-- MedX schema — PostgreSQL 16 + pgvector
--
-- Design notes:
--   * `listings` is keyed per BATCH, not per product. Recalls must be executable as
--     "disable every lot with batch X" and expiry is intrinsic to a lot, not a product.
--   * `demand_daily` is a materialized fact table rather than a view over orders,
--     because training must see zero-sale days. A join over order_items drops them
--     and biases the model toward optimism.
--   * Money is NUMERIC(10,2), never float.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy drug-name search
CREATE EXTENSION IF NOT EXISTS citext;    -- case-insensitive email

-- ============================================================================
-- Identity & compliance
-- ============================================================================

CREATE TYPE user_role AS ENUM ('buyer', 'seller', 'admin');

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           CITEXT UNIQUE NOT NULL,
    phone           TEXT UNIQUE,
    password_hash   TEXT NOT NULL,          -- argon2id
    role            user_role NOT NULL DEFAULT 'buyer',
    full_name       TEXT NOT NULL,
    email_verified  BOOLEAN NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE seller_status AS ENUM ('pending', 'verified', 'suspended');

-- A seller cannot list until license_number is verified. This is a legal
-- requirement (Form 20/21 retail/wholesale drug licence), not a trust feature.
CREATE TABLE sellers (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE RESTRICT,
    business_name       TEXT NOT NULL,
    license_number      TEXT NOT NULL,
    license_expiry      DATE NOT NULL,
    gstin               TEXT,
    status              seller_status NOT NULL DEFAULT 'pending',
    verified_at         TIMESTAMPTZ,
    address_line        TEXT NOT NULL,
    city                TEXT NOT NULL,
    state               TEXT NOT NULL,
    pincode             TEXT NOT NULL,
    region_code         TEXT NOT NULL,      -- forecasting granularity; see demand_daily
    rating              NUMERIC(3,2) NOT NULL DEFAULT 0.00
                            CHECK (rating >= 0 AND rating <= 5),
    rating_count        INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT seller_license_unexpired CHECK (license_expiry > '2000-01-01')
);

CREATE INDEX idx_sellers_region ON sellers(region_code) WHERE status = 'verified';

-- ============================================================================
-- Catalog
-- ============================================================================

-- Schedule H / H1 / X require a prescription. X additionally requires the seller
-- to retain records for 2 years.
CREATE TYPE drug_schedule AS ENUM ('OTC', 'H', 'H1', 'X');
CREATE TYPE drug_form AS ENUM (
    'tablet', 'capsule', 'syrup', 'injection', 'ointment',
    'drops', 'inhaler', 'powder', 'other'
);

CREATE TABLE drugs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_name          TEXT NOT NULL,
    composition         TEXT NOT NULL,      -- "Paracetamol 500mg + Caffeine 30mg"
    -- Normalized join key: lowercased, sorted, whitespace-stripped composition.
    -- Two brands of the same molecule share this. Substitute search depends on it.
    composition_key     TEXT NOT NULL,
    strength            TEXT NOT NULL,
    form                drug_form NOT NULL,
    manufacturer        TEXT NOT NULL,
    mrp                 NUMERIC(10,2) NOT NULL CHECK (mrp > 0),
    schedule_class      drug_schedule NOT NULL DEFAULT 'OTC',
    atc_code            TEXT,               -- WHO ATC; cold-start fallback tier
    pack_size           INTEGER NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (brand_name, strength, form, manufacturer)
);

CREATE INDEX idx_drugs_composition_key ON drugs(composition_key);
CREATE INDEX idx_drugs_atc ON drugs(atc_code);
CREATE INDEX idx_drugs_brand_trgm ON drugs USING gin (brand_name gin_trgm_ops);

-- Reserved for generic-substitute search. Nothing writes here yet — the column
-- exists so the migration doesn't need to run later on a live table.
CREATE TABLE drug_embeddings (
    drug_id     UUID PRIMARY KEY REFERENCES drugs(id) ON DELETE CASCADE,
    embedding   vector(768) NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_drug_embeddings_hnsw ON drug_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Inventory
-- ============================================================================

CREATE TYPE listing_status AS ENUM ('active', 'sold_out', 'expired', 'withdrawn', 'recalled');

CREATE TABLE listings (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    seller_id           UUID NOT NULL REFERENCES sellers(id) ON DELETE RESTRICT,
    drug_id             UUID NOT NULL REFERENCES drugs(id) ON DELETE RESTRICT,
    batch_number        TEXT NOT NULL,
    expiry_date         DATE NOT NULL,
    quantity_total      INTEGER NOT NULL CHECK (quantity_total > 0),
    quantity_available  INTEGER NOT NULL CHECK (quantity_available >= 0),
    mrp                 NUMERIC(10,2) NOT NULL CHECK (mrp > 0),
    current_price       NUMERIC(10,2) NOT NULL CHECK (current_price > 0),
    -- Seller-defined floor. Automated repricing may never go below this.
    price_floor         NUMERIC(10,2) NOT NULL CHECK (price_floor > 0),
    status              listing_status NOT NULL DEFAULT 'active',
    listed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT listing_qty_coherent  CHECK (quantity_available <= quantity_total),
    CONSTRAINT listing_price_le_mrp  CHECK (current_price <= mrp),
    CONSTRAINT listing_price_ge_floor CHECK (current_price >= price_floor),
    CONSTRAINT listing_floor_le_mrp  CHECK (price_floor <= mrp),
    UNIQUE (seller_id, drug_id, batch_number)
);

-- Selling expired stock is a criminal offence, not a policy violation. Enforce
-- it at the database so no application bug can produce it.
CREATE OR REPLACE FUNCTION reject_expired_active_listing() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'active' AND NEW.expiry_date <= CURRENT_DATE THEN
        RAISE EXCEPTION 'cannot have an active listing with expiry_date % (today is %)',
            NEW.expiry_date, CURRENT_DATE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_listings_not_expired
    BEFORE INSERT OR UPDATE ON listings
    FOR EACH ROW EXECUTE FUNCTION reject_expired_active_listing();

CREATE INDEX idx_listings_search ON listings(drug_id, status, expiry_date)
    WHERE status = 'active';
CREATE INDEX idx_listings_seller ON listings(seller_id, status);
CREATE INDEX idx_listings_expiry ON listings(expiry_date) WHERE status = 'active';
-- Recall path: must be fast.
CREATE INDEX idx_listings_batch ON listings(batch_number);

CREATE TYPE price_source AS ENUM ('seller', 'model', 'admin');

CREATE TABLE price_history (
    id              BIGSERIAL PRIMARY KEY,
    listing_id      UUID NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    price           NUMERIC(10,2) NOT NULL CHECK (price > 0),
    source          price_source NOT NULL,
    reason          TEXT,
    effective_from  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_price_history_listing ON price_history(listing_id, effective_from DESC);

-- ============================================================================
-- Prescriptions & orders
-- ============================================================================

CREATE TYPE prescription_status AS ENUM ('pending', 'approved', 'rejected', 'expired');

CREATE TABLE prescriptions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    file_key        TEXT NOT NULL,          -- object-store key, never a public URL
    doctor_name     TEXT,
    issued_on       DATE,
    status          prescription_status NOT NULL DEFAULT 'pending',
    reviewed_by     UUID REFERENCES users(id),
    reviewed_at     TIMESTAMPTZ,
    reject_reason   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_prescriptions_user ON prescriptions(user_id, status);

CREATE TYPE order_status AS ENUM (
    'pending_payment', 'awaiting_prescription', 'confirmed',
    'shipped', 'delivered', 'cancelled', 'refunded'
);

CREATE TABLE orders (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    buyer_id            UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status              order_status NOT NULL DEFAULT 'pending_payment',
    subtotal            NUMERIC(10,2) NOT NULL CHECK (subtotal >= 0),
    total               NUMERIC(10,2) NOT NULL CHECK (total >= 0),
    prescription_id     UUID REFERENCES prescriptions(id),
    shipping_pincode    TEXT NOT NULL,
    placed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_orders_buyer ON orders(buyer_id, placed_at DESC);

CREATE TABLE order_items (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id        UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    listing_id      UUID NOT NULL REFERENCES listings(id) ON DELETE RESTRICT,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    -- Denormalized at purchase time. The listing's price will move; what the
    -- buyer paid must not.
    unit_price      NUMERIC(10,2) NOT NULL CHECK (unit_price > 0),
    unit_mrp        NUMERIC(10,2) NOT NULL CHECK (unit_mrp > 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_listing ON order_items(listing_id);

-- ============================================================================
-- Forecasting substrate
-- ============================================================================

-- Rebuilt nightly by workers/tasks.py::rebuild_demand_daily.
-- MUST include zero-sale days — that is the entire reason this is a table and
-- not a view. A model trained only on days with sales learns that demand is
-- always positive and will never recommend a discount.
CREATE TABLE demand_daily (
    drug_id             UUID NOT NULL REFERENCES drugs(id) ON DELETE CASCADE,
    region_code         TEXT NOT NULL,
    day                 DATE NOT NULL,
    units_sold          INTEGER NOT NULL DEFAULT 0 CHECK (units_sold >= 0),
    -- Weighted by units, over listings that were active that day.
    avg_price_ratio     NUMERIC(5,4),       -- price ÷ mrp, in [0,1]
    active_listings     INTEGER NOT NULL DEFAULT 0,
    total_available     INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (drug_id, region_code, day)
);

CREATE INDEX idx_demand_daily_day ON demand_daily(day);
CREATE INDEX idx_demand_daily_lookup ON demand_daily(drug_id, region_code, day DESC);

-- Which fallback tier produced a score. Surfaced in the UI so we never project
-- false confidence, and used to gate automated repricing.
CREATE TYPE forecast_tier AS ENUM ('drug', 'atc_class', 'form_prior', 'rules_baseline');

CREATE TABLE lot_risk_scores (
    id                      BIGSERIAL PRIMARY KEY,
    listing_id              UUID NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    scored_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    days_to_expiry          INTEGER NOT NULL,
    p_sellout               NUMERIC(5,4) NOT NULL CHECK (p_sellout BETWEEN 0 AND 1),
    expected_units_p10      NUMERIC(10,2) NOT NULL,
    expected_units_p50      NUMERIC(10,2) NOT NULL,
    expected_units_p90      NUMERIC(10,2) NOT NULL,
    recommended_price       NUMERIC(10,2),  -- NULL when tier = rules_baseline
    tier                    forecast_tier NOT NULL,
    model_version           TEXT NOT NULL,

    UNIQUE (listing_id, scored_at)
);

CREATE INDEX idx_risk_scores_listing ON lot_risk_scores(listing_id, scored_at DESC);
CREATE INDEX idx_risk_scores_recent ON lot_risk_scores(scored_at DESC);

CREATE TYPE alert_tier AS ENUM ('watch', 'warning', 'critical');

CREATE TABLE alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    seller_id       UUID NOT NULL REFERENCES sellers(id) ON DELETE CASCADE,
    listing_id      UUID NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    tier            alert_tier NOT NULL,
    message         TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at    TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ
);

-- Supports the 7-day dedup cooldown: a lot sliding slowly toward expiry must
-- not re-alert at the same tier every night.
CREATE INDEX idx_alerts_dedup ON alerts(listing_id, tier, created_at DESC);
CREATE INDEX idx_alerts_seller ON alerts(seller_id, created_at DESC)
    WHERE acknowledged_at IS NULL;

-- ============================================================================
-- Audit
-- ============================================================================

-- Append-only. No UPDATE or DELETE grant is issued on this table in any
-- environment. Schedule X requires 2-year retention.
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    actor_id        UUID REFERENCES users(id),
    action          TEXT NOT NULL,          -- 'listing.price_changed', 'order.confirmed'
    entity_type     TEXT NOT NULL,
    entity_id       UUID NOT NULL,
    before          JSONB,
    after           JSONB,
    ip_address      INET,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id, created_at DESC);
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);
