# MedX — Architecture

A marketplace for near-expiry pharmaceuticals. Sellers (pharmacies, distributors) list
stock approaching expiry; buyers purchase it below MRP. A demand-forecasting model
predicts which lots will go unsold and recommends the discount that clears them.

## The core insight

Every unit that expires on a shelf is a total loss for the seller and a missed purchase
for a price-sensitive buyer. The marketplace only works if discounts are set correctly:
too shallow and stock still expires, too deep and the seller loses margin they didn't
need to give up. That pricing decision is a forecasting problem, which is why
forecasting is the first AI system we build rather than a later addition.

---

## System shape

```
                        ┌──────────────────────────┐
   buyers / sellers ───▶│  web  (Next.js 15, TS)   │
                        └────────────┬─────────────┘
                                     │ REST + SSE
                        ┌────────────▼─────────────┐
                        │  api  (FastAPI, Python)  │
                        │  auth · catalog · orders │
                        └──┬──────────┬─────────┬──┘
                           │          │         │
              ┌────────────▼──┐  ┌────▼────┐  ┌─▼─────────────┐
              │  Postgres 16  │  │  Redis  │  │  object store │
              │  + pgvector   │  │ cache + │  │ (prescriptions│
              │               │  │  queue  │  │  , invoices)  │
              └───────▲───────┘  └────┬────┘  └───────────────┘
                      │               │
                      │      ┌────────▼─────────┐
                      └──────┤ workers (Celery) │
                             │  nightly scoring │
                             │  alert dispatch  │
                             └────────┬─────────┘
                                      │
                             ┌────────▼─────────┐
                             │  ai/forecasting  │
                             │  LightGBM models │
                             └──────────────────┘
```

Four deployable units: `web`, `api`, `workers`, and Postgres/Redis as managed services.
`ai/` is a library imported by `workers` and `api`, not a separate service — it has no
independent scaling need until model inference exceeds ~50ms p99, at which point it
splits out behind its own endpoint without changing callers.

---

## Data model

The schema (`db/schema.sql`) has four groups.

**Identity & compliance** — `users`, `sellers`, `prescriptions`, `audit_log`.
Sellers carry `license_number` and `gstin` because a real pharmaceutical marketplace
cannot onboard an unverified seller. `audit_log` is append-only and records every
price change, order state transition, and prescription decision.

**Catalog** — `drugs` is the canonical product table, keyed on
(composition, strength, form) rather than brand name, so that two brands of the same
molecule are recognizably the same thing. This is what makes generic-substitute search
possible later, and it is why `drug_embeddings` (pgvector) exists in the schema now even
though nothing writes to it yet.

**Inventory & transactions** — `listings` is the central object: one row per
(seller, drug, batch, expiry). Note it is a *lot*, not a product — expiry and batch
number are intrinsic to it, and price moves over its lifetime. `price_history` records
every price the lot has held and whether a human or the model set it.

**Forecasting substrate** — `demand_daily` is the aggregated fact table the models train
on: units sold per (drug, region, day) with the average price and active listing count
that produced them. `lot_risk_scores` holds model output per lot per scoring run.
`alerts` holds what we decided to tell the seller about it.

`demand_daily` exists as a separate table rather than being computed from `order_items`
on demand because training reads it thousands of times and it must include days with
*zero* sales — which a join over orders silently drops, biasing the model toward
optimism. It is rebuilt nightly by `workers/tasks.py::rebuild_demand_daily`.

---

## The forecasting system

### What it predicts

Two models, trained together in `ai/forecasting/train.py`:

1. **Demand model** — daily units sold for a (drug, region), as a quantile regressor at
   P10/P50/P90. Quantiles rather than a point estimate because the decision we make with
   it is a risk decision: "will this clear" needs a distribution, not a mean.

2. **Sellout classifier** — probability a specific lot fully sells before its expiry.
   Trained on historical lots with their realized outcome.

The demand model takes **price ratio** (offer price ÷ MRP) as a feature. That is
deliberate and load-bearing — it is what makes the model invertible.

### Deriving price from the forecast

`ai/forecasting/predict.py::recommend_price` sweeps candidate discounts, asks the demand
model what each implies for expected units over the remaining shelf life, and returns the
shallowest discount that still hits the target sellout probability:

```
for each candidate price p:
    expected_units = Σ over remaining days of demand_model(features, price=p)
    if P(expected_units ≥ quantity_available) ≥ target:
        return p          # shallowest price that clears the lot
```

The seller keeps the margin they don't need to give away, and the buyer gets the discount
the stock actually requires. This is the pricing algorithm the original README promised.

### Cold start

A new marketplace has no sales history, so the model has nothing to learn from. This is
the single most likely reason a forecasting feature fails in production, so it is handled
explicitly in `ai/forecasting/baseline.py` via a hierarchical fallback:

```
drug-specific history (≥ 30 observations)
   ↓ else
ATC therapeutic class history        e.g. all NSAIDs in this region
   ↓ else
form + schedule prior                e.g. all OTC tablets nationally
   ↓ else
rules baseline                       days-to-expiry ladder, no ML
```

Every score carries the tier it came from, so the UI can say "estimated from category
data" instead of projecting false confidence. Lots scored from the rules baseline are
excluded from automated repricing — they generate a suggestion for a human, not an action.

### Serving

Scoring is **batch, not real-time**. A nightly Celery beat job scores every active listing
and writes to `lot_risk_scores`. Expiry risk changes on a scale of days; computing it per
request would spend latency on information that doesn't move. The API reads the stored
score. A `POST /forecast/score` endpoint exists for on-demand recomputation when a seller
changes price or quantity.

### Smart alerts

`ai/alerts/rules.py` turns a risk score into an action. The tiering is intentionally
coarse — sellers ignore noisy alert streams, so the bar for interrupting someone is high:

| Tier | Condition | Action |
|---|---|---|
| `critical` | P(sellout) < 0.3, < 30 days left | Push + email, recommend price now |
| `warning` | P(sellout) < 0.6, < 90 days left | In-app, batched into a daily digest |
| `watch` | P(sellout) < 0.8 | Dashboard indicator only, no notification |
| — | otherwise | Nothing |

Alerts deduplicate on (listing, tier) with a 7-day cooldown so a lot sliding slowly toward
expiry doesn't generate the same warning nightly.

---

## Regulatory constraints

You said this is a real startup MVP, so this is a design constraint rather than a footnote.
In India (the original project's context) this is governed by the Drugs and Cosmetics Act,
1940 and the Drugs and Cosmetics Rules, 1945:

- **Selling near-expiry stock is legal. Selling expired stock is not**, and the penalties
  are criminal, not civil. `listings` has a hard database-level check preventing an expiry
  date in the past, and the nightly job transitions lots to `expired` and delists them.
- **Schedule H, H1, and X drugs require a valid prescription.** `drugs.schedule_class`
  carries this, and the order flow blocks checkout on those items without a verified
  prescription. Schedule X additionally requires the seller to retain records for 2 years —
  hence `audit_log` retention.
- **Sellers must hold a valid retail/wholesale drug licence** (Form 20/21). Captured at
  onboarding in `sellers.license_number`, verified before a seller can list.
- **Batch and expiry traceability is mandatory** for recalls. This is why `listings` is
  keyed per batch rather than per product — a recall must be executable as
  "disable every lot with batch X", and that query has to be fast.

None of this is optional infrastructure you add later; retrofitting per-batch traceability
onto a per-product schema is a migration you do not want to run.

---

## What is deliberately not built yet

- **Generic substitute search** — schema support is in place (`drugs.composition_key`,
  `drug_embeddings`), no implementation. Needs a real drug catalog first.
- **Symptom checker** — high liability surface, needs medical review and clear
  "not medical advice" boundaries before it ships. Not an MVP feature.
- **Payments** — the order state machine has the seams for it (`orders.status`), but
  integrating a PSP means PCI scope and is its own workstream.

## Build order

1. **Foundation** — schema, auth, seller onboarding with licence capture, listing CRUD.
2. **Marketplace** — search, cart, order state machine, prescription upload + verification.
3. **Forecasting** — `demand_daily` pipeline, baseline rules scorer, alerts. Ships before
   any ML: the rules baseline is a working product and becomes the model's control group.
4. **Models** — train LightGBM once there is history, A/B against the rules baseline,
   promote only on a win.
5. **Automated repricing** — seller opt-in, model sets prices within a floor they define.

Steps 3 and 4 in that order matter. Shipping the rules baseline first gives you a
functioning feature on day one and an honest measurement of whether the model beats it.
