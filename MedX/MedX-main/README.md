# MedX

A marketplace for near-expiry pharmaceuticals. Sellers list stock approaching its
expiry date; buyers get it below MRP. A demand-forecasting model predicts which
lots won't clear in time and recommends the discount that would clear them.

Rebuilt from a 2019 hackathon prototype. Nothing of the original code survives —
see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

## Why the forecasting model is the whole product

Every unit that expires on a shelf is a total loss for the seller and a missed
purchase for someone who couldn't afford it at MRP. The marketplace only works if
the discount is right: too shallow and stock still expires, too deep and the
seller gave away margin they didn't need to.

Because price is an input to the demand model, that model **inverts**. Ask it what
demand would be at each candidate price, and take the shallowest discount that
still clears the lot before expiry:

```python
for candidate_price in discounts:                     # ai/forecasting/predict.py
    if P(demand over remaining days ≥ stock) ≥ 0.85:
        return candidate_price                        # cheapest price that works
```

That is the pricing engine, obtained from the forecasting model rather than built
as a second system.

## Stack

| Layer | Choice |
|---|---|
| Frontend | Next.js 15 · React · TypeScript · Tailwind |
| API | FastAPI · Pydantic v2 · SQLAlchemy 2 |
| Database | PostgreSQL 16 + pgvector |
| Cache / queue | Redis · Celery |
| ML | LightGBM (quantile regression + binary classifier) · pandas |
| Infra | Docker Compose → containers + managed Postgres/Redis |

## Quickstart

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # → JWT_SECRET
docker compose up -d

curl localhost:8000/health          # {"status":"ok","database":true}
open http://localhost:8000/docs     # OpenAPI
```

`db/schema.sql` runs automatically on first boot. To reapply after editing it:
`docker compose down -v && docker compose up -d`.

## Layout

```
api/app/          FastAPI — config, models, auth, routers
  routers/forecast.py    risk scores, on-demand rescoring, seller dashboard
ai/forecasting/   the models
  features.py            market vs lot feature split, leak-safe lag windows
  baseline.py            cold-start fallback tiers
  train.py               LightGBM training, forward-chaining CV
  predict.py             scoring + price inversion
ai/alerts/rules.py       risk score → alert tier → channel
workers/tasks.py  Celery — nightly demand rollup, scoring, expiry sweep
db/schema.sql     source of truth for all constraints
```

## Training

The system runs without a trained model. `Forecaster` falls back to a rules
baseline (a days-to-expiry markdown ladder) when no bundle exists, so the product
works on day one and the ladder becomes the control group the model must beat.

```bash
python -m ai.forecasting.train \
  --demand-csv data/demand_daily.csv \
  --lots-csv   data/closed_lots.csv \
  --out models/current
```

Validation is a **forward-chaining time split**, never random — a random split
lets the model see next week while predicting this week, producing a great
offline number and a useless model.

## Tests

```bash
pip install -r api/requirements.txt pytest
pytest                 # 86 tests
```

`test_train_integration.py` trains real LightGBM models on synthetic data with a
known price signal, then saves, reloads, and scores through the production path —
so the pipeline is proven to run, not merely to type-check. Representative output:

| model | pinball | coverage | target |
|---|---|---|---|
| P10 | 0.208 | 0.179 | 0.10 |
| P50 | 0.611 | 0.607 | 0.50 |
| P90 | 0.332 | **0.929** | 0.90 |

Coverage is the metric that matters for a quantile model: ~90% of realized demand
should fall below the P90 prediction. A model can post excellent average error and
still be systematically overconfident.

The suite also asserts the signal the pricing engine depends on — that the model
recovers a ~3x demand lift between 95% and 45% of MRP, with `price_ratio` the
highest-gain feature. If price stops being predictive, inverting the model for a
recommendation becomes meaningless, so that relationship is a test, not an
assumption.

## Design decisions worth knowing

**Listings are keyed per batch, not per product.** Expiry is intrinsic to a lot,
and a recall must run as "disable every lot with batch X". Retrofitting per-batch
traceability onto a per-product schema is a migration you don't want.

**`demand_daily` is a table, not a view.** It must contain zero-sale days. A join
over `order_items` silently drops them, and a model trained only on days that had
sales learns demand is always positive — it will never recommend a discount.

**Cold start is handled explicitly**, degrading drug → ATC class → form prior →
rules baseline. Every score carries the tier that produced it, so the UI can say
"estimated from category data" instead of projecting false confidence. Baseline-tier
scores never drive automated repricing.

**Passwords are Argon2id.** The prototype used unsalted MD5.

**Regulatory constraints are schema-level.** Expired lots are blocked by a database
trigger, not application logic. Sellers can't list until their drug licence is
verified. Schedule H/H1/X items require a verified prescription at checkout.
`audit_log` is append-only. See the compliance section in `docs/ARCHITECTURE.md`.

## Status

Built: schema, forecasting models, alerting, nightly pipeline, forecast API,
Docker infra.

Not built: `web/` frontend, auth/listing/order routers beyond the forecast
surface, payments, generic-substitute search (schema support is in place),
symptom checker (deliberately deferred — high liability, needs medical review).

Build order and rationale in `docs/ARCHITECTURE.md`.
