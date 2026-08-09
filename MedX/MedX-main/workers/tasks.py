"""Celery tasks: nightly demand rollup, lot scoring, and alert dispatch.

Order matters and the beat schedule enforces it. `rebuild_demand_daily` must
finish before `score_all_lots` runs, or the models score today against yesterday's
aggregates. They are chained rather than independently scheduled for that reason.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from celery import Celery, chain
from celery.schedules import crontab
from sqlalchemy import create_engine, text

from ai.alerts.rules import build_alert, suppressed_by_cooldown
from ai.forecasting.baseline import resolve_history
from ai.forecasting.features import LotContext, build_history_features
from ai.forecasting.predict import Forecaster

log = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/current"))

app = Celery("medx", broker=REDIS_URL, backend=REDIS_URL)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Instantiated lazily so a worker without a model bundle still boots and runs the
# rules baseline instead of crash-looping.
_forecaster: Forecaster | None = None


def get_forecaster() -> Forecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster(MODEL_DIR)
    return _forecaster


app.conf.beat_schedule = {
    "nightly-forecast": {
        "task": "workers.tasks.nightly_pipeline",
        "schedule": crontab(hour=2, minute=30),
    },
    "expire-lots": {
        "task": "workers.tasks.expire_lots",
        "schedule": crontab(hour=0, minute=5),
    },
}
app.conf.timezone = "Asia/Kolkata"


@app.task(name="workers.tasks.nightly_pipeline")
def nightly_pipeline():
    """Chain the rollup into scoring so scoring never reads stale aggregates."""
    return chain(rebuild_demand_daily.si(), score_all_lots.si())()


# ---------------------------------------------------------------- demand rollup

# The LEFT JOIN and COALESCE are the whole point: this must emit a row for every
# (drug, region, day) a listing was active, including days with no sales. A model
# trained only on days that had sales learns demand is always positive and will
# never recommend a discount.
_REBUILD_SQL = text(
    """
    INSERT INTO demand_daily (
        drug_id, region_code, day, units_sold,
        avg_price_ratio, active_listings, total_available
    )
    SELECT
        l.drug_id,
        s.region_code,
        d.day::date,
        COALESCE(SUM(oi.quantity), 0)                              AS units_sold,
        AVG(l.current_price / NULLIF(l.mrp, 0))::numeric(5,4)      AS avg_price_ratio,
        COUNT(DISTINCT l.id)                                       AS active_listings,
        COALESCE(SUM(DISTINCT l.quantity_available), 0)            AS total_available
    FROM generate_series(
            (CURRENT_DATE - make_interval(days => :lookback_days)),
            CURRENT_DATE - 1,
            '1 day'
         ) AS d(day)
    JOIN listings l
      ON l.listed_at::date <= d.day::date
     AND (l.status = 'active' OR l.updated_at::date >= d.day::date)
    JOIN sellers s ON s.id = l.seller_id
    LEFT JOIN order_items oi
      ON oi.listing_id = l.id
    LEFT JOIN orders o
      ON o.id = oi.order_id
     AND o.placed_at::date = d.day::date
     AND o.status NOT IN ('cancelled', 'refunded')
    GROUP BY l.drug_id, s.region_code, d.day::date
    ON CONFLICT (drug_id, region_code, day) DO UPDATE SET
        units_sold      = EXCLUDED.units_sold,
        avg_price_ratio = EXCLUDED.avg_price_ratio,
        active_listings = EXCLUDED.active_listings,
        total_available = EXCLUDED.total_available
    """
)


@app.task(name="workers.tasks.rebuild_demand_daily")
def rebuild_demand_daily(lookback_days: int = 120) -> int:
    """Recompute the demand fact table for the trailing window.

    Recomputed rather than appended because orders get cancelled and refunded
    after the fact; an append-only rollup would keep counting sales that were
    reversed.
    """
    with engine.begin() as conn:
        result = conn.execute(_REBUILD_SQL, {"lookback_days": lookback_days})
    log.info("demand_daily: upserted %s rows", result.rowcount)
    return result.rowcount


# --------------------------------------------------------------------- scoring

_ACTIVE_LOTS_SQL = text(
    """
    SELECT
        l.id AS listing_id, l.drug_id, l.expiry_date, l.quantity_available,
        l.mrp, l.current_price, l.price_floor,
        s.id AS seller_id, s.region_code, s.rating AS seller_rating,
        d.brand_name, d.atc_code, d.form::text AS form,
        d.schedule_class::text AS schedule_class
    FROM listings l
    JOIN sellers s ON s.id = l.seller_id
    JOIN drugs d   ON d.id = l.drug_id
    WHERE l.status = 'active' AND l.quantity_available > 0
    """
)

_DEMAND_SQL = text(
    "SELECT drug_id::text, region_code, day, units_sold, avg_price_ratio, "
    "active_listings, total_available FROM demand_daily WHERE day >= :since"
)

_DRUG_META_SQL = text(
    "SELECT id::text AS drug_id, atc_code, form::text AS form, "
    "schedule_class::text AS schedule_class FROM drugs"
)


@app.task(name="workers.tasks.score_all_lots")
def score_all_lots() -> dict[str, int]:
    """Score every active lot and emit alerts for the ones that need attention."""
    forecaster = get_forecaster()
    today = date.today()

    with engine.begin() as conn:
        lots = pd.read_sql(_ACTIVE_LOTS_SQL, conn)
        demand = pd.read_sql(
            _DEMAND_SQL, conn, params={"since": today - timedelta(days=120)}
        )
        drug_meta = pd.read_sql(_DRUG_META_SQL, conn)

    if lots.empty:
        log.info("no active lots to score")
        return {"scored": 0, "alerts": 0}

    demand = build_history_features(demand) if not demand.empty else demand

    scored = alerts_created = 0
    for lot in lots.itertuples(index=False):
        ctx = LotContext(
            drug_id=str(lot.drug_id),
            region_code=lot.region_code,
            expiry_date=lot.expiry_date,
            quantity_available=int(lot.quantity_available),
            mrp=float(lot.mrp),
            current_price=float(lot.current_price),
            price_floor=float(lot.price_floor),
            seller_rating=float(lot.seller_rating),
            is_prescription_only=lot.schedule_class in ("H", "H1", "X"),
            atc_code=lot.atc_code,
            form=lot.form,
        )

        history, tier = resolve_history(
            demand,
            drug_id=str(lot.drug_id),
            region_code=lot.region_code,
            atc_code=lot.atc_code,
            form=lot.form,
            schedule_class=lot.schedule_class,
            drug_meta=drug_meta,
        )

        score = forecaster.score_lot(
            str(lot.listing_id), ctx, history, tier, as_of=today
        )
        _persist_score(score)
        scored += 1

        alert = build_alert(score, lot.brand_name, float(lot.mrp))
        if alert and _emit_alert(alert, str(lot.seller_id)):
            alerts_created += 1

    log.info("scored %s lots, created %s alerts", scored, alerts_created)
    return {"scored": scored, "alerts": alerts_created}


def _persist_score(score) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO lot_risk_scores (
                    listing_id, days_to_expiry, p_sellout,
                    expected_units_p10, expected_units_p50, expected_units_p90,
                    recommended_price, tier, model_version
                ) VALUES (
                    :listing_id, :days, :p, :p10, :p50, :p90, :rec, :tier, :version
                )
                """
            ),
            {
                "listing_id": score.listing_id,
                "days": score.days_to_expiry,
                "p": score.p_sellout,
                "p10": score.expected_units_p10,
                "p50": score.expected_units_p50,
                "p90": score.expected_units_p90,
                "rec": score.recommended_price,
                "tier": score.tier.value,
                "version": score.model_version,
            },
        )


def _emit_alert(alert, seller_id: str) -> bool:
    """Insert an alert unless an identical-tier one fired inside the cooldown."""
    import json

    with engine.begin() as conn:
        last = conn.execute(
            text(
                "SELECT created_at FROM alerts WHERE listing_id = :lid AND tier = :tier "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"lid": alert.listing_id, "tier": alert.tier.value},
        ).scalar()

        if suppressed_by_cooldown(alert, last):
            return False

        conn.execute(
            text(
                "INSERT INTO alerts (seller_id, listing_id, tier, message, payload) "
                "VALUES (:sid, :lid, :tier, :msg, CAST(:payload AS jsonb))"
            ),
            {
                "sid": seller_id,
                "lid": alert.listing_id,
                "tier": alert.tier.value,
                "msg": alert.message,
                "payload": json.dumps(alert.payload),
            },
        )
    return True


# ------------------------------------------------------------------- expiry

@app.task(name="workers.tasks.expire_lots")
def expire_lots() -> int:
    """Delist lots that reached expiry.

    Selling expired stock is a criminal offence, so this runs at 00:05 daily and
    independently of the forecasting pipeline — it must not be skipped because a
    model failed to load.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE listings SET status = 'expired', updated_at = now() "
                "WHERE status = 'active' AND expiry_date <= CURRENT_DATE"
            )
        )
    if result.rowcount:
        log.warning("delisted %s expired lots", result.rowcount)
    return result.rowcount
