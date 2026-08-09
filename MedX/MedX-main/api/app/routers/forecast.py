"""Forecast endpoints.

Scoring is batch, not per-request: expiry risk moves on a scale of days, so
computing it on every read would spend latency on information that has not
changed. These endpoints read the stored score. `POST /forecast/score/{id}`
exists for on-demand recomputation after a seller changes price or quantity.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ai.forecasting.baseline import Tier, resolve_history
from ai.forecasting.features import LotContext, build_history_features
from ai.forecasting.predict import Forecaster

from ..auth import CurrentSeller
from ..config import get_settings
from ..db import get_db
from ..models import Listing, LotRiskScore

router = APIRouter(prefix="/forecast", tags=["forecast"])

# One Forecaster per process. The boosters are read-only and safe to share.
_forecaster: Forecaster | None = None


def get_forecaster() -> Forecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster(get_settings().model_dir)
    return _forecaster


class RiskResponse(BaseModel):
    listing_id: uuid.UUID
    days_to_expiry: int
    p_sellout: float
    expected_units_p50: float
    recommended_price: float | None
    tier: str
    # False when the score came from category-level or baseline data. The UI must
    # qualify the number rather than implying precision we do not have.
    is_drug_specific: bool
    model_version: str
    scored_at: str | None = None


@router.get("/listing/{listing_id}", response_model=RiskResponse)
def get_listing_risk(
    listing_id: uuid.UUID,
    seller: CurrentSeller,
    db: Annotated[Session, Depends(get_db)],
):
    """Latest stored risk score for one of the caller's own lots."""
    listing = db.get(Listing, listing_id)
    if listing is None or listing.seller_id != seller.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Listing not found")

    score = (
        db.query(LotRiskScore)
        .filter(LotRiskScore.listing_id == listing_id)
        .order_by(LotRiskScore.scored_at.desc())
        .first()
    )
    if score is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No score yet — lots are scored nightly. POST /forecast/score/{id} to "
            "compute one now.",
        )

    return RiskResponse(
        listing_id=listing_id,
        days_to_expiry=score.days_to_expiry,
        p_sellout=float(score.p_sellout),
        expected_units_p50=float(score.expected_units_p50),
        recommended_price=(
            float(score.recommended_price) if score.recommended_price else None
        ),
        tier=score.tier,
        is_drug_specific=score.tier == Tier.DRUG.value,
        model_version=score.model_version,
        scored_at=score.scored_at.isoformat(),
    )


@router.post("/score/{listing_id}", response_model=RiskResponse)
def score_now(
    listing_id: uuid.UUID,
    seller: CurrentSeller,
    db: Annotated[Session, Depends(get_db)],
    forecaster: Annotated[Forecaster, Depends(get_forecaster)],
):
    """Recompute a lot's score immediately, e.g. after a price change."""
    listing = db.get(Listing, listing_id)
    if listing is None or listing.seller_id != seller.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Listing not found")

    drug = listing.drug
    ctx = LotContext(
        drug_id=str(listing.drug_id),
        region_code=seller.region_code,
        expiry_date=listing.expiry_date,
        quantity_available=listing.quantity_available,
        mrp=float(listing.mrp),
        current_price=float(listing.current_price),
        price_floor=float(listing.price_floor),
        seller_rating=float(seller.rating),
        is_prescription_only=drug.requires_prescription,
        atc_code=drug.atc_code,
        form=drug.form,
    )

    demand = pd.read_sql(
        text(
            "SELECT drug_id::text, region_code, day, units_sold, avg_price_ratio, "
            "active_listings, total_available FROM demand_daily WHERE day >= :since"
        ),
        db.connection(),
        params={"since": date.today() - timedelta(days=120)},
    )
    drug_meta = pd.read_sql(
        text(
            "SELECT id::text AS drug_id, atc_code, form::text AS form, "
            "schedule_class::text AS schedule_class FROM drugs"
        ),
        db.connection(),
    )

    if not demand.empty:
        demand = build_history_features(demand)

    history, tier = resolve_history(
        demand,
        drug_id=str(listing.drug_id),
        region_code=seller.region_code,
        atc_code=drug.atc_code,
        form=drug.form,
        schedule_class=drug.schedule_class,
        drug_meta=drug_meta,
    )

    score = forecaster.score_lot(str(listing_id), ctx, history, tier)

    db.add(
        LotRiskScore(
            listing_id=listing_id,
            days_to_expiry=score.days_to_expiry,
            p_sellout=score.p_sellout,
            expected_units_p10=score.expected_units_p10,
            expected_units_p50=score.expected_units_p50,
            expected_units_p90=score.expected_units_p90,
            recommended_price=score.recommended_price,
            tier=score.tier.value,
            model_version=score.model_version,
        )
    )
    db.commit()

    return RiskResponse(
        listing_id=listing_id,
        days_to_expiry=score.days_to_expiry,
        p_sellout=score.p_sellout,
        expected_units_p50=score.expected_units_p50,
        recommended_price=score.recommended_price,
        tier=score.tier.value,
        is_drug_specific=score.tier is Tier.DRUG,
        model_version=score.model_version,
    )


class SellerDashboard(BaseModel):
    total_active_lots: int
    at_risk_lots: int
    units_at_risk: int
    value_at_risk: float


@router.get("/dashboard", response_model=SellerDashboard)
def seller_dashboard(
    seller: CurrentSeller, db: Annotated[Session, Depends(get_db)]
) -> SellerDashboard:
    """Headline numbers: how much stock is projected not to clear, and its value.

    `value_at_risk` is at cost-proxy (current price), not MRP — MRP would flatter
    the number, and the seller is not going to realise MRP on this stock.
    """
    row = db.execute(
        text(
            """
            WITH latest AS (
                SELECT DISTINCT ON (listing_id) listing_id, p_sellout
                FROM lot_risk_scores
                ORDER BY listing_id, scored_at DESC
            )
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE latest.p_sellout < 0.6) AS at_risk,
                COALESCE(SUM(l.quantity_available)
                         FILTER (WHERE latest.p_sellout < 0.6), 0) AS units,
                COALESCE(SUM(l.quantity_available * l.current_price)
                         FILTER (WHERE latest.p_sellout < 0.6), 0) AS value
            FROM listings l
            LEFT JOIN latest ON latest.listing_id = l.id
            WHERE l.seller_id = :sid AND l.status = 'active'
            """
        ),
        {"sid": seller.id},
    ).one()

    return SellerDashboard(
        total_active_lots=row.total,
        at_risk_lots=row.at_risk,
        units_at_risk=int(row.units),
        value_at_risk=float(row.value),
    )
