"""Cold-start fallback for lots with no usable sales history.

A new marketplace has no history, which is the most common reason a forecasting
feature fails on launch. Rather than let the model extrapolate from nothing, we
degrade explicitly through four tiers and record which one produced each score.

    drug          this drug in this region, >= MIN_OBS_DRUG observations
    atc_class     same ATC therapeutic class in this region
    form_prior    same dosage form + schedule class, nationally
    rules_baseline  no data at all — a days-to-expiry ladder, no ML

Scores from `rules_baseline` never drive automated repricing. They produce a
suggestion for a human. The tier travels with the score all the way to the UI so
we can say "estimated from category data" instead of implying precision we
don't have.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

MIN_OBS_DRUG = 30       # ~1 month of daily observations
MIN_OBS_CLASS = 100


class Tier(str, Enum):
    DRUG = "drug"
    ATC_CLASS = "atc_class"
    FORM_PRIOR = "form_prior"
    RULES_BASELINE = "rules_baseline"

    @property
    def supports_auto_repricing(self) -> bool:
        return self is not Tier.RULES_BASELINE


@dataclass(frozen=True)
class BaselineEstimate:
    daily_units_p10: float
    daily_units_p50: float
    daily_units_p90: float
    tier: Tier


def resolve_history(
    demand: pd.DataFrame,
    drug_id: str,
    region_code: str,
    atc_code: str | None,
    form: str,
    schedule_class: str,
    drug_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, Tier]:
    """Return the most specific history frame with enough observations, and its tier.

    `drug_meta` maps drug_id -> (atc_code, form, schedule_class) and is used to
    widen the selection at each fallback step.
    """
    exact = demand[
        (demand["drug_id"] == drug_id) & (demand["region_code"] == region_code)
    ]
    if len(exact) >= MIN_OBS_DRUG:
        return exact, Tier.DRUG

    if atc_code:
        sibling_ids = drug_meta.loc[drug_meta["atc_code"] == atc_code, "drug_id"]
        klass = demand[
            demand["drug_id"].isin(sibling_ids)
            & (demand["region_code"] == region_code)
        ]
        if len(klass) >= MIN_OBS_CLASS:
            return _collapse(klass), Tier.ATC_CLASS

    prior_ids = drug_meta.loc[
        (drug_meta["form"] == form) & (drug_meta["schedule_class"] == schedule_class),
        "drug_id",
    ]
    prior = demand[demand["drug_id"].isin(prior_ids)]
    if len(prior) >= MIN_OBS_CLASS:
        return _collapse(prior), Tier.FORM_PRIOR

    return demand.iloc[0:0], Tier.RULES_BASELINE


def _collapse(frame: pd.DataFrame) -> pd.DataFrame:
    """Average a multi-drug frame down to one series per day.

    Class-level history describes a category, not a product, so per-day mean is
    the right aggregation — summing would make a class look like it sells N times
    as fast as any member of it.
    """
    return (
        frame.groupby("day", as_index=False)
        .agg(
            units_sold=("units_sold", "mean"),
            avg_price_ratio=("avg_price_ratio", "mean"),
            active_listings=("active_listings", "mean"),
            total_available=("total_available", "mean"),
        )
        .assign(drug_id="__aggregate__", region_code="__aggregate__")
    )


# Discount ladder used when there is no data of any kind. These are starting
# heuristics from standard retail-pharmacy markdown practice, not learned values;
# they exist to make the product functional on day one and to serve as the
# control group the trained model must beat.
_RULES_LADDER = [
    # (days_to_expiry_below, discount_fraction, assumed_sellout_probability)
    (30, 0.50, 0.35),
    (60, 0.35, 0.50),
    (90, 0.25, 0.65),
    (180, 0.15, 0.75),
    (10**6, 0.05, 0.85),
]


def rules_baseline(days_to_expiry: int, quantity_available: int) -> BaselineEstimate:
    """Days-to-expiry markdown ladder. No model, no history, no false precision."""
    for threshold, _discount, p_sellout in _RULES_LADDER:
        if days_to_expiry < threshold:
            break

    # Spread the implied sellout over the remaining days to get a nominal rate.
    days = max(days_to_expiry, 1)
    p50 = (quantity_available * p_sellout) / days

    return BaselineEstimate(
        daily_units_p10=p50 * 0.3,
        daily_units_p50=p50,
        daily_units_p90=p50 * 2.5,
        tier=Tier.RULES_BASELINE,
    )


def rules_discount(days_to_expiry: int) -> float:
    """Discount fraction the ladder suggests. Used for the human-facing suggestion."""
    for threshold, discount, _p in _RULES_LADDER:
        if days_to_expiry < threshold:
            return discount
    return _RULES_LADDER[-1][1]
