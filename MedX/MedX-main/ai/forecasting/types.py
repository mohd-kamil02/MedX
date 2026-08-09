"""Data contracts shared across the forecasting and alerting layers.

Deliberately dependency-free (stdlib only). `LotScore` used to live in
`predict.py`, which meant anything reading a score — the alert rules, the API
serializers, the tests — had to import LightGBM just to name the type. The score
is a data contract, not model code, so it belongs on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import Tier

# The sellout probability we aim for when recommending a price. Below 1.0 on
# purpose: pricing every lot to a near-certain clear means giving away margin on
# stock that would have sold anyway.
DEFAULT_TARGET_SELLOUT = 0.85


@dataclass(frozen=True)
class LotScore:
    """One scoring run's verdict on one lot."""

    listing_id: str
    days_to_expiry: int
    p_sellout: float
    expected_units_p10: float
    expected_units_p50: float
    expected_units_p90: float
    recommended_price: float | None
    tier: Tier
    model_version: str

    @property
    def clears_comfortably(self) -> bool:
        return self.p_sellout >= DEFAULT_TARGET_SELLOUT
