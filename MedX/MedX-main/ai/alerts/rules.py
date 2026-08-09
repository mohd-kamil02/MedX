"""Turn a lot risk score into a seller-facing alert — or into nothing.

The hard part of an alert system is not generating alerts, it is not generating
them. Sellers who get a notification every night stop reading notifications, and
the one that mattered goes unread with the rest. So: three tiers, a high bar for
interrupting anyone, and a cooldown so a lot drifting slowly toward expiry does
not re-alert every night at the same severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ai.forecasting.baseline import Tier
from ai.forecasting.types import LotScore

# A lot must not re-alert at the same tier inside this window.
COOLDOWN = timedelta(days=7)


class AlertTier(str, Enum):
    WATCH = "watch"        # dashboard indicator only, no notification
    WARNING = "warning"    # in-app, batched into a daily digest
    CRITICAL = "critical"  # push + email, immediate


class Channel(str, Enum):
    DASHBOARD = "dashboard"
    DIGEST = "digest"
    PUSH = "push"
    EMAIL = "email"


_CHANNELS: dict[AlertTier, tuple[Channel, ...]] = {
    AlertTier.WATCH: (Channel.DASHBOARD,),
    AlertTier.WARNING: (Channel.DASHBOARD, Channel.DIGEST),
    AlertTier.CRITICAL: (Channel.DASHBOARD, Channel.PUSH, Channel.EMAIL),
}


@dataclass(frozen=True)
class Alert:
    listing_id: str
    tier: AlertTier
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> tuple[Channel, ...]:
        return _CHANNELS[self.tier]


def classify(score: LotScore) -> AlertTier | None:
    """Decide the severity of a score, or None to stay silent.

    Both probability and time matter. A lot with a 50% sellout chance and 200 days
    left is not urgent — there is time for the picture to change. The same
    probability with 20 days left is.
    """
    p, days = score.p_sellout, score.days_to_expiry

    if days <= 0:
        return AlertTier.CRITICAL  # already expired; must be delisted now
    if p < 0.30 and days < 30:
        return AlertTier.CRITICAL
    if p < 0.60 and days < 90:
        return AlertTier.WARNING
    if p < 0.80:
        return AlertTier.WATCH
    return None


def build_alert(score: LotScore, drug_name: str, mrp: float) -> Alert | None:
    """Compose the alert for a score, or None if it does not warrant one."""
    tier = classify(score)
    if tier is None:
        return None

    payload: dict[str, Any] = {
        "p_sellout": score.p_sellout,
        "days_to_expiry": score.days_to_expiry,
        "expected_units_p50": score.expected_units_p50,
        "forecast_tier": score.tier.value,
        "model_version": score.model_version,
        # Surfaced so the UI can qualify the claim rather than implying precision
        # the estimate does not have.
        "estimate_is_provisional": score.tier is not Tier.DRUG,
    }

    if score.days_to_expiry <= 0:
        return Alert(
            listing_id=score.listing_id,
            tier=tier,
            message=f"{drug_name} has expired and has been delisted.",
            payload=payload,
        )

    message = (
        f"{drug_name}: {int(score.p_sellout * 100)}% likely to sell out before "
        f"expiry in {score.days_to_expiry} days."
    )

    if score.recommended_price is not None:
        discount = round((1 - score.recommended_price / mrp) * 100)
        payload["recommended_price"] = score.recommended_price
        payload["recommended_discount_pct"] = discount
        message += f" Pricing at ₹{score.recommended_price:.2f} ({discount}% off) should clear it."
    elif score.tier is Tier.RULES_BASELINE:
        message += " Not enough sales history for a price recommendation yet."
    else:
        # recommend_price exhausted the grid without clearing the seller's floor.
        message += " No price above your floor is projected to clear this lot."
        payload["needs_human_review"] = True

    return Alert(score.listing_id, tier, message, payload)


def suppressed_by_cooldown(
    alert: Alert, last_sent_at: datetime | None, now: datetime | None = None
) -> bool:
    """True if an identical-tier alert for this lot fired inside the cooldown."""
    if last_sent_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - last_sent_at) < COOLDOWN
