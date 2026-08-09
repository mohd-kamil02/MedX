"""Alert tiering and suppression tests.

The hard part of an alert system is not generating alerts, it is not generating
them. These tests pin the "stay silent" cases as tightly as the noisy ones.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ai.alerts.rules import (
    COOLDOWN,
    Alert,
    AlertTier,
    Channel,
    build_alert,
    classify,
    suppressed_by_cooldown,
)
from ai.forecasting.baseline import Tier
from ai.forecasting.types import LotScore


def score(p_sellout, days, recommended=None, tier=Tier.DRUG):
    return LotScore(
        listing_id="lot-1",
        days_to_expiry=days,
        p_sellout=p_sellout,
        expected_units_p10=10.0,
        expected_units_p50=20.0,
        expected_units_p90=40.0,
        recommended_price=recommended,
        tier=tier,
        model_version="test-1",
    )


class TestClassification:
    def test_healthy_lot_stays_silent(self):
        """The most important case. A lot that will clear generates nothing."""
        assert classify(score(0.95, 120)) is None

    def test_expired_is_always_critical(self):
        assert classify(score(0.99, 0)) is AlertTier.CRITICAL

    def test_low_probability_and_near_expiry_is_critical(self):
        assert classify(score(0.20, 20)) is AlertTier.CRITICAL

    def test_low_probability_but_far_out_is_only_a_warning(self):
        """Same 20% risk, 200 days left — there is still time for it to change."""
        assert classify(score(0.20, 200)) is AlertTier.WATCH

    def test_middling_risk_within_a_quarter_is_a_warning(self):
        assert classify(score(0.45, 60)) is AlertTier.WARNING

    def test_mild_risk_is_watch_only(self):
        assert classify(score(0.75, 300)) is AlertTier.WATCH

    @pytest.mark.parametrize("p", [0.80, 0.85, 0.99])
    def test_at_or_above_threshold_is_silent(self, p):
        assert classify(score(p, 300)) is None


class TestChannels:
    def test_watch_never_notifies(self):
        """A dashboard indicator only. Interrupting someone needs a higher bar."""
        alert = Alert("lot-1", AlertTier.WATCH, "msg")
        assert alert.channels == (Channel.DASHBOARD,)
        assert Channel.PUSH not in alert.channels
        assert Channel.EMAIL not in alert.channels

    def test_warning_batches_into_digest_not_push(self):
        alert = Alert("lot-1", AlertTier.WARNING, "msg")
        assert Channel.DIGEST in alert.channels
        assert Channel.PUSH not in alert.channels

    def test_critical_escalates_to_push_and_email(self):
        alert = Alert("lot-1", AlertTier.CRITICAL, "msg")
        assert Channel.PUSH in alert.channels
        assert Channel.EMAIL in alert.channels


class TestMessageBuilding:
    def test_healthy_lot_produces_no_alert(self):
        assert build_alert(score(0.95, 120), "Crocin 500mg", 50.0) is None

    def test_includes_recommended_price_and_discount(self):
        alert = build_alert(score(0.22, 45, recommended=30.0), "Crocin 500mg", 50.0)
        assert alert is not None
        assert "30.00" in alert.message
        assert alert.payload["recommended_price"] == 30.0
        assert alert.payload["recommended_discount_pct"] == 40  # 30 off 50

    def test_baseline_tier_admits_it_cannot_price(self):
        alert = build_alert(
            score(0.30, 45, recommended=None, tier=Tier.RULES_BASELINE),
            "New Drug", 50.0,
        )
        assert "history" in alert.message.lower()
        assert "recommended_price" not in alert.payload

    def test_no_viable_price_flags_for_human_review(self):
        """Model tier but no price cleared the floor — a person must decide."""
        alert = build_alert(score(0.30, 45, recommended=None), "Crocin", 50.0)
        assert alert.payload["needs_human_review"] is True

    def test_provisional_flag_tracks_tier(self):
        drug_tier = build_alert(score(0.30, 45, recommended=30.0), "X", 50.0)
        class_tier = build_alert(
            score(0.30, 45, recommended=30.0, tier=Tier.ATC_CLASS), "X", 50.0
        )
        assert drug_tier.payload["estimate_is_provisional"] is False
        assert class_tier.payload["estimate_is_provisional"] is True

    def test_expired_message_says_delisted(self):
        alert = build_alert(score(0.10, 0), "Crocin 500mg", 50.0)
        assert alert.tier is AlertTier.CRITICAL
        assert "expired" in alert.message.lower()


class TestCooldown:
    def test_first_alert_is_never_suppressed(self):
        alert = Alert("lot-1", AlertTier.WARNING, "msg")
        assert suppressed_by_cooldown(alert, None) is False

    def test_repeat_inside_window_is_suppressed(self):
        """A lot drifting toward expiry must not re-alert nightly."""
        now = datetime.now(timezone.utc)
        alert = Alert("lot-1", AlertTier.WARNING, "msg")
        assert suppressed_by_cooldown(alert, now - timedelta(days=1), now) is True

    def test_alert_allowed_once_window_elapses(self):
        now = datetime.now(timezone.utc)
        alert = Alert("lot-1", AlertTier.WARNING, "msg")
        just_past = now - COOLDOWN - timedelta(seconds=1)
        assert suppressed_by_cooldown(alert, just_past, now) is False

    def test_boundary_is_exclusive(self):
        now = datetime.now(timezone.utc)
        alert = Alert("lot-1", AlertTier.WARNING, "msg")
        assert suppressed_by_cooldown(alert, now - COOLDOWN, now) is False
