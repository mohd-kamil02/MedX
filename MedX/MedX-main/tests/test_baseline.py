"""Cold-start fallback tests.

The safety property under test: a score derived from thin or absent data must
never drive an automated price change. Everything else here supports that.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from ai.forecasting.baseline import (
    MIN_OBS_CLASS,
    MIN_OBS_DRUG,
    Tier,
    resolve_history,
    rules_baseline,
    rules_discount,
)


@pytest.fixture
def drug_meta():
    return pd.DataFrame(
        [
            {"drug_id": "para-1", "atc_code": "N02BE", "form": "tablet",
             "schedule_class": "OTC"},
            {"drug_id": "para-2", "atc_code": "N02BE", "form": "tablet",
             "schedule_class": "OTC"},
            {"drug_id": "amox-1", "atc_code": "J01CA", "form": "capsule",
             "schedule_class": "H"},
        ]
    )


def _demand(drug_id, region, n_days, units=3):
    start = date.today() - timedelta(days=n_days)
    return pd.DataFrame(
        [
            {
                "drug_id": drug_id,
                "region_code": region,
                "day": start + timedelta(days=i),
                "units_sold": units,
                "avg_price_ratio": 0.7,
                "active_listings": 2,
                "total_available": 50,
            }
            for i in range(n_days)
        ]
    )


class TestTierResolution:
    def test_ample_history_uses_drug_tier(self, drug_meta):
        demand = _demand("para-1", "MH-01", MIN_OBS_DRUG + 10)
        _, tier = resolve_history(
            demand, "para-1", "MH-01", "N02BE", "tablet", "OTC", drug_meta
        )
        assert tier is Tier.DRUG

    def test_thin_drug_history_falls_back_to_class(self, drug_meta):
        """Below the drug threshold but the ATC class has enough between them."""
        demand = pd.concat(
            [
                _demand("para-1", "MH-01", MIN_OBS_DRUG - 5),
                _demand("para-2", "MH-01", MIN_OBS_CLASS),
            ]
        )
        _, tier = resolve_history(
            demand, "para-1", "MH-01", "N02BE", "tablet", "OTC", drug_meta
        )
        assert tier is Tier.ATC_CLASS

    def test_unknown_drug_borrows_from_its_class(self, drug_meta):
        demand = _demand("para-2", "MH-01", MIN_OBS_CLASS + 20)
        _, tier = resolve_history(
            demand, "brand-new", "MH-01", "N02BE", "tablet", "OTC", drug_meta
        )
        assert tier is Tier.ATC_CLASS

    def test_no_data_anywhere_falls_to_rules(self, drug_meta):
        _, tier = resolve_history(
            pd.DataFrame(columns=["drug_id", "region_code", "day", "units_sold"]),
            "unknown", "XX-99", None, "syrup", "X", drug_meta,
        )
        assert tier is Tier.RULES_BASELINE

    def test_other_region_does_not_count_as_drug_history(self, drug_meta):
        """Demand is regional. Delhi sales say little about Mumbai."""
        demand = _demand("para-1", "DL-01", MIN_OBS_DRUG + 50)
        _, tier = resolve_history(
            demand, "para-1", "MH-01", "N02BE", "tablet", "OTC", drug_meta
        )
        assert tier is not Tier.DRUG

    def test_class_aggregate_collapses_to_one_row_per_day(self, drug_meta):
        """Two drugs on the same day must average, not sum.

        Summing would make the category look like it sells twice as fast as any
        member of it, and every lot in that class would be under-discounted.
        """
        demand = pd.concat(
            [
                _demand("para-1", "MH-01", MIN_OBS_CLASS, units=4),
                _demand("para-2", "MH-01", MIN_OBS_CLASS, units=6),
            ]
        )
        history, tier = resolve_history(
            demand, "brand-new", "MH-01", "N02BE", "tablet", "OTC", drug_meta
        )
        assert tier is Tier.ATC_CLASS
        assert history["day"].is_unique
        assert history["units_sold"].iloc[0] == pytest.approx(5.0)  # mean, not 10


class TestAutoRepricingGate:
    """The safety property. A guess may inform a human; it may not act."""

    def test_rules_baseline_cannot_auto_reprice(self):
        assert Tier.RULES_BASELINE.supports_auto_repricing is False

    @pytest.mark.parametrize(
        "tier", [Tier.DRUG, Tier.ATC_CLASS, Tier.FORM_PRIOR]
    )
    def test_data_backed_tiers_may_auto_reprice(self, tier):
        assert tier.supports_auto_repricing is True


class TestRulesLadder:
    @pytest.mark.parametrize(
        "days,expected",
        [(10, 0.50), (29, 0.50), (45, 0.35), (75, 0.25), (150, 0.15), (400, 0.05)],
    )
    def test_discount_deepens_as_expiry_nears(self, days, expected):
        assert rules_discount(days) == expected

    def test_discount_is_monotonic(self):
        """Never suggest a deeper discount for a lot with more time left."""
        discounts = [rules_discount(d) for d in (5, 25, 45, 75, 150, 400)]
        assert discounts == sorted(discounts, reverse=True)

    def test_estimate_is_tagged_as_baseline(self):
        assert rules_baseline(30, 100).tier is Tier.RULES_BASELINE

    def test_quantiles_are_ordered(self):
        est = rules_baseline(45, 100)
        assert est.daily_units_p10 <= est.daily_units_p50 <= est.daily_units_p90

    def test_zero_days_does_not_divide_by_zero(self):
        est = rules_baseline(0, 50)
        assert est.daily_units_p50 > 0

    def test_larger_stock_implies_higher_required_rate(self):
        """200 units in the same window needs a faster clip than 100."""
        small = rules_baseline(45, 100).daily_units_p50
        large = rules_baseline(45, 200).daily_units_p50
        assert large > small
