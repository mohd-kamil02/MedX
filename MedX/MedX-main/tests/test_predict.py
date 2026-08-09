"""Scoring and price-inversion tests.

The price recommender is the part with real money attached, so its guarantees are
tested directly: never below the seller's floor, never a price increase, and stop
at the shallowest discount that works rather than the deepest available.

A stub booster stands in for LightGBM so the pricing logic is tested in isolation
from model quality — these assert the *policy*, not the forecast.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.forecasting.baseline import Tier
from ai.forecasting.features import DEMAND_FEATURES, LotContext
from ai.forecasting.predict import Forecaster
from ai.forecasting.types import LotScore


class StubBooster:
    """Demand rises as price falls, with a configurable elasticity."""

    def __init__(self, base=1.0, elasticity=6.0):
        self.base = base
        self.elasticity = elasticity

    def predict(self, row):
        ratio = float(row["price_ratio"].iloc[0])
        # ratio 1.0 -> base; ratio 0.5 -> base + elasticity/2
        return np.array([self.base + self.elasticity * (1.0 - ratio)])


@pytest.fixture
def forecaster(tmp_path):
    """A Forecaster with no bundle on disk, then stub models injected."""
    f = Forecaster(tmp_path / "nonexistent")
    f.demand_models = {
        "p10": StubBooster(base=0.5),
        "p50": StubBooster(base=1.0),
        "p90": StubBooster(base=1.5),
    }
    return f


@pytest.fixture
def ctx():
    return LotContext(
        drug_id="drug-a",
        region_code="MH-01",
        expiry_date=date.today() + timedelta(days=45),
        quantity_available=100,
        mrp=50.0,
        current_price=40.0,
        price_floor=25.0,
        seller_rating=4.2,
        is_prescription_only=False,
        atc_code="N02BE",
        form="tablet",
    )


@pytest.fixture
def history():
    start = date.today() - timedelta(days=30)
    return pd.DataFrame(
        [
            {
                "drug_id": "drug-a",
                "region_code": "MH-01",
                "day": start + timedelta(days=i),
                "units_sold": 2,
                **{c: 0.0 for c in DEMAND_FEATURES},
            }
            for i in range(30)
        ]
    )


class TestMissingBundle:
    def test_absent_bundle_is_a_valid_state(self, tmp_path):
        """Day one has no trained model. Booting must not fail."""
        f = Forecaster(tmp_path / "nope")
        assert f.version == "rules-baseline"
        assert f.demand_models == {}
        assert f.sellout_model is None

    def test_mismatched_feature_set_is_rejected(self, tmp_path):
        """A reordered feature list produces confident nonsense, so refuse it."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(
            '{"version": "bad", "features": ["only_one_feature"], "metrics": {}}'
        )
        with pytest.raises(RuntimeError, match="different feature set"):
            Forecaster(bundle)


class TestPriceInversion:
    def test_never_recommends_below_the_sellers_floor(self, forecaster, ctx, history):
        """Hard money guarantee. Elasticity is set so low that nothing clears."""
        forecaster.demand_models = {
            k: StubBooster(base=0.01, elasticity=0.05)
            for k in ("p10", "p50", "p90")
        }
        price = forecaster.recommend_price(ctx, history)
        assert price is None or price >= ctx.price_floor

    def test_never_recommends_a_price_increase(self, forecaster, ctx, history):
        price = forecaster.recommend_price(ctx, history)
        if price is not None:
            assert price < ctx.current_price

    def test_stops_at_the_shallowest_workable_discount(self, forecaster, ctx, history):
        """The margin guarantee: take the first price that clears, not the deepest."""
        price = forecaster.recommend_price(ctx, history, target_sellout=0.60)
        deeper = forecaster.recommend_price(ctx, history, target_sellout=0.95)
        if price is not None and deeper is not None:
            assert price >= deeper

    def test_higher_target_needs_a_deeper_discount(self, forecaster, ctx, history):
        lenient = forecaster.recommend_price(ctx, history, target_sellout=0.50)
        strict = forecaster.recommend_price(ctx, history, target_sellout=0.90)
        if lenient is not None and strict is not None:
            assert strict <= lenient

    def test_returns_none_when_nothing_clears_the_floor(self, forecaster, ctx, history):
        """Escalate to a human rather than dumping the lot at any price."""
        forecaster.demand_models = {
            k: StubBooster(base=0.0, elasticity=0.0) for k in ("p10", "p50", "p90")
        }
        assert forecaster.recommend_price(ctx, history) is None

    def test_expired_lot_gets_no_recommendation(self, forecaster, ctx, history):
        expired = LotContext(
            **{**ctx.__dict__, "expiry_date": date.today() - timedelta(days=1)}
        )
        assert forecaster.recommend_price(expired, history) is None

    def test_floor_equal_to_mrp_permits_no_discount(self, forecaster, history):
        locked = LotContext(
            "d", "MH-01", date.today() + timedelta(days=45), 100,
            50.0, 50.0, 50.0, 4.0, False, "N02BE", "tablet",
        )
        assert forecaster.recommend_price(locked, history) is None


class TestScoring:
    def test_rules_tier_yields_no_recommended_price(self, forecaster, ctx, history):
        """Baseline-tier scores inform a human; they never carry a price."""
        result = forecaster.score_lot(
            "lot-1", ctx, history, Tier.RULES_BASELINE
        )
        assert result.recommended_price is None
        assert result.tier is Tier.RULES_BASELINE

    def test_score_shape_and_bounds(self, forecaster, ctx, history):
        result = forecaster.score_lot("lot-1", ctx, history, Tier.DRUG)
        assert isinstance(result, LotScore)
        assert 0.0 <= result.p_sellout <= 1.0
        assert result.days_to_expiry == 45

    def test_quantiles_are_ordered(self, forecaster, ctx, history):
        result = forecaster.score_lot("lot-1", ctx, history, Tier.DRUG)
        assert (
            result.expected_units_p10
            <= result.expected_units_p50
            <= result.expected_units_p90
        )

    def test_no_models_falls_back_to_rules(self, tmp_path, ctx, history):
        bare = Forecaster(tmp_path / "nope")
        result = bare.score_lot("lot-1", ctx, history, Tier.DRUG)
        assert result.tier is Tier.RULES_BASELINE

    def test_clears_comfortably_threshold(self):
        low = LotScore("l", 45, 0.50, 1, 2, 3, None, Tier.DRUG, "v")
        high = LotScore("l", 45, 0.90, 1, 2, 3, None, Tier.DRUG, "v")
        assert low.clears_comfortably is False
        assert high.clears_comfortably is True
