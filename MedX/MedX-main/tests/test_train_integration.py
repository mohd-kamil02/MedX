"""End-to-end training test against real LightGBM.

Everything else in this suite stubs the model out to test policy. This one
actually trains, saves, reloads, and scores — the path that runs in production —
so the pipeline is proven to execute rather than merely to type-check.

Data is synthetic but built with a real signal: demand rises as price falls. If
training works, the models should recover that relationship.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ai.forecasting.baseline import Tier
from ai.forecasting.features import (
    DEMAND_FEATURES,
    SELLOUT_FEATURES,
    LotContext,
    build_history_features,
)
from ai.forecasting.predict import Forecaster
from ai.forecasting.train import (
    save_bundle,
    time_splits,
    train_demand_models,
    train_sellout_model,
)

pytest.importorskip("lightgbm")

N_DAYS = 400  # enough for 4 forward-chaining folds of 28 days


@pytest.fixture(scope="module")
def synthetic_demand():
    """Demand driven by price, weekday, and noise — a signal to be recovered."""
    rng = np.random.default_rng(7)
    start = date.today() - timedelta(days=N_DAYS)
    rows = []
    for drug in ("drug-a", "drug-b"):
        for i in range(N_DAYS):
            day = start + timedelta(days=i)
            price_ratio = rng.uniform(0.4, 1.0)
            weekday_lift = 1.4 if day.weekday() < 5 else 0.7
            # The relationship the model must learn: cheaper -> more units.
            expected = (6.0 * (1.0 - price_ratio) + 0.5) * weekday_lift
            rows.append(
                {
                    "drug_id": drug,
                    "region_code": "MH-01",
                    "day": day,
                    "units_sold": int(rng.poisson(max(expected, 0.01))),
                    "avg_price_ratio": price_ratio,
                    "active_listings": rng.integers(1, 8),
                    "total_available": rng.integers(50, 400),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained(synthetic_demand):
    models, metrics = train_demand_models(synthetic_demand, n_folds=4)
    return models, metrics


class TestTimeSplits:
    def test_validation_always_follows_training(self, synthetic_demand):
        """No fold may train on data that comes after what it validates on."""
        df = build_history_features(synthetic_demand).reset_index(drop=True)
        for train_idx, valid_idx in time_splits(df, n_folds=4):
            assert df.loc[train_idx, "day"].max() < df.loc[valid_idx, "day"].min()

    def test_folds_are_non_empty(self, synthetic_demand):
        df = build_history_features(synthetic_demand).reset_index(drop=True)
        for train_idx, valid_idx in time_splits(df, n_folds=4):
            assert len(train_idx) > 0 and len(valid_idx) > 0

    def test_insufficient_history_raises_clearly(self):
        """Better a loud error than a model trained on two weeks of data."""
        tiny = pd.DataFrame(
            {
                "drug_id": ["d"] * 10,
                "region_code": ["r"] * 10,
                "day": pd.date_range("2026-01-01", periods=10),
                "units_sold": [1] * 10,
                "avg_price_ratio": [0.8] * 10,
                "active_listings": [1] * 10,
                "total_available": [10] * 10,
            }
        )
        with pytest.raises(ValueError, match="distinct days"):
            list(time_splits(tiny, n_folds=4))


class TestDemandTraining:
    def test_produces_all_three_quantiles(self, trained):
        models, _ = trained
        assert set(models) == {"p10", "p50", "p90"}

    def test_quantiles_are_ordered_on_average(self, trained, synthetic_demand):
        """P10 <= P50 <= P90 in aggregate — the basic sanity of a quantile model."""
        models, _ = trained
        X = build_history_features(synthetic_demand)[DEMAND_FEATURES]
        p10, p50, p90 = (models[k].predict(X).mean() for k in ("p10", "p50", "p90"))
        assert p10 <= p50 <= p90

    def test_p90_coverage_is_roughly_calibrated(self, trained, synthetic_demand):
        """The metric that matters: ~90% of actuals should fall under P90."""
        _, metrics = trained
        assert 0.75 <= metrics["p90"]["coverage"] <= 0.99

    def test_p10_coverage_is_roughly_calibrated(self, trained):
        _, metrics = trained
        assert 0.02 <= metrics["p10"]["coverage"] <= 0.35

    def test_model_learns_that_cheaper_sells_more(self, trained, synthetic_demand):
        """The signal the whole pricing engine depends on.

        If this fails, price inversion is meaningless — the recommender would be
        walking a discount grid the model does not respond to.
        """
        models, _ = trained
        X = build_history_features(synthetic_demand)[DEMAND_FEATURES].copy()

        expensive = X.assign(price_ratio=0.95)
        cheap = X.assign(price_ratio=0.45)

        assert models["p50"].predict(cheap).mean() > models["p50"].predict(
            expensive
        ).mean()


class TestSelloutTraining:
    @pytest.fixture
    def lots(self, synthetic_demand):
        """Closed lots with an outcome correlated to shelf life and stock."""
        rng = np.random.default_rng(11)
        feat = build_history_features(synthetic_demand).dropna(
            subset=["units_mean_7d"]
        )
        sample = feat.sample(400, random_state=3).reset_index(drop=True)

        sample["days_to_expiry"] = rng.integers(10, 200, len(sample))
        sample["log_quantity_available"] = np.log1p(
            rng.integers(10, 500, len(sample))
        )
        sample["seller_rating"] = rng.uniform(2.0, 5.0, len(sample))
        sample["is_prescription_only"] = rng.integers(0, 2, len(sample))
        sample["listed_at"] = sample["day"]

        capacity = sample["days_to_expiry"] * (1.0 - sample["price_ratio"]) * 0.15
        sample["sold_out"] = (
            capacity > sample["log_quantity_available"] * 0.5
        ).astype(int)
        return sample

    def test_trains_and_beats_random(self, lots):
        _, metrics = train_sellout_model(lots)
        assert metrics["auc"] > 0.6, "classifier is no better than chance"

    def test_probabilities_are_well_formed(self, lots):
        model, _ = train_sellout_model(lots)
        preds = model.predict(lots[SELLOUT_FEATURES])
        assert ((preds >= 0) & (preds <= 1)).all()

    def test_missing_label_column_is_rejected(self, lots):
        with pytest.raises(ValueError, match="missing columns"):
            train_sellout_model(lots.drop(columns=["sold_out"]))


class TestBundleRoundTrip:
    def test_save_reload_and_score(self, trained, synthetic_demand, tmp_path):
        """The full production path: train -> disk -> reload -> score a real lot."""
        models, metrics = trained
        save_bundle(tmp_path, models, None, metrics, version="test-1")

        loaded = Forecaster(tmp_path)
        assert loaded.version == "test-1"
        assert set(loaded.demand_models) == {"p10", "p50", "p90"}

        ctx = LotContext(
            drug_id="drug-a",
            region_code="MH-01",
            expiry_date=date.today() + timedelta(days=45),
            quantity_available=100,
            mrp=50.0,
            current_price=45.0,
            price_floor=20.0,
            seller_rating=4.2,
            is_prescription_only=False,
            atc_code="N02BE",
            form="tablet",
        )
        history = build_history_features(synthetic_demand)
        history = history[history.drug_id == "drug-a"]

        result = loaded.score_lot("lot-1", ctx, history, Tier.DRUG)

        assert 0.0 <= result.p_sellout <= 1.0
        assert result.days_to_expiry == 45
        assert result.model_version == "test-1"
        if result.recommended_price is not None:
            assert ctx.price_floor <= result.recommended_price < ctx.current_price

    def test_manifest_records_both_feature_sets(self, trained, tmp_path):
        import json

        models, metrics = trained
        save_bundle(tmp_path, models, None, metrics, version="test-2")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["features"] == DEMAND_FEATURES
        assert manifest["sellout_features"] == SELLOUT_FEATURES
