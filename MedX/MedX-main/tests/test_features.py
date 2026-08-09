"""Feature-engineering tests.

The two things worth guarding here are label leakage and the market/lot feature
split. Both are silent failures: leakage produces a model that scores brilliantly
offline and fails in production, and a mismatched feature set produces confident
predictions from the wrong columns. Neither raises an exception on its own.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ai.forecasting.features import (
    DEMAND_FEATURES,
    LOT_FEATURES,
    MARKET_FEATURES,
    SELLOUT_FEATURES,
    LotContext,
    build_history_features,
    lot_feature_row,
)


@pytest.fixture
def dense_demand():
    """120 days x 2 drugs, including zero-sale days.

    Density matters: the lag features are only meaningful if days with no sales
    are present as rows. A sparse frame silently changes what they mean.
    """
    rng = np.random.default_rng(42)
    start = date.today() - timedelta(days=120)
    rows = [
        {
            "drug_id": drug,
            "region_code": "MH-01",
            "day": start + timedelta(days=i),
            "units_sold": int(rng.poisson(3 if drug == "drug-a" else 0.2)),
            "avg_price_ratio": 0.7,
            "active_listings": 4,
            "total_available": 200,
        }
        for drug in ("drug-a", "drug-b")
        for i in range(120)
    ]
    return pd.DataFrame(rows)


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


class TestLeakage:
    """A row must never see its own target. This is the highest-value test here."""

    def test_rolling_mean_excludes_current_day(self, dense_demand):
        feat = build_history_features(dense_demand)
        a = feat[feat.drug_id == "drug-a"].sort_values("day")

        expected = a["units_sold"].shift(1).rolling(7, min_periods=1).mean()
        pd.testing.assert_series_equal(
            a["units_mean_7d"], expected, check_names=False
        )

    def test_first_row_of_each_group_has_no_history(self, dense_demand):
        """No prior day exists, so the lag must be NaN — not 0.

        NaN reads as "unknown" to LightGBM, which learns a branch for it. Zero
        would assert "no demand", a much stronger and false claim.
        """
        feat = build_history_features(dense_demand)
        firsts = feat.sort_values("day").groupby(["drug_id", "region_code"]).head(1)
        assert firsts["units_mean_7d"].isna().all()

    def test_groups_do_not_bleed_into_each_other(self, dense_demand):
        """drug-b's sparse history must not inherit drug-a's busy history."""
        feat = build_history_features(dense_demand)
        b = feat[feat.drug_id == "drug-b"].sort_values("day")
        expected = b["units_sold"].shift(1).rolling(28, min_periods=1).mean()
        pd.testing.assert_series_equal(
            b["units_mean_28d"], expected, check_names=False
        )


class TestFeatureSets:
    def test_market_and_lot_features_are_disjoint(self):
        assert set(MARKET_FEATURES).isdisjoint(LOT_FEATURES)

    def test_demand_model_sees_market_features_only(self):
        assert DEMAND_FEATURES == MARKET_FEATURES

    def test_sellout_model_sees_both(self):
        assert set(SELLOUT_FEATURES) == set(MARKET_FEATURES) | set(LOT_FEATURES)

    def test_history_frame_supplies_every_demand_feature(self, dense_demand):
        """Regression test: training does df[DEMAND_FEATURES] and must not KeyError.

        This broke once — lot-level columns were in the demand list, and nothing
        in the pipeline produced them at (drug, region, day) granularity.
        """
        feat = build_history_features(dense_demand)
        missing = set(DEMAND_FEATURES) - set(feat.columns)
        assert not missing, f"training would KeyError on: {sorted(missing)}"

    def test_training_matrix_is_all_numeric(self, dense_demand):
        X = build_history_features(dense_demand)[DEMAND_FEATURES]
        assert X.select_dtypes("number").shape[1] == X.shape[1]


class TestScoringRow:
    def test_returns_requested_columns_in_order(self, dense_demand, ctx):
        feat = build_history_features(dense_demand)
        hist = feat[feat.drug_id == "drug-a"]

        demand_row = lot_feature_row(ctx, hist, date.today(), columns=DEMAND_FEATURES)
        sellout_row = lot_feature_row(ctx, hist, date.today(), columns=SELLOUT_FEATURES)

        assert list(demand_row.columns) == DEMAND_FEATURES
        assert list(sellout_row.columns) == SELLOUT_FEATURES

    def test_price_override_changes_price_ratio(self, dense_demand, ctx):
        """The hinge of the whole pricing engine: price must be a live input."""
        feat = build_history_features(dense_demand)
        hist = feat[feat.drug_id == "drug-a"]

        cheap = lot_feature_row(ctx, hist, date.today(), price_override=25.0)
        dear = lot_feature_row(ctx, hist, date.today(), price_override=45.0)

        assert cheap["price_ratio"].iloc[0] == pytest.approx(0.5)
        assert dear["price_ratio"].iloc[0] == pytest.approx(0.9)

    def test_empty_history_does_not_raise(self, ctx):
        """Cold start must degrade, never crash mid-scoring."""
        row = lot_feature_row(
            ctx, pd.DataFrame(), date.today(), columns=SELLOUT_FEATURES
        )
        assert row.shape == (1, len(SELLOUT_FEATURES))
        assert not row.isna().any().any()

    def test_lot_features_absent_from_demand_row(self, dense_demand, ctx):
        feat = build_history_features(dense_demand)
        row = lot_feature_row(ctx, feat, date.today(), columns=DEMAND_FEATURES)
        assert "seller_rating" not in row.columns
        assert "days_to_expiry" not in row.columns


class TestLotContext:
    def test_days_to_expiry_never_negative(self):
        expired = LotContext(
            "d", "r", date.today() - timedelta(days=10), 5, 50.0, 40.0,
            25.0, 4.0, False, None, "tablet",
        )
        assert expired.days_to_expiry(date.today()) == 0

    def test_price_ratio(self, ctx):
        assert ctx.price_ratio == pytest.approx(0.8)  # 40 / 50


class TestZeroSaleStreak:
    def test_counts_consecutive_zero_days(self):
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
        frame = pd.DataFrame(
            {
                "drug_id": ["d"] * 6,
                "region_code": ["r"] * 6,
                "day": days,
                "units_sold": [5, 0, 0, 0, 2, 0],
                "avg_price_ratio": [0.8] * 6,
                "active_listings": [1] * 6,
                "total_available": [10] * 6,
            }
        )
        streak = build_history_features(frame)["zero_sale_streak"].tolist()
        # Row i counts zero-days strictly before it: after 5,0,0,0 -> 0,0,1,2,
        # then a sale resets, so the row after 2 is back to 0.
        assert streak == [0, 0, 1, 2, 3, 0]

    def test_resets_between_drugs(self):
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(3)]
        frame = pd.DataFrame(
            {
                "drug_id": ["a", "a", "a", "b", "b", "b"],
                "region_code": ["r"] * 6,
                "day": days * 2,
                "units_sold": [0, 0, 0, 4, 4, 4],
                "avg_price_ratio": [0.8] * 6,
                "active_listings": [1] * 6,
                "total_available": [10] * 6,
            }
        )
        feat = build_history_features(frame)
        assert feat[feat.drug_id == "b"]["zero_sale_streak"].max() == 0
