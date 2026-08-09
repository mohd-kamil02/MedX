"""Feature engineering for the demand and sellout models.

Both models read from `demand_daily`. The critical property of that table is that
it contains zero-sale days; the lag and rolling features below are only meaningful
because of it. If you ever swap the source for a query over `order_items`, these
features silently become "demand conditional on a sale having occurred", which is
a different and much more optimistic quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# Lag windows in days. 7 and 28 capture weekly and monthly cycles; 91 gives a
# quarter-length trend without reaching so far back that a new listing has no
# history at all.
LAG_WINDOWS = (7, 28, 91)

# Market-level: describes demand for a (drug, region) on a day. Every column here
# is derivable from `demand_daily` alone, which is what makes the demand model
# trainable directly off that table.
MARKET_FEATURES = [
    "price_ratio",
    "active_listings",
    "log_total_available",
    "dow",
    "month",
    "is_weekend",
    *[f"units_mean_{w}d" for w in LAG_WINDOWS],
    *[f"units_std_{w}d" for w in LAG_WINDOWS],
    *[f"price_ratio_mean_{w}d" for w in LAG_WINDOWS],
    "days_since_first_sale",
    "zero_sale_streak",
]

# Lot-level: properties of one seller's specific batch. These have no meaning at
# (drug, region, day) granularity — a single day's market demand is not a function
# of one seller's shelf life or rating.
LOT_FEATURES = [
    "days_to_expiry",
    "log_quantity_available",
    "seller_rating",
    "is_prescription_only",
]

# The demand model predicts market demand, so it sees market features only.
DEMAND_FEATURES = MARKET_FEATURES

# The sellout classifier predicts an outcome for one lot, so it sees both.
SELLOUT_FEATURES = MARKET_FEATURES + LOT_FEATURES


@dataclass(frozen=True)
class LotContext:
    """Everything about a listing the model needs, independent of the day scored."""

    drug_id: str
    region_code: str
    expiry_date: date
    quantity_available: int
    mrp: float
    current_price: float
    price_floor: float
    seller_rating: float
    is_prescription_only: bool
    atc_code: str | None
    form: str

    @property
    def price_ratio(self) -> float:
        return self.current_price / self.mrp if self.mrp else 1.0

    def days_to_expiry(self, as_of: date) -> int:
        return max((self.expiry_date - as_of).days, 0)


def build_history_features(demand: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features to a (drug_id, region_code, day) demand frame.

    `demand` must be dense — one row per drug/region/day including zeros. Rolling
    windows are shifted by one day so a row never sees its own target, which would
    leak the label into training.
    """
    if demand.empty:
        return demand.assign(**{c: np.nan for c in DEMAND_FEATURES if c not in demand})

    df = demand.sort_values(["drug_id", "region_code", "day"]).copy()
    df["day"] = pd.to_datetime(df["day"])

    keys = ["drug_id", "region_code"]
    grp = df.groupby(keys, sort=False)

    # shift(1) before rolling: a row must never see its own target, which would
    # leak the label into training.
    df["_lag_units"] = grp["units_sold"].shift(1)
    df["_lag_ratio"] = grp["avg_price_ratio"].shift(1)
    lagged = df.groupby(keys, sort=False)

    for window in LAG_WINDOWS:
        df[f"units_mean_{window}d"] = lagged["_lag_units"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        df[f"units_std_{window}d"] = lagged["_lag_units"].transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
        df[f"price_ratio_mean_{window}d"] = lagged["_lag_ratio"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )

    df = df.drop(columns=["_lag_units", "_lag_ratio"])

    # `avg_price_ratio` is the market's realized price level that day; it is the
    # same quantity the scorer varies when inverting the model for a price.
    df["price_ratio"] = df["avg_price_ratio"].astype(float).fillna(1.0)
    df["log_total_available"] = np.log1p(df["total_available"].fillna(0))

    # Calendar effects: pharmacy demand has a clear weekly cycle.
    df["dow"] = df["day"].dt.dayofweek
    df["month"] = df["day"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["days_since_first_sale"] = df.groupby(keys, sort=False).cumcount()
    df["zero_sale_streak"] = _zero_streak(df)

    # Lag columns are NaN for the first rows of each series, and deliberately left
    # that way. LightGBM branches on missing natively, so NaN reads as "no history
    # yet" — whereas filling a std with 0 would assert demand was perfectly stable,
    # which is a much stronger and false claim about a lot we know nothing about.
    return df


def _zero_streak(df: pd.DataFrame) -> pd.Series:
    """Consecutive zero-sale days immediately preceding each row.

    A long streak is the strongest single signal that a lot is not moving, and it
    is the feature that most often drives a `critical` alert.
    """
    out = np.zeros(len(df), dtype=np.int32)
    streak = 0
    prev_key = None
    prev_units = None

    for i, (key, units) in enumerate(
        zip(
            zip(df["drug_id"].to_numpy(), df["region_code"].to_numpy()),
            df["units_sold"].to_numpy(),
        )
    ):
        if key != prev_key:
            streak = 0
        else:
            streak = streak + 1 if prev_units == 0 else 0
        out[i] = streak
        prev_key, prev_units = key, units

    return pd.Series(out, index=df.index)


def lot_feature_row(
    ctx: LotContext,
    history: pd.DataFrame,
    as_of: date,
    price_override: float | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single-row feature frame for scoring one lot on one day.

    `columns` selects the feature set — DEMAND_FEATURES for the demand model,
    SELLOUT_FEATURES for the classifier. Defaults to the demand set.

    `price_override` is what makes the model invertible: `recommend_price` sweeps
    candidate prices through this function to ask "what would demand be at this
    price", then picks the shallowest discount that clears the lot.
    """
    columns = columns or DEMAND_FEATURES
    price = price_override if price_override is not None else ctx.current_price

    row: dict[str, float] = {
        "price_ratio": price / ctx.mrp if ctx.mrp else 1.0,
        "dow": as_of.weekday(),
        "month": as_of.month,
        "is_weekend": int(as_of.weekday() >= 5),
        # Lot-level; ignored when `columns` is DEMAND_FEATURES.
        "days_to_expiry": ctx.days_to_expiry(as_of),
        "log_quantity_available": float(np.log1p(ctx.quantity_available)),
        "seller_rating": ctx.seller_rating,
        "is_prescription_only": int(ctx.is_prescription_only),
    }

    if history.empty:
        # Caller should have routed to the cold-start baseline; fill neutrally so
        # this can never raise mid-scoring.
        for col in columns:
            row.setdefault(col, 0.0)
        row["active_listings"] = 1.0
        row["log_total_available"] = row["log_quantity_available"]
    else:
        recent = history.sort_values("day").iloc[-1]
        for col in columns:
            if col not in row:
                value = recent.get(col)
                row[col] = 0.0 if value is None or pd.isna(value) else float(value)

    return pd.DataFrame([row])[columns]
