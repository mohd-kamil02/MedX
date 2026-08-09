"""Train the demand quantile models and the sellout classifier.

Two artifacts come out of this:

  demand_p10 / p50 / p90   LightGBM quantile regressors on daily units sold
  sellout                  LightGBM binary classifier, P(lot clears before expiry)

Quantiles rather than a point estimate because the decision downstream is a risk
decision. "Expected demand is 4/day" does not tell you whether to discount; "there
is a 20% chance demand is below 1.2/day" does.

Validation is a forward-chaining time split, never a random split. A random split
lets the model see next week while predicting this week, which produces a
beautiful offline score and a useless model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import DEMAND_FEATURES, SELLOUT_FEATURES, build_history_features

log = logging.getLogger(__name__)

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

_BASE_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}


def time_splits(df: pd.DataFrame, n_folds: int = 4, horizon_days: int = 28):
    """Forward-chaining splits: train on everything before a cutoff, validate after.

    Yields (train_idx, valid_idx) with each fold's validation window immediately
    following its training window — the same shape as production, where the model
    only ever sees the past.
    """
    days = np.sort(df["day"].unique())
    if len(days) < n_folds * horizon_days:
        raise ValueError(
            f"need >= {n_folds * horizon_days} distinct days to build {n_folds} folds, "
            f"got {len(days)}. Train on a single split or wait for more history."
        )

    for fold in range(n_folds):
        end = len(days) - (n_folds - fold - 1) * horizon_days
        cut = end - horizon_days
        train_days, valid_days = days[:cut], days[cut:end]
        yield (
            df.index[df["day"].isin(train_days)],
            df.index[df["day"].isin(valid_days)],
        )


def train_demand_models(
    demand: pd.DataFrame, n_folds: int = 4
) -> tuple[dict[str, lgb.Booster], dict]:
    """Fit the P10/P50/P90 quantile regressors on daily units sold."""
    df = build_history_features(demand).dropna(subset=["units_sold"])
    df = df.reset_index(drop=True)

    X, y = df[DEMAND_FEATURES], df["units_sold"]
    models: dict[str, lgb.Booster] = {}
    metrics: dict[str, dict] = {}

    *_, (train_idx, valid_idx) = time_splits(df, n_folds=n_folds)

    for name, alpha in QUANTILES.items():
        params = {**_BASE_PARAMS, "alpha": alpha}
        booster = lgb.train(
            params,
            lgb.Dataset(X.loc[train_idx], y.loc[train_idx]),
            num_boost_round=800,
            valid_sets=[lgb.Dataset(X.loc[valid_idx], y.loc[valid_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models[name] = booster

        pred = booster.predict(X.loc[valid_idx])
        metrics[name] = {
            "pinball_loss": float(_pinball(y.loc[valid_idx].to_numpy(), pred, alpha)),
            "best_iteration": booster.best_iteration,
            # Coverage is the metric that actually matters: for a well-calibrated
            # P90, ~90% of realized demand should fall below the prediction.
            "coverage": float((y.loc[valid_idx].to_numpy() <= pred).mean()),
        }
        log.info("demand_%s: %s", name, metrics[name])

    return models, metrics


def train_sellout_model(lots: pd.DataFrame) -> tuple[lgb.Booster, dict]:
    """Fit P(lot fully sells before expiry) on historical lots with known outcomes.

    `lots` must carry the SELLOUT_FEATURES columns as of listing time plus a
    boolean `sold_out` label. Only closed lots (expired or sold out) belong here —
    including still-active lots labels an unresolved outcome as a failure.

    This model sees lot-level features (shelf life, quantity, seller rating) that
    the demand model does not, because the outcome it predicts belongs to one lot.
    """
    required = {"sold_out", *SELLOUT_FEATURES}
    missing = required - set(lots.columns)
    if missing:
        raise ValueError(f"lots frame is missing columns: {sorted(missing)}")

    lots = lots.sort_values("listed_at").reset_index(drop=True)
    cut = int(len(lots) * 0.8)
    train, valid = lots.iloc[:cut], lots.iloc[cut:]

    params = {
        **{k: v for k, v in _BASE_PARAMS.items() if k not in ("objective", "metric")},
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
    }
    booster = lgb.train(
        params,
        lgb.Dataset(train[SELLOUT_FEATURES], train["sold_out"]),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(valid[SELLOUT_FEATURES], valid["sold_out"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = booster.predict(valid[SELLOUT_FEATURES])
    metrics = {
        "auc": float(_auc(valid["sold_out"].to_numpy(), pred)),
        "brier": float(np.mean((pred - valid["sold_out"].to_numpy()) ** 2)),
        "base_rate": float(valid["sold_out"].mean()),
        "best_iteration": booster.best_iteration,
    }
    log.info("sellout: %s", metrics)
    return booster, metrics


def _pinball(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    delta = y - pred
    return np.mean(np.maximum(alpha * delta, (alpha - 1) * delta))


def _auc(y: np.ndarray, pred: np.ndarray) -> float:
    """ROC AUC via the rank identity — avoids a sklearn dependency."""
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = pd.Series(pred).rank().to_numpy()
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def save_bundle(
    out_dir: Path,
    demand_models: dict[str, lgb.Booster],
    sellout_model: lgb.Booster | None,
    metrics: dict,
    version: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, booster in demand_models.items():
        booster.save_model(str(out_dir / f"demand_{name}.txt"))
    if sellout_model is not None:
        sellout_model.save_model(str(out_dir / "sellout.txt"))

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "trained_at": date.today().isoformat(),
                # Recorded so the serving side can refuse a bundle trained on a
                # different or reordered feature set — silently mismatched columns
                # produce confident nonsense rather than an error.
                "features": DEMAND_FEATURES,
                "sellout_features": SELLOUT_FEATURES,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    log.info("wrote model bundle %s to %s", version, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedX forecasting models")
    parser.add_argument("--demand-csv", type=Path, required=True)
    parser.add_argument("--lots-csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/current"))
    parser.add_argument("--version", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    demand = pd.read_csv(args.demand_csv, parse_dates=["day"])
    demand_models, metrics = train_demand_models(demand)

    sellout_model = None
    if args.lots_csv and args.lots_csv.exists():
        lots = pd.read_csv(args.lots_csv, parse_dates=["listed_at"])
        sellout_model, sellout_metrics = train_sellout_model(lots)
        metrics["sellout"] = sellout_metrics
    else:
        log.warning(
            "no --lots-csv; skipping sellout classifier. Scoring will fall back to "
            "integrating the demand quantiles, which is less accurate but usable."
        )

    save_bundle(args.out, demand_models, sellout_model, metrics, args.version)


if __name__ == "__main__":
    main()
