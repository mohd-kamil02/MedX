"""Scoring and price recommendation.

This module is where the forecast becomes a product decision. `score_lot` answers
"will this clear?"; `recommend_price` inverts the demand model to answer "what
would make it clear?" — which is the dynamic pricing engine, obtained from the
same model rather than as a second system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baseline import BaselineEstimate, Tier, rules_baseline, rules_discount
from .features import (
    DEMAND_FEATURES, SELLOUT_FEATURES, LotContext, lot_feature_row,
)
from .types import DEFAULT_TARGET_SELLOUT, LotScore

log = logging.getLogger(__name__)

# Discount grid searched by recommend_price, as a fraction off MRP. Coarse by
# design — sellers do not act on a 3% difference, and a fine grid implies a
# precision the model does not have.
_DISCOUNT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


class Forecaster:
    """Loads a model bundle and scores lots.

    Instantiate once per worker process; the boosters are read-only and safe to
    share across threads.
    """

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        manifest_path = bundle_dir / "manifest.json"

        if not manifest_path.exists():
            # Perfectly valid state on day one: no bundle yet, everything routes
            # to the rules baseline until there is enough history to train.
            log.warning("no model bundle at %s; running rules-baseline only", bundle_dir)
            self.version = "rules-baseline"
            self.demand_models: dict[str, lgb.Booster] = {}
            self.sellout_model: lgb.Booster | None = None
            return

        manifest = json.loads(manifest_path.read_text())
        self.version = manifest["version"]

        if manifest.get("features") != DEMAND_FEATURES:
            raise RuntimeError(
                f"bundle {self.version} was trained on a different feature set. "
                "Retrain before serving — silently reordered features produce "
                "confident nonsense."
            )

        self.demand_models = {
            name: lgb.Booster(model_file=str(bundle_dir / f"demand_{name}.txt"))
            for name in ("p10", "p50", "p90")
        }
        sellout_path = bundle_dir / "sellout.txt"
        self.sellout_model = (
            lgb.Booster(model_file=str(sellout_path)) if sellout_path.exists() else None
        )

    # ------------------------------------------------------------------ scoring

    def score_lot(
        self,
        listing_id: str,
        ctx: LotContext,
        history: pd.DataFrame,
        tier: Tier,
        as_of: date | None = None,
        target_sellout: float = DEFAULT_TARGET_SELLOUT,
    ) -> LotScore:
        as_of = as_of or date.today()
        days = ctx.days_to_expiry(as_of)

        if tier is Tier.RULES_BASELINE or not self.demand_models:
            return self._score_from_rules(listing_id, ctx, days)

        daily = self._predict_daily(ctx, history, as_of, price=ctx.current_price)
        totals = {k: v * days for k, v in daily.items()}

        p_sellout = self._sellout_probability(ctx, history, as_of, totals)

        recommended = None
        if tier.supports_auto_repricing and p_sellout < target_sellout:
            recommended = self.recommend_price(
                ctx, history, as_of, target_sellout=target_sellout
            )

        return LotScore(
            listing_id=listing_id,
            days_to_expiry=days,
            p_sellout=round(float(p_sellout), 4),
            expected_units_p10=round(totals["p10"], 2),
            expected_units_p50=round(totals["p50"], 2),
            expected_units_p90=round(totals["p90"], 2),
            recommended_price=recommended,
            tier=tier,
            model_version=self.version,
        )

    def _score_from_rules(
        self, listing_id: str, ctx: LotContext, days: int
    ) -> LotScore:
        est: BaselineEstimate = rules_baseline(days, ctx.quantity_available)
        suggested = round(ctx.mrp * (1 - rules_discount(days)), 2)

        return LotScore(
            listing_id=listing_id,
            days_to_expiry=days,
            p_sellout=round(
                min(est.daily_units_p50 * days / max(ctx.quantity_available, 1), 1.0), 4
            ),
            expected_units_p10=round(est.daily_units_p10 * days, 2),
            expected_units_p50=round(est.daily_units_p50 * days, 2),
            expected_units_p90=round(est.daily_units_p90 * days, 2),
            # Deliberately None: a rules-tier score must not drive automated
            # repricing. The ladder's suggestion rides in payload for a human.
            recommended_price=None,
            tier=Tier.RULES_BASELINE,
            model_version=f"{self.version} (suggested={suggested})",
        )

    def _predict_daily(
        self, ctx: LotContext, history: pd.DataFrame, as_of: date, price: float
    ) -> dict[str, float]:
        row = lot_feature_row(
            ctx, history, as_of, price_override=price, columns=DEMAND_FEATURES
        )
        return {
            name: max(float(model.predict(row)[0]), 0.0)
            for name, model in self.demand_models.items()
        }

    def _sellout_probability(
        self,
        ctx: LotContext,
        history: pd.DataFrame,
        as_of: date,
        totals: dict[str, float],
    ) -> float:
        """P(cumulative demand >= quantity_available before expiry).

        Prefers the trained classifier. Without one, falls back to a normal
        approximation over the quantile spread — cruder, but honest: the P10/P90
        band already encodes the model's uncertainty.
        """
        if self.sellout_model is not None:
            row = lot_feature_row(ctx, history, as_of, columns=SELLOUT_FEATURES)
            return float(np.clip(self.sellout_model.predict(row)[0], 0.0, 1.0))

        need = ctx.quantity_available
        mu = totals["p50"]
        sigma = max((totals["p90"] - totals["p10"]) / 2.56, 1e-6)
        z = (mu - need) / sigma
        return float(np.clip(0.5 * (1 + _erf(z / np.sqrt(2))), 0.0, 1.0))

    # ------------------------------------------------------- price inversion

    def recommend_price(
        self,
        ctx: LotContext,
        history: pd.DataFrame,
        as_of: date | None = None,
        target_sellout: float = DEFAULT_TARGET_SELLOUT,
    ) -> float | None:
        """The shallowest price that still clears the lot before expiry.

        This is the pricing engine. Because `price_ratio` is a demand-model
        feature, we can ask the model what demand would be at each candidate
        price and stop at the first one that hits the target — the seller keeps
        every rupee of margin the stock did not require them to give up.

        Returns None if no price above the seller's floor achieves the target;
        the caller should escalate to a human rather than dumping the lot.
        """
        as_of = as_of or date.today()
        days = ctx.days_to_expiry(as_of)
        if days <= 0 or not self.demand_models:
            return None

        need = ctx.quantity_available

        for discount in _DISCOUNT_GRID:
            candidate = round(ctx.mrp * (1 - discount), 2)
            if candidate < ctx.price_floor:
                break  # grid is ascending in discount; nothing deeper is allowed
            if candidate >= ctx.current_price:
                continue  # never recommend a price increase

            daily = self._predict_daily(ctx, history, as_of, price=candidate)
            totals = {k: v * days for k, v in daily.items()}
            probe = replace(ctx, current_price=candidate)
            p = self._sellout_probability(probe, history, as_of, totals)
            if p >= target_sellout:
                return candidate

        return None


def _erf(x: float) -> float:
    """Abramowitz-Stegun 7.1.26. Avoids a scipy dependency for one function."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911,
    )
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
