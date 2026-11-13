"""Run one experiment on the DEVELOPMENT period (walk-forward test years 2016-2021) and log it.

The held-out period (config.TEST_START onward) is never read here.
"""
import json
import time
from functools import lru_cache

import pandas as pd

import config
from backtest.engine import run_backtest
from data.loader import load_prices
from features.build import FEATURE_GROUPS, build_panel
from models.walkforward import probs_to_weights, walk_forward_predict

LOG = config.RESULTS_DIR / "experiments.csv"
FIRST_YEAR, LAST_YEAR = 2016, 2021


@lru_cache(maxsize=None)
def _prices() -> dict:
    return load_prices()


@lru_cache(maxsize=None)
def _panel(groups: tuple) -> pd.DataFrame:
    panel = build_panel(_prices(), list(groups))
    # hard cut so the dev loop cannot see held-out data
    return panel[panel.index.get_level_values("date") <= config.DEV_END]


def run_experiment(name: str, model: str, groups: tuple, model_params: dict, threshold: float = 0.5,
                   max_names: int | None = None, note: str = "") -> dict:
    """Walk-forward validate one config, append a row to the experiment log, return the row."""
    panel = _panel(groups)
    features = [c for g in groups for c in FEATURE_GROUPS[g]]
    prob = walk_forward_predict(panel, features, model, FIRST_YEAR, LAST_YEAR, model_params)
    weights = probs_to_weights(prob, threshold, max_names)
    metrics = run_backtest(weights, _prices()["close"].loc[: config.DEV_END])["metrics"]
    label = panel.loc[prob.index, "label"]
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "model": model, "groups": "+".join(groups),
        "params": json.dumps(model_params), "threshold": threshold, "max_names": max_names,
        "accuracy": round(float(((prob > 0.5) == (label > 0.5)).mean()), 4),  # sanity check only
        **{k: round(float(v), 4) for k, v in metrics.items()}, "note": note,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return row


if __name__ == "__main__":
    for model in ("logreg", "lgbm"):
        r = run_experiment(f"first_{model}", model, tuple(FEATURE_GROUPS), {}, note="phase2 all features")
        print(model, {k: r[k] for k in ("accuracy", "ann_return", "sharpe", "max_drawdown", "turnover")})
