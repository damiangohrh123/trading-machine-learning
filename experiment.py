"""Run one experiment on the DEVELOPMENT period (walk-forward test years 2016-2021) and log it.

The held-out period (config.TEST_START onward) is never touched here.
"""
from __future__ import annotations

import json
import time

import pandas as pd

import config
from backtest.engine import run_backtest
from data.loader import load_prices
from features.build import FEATURE_GROUPS, build_panel
from models.walkforward import probs_to_weights, walk_forward_predict

LOG = config.RESULTS_DIR / "experiments.csv"
FIRST_TEST_YEAR, LAST_TEST_YEAR = 2016, 2021

_prices = None
_panels: dict = {}


def _get_panel(groups: tuple[str, ...]) -> pd.DataFrame:
    global _prices
    if _prices is None:
        _prices = load_prices()
    if groups not in _panels:
        p = build_panel(_prices, list(groups))
        _panels[groups] = p[p.index.get_level_values("date") <= config.DEV_END]  # hard cut: no test data
    return _panels[groups]


def run_experiment(name: str, model: str = "lgbm", groups: tuple[str, ...] | None = None,
                   model_params: dict | None = None, threshold: float = 0.5,
                   max_names: int | None = None, note: str = "") -> dict:
    groups = tuple(groups or FEATURE_GROUPS)
    panel = _get_panel(groups)
    feats = [c for g in groups for c in FEATURE_GROUPS[g]]
    prob = walk_forward_predict(panel, feats, model, FIRST_TEST_YEAR, LAST_TEST_YEAR, model_params)
    weights = probs_to_weights(prob, threshold, max_names)
    close = _prices["close"].loc[: config.DEV_END]
    res = run_backtest(weights, close)
    m = res["metrics"]
    # classification quality (sanity only; trading metrics are what we optimise)
    lab = panel.loc[prob.index, "label"]
    acc = float(((prob > 0.5) == (lab > 0.5)).mean())
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "model": model,
        "groups": "+".join(groups), "params": json.dumps(model_params or {}),
        "threshold": threshold, "max_names": max_names, "accuracy": round(acc, 4),
        **{k: round(v, 4) for k, v in m.items()}, "note": note,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    res["row"] = row
    return res


if __name__ == "__main__":
    for model in ("logreg", "lgbm"):
        r = run_experiment(f"first_{model}", model=model, note="phase2 all features")
        print(model, {k: r["row"][k] for k in ("accuracy", "ann_return", "sharpe", "max_drawdown", "turnover")})
