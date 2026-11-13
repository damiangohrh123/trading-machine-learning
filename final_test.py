"""Phase 4: ONE run on the held-out period (2022-2025). No tuning after this.

Strategies evaluated (all fixed before looking at the test period):
  - best config from the dev loop (results/best_config.json)
  - the untuned Phase-2 LightGBM (reported for transparency, not selected)
  - baselines: SPY buy&hold, equal-weight monthly, SMA 50/200
Models are retrained each calendar year on all earlier data (expanding window + embargo).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from backtest.baselines import equal_weight_buy_and_hold, sma_crossover, spy_buy_and_hold
from backtest.engine import run_backtest
from data.loader import load_prices
from features.build import FEATURE_GROUPS, build_panel
from models.walkforward import MODELS, probs_to_weights, walk_forward_predict

R = config.RESULTS_DIR


def main():
    prices = load_prices()
    close = prices["close"]
    best = json.loads((R / "best_config.json").read_text())["cfg"]
    runs = {
        "ml_best_dev_config": best,
        "ml_lgbm_untuned": dict(model="lgbm", groups=list(FEATURE_GROUPS), model_params={}, threshold=0.5, max_names=None),
    }
    results, returns = {}, {}
    for name, cfg in runs.items():
        panel = build_panel(prices, cfg["groups"])
        feats = [c for g in cfg["groups"] for c in FEATURE_GROUPS[g]]
        prob = walk_forward_predict(panel, feats, cfg["model"], 2022, 2025, cfg["model_params"])
        w = probs_to_weights(prob, cfg["threshold"], cfg["max_names"])
        res = run_backtest(w, close)
        results[name], returns[name] = res["metrics"], res["net_returns"]

    start, end = config.TEST_START, config.END
    base = {
        "spy_buy_hold": spy_buy_and_hold(close),
        "equal_weight_monthly": equal_weight_buy_and_hold(close, config.TICKERS),
        "sma_50_200": sma_crossover(close, config.TICKERS),
    }
    for name, w in base.items():
        res = run_backtest(w.loc[start:end], close)
        results[name], returns[name] = res["metrics"], res["net_returns"]

    table = pd.DataFrame(results).T
    table.to_csv(R / "final_test_metrics.csv")
    pd.DataFrame(returns).to_csv(R / "final_test_returns.csv")
    print(table.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, r in returns.items():
        (1 + r.loc[start:end]).cumprod().plot(ax=ax, label=name)
    ax.set_title("Held-out 2022-2025: net equity curves (after costs)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(R / "final_test_equity.png", dpi=130)

    # interpretability: standardized logistic-regression coefficients (fit on all dev data)
    if best["model"] == "logreg":
        panel = build_panel(prices, best["groups"])
        feats = [c for g in best["groups"] for c in FEATURE_GROUPS[g]]
        dev = panel[(panel.index.get_level_values("date") <= config.DEV_END) & panel["label"].notna()]
        m = MODELS["logreg"](**best["model_params"]).fit(dev[feats].values, dev["label"].values)
        coef = pd.Series(m[-1].coef_[0], index=feats).sort_values()
        coef.to_csv(R / "feature_coefficients.csv", header=["coef"])
        fig, ax = plt.subplots(figsize=(7, 5))
        coef.plot.barh(ax=ax); ax.set_title("Logistic regression coefficients (standardized, dev data)")
        fig.tight_layout(); fig.savefig(R / "feature_coefficients.png", dpi=130)


if __name__ == "__main__":
    main()
