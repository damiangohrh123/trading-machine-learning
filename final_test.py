"""Phase 4: ONE run on the held-out period (2022-2025). No tuning after this.

Evaluated: the best dev config (results/best_config.json), the untuned LightGBM (reported, not
selected), and the baselines. Models retrain each calendar year on all earlier data plus an embargo.
"""
import json

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import pandas as pd

import config
from backtest.baselines import equal_weight, sma_crossover, spy_buy_and_hold
from backtest.engine import run_backtest
from data.loader import load_prices
from features.build import FEATURE_GROUPS, build_panel
from models.walkforward import MODELS, probs_to_weights, walk_forward_predict

R = config.RESULTS_DIR


def ml_returns(prices: dict, cfg: dict) -> dict:
    panel = build_panel(prices, cfg["groups"])
    features = [c for g in cfg["groups"] for c in FEATURE_GROUPS[g]]
    prob = walk_forward_predict(panel, features, cfg["model"], 2022, 2025, cfg["model_params"])
    return run_backtest(probs_to_weights(prob, cfg["threshold"], cfg["max_names"]), prices["close"])


def main():
    prices = load_prices()
    close = prices["close"]
    best = json.loads((R / "best_config.json").read_text())["cfg"]
    untuned_lgbm = dict(model="lgbm", groups=list(FEATURE_GROUPS), model_params={}, threshold=0.5, max_names=None)
    runs = {"ml_best_dev_config": ml_returns(prices, best), "ml_lgbm_untuned": ml_returns(prices, untuned_lgbm)}

    start = config.TEST_START
    baselines = {"spy_buy_hold": spy_buy_and_hold(close), "equal_weight": equal_weight(close, config.TICKERS),
                 "sma_50_200": sma_crossover(close, config.TICKERS)}
    for name, w in baselines.items():
        runs[name] = run_backtest(w.loc[start:config.END], close)

    table = pd.DataFrame({n: r["metrics"] for n, r in runs.items()}).T
    returns = pd.DataFrame({n: r["net_returns"] for n, r in runs.items()})
    table.to_csv(R / "final_test_metrics.csv")
    returns.to_csv(R / "final_test_returns.csv")
    print(table.round(3).to_string())

    ax = (1 + returns).cumprod().plot(figsize=(9, 4.5), title="Held-out 2022-2025: net equity curves (after costs)")
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(R / "final_test_equity.png", dpi=130)

    # standardized logistic coefficients from a fit on all dev data (fails loudly for non-logreg configs)
    panel = build_panel(prices, best["groups"])
    features = [c for g in best["groups"] for c in FEATURE_GROUPS[g]]
    dev = panel[(panel.index.get_level_values("date") <= config.DEV_END) & panel["label"].notna()]
    model = MODELS[best["model"]](**best["model_params"]).fit(dev[features].values, dev["label"].values)
    coef = pd.Series(model[-1].coef_[0], index=features).sort_values()
    coef.to_csv(R / "feature_coefficients.csv", header=["coef"])
    ax = coef.plot.barh(figsize=(7, 5), title="Logistic regression coefficients (standardized, dev data)")
    ax.figure.tight_layout()
    ax.figure.savefig(R / "feature_coefficients.png", dpi=130)


if __name__ == "__main__":
    main()
