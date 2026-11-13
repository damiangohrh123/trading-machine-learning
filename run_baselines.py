"""Phase 1: baselines on the DEVELOPMENT period only (walk-forward test years 2016-2021)."""
from __future__ import annotations

import pandas as pd

import config
from backtest.baselines import equal_weight_buy_and_hold, sma_crossover, spy_buy_and_hold
from backtest.engine import run_backtest
from data.loader import load_prices

EVAL_START, EVAL_END = "2016-01-01", config.DEV_END


def main():
    close = load_prices()["close"]
    tickers = config.TICKERS
    strategies = {
        "spy_buy_hold": spy_buy_and_hold(close),
        "equal_weight_monthly": equal_weight_buy_and_hold(close, tickers),
        "sma_50_200": sma_crossover(close, tickers),
    }
    rows = {}
    for name, w in strategies.items():
        w = w.loc[EVAL_START:EVAL_END]
        res = run_backtest(w, close.loc[:EVAL_END])
        rows[name] = res["metrics"]
        res["net_returns"].to_csv(config.RESULTS_DIR / f"returns_{name}.csv")
    table = pd.DataFrame(rows).T
    config.RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(config.RESULTS_DIR / "baselines_dev.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
