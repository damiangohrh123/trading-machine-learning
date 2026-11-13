"""Phase 1: baselines on the DEVELOPMENT evaluation years (2016-2021)."""
import pandas as pd

import config
from backtest.baselines import equal_weight, sma_crossover, spy_buy_and_hold
from backtest.engine import run_backtest
from data.loader import load_prices

EVAL_START, EVAL_END = "2016-01-01", config.DEV_END


def main():
    close = load_prices()["close"].loc[:EVAL_END]
    strategies = {
        "spy_buy_hold": spy_buy_and_hold(close),
        "equal_weight": equal_weight(close, config.TICKERS),
        "sma_50_200": sma_crossover(close, config.TICKERS),
    }
    table = pd.DataFrame({name: run_backtest(w.loc[EVAL_START:], close)["metrics"]
                          for name, w in strategies.items()}).T
    table.to_csv(config.RESULTS_DIR / "baselines_dev.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
