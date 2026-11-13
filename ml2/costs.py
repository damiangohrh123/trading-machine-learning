"""Cost sensitivity: how net Sharpe falls as one-way trading costs rise (cached predictions)."""
import json

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summarize
from ml2 import dataset, explore, portfolio, settings as S

COSTS = [0.0, 3.5, 7.0, 14.0, 21.0]  # one-way bps; 7 is the setting used everywhere else
MODELS = ("gru", "ensemble")
BOOKS = {"base decile, 5-day book": portfolio.decile(0.1, 5), "buffer 10/40": portfolio.buffered(0.1, 0.4)}


def main():
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    rows = []
    for model in MODELS:
        for tag, (a, b) in (("dev", S.DEV_YEARS), ("final", S.FINAL_YEARS)):
            years = list(range(a, b + 1))
            preds = explore.preds_for(ds, best, model, years, tag)
            for book, fn in BOOKS.items():
                for mode in ("ls", "long"):
                    weights = {y: fn(preds[y], mode) for y in years}  # built once, repriced at every cost level
                    sharpes, turnover = [], 0.0
                    for cost in COSTS:
                        runs = [run_backtest(weights[y], ds.close, cost_bps=cost, slippage_bps=0.0,
                                             borrow_bps=S.BORROW_BPS) for y in years]
                        m = summarize(pd.concat([r["net_returns"] for r in runs]),
                                      pd.concat([r["turnover"] for r in runs]))
                        sharpes.append(m["sharpe"])
                        turnover = m["turnover"]
                    slope, intercept = np.polyfit(COSTS, sharpes, 1)  # Sharpe is close to linear in cost
                    breakeven = -intercept / slope if intercept > 0 else float("nan")
                    rows.append({"model": model, "period": tag, "book": book, "mode": mode,
                                 "turnover_per_year": turnover,
                                 **{f"sharpe_{c:g}bps": s for c, s in zip(COSTS, sharpes)},
                                 "breakeven_bps": breakeven})
    table = pd.DataFrame(rows)
    table.to_csv(S.RESULTS / "cost_sensitivity.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
