"""Block-bootstrap confidence intervals for the ensemble results (reads the cached predictions).

Resampling blocks of days keeps the autocorrelation of returns, so the intervals are not too narrow.
Paired statistics resample the same dates for both series.
"""
import pickle

import numpy as np
import pandas as pd

from ml2 import dataset, evaluate, settings as S
from ml2.final import factor_preds

BLOCK, N_BOOT, SEED = 20, 5000, 7
sharpe = lambda r: r.mean() / r.std() * np.sqrt(252)  # noqa: E731


def boot_ci(series: list[pd.Series], stat) -> tuple[float, float, float, float]:
    """Point estimate, 95% interval and the share of resamples above zero for stat(*series)."""
    data = pd.concat(series, axis=1).dropna().values
    n = len(data)
    rng = np.random.default_rng(SEED)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n, size=-(-n // BLOCK))
        idx = ((starts[:, None] + np.arange(BLOCK)) % n).ravel()[:n]  # circular blocks of consecutive days
        out[i] = stat(*data[idx].T)
    low, high = np.percentile(out, [2.5, 97.5])
    return stat(*data.T), low, high, float((out > 0).mean())


def main():
    ds = dataset.load()
    fwd = ds.fwd[S.HORIZON]
    rows = []
    for tag, (a, b) in (("dev", S.DEV_YEARS), ("final", S.FINAL_YEARS)):
        years = list(range(a, b + 1))
        preds = pickle.loads((S.RESULTS / f"preds_{tag}.pkl").read_bytes())
        ls = pd.concat([evaluate.backtest(preds[y], ds.close, 0.1, "ls", 5)["net_returns"] for y in years])
        lo = pd.concat([evaluate.backtest(preds[y], ds.close, 0.1, "long", 5)["net_returns"] for y in years])
        mom = factor_preds(ds, years)["factor_momentum_12_1"]
        mom_ls = pd.concat([evaluate.backtest(mom[y], ds.close, 0.1, "ls", 5)["net_returns"] for y in years])
        ew = evaluate.equal_weight_universe(lo.index, ds.close)["net_returns"]
        ic = pd.concat([evaluate.daily_ic(preds[y], fwd) for y in years])
        tests = {
            "rank IC (mean per day)": ([ic], lambda x: x.mean()),
            "long-short Sharpe": ([ls], sharpe),
            "long-only top decile Sharpe": ([lo], sharpe),
            "long-short Sharpe minus momentum factor": ([ls, mom_ls], lambda x, y: sharpe(x) - sharpe(y)),
            "long-only minus equal-weight (annual return)": ([lo, ew], lambda x, y: (x - y).mean() * 252),
        }
        for name, (series, stat) in tests.items():
            est, low, high, p_pos = boot_ci(series, stat)
            rows.append({"period": tag, "statistic": name, "estimate": est, "ci_low": low, "ci_high": high,
                         "share_positive": p_pos})
    table = pd.DataFrame(rows)
    table.to_csv(S.RESULTS / "significance.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
