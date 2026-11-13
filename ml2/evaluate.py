"""Predictions to rank IC, long/short portfolios and net-of-cost backtests."""
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summarize
from ml2 import settings as S


def to_wide(ds, rows: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Scatter per-row scores into a date x ticker frame (NaN where there is no prediction)."""
    wide = np.full(ds.close.shape, np.nan, dtype="float32")
    wide[ds.t[rows], ds.s[rows]] = scores
    return pd.DataFrame(wide, index=ds.dates, columns=ds.tickers)


def daily_ic(pred: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Spearman correlation between prediction and realised forward return, per date."""
    p, f = pred.where(fwd.notna()), fwd.where(pred.notna())
    return p.rank(axis=1).corrwith(f.rank(axis=1), axis=1).dropna()


def weights(pred: pd.DataFrame, top_frac: float, mode: str, hold: int) -> pd.DataFrame:
    """Top/bottom `top_frac` portfolios. 'long' is gross 1 long-only; 'ls' is 0.5 long and 0.5 short."""
    pred = pred.loc[pred.notna().any(axis=1)]
    pct = pred.rank(axis=1, pct=True)
    long = (pct > 1 - top_frac).astype(float)
    long = long.div(long.sum(axis=1), axis=0)
    if mode == "long":
        w = long
    else:
        short = (pct <= top_frac).astype(float)
        w = 0.5 * long - 0.5 * short.div(short.sum(axis=1), axis=0)
    # averaging the last `hold` daily books means each day trades only 1/hold of the portfolio
    return w.rolling(hold, min_periods=1).mean()


def backtest(pred: pd.DataFrame, close: pd.DataFrame, top_frac: float, mode: str, hold: int) -> dict:
    return run_backtest(weights(pred, top_frac, mode, hold), close, cost_bps=S.COST_BPS, slippage_bps=0.0,
                        borrow_bps=S.BORROW_BPS)


def equal_weight_universe(pred_days: pd.Index, close: pd.DataFrame) -> dict:
    """Baseline: hold every tradable ticker equally (daily rebalanced, rebalancing cost ignored)."""
    w = pd.DataFrame(1.0 / close.shape[1], index=pred_days, columns=close.columns)
    return run_backtest(w, close, cost_bps=S.COST_BPS, slippage_bps=0.0)


def score_folds(preds: dict[int, pd.DataFrame], ds, top_frac: float, hold: int) -> dict:
    """Per-fold and pooled metrics. IC is always measured against the 5-day forward return."""
    fwd = ds.fwd[S.HORIZON]
    ics = {y: daily_ic(p, fwd) for y, p in preds.items()}
    ls = {y: backtest(p, ds.close, top_frac, "ls", hold) for y, p in preds.items()}
    lo = {y: backtest(p, ds.close, top_frac, "long", hold) for y, p in preds.items()}
    pool = lambda runs, k: pd.concat([r[k] for r in runs.values()])  # noqa: E731
    ic_all = pd.concat(ics.values())
    ls_m = summarize(pool(ls, "net_returns"), pool(ls, "turnover"))
    lo_m = summarize(pool(lo, "net_returns"), pool(lo, "turnover"))
    return {
        "ic_mean": float(ic_all.mean()), "ic_ir": float(ic_all.mean() / ic_all.std() * np.sqrt(252)),
        "ls_sharpe": float(ls_m["sharpe"]), "ls_ann_return": float(ls_m["ann_return"]),
        "ls_max_dd": float(ls_m["max_drawdown"]), "ls_turnover": float(ls_m["turnover"]),
        "long_sharpe": float(lo_m["sharpe"]), "long_ann_return": float(lo_m["ann_return"]),
        "fold_ic": {y: float(v.mean()) for y, v in ics.items()},
        "fold_sharpe": {y: float(summarize(r["net_returns"], r["turnover"])["sharpe"]) for y, r in ls.items()},
        "ls_returns": pool(ls, "net_returns"), "long_returns": pool(lo, "net_returns"),
    }
