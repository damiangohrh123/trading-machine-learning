from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "hit_rate", "n_days")}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    ann_ret = equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "turnover": float(turnover.mean() * TRADING_DAYS) if turnover is not None else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }
