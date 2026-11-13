"""Vectorised long-only backtester."""
import pandas as pd

import config
from backtest.metrics import summarize


def run_backtest(weights: pd.DataFrame, close: pd.DataFrame,
                 cost_bps: float = config.COST_BPS, slippage_bps: float = config.SLIPPAGE_BPS) -> dict:
    """weights: date x ticker targets decided at each date's close (sum <= 1, rest is cash)."""
    rets = close[weights.columns].pct_change().fillna(0.0)
    w = weights.reindex(rets.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)  # weights set at close t earn the t+1 return (no look-ahead)
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps + slippage_bps) / 1e4
    # report only the dates where the strategy produced weights
    days = weights.index
    return {
        "net_returns": net.loc[days],
        "gross_returns": gross.loc[days],
        "turnover": turnover.loc[days],
        "metrics": summarize(net.loc[days], turnover.loc[days]),
    }
