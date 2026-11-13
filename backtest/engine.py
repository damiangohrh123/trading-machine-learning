"""Vectorised long-only backtester.

Timing convention (no look-ahead): weights decided with information up to the close of
day t are applied to the return of day t+1 (close t -> close t+1).
"""
from __future__ import annotations

import pandas as pd

import config
from backtest.metrics import summarize


def run_backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    cost_bps: float = config.COST_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
) -> dict:
    """weights: date x ticker target weights decided at close of date (sum <= 1, rest is cash)."""
    tickers = [t for t in weights.columns if t in close.columns]
    rets = close[tickers].pct_change()
    w = weights[tickers].reindex(rets.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)                      # weights actually held during day t
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps + slippage_bps) / 1e4
    net = gross - cost
    # restrict to dates where we have weights
    idx = weights.dropna(how="all").index
    net, gross, turnover = net.loc[net.index.isin(idx)], gross.loc[gross.index.isin(idx)], turnover.loc[turnover.index.isin(idx)]
    return {
        "net_returns": net,
        "gross_returns": gross,
        "turnover": turnover,
        "metrics": summarize(net, turnover),
    }
