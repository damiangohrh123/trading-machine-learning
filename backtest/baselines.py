from __future__ import annotations

import pandas as pd


def equal_weight_buy_and_hold(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Equal-weight, rebalanced monthly (cheap proxy for buy-and-hold of a basket)."""
    w = pd.DataFrame(0.0, index=close.index, columns=tickers)
    month = close.index.to_period("M")
    first_of_month = pd.Series(month, index=close.index).ne(pd.Series(month, index=close.index).shift())
    w.loc[:, :] = float("nan")
    w.loc[first_of_month.values, :] = 1.0 / len(tickers)
    return w.ffill().fillna(1.0 / len(tickers))


def spy_buy_and_hold(close: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=close.index, columns=["SPY"])
    w["SPY"] = 1.0
    return w


def sma_crossover(close: pd.DataFrame, tickers: list[str], fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Hold a ticker when fast SMA > slow SMA; equal-weight across held names, rest cash."""
    sig = pd.DataFrame(
        {t: (close[t].rolling(fast).mean() > close[t].rolling(slow).mean()).astype(float) for t in tickers}
    )
    sig[close[tickers].rolling(slow).mean().isna()] = 0.0
    return sig / len(tickers)
