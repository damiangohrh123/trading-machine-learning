import pandas as pd


def spy_buy_and_hold(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"SPY": 1.0}, index=close.index)


def equal_weight(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    # constant weights mean daily rebalancing, and the engine charges no cost for it
    return pd.DataFrame(1.0 / len(tickers), index=close.index, columns=tickers)


def sma_crossover(close: pd.DataFrame, tickers: list[str], fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Hold a ticker while its fast SMA is above its slow SMA; equal slots, rest is cash."""
    # SMAs are NaN during warm-up, and NaN comparisons are False, so no position is held then
    signal = pd.DataFrame({t: (close[t].rolling(fast).mean() > close[t].rolling(slow).mean()).astype(float)
                           for t in tickers})
    return signal / len(tickers)
