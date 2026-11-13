"""Download and cache daily adjusted prices."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

import config

CACHE = config.DATA_DIR / "prices.parquet"


def download(tickers: list[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                progress=False, group_by="column", threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:  # network / rate limit
            last_err = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Could not download price data: {last_err}")


def load_prices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return {'close': df, 'volume': df, 'high': df, 'low': df} indexed by date."""
    if CACHE.exists() and not refresh:
        flat = pd.read_parquet(CACHE)
    else:
        raw = download(config.TICKERS + config.MARKET_EXTRAS, config.START, config.END)
        flat = raw.copy()
        flat.columns = [f"{a}|{b}" for a, b in flat.columns]
        config.DATA_DIR.mkdir(exist_ok=True)
        flat.to_parquet(CACHE)
    cols = pd.MultiIndex.from_tuples([c.split("|") for c in flat.columns])
    df = flat.copy()
    df.columns = cols
    out = {f.lower(): df[f] for f in ("Close", "Volume", "High", "Low")}
    return out


if __name__ == "__main__":
    p = load_prices(refresh=True)
    c = p["close"]
    print(c.shape, c.index.min().date(), c.index.max().date())
    print("missing per ticker:\n", c.isna().sum()[c.isna().sum() > 0])
