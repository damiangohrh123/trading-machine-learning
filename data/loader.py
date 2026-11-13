"""Download and cache daily prices."""
import pandas as pd
import yfinance as yf

import config

CACHE = config.DATA_DIR / "prices.parquet"


def load_prices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return close/volume/high/low frames (date x ticker)."""
    if refresh or not CACHE.exists():
        # threads=False avoids a sqlite cache lock between parallel downloads
        raw = yf.download(config.TICKERS + config.MARKET_EXTRAS, start=config.START, end=config.END,
                          auto_adjust=True, progress=False, threads=False)
        assert not raw.empty, "price download returned no data"
        config.DATA_DIR.mkdir(exist_ok=True)
        # parquet needs flat string column names
        raw.columns = [f"{field}|{ticker}" for field, ticker in raw.columns]
        raw.to_parquet(CACHE)
    flat = pd.read_parquet(CACHE)
    flat.columns = pd.MultiIndex.from_tuples([c.split("|") for c in flat.columns])
    return {field.lower(): flat[field] for field in ("Close", "Volume", "High", "Low")}


if __name__ == "__main__":
    close = load_prices(refresh=True)["close"]
    assert not close.isna().any().any(), "missing prices"
    print(close.shape, close.index.min().date(), close.index.max().date())
