"""Download the S&P 500 + ETF universe and cache wide (date x ticker) price tables."""
import io
import time

import pandas as pd
import requests
import yfinance as yf

from ml2 import settings as S

EARNINGS = S.DATA / "v2_earnings.parquet"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BATCH = 50


def universe() -> pd.DataFrame:
    """Current S&P 500 members with GICS sector, plus ETFs. Survivorship bias: today's members only."""
    html = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    stocks = pd.DataFrame({"ticker": table["Symbol"].str.replace(".", "-", regex=False),
                           "sector": table["GICS Sector"]})
    etfs = pd.DataFrame({"ticker": S.ETFS, "sector": "ETF"})
    return pd.concat([stocks, etfs]).drop_duplicates("ticker").reset_index(drop=True)


def download(refresh: bool = False) -> None:
    """Fetch every ticker in batches and save one parquet per price field."""
    S.DATA.mkdir(exist_ok=True)
    uni = universe()
    uni.to_csv(S.DATA / "universe_v2.csv", index=False)
    tickers = list(uni["ticker"]) + S.MARKET
    parts = {f: [] for f in S.FIELDS}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        # threads=False avoids a sqlite cache lock between parallel downloads
        raw = yf.download(batch, start=S.START, auto_adjust=True, progress=False, threads=False)
        for f in S.FIELDS:
            parts[f].append(raw[f])
        print(f"downloaded {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)
    for f in S.FIELDS:
        wide = pd.concat(parts[f], axis=1, sort=True)
        wide = wide.dropna(axis=1, how="all")  # tickers Yahoo returned nothing for
        wide.to_parquet(S.DATA / f"v2_{f.lower()}.parquet")
    missing = sorted(set(tickers) - set(wide.columns))
    print("tickers with no data:", missing)


def download_earnings() -> None:
    """Earnings dates and EPS surprise for every stock, one row per report. Needs about five minutes."""
    stocks = universe().query("sector != 'ETF'")["ticker"]
    rows = []
    for i, t in enumerate(stocks):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=100)
        except Exception as err:  # Yahoo drops some tickers; they simply have no earnings features
            print(f"{t}: {err}")
            continue
        if e is None or e.empty:
            continue
        local = e.index.tz_localize(None)
        rows.append(pd.DataFrame({"ticker": t, "date": local.normalize(), "after_close": local.hour >= 16,
                                  "surprise": e["Surprise(%)"].values}))
        if i % 100 == 99:
            print(f"earnings {i + 1}/{len(stocks)}", flush=True)
    pd.concat(rows).to_parquet(EARNINGS)
    print(f"earnings for {len(rows)} of {len(stocks)} stocks")


def earnings() -> pd.DataFrame:
    return pd.read_parquet(EARNINGS)


def drop_incomplete_days(px: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop days where under half the stocks have a price.

    Holidays that only carry a VIX row, or a day Yahoo has not filled yet, would otherwise put an all-missing
    row into every rolling window that contains it and silently remove all later rows.
    """
    stocks = [t for t in px["close"].columns if t != "^VIX"]
    keep = px["close"][stocks].notna().mean(axis=1) >= 0.5
    return {k: v.loc[keep] for k, v in px.items()}


def load() -> dict[str, pd.DataFrame]:
    """Wide float32 tables keyed by lowercase field name, on complete trading days only."""
    raw = {f.lower(): pd.read_parquet(S.DATA / f"v2_{f.lower()}.parquet").astype("float32") for f in S.FIELDS}
    return drop_incomplete_days(raw)


def sectors() -> pd.Series:
    return pd.read_csv(S.DATA / "universe_v2.csv").set_index("ticker")["sector"]


if __name__ == "__main__":
    download()
    # upcoming report dates change, so refresh the earnings table weekly
    if not EARNINGS.exists() or time.time() - EARNINGS.stat().st_mtime > 7 * 86400:
        download_earnings()
