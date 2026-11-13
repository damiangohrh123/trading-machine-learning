"""Feature engineering. Every feature at date t uses only data up to and including close t."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

FEATURE_GROUPS = {
    "momentum": ["ret_1", "ret_5", "ret_10", "ret_21", "ret_63"],
    "volatility": ["vol_10", "vol_21", "vol_ratio"],
    "technical": ["rsi_14", "macd_hist", "dist_sma50", "dist_sma200"],
    "volume": ["vol_chg_5", "vol_z_21"],
    "market": ["vix", "vix_chg_5", "spy_ret_21", "spy_vol_21"],
    "range": ["range_21", "close_in_range_21"],
}


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ticker_features(close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    r1 = close.pct_change()
    f = pd.DataFrame(index=close.index)
    for n in (1, 5, 10, 21, 63):
        f[f"ret_{n}"] = close.pct_change(n)
    f["vol_10"] = r1.rolling(10).std()
    f["vol_21"] = r1.rolling(21).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_21"]
    f["rsi_14"] = _rsi(close)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    f["dist_sma50"] = close / close.rolling(50).mean() - 1
    f["dist_sma200"] = close / close.rolling(200).mean() - 1
    f["vol_chg_5"] = volume.rolling(5).mean() / volume.rolling(21).mean() - 1
    f["vol_z_21"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
    hh, ll = high.rolling(21).max(), low.rolling(21).min()
    f["range_21"] = (hh - ll) / close
    f["close_in_range_21"] = (close - ll) / (hh - ll).replace(0, np.nan)
    return f


def build_panel(prices: dict[str, pd.DataFrame], groups: list[str] | None = None) -> pd.DataFrame:
    """Long panel indexed by (date, ticker) with features, fwd_ret and label."""
    groups = groups or list(FEATURE_GROUPS)
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    close, volume, high, low = (prices[k] for k in ("close", "volume", "high", "low"))

    vix = close["^VIX"]
    spy = close["SPY"]
    market = pd.DataFrame(
        {
            "vix": vix,
            "vix_chg_5": vix.pct_change(5),
            "spy_ret_21": spy.pct_change(21),
            "spy_vol_21": spy.pct_change().rolling(21).std(),
        }
    )

    frames = []
    for t in config.TICKERS:
        f = _ticker_features(close[t], volume[t], high[t], low[t])
        f = f.join(market)
        f["fwd_ret"] = close[t].shift(-config.HORIZON) / close[t] - 1   # label source: FUTURE data
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    panel = panel.set_index(["date", "ticker"]).sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=cols)                       # features must exist
    panel["label"] = (panel["fwd_ret"] > 0).astype(float).where(panel["fwd_ret"].notna())
    return panel[cols + ["fwd_ret", "label"]]
