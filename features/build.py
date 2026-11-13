"""Feature engineering. Every feature at date t uses only data up to and including close t."""
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
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _ticker_features(close, volume, high, low) -> pd.DataFrame:
    daily = close.pct_change()
    f = pd.DataFrame(index=close.index)
    for n in (1, 5, 10, 21, 63):
        f[f"ret_{n}"] = close.pct_change(n)
    f["vol_10"] = daily.rolling(10).std()
    f["vol_21"] = daily.rolling(21).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_21"]
    f["rsi_14"] = _rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close  # scaled by price
    f["dist_sma50"] = close / close.rolling(50).mean() - 1
    f["dist_sma200"] = close / close.rolling(200).mean() - 1
    f["vol_chg_5"] = volume.rolling(5).mean() / volume.rolling(21).mean() - 1
    f["vol_z_21"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
    top, bottom = high.rolling(21).max(), low.rolling(21).min()
    f["range_21"] = (top - bottom) / close
    f["close_in_range_21"] = (close - bottom) / (top - bottom).replace(0, np.nan)
    return f


def build_panel(prices: dict[str, pd.DataFrame], groups: list[str]) -> pd.DataFrame:
    """Long panel indexed by (date, ticker) with the chosen features, fwd_ret and label."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    close, volume, high, low = (prices[k] for k in ("close", "volume", "high", "low"))

    # market-wide context, shared by every ticker
    market = pd.DataFrame({
        "vix": close["^VIX"],
        "vix_chg_5": close["^VIX"].pct_change(5),
        "spy_ret_21": close["SPY"].pct_change(21),
        "spy_vol_21": close["SPY"].pct_change().rolling(21).std(),
    })

    frames = []
    for t in config.TICKERS:
        f = _ticker_features(close[t], volume[t], high[t], low[t]).join(market)
        f["fwd_ret"] = close[t].shift(-config.HORIZON) / close[t] - 1  # label source: uses FUTURE prices
        f["ticker"] = t
        f.index.name = "date"
        frames.append(f)
    panel = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)  # drop warm-up rows
    # the last HORIZON days have no future price, so their label stays NaN
    panel["label"] = (panel["fwd_ret"] > 0).astype(float).where(panel["fwd_ret"].notna())
    return panel[cols + ["fwd_ret", "label"]]
