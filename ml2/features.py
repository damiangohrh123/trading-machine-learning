"""Wide-table feature engineering. A feature at date t only uses data up to the close of t."""
import numpy as np
import pandas as pd

# stock-level features (differ per ticker) grouped for ablation; market/calendar are shared per date
GROUPS = {
    "momentum": ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_21", "ret_42", "ret_63", "ret_126", "mom_12_1"],
    "volatility": ["vol_5", "vol_10", "vol_21", "vol_63", "vol_ratio", "down_vol_21", "skew_21", "kurt_21",
                   "max_ret_21", "min_ret_21"],
    "technical": ["rsi_14", "macd_hist", "boll_z_20", "dist_sma_10", "dist_sma_20", "dist_sma_50",
                  "dist_sma_100", "dist_sma_200", "sma_ratio_50_200"],
    "range": ["dist_52w_high", "dist_52w_low", "range_21", "close_in_range_21"],
    "volume": ["vol_chg_5", "vol_z_21", "dollar_vol_21", "amihud_21"],
    "intraday": ["gap_1", "gap_mean_5", "gap_mean_21", "intra_1", "intra_mean_5", "intra_mean_21"],
    "beta": ["beta_63", "idio_vol_63", "rel_ret_5", "rel_ret_21"],
    "sector": ["sec_rel_ret_5", "sec_rel_ret_21", "sec_rel_vol_21"],
    "market": ["vix", "vix_chg_5", "vix_z_63", "spy_ret_5", "spy_ret_21", "spy_ret_63", "spy_vol_21"],
    "calendar": ["dow", "month"],
    "residual": ["sec_rel_ret_1", "sec_rel_ret_10", "idio_ret_5", "idio_ret_21"],
    "volvol": ["vol_of_vol_21", "vol_trend_126", "autocorr_21"],
    "liquidity": ["dvol_ratio_126", "vol_chg_63"],
    "breadth": ["disp_1", "disp_21", "breadth_1", "breadth_sma50", "spy_dd_252", "spy_above_sma200"],
    "earnings": ["days_since_earn", "last_surprise", "surprise_mean_4"],
    "earn_ahead": ["days_to_earn"],  # uses the announced date of the next report, kept apart in case it leaks
}
SHARED = ("market", "calendar", "breadth")  # same value for every ticker on a date, so no cross-sectional rank
NEW_GROUPS = ("residual", "volvol", "liquidity", "breadth")
EARN_GROUPS = ("earnings", "earn_ahead")
EARN_CAP = 126  # business days; also the value for a stock with no report before or after


def earnings_wide(events: pd.DataFrame, dates: pd.DatetimeIndex, stocks: list[str]) -> dict[str, pd.DataFrame]:
    """Earnings features per date and stock. A report counts from the first close that reflects it:
    the same day if released before the close, the next business day if released after."""
    day = dates.values.astype("datetime64[D]")
    out = {n: np.full((len(dates), len(stocks)), v, dtype="float32")
           for n, v in (("days_since_earn", EARN_CAP), ("days_to_earn", EARN_CAP), ("last_surprise", 0), ("surprise_mean_4", 0))}
    by_ticker = {t: g for t, g in events.groupby("ticker")}
    for j, t in enumerate(stocks):
        if t not in by_ticker:
            continue
        e = by_ticker[t].sort_values("date")
        d = e["date"].values.astype("datetime64[D]")
        react = np.where(e["after_close"].values, np.busday_offset(d, 1, roll="backward"),
                         np.busday_offset(d, 0, roll="forward"))
        keep = np.r_[True, np.diff(react) > 0]  # one report per reaction day
        react, sur = react[keep], np.clip(e["surprise"].fillna(0).values[keep], -50, 50) / 10
        mean4 = np.array([sur[max(0, k - 3):k + 1].mean() for k in range(len(sur))])
        k = np.searchsorted(react, day, side="right") - 1          # last report reflected by each close
        seen = k >= 0
        out["days_since_earn"][seen, j] = np.minimum(np.busday_count(react[k[seen]], day[seen]), EARN_CAP)
        out["last_surprise"][seen, j], out["surprise_mean_4"][seen, j] = sur[k[seen]], mean4[k[seen]]
        nxt = k + 1 < len(react)
        out["days_to_earn"][nxt, j] = np.minimum(np.busday_count(day[nxt], react[k[nxt] + 1]), EARN_CAP)
    return {n: pd.DataFrame(v, index=dates, columns=stocks) for n, v in out.items()}


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def compute_wide(px: dict[str, pd.DataFrame], sector: pd.Series, events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return ({feature: date x ticker frame}, close frame of tradable tickers)."""
    vix = px["close"]["^VIX"]
    stocks = [t for t in px["close"].columns if t != "^VIX"]
    c, o, h, l, v = (px[k][stocks] for k in ("close", "open", "high", "low", "volume"))
    r = c.pct_change()
    spy = r["SPY"]
    f = {}

    for n in (1, 2, 3, 5, 10, 21, 42, 63, 126):
        f[f"ret_{n}"] = c.pct_change(n)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1  # 12-month momentum skipping the last month

    for n in (5, 10, 21, 63):
        f[f"vol_{n}"] = r.rolling(n).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_63"]
    f["down_vol_21"] = r.clip(upper=0).rolling(21).std()
    f["skew_21"] = r.rolling(21).skew()
    f["kurt_21"] = r.rolling(21).kurt()
    f["max_ret_21"] = r.rolling(21).max()
    f["min_ret_21"] = r.rolling(21).min()

    f["rsi_14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c  # scaled by price
    f["boll_z_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    for n in (10, 20, 50, 100, 200):
        f[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    f["sma_ratio_50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1

    f["dist_52w_high"] = c / c.rolling(252).max() - 1
    f["dist_52w_low"] = c / c.rolling(252).min() - 1
    top, bottom = h.rolling(21).max(), l.rolling(21).min()
    f["range_21"] = (top - bottom) / c
    f["close_in_range_21"] = (c - bottom) / (top - bottom).replace(0, np.nan)

    dollar = (c * v).replace(0, np.nan)
    f["vol_chg_5"] = v.rolling(5).mean() / v.rolling(21).mean() - 1
    f["vol_z_21"] = (v - v.rolling(21).mean()) / v.rolling(21).std()
    f["dollar_vol_21"] = np.log(dollar.rolling(21).mean())
    f["amihud_21"] = np.log((r.abs() / dollar).rolling(21).mean().replace(0, np.nan))  # price impact proxy

    gap, intra = o / c.shift(1) - 1, c / o - 1
    f["gap_1"], f["intra_1"] = gap, intra
    for n in (5, 21):
        f[f"gap_mean_{n}"] = gap.rolling(n).mean()
        f[f"intra_mean_{n}"] = intra.rolling(n).mean()

    spy_var = spy.rolling(63).var()
    beta = r.rolling(63).cov(spy).div(spy_var, axis=0)
    f["beta_63"] = beta
    f["idio_vol_63"] = (r.rolling(63).var() - beta.pow(2).mul(spy_var, axis=0)).clip(lower=0) ** 0.5
    f["rel_ret_5"] = f["ret_5"].sub(f["ret_5"]["SPY"], axis=0)
    f["rel_ret_21"] = f["ret_21"].sub(f["ret_21"]["SPY"], axis=0)

    # sector-relative: stock minus the mean of its sector on the same date
    groups = sector.reindex(stocks).values
    for name, base in (("sec_rel_ret_5", "ret_5"), ("sec_rel_ret_21", "ret_21"), ("sec_rel_vol_21", "vol_21")):
        f[name] = f[base] - f[base].T.groupby(groups).transform("mean").T

    for name, base in (("sec_rel_ret_1", "ret_1"), ("sec_rel_ret_10", "ret_10")):
        f[name] = f[base] - f[base].T.groupby(groups).transform("mean").T
    spy_r5, spy_r21 = c["SPY"].pct_change(5), c["SPY"].pct_change(21)
    f["idio_ret_5"] = f["ret_5"].sub(beta.mul(spy_r5, axis=0))
    f["idio_ret_21"] = f["ret_21"].sub(beta.mul(spy_r21, axis=0))

    f["vol_of_vol_21"] = f["vol_5"].rolling(21).std() / f["vol_21"]
    f["vol_trend_126"] = f["vol_21"] / r.rolling(126).std()
    f["autocorr_21"] = r.rolling(21).corr(r.shift(1)).replace([np.inf, -np.inf], np.nan)
    f["dvol_ratio_126"] = np.log(dollar.rolling(21).mean() / dollar.rolling(126).mean())
    f["vol_chg_63"] = v.rolling(21).mean() / v.rolling(63).mean() - 1

    def shared(series):  # broadcast a per-date series over all tickers
        return pd.DataFrame(np.repeat(series.reindex(c.index).values[:, None], len(stocks), axis=1),
                            index=c.index, columns=stocks)
    f["vix"] = shared(vix)
    f["vix_chg_5"] = shared(vix.pct_change(5))
    f["vix_z_63"] = shared((vix - vix.rolling(63).mean()) / vix.rolling(63).std())
    spy_c = c["SPY"]
    f["spy_ret_5"], f["spy_ret_21"], f["spy_ret_63"] = (shared(spy_c.pct_change(n)) for n in (5, 21, 63))
    f["spy_vol_21"] = shared(spy.rolling(21).std())
    f["disp_1"] = shared(f["ret_1"].std(axis=1))
    f["disp_21"] = shared(f["ret_21"].std(axis=1))
    f["breadth_1"] = shared((f["ret_1"] > 0).mean(axis=1))
    f["breadth_sma50"] = shared((f["dist_sma_50"] > 0).mean(axis=1))
    f["spy_dd_252"] = shared(spy_c / spy_c.rolling(252).max() - 1)
    f["spy_above_sma200"] = shared((spy_c > spy_c.rolling(200).mean()).astype(float))
    f["dow"] = shared(pd.Series(c.index.day_of_week, index=c.index).astype("float32"))
    f["month"] = shared(pd.Series(c.index.month, index=c.index).astype("float32"))
    f.update(earnings_wide(events, c.index, stocks))
    return {k: v_.astype("float32") for k, v_ in f.items()}, c
