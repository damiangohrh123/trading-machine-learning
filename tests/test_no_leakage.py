import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from features.build import FEATURE_GROUPS, build_panel  # noqa: E402
from models.walkforward import yearly_folds  # noqa: E402


def _fake_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    names = config.TICKERS + config.MARKET_EXTRAS
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(names))), axis=0)), index=idx, columns=names)
    vol = pd.DataFrame(rng.integers(1_000, 5_000, (n, len(names))).astype(float), index=idx, columns=names)
    return {"close": close, "volume": vol, "high": close * 1.01, "low": close * 0.99}


def test_features_do_not_use_future_data():
    """Changing prices AFTER date t must not change any feature at date t."""
    p1 = _fake_prices()
    p2 = {k: v.copy() for k, v in p1.items()}
    cut = p1["close"].index[500]
    for k in ("close", "high", "low"):
        p2[k].loc[p2[k].index > cut] *= 1.5
    p2["volume"].loc[p2["volume"].index > cut] *= 3
    a, b = build_panel(p1), build_panel(p2)
    feat_cols = [c for g in FEATURE_GROUPS.values() for c in g]
    ia = a.index[a.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(a.loc[ia, feat_cols], b.loc[ia, feat_cols])


def test_walk_forward_embargo():
    dates = pd.bdate_range("2012-01-01", "2021-12-31")
    for train_end, test_start, _ in yearly_folds(dates, 2016, 2021):
        gap = len(dates[(dates > train_end) & (dates < test_start)])
        assert gap >= config.EMBARGO
        # labels at train_end look HORIZON days ahead: must land before test_start
        pos = dates.get_loc(train_end)
        assert dates[pos + config.HORIZON] < test_start


def test_backtest_applies_weights_next_day():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [100, 110, 121, 121, 121.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0, 0, 0, 0]}, index=idx)   # decided at close of day 0
    res = run_backtest(w, close, cost_bps=0, slippage_bps=0)
    r = res["gross_returns"]
    assert abs(r.iloc[0]) < 1e-12              # no return on decision day
    assert abs(r.iloc[1] - 0.10) < 1e-9        # earns day-1 return
    assert abs(r.iloc[2]) < 1e-12              # position gone after day 1 decision


def test_costs_reduce_returns():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0, 1.0, 0, 1.0, 0]}, index=idx)
    free = run_backtest(w, close, 0, 0)["net_returns"].sum()
    paid = run_backtest(w, close, 10, 5)["net_returns"].sum()
    assert paid < free
