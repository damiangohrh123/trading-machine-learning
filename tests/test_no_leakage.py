import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from features.build import FEATURE_GROUPS, build_panel  # noqa: E402
from models.walkforward import yearly_folds  # noqa: E402

ALL = list(FEATURE_GROUPS)


def _fake_prices(n=700, seed=0):
    """Random-walk prices for every configured ticker."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    names = config.TICKERS + config.MARKET_EXTRAS
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(names))), axis=0)),
                         index=idx, columns=names)
    volume = pd.DataFrame(rng.integers(1_000, 5_000, (n, len(names))).astype(float), index=idx, columns=names)
    return {"close": close, "volume": volume, "high": close * 1.01, "low": close * 0.99}


def test_features_do_not_use_future_data():
    # changing prices AFTER a cut date must leave every feature at or before it unchanged
    p1 = _fake_prices()
    p2 = {k: v.copy() for k, v in p1.items()}
    cut = p1["close"].index[500]
    for k in ("close", "high", "low"):
        p2[k].loc[p2[k].index > cut] *= 1.5
    p2["volume"].loc[p2["volume"].index > cut] *= 3
    a, b = build_panel(p1, ALL), build_panel(p2, ALL)
    cols = [c for g in FEATURE_GROUPS.values() for c in g]
    before = a.index[a.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(a.loc[before, cols], b.loc[before, cols])


def test_walk_forward_embargo():
    dates = pd.bdate_range("2012-01-01", "2021-12-31")
    for train_end, test_start, _ in yearly_folds(dates, 2016, 2021):
        assert len(dates[(dates > train_end) & (dates < test_start)]) >= config.EMBARGO
        # the label at train_end looks HORIZON days ahead and must land before the test period
        assert dates[dates.get_loc(train_end) + config.HORIZON] < test_start


def test_backtest_applies_weights_next_day():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [100, 110, 121, 121, 121.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 0, 0, 0, 0]}, index=idx)  # decided at the close of day 0
    gross = run_backtest(weights, close, cost_bps=0, slippage_bps=0)["gross_returns"]
    assert abs(gross.iloc[0]) < 1e-12        # nothing earned on the decision day
    assert abs(gross.iloc[1] - 0.10) < 1e-9  # earns the day-1 return
    assert abs(gross.iloc[2]) < 1e-12        # position was closed by the day-1 decision


def test_costs_reduce_returns():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 0, 1.0, 0, 1.0, 0]}, index=idx)  # trades every day
    free = run_backtest(weights, close, 0, 0)["net_returns"].sum()
    paid = run_backtest(weights, close, 10, 5)["net_returns"].sum()
    assert paid < free
