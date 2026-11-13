import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml2 import data, dataset, evaluate, features, forward, models, portfolio, search, settings as S  # noqa: E402
from ml2.walkforward import fold_preds  # noqa: E402

CFG = {"groups": ["momentum", "market"], "cs_mode": "raw", "target": "rank", "horizon": 5, "model": "xgb",
       "params": {}}


def fake_dataset(T=900, n=12, seed=0) -> dataset.Dataset:
    """Random-walk prices with every row and feature present, about 3.5 years from 2012."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=T)
    tickers = pd.Index([f"T{i}" for i in range(n)])
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)), index=dates, columns=tickers)
    t, s = (a.ravel() for a in np.meshgrid(np.arange(T), np.arange(n), indexing="ij"))
    raw = [c for cols in features.GROUPS.values() for c in cols]
    names = raw + ["cs_" + c for g, cols in features.GROUPS.items() if g not in features.SHARED for c in cols]
    meta = {"names": names, "t": t.astype("int32"), "s": s.astype("int32"), "dates": dates, "tickers": tickers,
            "close": close, "fwd": {h: close.shift(-h) / close - 1 for h in dataset.HORIZONS},
            "seq": np.zeros((T, n, 5), "float32")}
    return dataset.Dataset(rng.normal(size=(len(t), len(names))).astype("float32"), meta)


def capture(monkeypatch):
    """Replace the model with a stub that records what each fold trained on."""
    seen = []

    def stub(name, params, Xtr, ytr, Xte, ctx):
        seen.append(ctx)
        return np.zeros(len(Xte))
    monkeypatch.setattr(models, "fit_predict", stub)
    return seen


def test_days_without_stock_prices_are_dropped_so_rolling_features_survive():
    idx = pd.bdate_range("2026-05-20", periods=6)
    close = pd.DataFrame(100.0, index=idx, columns=["SPY", "A", "B", "^VIX"])
    close.loc[idx[2], ["SPY", "A", "B"]] = np.nan  # a holiday that still has a VIX row
    px = {"close": close, "volume": close.copy()}
    out = data.drop_incomplete_days(px)
    assert idx[2] not in out["close"].index and len(out["close"]) == 5
    assert out["close"].index.equals(out["volume"].index)  # every table gets the same days


def test_v2_features_do_not_use_future_data():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=400)
    cols = ["SPY", "A", "B", "C", "D", "^VIX"]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, len(cols))), axis=0)), index=idx, columns=cols)
    vol = pd.DataFrame(rng.integers(1_000, 5_000, close.shape).astype(float), index=idx, columns=cols)
    px = {"close": close, "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "volume": vol}
    sector = pd.Series({"SPY": "ETF", "A": "X", "B": "X", "C": "Y", "D": "Y"})
    cut = idx[300]
    px2 = {k: v.copy() for k, v in px.items()}
    for k in ("close", "open", "high", "low"):
        px2[k].loc[idx > cut] *= 1.7  # change only the future
    px2["volume"].loc[idx > cut] *= 4
    a, _ = features.compute_wide(px, sector)
    b, _ = features.compute_wide(px2, sector)
    for name in a:  # every feature up to the cut must be identical
        pd.testing.assert_frame_equal(a[name].loc[:cut], b[name].loc[:cut], check_exact=False, rtol=1e-4, atol=1e-6)


def test_folds_respect_embargo_and_test_year(monkeypatch):
    ds = fake_dataset()
    seen = capture(monkeypatch)
    list(fold_preds(ds, [2014], CFG))
    ctx = seen[0]
    first_t = int(np.argmax(ds.dates.year.values == 2014))
    assert ds.t[ctx["tr"]].max() <= first_t - max(S.EMBARGO, CFG["horizon"]) - 1  # gap before the test year
    assert set(ds.dates.year[ds.t[ctx["te"]]]) == {2014}
    assert not set(ctx["tr"]) & set(ctx["te"])


def test_recency_weights_grow_with_time_and_halve_at_halflife(monkeypatch):
    ds = fake_dataset()
    seen = capture(monkeypatch)
    list(fold_preds(ds, [2014], {**CFG, "params": {"recency_halflife": 1}}))
    ctx = seen[0]
    w, t = ctx["weight"], ds.t[ctx["tr"]]
    first_t = int(np.argmax(ds.dates.year.values == 2014))
    assert np.all(np.diff(w[np.argsort(t)]) >= 0)                        # newer rows weigh more
    assert abs(w[t == first_t - 252][0] - 0.5) < 1e-9                    # one year back is half weight


def test_targets_are_ranks_and_unknown_in_the_last_horizon_days():
    ds = fake_dataset()
    y = ds.target("rank", 5)
    assert np.nanmin(y) >= -0.5 and np.nanmax(y) <= 0.5
    last_days = ds.t >= len(ds.dates) - 5
    assert np.isnan(y[last_days]).all() and not np.isnan(y[~last_days]).any()


def test_ic_is_one_for_perfect_and_minus_one_for_reversed_predictions():
    ds = fake_dataset()
    fwd = ds.fwd[5]
    assert evaluate.daily_ic(fwd, fwd).mean() > 0.999
    assert evaluate.daily_ic(-fwd, fwd).mean() < -0.999


def test_forward_scoring_matches_a_hand_computed_ranking():
    names = [f"S{i}" for i in range(20)]
    pred = pd.Series(np.arange(20.0), index=names)                         # S19 ranks best, S0 worst
    fwd = pd.Series(np.arange(20.0) / 100, index=names)                     # outcomes agree with the ranking
    r = forward.realized(pred, fwd, frac=0.1)
    assert abs(r["rank_ic"] - 1) < 1e-9
    assert abs(r["long_return"] - 0.185) < 1e-9 and abs(r["short_return"] - 0.005) < 1e-9  # top 2 and bottom 2
    assert abs(r["long_minus_short"] - 0.18) < 1e-9 and r["n"] == 20
    assert forward.realized(pred, -fwd)["rank_ic"] < -0.999                 # reversed outcomes flip the sign


def _row(ic: float, folds: list[float], sharpe: float) -> dict:
    return {"valid_ic": ic, "fold_ic": json.dumps(dict(zip(["2019", "2020", "2021"], folds))), "valid_ls_sharpe": sharpe}


def test_adoption_gate_needs_ic_gain_two_fold_wins_and_no_sharpe_loss():
    ref = _row(0.0334, [0.0287, 0.0767, -0.0055], 1.22)
    assert search.gate(_row(0.0380, [0.03, 0.08, 0.01], 1.30), ref)             # all three conditions hold
    assert not search.gate(_row(0.0359, [0.0111, 0.0892, 0.0073], 0.995), ref)  # the MLP case: Sharpe is lower
    assert not search.gate(_row(0.0350, [0.03, 0.08, 0.01], 1.30), ref)          # gain 0.0016 is under 0.003
    assert not search.gate(_row(0.0380, [0.03, 0.05, -0.01], 1.30), ref)         # wins only 1 of 3 years


def _pred(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.bdate_range("2020-01-01", periods=len(next(iter(values.values())))))


def test_long_short_book_is_dollar_neutral():
    pred = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 30)), index=pd.bdate_range("2020-01-01", periods=20))
    daily = evaluate.weights(pred, 0.1, "ls", hold=1)
    assert np.allclose(daily.sum(axis=1), 0, atol=1e-9)       # long and short sides offset
    assert np.allclose(daily.abs().sum(axis=1), 1, atol=1e-9)  # one daily book has gross exposure 1
    held = evaluate.weights(pred, 0.1, "ls", hold=5)
    assert np.allclose(held.sum(axis=1), 0, atol=1e-9)
    # averaging books lets a name that flips from long to short cancel, so gross can only shrink
    assert (held.abs().sum(axis=1) <= 1 + 1e-9).all()


def test_buffer_keeps_a_position_until_it_falls_below_the_exit_rank():
    # 10 names; X is top-ranked on day 1, slips to rank 2 (top 20%) on day 2, then to rank 5 on day 3
    base = {f"N{i}": [float(i)] * 3 for i in range(9)}
    base["X"] = [100.0, 7.5, 4.5]  # N8, N7 are higher on day 2; day 3 puts X above N0-N3 only
    w = portfolio.buffered(0.1, 0.3)(_pred(base), "long")
    assert w["X"].iloc[0] == 1.0       # entered as the top name
    assert w["X"].iloc[1] > 0          # still inside the 30% exit band, so kept
    assert w["X"].iloc[2] == 0         # dropped below the exit band, so sold
