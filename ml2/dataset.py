"""Stack wide features into one row-per-(date, ticker) table with ranks, targets and GRU sequences."""
import numpy as np
import pandas as pd

from ml2 import data, features, settings as S

CACHE_X = S.DATA / "v2_X.npy"
CACHE_META = S.DATA / "v2_meta.pkl"
HORIZONS = (5, 10, 21)


class Dataset:
    def __init__(self, X, meta):
        self.X = X                              # (rows, all features) float32, sorted by date
        self.__dict__.update(meta)              # names, t, s, dates, tickers, close, fwd, seq
        self.col = {n: i for i, n in enumerate(self.names)}

    def select(self, groups: list[str], cs_mode: str) -> list[str]:
        """Feature names for the chosen groups; cs_mode is 'raw', 'cs' (cross-sectional rank) or 'both'."""
        names = []
        for g in groups:
            for n in features.GROUPS[g]:
                if g in features.SHARED:
                    names.append(n)
                    continue
                if cs_mode in ("raw", "both"):
                    names.append(n)
                if cs_mode in ("cs", "both"):
                    names.append("cs_" + n)
        return names

    def matrix(self, names: list[str]) -> np.ndarray:
        return self.X[:, [self.col[n] for n in names]]

    def target(self, kind: str, horizon: int) -> np.ndarray:
        """Per-row training target built from forward returns; NaN where the future is unknown."""
        fwd = self.fwd[horizon]
        pct = fwd.rank(axis=1, pct=True)
        if kind == "rank":
            y = pct - 0.5
        elif kind == "excess":  # cross-sectional z-score of the forward return, clipped
            y = fwd.sub(fwd.mean(axis=1), axis=0).div(fwd.std(axis=1), axis=0).clip(-3, 3)
        elif kind == "resid":  # beta-adjusted return: strip the part explained by SPY, then rank
            beta = self.X[:, self.col["beta_63"]]
            resid = np.full(self.close.shape, np.nan, dtype="float32")
            resid[self.t, self.s] = fwd.values[self.t, self.s] - beta * fwd["SPY"].values[self.t]
            y = pd.DataFrame(resid, index=fwd.index, columns=fwd.columns).rank(axis=1, pct=True) - 0.5
        elif kind == "class_top":  # top quintile vs the rest
            y = (pct >= 0.8).astype("float32").where(fwd.notna())
        else:
            raise ValueError(kind)
        return y.values[self.t, self.s].astype("float32")


def build() -> Dataset:
    px = data.load()
    wide, close = features.compute_wide(px, data.sectors())
    for group, cols in features.GROUPS.items():
        if group not in features.SHARED:
            for n in cols:
                wide["cs_" + n] = (wide[n].rank(axis=1, pct=True) - 0.5).astype("float32")
    names = list(wide)
    cube = np.stack([wide[n].values for n in names], axis=-1)       # (dates, tickers, features)
    ok = np.isfinite(cube).all(axis=-1)                              # keep rows with every feature defined
    t, s = np.nonzero(ok)
    X = cube[ok]
    del cube

    stocks = list(close.columns)
    h, l = px["high"][stocks], px["low"][stocks]
    scale = wide["vol_63"] + 1e-4
    rng = (h - l) / close
    seq = np.stack([(wide["ret_1"] / scale).clip(-5, 5), (wide["gap_1"] / scale).clip(-5, 5),
                    (wide["intra_1"] / scale).clip(-5, 5), wide["vol_z_21"].clip(-5, 5),
                    (rng / rng.rolling(21).mean() - 1).clip(-3, 3)], axis=-1)
    meta = {
        "names": names, "t": t.astype("int32"), "s": s.astype("int32"), "dates": close.index,
        "tickers": pd.Index(stocks), "close": close,
        "fwd": {k: close.shift(-k) / close - 1 for k in HORIZONS},
        "seq": np.nan_to_num(seq.astype("float32")),                 # (dates, tickers, 5)
    }
    return Dataset(X, meta)


def save(ds: Dataset) -> None:
    np.save(CACHE_X, ds.X)
    pd.to_pickle({k: getattr(ds, k) for k in ("names", "t", "s", "dates", "tickers", "close", "fwd", "seq")}, CACHE_META)


def load() -> Dataset:
    return Dataset(np.load(CACHE_X), pd.read_pickle(CACHE_META))


if __name__ == "__main__":
    ds = build()
    save(ds)
    print("rows", len(ds.X), "features", len(ds.names), "tickers", len(ds.tickers),
          "dates", ds.dates[ds.t.min()].date(), "to", ds.dates[ds.t.max()].date())
