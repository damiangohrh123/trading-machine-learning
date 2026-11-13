"""Alternative score transforms and portfolio constructions built on the same model predictions."""
import numpy as np
import pandas as pd

from ml2 import evaluate


def ema(pred: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially smooth each ticker's score over time so rankings change more slowly."""
    days = pred.index[pred.notna().any(axis=1)]
    return pred.loc[days].ewm(halflife=halflife).mean()


def blend(pred: pd.DataFrame, other: pd.DataFrame, w: float) -> pd.DataFrame:
    """Cross-sectional rank average: (1 - w) * model + w * another signal."""
    days = pred.index[pred.notna().any(axis=1)]
    return (1 - w) * pred.loc[days].rank(axis=1, pct=True) + w * other.loc[days].rank(axis=1, pct=True)


def buffered(enter: float, exit_: float):
    """Hysteresis: open a position in the top `enter` fraction, keep it until it drops below `exit_`."""
    def fn(pred: pd.DataFrame, mode: str) -> pd.DataFrame:
        pred = pred.loc[pred.notna().any(axis=1)]
        pct = pred.rank(axis=1, pct=True).values
        valid = ~np.isnan(pct)
        long = np.zeros_like(pct, dtype=bool)
        short = np.zeros_like(pct, dtype=bool)
        for i in range(len(pct)):
            prev_l = long[i - 1] if i else np.zeros(pct.shape[1], bool)
            prev_s = short[i - 1] if i else np.zeros(pct.shape[1], bool)
            long[i] = valid[i] & ((pct[i] > 1 - enter) | (prev_l & (pct[i] > 1 - exit_)))
            short[i] = valid[i] & ((pct[i] <= enter) | (prev_s & (pct[i] <= exit_)))
        lw = pd.DataFrame(long.astype(float), index=pred.index, columns=pred.columns)
        lw = lw.div(lw.sum(axis=1), axis=0)
        if mode == "long":
            return lw
        sw = pd.DataFrame(short.astype(float), index=pred.index, columns=pred.columns)
        return 0.5 * lw - 0.5 * sw.div(sw.sum(axis=1), axis=0)
    return fn


def sector_neutral(sector: pd.Series, top_frac: float, hold: int):
    """Pick the top/bottom fraction inside each sector so the book carries no sector bet."""
    def fn(pred: pd.DataFrame, mode: str) -> pd.DataFrame:
        pred = pred.loc[pred.notna().any(axis=1)]
        groups = sector.reindex(pred.columns).values
        pct = pred.T.groupby(groups).rank(pct=True).T
        long = (pct > 1 - top_frac).astype(float)
        long = long.div(long.sum(axis=1), axis=0)
        if mode == "long":
            w = long
        else:
            short = (pct <= top_frac).astype(float)
            w = 0.5 * long - 0.5 * short.div(short.sum(axis=1), axis=0)
        return w.rolling(hold, min_periods=1).mean()
    return fn


def decile(top_frac: float, hold: int):
    return lambda pred, mode: evaluate.weights(pred, top_frac, mode, hold)
