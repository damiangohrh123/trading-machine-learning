"""Expanding-window yearly retraining with an embargo gap."""
import numpy as np
import pandas as pd

from ml2 import evaluate, models, settings as S


def fold_preds(ds, years: list[int], cfg: dict, deadline: float | None = None):
    """Yield (year, predictions) after training on everything before that year (minus the embargo).

    cfg keys: groups, cs_mode, target, horizon, model, params. params may hold `recency_halflife`
    (years; 0 or absent = equal weights), which weights recent training rows more for tree models.
    deadline (epoch seconds) makes neural nets raise deep.TrialTimeout when they run past it.
    """
    params = dict(cfg["params"])
    halflife = params.pop("recency_halflife", 0)
    cols = ds.select(cfg["groups"], cfg["cs_mode"])
    X = ds.matrix(cols)
    y = ds.target(cfg["target"], cfg["horizon"])
    row_year = ds.dates.year.values[ds.t]
    embargo = max(S.EMBARGO, cfg["horizon"])
    for year in years:
        first_t = int(np.argmax(ds.dates.year.values == year))
        train = np.flatnonzero((ds.t <= first_t - embargo - 1) & ~np.isnan(y))
        test = np.flatnonzero(row_year == year)
        weight = 0.5 ** ((first_t - ds.t[train]) / (halflife * 252)) if halflife else None
        ctx = {"ds": ds, "tr": train, "te": test, "target": cfg["target"], "weight": weight, "deadline": deadline}
        scores = models.fit_predict(cfg["model"], params, X[train], y[train], X[test], ctx)
        yield year, evaluate.to_wide(ds, test, scores)


def run_folds(ds, years: list[int], cfg: dict) -> dict[int, pd.DataFrame]:
    return dict(fold_preds(ds, years, cfg))
