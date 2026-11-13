"""Expanding-window yearly retraining with an embargo gap."""
import numpy as np
import pandas as pd

from ml2 import evaluate, models, settings as S


def run_folds(ds, years: list[int], cfg: dict) -> dict[int, pd.DataFrame]:
    """Train on everything before each test year (minus the embargo) and predict that year.

    cfg keys: groups, cs_mode, target, horizon, model, params.
    """
    cols = ds.select(cfg["groups"], cfg["cs_mode"])
    X = ds.matrix(cols)
    y = ds.target(cfg["target"], cfg["horizon"])
    row_year = ds.dates.year.values[ds.t]
    embargo = max(S.EMBARGO, cfg["horizon"])
    preds = {}
    for year in years:
        first_t = int(np.argmax(ds.dates.year.values == year))
        train = np.flatnonzero((ds.t <= first_t - embargo - 1) & ~np.isnan(y))
        test = np.flatnonzero(row_year == year)
        scores = models.fit_predict(cfg["model"], cfg["params"], X[train], y[train], X[test],
                                    {"ds": ds, "tr": train, "te": test, "target": cfg["target"]})
        preds[year] = evaluate.to_wide(ds, test, scores)
    return preds
