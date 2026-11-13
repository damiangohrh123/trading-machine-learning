"""Walk-forward (expanding window) training with an embargo gap."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def make_logreg(C: float = 0.1, **_):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))


def make_lgbm(n_estimators: int = 200, learning_rate: float = 0.03, num_leaves: int = 15,
              min_child_samples: int = 200, subsample: float = 0.8, colsample_bytree: float = 0.8,
              reg_lambda: float = 5.0, **_):
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        random_state=config.SEED, verbose=-1, n_jobs=4,
    )


MODELS: dict[str, Callable] = {"logreg": make_logreg, "lgbm": make_lgbm}


def yearly_folds(dates: pd.DatetimeIndex, first_test_year: int, last_test_year: int,
                 embargo: int = config.EMBARGO):
    """Yield (train_end_date, test_start, test_end). Train uses dates <= train_end_date, where
    train_end_date is `embargo` trading days before the first test day, so that no training label
    (which looks `HORIZON` days ahead) overlaps the test period."""
    udates = pd.DatetimeIndex(sorted(dates.unique()))
    for year in range(first_test_year, last_test_year + 1):
        test_dates = udates[(udates >= f"{year}-01-01") & (udates <= f"{year}-12-31")]
        if len(test_dates) == 0:
            continue
        first_pos = udates.get_loc(test_dates[0])
        train_end = udates[first_pos - embargo - 1]
        yield train_end, test_dates[0], test_dates[-1]


def walk_forward_predict(panel: pd.DataFrame, features: list[str], model_name: str,
                         first_test_year: int, last_test_year: int, model_params: dict | None = None,
                         return_importance: bool = False):
    model_params = model_params or {}
    dates = panel.index.get_level_values("date")
    preds = []
    imps = []
    for train_end, test_start, test_end in yearly_folds(dates, first_test_year, last_test_year):
        tr = panel[(dates <= train_end) & panel["label"].notna()]
        te = panel[(dates >= test_start) & (dates <= test_end)]
        model = MODELS[model_name](**model_params)
        model.fit(tr[features].values, tr["label"].values)
        p = model.predict_proba(te[features].values)[:, 1]
        preds.append(pd.Series(p, index=te.index, name="prob"))
        if return_importance and hasattr(model, "feature_importances_"):
            imps.append(pd.Series(model.feature_importances_, index=features))
    out = pd.concat(preds)
    if return_importance:
        return out, (pd.concat(imps, axis=1).mean(axis=1) if imps else None)
    return out


def probs_to_weights(prob: pd.Series, threshold: float = 0.5, max_names: int | None = None) -> pd.DataFrame:
    """Long-only: hold names with prob > threshold, equal-weighted at 1/N_universe each (rest cash).
    If max_names is set, keep only the top-k probabilities per day."""
    p = prob.unstack("ticker")
    sel = (p > threshold).astype(float)
    if max_names is not None:
        rank = p.rank(axis=1, ascending=False, method="first")
        sel = sel * (rank <= max_names).astype(float)
        return sel / max_names
    return sel / p.shape[1]
