"""Walk-forward (expanding window) training with an embargo gap."""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def make_logreg(C: float = 0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))


def make_lgbm(n_estimators: int = 200, learning_rate: float = 0.03, num_leaves: int = 15,
              min_child_samples: int = 200):
    # strong regularisation (subsampling, L2, large leaves) because daily returns are mostly noise
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                          min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=5.0, random_state=config.SEED,
                          verbose=-1, n_jobs=4)


MODELS = {"logreg": make_logreg, "lgbm": make_lgbm}


def yearly_folds(dates: pd.DatetimeIndex, first_year: int, last_year: int, embargo: int = config.EMBARGO):
    """Yield (train_end, test_start, test_end) per calendar year.

    Train uses dates <= train_end, which sits `embargo` trading days before the first test day,
    so no training label (which looks HORIZON days ahead) overlaps the test period.
    """
    days = pd.DatetimeIndex(sorted(dates.unique()))
    for year in range(first_year, last_year + 1):
        test_days = days[(days >= f"{year}-01-01") & (days <= f"{year}-12-31")]
        first = days.get_loc(test_days[0])
        yield days[first - embargo - 1], test_days[0], test_days[-1]


def walk_forward_predict(panel: pd.DataFrame, features: list[str], model_name: str,
                         first_year: int, last_year: int, model_params: dict) -> pd.Series:
    """Out-of-sample probability of an up-move for every (date, ticker) in the test years."""
    dates = panel.index.get_level_values("date")
    preds = []
    for train_end, test_start, test_end in yearly_folds(dates, first_year, last_year):
        train = panel[(dates <= train_end) & panel["label"].notna()]
        test = panel[(dates >= test_start) & (dates <= test_end)]
        model = MODELS[model_name](**model_params).fit(train[features].values, train["label"].values)
        proba = model.predict_proba(test[features].values)[:, 1]
        preds.append(pd.Series(proba, index=test.index, name="prob"))
    return pd.concat(preds)


def probs_to_weights(prob: pd.Series, threshold: float, max_names: int | None) -> pd.DataFrame:
    """Long-only weights: names with prob > threshold get 1/N_universe each, the rest is cash.

    With max_names set, only the top-k probabilities per day are kept, each with weight 1/k.
    """
    p = prob.unstack("ticker")
    held = (p > threshold).astype(float)
    if max_names is None:
        return held / p.shape[1]
    top_k = (p.rank(axis=1, ascending=False, method="first") <= max_names).astype(float)
    return held * top_k / max_names
