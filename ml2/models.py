"""Model zoo behind one interface: fit on (Xtr, ytr), return a score per test row (higher = better)."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml2 import deep, settings as S

_NET = dict(hidden=256, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=8, batch=8192)
_XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
            min_child_weight=100, reg_lambda=10.0)
DEFAULTS = {
    "xgb": _XGB,
    "xgb_rank": {**_XGB, "objective": "rank:pairwise"},
    "lgbm": dict(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
                 min_child_samples=500, reg_lambda=10.0),
    "ridge": dict(alpha=1000.0),
    "mlp": _NET,
    "gru": {**_NET, "seq_len": 30},
    "tcn": {**_NET, "seq_len": 30},
    "transformer": {**_NET, "seq_len": 30},
}
SEQUENCE_ARCH = {"mlp": "none", "gru": "gru", "tcn": "tcn", "transformer": "transformer"}


def fit_predict(name: str, params: dict, Xtr, ytr, Xte, ctx: dict) -> np.ndarray:
    """ctx carries the dataset, row indices, target kind and optional per-row sample weights."""
    classify = ctx["target"] == "class_top"
    weight = ctx.get("weight")
    if name == "xgb":
        cls = xgb.XGBClassifier if classify else xgb.XGBRegressor
        m = cls(tree_method="hist", device="cuda", random_state=S.SEED, **params)
        m.fit(Xtr, ytr, sample_weight=weight)
        m.set_params(device="cpu")  # numpy inputs live on the CPU, so predict there
        return m.predict_proba(Xte)[:, 1] if classify else m.predict(Xte)
    if name == "xgb_rank":
        assert ctx["target"] == "rank" and weight is None, "xgb_rank needs the rank target and no weights"
        grade = np.clip(np.floor((ytr + 0.5) * 5), 0, 4).astype(int)  # quintile grades 0-4
        m = xgb.XGBRanker(tree_method="hist", device="cuda", random_state=S.SEED, **params)
        m.fit(Xtr, grade, qid=ctx["ds"].t[ctx["tr"]])  # one query per date; rows are date-sorted
        m.set_params(device="cpu")
        return m.predict(Xte)
    if name == "lgbm":
        cls = lgb.LGBMClassifier if classify else lgb.LGBMRegressor
        m = cls(subsample_freq=1, random_state=S.SEED, verbose=-1, n_jobs=16, **params)
        m.fit(Xtr, ytr, sample_weight=weight)
        return m.predict_proba(Xte)[:, 1] if classify else m.predict(Xte)
    if name == "ridge":
        assert not classify and weight is None, "ridge is plain regression"
        return make_pipeline(StandardScaler(), Ridge(**params)).fit(Xtr, ytr).predict(Xte)
    if name in SEQUENCE_ARCH:
        assert weight is None, "sample weights are only supported for tree models"
        return deep.fit_predict(params, Xtr, ytr, Xte, ctx, SEQUENCE_ARCH[name])
    raise ValueError(name)
