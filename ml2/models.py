"""Model zoo behind one interface: fit on (Xtr, ytr), return a score per test row (higher = better)."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml2 import deep, settings as S

DEFAULTS = {
    "xgb": dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
                min_child_weight=100, reg_lambda=10.0),
    "lgbm": dict(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
                 min_child_samples=500, reg_lambda=10.0),
    "ridge": dict(alpha=1000.0),
    "mlp": dict(hidden=256, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=8, batch=8192),
    "gru": dict(hidden=256, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=8, batch=8192, seq_len=30),
}


def fit_predict(name: str, params: dict, Xtr, ytr, Xte, ctx: dict) -> np.ndarray:
    classify = ctx["target"] == "class_top"
    if name == "xgb":
        cls = xgb.XGBClassifier if classify else xgb.XGBRegressor
        m = cls(tree_method="hist", device="cuda", random_state=S.SEED, **params).fit(Xtr, ytr)
        m.set_params(device="cpu")  # numpy inputs live on the CPU, so predict there
        return m.predict_proba(Xte)[:, 1] if classify else m.predict(Xte)
    if name == "lgbm":
        cls = lgb.LGBMClassifier if classify else lgb.LGBMRegressor
        m = cls(subsample_freq=1, random_state=S.SEED, verbose=-1, n_jobs=16, **params).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1] if classify else m.predict(Xte)
    if name == "ridge":
        assert not classify, "ridge is regression only"
        return make_pipeline(StandardScaler(), Ridge(**params)).fit(Xtr, ytr).predict(Xte)
    if name in ("mlp", "gru"):
        return deep.fit_predict(params, Xtr, ytr, Xte, ctx, use_seq=name == "gru")
    raise ValueError(name)
