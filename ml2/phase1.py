"""Phase 1: compare training targets for three model families on the tune years, then validate on 2019-2021.

    python -m ml2.phase1 tune      # TUNE_YEARS only
    python -m ml2.phase1 valid     # VALID_YEARS, read after the tune table is fixed
"""
import gc
import json
import sys

import pandas as pd
import torch

from ml2 import dataset, evaluate, settings as S
from ml2.search import TUNE_YEARS, VALID_YEARS
from ml2.walkforward import run_folds

TARGETS = [("rank", 5), ("excess", 5), ("resid", 5), ("sector", 5), ("volscaled", 5), ("triple", 5),
           ("multi", 21), ("rank", 10)]
DROPPED = {"resid", "volscaled", "triple"}  # below the rank baseline on every model on the tune years
KEYS = ("ic_mean", "ic_ir", "ls_sharpe", "ls_turnover", "long_sharpe")


def base_cfgs() -> dict:
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    xgb = {k: v for k, v in best.items() if k not in ("overrides", "top_frac", "hold")}
    gru = best["overrides"]["gru"]
    lgbm = {**xgb, "model": "lgbm", "params": dict(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                                  subsample=0.7, colsample_bytree=0.5, min_child_samples=500,
                                                  reg_lambda=10.0)}
    return {"xgb": xgb, "lgbm": lgbm, "gru": gru}


def main(stage: str) -> None:
    years = TUNE_YEARS if stage == "tune" else VALID_YEARS
    targets = TARGETS if stage == "tune" else [t for t in TARGETS if t[0] not in DROPPED]
    out = S.RESULTS / f"phase1_{stage}.csv"
    ds = dataset.load()
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    seen = set(zip(done.get("model", []), done.get("target", []), done.get("horizon", [])))
    rows = done.to_dict("records")
    for target, horizon in targets:
        for name, cfg in base_cfgs().items():
            if (name, target, horizon) in seen:
                continue
            m = evaluate.score_folds(run_folds(ds, years, {**cfg, "target": target, "horizon": horizon}), ds, 0.1, 5)
            rows.append({"model": name, "target": target, "horizon": horizon, **{k: m[k] for k in KEYS}})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(rows[-1], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
