"""Phase 4: do earnings features help? Adds each earnings group to the adopted XGBoost, MLP and GRU configs.

    python -m ml2.phase4 tune|valid     # never the final years
"""
import gc
import json
import sys

import pandas as pd
import torch

from ml2 import dataset, evaluate, settings as S
from ml2.features import EARN_GROUPS
from ml2.search import TUNE_YEARS, VALID_YEARS
from ml2.walkforward import run_folds

KEYS = ("ic_mean", "ic_ir", "ls_sharpe", "ls_turnover", "long_sharpe")
VARIANTS = {"base": [], "+earnings": ["earnings"], "+earn_ahead": ["earn_ahead"], "+both": list(EARN_GROUPS)}


def base_cfgs() -> dict:
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    xgb = {k: v for k, v in best.items() if k not in ("overrides", "top_frac", "hold")}
    return {"xgb": xgb, **best["overrides"]}


def main(stage: str) -> None:
    years = TUNE_YEARS if stage == "tune" else VALID_YEARS
    out = S.RESULTS / f"phase4_{stage}.csv"
    ds = dataset.load()
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    seen = {(r["model"], r["variant"]) for r in rows}
    for name, cfg in base_cfgs().items():
        for variant, extra in VARIANTS.items():
            if (name, variant) in seen:
                continue
            m = evaluate.score_folds(run_folds(ds, years, {**cfg, "groups": cfg["groups"] + extra}), ds, 0.1, 5)
            rows.append({"model": name, "variant": variant, **{k: m[k] for k in KEYS}})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(rows[-1], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
