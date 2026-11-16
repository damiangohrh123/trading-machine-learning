"""Phase 2: turnover-penalised training for the GRU and MLP, scored on the tune years (then validation).

    python -m ml2.phase2 tune|valid
"""
import gc
import json
import sys

import pandas as pd
import torch

from ml2 import dataset, evaluate, portfolio, settings as S
from ml2.search import TUNE_YEARS, VALID_YEARS
from ml2.walkforward import run_folds

LAMBDAS = [None, 0.0, 0.1, 0.3, 1.0, 3.0]  # None = the original batch training
KEYS = ("ic_mean", "ls_sharpe", "ls_turnover", "long_sharpe")


def main(stage: str) -> None:
    years = TUNE_YEARS if stage == "tune" else VALID_YEARS
    out = S.RESULTS / f"phase2_{stage}.csv"
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())["overrides"]
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    seen = {(r["model"], r["lam"]) for r in rows}
    for name in ("gru", "mlp"):
        for lam in LAMBDAS:
            if (name, str(lam)) in seen:
                continue
            cfg = json.loads(json.dumps(best[name]))
            if lam is not None:
                cfg["params"].update(turn_lambda=lam, epochs=12)
            preds = run_folds(ds, years, cfg)
            m = evaluate.score_folds(preds, ds, 0.1, 5)
            sm = evaluate.score_folds({y: portfolio.ema(p, 5) for y, p in preds.items()}, ds, 0.1, 1,
                                      portfolio.decile(0.1, 1))
            rows.append({"model": name, "lam": str(lam), **{k: m[k] for k in KEYS},
                         "ema5_sharpe": sm["ls_sharpe"], "ema5_turnover": sm["ls_turnover"]})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(rows[-1], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
