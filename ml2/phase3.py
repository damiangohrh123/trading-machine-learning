"""Phase 3: feature-group ablation for XGBoost with the sector-neutral target.

    python -m ml2.phase3 tune|valid     # add each new group, add all, and drop each old group
"""
import gc
import json
import sys

import pandas as pd
import torch

from ml2 import dataset, evaluate, settings as S
from ml2.features import NEW_GROUPS
from ml2.search import TUNE_YEARS, VALID_YEARS
from ml2.walkforward import run_folds

KEYS = ("ic_mean", "ic_ir", "ls_sharpe", "ls_turnover", "long_sharpe")


def variants(base_groups: list[str]) -> dict[str, list[str]]:
    out = {"base": base_groups}
    out.update({f"+{g}": base_groups + [g] for g in NEW_GROUPS})
    out["+all new"] = base_groups + list(NEW_GROUPS)
    out.update({f"-{g}": [x for x in base_groups if x != g] for g in base_groups if g != "momentum"})
    # combinations picked from the tune-year table: drop market (best), intraday and volatility; add residual
    no_market = [x for x in base_groups if x != "market"]
    out["-market -intraday"] = [x for x in no_market if x != "intraday"]
    out["-market -volatility"] = [x for x in no_market if x != "volatility"]
    out["-market +residual"] = no_market + ["residual"]
    out["-market -intraday -volatility +residual"] = [x for x in no_market if x not in ("intraday", "volatility")] + ["residual"]
    return out


def main(stage: str) -> None:
    years = TUNE_YEARS if stage == "tune" else VALID_YEARS
    out = S.RESULTS / f"phase3_{stage}.csv"
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    cfg = {k: v for k, v in best.items() if k not in ("overrides", "top_frac", "hold")}
    cfg.update(target="sector")
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    seen = {r["variant"] for r in rows}
    for name, groups in variants(best["groups"]).items():
        if name in seen:
            continue
        m = evaluate.score_folds(run_folds(ds, years, {**cfg, "groups": groups}), ds, 0.1, 5)
        rows.append({"variant": name, **{k: m[k] for k in KEYS}})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(rows[-1], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
