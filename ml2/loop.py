"""Iterative improvement loop on the DEVELOPMENT years only (the final test years are never read).

Each pass applies ONE change to the current best config and keeps it only if it improves enough AND
wins in most folds. Model/feature changes are judged on rank IC; portfolio changes on net Sharpe.
"""
import copy
import json
import sys
import time

import optuna
import pandas as pd

from ml2 import dataset, evaluate, models, settings as S
from ml2.features import GROUPS
from ml2.walkforward import run_folds

LOOP_YEARS = list(range(S.DEV_YEARS[0], S.DEV_YEARS[1] + 1))  # every dev year, 2013-2021
MAX_PASSES, PATIENCE = 50, 10
MIN_IC_GAIN, MIN_SHARPE_GAIN = 0.0015, 0.15
MIN_FOLD_WINS = 5  # a change must beat the current best in at least 5 of the 9 folds
OPTUNA_TRIALS = 25
LOG = S.RESULTS / "experiments.csv"
BEST = S.RESULTS / "best_config.json"

START = {"groups": list(GROUPS), "cs_mode": "cs", "target": "rank", "horizon": 5, "model": "xgb",
         "params": models.DEFAULTS["xgb"], "top_frac": 0.1, "hold": 5}


def setk(**kw):
    return lambda c: {**c, **kw}


def without(group):
    return lambda c: {**c, "groups": [g for g in c["groups"] if g != group]}


def xgb_param(**kw):
    return lambda c: {**c, "params": {**c["params"], **kw}}


def switch(model, **overrides):
    return lambda c: {**c, "model": model, "params": {**models.DEFAULTS[model], **overrides}}


# (description, change, kind): kind 'model' retrains and is judged on IC, 'portfolio' re-scores only
CANDIDATES = [
    ("raw features instead of ranks", setk(cs_mode="raw"), "model"),
    ("raw + rank features", setk(cs_mode="both"), "model"),
    ("target: cross-sectional z-score", setk(target="excess"), "model"),
    ("target: top-quintile classifier", setk(target="class_top"), "model"),
    ("train on 10-day horizon", setk(horizon=10), "model"),
    ("train on 21-day horizon", setk(horizon=21), "model"),
    *[(f"drop {g} features", without(g), "model") for g in
      ("momentum", "volatility", "technical", "range", "volume", "intraday", "beta", "sector", "market", "calendar")],
    ("xgb depth 4", xgb_param(max_depth=4), "model"),
    ("xgb depth 8", xgb_param(max_depth=8), "model"),
    ("xgb lr 0.03, 500 trees", xgb_param(learning_rate=0.03, n_estimators=500), "model"),
    ("xgb min_child_weight 500", xgb_param(min_child_weight=500), "model"),
    ("xgb colsample 0.3", xgb_param(colsample_bytree=0.3), "model"),
    ("xgb reg_lambda 50", xgb_param(reg_lambda=50.0), "model"),
    ("model: lightgbm", switch("lgbm"), "model"),
    ("model: MLP (GPU)", switch("mlp"), "model"),
    ("model: GRU + MLP (GPU)", switch("gru"), "model"),
    ("model: ridge", switch("ridge"), "model"),
    ("model: bigger MLP (512, 12 epochs)", switch("mlp", hidden=512, dropout=0.3, epochs=12), "model"),
    ("model: GRU with 60-day window", switch("gru", seq_len=60, hidden=512, dropout=0.3, epochs=12), "model"),
    ("model: bigger lightgbm (127 leaves)", switch("lgbm", num_leaves=127, n_estimators=600, learning_rate=0.03), "model"),
    ("model: deeper xgboost (depth 10)", switch("xgb", max_depth=10, n_estimators=600, learning_rate=0.03), "model"),
    ("portfolio: top/bottom 5%", setk(top_frac=0.05), "portfolio"),
    ("portfolio: top/bottom 20%", setk(top_frac=0.2), "portfolio"),
    ("portfolio: hold 10 days", setk(hold=10), "portfolio"),
    ("portfolio: hold 21 days", setk(hold=21), "portfolio"),
]


def log(name: str, desc: str, cfg: dict, m: dict, secs: float, kept: bool) -> None:
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "change": desc, "kept": kept,
           "model": cfg["model"], "cs_mode": cfg["cs_mode"], "target": cfg["target"], "horizon": cfg["horizon"],
           "n_groups": len(cfg["groups"]), "top_frac": cfg["top_frac"], "hold": cfg["hold"],
           "params": json.dumps(cfg["params"]),
           **{k: round(m[k], 4) for k in ("ic_mean", "ic_ir", "ls_sharpe", "ls_ann_return", "ls_turnover", "long_sharpe")},
           "fold_ic": json.dumps({y: round(v, 4) for y, v in m["fold_ic"].items()}), "secs": round(secs)}
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def tune_xgb(ds, cfg: dict) -> dict:
    """Optuna search over xgboost hyperparameters, scored by mean rank IC on the loop folds."""
    fwd = ds.fwd[S.HORIZON]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600), max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
            min_child_weight=trial.suggest_float("min_child_weight", 20, 2000, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True))
        preds = run_folds(ds, LOOP_YEARS, {**cfg, "params": params})
        return float(pd.concat([evaluate.daily_ic(p, fwd) for p in preds.values()]).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=S.SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return study.best_params


def accepted(kind: str, new: dict, best: dict) -> bool:
    if kind == "portfolio":
        wins = sum(new["fold_sharpe"][y] > best["fold_sharpe"][y] for y in LOOP_YEARS)
        return new["ls_sharpe"] - best["ls_sharpe"] > MIN_SHARPE_GAIN and wins >= MIN_FOLD_WINS
    wins = sum(new["fold_ic"][y] > best["fold_ic"][y] for y in LOOP_YEARS)
    return new["ic_mean"] - best["ic_mean"] > MIN_IC_GAIN and wins >= MIN_FOLD_WINS


def main(done: int = 0, patience: int = PATIENCE):
    """done > 0 resumes from the saved best config and skips the first `done` candidates already tried."""
    ds = dataset.load()
    best_cfg = json.loads(BEST.read_text()) if done else copy.deepcopy(START)
    t0 = time.time()
    best_preds = run_folds(ds, LOOP_YEARS, best_cfg)
    best = evaluate.score_folds(best_preds, ds, best_cfg["top_frac"], best_cfg["hold"])
    log("resume" if done else "start", "starting config", best_cfg, best, time.time() - t0, True)
    print(f"start ic={best['ic_mean']:.4f} sharpe={best['ls_sharpe']:.2f}", flush=True)

    misses, history = 0, []
    candidates = CANDIDATES + [("optuna: tune xgb hyperparameters", None, "optuna")]
    for i, (desc, change, kind) in enumerate(candidates[:MAX_PASSES], 1):
        if i <= done:
            continue
        t0 = time.time()
        if kind == "optuna":
            cfg = {**best_cfg, "model": "xgb", "params": tune_xgb(ds, {**best_cfg, "model": "xgb"})}
            kind = "model"
        else:
            cfg = change(copy.deepcopy(best_cfg))
        preds = best_preds if kind == "portfolio" else run_folds(ds, LOOP_YEARS, cfg)
        new = evaluate.score_folds(preds, ds, cfg["top_frac"], cfg["hold"])
        keep = accepted(kind, new, best)
        log(f"pass_{i:02d}", desc, cfg, new, time.time() - t0, keep)
        history.append({"pass": i, "change": desc, "kept": bool(keep), "ic": new["ic_mean"], "sharpe": new["ls_sharpe"]})
        print(f"pass {i:2d} | {desc:36s} ic={new['ic_mean']:.4f} sharpe={new['ls_sharpe']:+.2f} "
              f"{'KEEP' if keep else 'drop'} ({time.time() - t0:.0f}s)", flush=True)
        if keep:
            best_cfg, best, misses = cfg, new, 0
            best_preds = preds
        else:
            misses += 1
            if misses >= patience:
                print(f"stopping: {patience} passes in a row without improvement")
                break
    BEST.write_text(json.dumps(best_cfg, indent=2))
    (S.RESULTS / f"loop_history_from_{done}.json").write_text(json.dumps(history, indent=2))
    print("best:", json.dumps(best_cfg))


if __name__ == "__main__":
    # usage: python -m ml2.loop [candidates_already_done [patience]]
    main(*(int(a) for a in sys.argv[1:3]))
