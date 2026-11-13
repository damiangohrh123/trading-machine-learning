"""Unattended Optuna search over models, features, targets and training tweaks.

    python -m ml2.search run HOURS     # search on the tune years; resumable (trials live in search.db)
    python -m ml2.search finalists N   # score the N best tune-year trials once on the validation years

Protection against picking luck: trials are scored on TUNE_YEARS only. The VALID_YEARS block is read
only by `finalists`, and a finalist must beat the baseline config there to count as real.
"""
import gc
import json
import sys
import time

import optuna
import pandas as pd
import torch

from ml2 import dataset, deep, evaluate, models, settings as S
from ml2.features import GROUPS
from ml2.walkforward import fold_preds

TUNE_YEARS = list(range(2013, 2019))
VALID_YEARS = [2019, 2020, 2021]
DB = f"sqlite:///{(S.RESULTS / 'search.db').as_posix()}"
STUDY = "v2_search"
LOG = S.RESULTS / "search.csv"
MODEL_NAMES = ["xgb", "lgbm", "xgb_rank", "mlp", "gru", "tcn", "transformer"]
MAX_TRIAL_SECS = 600  # a trial past this budget is pruned, so heavy configs cannot stall the search


def _net_params(trial, sequence: bool) -> dict:
    p = dict(hidden=trial.suggest_categorical("hidden", [128, 256, 512, 1024]),
             dropout=trial.suggest_float("dropout", 0.0, 0.5),
             lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
             weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
             epochs=trial.suggest_int("epochs", 4, 16),
             batch=trial.suggest_categorical("batch", [4096, 8192, 16384]),
             loss=trial.suggest_categorical("loss", ["mse", "ic"]))
    if sequence:
        p["seq_len"] = trial.suggest_categorical("seq_len", [20, 30, 60])
        p["seq_hidden"] = trial.suggest_categorical("seq_hidden", [32, 64, 128])
    return p


def _tree_params(trial, model: str) -> dict:
    common = dict(n_estimators=trial.suggest_int("n_estimators", 100, 800),
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                  subsample=trial.suggest_float("subsample", 0.5, 1.0),
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
                  reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True))
    if model == "lgbm":
        return {**common, "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 100, 2000, log=True),
                "recency_halflife": trial.suggest_categorical("recency_halflife", [0, 2, 5])}
    p = {**common, "max_depth": trial.suggest_int("max_depth", 3, 10),
         "min_child_weight": trial.suggest_float("min_child_weight", 20, 2000, log=True)}
    if model == "xgb_rank":
        p["objective"] = trial.suggest_categorical("rank_objective", ["rank:pairwise", "rank:ndcg"])
    else:
        p["recency_halflife"] = trial.suggest_categorical("recency_halflife", [0, 2, 5])  # years
    return p


def sample_cfg(trial) -> dict:
    model = trial.suggest_categorical("model", MODEL_NAMES)
    # momentum always stays so at least one stock-level group is present
    groups = ["momentum"] + [g for g in GROUPS if g != "momentum" and trial.suggest_categorical(f"use_{g}", [True, False])]
    cfg = {"model": model, "groups": groups, "cs_mode": trial.suggest_categorical("cs_mode", ["raw", "cs", "both"]),
           "horizon": trial.suggest_categorical("horizon", [5, 10])}
    cfg["target"] = "rank" if model == "xgb_rank" else trial.suggest_categorical("target", ["rank", "excess", "class_top"])
    if model in models.SEQUENCE_ARCH:
        cfg["params"] = _net_params(trial, model != "mlp")
        if cfg["target"] == "class_top":
            cfg["params"]["loss"] = "mse"  # ignored by the classifier head, kept explicit for the log
    else:
        cfg["params"] = _tree_params(trial, model)
    return cfg


def log_trial(study, trial) -> None:
    row = {"number": trial.number, "state": trial.state.name, "ic_tune": trial.value,
           "secs": round((trial.datetime_complete - trial.datetime_start).total_seconds()),
           "fold_ic": json.dumps(trial.user_attrs.get("fold_ic")), "cfg": json.dumps(trial.user_attrs.get("cfg"))}
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    best = study.best_value if any(t.value is not None for t in study.trials) else float("nan")
    print(f"trial {trial.number:4d} {trial.state.name:8s} ic={trial.value if trial.value is not None else float('nan'):.4f} "
          f"best={best:.4f} ({row['secs']}s)", flush=True)


def run(hours: float) -> None:
    ds = dataset.load()
    fwd = ds.fwd[S.HORIZON]

    def objective(trial):
        cfg = sample_cfg(trial)
        trial.set_user_attr("cfg", cfg)
        ics = []
        deadline = time.time() + MAX_TRIAL_SECS
        try:
            for step, (_, pred) in enumerate(fold_preds(ds, TUNE_YEARS, cfg, deadline)):
                ics.append(float(evaluate.daily_ic(pred, fwd).mean()))
                trial.report(sum(ics) / len(ics), step)  # running mean IC lets weak trials stop early
                if trial.should_prune() or time.time() > deadline:
                    raise optuna.TrialPruned()
        except (optuna.TrialPruned, deep.TrialTimeout):
            trial.set_user_attr("fold_ic", ics)
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        trial.set_user_attr("fold_ic", ics)
        gc.collect()
        torch.cuda.empty_cache()
        return sum(ics) / len(ics)

    study = optuna.create_study(study_name=STUDY, storage=DB, load_if_exists=True, direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1))
    # seed by trial count so a restart does not replay the random draws of earlier runs
    study.sampler = optuna.samplers.TPESampler(seed=S.SEED + len(study.trials), n_startup_trials=30)
    # out-of-memory and bad-hyperparameter errors fail the trial only; anything else stops the search
    study.optimize(objective, timeout=hours * 3600, callbacks=[log_trial],
                   catch=(torch.cuda.OutOfMemoryError, ValueError))


def finalists(n: int) -> None:
    """Score the n best tune-year trials (plus the baseline) once on the validation years."""
    ds = dataset.load()
    study = optuna.load_study(study_name=STUDY, storage=DB)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    unique = {json.dumps(t.user_attrs["cfg"], sort_keys=True): t for t in sorted(done, key=lambda t: t.value)}
    top = sorted(unique.values(), key=lambda t: t.value, reverse=True)[:n]  # identical configs count once
    base = json.loads((S.RESULTS / "best_config.json").read_text())
    entries = [("baseline", base, None)] + [(f"trial_{t.number}", t.user_attrs["cfg"], t.value) for t in top]
    rows = []
    for name, cfg, tune_ic in entries:
        preds = dict(fold_preds(ds, VALID_YEARS, cfg))
        m = evaluate.score_folds(preds, ds, 0.1, 5)
        rows.append({"name": name, "tune_ic": tune_ic, "valid_ic": m["ic_mean"], "valid_ic_ir": m["ic_ir"],
                     "valid_ls_sharpe": m["ls_sharpe"], "valid_long_sharpe": m["long_sharpe"],
                     "fold_ic": json.dumps({y: round(v, 4) for y, v in m["fold_ic"].items()}),
                     "model": cfg["model"], "cfg": json.dumps(cfg)})
        gc.collect()
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(S.RESULTS / "search_finalists.csv", index=False)
    print(f"{len(done)} completed trials of {len(study.trials)}")
    print(table.drop(columns=["cfg", "fold_ic"]).round(4).to_string())


if __name__ == "__main__":
    {"run": lambda a: run(float(a)), "finalists": lambda a: finalists(int(a))}[sys.argv[1]](sys.argv[2])
