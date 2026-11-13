"""Phase 3: greedy improvement loop on the DEVELOPMENT period.

Each pass applies ONE pre-listed change to the current best config, runs walk-forward
validation (2016-2021), logs it, and keeps it only if net Sharpe improves by > MIN_GAIN.
Stops after MAX_PASSES passes or PATIENCE consecutive non-improvements.
The held-out period is never read here.
"""
from __future__ import annotations

import copy
import json

import config
from experiment import run_experiment
from features.build import FEATURE_GROUPS

MAX_PASSES = 10
PATIENCE = 3
MIN_GAIN = 0.05

ALL = tuple(FEATURE_GROUPS)

START = dict(model="logreg", groups=ALL, model_params={"C": 0.1}, threshold=0.5, max_names=None)


def without(*drop):
    return lambda c: {**c, "groups": tuple(g for g in c["groups"] if g not in drop)}


def setp(**kw):
    return lambda c: {**c, "model_params": {**c["model_params"], **kw}}


def setk(**kw):
    return lambda c: {**c, **kw}


CANDIDATES = [
    ("threshold 0.52", setk(threshold=0.52)),
    ("top-10 names by prob", setk(max_names=10)),
    ("drop volume features", without("volume")),
    ("logreg C=0.01 (stronger reg)", setp(C=0.01)),
    ("drop range features", without("range")),
    ("lgbm, heavy regularisation", lambda c: {**c, "model": "lgbm", "model_params": dict(
        n_estimators=100, num_leaves=7, min_child_samples=500, learning_rate=0.03)}),
    ("threshold 0.55", setk(threshold=0.55)),
    ("logreg C=1.0", lambda c: {**c, "model": "logreg", "model_params": {"C": 1.0}}),
    ("drop market features", without("market")),
    ("top-5 names by prob", setk(max_names=5)),
]


def main():
    best_cfg = copy.deepcopy(START)
    best = run_experiment("loop_start", note="loop baseline", **best_cfg)["row"]
    print(f"start sharpe={best['sharpe']}  cfg={best_cfg['model']} {best_cfg['model_params']}")
    fails = 0
    history = []
    for i, (desc, fn) in enumerate(CANDIDATES[:MAX_PASSES], 1):
        cfg = fn(copy.deepcopy(best_cfg))
        row = run_experiment(f"loop_{i:02d}", note=desc, **cfg)["row"]
        gain = row["sharpe"] - best["sharpe"]
        kept = gain > MIN_GAIN
        history.append({"pass": i, "change": desc, "sharpe": row["sharpe"], "gain": round(gain, 4),
                        "kept": kept, "turnover": row["turnover"], "ann_return": row["ann_return"],
                        "max_drawdown": row["max_drawdown"]})
        print(f"pass {i:2d} | {desc:32s} sharpe={row['sharpe']:.3f} gain={gain:+.3f} {'KEEP' if kept else 'drop'}")
        if kept:
            best_cfg, best, fails = cfg, row, 0
        else:
            fails += 1
            if fails >= PATIENCE:
                print(f"stopping: {PATIENCE} passes in a row without improvement")
                break
    final = {"cfg": {**best_cfg, "groups": list(best_cfg["groups"])}, "dev_metrics": {
        k: best[k] for k in ("accuracy", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "hit_rate")}}
    (config.RESULTS_DIR / "best_config.json").write_text(json.dumps(final, indent=2))
    (config.RESULTS_DIR / "loop_history.json").write_text(json.dumps(history, indent=2))
    print("best:", json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
