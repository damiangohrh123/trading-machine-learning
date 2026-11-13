"""Phase 3: greedy improvement loop on the DEVELOPMENT period.

Each pass applies ONE listed change to the current best config, validates it, and keeps it only if
net Sharpe improves by more than MIN_GAIN. Stops after MAX_PASSES passes or PATIENCE passes in a
row without improvement. The held-out period is never read here.
"""
import json

import config
from experiment import run_experiment
from features.build import FEATURE_GROUPS

MAX_PASSES = 10
PATIENCE = 3
MIN_GAIN = 0.05

START = dict(model="logreg", groups=tuple(FEATURE_GROUPS), model_params={"C": 0.1}, threshold=0.5, max_names=None)


def without(group):
    return lambda c: {**c, "groups": tuple(g for g in c["groups"] if g != group)}


def with_params(model=None, **params):
    return lambda c: {**c, "model": model or c["model"], "model_params": params}


def with_setting(**kw):
    return lambda c: {**c, **kw}


# (description, change) in the order they are tried
CANDIDATES = [
    ("threshold 0.52", with_setting(threshold=0.52)),
    ("top-10 names by prob", with_setting(max_names=10)),
    ("drop volume features", without("volume")),
    ("logreg C=0.01 (stronger reg)", with_params(C=0.01)),
    ("drop range features", without("range")),
    ("lgbm, heavy regularisation", with_params("lgbm", n_estimators=100, num_leaves=7,
                                               min_child_samples=500, learning_rate=0.03)),
    ("threshold 0.55", with_setting(threshold=0.55)),
    ("logreg C=1.0", with_params("logreg", C=1.0)),
    ("drop market features", without("market")),
    ("top-5 names by prob", with_setting(max_names=5)),
]


def main():
    best_cfg = START
    best = run_experiment("loop_start", note="loop baseline", **best_cfg)
    print(f"start sharpe={best['sharpe']}")
    history, misses = [], 0
    for i, (desc, change) in enumerate(CANDIDATES[:MAX_PASSES], 1):
        cfg = change(best_cfg)
        row = run_experiment(f"loop_{i:02d}", note=desc, **cfg)
        gain = row["sharpe"] - best["sharpe"]
        kept = gain > MIN_GAIN
        history.append({"pass": i, "change": desc, "sharpe": row["sharpe"], "gain": round(gain, 4), "kept": kept})
        print(f"pass {i:2d} | {desc:32s} sharpe={row['sharpe']:.3f} gain={gain:+.3f} {'KEEP' if kept else 'drop'}")
        if kept:
            best_cfg, best, misses = cfg, row, 0
        else:
            misses += 1
            if misses >= PATIENCE:
                print(f"stopping: {PATIENCE} passes in a row without improvement")
                break
    final = {"cfg": {**best_cfg, "groups": list(best_cfg["groups"])}, "dev_sharpe": best["sharpe"]}
    (config.RESULTS_DIR / "best_config.json").write_text(json.dumps(final, indent=2))
    (config.RESULTS_DIR / "loop_history.json").write_text(json.dumps(history, indent=2))
    print("best:", final)


if __name__ == "__main__":
    main()
