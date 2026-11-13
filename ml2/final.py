"""Evaluate every model on the dev years (to pick a headline model), then ONCE on the final years.

All models share the features/target found by the dev loop. The headline model is chosen by dev
net Sharpe only; the final years are never used for any choice.
"""
import json

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.metrics import summarize
from ml2 import dataset, evaluate, models, settings as S
from ml2.walkforward import run_folds

MEMBERS = ["ridge", "xgb", "lgbm", "mlp", "gru"]
ENSEMBLE_OF = ["xgb", "lgbm", "mlp", "gru"]
FACTORS = {"factor_momentum_12_1": ("mom_12_1", 1.0), "factor_reversal_5d": ("ret_5", -1.0)}


def member_cfg(best: dict, name: str) -> dict:
    """Config for one ensemble member: a search-adopted override if present, else the loop's best settings."""
    if name in best.get("overrides", {}):
        return {**best, **best["overrides"][name]}
    # reuse tuned params only for the model the loop tuned; other models use their defaults
    params = best["params"] if best["model"] == name else models.DEFAULTS[name]
    return {**best, "model": name, "params": params}


def ensemble(preds: dict[str, dict]) -> dict[int, pd.DataFrame]:
    """Average of each member's cross-sectional rank, per date."""
    years = preds[ENSEMBLE_OF[0]].keys()
    return {y: sum(preds[m][y].rank(axis=1, pct=True) for m in ENSEMBLE_OF) / len(ENSEMBLE_OF) for y in years}


def factor_preds(ds, years: list[int]) -> dict[str, dict[int, pd.DataFrame]]:
    """Simple hand-made signals as baselines for the ML models."""
    out = {}
    for name, (col, sign) in FACTORS.items():
        out[name] = {}
        for y in years:
            rows = np.flatnonzero(ds.dates.year.values[ds.t] == y)
            out[name][y] = evaluate.to_wide(ds, rows, sign * ds.X[rows, ds.col[col]])
    return out


def table(ds, preds: dict[str, dict], top_frac: float, hold: int):
    rows, curves = {}, {}
    for name, p in preds.items():
        m = evaluate.score_folds(p, ds, top_frac, hold)
        rows[name] = {k: v for k, v in m.items() if not isinstance(v, (dict, pd.Series))}
        curves[name] = m["ls_returns"], m["long_returns"]
    first = next(iter(preds.values()))
    days = pd.DatetimeIndex(np.concatenate([p.index[p.notna().any(axis=1)] for p in first.values()]))
    ew = evaluate.equal_weight_universe(days, ds.close)
    rows["equal_weight_universe"] = {"ls_sharpe": np.nan, "long_sharpe": ew["metrics"]["sharpe"],
                                     "long_ann_return": ew["metrics"]["ann_return"]}
    return pd.DataFrame(rows).T, curves, ew["net_returns"]


def main():
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    top_frac = best["top_frac"]
    members = [m for m in MEMBERS if not (m == "ridge" and best["target"] == "class_top")]

    results = {}
    for tag, (a, b) in (("dev", S.DEV_YEARS), ("final", S.FINAL_YEARS)):
        years = list(range(a, b + 1))
        preds = {m: run_folds(ds, years, member_cfg(best, m)) for m in members}
        preds["ensemble"] = ensemble(preds)
        preds.update(factor_preds(ds, years))
        tbl, curves, ew = table(ds, preds, top_frac, best["hold"])
        tbl.to_csv(S.RESULTS / f"{tag}_metrics.csv")
        results[tag] = (tbl, curves, ew)
        print(f"\n== {tag} {a}-{b}\n{tbl.round(3).to_string()}", flush=True)

    # headline model: best dev net long-short Sharpe among the ML models (decided without the final years)
    dev_tbl = results["dev"][0]
    headline = dev_tbl.loc[members + ["ensemble"], "ls_sharpe"].astype(float).idxmax()
    print("\nheadline model (by dev Sharpe):", headline)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    curves, ew = results["final"][1], results["final"][2]
    for name in members + ["ensemble"] + list(FACTORS):
        (1 + curves[name][0]).cumprod().plot(ax=axes[0], label=name)
        (1 + curves[name][1]).cumprod().plot(ax=axes[1], label=name)
    (1 + ew).cumprod().plot(ax=axes[1], label="equal_weight_universe", color="black", ls="--")
    axes[0].set_title(f"Final years: long-short net equity (top/bottom {top_frac:.0%})")
    axes[1].set_title("Final years: long-only net equity")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(S.RESULTS / "final_equity.png", dpi=130)
    (S.RESULTS / "headline.json").write_text(json.dumps({"headline": headline, "config": best}, indent=2))


if __name__ == "__main__":
    main()
