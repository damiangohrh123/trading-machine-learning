"""Exploration round: portfolio constructions and a beta-adjusted target.

Every choice is made on the DEV years. The final years are then scored as a second look; they were
already seen once in ml2.final, so that run is a sanity check, not a clean test.
"""
import json
import pickle

import pandas as pd

from ml2 import data, dataset, evaluate, portfolio, settings as S
from ml2.final import ENSEMBLE_OF, ensemble, factor_preds, member_cfg
from ml2.walkforward import run_folds

KEYS = ("ic_mean", "ls_sharpe", "ls_ann_return", "ls_turnover", "long_sharpe", "long_ann_return")


def ensemble_preds(ds, best: dict, years: list[int], tag: str) -> dict[int, pd.DataFrame]:
    """Rank-average ensemble predictions per year, cached on disk because training takes minutes."""
    path = S.RESULTS / f"preds_{tag}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    preds = ensemble({m: run_folds(ds, years, member_cfg(best, m)) for m in ENSEMBLE_OF})
    preds = {y: p.loc[p.notna().any(axis=1)] for y, p in preds.items()}
    path.write_bytes(pickle.dumps(preds))
    return preds


def variants(sector: pd.Series) -> dict:
    """name -> (transform(preds, momentum) -> preds, weight_fn). Smoothed scores use a 1-day book."""
    same = lambda p, m: p  # noqa: E731
    ema = lambda hl: (lambda p, m: {y: portfolio.ema(f, hl) for y, f in p.items()})  # noqa: E731
    mix = lambda w: (lambda p, m: {y: portfolio.blend(f, m[y], w) for y, f in p.items()})  # noqa: E731
    ema5_mix = lambda p, m: {y: portfolio.blend(portfolio.ema(f, 5), m[y], 0.25) for y, f in p.items()}  # noqa: E731
    return {
        "base: decile, 5-day book": (same, portfolio.decile(0.1, 5)),
        "ema halflife 2": (ema(2), portfolio.decile(0.1, 1)),
        "ema halflife 5": (ema(5), portfolio.decile(0.1, 1)),
        "ema halflife 10": (ema(10), portfolio.decile(0.1, 1)),
        "buffer enter 10% / exit 25%": (same, portfolio.buffered(0.1, 0.25)),
        "buffer enter 10% / exit 40%": (same, portfolio.buffered(0.1, 0.4)),
        "sector-neutral": (same, portfolio.sector_neutral(sector, 0.1, 5)),
        "blend 25% momentum": (mix(0.25), portfolio.decile(0.1, 5)),
        "blend 50% momentum": (mix(0.5), portfolio.decile(0.1, 5)),
        "ema 5 + buffer 10/25": (ema(5), portfolio.buffered(0.1, 0.25)),
        "ema 5 + 25% momentum": (ema5_mix, portfolio.decile(0.1, 1)),
        "ema 5 + 25% momentum + buffer 10/25": (ema5_mix, portfolio.buffered(0.1, 0.25)),
    }


def score_all(ds, preds, mom, sector) -> tuple[pd.DataFrame, dict]:
    rows, folds = {}, {}
    for name, (transform, weight_fn) in variants(sector).items():
        m = evaluate.score_folds(transform(preds, mom), ds, 0.1, 5, weight_fn)
        rows[name] = {k: m[k] for k in KEYS}
        folds[name] = m["fold_sharpe"]
    return pd.DataFrame(rows).T, folds


def main():
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    sector = data.sectors()
    dev_years = list(range(S.DEV_YEARS[0], S.DEV_YEARS[1] + 1))
    final_years = list(range(S.FINAL_YEARS[0], S.FINAL_YEARS[1] + 1))

    # 1) beta-adjusted target for the tuned XGBoost, compared with the loop's best on the dev years
    resid = evaluate.score_folds(run_folds(ds, dev_years, {**best, "target": "resid"}), ds, 0.1, 5)
    print("beta-adjusted target (xgb): ic=%.4f ls_sharpe=%.2f  [rank target xgb: ic=0.0324 ls_sharpe=0.69]"
          % (resid["ic_mean"], resid["ls_sharpe"]), flush=True)

    # 2) portfolio variants on dev; momentum factor scores are the blend partner
    dev = ensemble_preds(ds, best, dev_years, "dev")
    mom_dev = factor_preds(ds, dev_years)["factor_momentum_12_1"]
    table, folds = score_all(ds, dev, mom_dev, sector)
    base = "base: decile, 5-day book"
    table["fold_wins_vs_base"] = [sum(folds[n][y] > folds[base][y] for y in dev_years) for n in table.index]
    table.to_csv(S.RESULTS / "explore_dev.csv")
    print("\n== dev (variants on the ensemble)\n" + table.round(3).to_string(), flush=True)

    # 3) pick by dev long-short Sharpe among variants that win at least 5 of 9 folds, then second look
    eligible = table[(table["fold_wins_vs_base"] >= 5) | (table.index == base)]
    chosen = eligible["ls_sharpe"].astype(float).idxmax()
    print("\nchosen on dev:", chosen, flush=True)
    final = ensemble_preds(ds, best, final_years, "final")
    mom_final = factor_preds(ds, final_years)["factor_momentum_12_1"]
    table_f, _ = score_all(ds, final, mom_final, sector)
    table_f.to_csv(S.RESULTS / "explore_final.csv")
    print("\n== final second look (all variants shown; selection used dev only)\n" + table_f.round(3).to_string())
    (S.RESULTS / "explore_chosen.json").write_text(json.dumps({"chosen_on_dev": chosen}))


if __name__ == "__main__":
    main()
