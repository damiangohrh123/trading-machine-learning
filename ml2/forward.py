"""Forward test: log predictions before the outcome is known, then score them once prices arrive.

    python -m ml2.data && python -m ml2.dataset      # refresh prices and features first
    python -m ml2.forward predict [model] [as_of]    # model defaults to gru; 'all' logs every member + ensemble
    python -m ml2.forward score                      # score every logged prediction whose outcome is known

A live prediction (no as_of) trains on everything labelled so far and ranks the latest day. Giving as_of
replays a past day into results/forward/replay, which tests the code but is not a forward record.
"""
import json
import sys

import numpy as np
import pandas as pd

from ml2 import dataset, models, settings as S
from ml2.features import EARN_GROUPS
from ml2.final import ENSEMBLE_OF, member_cfg

OUT = S.RESULTS.parent / "forward"
LOG = OUT / "forward_log.csv"
FILES = "predictions_*.csv"
EARN_MODELS = ("xgb", "mlp", "gru")  # the models round four tested with the earnings groups; LightGBM stays as is


def member_scores(ds, best: dict, model: str, t_pred: int, extra: tuple = ()) -> np.ndarray:
    """Train one model on every day labelled before t_pred and score the stocks on day t_pred.

    extra adds feature groups to the adopted config, which is how the earnings variants are logged."""
    cfg = member_cfg(best, model)
    X = ds.matrix(ds.select(cfg["groups"] + list(extra), cfg["cs_mode"]))
    y = ds.target(cfg["target"], cfg["horizon"])
    # train only on rows whose label window closed before the prediction day, with the usual embargo
    train = np.flatnonzero((ds.t <= t_pred - max(S.EMBARGO, cfg["horizon"]) - 1) & ~np.isnan(y))
    test = np.flatnonzero(ds.t == t_pred)
    params = dict(cfg["params"])
    assert not params.pop("recency_halflife", 0), "the forward test does not support recency weighting"
    ctx = {"ds": ds, "tr": train, "te": test, "target": cfg["target"], "weight": None, "deadline": None}
    return models.fit_predict(cfg["model"], params, X[train], y[train], X[test], ctx)


def predict(model: str = "gru", as_of: str | None = None) -> None:
    """Log one model's ranking, or every member plus the ensemble with model 'all'."""
    ds = dataset.load()
    best = json.loads((S.RESULTS / "best_config.json").read_text())
    t_pred = int(ds.t.max()) if as_of is None else ds.dates.get_loc(pd.Timestamp(as_of))
    tickers = ds.tickers[ds.s[np.flatnonzero(ds.t == t_pred)]]
    names = ["ridge"] + ENSEMBLE_OF if model == "all" else [model]
    scores = {m: member_scores(ds, best, m, t_pred) for m in names}
    if model == "all":  # same rule as ml2.final.ensemble: average of the members' cross-sectional ranks
        scores["ensemble"] = sum(pd.Series(scores[m]).rank(pct=True).values for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
        # earnings variants (round four) are logged next to the adopted models, not instead of them
        for m in EARN_MODELS:
            scores[m + "_earn"] = member_scores(ds, best, m, t_pred, EARN_GROUPS)
        scores["ensemble_earn"] = sum(
            pd.Series(scores[m + "_earn" if m in EARN_MODELS else m]).rank(pct=True).values
            for m in ENSEMBLE_OF) / len(ENSEMBLE_OF)
    folder = OUT if as_of is None else OUT / "replay"
    folder.mkdir(parents=True, exist_ok=True)
    for name, score in scores.items():
        pct = pd.Series(score).rank(pct=True).values
        out = pd.DataFrame({"date": ds.dates[t_pred].date(), "ticker": tickers, "score": score, "pct": pct,
                            "side": np.where(pct > 0.9, "long", np.where(pct <= 0.1, "short", "none")),
                            "model": name})
        path = folder / f"predictions_{ds.dates[t_pred].date()}_{name}.csv"
        out.to_csv(path, index=False)
        print(f"saved {path.relative_to(S.ROOT)}: {(out.side == 'long').sum()} long, {(out.side == 'short').sum()} short")


def realized(pred: pd.Series, fwd: pd.Series, frac: float = 0.1) -> dict:
    """Outcome of one day's ranking: rank IC and the mean forward return of the top and bottom groups."""
    both = pd.concat([pred, fwd], axis=1, keys=["p", "f"]).dropna()
    pct = both["p"].rank(pct=True)
    long, short = both["f"][pct > 1 - frac].mean(), both["f"][pct <= frac].mean()
    return {"rank_ic": both["p"].corr(both["f"], method="spearman"), "long_return": long, "short_return": short,
            "long_minus_short": long - short, "universe_return": both["f"].mean(), "n": len(both)}


def score() -> None:
    ds = dataset.load()
    fwd = ds.fwd[S.HORIZON]
    done = set(pd.read_csv(LOG)["file"]) if LOG.exists() else set()
    rows = []
    for kind, folder in (("live", OUT), ("replay", OUT / "replay")):
        for path in sorted(folder.glob(FILES)):
            key = f"{kind}/{path.name}"
            if key in done:
                continue
            pred = pd.read_csv(path)
            day = pd.Timestamp(pred["date"].iloc[0])
            t = ds.dates.get_loc(day)
            if t + S.HORIZON >= len(ds.dates):
                print(f"pending: {key} (outcome needs {S.HORIZON} more trading days)")
                continue
            rows.append({"file": key, "date": day.date(), "model": pred["model"].iloc[0],
                         **realized(pred.set_index("ticker")["score"], fwd.iloc[t])})
    if rows:
        new = pd.DataFrame(rows)
        new.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        print(new.round(4).to_string(index=False))


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"predict": predict, "score": score}[command](*args)
