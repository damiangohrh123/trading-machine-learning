"""v2 results page. Run: streamlit run app/streamlit_app.py, then open "2 v2 results" in the sidebar."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

R = Path(__file__).resolve().parents[2] / "results" / "v2"
FORWARD = R.parent / "forward"

st.title("v2: cross-sectional ML (GPU)")
st.warning("Research only, not financial advice. Backtests and simulations; no real trading.")

final, tail, sig, cost, search, forward = st.tabs(
    ["Final years", "Pristine tail", "Significance", "Costs", "Search", "Forward test"])


def table(name: str, caption: str) -> pd.DataFrame:
    st.caption(caption)
    df = pd.read_csv(R / name, index_col=0)
    st.dataframe(df.style.format("{:.3f}"), width="stretch")
    return df


with final:
    table("final_metrics.csv", "2022 to Nov 2026, net of costs, base decile book. The ensemble is the headline model.")
    st.image(str(R / "final_equity.png"))
    table("dev_metrics.csv", "Dev 2013-2021. Includes years the search tuned on, so it overstates every model.")

with tail:
    table("pristine_metrics.csv", "2026-05-23 onward: days no earlier decision saw. About 118 trading days, so "
          "every interval is wide.")

with sig:
    df = pd.read_csv(R / "significance.csv")
    model = st.radio("Model", sorted(df["model"].unique()), horizontal=True)
    st.caption("95% block-bootstrap intervals (20-day blocks, 5000 resamples).")
    st.dataframe(df[df["model"] == model].drop(columns="model").style.format(
        {c: "{:.3f}" for c in ("estimate", "ci_low", "ci_high", "share_positive")}), width="stretch")

with cost:
    df = pd.read_csv(R / "cost_sensitivity.csv")
    st.caption("Net Sharpe as one-way trading cost rises. The results elsewhere use 7 bps.")
    model = st.radio("Model ", sorted(df["model"].unique()), horizontal=True)
    st.dataframe(df[df["model"] == model].drop(columns="model").style.format("{:.2f}", subset=df.columns[4:]),
                 width="stretch")

with search:
    df = pd.read_csv(R / "search.csv").drop_duplicates("number", keep="last").sort_values("number")
    df["model"] = df["cfg"].map(lambda s: json.loads(s)["model"])
    st.write(f"{len(df)} trials: " + ", ".join(f"{k.lower()} {v}" for k, v in df["state"].value_counts().items()))
    # pruned trials hold only a partial score, so charts and summaries use completed trials
    df = df[df["state"] == "COMPLETE"].copy()
    df["best_so_far"] = df["ic_tune"].cummax()
    st.line_chart(df.set_index("number")[["ic_tune", "best_so_far"]])
    by_model = df.groupby("model")["ic_tune"].agg(["count", "mean", "max"]).sort_values("mean", ascending=False)
    st.caption("Tune-year rank IC (2013-2018) by model family. Finalists are scored on 2019-2021 before adoption.")
    st.dataframe(by_model.style.format({"mean": "{:.4f}", "max": "{:.4f}"}), width="stretch")
    finalists = R / "search_finalists.csv"
    if finalists.exists():
        st.caption("Latest finalist check on the validation years 2019-2021.")
        st.dataframe(pd.read_csv(finalists).drop(columns=["cfg", "fold_ic"]).style.format(
            {c: "{:.4f}" for c in ("tune_ic", "valid_ic", "valid_ic_ir", "valid_ls_sharpe", "valid_long_sharpe")}),
            width="stretch")

with forward:
    log = FORWARD / "forward_log.csv"
    files = sorted(FORWARD.glob("predictions_*.csv"))
    days = sorted({f.stem.split("_")[1] for f in files})
    st.write(f"{len(files)} live prediction file(s) over {len(days)} day(s), latest {days[-1] if days else 'none'}.")
    live = pd.read_csv(log) if log.exists() else pd.DataFrame()
    live = live[live["file"].str.startswith("live/")] if len(live) else live
    if live.empty:
        st.info("No outcomes scored yet. A prediction is scored once its 5 trading days have passed.")
    else:
        st.caption("Live days only (replays excluded). Intervals are 95% block bootstraps with 5-day blocks, since "
                   "5-day outcomes of neighbouring days overlap. They are wide until a few hundred days exist.")
        rng, rows = np.random.default_rng(0), []
        for name, g in live.sort_values("date").groupby("model"):
            row = {"model": name, "days": len(g)}
            for col in ("rank_ic", "long_minus_short"):
                x = g[col].dropna().to_numpy()
                n_blocks = -(-len(x) // 5)
                starts = rng.integers(0, max(len(x) - 4, 1), (2000, n_blocks))
                means = np.array([np.concatenate([x[s:s + 5] for s in st_])[:len(x)].mean() for st_ in starts])
                row[col] = x.mean()
                row[col + "_low"], row[col + "_high"] = np.percentile(means, [2.5, 97.5])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.4f}", subset=pd.IndexSlice[:, "rank_ic":]),
                     width="stretch")
        cum = live.pivot_table(index="date", columns="model", values="long_minus_short").sort_index().fillna(0).cumsum()
        st.caption("Cumulative top-minus-bottom 5-day return, summed over days (overlapping, so not a tradable curve).")
        st.line_chart(cum)
        st.dataframe(live, width="stretch")
