"""v2 results page. Run: streamlit run app/streamlit_app.py, then open "2 v2 results" in the sidebar."""
import json
from pathlib import Path

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
    table("final_metrics.csv", "2022 to Nov 2026, net of costs, base decile book. The GRU is the headline model.")
    st.image(str(R / "final_equity.png"))
    table("dev_metrics.csv", "Dev 2013-2021. Includes years the search tuned on, so it overstates the GRU.")

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
    st.write(f"{len(files)} live prediction file(s) logged: " + ", ".join(f.stem.removeprefix("predictions_") for f in files))
    if log.exists():
        st.dataframe(pd.read_csv(log), width="stretch")
    else:
        st.info("No outcomes scored yet. Run `python -m ml2.forward score` once 5 trading days have passed.")
