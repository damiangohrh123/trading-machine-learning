"""Dashboard: streamlit run app/streamlit_app.py  (research only, not financial advice)."""
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="ML walk-forward backtest", layout="wide")
st.title("ML walk-forward backtest")
st.warning("Research only, not financial advice. Backtests and simulations; no real trading.")

if not (R / "final_test_metrics.csv").exists():
    st.error("No results yet. Run: python run_baselines.py, python loop.py, python final_test.py")
    st.stop()

st.header("Held-out 2022-2025 (net of costs)")
metrics = pd.read_csv(R / "final_test_metrics.csv", index_col=0)
st.dataframe(metrics.style.format("{:.3f}"), width="stretch")

rets = pd.read_csv(R / "final_test_returns.csv", index_col=0, parse_dates=True)
choices = st.multiselect("Strategies", list(rets.columns), default=list(rets.columns))
if choices:
    st.line_chart((1 + rets[choices].fillna(0)).cumprod())

c1, c2 = st.columns(2)
with c1:
    st.subheader("Development experiments (2016-2021)")
    if (R / "experiments.csv").exists():
        exp = pd.read_csv(R / "experiments.csv")
        st.dataframe(exp[["name", "model", "threshold", "max_names", "sharpe", "ann_return",
                          "max_drawdown", "turnover", "note"]], width="stretch")
with c2:
    st.subheader("Logistic regression coefficients")
    if (R / "feature_coefficients.csv").exists():
        st.bar_chart(pd.read_csv(R / "feature_coefficients.csv", index_col=0)["coef"])
