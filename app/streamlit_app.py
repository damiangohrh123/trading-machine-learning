"""Dashboard: streamlit run app/streamlit_app.py (research only, not financial advice)."""
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="ML walk-forward backtest", layout="wide")
st.title("ML walk-forward backtest")
st.warning("Research only, not financial advice. Backtests and simulations; no real trading.")

# held-out results: table plus selectable equity curves
st.header("Held-out 2022-2025 (net of costs)")
st.dataframe(pd.read_csv(R / "final_test_metrics.csv", index_col=0).style.format("{:.3f}"), width="stretch")
returns = pd.read_csv(R / "final_test_returns.csv", index_col=0, parse_dates=True)
chosen = st.multiselect("Strategies", list(returns.columns), default=list(returns.columns))
if chosen:
    st.line_chart((1 + returns[chosen]).cumprod())

left, right = st.columns(2)
with left:
    st.subheader("Development experiments (2016-2021)")
    experiments = pd.read_csv(R / "experiments.csv")
    st.dataframe(experiments[["name", "model", "threshold", "max_names", "sharpe", "ann_return",
                              "max_drawdown", "turnover", "note"]], width="stretch")
with right:
    st.subheader("Logistic regression coefficients")
    st.bar_chart(pd.read_csv(R / "feature_coefficients.csv", index_col=0)["coef"])
