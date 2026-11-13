"""Central configuration. Research only, not financial advice."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Liquid, long-history US ETFs and large caps (survivorship caveat: large caps are
# chosen with hindsight; ETFs are used to limit this bias).
TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV",
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "AAPL", "MSFT",
]
MARKET_EXTRAS = ["^VIX"]  # context features only, never traded

START = "2011-01-01"        # a little earlier than DEV_START for indicator warm-up
DEV_START = "2012-01-01"
DEV_END = "2021-12-31"      # development period: all tuning happens here
TEST_START = "2022-01-01"   # final held-out period: touched only in Phase 4
END = "2025-12-31"

HORIZON = 5                 # label horizon in trading days
EMBARGO = HORIZON           # gap between train end and test start (avoid label leakage)
COST_BPS = 5.0              # one-way transaction cost
SLIPPAGE_BPS = 2.0          # one-way slippage
SEED = 42
