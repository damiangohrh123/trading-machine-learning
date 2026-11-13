"""Settings for the large cross-sectional ML pipeline (v2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "v2"

START = "2004-01-01"       # a year of warm-up before the first training rows
DEV_YEARS = (2013, 2021)   # walk-forward test years used for ALL tuning
FINAL_YEARS = (2022, 2026) # evaluated once at the end; 2026 never appeared in v1
PRISTINE_FROM = "2026-05-23"  # first day after the data every earlier analysis used (a bug had hidden later days)
HORIZON = 5               # forward-return horizon in trading days
EMBARGO = 5                # gap between training end and test start
COST_BPS = 7.0             # one-way cost plus slippage
BORROW_BPS = 30.0          # annual borrow cost on short exposure
SEED = 42

ETFS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV",
        "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU"]
MARKET = ["^VIX"]  # context only, never traded
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
