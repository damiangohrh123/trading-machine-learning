import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize(returns: pd.Series, turnover: pd.Series) -> dict:
    """Annualised performance statistics from daily net returns and daily turnover."""
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "ann_return": equity.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1,
        "ann_vol": returns.std() * np.sqrt(TRADING_DAYS),
        "sharpe": returns.mean() / returns.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": drawdown.min(),
        "turnover": turnover.mean() * TRADING_DAYS,  # traded value per year, in portfolio multiples
        "hit_rate": (returns > 0).mean(),
        "n_days": len(returns),
    }
