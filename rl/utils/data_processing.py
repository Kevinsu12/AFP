"""Data processing utilities."""

import pandas as pd


def winsorize_cs(x: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsorize a series at the p and (1-p) quantiles.
    
    Args:
        x: Input series
        p: Quantile threshold (default 0.01 for 1%/99%)
        
    Returns:
        Winsorized series
    """
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lo, hi)


def zscore_cs(x: pd.Series) -> pd.Series:
    """Z-score normalize a series.
    
    Args:
        x: Input series
        
    Returns:
        Z-score normalized series
    """
    return (x - x.mean()) / (x.std(ddof=0) + 1e-8)
