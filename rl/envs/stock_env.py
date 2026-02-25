"""Stock market environment builder for portfolio management."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from ..utils.data_loader import load_parquet_data


def build_environment_from_df(
    df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str = "ret_next",
    id_col: str = "ticker",
    date_col: str = "date",
    window: int = 48,
    min_obs_in_window: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, List[np.ndarray]]:
    """Build environment from stock data for fixed-universe portfolio training.
    
    Args:
        df: DataFrame with stock data
        factor_cols: List of factor column names
        ret_col: Name of return column
        id_col: Name of asset ID column
        date_col: Name of date column
        window: Total window size for features (default: 48)
        min_obs_in_window: Minimum observations required in window
        
    Returns:
        Tuple of (X_train, R_train, ids_t, dates, M_feat_list) where:
        - X_train: List of feature arrays (N_t, W, K)
        - R_train: List of return arrays (N_t,)
        - ids_t: List of asset IDs for each time step
        - dates: List of dates
        - M_feat_list: List of per-factor mask arrays (N_t, W, K)
    """
    df = df.copy()
    # Normalize common column names if needed (case-insensitive)
    lower_map = {col.lower(): col for col in df.columns}
    for expected in [date_col, id_col, ret_col]:
        if expected not in df.columns and expected.lower() in lower_map:
            df = df.rename(columns={lower_map[expected.lower()]: expected})

    missing = [c for c in [date_col, id_col, ret_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values([id_col, date_col], inplace=True)

    all_dates = df[date_col].drop_duplicates().sort_values().tolist()
    all_ids = df[id_col].drop_duplicates().tolist()

    factor_panels = []
    for col in factor_cols:
        factor = df.pivot_table(index=date_col, columns=id_col, values=col, aggfunc="last")
        factor = factor.reindex(index=all_dates, columns=all_ids)
        factor_panels.append(factor.to_numpy(dtype=np.float32))
    
    FACTORS = np.stack(factor_panels, axis=-1)

    RET = df.pivot_table(index=date_col, columns=id_col, values=ret_col, aggfunc="last")
    RET = RET.reindex(index=all_dates, columns=all_ids)
    RET = RET.to_numpy(dtype=np.float32)

    X_train, R_train, ids_t, dates = [], [], [], []
    M_feat_list = []
    T_full, N_full = FACTORS.shape[0], FACTORS.shape[1]

    for t in range(window, T_full):
        win = FACTORS[t - window : t, :, :]        
        r_t = RET[t, :]                         

        tradable = ~np.isnan(r_t)             

        if tradable.sum() == 0:
            continue

        # Compute per-factor validity mask
        valid_feat = ~np.isnan(win)  # shape (W, N_full, K)
        valid_ts = valid_feat.any(axis=2)  # a time-row is valid if *any* factor present
        real_counts = valid_ts.sum(axis=0)     
        enough_hist = real_counts >= min_obs_in_window

        keep = tradable & enough_hist         
        if keep.sum() < 1:
            continue

        win_filled = win.copy()                 
        nan_mask = np.isnan(win_filled)
        win_filled[nan_mask] = 0

        X_t = win_filled[:, keep, :].transpose(1, 0, 2)   # (N_t, W, K)
        R_t = r_t[keep]                                   
        M_feat_t = valid_feat[:, keep, :].transpose(1, 0, 2)  # (N_t, W, K)

        X_train.append(X_t)
        R_train.append(R_t)
        ids_t.append(np.array(all_ids)[keep])
        dates.append(all_dates[t])
        M_feat_list.append(M_feat_t)
    
    return X_train, R_train, ids_t, dates, M_feat_list


def build_environment(
    data_dir: str,
    split: str = "processed_weekly_panel.parquet",
    ret_col: str = "ret_next",
    id_col: str = "ticker",
    date_col: str = "date",
    window: int = 48,
    min_obs_in_window: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, List[np.ndarray]]:
    """Build environment from parquet data files for fixed-universe training.
    
    Args:
        data_dir: Path to data directory containing train/test/val splits
        split: Data split to load ('train', 'test', 'val')
        ret_col: Name of return column
        id_col: Name of asset ID column
        date_col: Name of date column
        window: Total window size for features (default: 48)
        length: Length of sequence to extract (default: 24)
        min_obs_in_window: Minimum observations required in window
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        Tuple of (X_train, R_train, ids_t, dates, M_feat_list) where:
        - X_train: List of feature arrays (N_t, W, K)
        - R_train: List of return arrays (N_t,)
        - ids_t: List of asset IDs for each time step
        - dates: List of dates
        - M_feat_list: List of per-factor mask arrays (N_t, W, K)
    """
    # Load data from parquet files
    df = load_parquet_data(
        data_dir=data_dir,
        split=split,
        start_date=start_date,
        end_date=end_date
    )
    
    # Automatically detect factor columns (all columns except ret_col, id_col, and date_col)
    factor_cols = [col for col in df.columns if col not in [ret_col, id_col, date_col]]
    
    # Use the dataframe-based function
    return build_environment_from_df(
        df=df,
        factor_cols=factor_cols,
        ret_col=ret_col,
        id_col=id_col,
        date_col=date_col,
        window=window,
        min_obs_in_window=min_obs_in_window
    )
