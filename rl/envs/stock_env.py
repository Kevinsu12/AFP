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
    id_col: str = "permno",
    date_col: str = "date",
    window: int = 48,
    min_obs_in_window: int = 1,
    stochastic: bool = False,
    rebalance_prob: float = 0.3,
    max_change: int = 2,
    min_portfolio_size: int = 3,
    max_portfolio_size: int = 15
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, List[np.ndarray]]:
    """Build environment from stock data with stochastic portfolio rebalancing.
    
    Args:
        df: DataFrame with stock data
        factor_cols: List of factor column names
        ret_col: Name of return column
        id_col: Name of asset ID column
        date_col: Name of date column
        window: Total window size for features (default: 48)
        length: Length of sequence to extract (default: 24)
        min_obs_in_window: Minimum observations required in window
        stochastic: Enable stochastic portfolio selection
        rebalance_prob: Probability of rebalancing each time step (0.0-1.0)
        max_change: Maximum number of stocks to add/remove per rebalancing
        min_portfolio_size: Minimum portfolio size
        max_portfolio_size: Maximum portfolio size
        
    Returns:
        Tuple of (X_train, R_train, ids_t, dates, M_feat_list) where:
        - X_train: List of feature arrays (N_t, W, K)
        - R_train: List of return arrays (N_t,)
        - ids_t: List of asset IDs for each time step
        - dates: List of dates
        - M_feat_list: List of per-factor mask arrays (N_t, W, K)
    """
    df = df.copy()
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
    M_list = []  
    M_feat_list = []
    T_full, N_full = FACTORS.shape[0], FACTORS.shape[1]

    # Initialize portfolio for stochastic mode
    current_portfolio = None
    
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

        # Stochastic portfolio selection logic
        if stochastic:
            # First time step: initialize random portfolio
            if current_portfolio is None:
                available_stocks = np.where(keep)[0]
                if len(available_stocks) < min_portfolio_size:
                    continue
                
                # Random initial portfolio size (3-11 stocks)
                initial_size = min(
                    np.random.randint(min_portfolio_size, min(max_portfolio_size, len(available_stocks)) + 1),
                    len(available_stocks)
                )
                current_portfolio = np.random.choice(available_stocks, size=initial_size, replace=False)
            
            # Check if we should rebalance (30% chance)
            if np.random.random() < rebalance_prob:
                available_stocks = np.where(keep)[0]
                current_available = np.intersect1d(current_portfolio, available_stocks)
                
                if len(current_available) >= min_portfolio_size:
                    # Determine how many stocks to change (1-2 stocks)
                    num_changes = min(max_change, len(current_available))
                    
                    # Randomly decide: add, remove, or both
                    action = np.random.choice(['add', 'remove', 'both'])
                    
                    if action == 'add' and len(current_available) < max_portfolio_size:
                        # Add stocks
                        available_to_add = np.setdiff1d(available_stocks, current_available)
                        if len(available_to_add) > 0:
                            num_to_add = min(num_changes, len(available_to_add), 
                                           max_portfolio_size - len(current_available))
                            new_stocks = np.random.choice(available_to_add, size=num_to_add, replace=False)
                            current_portfolio = np.concatenate([current_available, new_stocks])
                    
                    elif action == 'remove' and len(current_available) > min_portfolio_size:
                        # Remove stocks
                        num_to_remove = min(num_changes, len(current_available) - min_portfolio_size)
                        stocks_to_remove = np.random.choice(current_available, size=num_to_remove, replace=False)
                        current_portfolio = np.setdiff1d(current_available, stocks_to_remove)
                    
                    elif action == 'both' and len(current_available) > min_portfolio_size:
                        # Both add and remove
                        num_to_remove = min(num_changes // 2, len(current_available) - min_portfolio_size)
                        num_to_add = min(num_changes - num_to_remove, 
                                       max_portfolio_size - len(current_available))
                        
                        if num_to_remove > 0:
                            stocks_to_remove = np.random.choice(current_available, size=num_to_remove, replace=False)
                            current_available = np.setdiff1d(current_available, stocks_to_remove)
                        
                        if num_to_add > 0:
                            available_to_add = np.setdiff1d(available_stocks, current_available)
                            if len(available_to_add) > 0:
                                new_stocks = np.random.choice(available_to_add, size=num_to_add, replace=False)
                                current_available = np.concatenate([current_available, new_stocks])
                        
                        current_portfolio = current_available
                    
            
            # Use current portfolio for this time step
            keep = np.zeros_like(keep)
            keep[current_portfolio] = True
            
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
    stochastic: bool = False,
    rebalance_prob: float = 0.3,
    max_change: int = 2,
    min_portfolio_size: int = 3,
    max_portfolio_size: int = 15
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, List[np.ndarray]]:
    """Build environment from parquet data files with stochastic portfolio selection.
    
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
        stochastic: Enable stochastic portfolio selection
        rebalance_prob: Probability of rebalancing each time step (0.0-1.0)
        max_change: Maximum number of stocks to add/remove per rebalancing
        min_portfolio_size: Minimum portfolio size
        max_portfolio_size: Maximum portfolio size
    
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
        min_obs_in_window=min_obs_in_window,
        stochastic=stochastic,
        rebalance_prob=rebalance_prob,
        max_change=max_change,
        min_portfolio_size=min_portfolio_size,
        max_portfolio_size=max_portfolio_size
    )
