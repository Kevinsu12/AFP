"""Tests for environment components."""

import pytest
import pandas as pd
import numpy as np

from rl.envs import build_environment_from_df


def test_build_environment():
    """Test environment building with synthetic data."""
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    assets = ['A', 'B', 'C']
    
    data = []
    for date in dates:
        for asset in assets:
            data.append({
                'date': date,
                'permno': asset,
                'ret_next': np.random.normal(0, 0.02),
                'log_mktcap': np.random.normal(10, 1),
                'ret_lag1': np.random.normal(0, 0.02),
                'mom_12_2': np.random.normal(1, 0.1),
                'vol_12': np.random.normal(0.02, 0.01),
                'turnover': np.random.normal(0.1, 0.05),
                'log_dollar_vol': np.random.normal(5, 1),
                'amihud': np.random.normal(0.001, 0.0005),
                'log_adjprc': np.random.normal(4, 0.5),
            })
    
    df = pd.DataFrame(data)
    factor_cols = ['log_mktcap', 'ret_lag1', 'mom_12_2', 'vol_12', 'turnover', 'log_dollar_vol', 'amihud', 'log_adjprc']
    
    X_train, R_train, ids_t, dates, M_list, M_feat_list = build_environment_from_df(
        df, factor_cols, window=12, min_obs_in_window=1
    )
    
    # Check that we get some data
    assert len(X_train) > 0
    assert len(R_train) > 0
    assert len(ids_t) > 0
    assert len(dates) > 0
    assert len(M_list) > 0
    
    # Check shapes
    for i in range(len(X_train)):
        N, W, K = X_train[i].shape
        assert W == 12  # window size
        assert K == len(factor_cols)  # number of factors
        assert R_train[i].shape == (N,)  # returns match number of assets
        assert M_list[i].shape == (N, W)  # mask shape matches
        
        # Check that weights sum to 1 (if we had a policy)
        assert not np.isnan(X_train[i]).any()
        assert not np.isnan(R_train[i]).any()
