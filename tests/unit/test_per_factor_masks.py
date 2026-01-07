"""Tests for per-factor mask functionality."""

import pytest
import numpy as np
import pandas as pd
import torch

from rl.envs import build_environment_from_df
from rl.models import TransformerPolicy


def test_per_factor_masks():
    """Test per-factor mask behavior with synthetic data."""
    # Create synthetic data: 2 assets × 4 months × 3 factors
    dates = pd.date_range('2020-01-01', periods=4, freq='M')
    assets = ['A', 'B']
    factors = ['factor1', 'factor2', 'factor3']
    
    data = []
    for date in dates:
        for asset in assets:
            data.append({
                'date': date,
                'permno': asset,
                'ret_next': np.random.normal(0, 0.02),
                'factor1': np.random.normal(0, 1),
                'factor2': np.random.normal(0, 1),
                'factor3': np.random.normal(0, 1),
            })
    
    df = pd.DataFrame(data)
    
    # Inject NaNs in one factor for one asset at one time
    df.loc[(df['permno'] == 'A') & (df['date'] == dates[1]), 'factor2'] = np.nan
    
    # Build environment with per-factor masks
    X_list, R_list, ids_t, dates_out, M_list, M_feat_list = build_environment_from_df(
        df, factors, window=2, min_obs_in_window=1
    )
    
    # Check shapes
    assert len(X_list) > 0
    X_t = X_list[0]
    M_t = M_list[0]
    M_feat_t = M_feat_list[0]
    
    assert X_t.shape == (2, 2, 3)  # (N, W, K)
    assert M_t.shape == (2, 2)     # (N, W)
    assert M_feat_t.shape == (2, 2, 3)  # (N, W, K)
    
    # Check that X_t has zeros only where missing, not whole rows
    assert not np.isnan(X_t).any()  # No NaNs after zero-filling
    
    # Check that M_feat_t[n,w,k] = 0 only for missing factor
    # We injected NaN in factor2 for asset A at time 1
    # This should be reflected in the mask
    assert M_feat_t[0, 1, 1] == False  # factor2 for asset A at time 1 should be False
    assert M_feat_t[0, 1, 0] == True   # factor1 should be True
    assert M_feat_t[0, 1, 2] == True   # factor3 should be True
    
    # Check that M_t[n,w] = 1 if any factor present, 0 if all missing
    # Since we only made one factor missing, the row should still be valid
    assert M_t[0, 1] == True  # Asset A at time 1 should be valid (has factors 1 and 3)
    
    # Test policy with per-factor masks
    policy = TransformerPolicy(K=3, use_per_factor_mask=True)
    
    # Test forward pass with per-factor masks
    X_tensor = torch.tensor(X_t, dtype=torch.float32)
    M_tensor = torch.tensor(M_t, dtype=torch.bool)
    M_feat_tensor = torch.tensor(M_feat_t, dtype=torch.bool)
    
    weights = policy(X_tensor, M_tensor, M_feat_tensor)
    
    assert weights.shape == (2,)  # One weight per asset
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)  # Weights sum to 1
    assert torch.all(weights >= 0)  # All weights non-negative


def test_backward_compatibility():
    """Test that old call sites still work without per-factor masks."""
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=4, freq='M')
    assets = ['A', 'B']
    factors = ['factor1', 'factor2', 'factor3']
    
    data = []
    for date in dates:
        for asset in assets:
            data.append({
                'date': date,
                'permno': asset,
                'ret_next': np.random.normal(0, 0.02),
                'factor1': np.random.normal(0, 1),
                'factor2': np.random.normal(0, 1),
                'factor3': np.random.normal(0, 1),
            })
    
    df = pd.DataFrame(data)
    
    # Build environment (now always returns per-factor masks)
    X_list, R_list, ids_t, dates_out, M_list, M_feat_list = build_environment_from_df(
        df, factors, window=2, min_obs_in_window=1
    )
    
    # Test policy without per-factor masks
    policy = TransformerPolicy(K=3, use_per_factor_mask=False)
    
    X_tensor = torch.tensor(X_list[0], dtype=torch.float32)
    M_tensor = torch.tensor(M_list[0], dtype=torch.bool)
    M_feat_tensor = torch.tensor(M_feat_list[0], dtype=torch.bool)
    
    weights = policy(X_tensor, M_tensor, M_feat=M_feat_tensor)  # M_feat provided but not used
    
    assert weights.shape == (2,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(weights >= 0)
