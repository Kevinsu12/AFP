"""Tests for model components."""

import pytest
import torch
import numpy as np

from rl.models import TransformerPolicy


def test_transformer_policy_forward():
    """Test TransformerPolicy forward pass shapes."""
    K = 8
    N = 10  # number of assets
    W = 12  # window size
    
    policy = TransformerPolicy(K=K, d_model=64, nhead=4, num_layers=2, hidden=64)
    
    # Test with valid data
    X_t = torch.randn(N, W, K)
    M_t = torch.ones(N, W, dtype=torch.bool)
    
    weights = policy(X_t, M_t)
    
    assert weights.shape == (N,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(weights >= 0)
    
    # Test with some missing data
    M_t[0, :5] = False  # First asset missing first 5 observations
    weights = policy(X_t, M_t)
    
    assert weights.shape == (N,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(weights >= 0)


def test_transformer_policy_different_sizes():
    """Test TransformerPolicy with different input sizes."""
    K = 5
    policy = TransformerPolicy(K=K, d_model=32, nhead=2, num_layers=1, hidden=32)
    
    # Test with different numbers of assets
    for N in [1, 5, 20, 100]:
        W = 10
        X_t = torch.randn(N, W, K)
        M_t = torch.ones(N, W, dtype=torch.bool)
        
        weights = policy(X_t, M_t)
        
        assert weights.shape == (N,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(weights >= 0)
