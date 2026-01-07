"""Reward functions for RL training."""

import torch
from typing import Optional, Dict, Any


class RewardFunction:
    """Reward function for RL training using excess Sharpe ratio."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize reward function.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def compute(self, portfolio_returns: torch.Tensor, equal_weight_returns: torch.Tensor, 
                info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute excess Sharpe ratio reward.
        
        Args:
            portfolio_returns: Portfolio returns (B, T) or (T,)
            equal_weight_returns: Equal weight returns (B, T) or (T,)
            info: Optional additional information (unused)
            
        Returns:
            Excess Sharpe ratio reward
        """
        excess = portfolio_returns - equal_weight_returns
        mu = excess.mean()
        sd = excess.std(unbiased=False) + 1e-8
        sharpe = mu / sd
        
        
        return sharpe