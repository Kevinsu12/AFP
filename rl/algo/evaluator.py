"""Evaluation functions for RL policies."""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

from ..models.rewards import RewardFunction


def evaluate_episode(
    policy: torch.nn.Module,
    X_val: List[np.ndarray],
    R_val: List[np.ndarray],
    t0: int = 0,
    length: int = 12,
    device: str = "cpu",
    M_feat_list: Optional[List[np.ndarray]] = None,
    risk_free_rate: float = 0.0,
    ids_list: Optional[List[np.ndarray]] = None,
    dates_list: Optional[List] = None,
    detailed: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Optional[List[Dict]]]:
    """Evaluate a policy on a sequence of time steps.
    
    Args:
        policy: The policy to evaluate
        X_train: List of feature arrays
        R_train: List of return arrays
        t0: Starting time index
        length: Number of time steps to evaluate
        device: Device to run on
        M_feat_list: Optional per-factor mask arrays
        risk_free_rate: Risk-free rate for Sharpe calculation
        ids_list: Optional list of asset IDs for each time step
        dates_list: Optional list of dates for each time step
        detailed: Whether to return detailed portfolio data
        
    Returns:
        Tuple of (port_returns, ew_returns, wealth_port, wealth_ew, sharpe_excess, detailed_results)
        If detailed=False, detailed_results will be None
    """
    policy.eval()
    port, ew, wealth_port, wealth_ew = [], [], [], []
    detailed_results = [] if detailed else None

    cum_p, cum_ew = 1.0, 1.0
    
    # Initialize reward function once
    reward_fn = RewardFunction(risk_free_rate=risk_free_rate)  

    for t in range(t0, min(t0 + length, len(X_val))):
        X_t = torch.tensor(X_val[t], dtype=torch.float32, device=device)   
        R_t = torch.tensor(R_val[t], dtype=torch.float32, device=device)  
        
        if M_feat_list is not None and t < len(M_feat_list):
            M_feat_t = torch.tensor(M_feat_list[t], dtype=torch.bool, device=device)
        else:
            M_feat_t = torch.ones_like(X_t, dtype=torch.bool)

        with torch.no_grad():
            # Debug: Check input data
            if t < 3:  # Only print first few times
                print(f"  Eval Debug - X_t shape: {X_t.shape}")
                print(f"  Eval Debug - M_feat_t shape: {M_feat_t.shape}")
                print(f"  Eval Debug - X_t sample: {X_t[0, 0, :5].cpu().numpy()}")
                print(f"  Eval Debug - M_feat_t sample: {M_feat_t[0, 0, :5].cpu().numpy()}")
            
            w_t = policy(X_t, M_feat_t)  # (N_t,)
            
            # Debug: Check weight distribution during evaluation
            if t < 3:  # Only print first few times
                print(f"  Eval Debug - Weights: {w_t.cpu().numpy()}")
                print(f"  Eval Debug - Weight sum: {w_t.sum().item():.6f}")
                print(f"  Eval Debug - Max weight: {w_t.max().item():.6f}")
                print(f"  Eval Debug - Min weight: {w_t.min().item():.6f}")
            
            r_p = (w_t * R_t).sum().item()     # portfolio simple return
            r_ew = R_t.mean().item()           # equal-weight return

        cum_p *= (1.0 + r_p)
        cum_ew *= (1.0 + r_ew)
        
        port.append(r_p)
        ew.append(r_ew)
        wealth_port.append(cum_p)
        wealth_ew.append(cum_ew)
        
        # Save detailed portfolio information if requested
        if detailed and detailed_results is not None:
            portfolio_details = {
                'time_step': t - t0,  # Relative to start of evaluation
                'date': dates_list[t] if dates_list and t < len(dates_list) else f"step_{t}",
                'asset_ids': ids_list[t].tolist() if ids_list and t < len(ids_list) else [],
                'portfolio_weights': w_t.cpu().numpy().tolist(),
                'returns': R_t.cpu().numpy().tolist(),
                'portfolio_return': r_p,
                'equal_weight_return': r_ew,
                'cumulative_wealth_port': cum_p,
                'cumulative_wealth_ew': cum_ew
            }
            detailed_results.append(portfolio_details)

    port = np.array(port, dtype=float)
    ew   = np.array(ew, dtype=float)
    wealth_port = np.array(wealth_port, dtype=float)
    wealth_ew   = np.array(wealth_ew, dtype=float)

    # Use the pre-initialized reward function
    sharpe_excess = reward_fn.compute(torch.tensor(port), torch.tensor(ew))

    return port, ew, wealth_port, wealth_ew, float(sharpe_excess), detailed_results
