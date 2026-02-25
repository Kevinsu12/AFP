"""Core SAC functions shared between pre-training and backtest."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from rl.configs.sac_config import Config
from rl.envs import build_environment_from_df
from rl.utils.data_loader import load_parquet_data


def cross_sectional_normalize(
    df: pd.DataFrame,
    factor_cols: List[str],
    date_col: str = "date",
    clip_range: Tuple[float, float] = (-3.0, 3.0),
) -> pd.DataFrame:
    """Apply cross-sectional z-score normalization per time step.
    
    For each factor, at each date:
        z = (x - mean) / std
        z = clip(z, clip_range)
    
    Args:
        df: DataFrame with factor data
        factor_cols: List of factor column names to normalize
        date_col: Name of date column
        clip_range: (min, max) for clipping z-scores
        
    Returns:
        DataFrame with normalized factor values
    """
    df = df.copy()
    
    for col in factor_cols:
        # Group by date, compute z-score within each date
        grouped = df.groupby(date_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        
        # Avoid division by zero (if all values same, std=0)
        std = std.replace(0, 1.0)
        
        # Z-score normalization
        df[col] = (df[col] - mean) / std
        
        # Clip to range
        df[col] = df[col].clip(lower=clip_range[0], upper=clip_range[1])
    
    return df


def uniform_prev_w(n_assets: int, device: str) -> torch.Tensor:
    """Create uniform portfolio weights."""
    return torch.full((n_assets,), 1.0 / n_assets, device=device)



def tensor_or_ones(
    X_t: torch.Tensor, t: int, M_feat_list: Optional[List[np.ndarray]]
) -> torch.Tensor:
    """Get mask tensor or create ones mask."""
    if M_feat_list is not None and t < len(M_feat_list):
        return torch.tensor(M_feat_list[t], dtype=torch.bool, device=X_t.device)
    return torch.ones_like(X_t, dtype=torch.bool)


def get_alpha(step: int, cfg: Config) -> float:
    """Linear decay from init_alpha to min_alpha."""
    sac = cfg.sac
    if step >= sac.alpha_decay_steps:
        return sac.min_alpha
    decay_ratio = step / sac.alpha_decay_steps
    return sac.init_alpha - (sac.init_alpha - sac.min_alpha) * decay_ratio


def sac_update(
    actor: torch.nn.Module,
    critic1: torch.nn.Module,
    critic2: torch.nn.Module,
    target1: torch.nn.Module,
    target2: torch.nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic1_opt: torch.optim.Optimizer,
    critic2_opt: torch.optim.Optimizer,
    batch: List[dict],
    alpha: float,
    cfg: Config,
    device: str,
    update_actor: bool = True,
) -> Tuple[Optional[float], float]:
    """SAC update step.
    
    Args:
        actor: Actor network
        critic1, critic2: Twin Q-critics
        target1, target2: Target critics
        actor_opt, critic1_opt, critic2_opt: Optimizers
        batch: List of transition dicts from replay buffer
        alpha: Entropy coefficient
        cfg: Configuration object
        device: Device string (cuda/cpu)
        update_actor: Whether to update actor this step (for policy delay)
        
    Returns:
        (actor_loss, critic_loss) - actor_loss is None if not updated
    """
    gamma = cfg.sac.gamma
    tau = cfg.sac.tau

    X_b = torch.tensor(np.stack([item["X"] for item in batch]), dtype=torch.float32, device=device)
    M_b = torch.ones_like(X_b, dtype=torch.bool) if batch[0]["M"] is None else \
          torch.tensor(np.stack([item["M"] for item in batch]), dtype=torch.bool, device=device)
    prev_w_b = torch.tensor(np.stack([item["prev_w"] for item in batch]), dtype=torch.float32, device=device)
    action_b = torch.tensor(np.stack([item["action"] for item in batch]), dtype=torch.float32, device=device)
    X_next_b = torch.tensor(np.stack([item["X_next"] for item in batch]), dtype=torch.float32, device=device)
    M_next_b = torch.ones_like(X_next_b, dtype=torch.bool) if batch[0]["M_next"] is None else \
               torch.tensor(np.stack([item["M_next"] for item in batch]), dtype=torch.bool, device=device)
    done_b = torch.tensor([1.0 if item["done"] else 0.0 for item in batch], dtype=torch.float32, device=device)
    reward_b = torch.tensor([item["reward"] for item in batch], dtype=torch.float32, device=device)

    # Critic update
    q1 = critic1(X_b, M_b, action_b, prev_w=prev_w_b)
    q2 = critic2(X_b, M_b, action_b, prev_w=prev_w_b)

    prev_w_next = action_b.detach()
    with torch.no_grad():
        dist_next = actor.get_dist(X_next_b, M_next_b, prev_w=prev_w_next)
        next_action = dist_next.rsample()
        logp_next = dist_next.log_prob(next_action)
        q1_t = target1(X_next_b, M_next_b, next_action, prev_w=prev_w_next)
        q2_t = target2(X_next_b, M_next_b, next_action, prev_w=prev_w_next)
        q_t = torch.min(q1_t, q2_t) - alpha * logp_next
        y = reward_b + (1.0 - done_b) * gamma * q_t

    critic_loss = ((q1 - y).pow(2) + (q2 - y).pow(2)).mean()

    critic1_opt.zero_grad()
    critic2_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic1_opt.step()
    critic2_opt.step()
    
    # Target update (every critic update, not just actor update)
    with torch.no_grad():
        for p_t, p in zip(target1.parameters(), critic1.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
        for p_t, p in zip(target2.parameters(), critic2.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    # Actor update (delayed)
    actor_loss_val: Optional[float] = None
    if update_actor:
        dist = actor.get_dist(X_b, M_b, prev_w=prev_w_b)
        action_pi = dist.rsample()
        logp_pi = dist.log_prob(action_pi)
        q1_pi = critic1(X_b, M_b, action_pi, prev_w=prev_w_b)
        q2_pi = critic2(X_b, M_b, action_pi, prev_w=prev_w_b)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp_pi - q_pi).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()
        actor_loss_val = float(actor_loss.detach().cpu())

    return actor_loss_val, float(critic_loss.detach().cpu())


def load_portfolio_data(
    cfg: Config,
    portfolio_tickers: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List, List[np.ndarray]]:
    """Load and filter data for a specific portfolio.
    
    Args:
        cfg: Configuration object
        portfolio_tickers: List of ticker symbols to include
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (X, R, ids, dates, M_feat) from build_environment_from_df
    """
    df = load_parquet_data(
        data_dir=cfg.env.data_dir,
        split=cfg.env.split,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Filter to portfolio tickers
    df = df[df[cfg.env.id_col].isin(portfolio_tickers)]
    
    # Columns to exclude from factors (not normalized, not used as features)
    exclude = {cfg.env.ret_col, cfg.env.id_col, cfg.env.date_col}
    exclude.update(cfg.env.exclude_cols)
    
    # Get factor columns (everything except excluded columns)
    factor_cols = [col for col in df.columns if col not in exclude]
    
    # Cross-sectional z-score normalization per time step (only factor columns)
    df = cross_sectional_normalize(
        df=df,
        factor_cols=factor_cols,
        date_col=cfg.env.date_col,
        clip_range=(-3.0, 3.0),
    )
    
    return build_environment_from_df(
        df=df,
        factor_cols=factor_cols,
        ret_col=cfg.env.ret_col,
        id_col=cfg.env.id_col,
        date_col=cfg.env.date_col,
        window=cfg.env.window,
        min_obs_in_window=cfg.env.min_obs_in_window,
    )


def compute_reward(
    w_t: torch.Tensor,
    R_t: torch.Tensor,
    prev_w: torch.Tensor,
    turnover_coef: float,
    reward_scale: float = 1.0,
    variance_coef: float = 0.0,
    recent_pnls: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute PnL - turnover - variance penalty.
    
    Returns:
        (pnl, turnover, reward_scalar) — pnl & turnover are un-scaled
    """
    pnl = (w_t * R_t).sum()
    turnover = torch.abs(w_t - prev_w).sum()
    var_penalty = 0.0
    if variance_coef > 0.0 and recent_pnls and len(recent_pnls) >= 2:
        var_penalty = float(np.var(recent_pnls))
    reward = reward_scale * (pnl - turnover_coef * turnover - variance_coef * var_penalty).item()
    return pnl, turnover, reward
