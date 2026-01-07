"""Training functions for RL policies."""

import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.optim as optim

from .evaluator import evaluate_episode
from ..models.rewards import RewardFunction


def train_policy(
    policy: torch.nn.Module,
    X_train: List[np.ndarray],
    R_train: List[np.ndarray],
    epochs: int = 20,
    length: int = 48,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    M_feat_list: Optional[List[np.ndarray]] = None,
    risk_free_rate: float = 0.0
) -> List[Dict[str, Any]]:
    """Train a policy using excess Sharpe ratio maximization.
    
    Args:
        policy: The policy to train
        X_train: Training feature arrays
        R_train: Training return arrays
        epochs: Number of training epochs
        length: Length of training sequences
        lr: Learning rate
        device: Device to train on
        seed: Random seed
        X_val: Optional validation features
        R_val: Optional validation returns
        M_feat_list: Optional training per-factor masks
        M_feat_val: Optional validation per-factor masks
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        List of training logs
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize reward function
    reward_fn = RewardFunction(risk_free_rate=risk_free_rate)
    
    policy.to(device).train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5
    )

    T = len(X_train)
    logs = []
    windows = []
    
    for start in range(0, T - length // 2 + 1, length):
        windows.append((start, start + length // 2))

    for ep in range(epochs):
        for (start, end) in windows:
            policy.train()
            candidates = list(range(start, min(start + length // 2, T)))
            t0 = random.choice(candidates)
            t1 = min(t0 + length // 2, T)

            port, ew = [], []
            for t in range(t0, t1):
                if t >= len(X_train):
                    break
                X_t = torch.tensor(X_train[t], dtype=torch.float32, device=device)
                R_t = torch.tensor(R_train[t], dtype=torch.float32, device=device)
                
                if M_feat_list is not None and t < len(M_feat_list):
                    M_feat_t = torch.tensor(M_feat_list[t], dtype=torch.bool, device=device)
                else:
                    M_feat_t = torch.ones_like(X_t, dtype=torch.bool)

                w_t = policy(X_t, M_feat_t)
                
                # Debug: Check weight distribution
                if ep == 0 and len(port) < 3:  # Only print first few times
                    print(f"  Debug - Weights: {w_t.detach().cpu().numpy()}")
                    print(f"  Debug - Weight sum: {w_t.sum().item():.6f}")
                    print(f"  Debug - Max weight: {w_t.max().item():.6f}")
                    print(f"  Debug - Min weight: {w_t.min().item():.6f}")
                
                port.append((w_t * R_t).sum())
                ew.append(R_t.mean())

            if len(port) == 0:
                continue
                
            port = torch.stack(port)
            ew   = torch.stack(ew)

            # Use the specified reward function
            reward = reward_fn.compute(port, ew)
            loss = -reward

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            logs.append({
                "epoch": ep + 1, 
                "window": (start, end),
                "loss": float(loss.detach().cpu()),
                "sharpe": float(reward.detach().cpu())
            })

        # Update learning rate based on performance
        train_sharpes = [x["sharpe"] for x in logs if x["epoch"] == ep + 1]
        mean_train_sharpe = float(np.mean(train_sharpes)) if train_sharpes else float("nan")
        scheduler.step(mean_train_sharpe)
        
        # Print progress every few epochs
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1}/{epochs}, Mean Sharpe: {mean_train_sharpe:.4f}, LR: {opt.param_groups[0]['lr']:.6f}")

    return logs
