"""Pre-training script: Train SAC on randomly sampled portfolios + final portfolio."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from rl.algo.replay_buffer import ReplayBuffer
from rl.algo.sac_core import (
    compute_reward,
    get_alpha,
    load_portfolio_data,
    sac_update,
    tensor_or_ones,
    uniform_prev_w,
)
from rl.configs.sac_config import Config
from rl.models import TransformerActor, TransformerQCritic
from rl.utils.data_loader import load_parquet_data


def sample_random_portfolios(
    all_tickers: List[str],
    num_portfolios: int,
    portfolio_size: int,
    seed: int,
) -> Dict[str, List[str]]:
    """Randomly sample portfolios from available tickers.
    
    Args:
        all_tickers: List of all available tickers
        num_portfolios: Number of portfolios to sample
        portfolio_size: Number of stocks per portfolio
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping portfolio ID to list of tickers
    """
    rng = random.Random(seed)
    
    if len(all_tickers) < portfolio_size:
        raise ValueError(
            f"Not enough tickers ({len(all_tickers)}) to sample portfolios of size {portfolio_size}"
        )
    
    portfolios = {}
    for i in range(num_portfolios):
        # Sample without replacement within each portfolio
        tickers = rng.sample(all_tickers, portfolio_size)
        portfolios[f"random_{i}"] = tickers
    
    return portfolios


def run_pretrain(cfg: Config | None = None):
    """Run pre-training on randomly sampled portfolios + optional final portfolio."""
    if cfg is None:
        cfg = Config.default()
    
    device = cfg.backtest.device
    seed = cfg.pretrain.seed
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print("=" * 60)
    print("PRE-TRAINING: Random Portfolio Sampling")
    print("=" * 60)
    print(f"Date range: {cfg.pretrain.start_date} to {cfg.pretrain.end_date}")
    print(f"Device: {device}")
    print(f"Reward scale: {cfg.sac.reward_scale}, Init alpha: {cfg.sac.init_alpha}")
    
    # Load parquet to get all available tickers
    print("\nLoading data to discover available tickers...")
    df = load_parquet_data(
        data_dir=cfg.env.data_dir,
        split=cfg.env.split,
        start_date=cfg.pretrain.start_date,
        end_date=cfg.pretrain.end_date,
    )
    all_tickers = sorted(df[cfg.env.id_col].unique().tolist())
    print(f"Found {len(all_tickers)} unique tickers in data")
    
    # Sample random portfolios
    print(f"\nSampling {cfg.pretrain.num_random_portfolios} random portfolios "
          f"(size={cfg.pretrain.portfolio_size})...")
    portfolios = sample_random_portfolios(
        all_tickers=all_tickers,
        num_portfolios=cfg.pretrain.num_random_portfolios,
        portfolio_size=cfg.pretrain.portfolio_size,
        seed=seed,
    )
    
    # Add "final" portfolio if configured
    if cfg.pretrain.include_final:
        portfolios_path = Path(cfg.pretrain.portfolios_file)
        if portfolios_path.exists():
            with open(portfolios_path) as f:
                final_portfolios = json.load(f)
            if "final" in final_portfolios:
                portfolios["final"] = final_portfolios["final"]
                print(f"Added 'final' portfolio: {portfolios['final']}")
            else:
                print("Warning: 'final' key not found in portfolios.json")
        else:
            print(f"Warning: {portfolios_path} not found, skipping final portfolio")
    
    print(f"Total portfolios for pre-training: {len(portfolios)}")
    
    # Load data for ALL portfolios upfront
    portfolio_data: Dict[str, dict] = {}
    all_dates_set = set()
    K = None
    
    print("\nLoading portfolio data...")
    for pid, tickers in portfolios.items():
        try:
            X, R, ids, dates, M_feat = load_portfolio_data(
                cfg, tickers, cfg.pretrain.start_date, cfg.pretrain.end_date
            )
        except Exception as e:
            print(f"  Skipping portfolio {pid}: {e}")
            continue
        
        if not dates or len(X) < 2:
            print(f"  Skipping portfolio {pid}: insufficient data")
            continue
        
        # Check portfolio has expected number of assets
        if X[0].shape[0] != cfg.pretrain.portfolio_size:
            print(f"  Skipping portfolio {pid}: got {X[0].shape[0]} assets, "
                  f"expected {cfg.pretrain.portfolio_size}")
            continue
        
        # Get K from first portfolio
        if K is None:
            _, _, K = X[0].shape
        
        # Store data indexed by date for this portfolio
        date_to_idx = {d: i for i, d in enumerate(dates)}
        portfolio_data[pid] = {
            "X": X,
            "R": R,
            "ids": ids,
            "dates": dates,
            "M_feat": M_feat,
            "date_to_idx": date_to_idx,
            "prev_w": None,
            "prev_R": None,
            "recent_pnls": [],
        }
        all_dates_set.update(dates)
    
    if not portfolio_data:
        raise ValueError("No valid portfolios loaded")
    
    print(f"\nSuccessfully loaded {len(portfolio_data)} portfolios")
    print(f"Number of factors: {K}")
    print(f"Total unique dates: {len(all_dates_set)}")
    
    # Sort all dates chronologically
    all_dates = sorted(all_dates_set)
    
    # Initialize models
    actor = TransformerActor.from_config(K, cfg.model).to(device)
    critic1 = TransformerQCritic.from_config(K, cfg.model).to(device)
    critic2 = TransformerQCritic.from_config(K, cfg.model).to(device)
    
    target1 = TransformerQCritic.from_config(K, cfg.model).to(device)
    target2 = TransformerQCritic.from_config(K, cfg.model).to(device)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    target1.eval()
    target2.eval()
    
    # Optimizers
    actor_opt = torch.optim.Adam(
        actor.parameters(), lr=cfg.sac.actor_lr, weight_decay=cfg.sac.weight_decay
    )
    critic1_opt = torch.optim.Adam(
        critic1.parameters(), lr=cfg.sac.critic_lr, weight_decay=cfg.sac.weight_decay
    )
    critic2_opt = torch.optim.Adam(
        critic2.parameters(), lr=cfg.sac.critic_lr, weight_decay=cfg.sac.weight_decay
    )
    
    # Shared replay buffer
    buffer = ReplayBuffer(cfg.sac.replay_size)
    
    all_logs = []
    global_step = 0
    critic_updates = 0
    
    # Process all portfolios at each time step
    for epoch in range(cfg.pretrain.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{cfg.pretrain.epochs}")
        print(f"{'='*60}")
        
        for pid in portfolio_data:
            portfolio_data[pid]["prev_w"] = None
            portfolio_data[pid]["prev_R"] = None
            portfolio_data[pid]["recent_pnls"] = []
        
        for t_idx, current_date in enumerate(all_dates[:-1]):  # -1 because we need next state
            next_date = all_dates[t_idx + 1]
            
            # Process each portfolio that has data for this date
            for pid, pdata in portfolio_data.items():
                date_to_idx = pdata["date_to_idx"]
                
                # Skip if this portfolio doesn't have data for this date
                if current_date not in date_to_idx:
                    pdata["prev_w"] = None  # Reset if gap in data
                    pdata["prev_R"] = None
                    continue
                
                t = date_to_idx[current_date]
                
                # Need next state - check if next date exists for this portfolio
                if next_date not in date_to_idx:
                    continue
                
                t_next = date_to_idx[next_date]
                
                X = pdata["X"]
                R = pdata["R"]
                M_feat = pdata["M_feat"]
                
                X_t = torch.tensor(X[t], dtype=torch.float32, device=device)
                R_t = torch.tensor(R[t], dtype=torch.float32, device=device)
                M_t = tensor_or_ones(X_t, t, M_feat)
                
                # Check asset count for BOTH current and next state
                n_expected = cfg.env.expected_assets
                if X_t.shape[0] != n_expected:
                    pdata["prev_w"] = None
                    pdata["prev_R"] = None
                    continue
                if X[t_next].shape[0] != n_expected:
                    # Next state has wrong asset count → skip this transition
                    pdata["prev_w"] = None
                    pdata["prev_R"] = None
                    continue
                
                prev_w = pdata["prev_w"]
                prev_R = pdata["prev_R"]
                if prev_w is None or prev_w.shape[0] != X_t.shape[0]:
                    prev_w = uniform_prev_w(X_t.shape[0], device)
                    prev_R = None
                
                # Compute drifted weights: actual holdings after last period's returns
                if prev_R is not None:
                    drifted_w = prev_w * (1.0 + prev_R)
                    drifted_w = drifted_w / drifted_w.sum().clamp(min=1e-8)
                else:
                    drifted_w = prev_w
                
                # Sample action (stochastic for exploration)
                dist = actor.get_dist(X_t, M_t, prev_w=drifted_w)
                w_t = dist.rsample()
                
                recent_pnls = pdata["recent_pnls"][-cfg.sac.variance_lookback:]
                pnl_t, _, reward = compute_reward(
                    w_t, R_t, drifted_w, cfg.sac.turnover_coef, cfg.sac.reward_scale,
                    variance_coef=cfg.sac.variance_coef, recent_pnls=recent_pnls,
                )
                pdata["recent_pnls"].append(float(pnl_t.detach().cpu()))
                
                # Check if this is the last valid step for this portfolio
                done = (t_next == len(X) - 1)
                
                # Store transition (prev_w = drifted = actual holdings when decision was made)
                buffer.add({
                    "X": X[t],
                    "M": M_feat[t] if M_feat is not None else None,
                    "prev_w": drifted_w.detach().cpu().numpy(),
                    "action": w_t.detach().cpu().numpy(),
                    "reward": reward,
                    "X_next": X[t_next],
                    "M_next": M_feat[t_next] if M_feat is not None else None,
                    "done": done,
                })
                
                # Update: target weights + returns for drift computation next step
                pdata["prev_w"] = w_t.detach()
                pdata["prev_R"] = R_t.detach()
                global_step += 1
            
            # Training update (after processing all portfolios for this time step)
            if len(buffer) < cfg.sac.batch_size or global_step < cfg.sac.start_steps:
                continue
            if global_step % cfg.sac.update_every != 0:
                continue
            
            for _ in range(cfg.sac.updates_per_step):
                alpha = get_alpha(critic_updates, cfg)
                critic_updates += 1
                update_actor = (critic_updates % cfg.sac.policy_delay == 0)
                
                batch = buffer.sample(cfg.sac.batch_size)
                actor_loss, critic_loss = sac_update(
                    actor, critic1, critic2, target1, target2,
                    actor_opt, critic1_opt, critic2_opt,
                    batch, alpha, cfg, device, update_actor,
                )
                
                if critic_updates % 100 == 0:
                    temp = float(actor.temperature.detach().cpu())
                    all_logs.append({
                        "date": str(current_date),
                        "global_step": global_step,
                        "critic_updates": critic_updates,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "alpha": alpha,
                        "temperature": temp,
                        "buffer_size": len(buffer),
                    })
                    print(f"  {current_date}: Steps={global_step}, Updates={critic_updates}, "
                          f"CriticLoss={critic_loss:.4f}, Alpha={alpha:.4f}, Temp={temp:.4f}")
    
    # Save pre-trained weights
    ckpt_dir = Path(cfg.pretrain.checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(actor.state_dict(), ckpt_dir / "actor.pt")
    torch.save(critic1.state_dict(), ckpt_dir / "critic1.pt")
    torch.save(critic2.state_dict(), ckpt_dir / "critic2.pt")
    torch.save(target1.state_dict(), ckpt_dir / "target1.pt")
    torch.save(target2.state_dict(), ckpt_dir / "target2.pt")
    
    # Save the sampled portfolios for reference
    portfolios_used_path = ckpt_dir / "portfolios_used.json"
    portfolios_used_path.write_text(json.dumps(portfolios, indent=2))
    
    # Save logs
    logs_path = ckpt_dir / "pretrain_logs.json"
    if all_logs:
        logs_path.write_text(pd.DataFrame(all_logs).to_json(orient="records"))
    
    print(f"\n{'='*60}")
    print("PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total portfolios trained: {len(portfolio_data)}")
    print(f"Total steps: {global_step}")
    print(f"Total critic updates: {critic_updates}")
    print(f"Final alpha: {get_alpha(critic_updates, cfg):.4f}")
    print(f"Saved checkpoints to: {ckpt_dir}")
    
    return all_logs


def main():
    run_pretrain()


if __name__ == "__main__":
    main()
