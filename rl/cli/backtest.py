"""Backtest script: Fine-tune pre-trained SAC on target portfolio."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize as scipy_minimize

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


def max_drawdown(pnls: List[float]) -> float:
    """Compute max drawdown from a list of daily PnLs."""
    cum = np.cumprod([1 + p for p in pnls])
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def enforce_weight_bounds(w: torch.Tensor, n_assets: int) -> torch.Tensor:
    """Clamp each weight to [1/(4N), 4/N], then renormalize to sum to 1."""
    min_w = 1.0 / (4 * n_assets)
    max_w = 4.0 / n_assets
    w = w.clamp(min=min_w, max=max_w)
    return w / w.sum()


def run_backtest(cfg: Config | None = None):
    """Run backtest with fine-tuning on target portfolio."""
    if cfg is None:
        cfg = Config.default()
    
    device = cfg.backtest.device
    seed = cfg.backtest.seed
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load portfolios
    portfolios_path = Path(cfg.pretrain.portfolios_file)
    with open(portfolios_path) as f:
        portfolios = json.load(f)
    
    portfolio_id = cfg.backtest.portfolio_id
    if portfolio_id not in portfolios:
        raise ValueError(f"Portfolio '{portfolio_id}' not found in {portfolios_path}")
    
    portfolio_tickers = portfolios[portfolio_id]
    print(f"Backtest portfolio: {portfolio_id}")
    print(f"Tickers: {portfolio_tickers}")
    print(f"Date range: {cfg.backtest.start_date} to {cfg.backtest.end_date}")
    print(f"Device: {device}")
    
    # Load data
    X, R, ids, dates, M_feat = load_portfolio_data(
        cfg, portfolio_tickers, cfg.backtest.start_date, cfg.backtest.end_date
    )
    
    if not dates or len(X) < 2:
        raise ValueError("Insufficient data for backtest")
    
    dates = [pd.to_datetime(d) for d in dates]
    print(f"Loaded {len(X)} time steps")
    
    # Get K from data
    _, _, K = X[0].shape
    
    # Initialize models
    actor = TransformerActor.from_config(K, cfg.model).to(device)
    critic1 = TransformerQCritic.from_config(K, cfg.model).to(device)
    critic2 = TransformerQCritic.from_config(K, cfg.model).to(device)
    target1 = TransformerQCritic.from_config(K, cfg.model).to(device)
    target2 = TransformerQCritic.from_config(K, cfg.model).to(device)
    
    # Load pre-trained weights if available
    if cfg.backtest.load_pretrained:
        pretrained_path = Path(cfg.backtest.pretrained_path)
        if (pretrained_path / "actor.pt").exists():
            print(f"Loading pre-trained weights from {pretrained_path}")
            actor.load_state_dict(torch.load(pretrained_path / "actor.pt", map_location=device))
            critic1.load_state_dict(torch.load(pretrained_path / "critic1.pt", map_location=device))
            critic2.load_state_dict(torch.load(pretrained_path / "critic2.pt", map_location=device))
            target1.load_state_dict(torch.load(pretrained_path / "target1.pt", map_location=device))
            target2.load_state_dict(torch.load(pretrained_path / "target2.pt", map_location=device))
        else:
            print(f"Warning: Pre-trained weights not found at {pretrained_path}, starting fresh")
    
    target1.eval()
    target2.eval()
    
    # Optimizers with LOWER learning rates for fine-tuning
    actor_opt = torch.optim.Adam(
        actor.parameters(), 
        lr=cfg.backtest.finetune_actor_lr,
        weight_decay=cfg.sac.weight_decay
    )
    critic1_opt = torch.optim.Adam(
        critic1.parameters(), 
        lr=cfg.backtest.finetune_critic_lr,
        weight_decay=cfg.sac.weight_decay
    )
    critic2_opt = torch.optim.Adam(
        critic2.parameters(), 
        lr=cfg.backtest.finetune_critic_lr,
        weight_decay=cfg.sac.weight_decay
    )
    
    print(f"Fine-tuning LRs - Actor: {cfg.backtest.finetune_actor_lr}, Critic: {cfg.backtest.finetune_critic_lr}")
    
    # Fresh replay buffer for backtest
    buffer = ReplayBuffer(cfg.sac.replay_size)
    
    total_steps = 0
    critic_updates = 0
    logs: List[dict] = []
    portfolio_steps: List[dict] = []
    recent_pnls: List[float] = []
    
    prev_w: Optional[torch.Tensor] = None      # previous target weights (stochastic)
    prev_w_det: Optional[torch.Tensor] = None  # previous target weights (deterministic)
    prev_R: Optional[torch.Tensor] = None      # previous returns (for drift)
    
    warmup_idx = cfg.backtest.warmup_days
    
    for t in range(len(X) - 1):
        X_t = torch.tensor(X[t], dtype=torch.float32, device=device)
        R_t = torch.tensor(R[t], dtype=torch.float32, device=device)
        M_t = tensor_or_ones(X_t, t, M_feat)
        
        # Check asset count for BOTH current and next state
        n_expected = cfg.env.expected_assets
        if X_t.shape[0] != n_expected:
            print(f"Warning: Expected {n_expected} assets, got {X_t.shape[0]} at {dates[t]}")
            prev_w = None
            prev_w_det = None
            prev_R = None
            continue
        if X[t + 1].shape[0] != n_expected:
            print(f"Warning: Next state has {X[t + 1].shape[0]} assets at {dates[t + 1]}, skipping")
            prev_w = None
            prev_w_det = None
            prev_R = None
            continue
        
        if prev_w is None or prev_w.shape[0] != X_t.shape[0]:
            prev_w = uniform_prev_w(X_t.shape[0], device)
            prev_R = None
        if prev_w_det is None or prev_w_det.shape[0] != X_t.shape[0]:
            prev_w_det = uniform_prev_w(X_t.shape[0], device)
        
        # Compute drifted weights: actual holdings after last period's returns
        if prev_R is not None:
            drifted_w = prev_w * (1.0 + prev_R)
            drifted_w = drifted_w / drifted_w.sum().clamp(min=1e-8)
            drifted_w_det = prev_w_det * (1.0 + prev_R)
            drifted_w_det = drifted_w_det / drifted_w_det.sum().clamp(min=1e-8)
        else:
            drifted_w = prev_w
            drifted_w_det = prev_w_det
        
        # Stochastic action for training (actor sees actual holdings)
        dist = actor.get_dist(X_t, M_t, prev_w=drifted_w)
        w_t = dist.rsample()
        if cfg.backtest.enforce_weight_bounds:
            w_t = enforce_weight_bounds(w_t, n_expected)
        
        # Deterministic action for logging
        with torch.no_grad():
            w_t_det = actor(X_t, M_t, prev_w=drifted_w_det)
            if cfg.backtest.enforce_weight_bounds:
                w_t_det = enforce_weight_bounds(w_t_det, n_expected)
        
        lookback_pnls = recent_pnls[-cfg.sac.variance_lookback:]
        pnl_stoch, _, reward = compute_reward(
            w_t, R_t, drifted_w, cfg.sac.turnover_coef, cfg.sac.reward_scale,
            variance_coef=cfg.sac.variance_coef, recent_pnls=lookback_pnls,
        )
        pnl_det, turnover_det, _ = compute_reward(
            w_t_det, R_t, drifted_w_det, cfg.sac.turnover_coef
        )
        recent_pnls.append(float(pnl_stoch.detach().cpu()))
        
        # Next state
        done = (t + 1) == (len(X) - 1)
        
        # Store transition (prev_w = drifted = actual holdings at decision time)
        buffer.add({
            "X": X[t],
            "M": M_feat[t] if M_feat is not None else None,
            "prev_w": drifted_w.detach().cpu().numpy(),
            "action": w_t.detach().cpu().numpy(),
            "reward": reward,
            "X_next": X[t + 1],
            "M_next": M_feat[t + 1] if M_feat is not None else None,
            "done": done,
        })
        
        # Save target weights + returns for drift computation next step
        prev_w = w_t.detach()
        prev_w_det = w_t_det.detach()
        prev_R = R_t.detach()
        total_steps += 1
        
        # Log portfolio (deterministic weights)
        portfolio_steps.append({
            "t": t,
            "date": dates[t].isoformat(),
            "asset_ids": ids[t].tolist() if ids is not None else portfolio_tickers,
            "weights": w_t_det.detach().cpu().numpy().tolist(),
            "returns": R[t].tolist(),
            "pnl": float(pnl_det.detach().cpu()),
            "turnover": float(turnover_det.detach().cpu()),
        })
        
        # Skip training during warmup
        if t < warmup_idx:
            continue
        if len(buffer) < cfg.sac.batch_size or total_steps < cfg.sac.start_steps:
            continue
        if total_steps % cfg.sac.update_every != 0:
            continue
        
        # Fine-tuning updates
        for _ in range(cfg.sac.updates_per_step):
            # Use backtest-specific alpha (lower exploration for fine-tuning)
            bt = cfg.backtest
            if critic_updates >= bt.finetune_alpha_decay_steps:
                alpha = bt.finetune_min_alpha
            else:
                ratio = critic_updates / bt.finetune_alpha_decay_steps
                alpha = bt.finetune_init_alpha - (bt.finetune_init_alpha - bt.finetune_min_alpha) * ratio
            critic_updates += 1
            update_actor = (critic_updates % cfg.sac.policy_delay == 0)
            
            batch = buffer.sample(cfg.sac.batch_size)
            actor_loss, critic_loss = sac_update(
                actor, critic1, critic2, target1, target2,
                actor_opt, critic1_opt, critic2_opt,
                batch, alpha, cfg, device, update_actor,
            )
            
            logs.append({
                "t": t,
                "date": dates[t].isoformat(),
                "reward": reward,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "alpha": alpha,
                "buffer_size": len(buffer),
            })
    
    # Save checkpoints
    if cfg.backtest.checkpoint_path:
        ckpt_dir = Path(cfg.backtest.checkpoint_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), ckpt_dir / "actor.pt")
        torch.save(critic1.state_dict(), ckpt_dir / "critic1.pt")
        torch.save(critic2.state_dict(), ckpt_dir / "critic2.pt")
        print(f"Saved fine-tuned checkpoints to {ckpt_dir}")
    
    # Compute summary statistics — SAC strategy
    pnls = [p["pnl"] for p in portfolio_steps]
    turnovers = [p["turnover"] for p in portfolio_steps]
    
    cumulative_return = np.prod([1 + p for p in pnls]) - 1
    avg_daily_return = np.mean(pnls)
    volatility = np.std(pnls) * np.sqrt(252)
    sharpe = (avg_daily_return * 252) / volatility if volatility > 0 else 0
    avg_turnover = np.mean(turnovers)
    mdd = max_drawdown(pnls)
    
    # Compute equal-weight benchmark (with realistic rebalancing turnover)
    n_assets = cfg.env.expected_assets
    ew_w = np.full(n_assets, 1.0 / n_assets)  # start at equal weight
    ew_pnls = []
    ew_turnovers = []
    for p in portfolio_steps:
        rets = np.array(p["returns"])
        # PnL using drifted weights (not forced 1/N)
        ew_pnl = float(np.dot(ew_w, rets))
        ew_pnls.append(ew_pnl)
        # After returns, weights drift
        ew_w_after = ew_w * (1.0 + rets)
        ew_w_after = ew_w_after / ew_w_after.sum()  # renormalize
        # Rebalance back to 1/N — that's the turnover
        ew_target = np.full(n_assets, 1.0 / n_assets)
        ew_turnover = float(np.abs(ew_w_after - ew_target).sum())
        ew_turnovers.append(ew_turnover)
        # Reset to equal weight for next period (daily rebalancing)
        ew_w = ew_target

    ew_cumulative = np.prod([1 + p for p in ew_pnls]) - 1
    ew_avg_daily = np.mean(ew_pnls)
    ew_vol = np.std(ew_pnls) * np.sqrt(252)
    ew_sharpe = (ew_avg_daily * 252) / ew_vol if ew_vol > 0 else 0
    ew_avg_turnover = np.mean(ew_turnovers)
    ew_mdd = max_drawdown(ew_pnls)
    
    # MVO benchmark (22-day lookback for expected returns & covariance)
    mvo_lookback = 22
    mvo_w = np.full(n_assets, 1.0 / n_assets)
    mvo_pnls = []
    mvo_turnovers = []
    
    for i, p in enumerate(portfolio_steps):
        rets = np.array(p["returns"])
        mvo_pnl = float(np.dot(mvo_w, rets))
        mvo_pnls.append(mvo_pnl)
        
        # Drift after returns
        mvo_w_after = mvo_w * (1.0 + rets)
        mvo_w_after = mvo_w_after / mvo_w_after.sum()
        
        # Compute new MVO target if we have enough history
        if i >= mvo_lookback:
            past = np.array([portfolio_steps[j]["returns"]
                             for j in range(i - mvo_lookback + 1, i + 1)])
            mu = past.mean(axis=0)
            cov = np.cov(past.T) + 1e-8 * np.eye(n_assets)
            
            def neg_util(w):
                return 0.5 * w @ cov @ w - mu @ w
            def neg_util_jac(w):
                return cov @ w - mu
            
            res = scipy_minimize(
                neg_util, np.full(n_assets, 1.0 / n_assets),
                jac=neg_util_jac, method="SLSQP",
                bounds=[(0.0, 1.0)] * n_assets,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            )
            mvo_target = res.x if res.success else np.full(n_assets, 1.0 / n_assets)
        else:
            mvo_target = np.full(n_assets, 1.0 / n_assets)
        
        mvo_turnover = float(np.abs(mvo_w_after - mvo_target).sum())
        mvo_turnovers.append(mvo_turnover)
        mvo_w = mvo_target
    
    mvo_cumulative = np.prod([1 + p for p in mvo_pnls]) - 1
    mvo_avg_daily = np.mean(mvo_pnls)
    mvo_vol = np.std(mvo_pnls) * np.sqrt(252)
    mvo_sharpe = (mvo_avg_daily * 252) / mvo_vol if mvo_vol > 0 else 0
    mvo_avg_turnover = np.mean(mvo_turnovers)
    mvo_mdd = max_drawdown(mvo_pnls)
    
    print(f"\n{'='*65}")
    print(f"  Backtest Results: {dates[0].date()} to {dates[-1].date()}")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'SAC':>12} {'EW':>12} {'MVO-22d':>12}")
    print(f"{'-'*65}")
    print(f"{'Cumulative Return':<25} {cumulative_return:>11.2%} {ew_cumulative:>11.2%} {mvo_cumulative:>11.2%}")
    print(f"{'Annualized Sharpe':<25} {sharpe:>12.2f} {ew_sharpe:>12.2f} {mvo_sharpe:>12.2f}")
    print(f"{'Annualized Volatility':<25} {volatility:>11.2%} {ew_vol:>11.2%} {mvo_vol:>11.2%}")
    print(f"{'Max Drawdown':<25} {mdd:>11.2%} {ew_mdd:>11.2%} {mvo_mdd:>11.2%}")
    print(f"{'Avg Daily Turnover':<25} {avg_turnover:>11.2%} {ew_avg_turnover:>11.2%} {mvo_avg_turnover:>11.2%}")
    print(f"{'-'*65}")
    print(f"Total critic updates: {critic_updates}")
    print(f"Final temperature: {float(actor.temperature.detach().cpu()):.4f}")
    
    return logs, portfolio_steps, ew_pnls, ew_turnovers, mvo_pnls, mvo_turnovers


def main(cfg: Config | None = None):
    """Run backtest and save results."""
    if cfg is None:
        cfg = Config.default()
    
    logs, portfolio_steps, ew_pnls, ew_turnovers, mvo_pnls, mvo_turnovers = run_backtest(cfg)
    
    # Save results
    output_path = Path(cfg.backtest.logs_path)
    if logs:
        output_path.write_text(pd.DataFrame(logs).to_json(orient="records"))
    
    portfolio_path = Path(cfg.backtest.portfolios_path)
    if portfolio_steps:
        portfolio_path.write_text(pd.DataFrame(portfolio_steps).to_json(orient="records"))
    
    # Save a clean CSV with one column per ticker weight + PnL + turnover
    if portfolio_steps:
        csv_rows = []
        for step in portfolio_steps:
            row = {"date": step["date"]}
            # Add weight per ticker
            tickers = step["asset_ids"]
            weights = step["weights"]
            for ticker, w in zip(tickers, weights):
                row[f"w_{ticker}"] = round(w, 6)
            # Add returns per ticker
            for ticker, r in zip(tickers, step["returns"]):
                row[f"ret_{ticker}"] = round(r, 6)
            row["portfolio_pnl"] = round(step["pnl"], 6)
            row["turnover"] = round(step["turnover"], 6)
            idx = len(csv_rows)
            row["equal_weight_pnl"] = round(ew_pnls[idx], 6)
            row["equal_weight_turnover"] = round(ew_turnovers[idx], 6)
            row["mvo_pnl"] = round(mvo_pnls[idx], 6)
            row["mvo_turnover"] = round(mvo_turnovers[idx], 6)
            csv_rows.append(row)
        
        tag = (f"tc{cfg.sac.turnover_coef}_vc{cfg.sac.variance_coef}"
               f"_vl{cfg.sac.variance_lookback}_rs{cfg.sac.reward_scale}")
        base = Path(cfg.backtest.portfolios_path).stem
        csv_path = Path(cfg.backtest.portfolios_path).parent / f"{base}_{tag}.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved portfolio weights CSV to {csv_path}")
    
    print(f"\nSaved logs to {output_path}")
    print(f"Saved portfolios to {portfolio_path}")


if __name__ == "__main__":
    main()
