"""Centralized configuration for SAC portfolio management."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    hidden: int = 64
    dropout: float = 0.15
    activation: str = "gelu"
    norm_first: bool = True
    ff_hidden_mult: int = 2
    max_len: int = 5000
    include_prev_w: bool = True
    temperature_init: float = 1.0


@dataclass
class EnvConfig:
    data_dir: str = "data"
    split: str = "processed_daily_panel.parquet"
    ret_col: str = "ret_next"
    id_col: str = "ticker"
    date_col: str = "date"
    window: int = 48
    min_obs_in_window: int = 1
    expected_assets: int = 10
    exclude_cols: tuple = ()


@dataclass
class SACConfig:
    replay_size: int = 50_000
    batch_size: int = 32
    start_steps: int = 50

    update_every: int = 1
    updates_per_step: int = 1
    policy_delay: int = 2

    gamma: float = 0.9
    tau: float = 0.005

    turnover_coef: float = 0.01
    variance_coef: float = 0
    variance_lookback: int = 22
    reward_scale: float = 500.0

    init_alpha: float = 0.02
    min_alpha: float = 0.001
    alpha_decay_steps: int = 7500

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class PretrainConfig:
    num_random_portfolios: int = 100
    portfolio_size: int = 10
    include_final: bool = True
    portfolios_file: str = "data/portfolios.json"

    start_date: str = "2012-01-01"
    end_date: str = "2021-12-31"
    epochs: int = 3

    checkpoint_path: str = "checkpoints/pretrained/"
    seed: int = 42


@dataclass
class BacktestConfig:
    portfolio_id: str = "final"
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"

    finetune_actor_lr: float = 1e-5
    finetune_critic_lr: float = 1e-4

    finetune_init_alpha: float = 0.01
    finetune_min_alpha: float = 0.001
    finetune_alpha_decay_steps: int = 300

    warmup_days: int = 50
    enforce_weight_bounds: bool = True

    load_pretrained: bool = True
    pretrained_path: str = "checkpoints/pretrained/"

    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    logs_path: str = "backtest_logs.json"
    portfolios_path: str = "backtest_portfolios.json"
    checkpoint_path: Optional[str] = "checkpoints/finetuned/"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def default(cls) -> "Config":
        return cls()

    def __post_init__(self):
        assert self.model.nhead > 0
        assert self.model.d_model % self.model.nhead == 0
        assert self.env.window > 0
        assert self.env.expected_assets > 0
        assert 0 <= self.sac.gamma <= 1
        assert 0 < self.sac.tau <= 1
        assert self.sac.init_alpha >= self.sac.min_alpha
