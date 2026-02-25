"""Transformer-based actor-critic models for SAC portfolio management."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

from .transformer_policy import TransformerBackbone

if TYPE_CHECKING:
    from rl.configs.sac_config import ModelConfig


class TransformerActor(nn.Module):
    """Actor that outputs portfolio weights via softmax with learnable temperature.
    
    Uses Gumbel-Softmax (RelaxedOneHotCategorical) for differentiable stochastic
    sampling during training, and softmax(scores / temperature) for deterministic
    evaluation.
    
    Ensures:
    - All weights are non-negative (long-only)
    - Weights sum to 1 (fully invested)
    - Learnable temperature controls weight concentration
    """

    def __init__(
        self,
        K: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        hidden: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        ff_hidden_mult: int = 4,
        max_len: int = 5000,
        include_prev_w: bool = False,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(
            K=K,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            ff_hidden_mult=ff_hidden_mult,
            max_len=max_len,
            include_prev_w=include_prev_w,
        )
        # Learnable temperature (log-parameterized for guaranteed positivity)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature, clamped to safe range [0.05, 5.0]."""
        return self.log_temperature.exp().clamp(min=0.05, max=5.0)

    @classmethod
    def from_config(cls, K: int, cfg: ModelConfig) -> TransformerActor:
        """Create actor from ModelConfig."""
        return cls(
            K=K,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
            activation=cfg.activation,
            norm_first=cfg.norm_first,
            ff_hidden_mult=cfg.ff_hidden_mult,
            max_len=cfg.max_len,
            include_prev_w=cfg.include_prev_w,
            temperature_init=cfg.temperature_init,
        )

    def forward(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return deterministic weights for evaluation.
        
        Computes softmax(scores / temperature).  Lower temperature → sharper
        (more concentrated) weights; higher → closer to uniform.
        
        Args:
            X_t: Factor values (B, N, W, K) or (N, W, K)
            M_feat: Per-factor masks
            prev_w: Previous portfolio weights
            
        Returns:
            weights: softmax portfolio weights (B, N) or (N,)
        """
        scores = self.backbone.scores(X_t, M_feat, prev_w=prev_w)
        dim = -1 if scores.dim() > 1 else 0
        return F.softmax(scores / self.temperature, dim=dim)

    def get_dist(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None = None,
    ) -> RelaxedOneHotCategorical:
        """Get Gumbel-Softmax distribution for stochastic sampling.
        
        The RelaxedOneHotCategorical (Gumbel-Softmax) distribution:
        - Produces differentiable samples on the simplex (sum to 1, all >= 0)
        - Supports rsample() for reparameterised gradients
        - Has well-defined log_prob() for SAC entropy computation
        
        Use dist.rsample() for training (reparameterised gradient).
        """
        scores = self.backbone.scores(X_t, M_feat, prev_w=prev_w)
        return RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=scores,
        )


class TransformerQCritic(nn.Module):
    """Q-critic that estimates Q(s, a) for portfolio actions.
    
    The critic takes state (factor history) and action (portfolio weights)
    and outputs a scalar Q-value representing expected cumulative reward.
    """

    def __init__(
        self,
        K: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        hidden: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        ff_hidden_mult: int = 4,
        max_len: int = 5000,
        include_prev_w: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(
            K=K,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            ff_hidden_mult=ff_hidden_mult,
            max_len=max_len,
            include_prev_w=include_prev_w,
        )
        # Q-head: takes combined embedding + action weight per asset
        self.q_head = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            self.backbone.get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_q_head()

    @classmethod
    def from_config(cls, K: int, cfg: ModelConfig) -> TransformerQCritic:
        """Create critic from ModelConfig."""
        return cls(
            K=K,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            hidden=cfg.hidden,
            dropout=cfg.dropout,
            activation=cfg.activation,
            norm_first=cfg.norm_first,
            ff_hidden_mult=cfg.ff_hidden_mult,
            max_len=cfg.max_len,
            include_prev_w=cfg.include_prev_w,
        )

    def _init_q_head(self) -> None:
        """Initialize Q-head weights."""
        for module in self.q_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        action: torch.Tensor,
        prev_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Q(s, a).
        
        Args:
            X_t: Factor values (B, N, W, K) or (N, W, K)
            M_feat: Per-factor masks
            action: Portfolio weights (B, N) or (N,)
            prev_w: Previous portfolio weights
            
        Returns:
            Q-value: Scalar (batch) or scalar (single)
        """
        h_asset = self.backbone.asset_embeddings(X_t, M_feat, prev_w=prev_w)
        combined = self.backbone.combined_embeddings(h_asset)

        # Append action weight to each asset's embedding
        if action.dim() == 1:
            action_feat = action.view(-1, 1)
        else:
            action_feat = action.view(action.shape[0], action.shape[1], 1)

        q_in = torch.cat([combined, action_feat], dim=-1)
        q_per_asset = self.q_head(q_in).squeeze(-1)

        # Sum over assets to get total Q-value
        if q_per_asset.dim() == 1:
            return q_per_asset.sum()
        return q_per_asset.sum(dim=1)
