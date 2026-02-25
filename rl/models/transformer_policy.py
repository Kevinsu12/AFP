"""Transformer backbone for portfolio management models."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.
    
    Implements the positional encoding from "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (time steps in factor history)
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (B*N, W, d_model)
            
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerBackbone(nn.Module):
    """Shared transformer backbone for actor and critic.
    
    This backbone processes factor history for each asset using a transformer
    encoder, then uses cross-attention to create portfolio-aware embeddings.
    
    Input shapes:
        X_t: (B, N, W, K) or (N, W, K) - factor values
        M_feat: (B, N, W, K) or (N, W, K) - per-factor validity masks
        prev_w: (B, N) or (N,) - previous portfolio weights (optional)
    
    Where:
        B = batch size
        N = number of assets (fixed, e.g., 15)
        W = window size (time steps of history)
        K = number of factors
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
        """Initialize transformer backbone.
        
        Args:
            K: Number of input factors
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            hidden: Hidden dimension for scoring MLP
            dropout: Dropout rate
            activation: Activation function name
            norm_first: Use pre-norm (True) or post-norm (False)
            ff_hidden_mult: Feedforward hidden dimension multiplier
            max_len: Maximum sequence length for positional encoding
            include_prev_w: Whether to include previous weights as input feature
        """
        super().__init__()

        self.K = K
        self.d_model = d_model
        self.hidden = hidden
        self.dropout = dropout
        self.activation_name = activation
        self.include_prev_w = include_prev_w

        # Input: concatenate features + per-factor masks → 2*K (or 2*(K+1) with prev_w)
        extra = 1 if include_prev_w else 0
        input_dim = 2 * (K + extra)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding for time dimension
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_hidden_mult,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-asset self-attention: each asset attends to all others
        # This gives each asset a UNIQUE context based on the full portfolio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Scoring network: combined_emb → scalar score per asset
        self.score = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        if activation_name.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {activation_name}. "
                f"Supported: {list(activations.keys())}"
            )
        return activations[activation_name.lower()]

    # Keep public alias for external use (e.g., TransformerQCritic)
    def get_activation(self, activation_name: str) -> nn.Module:
        """Public alias for _get_activation."""
        return self._get_activation(activation_name)

    def _init_weights(self) -> None:
        """Initialize network weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def _ensure_batch(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        """Ensure inputs have batch dimension."""
        squeeze = False
        if X_t.dim() == 3:
            X_t = X_t.unsqueeze(0)
            M_feat = M_feat.unsqueeze(0)
            if prev_w is not None and prev_w.dim() == 1:
                prev_w = prev_w.unsqueeze(0)
            squeeze = True
        return X_t, M_feat, prev_w, squeeze

    def _augment_inputs(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally append prev_w as an extra feature."""
        if not self.include_prev_w:
            return X_t, M_feat
        if prev_w is None:
            prev_w = torch.zeros(
                (X_t.shape[0], X_t.shape[1]), device=X_t.device, dtype=X_t.dtype
            )
        if prev_w.dim() == 1:
            prev_w = prev_w.unsqueeze(0)
        # Broadcast prev_w across time dimension: (B, N) → (B, N, W, 1)
        prev_feat = prev_w.view(X_t.shape[0], X_t.shape[1], 1, 1).expand(
            -1, -1, X_t.shape[2], 1
        )
        X_t = torch.cat([X_t, prev_feat], dim=-1)
        M_feat = torch.cat([M_feat, torch.ones_like(prev_feat, dtype=torch.bool)], dim=-1)
        return X_t, M_feat

    def asset_embeddings(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode per-asset embeddings from factor history.
        
        Args:
            X_t: Factor values (B, N, W, K) or (N, W, K)
            M_feat: Per-factor masks (B, N, W, K) or (N, W, K)
            prev_w: Previous weights (B, N) or (N,), optional
            
        Returns:
            h_asset: Per-asset embeddings (B, N, d_model) or (N, d_model)
        """
        X_t, M_feat, prev_w, squeeze = self._ensure_batch(X_t, M_feat, prev_w)
        X_t, M_feat = self._augment_inputs(X_t, M_feat, prev_w)

        # Concatenate features with masks: (B, N, W, 2K)
        x_in = torch.cat([X_t, M_feat.float()], dim=-1)
        bsz, n_assets, window, feat_dim = x_in.shape

        # Project and encode each asset's time series
        x = self.input_proj(x_in.view(bsz * n_assets, window, feat_dim))
        x = self.pos_encoding(x)  # (B*N, W, d_model)
        h_seq = self.seq_encoder(x)

        # Mean pool over time to get per-asset embedding
        h_asset = h_seq.mean(dim=1).view(bsz, n_assets, self.d_model)

        if squeeze:
            h_asset = h_asset.squeeze(0)
        return h_asset

    def combined_embeddings(self, h_asset: torch.Tensor) -> torch.Tensor:
        """Build combined embeddings using cross-asset attention context.
        
        Each asset attends to all assets (self-attention over the asset
        dimension), producing a UNIQUE context for every asset.  This lets
        the scoring network see how each asset relates to the rest of the
        portfolio, which is critical for producing differentiated weights.
        
        Args:
            h_asset: Per-asset embeddings (B, N, d_model) or (N, d_model)
            
        Returns:
            combined_emb: (B, N, 2*d_model) or (N, 2*d_model)
        """
        squeeze = False
        if h_asset.dim() == 2:
            h_asset = h_asset.unsqueeze(0)
            squeeze = True

        # Cross-asset self-attention: Q = K = V = h_asset
        # Each asset_i gets a unique context = weighted avg of all assets
        cross_ctx, _ = self.cross_attn(h_asset, h_asset, h_asset)  # (B, N, d_model)

        # Combine: [per-asset embedding | cross-asset context]
        combined_emb = torch.cat([h_asset, cross_ctx], dim=-1)  # (B, N, 2*d_model)

        if squeeze:
            combined_emb = combined_emb.squeeze(0)
        return combined_emb

    def scores(
        self,
        X_t: torch.Tensor,
        M_feat: torch.Tensor,
        prev_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-asset scores (logits before softmax).
        
        Args:
            X_t: Factor values (B, N, W, K) or (N, W, K)
            M_feat: Per-factor masks (B, N, W, K) or (N, W, K)
            prev_w: Previous weights (B, N) or (N,), optional
            
        Returns:
            scores: Per-asset scores (B, N) or (N,)
        """
        h_asset = self.asset_embeddings(X_t, M_feat, prev_w=prev_w)
        combined_emb = self.combined_embeddings(h_asset)
        return self.score(combined_emb).squeeze(-1)
