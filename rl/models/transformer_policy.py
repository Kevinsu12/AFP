"""Transformer-based policy for portfolio management."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.
    
    Implements the positional encoding from "Attention Is All You Need".
    """
    
    def __init__(self, d_model, max_len=5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (N, W, d_model)
            
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerPolicy(nn.Module):
    """Transformer-based policy for portfolio management.
    
    This policy uses a transformer encoder to process sequential factor data
    and outputs portfolio weights via cross-attention.
    """
    
    def __init__(self, K, d_model=64, nhead=4, num_layers=2, hidden=64, 
                 dropout=0.1, activation="gelu", norm_first=True, 
                 ff_hidden_mult=4, max_len=5000):
        """Initialize the transformer policy.
        
        Args:
            K: Number of input features (factors)
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            hidden: Hidden dimension for final scoring layer
            dropout: Dropout rate for attention and feedforward layers
            activation: Activation function ("gelu", "relu", "swish", etc.)
            norm_first: Whether to apply layer norm before attention/ff (Pre-LN vs Post-LN)
            ff_hidden_mult: Multiplier for feedforward hidden dimension (ff_hidden = d_model * ff_hidden_mult)
            max_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        # Always use per-factor masks, so input is 2*K (features + masks)
        input_dim = 2 * K
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding with configurable max length
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        # Transformer encoder with configurable parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * ff_hidden_mult,
            dropout=dropout, 
            activation=activation, 
            norm_first=norm_first,
            batch_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention with configurable dropout
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable portfolio query vector (better initialization)
        self.port_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Initialize scoring network weights properly
        self._init_weights()
        
        # Scoring network with configurable parameters
        self.score = nn.Sequential(
            nn.Linear(2 * d_model, hidden),  # 2*d_model for concatenated features
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def _get_activation(self, activation_name):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # SiLU is the same as Swish
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}. "
                           f"Supported: {list(activations.keys())}")
        return activations[activation_name.lower()]
    
    def _init_weights(self):
        """Initialize network weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, X_t, M_feat):
        """Forward pass through the policy.
        
        Args:
            X_t: Input tensor of shape (N, W, K) where N is number of assets,
                 W is window size, K is number of features
            M_feat: Per-factor mask tensor of shape (N, W, K)
            
        Returns:
            weights: Portfolio weights of shape (N,) that sum to 1
        """
        N, W, K = X_t.shape
        
        # Always concatenate features with per-factor masks
        x_in = torch.cat([X_t, M_feat.float()], dim=-1)  # (N, W, 2K)
        
        x = self.input_proj(x_in)
        
        # Add positional encoding along the time dimension (W)
        x = self.pos_encoding(x)  # Shape: (N, W, d_model)
            
        h_seq = self.seq_encoder(x) 
        h_asset = h_seq.mean(dim=1)  # Average over time dimension (N, d_model)
        
        # Use learnable portfolio query vector for cross-attention
        KV = h_asset.unsqueeze(0)  # (1, N, d_model) - keys/values are all asset embeddings
        Q = self.port_q.expand(1, 1, -1)  # (1, 1, d_model) - single query attends over all assets
        
        port_emb, attn_weights = self.cross_attn(Q, KV, KV)  # (1, 1, d_model)
        port_emb = port_emb.squeeze(0).expand(h_asset.size(0), -1)  # (N, d_model) - expand to all assets
        
        # Add some diversity by using both asset embeddings and portfolio context
        # Instead of just concatenating, use a more sophisticated combination
        asset_context = h_asset + 0.1 * port_emb  # Add portfolio context to asset embeddings
        combined_emb = torch.cat([h_asset, asset_context], dim=-1)  # (N, 2*d_model)
        
        scores = self.score(combined_emb).squeeze(-1)  
        
        # Add temperature scaling for better control over weight distribution
        temperature = 1.0
        logits = scores / temperature
        
        # Numerical stability with better handling
        logits = logits - logits.max(dim=0, keepdim=True)[0]
        weights = torch.softmax(logits, dim=0)
        
        # Add small amount of noise during training to encourage exploration
        if self.training:
            noise = torch.randn_like(weights) * 0.01
            weights = weights + noise
            weights = torch.softmax(weights, dim=0)
        

        return weights
