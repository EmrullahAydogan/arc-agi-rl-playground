"""
Attention Module for ARC Puzzles
Transformer-based attention for focusing on important grid regions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    Adds position information to embeddings
    """

    def __init__(self, d_model: int, max_len: int = 900):  # 30x30 = 900
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class GridAttentionLayer(nn.Module):
    """
    Single transformer attention layer for grids
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: Optional attention mask

        Returns:
            (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x, attn_weights


class GridTransformer(nn.Module):
    """
    Transformer for ARC grids
    Learns to attend to important regions
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_colors = num_colors
        self.d_model = d_model

        # Embedding layer: Convert colors to embeddings
        self.color_embedding = nn.Embedding(num_colors, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=grid_size * grid_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            GridAttentionLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection (for classification tasks)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        grid: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            grid: (batch, height, width) with values 0-9
            return_attention: Whether to return attention weights

        Returns:
            (output, attention_weights)
            - output: (batch, seq_len, d_model)
            - attention_weights: (batch, num_layers, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, h, w = grid.shape

        # Flatten grid to sequence
        grid_flat = grid.view(batch_size, -1)  # (batch, h*w)

        # Embed colors
        x = self.color_embedding(grid_flat.long())  # (batch, h*w, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer layers
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            if return_attention:
                all_attention_weights.append(attn_weights)

        # Output projection
        x = self.output_proj(x)

        # Return attention weights if requested
        if return_attention:
            # Stack attention weights: (batch, num_layers, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(all_attention_weights, dim=1)
            return x, attention_weights

        return x, None

    def get_attention_map(
        self,
        grid: np.ndarray,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> np.ndarray:
        """
        Get attention map for visualization

        Args:
            grid: (H, W) numpy array
            layer_idx: Which layer's attention to visualize (-1 = last layer)
            head_idx: Which attention head to visualize

        Returns:
            Attention map: (H, W) numpy array
        """
        h, w = grid.shape

        # Pad to grid_size if needed
        if h != self.grid_size or w != self.grid_size:
            padded = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            padded[:h, :w] = grid
            grid = padded

        # To tensor
        grid_tensor = torch.LongTensor(grid).unsqueeze(0)  # (1, H, W)

        with torch.no_grad():
            _, attention_weights = self.forward(grid_tensor, return_attention=True)

        # Extract specific layer and head
        # attention_weights: (1, num_layers, num_heads, seq_len, seq_len)
        attn = attention_weights[0, layer_idx, head_idx]  # (seq_len, seq_len)

        # Average attention received by each position
        attn_map = attn.mean(dim=0)  # (seq_len,)

        # Reshape to grid
        attn_map = attn_map.view(self.grid_size, self.grid_size)

        # Crop to original size
        attn_map = attn_map[:h, :w]

        return attn_map.cpu().numpy()


class AttentionPolicyHead(nn.Module):
    """
    Policy head with attention mechanism
    Uses transformer output to make decisions
    """

    def __init__(
        self,
        d_model: int = 128,
        num_actions: int = 9004,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.d_model = d_model
        self.num_actions = num_actions

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transformer_output: (batch, seq_len, d_model)

        Returns:
            action_logits: (batch, num_actions)
        """
        # Global average pooling
        x = transformer_output.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)

        # Action head
        action_logits = self.action_head(x)

        return action_logits


class TransformerAgent(nn.Module):
    """
    Complete Transformer-based agent
    Combines transformer with policy head
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        num_actions: int = 9004
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_colors = num_colors

        # Transformer
        self.transformer = GridTransformer(
            grid_size=grid_size,
            num_colors=num_colors,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )

        # Policy head
        self.policy_head = AttentionPolicyHead(
            d_model=d_model,
            num_actions=num_actions
        )

    def forward(
        self,
        grid: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            grid: (batch, H, W)
            return_attention: Whether to return attention weights

        Returns:
            (action_logits, attention_weights)
        """
        # Transformer
        transformer_output, attention_weights = self.transformer(
            grid,
            return_attention=return_attention
        )

        # Policy
        action_logits = self.policy_head(transformer_output)

        return action_logits, attention_weights

    def select_action(
        self,
        grid: np.ndarray,
        mode: str = 'exploit'
    ) -> int:
        """
        Select action for grid

        Args:
            grid: (H, W) numpy array
            mode: 'exploit' or 'explore'

        Returns:
            action: Integer action ID
        """
        h, w = grid.shape

        # Pad to grid_size
        padded = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        padded[:h, :w] = grid

        # To tensor
        grid_tensor = torch.LongTensor(padded).unsqueeze(0)

        with torch.no_grad():
            action_logits, _ = self.forward(grid_tensor)
            action_logits = action_logits[0]  # Remove batch dim

            if mode == 'explore':
                # Sample from distribution
                probs = F.softmax(action_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
            else:
                # Greedy
                action = torch.argmax(action_logits).item()

        return action

    def visualize_attention(
        self,
        grid: np.ndarray,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> np.ndarray:
        """Get attention map for visualization"""
        return self.transformer.get_attention_map(grid, layer_idx, head_idx)


# Utility functions

def visualize_attention_heatmap(
    attention_map: np.ndarray,
    original_grid: np.ndarray,
    cmap: str = 'hot'
) -> np.ndarray:
    """
    Create visualization of attention map

    Args:
        attention_map: (H, W) attention weights
        original_grid: (H, W) original grid
        cmap: Matplotlib colormap name

    Returns:
        Visualization grid (for display purposes)
    """
    # Normalize attention map
    attn_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    # For now, just return normalized attention
    # In a real implementation, you'd overlay this on the grid
    return attn_normalized
