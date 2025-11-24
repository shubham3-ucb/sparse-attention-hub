"""Utilities for Rotary Position Embedding (RoPE) operations."""

from typing import List, Optional, Tuple
import os

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input.

    Args:
        x: Input tensor of shape (..., head_dim) where head_dim is even.

    Returns:
        Rotated tensor with same shape as input.
    """
    x1: torch.Tensor = x[..., : x.shape[-1] // 2]
    x2: torch.Tensor = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys.

    Args:
        q: Query tensor of shape (..., seq_len, head_dim).
        k: Key tensor of shape (..., seq_len, head_dim).
        cos: Cosine embeddings of shape (..., seq_len, head_dim).
        sin: Sine embeddings of shape (..., seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    q_embed: torch.Tensor = (q * cos) + (rotate_half(q) * sin)
    k_embed: torch.Tensor = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def unapply_rotary_pos_emb(
    q_rot: torch.Tensor, k_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse rotary position embedding to get unroped queries and keys.

    This is the inverse of apply_rotary_pos_emb.
    
    Standard RoPE forward: x_rot = x * cos + rotate_half(x) * sin
    For a 2D rotation matrix R = [[cos, -sin], [sin, cos]]:
    - Forward: [x1_rot, x2_rot] = R @ [x1, x2]
    - Inverse: [x1, x2] = R^T @ [x1_rot, x2_rot] = [[cos, sin], [-sin, cos]] @ [x1_rot, x2_rot]
    
    So: x = x_rot * cos + rotate_half(x_rot) * sin
    (where rotate_half([a, b]) = [-b, a])

    Args:
        q_rot: Rotated query tensor of shape (..., seq_len, head_dim).
        k_rot: Rotated key tensor of shape (..., seq_len, head_dim).
        cos: Cosine embeddings of shape (..., seq_len, head_dim).
        sin: Sine embeddings of shape (..., seq_len, head_dim).

    Returns:
        Tuple of (unroped_q, unroped_k) with same shapes as inputs.
    """
    # Inverse RoPE: x = R^T @ x_rot where R^T = [[cos, sin], [-sin, cos]]
    # For [x1_rot, x2_rot]: [x1, x2] = [cos*x1_rot + sin*x2_rot, -sin*x1_rot + cos*x2_rot]
    # This equals: x_rot * cos + rotate_half_negate(x_rot) * sin
    # where rotate_half_negate([a, b]) = [b, -a] (opposite of rotate_half)
    # We can write as: x_rot * cos - rotate_half(x_rot) * sin (negate the sin part)
    # OR: x_rot * cos + rotate_half(-x_rot) * sin (negate input)
    # Simplest: negate sin: x_rot * cos - rotate_half(x_rot) * sin
    q_unroped: torch.Tensor = (q_rot * cos) - (rotate_half(q_rot) * sin)
    k_unroped: torch.Tensor = (k_rot * cos) - (rotate_half(k_rot) * sin)
    return q_unroped, k_unroped


def unapply_rotary_pos_emb_queries(
    q_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reverse rotary position embedding for queries only.

    Args:
        q_rot: Rotated query tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine embeddings of shape (batch, seq_len, head_dim) or (1, seq_len, head_dim).
        sin: Sine embeddings of shape (batch, seq_len, head_dim) or (1, seq_len, head_dim).

    Returns:
        Unroped query tensor with same shape as input.
    """
    # Reshape cos/sin to match q_rot: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    # q_rot is (batch, heads, seq_len, head_dim)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    # Inverse RoPE: x = R^T @ x_rot = x_rot * cos - rotate_half(x_rot) * sin
    q_unroped: torch.Tensor = (q_rot * cos) - (rotate_half(q_rot) * sin)
    return q_unroped


def unapply_rotary_pos_emb_keys(
    k_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reverse rotary position embedding for keys only.

    Args:
        k_rot: Rotated key tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine embeddings of shape (batch, seq_len, head_dim) or (1, seq_len, head_dim).
        sin: Sine embeddings of shape (batch, seq_len, head_dim) or (1, seq_len, head_dim).

    Returns:
        Unroped key tensor with same shape as input.
    """
    # Reshape cos/sin to match k_rot: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    # k_rot is (batch, heads, seq_len, head_dim)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    # Inverse RoPE: x = R^T @ x_rot = x_rot * cos - rotate_half(x_rot) * sin
    k_unroped: torch.Tensor = (k_rot * cos) - (rotate_half(k_rot) * sin)
    return k_unroped


def compute_rope_cos_sin(
    module: nn.Module,
    position_ids: torch.Tensor,
    seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin for RoPE from position_ids using the module's rotary embedding.

    Args:
        module: The attention module (should have rotary_emb attribute or config).
        position_ids: Position IDs tensor of shape (batch_size, seq_len).
        seq_len: Optional sequence length (inferred from position_ids if not provided).

    Returns:
        Tuple of (cos, sin) tensors of shape (batch_size, seq_len, head_dim).

    Raises:
        RuntimeError: If unable to compute cos/sin from the module.
    """
    # Method 1: Use module's rotary_emb if available directly
    if hasattr(module, "rotary_emb"):
        rotary_emb = module.rotary_emb
        try:
            # Some rotary_emb.forward() takes (x, position_ids), some just (position_ids)
            # Try both signatures
            try:
                # Llama-style: rotary_emb(x, position_ids) where x is dummy tensor
                # We need a dummy tensor with correct dtype/device - use position_ids shape
                dummy_x: torch.Tensor = torch.zeros(
                    position_ids.shape[0], 1, device=position_ids.device, dtype=torch.float32
                )
                cos, sin = rotary_emb(dummy_x, position_ids)
                return cos, sin
            except (TypeError, ValueError):
                # Some models: rotary_emb(position_ids) directly
                cos, sin = rotary_emb(position_ids)
                return cos, sin
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute cos/sin from module.rotary_emb.forward(): {e}"
            ) from e
    
    # Method 2: Try to get rotary_emb from sparse_meta_data (stored by adapter)
    # This is the most reliable method - the adapter stores it when available
    # Skip this check here - we'll check sparse_meta_data in base.py before calling this function
    
    # Method 3: Compute from config (fallback - requires manual computation)
    if hasattr(module, "config"):
        config = module.config
        if hasattr(config, "rope_theta"):
            # This would require implementing full RoPE computation
            # For now, raise error to indicate we need rotary_emb
            raise RuntimeError(
                "Module has rope_theta in config but no rotary_emb found in module hierarchy. "
                "Manual RoPE computation not yet implemented. "
                f"Module type: {type(module)}"
            )
    
    raise RuntimeError(
        "Unable to compute cos/sin: no rotary_emb found in module hierarchy. "
        f"Module type: {type(module)}, available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
    )


# Repositioning functions removed - keeping code clean for unroped mask computation only

