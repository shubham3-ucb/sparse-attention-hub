"""Repositioning utilities for sparse attention - extracted for modularity.

This module contains the position reassignment logic extracted from mask_attention_utils.py
to improve code organization and maintainability. All code is copied VERBATIM to ensure
100% functional equivalence.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .mask import Mask
from ..research_attention.rope_utils import (
    unapply_rotary_pos_emb_queries,
    unapply_rotary_pos_emb_keys,
    rotate_half,
)

# Default behavior: enable proportional prefix scaling unless explicitly disabled via kwargs
DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING: bool = True
# Ratio cap for prefix scaling (0 < alpha <= 1). Example: 0.75 compresses prefix
# to 75% of the available pre-chunk span. You can override via kwargs
# 'prefix_scale_alpha' if needed.
DEFAULT_PREFIX_SCALE_ALPHA: float = 1.0
# Default behavior: enable two-band prefix scaling (sinks+tail frozen, middle compressed)
DEFAULT_USE_TWO_BAND_PREFIX_SCALING: bool = True
DEFAULT_PACK_K_CHUNK_TRANSLATION: bool = True
# Hard cap for position IDs (inclusive). Enforce ALWAYS; fatal on violation.
# DEFAULT_MAX_POSITION_ID: int = 8192  # Original value - commented for half-length testing
DEFAULT_MAX_POSITION_ID: int = 4096  # Half-length testing: 8192 / 2 = 4096


def _scale_prefix_positions(
    prefix_positions: torch.Tensor,
    target_max_exclusive: int,
    epsilon: int = 1,
) -> torch.Tensor:
    """Affine-rescale prefix positions into [0, target_max_exclusive - 1 - epsilon].

    Args:
        prefix_positions: 1D tensor (long) of original prefix positions (on any device).
        target_max_exclusive: Upper bound (exclusive) for the target range; typically the
            first position of the current chunk, e.g., min_query_position.
        epsilon: Gap to keep before the chunk start to ensure clean separation.

    Returns:
        Tensor of same shape and device as ``prefix_positions`` with dtype long, scaled
        and clamped to the target range.
    """
    if prefix_positions.numel() == 0:
        return prefix_positions.clone()

    # Compute source range
    min_p: int = int(prefix_positions.min().item())
    max_p: int = int(prefix_positions.max().item())
    denom: int = max(1, max_p - min_p)

    # Compute target range [0, target_max_inclusive]
    target_max_inclusive: int = max(0, target_max_exclusive - 1 - max(0, epsilon))

    # Scale; clamp to avoid expansion (compress-only)
    s_raw: float = float(target_max_inclusive) / float(denom)
    s: float = min(1.0, s_raw)

    # Apply scaling on the same device; cast to float for arithmetic then back to long
    scaled: torch.Tensor = (prefix_positions.to(torch.float32) - float(min_p)) * s
    scaled = torch.round(scaled).to(dtype=torch.long, device=prefix_positions.device)
    return torch.clamp(scaled, min=0, max=target_max_inclusive)


def _apply_position_reassignment(
    queries: torch.Tensor,
    keys: torch.Tensor,
    rotary_emb: Any,
    position_ids_q: torch.Tensor,
    position_ids_k: torch.Tensor,
    sparse_attention_mask: Mask,
    exp_attention_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    training: bool,
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Apply position reassignment and return reroped attention weights.
    
    This function is an EXACT copy of lines 554-1174 from mask_attention_utils.py.
    NO LOGIC CHANGES - just extracted for modularity.
    
    Args:
        queries: Query tensor (batch, num_heads, seq_len_q, head_dim)
        keys: Key tensor (batch, num_kv_heads, seq_len_k, head_dim)
        rotary_emb: Rotary embedding module
        position_ids_q: Position IDs for queries (batch, seq_len_q)
        position_ids_k: Position IDs for keys (batch, seq_len_k)
        sparse_attention_mask: Sparse attention mask object
        exp_attention_weights: Pre-computed exponential attention weights
        attention_mask: Optional attention mask
        scaling: Attention scaling factor
        dropout: Dropout probability
        training: Whether model is in training mode
        kwargs: Additional keyword arguments
    
    Returns:
        Reroped exponential attention weights tensor
    """
    # EXACT COPY OF LINES 554-1174 FROM mask_attention_utils.py
    # NO LOGIC CHANGES - VERBATIM COPY
    
    reposition_start_time: float = time.time()
    try:
        # Compute cos/sin for queries
        # Current: Same position_ids for all heads (will be broadcasted in unapply_rotary_pos_emb_queries)
        # Future: Can extend to per-head position_ids: position_ids_q shape (batch, num_heads, seq_len)
        # Use same dtype as queries to avoid dtype mismatch (bfloat16 vs float32)
        seq_len_keys = keys.shape[2]
        seq_len_q = position_ids_q.shape[1]
        dummy_x_q = torch.zeros(1, position_ids_q.shape[1], device=queries.device, dtype=queries.dtype)
        cos_q, sin_q = rotary_emb(dummy_x_q, position_ids_q)
        # Compute cos/sin for keys
        # Current: Same position_ids for all heads (will be broadcasted in unapply_rotary_pos_emb_keys)
        # Future: Can extend to per-head position_ids: position_ids_k shape (batch, num_heads, seq_len)
        # Use same dtype as keys to avoid dtype mismatch (bfloat16 vs float32)
        dummy_x_k = torch.zeros(1, seq_len_keys, device=keys.device, dtype=keys.dtype)
        cos_k, sin_k = rotary_emb(dummy_x_k, position_ids_k)
        # Unrope
        unroped_queries = unapply_rotary_pos_emb_queries(queries, cos_q, sin_q)
        unroped_keys = unapply_rotary_pos_emb_keys(keys, cos_k, sin_k)
        # Verification: ensure unroped tensors are actually different from roped
        q_diff = torch.abs(queries - unroped_queries).max()
        k_diff = torch.abs(keys - unroped_keys).max()
        
        # Re-apply RoPE to unroped Q/K and verify we get same attention weights
        # Apply RoPE manually using the formula: x_rot = x * cos + rotate_half(x) * sin
        
        # Get sequence lengths
        seq_len_q = position_ids_q.shape[1]
        seq_len_k = keys.shape[2]  # (batch, num_kv_heads, seq_len_k, head_dim)
        
        # ====================================================================
        # Option 1: Gap Closure - Start current chunk at max_prefix + 1
        # ====================================================================
        # Strategy:
        # 1. Find max position among selected prefix keys per head
        # 2. Reassign prefix keys to 0, 1, 2, ... (contiguous)
        # 3. Start current chunk at max_prefix + 1
        # 4. Keep relative distances within current chunk
        # ====================================================================
        num_heads: int = queries.shape[1]
        batch_size: int = queries.shape[0]
        
        # Get actual position IDs for keys (if not provided, infer from queries)
        position_ids_k_actual: torch.Tensor = kwargs.get("position_ids_k", None)
        if position_ids_k_actual is None:
            min_q_pos: int = position_ids_q[0, 0].item()
            position_ids_k_actual = torch.arange(
                min_q_pos - seq_len_k + seq_len_q, min_q_pos + seq_len_q,
                device=keys.device, dtype=torch.long
            ).unsqueeze(0)  # (batch, seq_len_k)
        
        # Identify prefix vs current chunk boundary
        min_query_position: int = position_ids_q[0, 0].item()
        
        # OPTIMIZATION: Pre-compute boundary_inclusive once (same for all heads)
        # This is used in both two-band and proportional prefix scaling modes
        boundary_inclusive: int = max(0, min_query_position - 1)
        
        # Get dense mask to identify selected keys per head
        dense_mask = sparse_attention_mask.get_dense_mask()  # [batch, num_heads, seq_len_q, seq_len_k]
        assert dense_mask is not None, "[ERROR] dense_mask is None - cannot proceed with position reassignment"
        assert dense_mask.shape[0] == batch_size, f"[ERROR] dense_mask batch size mismatch: {dense_mask.shape[0]} != {batch_size}"
        assert dense_mask.shape[1] == num_heads, f"[ERROR] dense_mask num_heads mismatch: {dense_mask.shape[1]} != {num_heads}"
        
        # Initialize per-head position ID tensors
        position_ids_q_per_head: torch.Tensor = torch.zeros(
            batch_size, num_heads, seq_len_q, device=queries.device, dtype=torch.long
        )
        position_ids_k_per_head: torch.Tensor = torch.zeros(
            batch_size, num_heads, seq_len_k, device=keys.device, dtype=torch.long
        )
        
        # Toggle for proportional prefix scaling (clean minimal change)
        use_proportional_prefix_scaling: bool = bool(
            kwargs.get("use_proportional_prefix_scaling", DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING)
        )
        # Choose epsilon adaptively: when there is no space, drop the gap
        adaptive_epsilon: int = 1 if min_query_position > 1 else 0
        # Ratio cap alpha for prefix scaling
        prefix_scale_alpha: float = float(
            kwargs.get("prefix_scale_alpha", DEFAULT_PREFIX_SCALE_ALPHA)
        )
        # Clamp alpha into sensible range
        if prefix_scale_alpha <= 0.0:
            prefix_scale_alpha = 0.0
        if prefix_scale_alpha > 1.0:
            prefix_scale_alpha = 1.0
        # Hard cap: max allowed position id (inclusive)
        # ALWAYS use DEFAULT_MAX_POSITION_ID=8192 - DO NOT allow override via kwargs
        max_position_id_allowed: int = DEFAULT_MAX_POSITION_ID
        assert max_position_id_allowed >= 0, "[FATAL] max_position_id must be non-negative"
        # Two-band scaling controls
        use_two_band_prefix_scaling: bool = bool(
            kwargs.get("use_two_band_prefix_scaling", DEFAULT_USE_TWO_BAND_PREFIX_SCALING)
        )
        # Optional pack: translate K+Chunk down as a block (slope 1)
        pack_k_chunk_translation: bool = bool(
            kwargs.get("pack_k_chunk_translation", DEFAULT_PACK_K_CHUNK_TRANSLATION)
        )
        # Defaults for anchors during two-band scaling (stress: no anchors)
        num_sink_tokens: int = int(kwargs.get("num_sink_tokens", 0))
        prefix_freeze_tail_k: int = int(kwargs.get("prefix_freeze_tail_k", 0))
        enforce_monotone_prefix: bool = bool(kwargs.get("monotone_prefix", True))
        
        # Timing: per-head position reassignment
        per_head_start_time: float = time.time()
        
        # OPTIMIZATION: Cache position_ids_k_actual[0] once (accessed 20+ times per head)
        # This eliminates repeated tensor indexing operations
        position_ids_k_actual_0: torch.Tensor = position_ids_k_actual[0]  # (seq_len_k,)

        # OPTIMIZATION: Batch union computation for all heads at once (outside loop)
        # Shape: dense_mask[0, :, :, :] is (num_heads, seq_len_q, seq_len_k)
        # Use torch.any(dim=1) to get keys attended by ANY query per head: (num_heads, seq_len_k)
        # This is functionally identical to computing union_mask per head inside the loop
        union_masks_all_heads: torch.Tensor = torch.any(dense_mask[0, :, :, :] > 0, dim=1)  # (num_heads, seq_len_k)
        
        # OPTIMIZATION: Pre-compute all_current_chunk_mask once (same for all heads)
        # This mask only depends on position_ids_k_actual_0 and min_query_position, not on head_idx
        all_current_chunk_mask: torch.Tensor = position_ids_k_actual_0 >= min_query_position  # (seq_len_k,)
        all_current_chunk_key_indices_tensor: torch.Tensor = torch.nonzero(all_current_chunk_mask).squeeze(-1)  # (num_current_chunk_keys,)
        
        # OPTIMIZATION: Pre-compute sorted current chunk keys once (same for all heads)
        # Sorting is based on position_ids_k_actual_0, which doesn't depend on head_idx
        num_current_chunk_keys: int = all_current_chunk_key_indices_tensor.numel()
        if num_current_chunk_keys > 0:
            current_chunk_positions_tensor: torch.Tensor = position_ids_k_actual_0[all_current_chunk_key_indices_tensor]  # (num_current_chunk_keys,)
            sort_indices: torch.Tensor = torch.argsort(current_chunk_positions_tensor)  # GPU operation
            all_current_chunk_key_indices_sorted_tensor: torch.Tensor = all_current_chunk_key_indices_tensor[sort_indices]  # GPU gather
            current_chunk_original_pos_tensor: torch.Tensor = current_chunk_positions_tensor[sort_indices]  # Pre-sorted positions
        else:
            all_current_chunk_key_indices_sorted_tensor: Optional[torch.Tensor] = None
            current_chunk_original_pos_tensor: Optional[torch.Tensor] = None
        
        # OPTIMIZATION: Pre-compute query_positions_tensor once (same for all heads)
        # This is just position_ids_q[0, :] which doesn't depend on head_idx
        query_positions_tensor: torch.Tensor = position_ids_q[0, :]  # (seq_len_q,)

        # Process each head independently
        for head_idx in range(num_heads):
            # Per-head translation state for optional packing
            pack_delta: Optional[int] = None
            m_end_for_pack: Optional[int] = None
            # Get union of selected key indices for this head (pre-computed, batched)
            # OPTIMIZATION: Index into pre-computed batched union masks (functionally identical to per-head computation)
            union_mask: torch.Tensor = union_masks_all_heads[head_idx]  # (seq_len_k,)
            # OPTIMIZATION: Keep indices on GPU instead of converting to Python set
            union_key_indices_tensor: torch.Tensor = torch.nonzero(union_mask).squeeze(-1)  # (num_selected_keys,)

            # Separate prefix vs current chunk key indices (OPTIMIZED: vectorized GPU operations)
            num_union_keys: int = union_key_indices_tensor.numel()
            prefix_key_indices_tensor: Optional[torch.Tensor] = None
            current_chunk_key_indices_tensor: Optional[torch.Tensor] = None
            
            if num_union_keys > 0:
                # Batch extract all positions at once (single GPU operation)
                # OPTIMIZATION: Use cached tensor instead of repeated indexing
                union_key_positions_tensor: torch.Tensor = position_ids_k_actual_0[union_key_indices_tensor]  # (num_union_keys,)
                
                # Vectorized separation using GPU boolean masks
                prefix_mask: torch.Tensor = union_key_positions_tensor < min_query_position  # (num_union_keys,)
                prefix_key_indices_tensor = union_key_indices_tensor[prefix_mask]  # (num_prefix_keys,)
                current_chunk_key_indices_tensor = union_key_indices_tensor[~prefix_mask]  # (num_current_chunk_keys,)
                
                # OPTIMIZATION: Keep tensors on GPU - no CPU-GPU transfers
                # prefix_key_indices_tensor and current_chunk_key_indices_tensor stay as tensors
            else:
                # Empty case - set to None to indicate no keys
                prefix_key_indices_tensor = None
                current_chunk_key_indices_tensor = None

            # Sort prefix keys by their original position IDs (OPTIMIZED: GPU-based sorting)
            if prefix_key_indices_tensor is not None and prefix_key_indices_tensor.numel() > 0:
                # Already have tensor from vectorized separation above
                # OPTIMIZATION: Use cached tensor instead of repeated indexing
                prefix_positions_tensor: torch.Tensor = position_ids_k_actual_0[prefix_key_indices_tensor]  # (num_prefix_keys,)
                
                # OPTIMIZATION: Use torch.argsort for GPU-based sorting (functionally identical to Python sort)
                sort_indices: torch.Tensor = torch.argsort(prefix_positions_tensor)  # GPU operation
                prefix_key_indices_sorted_tensor: torch.Tensor = prefix_key_indices_tensor[sort_indices]  # GPU gather
                # OPTIMIZATION: Keep as tensor - no CPU-GPU transfer
                
                # Find max position among selected prefix keys (using sorted positions tensor)
                sorted_positions_tensor: torch.Tensor = prefix_positions_tensor[sort_indices]
                max_prefix_position: int = int(sorted_positions_tensor[-1].item())  # Last position after sorting
            else:
                prefix_key_indices_sorted_tensor: Optional[torch.Tensor] = None
                max_prefix_position: int = -1
                
            # Reassign prefix keys to contiguous positions [0, 1, 2, ...]
            num_prefix_keys: int = prefix_key_indices_sorted_tensor.numel() if prefix_key_indices_tensor is not None else 0
            if num_prefix_keys > 0:
                if use_two_band_prefix_scaling:
                    # ============================
                    # Two-band prefix scaling:
                    # - Freeze first S sinks (by order in prefix)
                    # - Freeze last K prefix tokens near the chunk
                    # - Compress only the middle band proportionally
                    # ============================
                    S: int = max(0, min(num_sink_tokens, num_prefix_keys))
                    # OPTIMIZATION: Reuse pre-computed boundary_inclusive (computed once outside loop, same for all heads)
                    # Enforce hard cap via alpha_eff so that max_new = T_pref + L_q <= B_max
                    # If boundary=0, alpha_eff=0; also ensure L_q <= B_max
                    assert seq_len_q <= (max_position_id_allowed + 1), (
                        f"[FATAL] Chunk length L_q={seq_len_q} exceeds allowed window {max_position_id_allowed+1}"
                    )
                    if boundary_inclusive > 0:
                        cap_ratio: float = float(max(0, max_position_id_allowed - seq_len_q)) / float(boundary_inclusive)
                        alpha_eff: float = max(0.0, min(prefix_scale_alpha, min(1.0, cap_ratio)))
                    else:
                        alpha_eff = 0.0
                    T_pref_inclusive: int = int(alpha_eff * float(boundary_inclusive))
                    # Debug print for cap diagnostics
                    # Sanity: estimated max must fit cap
                    assert (T_pref_inclusive + seq_len_q) <= max_position_id_allowed, (
                        f"[FATAL] Estimated max_new={T_pref_inclusive + seq_len_q} exceeds cap {max_position_id_allowed}"
                    )
                    # Determine requested K and cap it so that target_high_inclusive >= target_low
                    K_requested: int = max(0, min(prefix_freeze_tail_k, max(0, num_prefix_keys - S)))
                    K_cap_by_target: int = max(0, T_pref_inclusive - S)
                    K: int = max(0, min(K_requested, K_cap_by_target))
                    # Debug logging for two-band parameters (only log once per layer, first head)
                    if head_idx == 0 and os.environ.get("SPARSE_DEBUG_POSITIONS", "0").lower() in ("1", "true", "yes"):
                        layer_idx_debug: Optional[int] = kwargs.get("layer_idx", None)
                        layer_str: str = str(layer_idx_debug) if layer_idx_debug is not None else "?"
                        print(f"[DEBUG TWO-BAND] layer={layer_str} num_sink_tokens={num_sink_tokens} prefix_freeze_tail_k={prefix_freeze_tail_k} → S={S} K={K} num_prefix_keys={num_prefix_keys} T_pref={T_pref_inclusive}", flush=True)
                    start_mid: int = S
                    end_mid_exclusive: int = max(S, num_prefix_keys - K)
                    # Prepare original positions for the whole selected prefix (sorted)
                    # OPTIMIZATION: Use tensor directly - no CPU-GPU transfer
                    # OPTIMIZATION: Use cached tensor instead of repeated indexing
                    prefix_positions_tensor_full: torch.Tensor = position_ids_k_actual_0[prefix_key_indices_sorted_tensor]
                    # Freeze sinks: first S elements
                    if S > 0:
                        # OPTIMIZATION: Use tensor slicing - no CPU-GPU transfer
                        sink_indices: torch.Tensor = prefix_key_indices_sorted_tensor[:S]
                        # OPTIMIZATION: Use cached tensor instead of repeated indexing
                        sink_positions: torch.Tensor = position_ids_k_actual_0[sink_indices]
                        position_ids_k_per_head[0, head_idx, sink_indices] = sink_positions
                    # Freeze tail: last K elements
                    if K > 0:
                        # OPTIMIZATION: Use tensor slicing - no CPU-GPU transfer
                        tail_indices: torch.Tensor = prefix_key_indices_sorted_tensor[-K:]
                        # OPTIMIZATION: Use cached tensor instead of repeated indexing
                        tail_positions: torch.Tensor = position_ids_k_actual_0[tail_indices]
                        position_ids_k_per_head[0, head_idx, tail_indices] = tail_positions
                    # Middle band scaling
                    if end_mid_exclusive > start_mid:
                        # OPTIMIZATION: Use tensor slicing - no CPU-GPU transfer
                        middle_indices_sorted: torch.Tensor = prefix_key_indices_sorted_tensor[start_mid:end_mid_exclusive]
                        # OPTIMIZATION: Use cached tensor instead of repeated indexing
                        middle_orig_positions: torch.Tensor = position_ids_k_actual_0[middle_indices_sorted]
                        min_mid: int = int(middle_orig_positions.min().item())
                        max_mid: int = int(middle_orig_positions.max().item())
                        # Target band for middle: [S .. T_pref_inclusive - K]
                        target_low: int = S
                        target_high_inclusive: int = T_pref_inclusive - K
                        # Fatal if target range invalid (never silently fallback)
                        assert (target_high_inclusive >= target_low) and (max_mid >= min_mid), (
                            f"[FATAL] Invalid middle target range: target_low={target_low}, "
                            f"target_high_inclusive={target_high_inclusive}, S={S}, K={K}, T_pref={T_pref_inclusive}"
                        )
                        if target_high_inclusive >= target_low and max_mid >= min_mid:
                            target_span: int = target_high_inclusive - target_low
                            denom_mid: int = max(1, max_mid - min_mid)
                            s_raw_mid: float = float(target_span) / float(denom_mid)
                            s_mid: float = min(1.0, s_raw_mid)
                            # Scale with floor and clamp
                            middle_scaled: torch.Tensor = (
                                (middle_orig_positions.to(torch.float32) - float(min_mid)) * s_mid
                            )
                            middle_scaled = torch.floor(middle_scaled).to(dtype=torch.long, device=middle_orig_positions.device)
                            middle_new_positions: torch.Tensor = torch.clamp(
                                target_low + middle_scaled, min=target_low, max=target_high_inclusive
                            )
                            if enforce_monotone_prefix and middle_new_positions.numel() > 0:
                                # Ensure non-decreasing sequence to avoid duplicates regression
                                # OPTIMIZATION: Use GPU-based cummax + clamp (functionally identical to Python loop)
                                cummax_values, _ = torch.cummax(middle_new_positions, dim=0)
                                middle_new_positions = torch.clamp(cummax_values, max=target_high_inclusive)
                            # Assign middle band
                            position_ids_k_per_head[0, head_idx, middle_indices_sorted] = middle_new_positions
                        # (No else branch: invalid target is fatal above)
                    # Optional: translate-pack K+Chunk after middle band at m_end (always considered inside two-band branch)
                    if pack_k_chunk_translation:
                        # m_end is the end of the compressed middle band in target space
                        m_end_for_pack = int(max(S, T_pref_inclusive - K))
                        # Δ shifts K tail and Chunk together without changing their internal distances
                        pack_delta = int((m_end_for_pack + K + 1) - min_query_position)
                        # Overwrite tail positions with translated originals (ensure overwrite even if set earlier)
                        if K > 0:
                            # OPTIMIZATION: Use tensor slicing - no CPU-GPU transfer
                            tail_indices_re: torch.Tensor = prefix_key_indices_sorted_tensor[-K:]
                            # OPTIMIZATION: Use cached tensor instead of repeated indexing
                            tail_positions_orig: torch.Tensor = position_ids_k_actual_0[tail_indices_re]
                            translated_tail: torch.Tensor = (tail_positions_orig + pack_delta).to(
                                dtype=torch.long, device=position_ids_k_per_head.device
                            )
                            position_ids_k_per_head[0, head_idx, tail_indices_re] = translated_tail
                        # Hard cap verification post-pack (per head)
                        # Estimate maximum new positions via computed ranges
                        assert (m_end_for_pack + K + seq_len_q) <= max_position_id_allowed, (
                            f"[FATAL] Post-pack max_new={m_end_for_pack + K + seq_len_q} exceeds cap {max_position_id_allowed}"
                        )
                elif use_proportional_prefix_scaling:
                    # Scale prefix into [0, target_prefix_max_inclusive] using ratio cap alpha.
                    # OPTIMIZATION: Reuse pre-computed boundary_inclusive (computed once outside loop, same for all heads)
                    # Ratio-capped target inclusive bound; use epsilon=0 here to avoid a
                    # gratuitous -1 shift when the prefix already fits
                    target_prefix_max_inclusive: int = int(prefix_scale_alpha * float(boundary_inclusive))
                    # Ensure non-negative
                    target_prefix_max_inclusive = max(0, target_prefix_max_inclusive)
                    # Convert to exclusive bound for helper
                    target_prefix_max_exclusive: int = target_prefix_max_inclusive + 1
                    # OPTIMIZATION: Use tensor directly - no CPU-GPU transfer
                    # OPTIMIZATION: Use cached tensor instead of repeated indexing
                    prefix_positions_tensor: torch.Tensor = position_ids_k_actual_0[prefix_key_indices_sorted_tensor]
                    new_prefix_positions: torch.Tensor = _scale_prefix_positions(
                        prefix_positions=prefix_positions_tensor,
                        target_max_exclusive=target_prefix_max_exclusive,
                        epsilon=0,
                    )
                else:
                    # Vectorized assignment: create tensor of new positions and assign all at once
                    new_prefix_positions: torch.Tensor = torch.arange(
                        num_prefix_keys, device=position_ids_k_per_head.device, dtype=torch.long
                    )
                # Only assign bulk new_prefix_positions for non two-band modes
                if not use_two_band_prefix_scaling:
                    # OPTIMIZATION: Use tensor indexing - no CPU-GPU transfer
                    position_ids_k_per_head[0, head_idx, prefix_key_indices_sorted_tensor] = new_prefix_positions

            # Get all current chunk keys (not just selected ones) - vectorized
            # OPTIMIZATION: Reuse pre-computed all_current_chunk_key_indices_sorted_tensor (computed once outside loop)
            # This is functionally identical - sorting is based on position_ids_k_actual_0, same for all heads
            
            # Start current chunk at max_prefix + 1 (or 0 if no prefix)
            current_chunk_start: int = max_prefix_position + 1 if max_prefix_position >= 0 else 0
            
            # Assign current chunk key positions
            if all_current_chunk_key_indices_sorted_tensor is not None and all_current_chunk_key_indices_sorted_tensor.numel() > 0:
                # Keep current chunk unchanged ONLY for pure proportional mode (not when two-band is active)
                if use_proportional_prefix_scaling and not use_two_band_prefix_scaling:
                    # Keep current chunk positions UNCHANGED (copy original positions)
                    # OPTIMIZATION: Use tensor indexing - no CPU-GPU transfer
                    # OPTIMIZATION: Use cached tensor instead of repeated indexing
                    original_current_chunk_positions: torch.Tensor = position_ids_k_actual_0[all_current_chunk_key_indices_sorted_tensor]
                    position_ids_k_per_head[0, head_idx, all_current_chunk_key_indices_sorted_tensor] = original_current_chunk_positions
                else:
                    if pack_k_chunk_translation and pack_delta is not None:
                        # Pure translation for current chunk keys: new = orig + Δ
                        # OPTIMIZATION: Use tensor indexing - no CPU-GPU transfer
                        # OPTIMIZATION: Use cached tensor instead of repeated indexing
                        original_current_chunk_positions: torch.Tensor = position_ids_k_actual_0[all_current_chunk_key_indices_sorted_tensor]
                        translated_chunk: torch.Tensor = (original_current_chunk_positions + pack_delta).to(
                            dtype=torch.long, device=position_ids_k_per_head.device
                        )
                        position_ids_k_per_head[0, head_idx, all_current_chunk_key_indices_sorted_tensor] = translated_chunk
                    else:
                        # Default: contiguous placement after prefix
                        # OPTIMIZATION: Use tensor numel - no CPU-GPU transfer
                        num_current_chunk_keys: int = all_current_chunk_key_indices_sorted_tensor.numel()
                        new_current_chunk_positions: torch.Tensor = torch.arange(
                            current_chunk_start, current_chunk_start + num_current_chunk_keys,
                            device=position_ids_k_per_head.device, dtype=torch.long
                        )
                        # OPTIMIZATION: Use tensor indexing - no CPU-GPU transfer
                        position_ids_k_per_head[0, head_idx, all_current_chunk_key_indices_sorted_tensor] = new_current_chunk_positions

            # Create position_id -> key_index mapping for queries (OPTIMIZED: GPU-based lookup)
            if all_current_chunk_key_indices_sorted_tensor is not None and all_current_chunk_key_indices_sorted_tensor.numel() > 0:
                # Batch extract new positions (keep on GPU)
                # OPTIMIZATION: Reuse pre-computed current_chunk_original_pos_tensor (computed once outside loop)
                # OPTIMIZATION: Use tensor indexing - no CPU-GPU transfer
                current_chunk_new_positions_tensor: torch.Tensor = position_ids_k_per_head[0, head_idx, all_current_chunk_key_indices_sorted_tensor]  # (num_current_chunk_keys,)
                
                # OPTIMIZATION: Use searchsorted + advanced indexing instead of Python dict lookup
                # current_chunk_original_pos_tensor is already sorted (pre-computed outside loop)
                # query_positions_tensor is pre-computed outside loop (same for all heads)
                
                # Find insertion points (searchsorted returns insertion point for exact matches)
                lookup_indices: torch.Tensor = torch.searchsorted(
                    current_chunk_original_pos_tensor, 
                    query_positions_tensor, 
                    right=False
                )
                
                # Verify exact matches (same assertion behavior as original code)
                # Check bounds first to avoid out-of-bounds access
                valid_indices: torch.Tensor = lookup_indices < current_chunk_original_pos_tensor.shape[0]
                if not valid_indices.all():
                    # Some queries are out of bounds (position too large)
                    failed_mask: torch.Tensor = ~valid_indices
                    failed_query_positions: List[int] = query_positions_tensor[failed_mask].cpu().tolist()
                    max_available: int = int(current_chunk_original_pos_tensor.max().item())
                    assert False, (
                        f"[ERROR] Head {head_idx}: Query positions {failed_query_positions} exceed available key positions. "
                        f"Max available position: {max_available}. "
                        f"This indicates a mismatch between query and key positions."
                    )
                
                # Verify exact matches at the found indices
                exact_matches: torch.Tensor = current_chunk_original_pos_tensor[lookup_indices] == query_positions_tensor
                
                if not exact_matches.all():
                    # Find which queries failed to match (for better error message)
                    failed_mask: torch.Tensor = ~exact_matches
                    failed_query_positions: List[int] = query_positions_tensor[failed_mask].cpu().tolist()
                    available_positions: List[int] = current_chunk_original_pos_tensor.cpu().tolist()[:10]
                    assert False, (
                        f"[ERROR] Head {head_idx}: Query positions {failed_query_positions} have no matching key positions. "
                        f"Available key positions: {available_positions}{'...' if len(available_positions) >= 10 else ''}. "
                        f"This indicates a mismatch between query and key positions."
                    )
                
                # Use advanced indexing to get new positions (all on GPU, no CPU-GPU transfers)
                query_new_positions: torch.Tensor = current_chunk_new_positions_tensor[lookup_indices]
                position_ids_q_per_head[0, head_idx, :] = query_new_positions
            else:
                # No current chunk keys - queries should also be empty, but handle gracefully
                # query_positions_tensor is pre-computed outside loop (same for all heads)
                if query_positions_tensor.numel() > 0:
                    assert False, (
                        f"[ERROR] Head {head_idx}: Queries exist but no current chunk keys found. "
                        f"Query positions: {query_positions_tensor.cpu().tolist()}"
                    )
                # If no queries, nothing to assign (position_ids_q_per_head already initialized to zeros)
        
        per_head_elapsed: float = time.time() - per_head_start_time
        layer_idx: Optional[int] = kwargs.get("layer_idx", None)
        if os.environ.get("SPARSE_DEBUG"):
            print(f"[reposition] layer={layer_idx} per_head_reassign elapsed={per_head_elapsed:.3f}s ({num_heads} heads)", flush=True)
        
        # Timing: rotary_emb calls
        rotary_start_time: float = time.time()
        
        # Compute cos/sin per head with modified positions
        num_kv_heads: int = keys.shape[1]  # GQA: keys may have fewer heads
        head_ratio: int = num_heads // num_kv_heads
        
        cos_q_per_head_list: List[torch.Tensor] = []
        sin_q_per_head_list: List[torch.Tensor] = []
        cos_k_per_head_list: List[torch.Tensor] = []
        sin_k_per_head_list: List[torch.Tensor] = []
        
        # Compute cos/sin for all query heads
        # Use same dtype as queries to avoid dtype mismatch (bfloat16 vs float32)
        queries_dtype: torch.dtype = queries.dtype
        # OPTIMIZATION: Create dummy tensors once and reuse (rotary_emb doesn't modify them)
        dummy_x_q_all: torch.Tensor = torch.zeros(
            batch_size, seq_len_q, device=queries.device, dtype=queries_dtype
        )
        # OPTIMIZATION: Pre-allocate output tensors instead of building list + stacking
        # Get head_dim from first rotary_emb call
        pos_ids_q_first: torch.Tensor = position_ids_q_per_head[:, 0, :]  # (batch, seq_len_q)
        cos_q_first, sin_q_first = rotary_emb(dummy_x_q_all, pos_ids_q_first)
        head_dim_q: int = cos_q_first.shape[-1]  # Get head_dim from output shape
        
        # Pre-allocate output tensors
        cos_q_mod: torch.Tensor = torch.zeros(
            batch_size, num_heads, seq_len_q, head_dim_q,
            device=queries.device, dtype=queries_dtype
        )
        sin_q_mod: torch.Tensor = torch.zeros(
            batch_size, num_heads, seq_len_q, head_dim_q,
            device=queries.device, dtype=queries_dtype
        )
        
        # Assign first head (already computed)
        cos_q_mod[:, 0, :, :] = cos_q_first
        sin_q_mod[:, 0, :, :] = sin_q_first
        
        # Compute remaining query heads
        for head_idx in range(1, num_heads):
            pos_ids_q_head: torch.Tensor = position_ids_q_per_head[:, head_idx, :]  # (batch, seq_len_q)
            # OPTIMIZATION: Reuse dummy tensor (functionally identical - rotary_emb doesn't modify input)
            cos_q_head, sin_q_head = rotary_emb(dummy_x_q_all, pos_ids_q_head)
            # OPTIMIZATION: Direct assignment instead of list append
            cos_q_mod[:, head_idx, :, :] = cos_q_head
            sin_q_mod[:, head_idx, :, :] = sin_q_head
        
        # Compute cos/sin for key heads only (GQA)
        # Use same dtype as keys to avoid dtype mismatch (bfloat16 vs float32)
        keys_dtype: torch.dtype = keys.dtype
        # OPTIMIZATION: Create dummy tensor once and reuse (rotary_emb doesn't modify it)
        dummy_x_k_all: torch.Tensor = torch.zeros(
            batch_size, seq_len_k, device=keys.device, dtype=keys_dtype
        )
        # OPTIMIZATION: Pre-allocate output tensors instead of building list + stacking
        # Get head_dim from first rotary_emb call
        pos_ids_k_first: torch.Tensor = position_ids_k_per_head[:, 0, :]  # (batch, seq_len_k)
        cos_k_first, sin_k_first = rotary_emb(dummy_x_k_all, pos_ids_k_first)
        head_dim_k: int = cos_k_first.shape[-1]  # Get head_dim from output shape
        
        # Pre-allocate output tensors
        cos_k_mod: torch.Tensor = torch.zeros(
            batch_size, num_kv_heads, seq_len_k, head_dim_k,
            device=keys.device, dtype=keys_dtype
        )
        sin_k_mod: torch.Tensor = torch.zeros(
            batch_size, num_kv_heads, seq_len_k, head_dim_k,
            device=keys.device, dtype=keys_dtype
        )
        
        # Assign first key head (already computed)
        cos_k_mod[:, 0, :, :] = cos_k_first
        sin_k_mod[:, 0, :, :] = sin_k_first
        
        # Compute remaining key heads
        for kv_head_idx in range(1, num_kv_heads):
            query_head_idx: int = kv_head_idx * head_ratio
            pos_ids_k_head: torch.Tensor = position_ids_k_per_head[:, query_head_idx, :]  # (batch, seq_len_k)
            # OPTIMIZATION: Reuse dummy tensor (functionally identical - rotary_emb doesn't modify input)
            cos_k_head, sin_k_head = rotary_emb(dummy_x_k_all, pos_ids_k_head)
            # OPTIMIZATION: Direct assignment instead of list append
            cos_k_mod[:, kv_head_idx, :, :] = cos_k_head
            sin_k_mod[:, kv_head_idx, :, :] = sin_k_head
        
        rotary_elapsed: float = time.time() - rotary_start_time
        if os.environ.get("SPARSE_DEBUG"):
            print(f"[reposition] layer={layer_idx} rotary_emb elapsed={rotary_elapsed:.3f}s ({num_heads}Q+{num_kv_heads}K heads)", flush=True)
        
        # CRITICAL VERIFICATION: Ensure position reassignment actually happened
        # Check that position_ids_q_per_head and position_ids_k_per_head are NOT all zeros
        # (which would indicate they were never assigned and we're silently using original positions)
        q_positions_sum = position_ids_q_per_head.sum().item()
        k_positions_sum = position_ids_k_per_head.sum().item()
        assert q_positions_sum > 0, (
            f"[CRITICAL ERROR] position_ids_q_per_head is all zeros! "
            f"This means position reassignment failed silently. Sum={q_positions_sum}"
        )
        assert k_positions_sum > 0, (
            f"[CRITICAL ERROR] position_ids_k_per_head is all zeros! "
            f"This means position reassignment failed silently. Sum={k_positions_sum}"
        )
        
        # DEBUG: Log actual position ranges to verify repositioning
        if os.environ.get("SPARSE_DEBUG_POSITIONS", "0").lower() in ("1", "true", "yes"):
            q_max = position_ids_q_per_head.max().item()
            q_min = position_ids_q_per_head.min().item()
            k_max = position_ids_k_per_head.max().item()
            k_min = position_ids_k_per_head.min().item()
            print(f"[DEBUG POSITIONS] layer={layer_idx} Q range: [{q_min}, {q_max}], K range: [{k_min}, {k_max}], max_allowed={max_position_id_allowed}", flush=True)
        
        # Verify that reassigned positions were actually computed (not left as zeros)
        # Check that at least some positions are non-zero (unless legitimately all zeros for first prefill)
        # For keys: if there are any keys, at least some should have non-zero positions
        if seq_len_k > 0:
            k_positions_max = position_ids_k_per_head.max().item()
            # If all positions are 0, that's suspicious (unless it's a very edge case)
            # But we already checked sum > 0, so this is just an extra sanity check
            assert k_positions_max >= 0, (
                f"[CRITICAL ERROR] All key positions are negative or invalid! "
                f"Max position={k_positions_max}"
            )
        
        # Verify that we're using the reassigned positions, not original ones
        # The reassigned positions should be used in cos/sin computation above
        # If position_ids_q_per_head or position_ids_k_per_head were all zeros,
        # the RoPE computation would produce incorrect results
        # We've already verified they're not all zeros above
        
        # Apply RoPE with modified positions
        assert unroped_queries is not None, "[ERROR] unroped_queries is None"
        assert unroped_keys is not None, "[ERROR] unroped_keys is None"
        assert cos_q_mod is not None, "[ERROR] cos_q_mod is None"
        assert sin_q_mod is not None, "[ERROR] sin_q_mod is None"
        assert cos_k_mod is not None, "[ERROR] cos_k_mod is None"
        assert sin_k_mod is not None, "[ERROR] sin_k_mod is None"
        
        reroped_queries = (unroped_queries * cos_q_mod) + (rotate_half(unroped_queries) * sin_q_mod)
        reroped_keys = (unroped_keys * cos_k_mod) + (rotate_half(unroped_keys) * sin_k_mod)
        
        # Verify reroped tensors are valid
        assert reroped_queries.shape == queries.shape, f"[ERROR] reroped_queries shape mismatch: {reroped_queries.shape} != {queries.shape}"
        assert reroped_keys.shape == keys.shape, f"[ERROR] reroped_keys shape mismatch: {reroped_keys.shape} != {keys.shape}"
        assert not torch.isnan(reroped_queries).any(), "[ERROR] reroped_queries contains NaN"
        assert not torch.isnan(reroped_keys).any(), "[ERROR] reroped_keys contains NaN"
        assert not torch.isinf(reroped_queries).any(), "[ERROR] reroped_queries contains Inf"
        assert not torch.isinf(reroped_keys).any(), "[ERROR] reroped_keys contains Inf"
        
        # Verify reroped matches original (will be different due to position reassignment)
        reroped_q_diff = torch.abs(queries - reroped_queries).max()
        reroped_k_diff = torch.abs(keys - reroped_keys).max()
        
        # Import _compute_masked_exp_attention_weights from parent module
        # We need to call this function to recompute attention weights with reroped Q/K
        # Use absolute import to avoid circular dependency
        from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import _compute_masked_exp_attention_weights
        
        # Compute attention weights with re-roped Q/K
        exp_attention_weights_reroped: torch.Tensor = _compute_masked_exp_attention_weights(
            queries=reroped_queries,
            keys=reroped_keys,
            attention_mask=attention_mask,
            scaling=scaling,
            sparse_attention_mask=sparse_attention_mask,
            dropout=dropout,
            training=training,
        )
        
        # CRITICAL VERIFICATION: Ensure we're using reroped weights, not original
        assert exp_attention_weights_reroped is not None, (
            "[CRITICAL ERROR] exp_attention_weights_reroped is None! "
            "This means position reassignment failed and we would silently use original weights."
        )
        assert exp_attention_weights_reroped.shape == exp_attention_weights.shape, (
            f"[CRITICAL ERROR] Shape mismatch: reroped={exp_attention_weights_reroped.shape}, "
            f"original={exp_attention_weights.shape}"
        )
        # Verify reroped weights are actually different (or at least computed)
        # They should be different due to position reassignment
        weights_are_different = not torch.allclose(
            exp_attention_weights, exp_attention_weights_reroped, atol=1e-6, rtol=1e-6
        )
        assert weights_are_different, (
            "[CRITICAL ERROR] Reroped weights are identical to original! "
            "This suggests position reassignment had no effect or failed silently."
        )
        exp_attention_weights = exp_attention_weights_reroped.to(exp_attention_weights.dtype)
        
        reposition_total_elapsed: float = time.time() - reposition_start_time
        if os.environ.get("SPARSE_DEBUG"):
            print(f"[reposition] layer={layer_idx} TOTAL elapsed={reposition_total_elapsed:.3f}s", flush=True)
    except Exception as e:
        assert False, f"  [Unroped/Re-roped] Exception: {e}"
    
    return exp_attention_weights

