"""Utility functions for masked attention computation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .kv_utils import _get_num_key_value_groups, repeat_kv
from .mask import Mask

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
DEFAULT_MAX_POSITION_ID: int = 8192


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


def get_true_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Get the true (dense) attention output from the module.

    Args:
        module: The attention module (used for dropout training flag).
        queries: Query tensor of shape (..., seq_len_q, d_k).
        keys: Key tensor of shape (..., seq_len_k, d_k).
        values: Value tensor of shape (..., seq_len_k, d_v).
        attention_mask: Optional mask tensor to apply to attention weights.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability for attention weights.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple containing:
            - attention_output: Output tensor after applying attention.
            - attention_weights: Softmax-normalized attention weights.
    """
    num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
    key_states = repeat_kv(keys, num_key_value_groups)
    value_states = repeat_kv(values, num_key_value_groups)

    attn_weights = torch.matmul(queries, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        queries.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_inv_mask_sum(input_tensor: torch.Tensor, mask: Mask) -> torch.Tensor:
    """Apply inverse mask to input tensor and sum along the last dimension.

    This function efficiently computes the sum of applying the inverse mask to an input tensor
    using sparse representation and scatter operations, avoiding the need to create dense tensors.

    Args:
        input_tensor: Input tensor of shape (..., n) where n is the last dimension
        mask: Mask object to apply inverse mask with

    Returns:
        Sum tensor of shape (..., 1) with the last dimension reduced

    Note:
        - For full masks: returns sum of all input values
        - For empty masks: returns sum of all input values since empty mask means default case when no mask is built.
        - For sparse masks: efficiently computes sum using sparse operations
    """
    if input_tensor.shape != mask.shape:
        raise ValueError(
            f"input_tensor.shape must be {mask.shape}, got {input_tensor.shape}"
        )

    # Handle special cases
    if mask.is_full or mask.is_empty:
        # Full mask: sum all input values
        return input_tensor.sum(dim=-1, keepdim=True)
    result = mask.apply_inv_mask(input_tensor)
    return result.sum(dim=-1, keepdim=True)


# TODO(is there a better dense version of this function?)
def create_sampling_mask_with_per_head_budget(
    budgets: torch.Tensor,
    sampling_probability: torch.Tensor,
    seq_len_keys: int,
    start_idx: int,
    end_idx: int,
    dtype: torch.dtype = torch.float32,
) -> Mask:
    """Create a sampling mask with per-head budget using direct sparse construction.

    This function efficiently creates a sparse sampling mask by directly constructing
    the sparse representation without creating intermediate dense tensors.

    Args:
        budgets: Budget tensor of shape (b, h, q, 1) indicating how many elements to sample per row
        sampling_probability: Sampling probability tensor of shape (b, h, q, 1)
        seq_len_keys: Length of the key sequence dimension
        start_idx: Starting index for sampling range (inclusive)
        end_idx: Ending index for sampling range (exclusive)
        dtype: Data type for the mask

    Returns:
        Mask object with sparse sampling representation

    Note:
        - Uses direct sparse construction for memory efficiency
        - Generates random indices within [start_idx, end_idx) for each element
        - Creates proper ptr array for sparse representation
        - Assigns sampling probabilities as mask data values

    Important Note:
        - we use random sampling with replacement so the sampling probabilities might lead to be incorrect
    """
    batch_size, num_heads, seq_len_queries, _ = budgets.shape

    # Reshape budget to (num_rows,) for easier processing
    num_rows = batch_size * num_heads * seq_len_queries
    budgets_flat = budgets.view(num_rows)  # (num_rows,)

    # Calculate total number of elements to sample
    total_elements = int(budgets_flat.sum().item())

    # Create ptr array using cumulative sum of budgets
    ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=budgets.device),
            torch.cumsum(budgets_flat, dim=0),
        ]
    )  # (num_rows + 1,)

    # Generate random indices within sampling range for each element
    idx_in_row = torch.randint(
        low=start_idx,
        high=end_idx,
        size=(total_elements,),
        device=budgets.device,
        dtype=torch.long,
    )  # (total_elements,)

    # Create row indices by repeating each row index according to its budget
    positions = torch.arange(total_elements, device=budgets.device) + 1
    row_id = torch.searchsorted(ptr, positions, right=False) - 1  # (total_elements,)

    # Calculate global indices
    idx_global = idx_in_row + row_id * seq_len_keys  # (total_elements,)

    # Get sampling probabilities for each element
    sampling_prob_flat = sampling_probability.view(num_rows)  # (num_rows,)
    data_global = sampling_prob_flat[row_id]  # (total_elements,)

    # Create the sampling mask directly using sparse index construction
    sampling_mask = Mask.create_mask_from_indices(
        shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
        indices=idx_global,
        ptr=ptr,
        data=data_global,
        dtype=dtype,
    )

    return sampling_mask


def _compute_masked_exp_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    sparse_attention_mask: Mask,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Compute masked attention weights (common logic for numerator and denominator).

    Args:
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        sparse_attention_mask: Mask object for sparse attention
        dropout: Dropout probability
        training: Whether the model is in training mode

    Returns:
        Masked exponential attention weights tensor of shape (b, h, sq, sk)
    """
    # Calculate num_key_value_groups from tensor shapes
    num_key_value_groups: int = _get_num_key_value_groups(queries, keys)

    # Apply key-value grouping if needed
    key_states: torch.Tensor = repeat_kv(keys, num_key_value_groups)

    raw_attention_weights: torch.Tensor = (
        torch.matmul(queries, key_states.transpose(2, 3)) * scaling
    )

    if attention_mask is not None:
        raw_attention_weights = (
            raw_attention_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        )

    row_wise_max: torch.Tensor = torch.max(raw_attention_weights, dim=-1, keepdim=True)[
        0
    ]
    raw_attention_weights = raw_attention_weights - row_wise_max
    exp_attention_weights: torch.Tensor = torch.exp(raw_attention_weights)

    exp_attention_weights = sparse_attention_mask.apply_inv_mask(exp_attention_weights)

    # Apply dropout to attention weights if specified
    if dropout > 0.0 and training:
        exp_attention_weights = torch.nn.functional.dropout(
            exp_attention_weights, p=dropout, training=training
        )

    return exp_attention_weights


def _get_attention_denominator(exp_attention_weights: torch.Tensor) -> torch.Tensor:
    """Get attention denominator from pre-computed exponential attention weights.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    return torch.sum(exp_attention_weights, dim=-1, keepdim=True)


def _get_attention_numerator(
    exp_attention_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Get attention numerator from pre-computed exponential attention weights and prepared values.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)
        value_states: Prepared value tensor of shape (b, h, sq, d) - already grouped if needed

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    return torch.matmul(exp_attention_weights, value_states)


def get_attention_denominator(
    module: Optional[nn.Module],
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    return _get_attention_denominator(exp_attention_weights)


def get_attention_numerator(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention numerator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)

    return _get_attention_numerator(exp_attention_weights, value_states)


# GET MASKED ATTENTION OUTPUT


def get_masked_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    return_attention_weights: bool = False,
    sparse_meta_data: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Get masked attention output by dividing numerator by denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        return_attention_weights: Whether to return attention weights along with output
        **kwargs: Additional keyword arguments

    Returns:
        If return_attention_weights is False:
            Attention output tensor of shape (b, h, sq, d)
        If return_attention_weights is True:
            Tuple of (attention_output, attention_weights) where:
            - attention_output: tensor of shape (b, h, sq, d)
            - attention_weights: tensor of shape (b, h, sq, sk)
    """
    # Compute exponential attention weights once and reuse
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
    )
    
    
     # if sparse_attention_mask.get_density() < 0.3: import pdb; pdb.set_trace()
     
     # Print attention pattern for sample queries and heads
    # Commented out mask analysis section for performance (redundant with position reassignment logic below)
    # if sparse_attention_mask.get_density() < 0.3:
    #     # Analyze which key positions are preserved (not dropped) in the mask
    #     dense_mask = sparse_attention_mask.get_dense_mask()  # [batch, num_heads, seq_len_q, seq_len_k]
    #     num_heads = dense_mask.shape[1]
    #     seq_len_q = dense_mask.shape[2]
    #     seq_len_k = dense_mask.shape[3]
    #     layer_idx = kwargs.get("layer_idx", "?")
    #     
    #     # Per-head: union and intersection of keys across queries
    #     union_keys_per_head = []  # Keys attended by ANY query
    #     intersection_keys_per_head = []  # Keys attended by ALL queries
    #     for head_idx in range(num_heads):
    #         print(f"Head {head_idx}:")
    #         # Get keys for each query
    #         keys_per_query = []
    #         for q_idx in range(seq_len_q):
    #             active_keys = set(torch.nonzero(dense_mask[0, head_idx, q_idx] > 0).squeeze(-1).cpu().tolist())
    #             keys_per_query.append(active_keys)
    #         
    #         # Union: keys attended by ANY query
    #         union_keys = set()
    #         for q_keys in keys_per_query:
    #             union_keys |= q_keys
    #         union_keys_per_head.append(union_keys)
    #         
    #         # Intersection: keys attended by ALL queries
    #         if keys_per_query:
    #             intersection_keys = keys_per_query[0].copy()
    #             for q_keys in keys_per_query[1:]:
    #                 intersection_keys &= q_keys
    #         else:
    #             intersection_keys = set()
    #         intersection_keys_per_head.append(intersection_keys)
    #     
    #     # Print clean statistics
    #     print(f"\n[Mask Analysis] Layer {layer_idx}:")
    #     print(f"  Total queries: {seq_len_q}, Total key positions: {seq_len_k}")
    #     for head_idx in range(num_heads):
    #         union_count = len(union_keys_per_head[head_idx])
    #         intersection_count = len(intersection_keys_per_head[head_idx])
    #         union_pct = (union_count / seq_len_k) * 100
    #         intersection_pct = (intersection_count / seq_len_k) * 100 if seq_len_k > 0 else 0
    #         print(f"  Head {head_idx}: Union={union_count}/{seq_len_k} ({union_pct:.1f}%), Intersection={intersection_count}/{seq_len_k} ({intersection_pct:.1f}%)")
    #     
        # # Detailed analysis for sample head (head 0)
        # sample_head = 0
        # keys_per_query = []
        # for q_idx in range(seq_len_q):
        #     active_keys = set(torch.nonzero(dense_mask[0, sample_head, q_idx] > 0).squeeze(-1).cpu().tolist())
        #     keys_per_query.append(active_keys)
        
        # # Count how many queries share each key
        # key_to_query_count = {}
        # for q_idx, q_keys in enumerate(keys_per_query):
        #     for k in q_keys:
        #         key_to_query_count[k] = key_to_query_count.get(k, 0) + 1
        
        # # Statistics on key sharing
        # sharing_counts = {}
        # for count in key_to_query_count.values():
        #     sharing_counts[count] = sharing_counts.get(count, 0) + 1
        
        # print(f"\n  [Key Sharing Analysis] Head {sample_head}:")
        # print(f"    Keys attended by 1 query: {sharing_counts.get(1, 0)}")
        # print(f"    Keys attended by 2-5 queries: {sum(sharing_counts.get(i, 0) for i in range(2, 6))}")
        # print(f"    Keys attended by 6-10 queries: {sum(sharing_counts.get(i, 0) for i in range(6, 11))}")
        # print(f"    Keys attended by >10 queries: {sum(sharing_counts.get(i, 0) for i in range(11, seq_len_q+1))}")
        # print(f"    Keys attended by ALL queries: {len(intersection_keys_per_head[sample_head])}")
        # print()
        
    import random
    from sparse_attention_hub.sparse_attention.research_attention.rope_utils import (
        unapply_rotary_pos_emb_queries,
        unapply_rotary_pos_emb_keys,
        apply_rotary_pos_emb,
    )
    
    dense_mask = sparse_attention_mask.get_dense_mask()
    num_queries = queries.shape[2]
    num_heads = queries.shape[1]
    seq_len_keys = keys.shape[2]
    sample_queries = sorted(random.sample(range(min(num_queries, 256)), min(3, num_queries)))
    sample_heads = [0, min(1, num_heads - 1)]
    
    # Get cos/sin for unroping
    position_ids_q = kwargs.get("position_ids")  # Query positions
    position_ids_k = torch.arange(0, seq_len_keys, device=keys.device, dtype=torch.long).unsqueeze(0)  # Key positions [0, seq_len_keys-1]
    
    # Try to get rotary_emb from sparse_meta_data (preferred) or module
    rotary_emb = None
    if sparse_meta_data is not None:
        rotary_emb = sparse_meta_data.get("_rotary_emb")
    if rotary_emb is None:
        if hasattr(module, "rotary_emb"):
            rotary_emb = module.rotary_emb
        elif hasattr(module, "model") and hasattr(module.model, "rotary_emb"):
            rotary_emb = module.model.rotary_emb
    
    unroped_queries = queries
    unroped_keys = keys
    # Check if position reassignment is enabled (default: False)
    # When disabled, unroping for mask computation can still happen via EXTEND_CONTEXT in base.py
    # but repositioning/re-roping is skipped here
    enable_position_reassignment: bool = os.environ.get("ENABLE_POSITION_REASSIGNMENT", "0").lower() in ("1", "true", "yes")
    # if (rotary_emb is not None and position_ids_q is not None) and enable_position_reassignment:
    
    if (sparse_attention_mask.get_density() < 1.0) and (rotary_emb is not None and position_ids_q is not None) and enable_position_reassignment:
        # import pdb; pdb.set_trace()
        reposition_start_time: float = time.time()
        try:
            # Compute cos/sin for queries
            # Current: Same position_ids for all heads (will be broadcasted in unapply_rotary_pos_emb_queries)
            # Future: Can extend to per-head position_ids: position_ids_q shape (batch, num_heads, seq_len)
            # Use same dtype as queries to avoid dtype mismatch (bfloat16 vs float32)
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
            from sparse_attention_hub.sparse_attention.research_attention.rope_utils import rotate_half
            
            # Get sequence lengths
            seq_len_q = position_ids_q.shape[1]
            seq_len_k = keys.shape[2]  # (batch, num_kv_heads, seq_len_k, head_dim)
            
            # ====================================================================
            # Per-Head Position IDs: Reassign Contiguously Based on Selected Keys
            # ====================================================================
            # Strategy:
            # 1. Prefix keys (selected per head): Sort by original position, reassign 0,1,2,...
            # 2. Current chunk Q/K: All tokens get contiguous IDs starting from prefix_end
            #    Q and K for same token get same position ID
            # ====================================================================
            # COMMENTED OUT: Position reassignment disabled for testing
            # num_heads: int = queries.shape[1]
            # batch_size: int = queries.shape[0]
            # 
            # # Get actual position IDs for keys (if not provided, infer from queries)
            # position_ids_k_actual: torch.Tensor = kwargs.get("position_ids_k", None)
            # if position_ids_k_actual is None:
            #     # Infer: if queries start at X, keys likely go from 0 to X+seq_len_q-1
            #     min_q_pos: int = position_ids_q[0, 0].item()
            #     position_ids_k_actual = torch.arange(
            #         min_q_pos - seq_len_k + seq_len_q, min_q_pos + seq_len_q,
            #         device=keys.device, dtype=torch.long
            #     ).unsqueeze(0)  # (batch, seq_len_k)
            # 
            # # Identify prefix vs current chunk boundary
            # # Keys with position < min_query_position are prefix, others are current chunk
            # min_query_position: int = position_ids_q[0, 0].item()
            # max_query_position: int = position_ids_q[0, -1].item()
            # 
            # # Get dense mask to identify selected keys per head
            # dense_mask = sparse_attention_mask.get_dense_mask()  # [batch, num_heads, seq_len_q, seq_len_k]
            # 
            # # Initialize per-head position ID tensors
            # position_ids_q_per_head: torch.Tensor = torch.zeros(
            #     batch_size, num_heads, seq_len_q, device=queries.device, dtype=torch.long
            # )
            # position_ids_k_per_head: torch.Tensor = torch.zeros(
            #     batch_size, num_heads, seq_len_k, device=keys.device, dtype=torch.long
            # )
            # 
            # print(f"\n[Per-Head Position Reassignment]")
            # print(f"  Query positions: {min_query_position} to {max_query_position} ({seq_len_q} tokens)")
            # print(f"  Key positions: {position_ids_k_actual[0, 0].item()} to {position_ids_k_actual[0, -1].item()} ({seq_len_k} tokens)")
            # print(f"  Prefix boundary: keys with position < {min_query_position} are prefix")
            # 
            # # Process each head independently
            # for head_idx in range(num_heads):
            #     # Get union of selected key indices for this head
            #     # Optimization: if only one query, skip union computation
            #     if seq_len_q == 1:
            #         active_key_indices = torch.nonzero(dense_mask[0, head_idx, 0] > 0).squeeze(-1).cpu().tolist()
            #         union_key_indices: set = set(active_key_indices)
            #     else:
            #         union_key_indices: set = set()
            #         for q_idx in range(seq_len_q):
            #             active_key_indices = torch.nonzero(dense_mask[0, head_idx, q_idx] > 0).squeeze(-1).cpu().tolist()
            #             union_key_indices.update(active_key_indices)
            #     
            #     # Separate prefix vs current chunk key indices
            #     prefix_key_indices: List[int] = []
            #     current_chunk_key_indices: List[int] = []
            #     
            #     for k_idx in union_key_indices:
            #         k_pos: int = position_ids_k_actual[0, k_idx].item()
            #         if k_pos < min_query_position:
            #             prefix_key_indices.append(k_idx)
            #         else:
            #             current_chunk_key_indices.append(k_idx)
            #     
            #     # Sort prefix keys by their original position IDs
            #     prefix_key_indices_sorted: List[int] = sorted(
            #         prefix_key_indices,
            #         key=lambda idx: position_ids_k_actual[0, idx].item()
            #     )
            #     
            #     # Reassign prefix keys: 0, 1, 2, ...
            #     num_prefix_keys: int = len(prefix_key_indices_sorted)
            #     for new_pos, k_idx in enumerate(prefix_key_indices_sorted):
            #         position_ids_k_per_head[0, head_idx, k_idx] = new_pos
            #     
            #     # Reassign current chunk keys: ALL keys in current chunk get contiguous IDs
            #     # starting from num_prefix_keys, in order of their original positions
            #     # (Not just selected ones - all 4096 tokens in current chunk)
            #     all_current_chunk_key_indices: List[int] = []
            #     for k_idx in range(seq_len_k):
            #         k_pos: int = position_ids_k_actual[0, k_idx].item()
            #         if k_pos >= min_query_position:
            #             all_current_chunk_key_indices.append(k_idx)
            #     
            #     all_current_chunk_key_indices_sorted: List[int] = sorted(
            #         all_current_chunk_key_indices,
            #         key=lambda idx: position_ids_k_actual[0, idx].item()
            #     )
            #     
            #     # Assign contiguous IDs to all current chunk keys
            #     for offset, k_idx in enumerate(all_current_chunk_key_indices_sorted):
            #         position_ids_k_per_head[0, head_idx, k_idx] = num_prefix_keys + offset
            #     
            #     # Create position_id -> key_index mapping for O(1) lookup (optimization)
            #     pos_to_k_idx: Dict[int, int] = {
            #         position_ids_k_actual[0, k_idx].item(): k_idx
            #         for k_idx in all_current_chunk_key_indices_sorted
            #     }
            #     
            #     # For queries: All queries get contiguous IDs starting from num_prefix_keys
            #     # Q and K for same token get same position ID
            #     for q_idx in range(seq_len_q):
            #         q_pos: int = position_ids_q[0, q_idx].item()
            #         # Find corresponding key index using dictionary lookup (O(1) instead of O(seq_len_k))
            #         k_idx_for_q: Optional[int] = pos_to_k_idx.get(q_pos)
            #         
            #         if k_idx_for_q is not None:
            #             # Use same position ID as the corresponding key
            #             position_ids_q_per_head[0, head_idx, q_idx] = position_ids_k_per_head[0, head_idx, k_idx_for_q]
            #         else:
            #             # Fallback: assign sequentially based on query index
            #             position_ids_q_per_head[0, head_idx, q_idx] = num_prefix_keys + q_idx
            #     
            #     # Print summary for this head
            #     prefix_positions_original = [position_ids_k_actual[0, idx].item() for idx in prefix_key_indices_sorted]
            #     num_current_chunk_keys: int = len(all_current_chunk_key_indices_sorted)
            #     print(f"  Head {head_idx}:")
            #     if num_prefix_keys > 0:
            #         print(f"    Prefix keys: {num_prefix_keys} selected, original positions {prefix_positions_original[:5]}{'...' if len(prefix_positions_original) > 5 else ''} → new positions [0, ..., {num_prefix_keys-1}]")
            #     else:
            #         print(f"    Prefix keys: 0 selected (no prefix)")
            #     print(f"    Current chunk: {num_current_chunk_keys} keys total, {len(current_chunk_key_indices)} selected, Q/K positions [{num_prefix_keys}, ..., {num_prefix_keys + num_current_chunk_keys - 1}]")
            # 
            # # ====================================================================
            # # Compute cos/sin per head
            # # ====================================================================
            # # Generate RoPE embeddings for each head independently
            # # This allows future customization where each head has different position encodings
            # # Handle GQA: queries have num_heads, keys have num_kv_heads
            # # ====================================================================
            # num_kv_heads: int = keys.shape[1]  # GQA: keys may have fewer heads
            # head_ratio: int = num_heads // num_kv_heads  # e.g., 32 // 8 = 4
            # 
            # cos_q_per_head_list: List[torch.Tensor] = []
            # sin_q_per_head_list: List[torch.Tensor] = []
            # cos_k_per_head_list: List[torch.Tensor] = []
            # sin_k_per_head_list: List[torch.Tensor] = []
            # 
            # # Compute cos/sin for all query heads
            # for head_idx in range(num_heads):
            #     # Query cos/sin for this head
            #     pos_ids_q_head: torch.Tensor = position_ids_q_per_head[:, head_idx, :]  # (batch, seq_len_q)
            #     dummy_x_q_head: torch.Tensor = torch.zeros(
            #         batch_size, seq_len_q, device=queries.device, dtype=torch.float32
            #     )
            #     cos_q_head, sin_q_head = rotary_emb(dummy_x_q_head, pos_ids_q_head)  # (batch, seq_len_q, head_dim)
            #     cos_q_per_head_list.append(cos_q_head)
            #     sin_q_per_head_list.append(sin_q_head)
            # 
            # # Compute cos/sin for key heads only (GQA: fewer key heads)
            # for kv_head_idx in range(num_kv_heads):
            #     # Map to corresponding query head (use first query head in the group)
            #     query_head_idx: int = kv_head_idx * head_ratio
            #     pos_ids_k_head: torch.Tensor = position_ids_k_per_head[:, query_head_idx, :]  # (batch, seq_len_k)
            #     dummy_x_k_head: torch.Tensor = torch.zeros(
            #         batch_size, seq_len_k, device=keys.device, dtype=torch.float32
            #     )
            #     cos_k_head, sin_k_head = rotary_emb(dummy_x_k_head, pos_ids_k_head)  # (batch, seq_len_k, head_dim)
            #     cos_k_per_head_list.append(cos_k_head)
            #     sin_k_per_head_list.append(sin_k_head)
            # 
            # # Stack to get (batch, num_heads, seq_len, head_dim) shape for queries
            # cos_q_mod: torch.Tensor = torch.stack(cos_q_per_head_list, dim=1)  # (batch, num_heads, seq_len_q, head_dim)
            # sin_q_mod: torch.Tensor = torch.stack(sin_q_per_head_list, dim=1)  # (batch, num_heads, seq_len_q, head_dim)
            # # Stack to get (batch, num_kv_heads, seq_len, head_dim) shape for keys
            # cos_k_mod: torch.Tensor = torch.stack(cos_k_per_head_list, dim=1)  # (batch, num_kv_heads, seq_len_k, head_dim)
            # sin_k_mod: torch.Tensor = torch.stack(sin_k_per_head_list, dim=1)  # (batch, num_kv_heads, seq_len_k, head_dim)
            # # Apply RoPE to queries: q_rot = q * cos_q + rotate_half(q) * sin_q (using modified cos/sin)
            # reroped_queries = (unroped_queries * cos_q_mod) + (rotate_half(unroped_queries) * sin_q_mod)
            # # Apply RoPE to keys: k_rot = k * cos_k + rotate_half(k) * sin_k (using modified cos/sin)
            # reroped_keys = (unroped_keys * cos_k_mod) + (rotate_half(unroped_keys) * sin_k_mod)
            
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
            max_position_id_allowed: int = int(kwargs.get("max_position_id", DEFAULT_MAX_POSITION_ID))
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
                num_prefix_keys: int = prefix_key_indices_sorted_tensor.numel() if prefix_key_indices_sorted_tensor is not None else 0
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
                # Diagnostic prints (only if SPARSE_DEBUG enabled)
                # Commented out verbose per-head details - uncomment if needed for debugging
                # if debug:
                #     print(f"\n  [POSITION REASSIGNMENT] Head {head_idx}:")
                #     
                #     # Batch extract for diagnostics
                #     if len(prefix_key_indices_sorted) > 0:
                #         prefix_original_tensor: torch.Tensor = position_ids_k_actual[0, prefix_key_indices_sorted]
                #         prefix_assigned_tensor: torch.Tensor = position_ids_k_per_head[0, head_idx, prefix_key_indices_sorted]
                #         prefix_original_positions: List[int] = prefix_original_tensor.cpu().tolist()
                #         prefix_assigned_positions: List[int] = prefix_assigned_tensor.cpu().tolist()
                #         
                #         print(f"    PREFIX KEYS ({len(prefix_key_indices_sorted)} keys):")
                #         print(f"      Original positions: {prefix_original_positions[:10]}{'...' if len(prefix_original_positions) > 10 else ''} (max={max(prefix_original_positions)})")
                #         print(f"      → Reassigned to:     {prefix_assigned_positions[:10]}{'...' if len(prefix_assigned_positions) > 10 else ''}")
                #         print(f"      ✓ CONTIGUOUS: [0, 1, 2, ..., {len(prefix_assigned_positions)-1}]")
                #     else:
                #         print(f"    PREFIX KEYS: 0 keys (no prefix)")
                #     
                #     if len(all_current_chunk_key_indices_sorted) > 0:
                #         current_chunk_original_tensor: torch.Tensor = position_ids_k_actual[0, all_current_chunk_key_indices_sorted]
                #         current_chunk_assigned_tensor: torch.Tensor = position_ids_k_per_head[0, head_idx, all_current_chunk_key_indices_sorted]
                #         current_chunk_original_positions: List[int] = current_chunk_original_tensor.cpu().tolist()
                #         current_chunk_assigned_positions: List[int] = current_chunk_assigned_tensor.cpu().tolist()
                #         
                #         print(f"    CURRENT CHUNK KEYS ({len(all_current_chunk_key_indices_sorted)} keys):")
                #         print(f"      Original positions: {current_chunk_original_positions[:10]}{'...' if len(current_chunk_original_positions) > 10 else ''} (range: {min(current_chunk_original_positions)}-{max(current_chunk_original_positions)})")
                #         print(f"      → Reassigned to:     {current_chunk_assigned_positions[:10]}{'...' if len(current_chunk_assigned_positions) > 10 else ''}")
                #         print(f"      ✓ CONTIGUOUS: [{current_chunk_start}, {current_chunk_start+1}, ..., {current_chunk_start + len(all_current_chunk_key_indices_sorted) - 1}]")
                #     else:
                #         print(f"    CURRENT CHUNK KEYS: 0 keys")
                #     
                #     print(f"    Summary: Prefix max_original_pos={max_prefix_position}, Current chunk starts at {current_chunk_start}")

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
                
                # Adversarial override disabled
                # adversarial_base: int = 10_000_000
                # position_ids_q_per_head[0, head_idx, :] = 1
                # position_ids_k_per_head[0, head_idx, :] = (
                #     adversarial_base + torch.arange(
                #         seq_len_k, device=position_ids_k_per_head.device, dtype=torch.long
                #     )
                # )
                
                # Diagnostic print for queries (only if SPARSE_DEBUG enabled)
                # Commented out verbose query position details - uncomment if needed for debugging
                # if debug and seq_len_q > 0:
                #     query_assigned_tensor: torch.Tensor = position_ids_q_per_head[0, head_idx, :min(10, seq_len_q)]
                #     query_original_tensor: torch.Tensor = position_ids_q[0, :min(10, seq_len_q)]
                #     query_assigned_positions: List[int] = query_assigned_tensor.cpu().tolist()
                #     query_original_positions: List[int] = query_original_tensor.cpu().tolist()
                #     print(f"    QUERY POSITIONS ({seq_len_q} queries):")
                #     print(f"      Original positions: {query_original_positions}{'...' if seq_len_q > 10 else ''}")
                #     print(f"      → Reassigned to:     {query_assigned_positions}{'...' if seq_len_q > 10 else ''}")
                #     print(f"      (Queries match their corresponding key positions)")
            
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
            # ====================================================================
            # Compare Attention Patterns (Original vs Repositioned) - Selected Keys Only
            # ====================================================================
            # Reuse dense_mask from line 725 (already computed above)
            # dense_mask is already available from position reassignment section
                
                # Commented out verbose per-query comparisons - uncomment if needed for debugging
                # num_heads: int = exp_attention_weights.shape[1]
                # seq_len_q: int = exp_attention_weights.shape[2]
                # seq_len_k: int = exp_attention_weights.shape[3]
                # 
                # # Sample a few queries to analyze
                # sample_queries = [0, seq_len_q // 4, seq_len_q // 2, seq_len_q * 3 // 4, seq_len_q - 1]
                # sample_queries = [q for q in sample_queries if q < seq_len_q]
                # sample_head = 0  # Analyze head 0
                # 
                # print(f"\n  [Attention Pattern Comparison] (Selected Keys Only)")
                # for q_idx in sample_queries[:3]:  # Show first 3 sample queries
                #     # Get selected key indices for this query
                #     selected_k_indices = torch.nonzero(dense_mask[0, sample_head, q_idx] > 0).squeeze(-1).cpu().tolist()
                #     if len(selected_k_indices) == 0:
                #         continue
                #     
                #     # Get attention scores for selected keys only (before and after)
                #     orig_scores_selected = exp_attention_weights[0, sample_head, q_idx, selected_k_indices].cpu()
                #     reroped_scores_selected = exp_attention_weights_reroped[0, sample_head, q_idx, selected_k_indices].cpu()
                #     
                #     # Find max key (before and after)
                #     orig_max_idx_in_selected = torch.argmax(orig_scores_selected).item()
                #     reroped_max_idx_in_selected = torch.argmax(reroped_scores_selected).item()
                #     orig_max_k_idx = selected_k_indices[orig_max_idx_in_selected]
                #     reroped_max_k_idx = selected_k_indices[reroped_max_idx_in_selected]
                #     
                #     # Get top-3 keys (before and after)
                #     orig_top3_values, orig_top3_indices = torch.topk(orig_scores_selected, k=min(3, len(selected_k_indices)))
                #     reroped_top3_values, reroped_top3_indices = torch.topk(reroped_scores_selected, k=min(3, len(selected_k_indices)))
                #     orig_top3_k_indices = [selected_k_indices[i.item()] for i in orig_top3_indices]
                #     reroped_top3_k_indices = [selected_k_indices[i.item()] for i in reroped_top3_indices]
                #     
                #     # Compare max key
                #     max_changed = orig_max_k_idx != reroped_max_k_idx
                #     max_score_diff = abs(orig_scores_selected[orig_max_idx_in_selected].item() - reroped_scores_selected[reroped_max_idx_in_selected].item())
                #     
                #     print(f"    Q{q_idx}: {len(selected_k_indices)} selected keys")
                #     print(f"      Max key: {orig_max_k_idx} → {reroped_max_k_idx} {'(CHANGED)' if max_changed else '(same)'}, score_diff={max_score_diff:.6f}")
                #     print(f"      Top-3 keys (orig): {orig_top3_k_indices} with scores {[f'{v.item():.4f}' for v in orig_top3_values]}")
                #     print(f"      Top-3 keys (reroped): {reroped_top3_k_indices} with scores {[f'{v.item():.4f}' for v in reroped_top3_values]}")
                #     
                #     # Check if top-3 changed
                #     top3_same = set(orig_top3_k_indices) == set(reroped_top3_k_indices)
                #     print(f"      Top-3 same? {top3_same}")
                # 
                # # Compare overall statistics for selected keys
                # orig_selected_mean = exp_attention_weights[exp_attention_weights > 0].mean().item()
                # reroped_selected_mean = exp_attention_weights_reroped[exp_attention_weights_reroped > 0].mean().item()
                # orig_selected_std = exp_attention_weights[exp_attention_weights > 0].std().item()
                # reroped_selected_std = exp_attention_weights_reroped[exp_attention_weights_reroped > 0].std().item()
                # 
                # print(f"    Selected keys stats: mean {orig_selected_mean:.6f} → {reroped_selected_mean:.6f}, std {orig_selected_std:.6f} → {reroped_selected_std:.6f}")
                # 
                # threshold_abs = 0.2
                # threshold_rel = 0.2
                # abs_ok = weights_diff.item() < threshold_abs
                # rel_ok = weights_rel_diff < threshold_rel
                # passed = abs_ok and rel_ok
                # print(f"  [Attention Weights Verification] max_diff={weights_diff.item():.6f} (threshold={threshold_abs}, ok={abs_ok}), rel_diff={weights_rel_diff:.6f} (threshold={threshold_rel}, ok={rel_ok}) {'✓' if passed else '✗ FAILED'}")
                # print(f"    Original weights range: [{exp_attention_weights.min().item():.6f}, {orig_max:.6f}], Reroped range: [{exp_attention_weights_reroped.min().item():.6f}, {reroped_max:.6f}]")
                # print(f"    Note: Large diff expected when using per-head position reassignment (position IDs changed)")
                
                # Commented out detailed attention weights comparison - uncomment if needed for debugging
                # ====================================================================
                # Detailed Attention Weights Comparison (Original vs Repositioned)
                # ====================================================================
                # weights_diff_tensor: torch.Tensor = torch.abs(exp_attention_weights - exp_attention_weights_reroped)
                # 
                # # Find where max_diff in attention weights occurs
                # weights_max_diff_flat_idx = weights_diff_tensor.argmax()
                # weights_max_indices = torch.unravel_index(weights_max_diff_flat_idx.cpu(), weights_diff_tensor.shape)
                # w_batch, w_head, w_query, w_key = weights_max_indices[0].item(), weights_max_indices[1].item(), weights_max_indices[2].item(), weights_max_indices[3].item()
                # 
                # # Get percentiles of weight differences (selected keys only)
                # weights_diff_selected = weights_diff_tensor * dense_mask  # Only selected keys
                # weights_diff_flat = weights_diff_selected[weights_diff_selected > 0].flatten().cpu().float()
                # if len(weights_diff_flat) > 0:
                #     percentiles = [50, 75, 90, 95, 99, 99.9]
                #     max_quantile_size = 1000000  # 1M elements max for quantile
                #     if len(weights_diff_flat) > max_quantile_size:
                #         sample_indices = torch.randperm(len(weights_diff_flat), device=weights_diff_flat.device)[:max_quantile_size]
                #         weights_diff_sampled = weights_diff_flat[sample_indices]
                #         percentile_values = [torch.quantile(weights_diff_sampled, p/100.0).item() for p in percentiles]
                #     else:
                #         percentile_values = [torch.quantile(weights_diff_flat, p/100.0).item() for p in percentiles]
                # else:
                #     percentile_values = [0.0] * 6
                # 
                # # Get top-K query-key pairs with largest weight differences
                # top_k = min(10, weights_diff_tensor.numel())
                # weights_diff_flat_all = weights_diff_tensor.flatten().cpu().float()
                # top_k_values, top_k_indices = torch.topk(weights_diff_flat_all, k=top_k)
                # top_k_positions = []
                # for idx in top_k_indices:
                #     pos = torch.unravel_index(idx.cpu(), weights_diff_tensor.shape)
                #     top_k_positions.append((pos[0].item(), pos[1].item(), pos[2].item(), pos[3].item()))
                # 
                # # Per-head statistics for attention weights
                # weights_diff_per_head = weights_diff_selected.mean(dim=(0, 2, 3))
                # head_max_weight_diffs = torch.amax(weights_diff_selected, dim=(0, 2, 3))
                # 
                # print(f"\n  [Attention Weights Detailed Comparison]")
                # print(f"    Max weight diff: {weights_diff.item():.6f} at batch={w_batch}, head={w_head}, query={w_query}, key={w_key}")
                # print(f"      Original weight: {exp_attention_weights[w_batch, w_head, w_query, w_key].item():.6f}")
                # print(f"      Repositioned weight: {exp_attention_weights_reroped[w_batch, w_head, w_query, w_key].item():.6f}")
                # print(f"      Difference: {weights_diff_tensor[w_batch, w_head, w_query, w_key].item():.6f}")
                # 
                # if len(weights_diff_flat) > 0:
                #     print(f"\n    [Weight Difference Percentiles (Selected Keys Only)]")
                #     for p, v in zip(percentiles, percentile_values):
                #         print(f"      {p}th percentile: {v:.6f}")
                # 
                # # Debug: Compute raw QK^T scores (before normalization) for comparison
                # num_key_value_groups: int = _get_num_key_value_groups(queries, keys)
                # key_states_orig = repeat_kv(keys, num_key_value_groups)
                # key_states_reroped = repeat_kv(reroped_keys, num_key_value_groups)
                # 
                # raw_scores_orig = (torch.matmul(queries, key_states_orig.transpose(2, 3)) * scaling)
                # raw_scores_reroped = (torch.matmul(reroped_queries, key_states_reroped.transpose(2, 3)) * scaling)
                # 
                # # Get row-wise max for both
                # row_max_orig = raw_scores_orig.max(dim=-1, keepdim=True)[0]
                # row_max_reroped = raw_scores_reroped.max(dim=-1, keepdim=True)[0]
                # 
                # print(f"\n    [Top-{min(5, top_k)} Query-Key Pairs with Largest Weight Differences]")
                # for i, (val, pos) in enumerate(zip(top_k_values[:5], top_k_positions[:5])):
                #     b, h, q, k = pos
                #     orig_w = exp_attention_weights[b, h, q, k].item()
                #     reroped_w = exp_attention_weights_reroped[b, h, q, k].item()
                #     is_selected = dense_mask[b, h, q, k].item() > 0
                #     
                #     # Get raw QK^T scores (before normalization)
                #     raw_orig = raw_scores_orig[b, h, q, k].item()
                #     raw_reroped = raw_scores_reroped[b, h, q, k].item()
                #     
                #     # Get row-wise max for this query
                #     row_max_orig_q = row_max_orig[b, h, q, 0].item()
                #     row_max_reroped_q = row_max_reroped[b, h, q, 0].item()
                #     
                #     # Get normalized scores (before exp)
                #     normalized_orig = raw_orig - row_max_orig_q
                #     normalized_reroped = raw_reroped - row_max_reroped_q
                #     
                #     # Find which key is max in this row
                #     max_key_orig = raw_scores_orig[b, h, q, :].argmax().item()
                #     max_key_reroped = raw_scores_reroped[b, h, q, :].argmax().item()
                #     
                #     print(f"      #{i+1}: diff={val.item():.6f} at head={h}, query={q}, key={k} {'(selected)' if is_selected else '(NOT selected)'}")
                #     print(f"         orig_weight={orig_w:.6f}, reroped_weight={reroped_w:.6f}")
                #     print(f"         Raw QK^T: orig={raw_orig:.6f}, reroped={raw_reroped:.6f}, diff={raw_orig - raw_reroped:.6f}")
                #     print(f"         Row max: orig={row_max_orig_q:.6f} (key {max_key_orig}), reroped={row_max_reroped_q:.6f} (key {max_key_reroped})")
                #     print(f"         Normalized (before exp): orig={normalized_orig:.6f}, reroped={normalized_reroped:.6f}")
                #     if max_key_orig != max_key_reroped:
                #         print(f"         ⚠️  MAX KEY CHANGED: {max_key_orig} → {max_key_reroped}")
                # 
                # print(f"\n    [Per-Head Weight Differences (Selected Keys Only)]")
                # for h_idx in range(min(8, len(head_max_weight_diffs))):  # Show first 8 heads
                #     print(f"      Head {h_idx}: max_diff={head_max_weight_diffs[h_idx].item():.6f}, mean_diff={weights_diff_per_head[h_idx].item():.6f}")
                # 
                # # Analyze ranking changes for sample queries
                # print(f"\n    [Ranking Changes for Sample Queries]")
                # sample_queries_weights = [0, seq_len_q // 4, seq_len_q // 2] if seq_len_q > 1 else [0]
                # for q_idx in sample_queries_weights[:3]:
                #     if q_idx >= seq_len_q:
                #         continue
                #     # Get selected keys for this query
                #     selected_k_indices = torch.nonzero(dense_mask[0, sample_head, q_idx] > 0).squeeze(-1).cpu().tolist()
                #     if len(selected_k_indices) == 0:
                #         continue
                #     
                #     # Get attention scores for selected keys
                #     orig_scores = exp_attention_weights[0, sample_head, q_idx, selected_k_indices].cpu()
                #     reroped_scores = exp_attention_weights_reroped[0, sample_head, q_idx, selected_k_indices].cpu()
                #     
                #     # Get rankings (indices sorted by score, descending)
                #     orig_rankings = torch.argsort(orig_scores, descending=True)
                #     reroped_rankings = torch.argsort(reroped_scores, descending=True)
                #     
                #     # Check how many keys changed rank
                #     orig_ranked_keys = [selected_k_indices[i.item()] for i in orig_rankings]
                #     reroped_ranked_keys = [selected_k_indices[i.item()] for i in reroped_rankings]
                #     
                #     # Count rank changes
                #     rank_changes = sum(1 for i, (ok, rk) in enumerate(zip(orig_ranked_keys, reroped_ranked_keys)) if ok != rk)
                #     
                #     # Get top-5 scores for display
                #     top5_orig_scores = [orig_scores[orig_rankings[i]].item() for i in range(min(5, len(orig_rankings)))]
                #     top5_reroped_scores = [reroped_scores[reroped_rankings[i]].item() for i in range(min(5, len(reroped_rankings)))]
                #     
                #     print(f"      Q{q_idx}: {len(selected_k_indices)} selected keys, {rank_changes} keys changed rank")
                #     print(f"        Top-5 orig: keys={orig_ranked_keys[:5]}, scores={[f'{s:.4f}' for s in top5_orig_scores]}")
                #     print(f"        Top-5 reroped: keys={reroped_ranked_keys[:5]}, scores={[f'{s:.4f}' for s in top5_reroped_scores]}")
            
            # assert passed, f"Re-roped attention weights don't match original: diff={weights_diff.item()}, rel_diff={weights_rel_diff:.6f} (thresholds: abs<{threshold_abs}, rel<{threshold_rel})"
            # Overwrite with re-roped version after verification (cast to original dtype to match model precision)
            # TEST: Add random perturbation to verify it affects downstream (comment out for production)
            # perturbation = torch.randn_like(exp_attention_weights_reroped) * 0.0  # 10% noise
            # exp_attention_weights = (exp_attention_weights_reroped + perturbation).to(exp_attention_weights.dtype)
            
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
        

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)

    # Use internal helpers with pre-computed weights
    num: torch.Tensor = _get_attention_numerator(exp_attention_weights, value_states)
    den: torch.Tensor = _get_attention_denominator(exp_attention_weights)

    # Compute final attention output
    attention_output: torch.Tensor = (num / den).transpose(1, 2).contiguous()

    if return_attention_weights:
        # Normalize exponential weights to get attention probabilities
        attention_weights: torch.Tensor = exp_attention_weights / den
        return attention_output, attention_weights

    return attention_output
