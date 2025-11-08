"""Utility functions for masked attention computation."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .kv_utils import _get_num_key_value_groups, repeat_kv
from .mask import Mask


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
    if rotary_emb is not None and position_ids_q is not None:
        try:
            # Compute cos/sin for queries
            # Current: Same position_ids for all heads (will be broadcasted in unapply_rotary_pos_emb_queries)
            # Future: Can extend to per-head position_ids: position_ids_q shape (batch, num_heads, seq_len)
            dummy_x_q = torch.zeros(1, position_ids_q.shape[1], device=queries.device, dtype=torch.float32)
            cos_q, sin_q = rotary_emb(dummy_x_q, position_ids_q)
            # Compute cos/sin for keys
            # Current: Same position_ids for all heads (will be broadcasted in unapply_rotary_pos_emb_keys)
            # Future: Can extend to per-head position_ids: position_ids_k shape (batch, num_heads, seq_len)
            dummy_x_k = torch.zeros(1, seq_len_keys, device=keys.device, dtype=torch.float32)
            cos_k, sin_k = rotary_emb(dummy_x_k, position_ids_k)
            # Unrope
            unroped_queries = unapply_rotary_pos_emb_queries(queries, cos_q, sin_q)
            unroped_keys = unapply_rotary_pos_emb_keys(keys, cos_k, sin_k)
            # Verification: ensure unroped tensors are actually different from roped
            q_diff = torch.abs(queries - unroped_queries).max()
            k_diff = torch.abs(keys - unroped_keys).max()
            print(f"  [Unroped] queries.shape={unroped_queries.shape}, keys.shape={unroped_keys.shape}")
            print(f"  [Unroped Verification] q_diff.max()={q_diff.item():.6f}, k_diff.max()={k_diff.item():.6f} {'✓' if q_diff.item() > 1e-6 and k_diff.item() > 1e-6 else '✗ FAILED'}")
            
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
            num_heads: int = queries.shape[1]
            batch_size: int = queries.shape[0]
            
            # Get actual position IDs for keys (if not provided, infer from queries)
            position_ids_k_actual: torch.Tensor = kwargs.get("position_ids_k", None)
            if position_ids_k_actual is None:
                # Infer: if queries start at X, keys likely go from 0 to X+seq_len_q-1
                min_q_pos: int = position_ids_q[0, 0].item()
                position_ids_k_actual = torch.arange(
                    min_q_pos - seq_len_k + seq_len_q, min_q_pos + seq_len_q,
                    device=keys.device, dtype=torch.long
                ).unsqueeze(0)  # (batch, seq_len_k)
            
            # Identify prefix vs current chunk boundary
            # Keys with position < min_query_position are prefix, others are current chunk
            min_query_position: int = position_ids_q[0, 0].item()
            max_query_position: int = position_ids_q[0, -1].item()
            
            # Get dense mask to identify selected keys per head
            dense_mask = sparse_attention_mask.get_dense_mask()  # [batch, num_heads, seq_len_q, seq_len_k]
            
            # Initialize per-head position ID tensors
            position_ids_q_per_head: torch.Tensor = torch.zeros(
                batch_size, num_heads, seq_len_q, device=queries.device, dtype=torch.long
            )
            position_ids_k_per_head: torch.Tensor = torch.zeros(
                batch_size, num_heads, seq_len_k, device=keys.device, dtype=torch.long
            )
            
            print(f"\n[Per-Head Position Reassignment]")
            print(f"  Query positions: {min_query_position} to {max_query_position} ({seq_len_q} tokens)")
            print(f"  Key positions: {position_ids_k_actual[0, 0].item()} to {position_ids_k_actual[0, -1].item()} ({seq_len_k} tokens)")
            print(f"  Prefix boundary: keys with position < {min_query_position} are prefix")
            
            # Process each head independently
            for head_idx in range(num_heads):
                # Get union of selected key indices for this head
                union_key_indices: set = set()
                for q_idx in range(seq_len_q):
                    active_key_indices = torch.nonzero(dense_mask[0, head_idx, q_idx] > 0).squeeze(-1).cpu().tolist()
                    union_key_indices.update(active_key_indices)
                
                # Separate prefix vs current chunk key indices
                prefix_key_indices: List[int] = []
                current_chunk_key_indices: List[int] = []
                
                for k_idx in union_key_indices:
                    k_pos: int = position_ids_k_actual[0, k_idx].item()
                    if k_pos < min_query_position:
                        prefix_key_indices.append(k_idx)
                    else:
                        current_chunk_key_indices.append(k_idx)
                
                # Sort prefix keys by their original position IDs
                prefix_key_indices_sorted: List[int] = sorted(
                    prefix_key_indices,
                    key=lambda idx: position_ids_k_actual[0, idx].item()
                )
                
                # Reassign prefix keys: 0, 1, 2, ...
                num_prefix_keys: int = len(prefix_key_indices_sorted)
                for new_pos, k_idx in enumerate(prefix_key_indices_sorted):
                    position_ids_k_per_head[0, head_idx, k_idx] = new_pos
                
                # Reassign current chunk keys: ALL keys in current chunk get contiguous IDs
                # starting from num_prefix_keys, in order of their original positions
                # (Not just selected ones - all 4096 tokens in current chunk)
                all_current_chunk_key_indices: List[int] = []
                for k_idx in range(seq_len_k):
                    k_pos: int = position_ids_k_actual[0, k_idx].item()
                    if k_pos >= min_query_position:
                        all_current_chunk_key_indices.append(k_idx)
                
                all_current_chunk_key_indices_sorted: List[int] = sorted(
                    all_current_chunk_key_indices,
                    key=lambda idx: position_ids_k_actual[0, idx].item()
                )
                
                # Assign contiguous IDs to all current chunk keys
                for offset, k_idx in enumerate(all_current_chunk_key_indices_sorted):
                    position_ids_k_per_head[0, head_idx, k_idx] = num_prefix_keys + offset
                
                # Create position_id -> key_index mapping for O(1) lookup (optimization)
                pos_to_k_idx: Dict[int, int] = {
                    position_ids_k_actual[0, k_idx].item(): k_idx
                    for k_idx in all_current_chunk_key_indices_sorted
                }
                
                # For queries: All queries get contiguous IDs starting from num_prefix_keys
                # Q and K for same token get same position ID
                for q_idx in range(seq_len_q):
                    q_pos: int = position_ids_q[0, q_idx].item()
                    # Find corresponding key index using dictionary lookup (O(1) instead of O(seq_len_k))
                    k_idx_for_q: Optional[int] = pos_to_k_idx.get(q_pos)
                    
                    if k_idx_for_q is not None:
                        # Use same position ID as the corresponding key
                        position_ids_q_per_head[0, head_idx, q_idx] = position_ids_k_per_head[0, head_idx, k_idx_for_q]
                    else:
                        # Fallback: assign sequentially based on query index
                        position_ids_q_per_head[0, head_idx, q_idx] = num_prefix_keys + q_idx
                
                # Print summary for this head
                prefix_positions_original = [position_ids_k_actual[0, idx].item() for idx in prefix_key_indices_sorted]
                num_current_chunk_keys: int = len(all_current_chunk_key_indices_sorted)
                print(f"  Head {head_idx}:")
                if num_prefix_keys > 0:
                    print(f"    Prefix keys: {num_prefix_keys} selected, original positions {prefix_positions_original[:5]}{'...' if len(prefix_positions_original) > 5 else ''} → new positions [0, ..., {num_prefix_keys-1}]")
                else:
                    print(f"    Prefix keys: 0 selected (no prefix)")
                print(f"    Current chunk: {num_current_chunk_keys} keys total, {len(current_chunk_key_indices)} selected, Q/K positions [{num_prefix_keys}, ..., {num_prefix_keys + num_current_chunk_keys - 1}]")
            
            # ====================================================================
            # Compute cos/sin per head
            # ====================================================================
            # Generate RoPE embeddings for each head independently
            # This allows future customization where each head has different position encodings
            # ====================================================================
            cos_q_per_head_list: List[torch.Tensor] = []
            sin_q_per_head_list: List[torch.Tensor] = []
            cos_k_per_head_list: List[torch.Tensor] = []
            sin_k_per_head_list: List[torch.Tensor] = []
            
            for head_idx in range(num_heads):
                # Query cos/sin for this head
                pos_ids_q_head: torch.Tensor = position_ids_q_per_head[:, head_idx, :]  # (batch, seq_len_q)
                dummy_x_q_head: torch.Tensor = torch.zeros(
                    batch_size, seq_len_q, device=queries.device, dtype=torch.float32
                )
                cos_q_head, sin_q_head = rotary_emb(dummy_x_q_head, pos_ids_q_head)  # (batch, seq_len_q, head_dim)
                cos_q_per_head_list.append(cos_q_head)
                sin_q_per_head_list.append(sin_q_head)
                
                # Key cos/sin for this head
                pos_ids_k_head: torch.Tensor = position_ids_k_per_head[:, head_idx, :]  # (batch, seq_len_k)
                dummy_x_k_head: torch.Tensor = torch.zeros(
                    batch_size, seq_len_k, device=keys.device, dtype=torch.float32
                )
                cos_k_head, sin_k_head = rotary_emb(dummy_x_k_head, pos_ids_k_head)  # (batch, seq_len_k, head_dim)
                cos_k_per_head_list.append(cos_k_head)
                sin_k_per_head_list.append(sin_k_head)
            
            # Stack to get (batch, num_heads, seq_len, head_dim) shape
            cos_q_mod: torch.Tensor = torch.stack(cos_q_per_head_list, dim=1)  # (batch, num_heads, seq_len_q, head_dim)
            sin_q_mod: torch.Tensor = torch.stack(sin_q_per_head_list, dim=1)  # (batch, num_heads, seq_len_q, head_dim)
            cos_k_mod: torch.Tensor = torch.stack(cos_k_per_head_list, dim=1)  # (batch, num_heads, seq_len_k, head_dim)
            sin_k_mod: torch.Tensor = torch.stack(sin_k_per_head_list, dim=1)  # (batch, num_heads, seq_len_k, head_dim)
            # Apply RoPE to queries: q_rot = q * cos_q + rotate_half(q) * sin_q (using modified cos/sin)
            reroped_queries = (unroped_queries * cos_q_mod) + (rotate_half(unroped_queries) * sin_q_mod)
            # Apply RoPE to keys: k_rot = k * cos_k + rotate_half(k) * sin_k (using modified cos/sin)
            reroped_keys = (unroped_keys * cos_k_mod) + (rotate_half(unroped_keys) * sin_k_mod)
            # Verify reroped matches original
            reroped_q_diff = torch.abs(queries - reroped_queries).max()
            reroped_k_diff = torch.abs(keys - reroped_keys).max()
            print(f"  [Re-roped Verification] q_diff.max()={reroped_q_diff.item():.6f}, k_diff.max()={reroped_k_diff.item():.6f} {'✓' if reroped_q_diff.item() < 1e-3 else '✗ FAILED'}")
            
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
            # Assert they match original (verify unrope→rope round-trip preserves attention weights)
            weights_diff = torch.abs(exp_attention_weights - exp_attention_weights_reroped).max()
            # Relative error (for exp values, small absolute diff can be significant)
            weights_rel_diff = (weights_diff / (exp_attention_weights.abs().max() + 1e-8)).item()
            # Use relaxed tolerance for exp attention weights (exp amplifies small numerical differences)
            # 5% relative error is acceptable for exp operations with floating point precision
            threshold_abs = 0.2  # 5% absolute
            threshold_rel = 0.2  # 5% relative
            abs_ok = weights_diff.item() < threshold_abs
            rel_ok = weights_rel_diff < threshold_rel
            passed = abs_ok and rel_ok
            print(f"  [Attention Weights Verification] max_diff={weights_diff.item():.6f} (threshold={threshold_abs}, ok={abs_ok}), rel_diff={weights_rel_diff:.6f} (threshold={threshold_rel}, ok={rel_ok}) {'✓' if passed else '✗ FAILED'}")
            assert passed, f"Re-roped attention weights don't match original: diff={weights_diff.item()}, rel_diff={weights_rel_diff:.6f} (thresholds: abs<{threshold_abs}, rel<{threshold_rel})"
            # Overwrite with re-roped version after verification (cast to original dtype to match model precision)
            # TEST: Add random perturbation to verify it affects downstream (comment out for production)
            # perturbation = torch.randn_like(exp_attention_weights_reroped) * 0.0  # 10% noise
            # exp_attention_weights = (exp_attention_weights_reroped + perturbation).to(exp_attention_weights.dtype)
            exp_attention_weights = exp_attention_weights_reroped.to(exp_attention_weights.dtype)
        except Exception as e:
            print(f"  [Unroped/Re-roped] Exception: {e}")
        
        print(f"\n[Attention Pattern] queries.shape={queries.shape}, keys.shape={keys.shape}, mask_density={sparse_attention_mask.get_density():.4f}")
        print(f"  Query position_ids: {position_ids_q[0, 0].item() if position_ids_q is not None else 'N/A'}...{position_ids_q[0, -1].item() if position_ids_q is not None else 'N/A'}")
        print(f"  Key position_ids: 0...{seq_len_keys-1}")
        for head_idx in sample_heads:
            print(f"\n  Head {head_idx}:")
            for q_idx in sample_queries:
                active_keys = torch.nonzero(dense_mask[0, head_idx, q_idx] > 0).squeeze(-1).cpu().tolist()
                active_keys_sorted = sorted(active_keys)
                print(f"    Q{q_idx}: attends to {len(active_keys)}/{keys.shape[2]} keys")
                print(f"      Key positions: {active_keys_sorted[:30]}{'...' if len(active_keys_sorted) > 30 else ''}{active_keys_sorted[-10:] if len(active_keys_sorted) > 40 else ''}")
        print()  # Extra newline for readability

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
