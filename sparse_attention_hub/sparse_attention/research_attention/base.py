"""Base classes for research attention mechanisms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch import nn
import os
import numpy as np
import json

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

from ..base import SparseAttention, SparseAttentionConfig
from ..utils.mask import Mask
from ..utils.mask_attention_utils import (
    get_masked_attention_output,
    get_true_attention_output,
)
from .maskers.base import MaskerConfig, ResearchMasker
from .maskers.sampling.base import SamplingMasker
from .rope_utils import (
    unapply_rotary_pos_emb,
    unapply_rotary_pos_emb_queries,
    unapply_rotary_pos_emb_keys,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_queries,
    apply_rotary_pos_emb_keys,
    compute_rope_cos_sin,
)

MicroMetricLogger.register_metric("research_attention_density", float)
MicroMetricLogger.register_metric("research_attention_output_error", float)
MicroMetricLogger.register_metric("research_attention_weight_diff", dict)
MicroMetricLogger.register_metric("research_mask_roped_vs_unroped", dict)


@dataclass
class ResearchAttentionConfig(SparseAttentionConfig):
    """Configuration class for research attention mechanisms."""

    masker_configs: List[MaskerConfig]
    # Optional: translate-pack K+Chunk (keep slope 1 for K+Chunk; only translate)
    pack_k_chunk_translation: bool = False


class ResearchAttention(SparseAttention):
    """Base class for research attention mechanisms with maskers."""

    maskers: List[ResearchMasker]
    
    # Class-level variables for mask logging
    _sample_idx: int = 0  # Track current sample index
    _prev_seq_len_k: int = 0  # Track previous seq_len_k to detect new samples
    _mask_save_dir: Optional[Path] = None  # Directory for saving masks

    def __init__(
        self,
        sparse_attention_config: SparseAttentionConfig,
        maskers: List[ResearchMasker],
    ) -> None:
        """Initialize research attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            maskers: List of research maskers to apply.

        Raises:
            ValueError: If more than one sampling masker is provided.
        """
        super().__init__(sparse_attention_config)

        # Validate that there's at most one sampling masker
        sampling_masker_count: int = sum(
            1 for masker in maskers if isinstance(masker, SamplingMasker)
        )
        if sampling_masker_count > 1:
            raise ValueError(
                "Only one sampling masker supported for efficiency; consider implementing all sampling logic in one masker"
            )

        self.maskers = maskers
        
        # Initialize mask saving directory if enabled
        if os.environ.get("SAVE_MASKS", "0").lower() in ("1", "true", "yes"):
            self._init_mask_save_dir()

    def _init_mask_save_dir(self) -> None:
        """Initialize directory for saving masks."""
        # Get model name from environment or use default
        model_name: str = os.environ.get("MODEL_NAME", "unknown_model")
        # Sanitize model name for filesystem
        model_name_safe: str = model_name.replace("/", "_").replace("-", "_")
        
        # Create /data/masks/{model_name} directory
        base_dir: Path = Path("/data/masks")
        self._mask_save_dir = base_dir / model_name_safe
        self._mask_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset sample tracking
        ResearchAttention._sample_idx = 0
        ResearchAttention._prev_seq_len_k = 0
        
        print(f"[MaskLogging] Initialized mask save directory: {self._mask_save_dir}")
        print(f"[MaskLogging] Will save masks only for first sample (sample_idx=0)")

    def _update_sample_idx(self, seq_len_q: int, seq_len_k: int) -> None:
        """Update sample index based on chunk progression.
        
        Detects new sample when:
        - First chunk: seq_len_q == seq_len_k (no prefix yet)
        - Reset detected: seq_len_k decreases significantly
        """
        # Detect new sample: reset to first chunk (seq_q == seq_k) after generation (seq_q == 1)
        # or significant decrease in seq_len_k
        if (seq_len_q == seq_len_k and seq_len_q >= 100 and 
            ResearchAttention._prev_seq_len_k > 0 and 
            ResearchAttention._prev_seq_len_k < seq_len_k * 0.5):
            # New sample detected
            ResearchAttention._sample_idx += 1
            print(f"[MaskLogging] New sample detected: sample_idx={ResearchAttention._sample_idx}")
        
        ResearchAttention._prev_seq_len_k = seq_len_k

    def _save_mask(
        self,
        mask: torch.Tensor,
        mask_type: str,  # "computed" or "roped"
        layer_idx: int,
        head_idx: int,
        seq_len_q: int,
        seq_len_k: int,
        sample_idx: int,
    ) -> None:
        """Save mask to disk as numpy array.
        
        Args:
            mask: Mask tensor to save (seq_len_q, seq_len_k)
            mask_type: Type of mask ("computed" or "roped")
            layer_idx: Layer index
            head_idx: Head index
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            sample_idx: Sample index
        """
        if self._mask_save_dir is None:
            return
        
        # Only save for first sample
        if sample_idx != 0:
            return
        
        # Create subdirectory: layer_{layer_idx}/head_{head_idx}/
        layer_dir: Path = self._mask_save_dir / f"layer_{layer_idx}" / f"head_{head_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: mask_{type}_l{layer_idx}_h{head_idx}_q{seq_len_q}_k{seq_len_k}_s{sample_idx}.npy
        filename: str = f"mask_{mask_type}_l{layer_idx}_h{head_idx}_q{seq_len_q}_k{seq_len_k}_s{sample_idx}.npy"
        filepath: Path = layer_dir / filename
        
        # Convert to numpy and save
        # Handle BFloat16: NumPy doesn't support bfloat16, convert to float32 first
        mask_cpu: torch.Tensor = mask.detach().cpu()
        if mask_cpu.dtype == torch.bfloat16:
            mask_cpu = mask_cpu.to(torch.float32)
        mask_np: np.ndarray = mask_cpu.numpy()
        np.save(filepath, mask_np)
        
        # Save metadata JSON file
        metadata_file: Path = layer_dir / f"metadata_l{layer_idx}_h{head_idx}_q{seq_len_q}_k{seq_len_k}_s{sample_idx}.json"
        metadata: Dict[str, Any] = {
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "seq_len_q": seq_len_q,
            "seq_len_k": seq_len_k,
            "sample_idx": sample_idx,
            "mask_type": mask_type,
            "mask_shape": list(mask_np.shape),
            "model_name": os.environ.get("MODEL_NAME", "unknown_model"),
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _prepare_unroped_qk_for_mask(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        module: nn.Module,
        sparse_meta_data: Dict[Any, Any],
        kwargs: Dict[str, Any],
        layer_idx: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Prepare unroped Q/K for mask computation if EXTEND_CONTEXT is enabled.
        
        Returns:
            Tuple of (queries_for_mask, keys_for_mask, unroped_used)
        """
        extend_context: bool = os.environ.get("EXTEND_CONTEXT", "0").lower() in ("1", "true", "yes")
        queries_for_mask: torch.Tensor = queries
        keys_for_mask: torch.Tensor = keys
        unroped_used: bool = False
        
        if extend_context:
            cos: Optional[torch.Tensor] = kwargs.get("cos")
            sin: Optional[torch.Tensor] = kwargs.get("sin")
            
            if cos is None or sin is None:
                rotary_emb: Optional[Any] = sparse_meta_data.get("_rotary_emb")
                position_ids: Optional[torch.Tensor] = kwargs.get("position_ids")
                seq_len_keys: int = keys.shape[2]
                seq_len_queries: int = queries.shape[2]
                
                if rotary_emb is not None:
                    try:
                        keys_position_ids: torch.Tensor = torch.arange(
                            0, seq_len_keys, device=keys.device, dtype=torch.long
                        ).unsqueeze(0)
                        if position_ids is not None and position_ids.shape[1] == seq_len_queries:
                            queries_position_ids: torch.Tensor = position_ids
                        else:
                            queries_position_ids: torch.Tensor = torch.arange(
                                0, seq_len_queries, device=queries.device, dtype=torch.long
                            ).unsqueeze(0)
                        dummy_x_keys: torch.Tensor = torch.zeros(
                            1, seq_len_keys, device=keys.device, dtype=torch.float32
                        )
                        cos_keys, sin_keys = rotary_emb(dummy_x_keys, keys_position_ids)
                        dummy_x_queries: torch.Tensor = torch.zeros(
                            1, seq_len_queries, device=queries.device, dtype=torch.float32
                        )
                        cos_queries, sin_queries = rotary_emb(dummy_x_queries, queries_position_ids)
                        cos = (cos_queries, cos_keys)
                        sin = (sin_queries, sin_keys)
                    except (TypeError, ValueError):
                        try:
                            keys_position_ids: torch.Tensor = torch.arange(
                                0, seq_len_keys, device=keys.device, dtype=torch.long
                            ).unsqueeze(0)
                            cos_keys, sin_keys = rotary_emb(keys_position_ids)
                            if position_ids is not None and position_ids.shape[1] == seq_len_queries:
                                queries_position_ids: torch.Tensor = position_ids
                            else:
                                queries_position_ids: torch.Tensor = torch.arange(
                                    0, seq_len_queries, device=queries.device, dtype=torch.long
                                ).unsqueeze(0)
                            cos_queries, sin_queries = rotary_emb(queries_position_ids)
                            cos = (cos_queries, cos_keys)
                            sin = (sin_queries, sin_keys)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to compute cos/sin from rotary_emb in sparse_meta_data: {e}"
                            ) from e
                elif rotary_emb is None:
                    seq_len_keys: int = keys.shape[2]
                    full_position_ids: torch.Tensor = torch.arange(
                        0, seq_len_keys, device=keys.device, dtype=torch.long
                    ).unsqueeze(0)
                    try:
                        computed_cos, computed_sin = compute_rope_cos_sin(
                            module=module,
                            position_ids=full_position_ids,
                            seq_len=seq_len_keys,
                        )
                        cos = computed_cos
                        sin = computed_sin
                    except Exception as e:
                        raise RuntimeError(
                            f"EXTEND_CONTEXT enabled but failed to compute cos/sin at layer {layer_idx}. "
                            f"Original error: {e}"
                        ) from e
                else:
                    available_kwargs = list(kwargs.keys())
                    raise ValueError(
                        f"EXTEND_CONTEXT enabled but rotary_emb not available "
                        f"at layer {layer_idx}. Available kwargs: {available_kwargs}."
                    )
            
            try:
                if isinstance(cos, tuple) and isinstance(sin, tuple):
                    cos_queries, cos_keys = cos
                    sin_queries, sin_keys = sin
                    # Step 1: Unrope Q/K
                    queries_unroped: torch.Tensor = unapply_rotary_pos_emb_queries(
                        queries, cos_queries, sin_queries
                    )
                    keys_unroped: torch.Tensor = unapply_rotary_pos_emb_keys(
                        keys, cos_keys, sin_keys
                    )
                    
                    # Step 2: Apply scaled RoPE for mask computation
                    # Scale position IDs to [0, 8191] range for valid cos/sin computation
                    # Keys: [0, ..., seq_len_keys-1] → [0, ..., 8191]
                    # Queries: [min_q_pos, ..., max_q_pos] → [8191-seq_len_queries, ..., 8191]
                    max_position_id: int = 8191
                    
                    # Check if scaling is needed
                    needs_scaling: bool = False
                    if position_ids is not None and position_ids.shape[1] == seq_len_queries:
                        max_q_pos: int = int(position_ids[0, -1].item())
                        needs_scaling = max_q_pos > max_position_id
                    else:
                        needs_scaling = seq_len_queries - 1 > max_position_id
                    
                    # Only scale if position IDs exceed valid range
                    if needs_scaling and rotary_emb is not None:
                        # Use same scale factor for both keys and queries
                        # Keys: [0, ..., seq_len_keys-1] → [0, ..., 8191]
                        # Queries: [seq_len_keys-seq_len_queries, ..., seq_len_keys-1] → [scaled_start, ..., 8191]
                        if seq_len_keys > 1:
                            scale_factor: float = max_position_id / (seq_len_keys - 1)
                            
                            # Scale key position IDs: [0, ..., seq_len_keys-1] → [0, ..., 8191]
                            keys_position_ids_scaled: torch.Tensor = (
                                torch.arange(0, seq_len_keys, device=keys.device, dtype=torch.float32) * scale_factor
                            ).long().unsqueeze(0)
                            
                            # Scale query position IDs using same factor
                            # Queries correspond to last seq_len_queries positions of keys
                            if position_ids is not None and position_ids.shape[1] == seq_len_queries:
                                # Use actual query position IDs from kwargs, scale them
                                queries_position_ids_scaled: torch.Tensor = (
                                    position_ids.float() * scale_factor
                                ).long()
                            else:
                                # Fallback: queries are last seq_len_queries positions of keys
                                query_start_pos: int = seq_len_keys - seq_len_queries
                                queries_position_ids_scaled: torch.Tensor = (
                                    torch.arange(query_start_pos, seq_len_keys, device=queries.device, dtype=torch.float32) * scale_factor
                                ).long().unsqueeze(0)
                        else:
                            keys_position_ids_scaled: torch.Tensor = torch.zeros(1, seq_len_keys, device=keys.device, dtype=torch.long)
                            queries_position_ids_scaled: torch.Tensor = torch.zeros(1, seq_len_queries, device=queries.device, dtype=torch.long)
                        
                        # Compute cos/sin with scaled position IDs
                        dummy_x_keys_scaled: torch.Tensor = torch.zeros(
                            1, seq_len_keys, device=keys.device, dtype=torch.float32
                        )
                        cos_keys_scaled, sin_keys_scaled = rotary_emb(dummy_x_keys_scaled, keys_position_ids_scaled)
                        
                        dummy_x_queries_scaled: torch.Tensor = torch.zeros(
                            1, seq_len_queries, device=queries.device, dtype=torch.float32
                        )
                        cos_queries_scaled, sin_queries_scaled = rotary_emb(dummy_x_queries_scaled, queries_position_ids_scaled)
                        
                        # Apply RoPE to unroped Q/K with scaled cos/sin
                        queries_for_mask: torch.Tensor = apply_rotary_pos_emb_queries(
                            queries_unroped, cos_queries_scaled, sin_queries_scaled
                        )
                        keys_for_mask: torch.Tensor = apply_rotary_pos_emb_keys(
                            keys_unroped, cos_keys_scaled, sin_keys_scaled
                        )
                    else:
                        # No scaling needed: use unroped Q/K directly for mask computation
                        queries_for_mask: torch.Tensor = queries_unroped
                        keys_for_mask: torch.Tensor = keys_unroped
                else:
                    # Tuple case not used, but handle for completeness
                    queries_unroped, keys_unroped = unapply_rotary_pos_emb(
                        queries, keys, cos, sin
                    )
                    # For non-tuple case, use unroped Q/K directly (scaling logic would need cos/sin as tuple)
                    queries_for_mask: torch.Tensor = queries_unroped
                    keys_for_mask: torch.Tensor = keys_unroped
                unroped_used = True
                
                # COMMENTED OUT: Verification check not needed when using unroped Q/K directly
                # q_diff: torch.Tensor = torch.abs(queries - queries_for_mask).max()
                # k_diff: torch.Tensor = torch.abs(keys - keys_for_mask).max()
                # 
                # if q_diff.item() < 1e-6 and k_diff.item() < 1e-6:
                #     # Critical error - unroping failed
                #     pass  # Keep check but don't raise (matches original behavior)
            except Exception as e:
                raise RuntimeError(
                    f"EXTEND_CONTEXT enabled but failed to unrope Q/K at layer {layer_idx}. "
                    f"Original error: {e}"
                ) from e
        
        return queries_for_mask, keys_for_mask, unroped_used

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute research attention mechanism with masking.

        Args:
            module: The attention module
            queries: Query tensor of shape (b, h, sk, d)
            keys: Key tensor of shape (b, h, sq, d)
            values: Value tensor of shape (b, h, sq, d)
            attention_mask: Optional attention mask of shape (b, h, sq, sk)
            scaling: Scaling factor for attention weights
            dropout: Dropout probability
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of attention output and optional attention weights.
        """
        # Create an empty Mask object
        mask_shape: Tuple[int, int, int, int] = (
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            keys.shape[2],
        )
        sparse_attention_mask: Mask = Mask.create_empty_mask(
            mask_shape, dtype=queries.dtype, device=queries.device
        )

        # Optional debug printing (set SPARSE_DEBUG=1 in env to enable)
        debug = os.environ.get("SPARSE_DEBUG")
        layer_idx = kwargs.get("layer_idx", None)
        
        # Prepare unroped Q/K for mask computation if EXTEND_CONTEXT is enabled
        queries_for_mask, keys_for_mask, unroped_used = self._prepare_unroped_qk_for_mask(
            queries, keys, module, sparse_meta_data, kwargs, layer_idx
        )
        extend_context: bool = os.environ.get("EXTEND_CONTEXT", "0").lower() in ("1", "true", "yes")

        # Apply all maskers sequentially, each one on the output of the previous one
        for masker in self.maskers:
            masker_name = getattr(masker, "__class__", type(masker)).__name__
            
            sparse_attention_mask = masker.add_mask(
                keys=keys_for_mask,  # Use unroped keys when EXTEND_CONTEXT=1, roped otherwise
                queries=queries_for_mask,  # Use unroped queries when EXTEND_CONTEXT=1, roped otherwise
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_attention_mask,
                **kwargs,
            )

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={"layer_idx": kwargs["layer_idx"]},
            )
        
        # Compare mask computed with roped vs unroped Q/K if enabled
        # Also supports sanity check: compare roped vs roped (should be identical)
        if MicroMetricLogger().is_metric_enabled("research_mask_roped_vs_unroped"):
            compare_mask_flag: bool = os.environ.get("COMPARE_MASK_ROPED_VS_UNROPED", "0").lower() in ("1", "true", "yes")
            
            # Run comparison if flag is set
            # Case 1: EXTEND_CONTEXT=1 -> Compare unroped mask (already computed) vs roped mask (baseline)
            # Case 2: EXTEND_CONTEXT=0 -> Compare roped mask (already computed) vs roped mask (sanity check, should be identical)
            if compare_mask_flag:
                layer_idx: int = kwargs["layer_idx"]
                # Sample only specific layers to reduce overhead
                sample_layers: List[int] = [15]  # First, middle, and last layers
                
                if layer_idx in sample_layers:
                    # DEBUG: Verify position IDs used for unroping match those used for roping
                    # This is critical - if position IDs don't match, unroping will be incorrect
                    position_ids_q_from_kwargs: Optional[torch.Tensor] = kwargs.get("position_ids")
                    seq_len_keys: int = keys.shape[2]
                    seq_len_queries: int = queries.shape[2]
                    
                    # Expected position IDs for keys (should be [0, 1, 2, ..., seq_len_keys-1] if no reassignment)
                    # Expected position IDs for queries (from kwargs, or [0, 1, 2, ..., seq_len_queries-1])
                    expected_pos_ids_k: torch.Tensor = torch.arange(0, seq_len_keys, device=keys.device, dtype=torch.long)
                    
                    if position_ids_q_from_kwargs is not None:
                        expected_pos_ids_q: torch.Tensor = position_ids_q_from_kwargs
                    else:
                        expected_pos_ids_q: torch.Tensor = torch.arange(0, seq_len_queries, device=queries.device, dtype=torch.long)
                    
                    # If unroped Q/K were used, verify the position IDs that were used for unroping
                    if extend_context and unroped_used:
                        # Check if position IDs in sparse_meta_data match what we expect
                        # (This is where position reassignment might have changed them)
                        pos_ids_in_meta: Optional[torch.Tensor] = sparse_meta_data.get("position_ids_q") if sparse_meta_data else None
                        pos_ids_k_in_meta: Optional[torch.Tensor] = sparse_meta_data.get("position_ids_k") if sparse_meta_data else None
                        
                        # If position reassignment happened, the actual position IDs might be different
                        # We need to check what position IDs were actually used for roping the incoming Q/K
                        # For now, we'll assert that if no reassignment happened, position IDs should match
                        enable_reassignment: bool = os.environ.get("ENABLE_POSITION_REASSIGNMENT", "0").lower() in ("1", "true", "yes")
                        
                        if not enable_reassignment:
                            # No reassignment - position IDs should match expected
                            if position_ids_q_from_kwargs is not None:
                                assert position_ids_q_from_kwargs.shape[1] == seq_len_queries, (
                                    f"Position IDs shape mismatch: position_ids_q.shape[1]={position_ids_q_from_kwargs.shape[1]}, "
                                    f"seq_len_queries={seq_len_queries}"
                                )
                        
                        # # DEBUG: Set breakpoint here to inspect position IDs
                        # import pdb; pdb.set_trace()
                        
                    
                    # Compute baseline mask with roped Q/K
                    mask_shape: Tuple[int, ...] = (
                        queries.shape[0],  # batch_size
                        queries.shape[1],  # num_heads
                        queries.shape[2],  # seq_len_q
                        keys.shape[2],     # seq_len_k
                    )
                    sparse_attention_mask_roped: Mask = Mask.create_empty_mask(
                        shape=mask_shape,
                        dtype=torch.float32,
                        device=queries.device,
                    )
                    
                    # Apply all maskers with roped Q/K (baseline)
                    for masker in self.maskers:
                        sparse_attention_mask_roped = masker.add_mask(
                            keys=keys,  # Use roped keys
                            queries=queries,  # Use roped queries
                            values=values,
                            attention_mask=attention_mask,
                            scaling=scaling,
                            dropout=dropout,
                            sparse_meta_data=sparse_meta_data,
                            previous_mask=sparse_attention_mask_roped,
                            **kwargs,
                        )
                    
                    # Get dense masks for comparison
                    # Note: mask_computed is the mask already computed (could be unroped or roped depending on EXTEND_CONTEXT)
                    # mask_roped is always computed with roped Q/K (baseline)
                    mask_computed: torch.Tensor = sparse_attention_mask.get_dense_mask()  # (batch, num_heads, seq_len_q, seq_len_k)
                    mask_roped: torch.Tensor = sparse_attention_mask_roped.get_dense_mask()  # (batch, num_heads, seq_len_q, seq_len_k)
                    
                    # Determine comparison type for metadata
                    comparison_type: str = "unroped_vs_roped" if (extend_context and unroped_used) else "roped_vs_roped"
                    
                    # Ensure shapes match
                    if mask_computed.shape == mask_roped.shape:
                        batch_size: int = mask_computed.shape[0]
                        num_heads: int = mask_computed.shape[1]
                        seq_len_q: int = mask_computed.shape[2]
                        seq_len_k: int = mask_computed.shape[3]
                        
                        # Sample only 1 head to reduce memory (head 0)
                        sample_head_idx: int = 0
                        
                        # Extract masks for sampled head
                        mask_computed_head: torch.Tensor = mask_computed[0, sample_head_idx, :, :]  # (seq_len_q, seq_len_k)
                        mask_roped_head: torch.Tensor = mask_roped[0, sample_head_idx, :, :]  # (seq_len_q, seq_len_k)
                        
                        # Update sample index tracking (for mask saving)
                        if os.environ.get("SAVE_MASKS", "0").lower() in ("1", "true", "yes"):
                            self._update_sample_idx(seq_len_q, seq_len_k)
                        
                        # Save masks if enabled (only for first sample)
                        if os.environ.get("SAVE_MASKS", "0").lower() in ("1", "true", "yes"):
                            current_sample_idx: int = ResearchAttention._sample_idx
                            self._save_mask(
                                mask=mask_computed_head,
                                mask_type="computed",
                                layer_idx=layer_idx,
                                head_idx=sample_head_idx,
                                seq_len_q=seq_len_q,
                                seq_len_k=seq_len_k,
                                sample_idx=current_sample_idx,
                            )
                            self._save_mask(
                                mask=mask_roped_head,
                                mask_type="roped",
                                layer_idx=layer_idx,
                                head_idx=sample_head_idx,
                                seq_len_q=seq_len_q,
                                seq_len_k=seq_len_k,
                                sample_idx=current_sample_idx,
                            )
                        
                        # Compute comparison metrics
                        with torch.no_grad():
                            # Convert to boolean for set operations
                            mask_computed_bool: torch.Tensor = mask_computed_head > 0
                            mask_roped_bool: torch.Tensor = mask_roped_head > 0
                            
                            # Intersection: keys selected in both
                            intersection: torch.Tensor = mask_computed_bool & mask_roped_bool
                            num_intersection: int = intersection.sum().item()
                            
                            # Union: keys selected in either
                            union: torch.Tensor = mask_computed_bool | mask_roped_bool
                            num_union: int = union.sum().item()
                            
                            # Difference: keys selected in one but not the other
                            only_computed: torch.Tensor = mask_computed_bool & (~mask_roped_bool)
                            only_roped: torch.Tensor = mask_roped_bool & (~mask_computed_bool)
                            num_only_computed: int = only_computed.sum().item()
                            num_only_roped: int = only_roped.sum().item()
                            
                            # Jaccard similarity: intersection / union
                            jaccard: float = (num_intersection / num_union) if num_union > 0 else 0.0
                            
                            # Overlap percentage: intersection / (average of both mask sizes)
                            num_computed: int = mask_computed_bool.sum().item()
                            num_roped: int = mask_roped_bool.sum().item()
                            avg_mask_size: float = (num_computed + num_roped) / 2.0
                            overlap_percentage: float = (num_intersection / avg_mask_size) if avg_mask_size > 0 else 0.0
                            
                            # Total difference count
                            total_diff: int = num_only_computed + num_only_roped
                            
                            # Difference percentage
                            diff_percentage: float = (total_diff / num_union) if num_union > 0 else 0.0
                        
                        # Log comprehensive metrics
                        MicroMetricLogger().log(
                            "research_mask_roped_vs_unroped",
                            {
                                "jaccard_similarity": jaccard,
                                "overlap_percentage": overlap_percentage,
                                "num_intersection": num_intersection,
                                "num_union": num_union,
                                "num_only_computed": num_only_computed,  # Keys only in computed mask (unroped if EXTEND_CONTEXT=1, roped if EXTEND_CONTEXT=0)
                                "num_only_roped": num_only_roped,  # Keys only in roped mask
                                "total_diff": total_diff,
                                "diff_percentage": diff_percentage,
                                "num_computed": num_computed,  # Total keys in computed mask
                                "num_roped": num_roped,  # Total keys in roped mask
                            },
                            metadata={
                                "layer_idx": layer_idx,
                                "head_idx": sample_head_idx,
                                "batch_size": batch_size,
                                "seq_len_q": seq_len_q,
                                "seq_len_k": seq_len_k,
                                "queries_shape": list(queries.shape),
                                "keys_shape": list(keys.shape),
                                "mask_shape": list(mask_computed.shape),
                                "comparison_type": comparison_type,  # "unroped_vs_roped" or "roped_vs_roped"
                                "extend_context": extend_context,
                                "unroped_used": unroped_used,
                            },
                        )
                    
                    # Clean up
                    del sparse_attention_mask_roped, mask_computed, mask_roped
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        # Normal path: use existing get_masked_attention_output
        # Note: Unroped Q/K are used for mask computation above, but normal attention uses roped Q/K
        # Thread config toggles into kwargs (no env flags)
        kwargs.setdefault("pack_k_chunk_translation", getattr(self.sparse_attention_config, "pack_k_chunk_translation", False))
        attention_output, attention_weights = get_masked_attention_output(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            return_attention_weights=True,
            sparse_meta_data=sparse_meta_data,
            **kwargs,
        )

        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            true_attention_output, _ = get_true_attention_output(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
            error = torch.norm(true_attention_output - attention_output) / torch.norm(
                true_attention_output
            )
            MicroMetricLogger().log(
                "research_attention_output_error",
                float(error.item()),
                metadata={"layer_idx": kwargs["layer_idx"]},
            )

        # Log attention weight differences (sparse vs dense) for sampled layers and heads
        # Memory-efficient: only compute for 1-2 layers to avoid OOM
        # import pdb; pdb.set_trace()
        if MicroMetricLogger().is_metric_enabled("research_attention_weight_diff"):
            layer_idx: int = kwargs["layer_idx"]
            # Sample only 1-2 layers to reduce memory usage (compute dense attention is expensive)
            sample_layers: List[int] = [15]  # Just first and middle layer
            
            if layer_idx in sample_layers and attention_weights is not None:
                try:
                    # Get dense attention weights (memory intensive - may fail on large sequences)
                    _, dense_attention_weights = get_true_attention_output(
                        module,
                        queries,
                        keys,
                        values,
                        attention_mask,
                        scaling,
                        dropout,
                        **kwargs,
                    )
                    
                    # Ensure shapes match
                    if dense_attention_weights.shape == attention_weights.shape:
                        # Sample only 1 head to reduce memory (head 0)
                        num_heads: int = attention_weights.shape[1]
                        sample_heads: List[int] = [10]  # Just first head
                        
                        for head_idx in sample_heads:
                            sparse_weights_head: torch.Tensor = attention_weights[:, head_idx, :, :]  # (batch, seq_q, seq_k)
                            dense_weights_head: torch.Tensor = dense_attention_weights[:, head_idx, :, :]
                            
                            # Get shape information for context
                            batch_size: int = attention_weights.shape[0]
                            seq_len_q: int = attention_weights.shape[2]
                            seq_len_k: int = attention_weights.shape[3]
                            queries_shape: Tuple[int, ...] = queries.shape
                            keys_shape: Tuple[int, ...] = keys.shape
                            
                            # Compute differences (move to CPU if needed to save GPU memory)
                            with torch.no_grad():
                                diff: torch.Tensor = torch.abs(sparse_weights_head - dense_weights_head)
                                max_diff: float = diff.max().item()
                                mean_diff: float = diff.mean().item()
                                l2_diff: float = torch.norm(sparse_weights_head - dense_weights_head).item()
                                
                                # Compute relative L2 difference (normalized by dense weights norm)
                                dense_norm: float = torch.norm(dense_weights_head).item()
                                l2_diff_relative: float = (l2_diff / dense_norm) if dense_norm > 0 else 0.0
                            
                            # Log per head with comprehensive metrics and shape context
                            MicroMetricLogger().log(
                                "research_attention_weight_diff",
                                {
                                    "max_diff": max_diff,
                                    "mean_diff": mean_diff,
                                    "l2_diff": l2_diff,
                                    "l2_diff_relative": l2_diff_relative,
                                },
                                metadata={
                                    "layer_idx": layer_idx,
                                    "head_idx": head_idx,
                                    "batch_size": batch_size,
                                    "seq_len_q": seq_len_q,
                                    "seq_len_k": seq_len_k,
                                    "queries_shape": list(queries_shape),
                                    "keys_shape": list(keys_shape),
                                    "attention_shape": list(attention_weights.shape),
                                },
                            )
                    
                    # Clean up dense attention weights immediately to free memory
                    del dense_attention_weights
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    msg_lower = str(e).lower()
                    if "out of memory" in msg_lower or "cuda" in msg_lower:
                        raise RuntimeError(
                            f"[FATAL] research_attention_weight_diff dense comparison failed at layer {layer_idx} due to OOM/CUDA error. "
                            f"This run requires dense-vs-sparse metrics; refusing to continue silently. "
                            f"Action: reduce PREFILL_CHUNK_SIZE, NUM_SAMPLES, or effective sequence length (e.g., set MAX_CONTEXT_LENGTH) and re-run."
                        ) from e
                    # Re-raise non-OOM errors
                    raise

        return attention_output, attention_weights

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "ResearchAttention":
        """Create research attention instance from configuration.

        Args:
            config: Configuration for the research attention mechanism.

        Returns:
            Instance of the research attention mechanism.

        Raises:
            TypeError: If config is not a ResearchAttentionConfig.
        """
        if not isinstance(config, ResearchAttentionConfig):
            raise TypeError(f"Expected ResearchAttentionConfig, got {type(config)}")

        # Create ResearchMasker objects from the configs using the factory method
        maskers: List[ResearchMasker] = []
        for masker_config in config.masker_configs:
            masker: ResearchMasker = ResearchMasker.create_masker_from_config(
                masker_config
            )
            maskers.append(masker)

        return cls(config, maskers)
