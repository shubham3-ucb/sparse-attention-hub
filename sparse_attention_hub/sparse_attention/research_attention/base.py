"""Base classes for research attention mechanisms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import os

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
    compute_rope_cos_sin,
)

MicroMetricLogger.register_metric("research_attention_density", float)
MicroMetricLogger.register_metric("research_attention_output_error", float)


@dataclass
class ResearchAttentionConfig(SparseAttentionConfig):
    """Configuration class for research attention mechanisms."""

    masker_configs: List[MaskerConfig]


class ResearchAttention(SparseAttention):
    """Base class for research attention mechanisms with maskers."""

    maskers: List[ResearchMasker]

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
        
        # Check for EXTEND_CONTEXT flag for unroped mask computation
        extend_context: bool = os.environ.get("EXTEND_CONTEXT", "0").lower() in ("1", "true", "yes")
        
        # For mask computation: use unroped Q/K if EXTEND_CONTEXT is enabled and cos/sin available
        # For actual attention: always use roped Q/K (existing behavior)
        queries_for_mask: torch.Tensor = queries
        keys_for_mask: torch.Tensor = keys
        unroped_used: bool = False
        
        if extend_context:
            # Try to get cos/sin from kwargs first (preferred method if passed directly)
            cos: Optional[torch.Tensor] = kwargs.get("cos")
            sin: Optional[torch.Tensor] = kwargs.get("sin")
            
            # If not in kwargs, try to get from sparse_meta_data or compute from position_ids
            if cos is None or sin is None:
                # First, try to get rotary_emb from sparse_meta_data (stored by adapter)
                rotary_emb: Optional[Any] = sparse_meta_data.get("_rotary_emb")
                position_ids: Optional[torch.Tensor] = kwargs.get("position_ids")
                
                # Important: We need cos/sin for BOTH query and key sequences separately
                # - Queries: use position_ids from kwargs (positions for current chunk)
                # - Keys: use full range [0, seq_len_keys-1] (all positions in KV cache)
                seq_len_keys: int = keys.shape[2]  # Full key sequence length (KV cache)
                seq_len_queries: int = queries.shape[2]  # Query sequence length (chunk)
                
                if rotary_emb is not None:
                    try:
                        # Compute cos/sin for keys: full range [0, seq_len_keys-1]
                        keys_position_ids: torch.Tensor = torch.arange(
                            0, seq_len_keys, device=keys.device, dtype=torch.long
                        ).unsqueeze(0)  # Shape: [1, seq_len_keys]
                        
                        # Compute cos/sin for queries: use position_ids from kwargs if available
                        # Otherwise assume queries start at position 0 (first chunk)
                        if position_ids is not None and position_ids.shape[1] == seq_len_queries:
                            queries_position_ids: torch.Tensor = position_ids  # Shape: [1, seq_len_queries]
                        else:
                            # Fallback: assume queries are at positions [0, seq_len_queries-1]
                            queries_position_ids: torch.Tensor = torch.arange(
                                0, seq_len_queries, device=queries.device, dtype=torch.long
                            ).unsqueeze(0)
                        
                        # Llama-style: rotary_emb(x, position_ids)
                        # Compute for keys
                        dummy_x_keys: torch.Tensor = torch.zeros(
                            1, seq_len_keys, device=keys.device, dtype=torch.float32
                        )
                        cos_keys, sin_keys = rotary_emb(dummy_x_keys, keys_position_ids)
                        
                        # Compute for queries
                        dummy_x_queries: torch.Tensor = torch.zeros(
                            1, seq_len_queries, device=queries.device, dtype=torch.float32
                        )
                        cos_queries, sin_queries = rotary_emb(dummy_x_queries, queries_position_ids)
                        
                        # Store both - we'll use them separately in unapply
                        cos = (cos_queries, cos_keys)
                        sin = (sin_queries, sin_keys)
                        
                        if debug:
                            msg = (
                                f"[extend_context] layer={layer_idx} computed cos/sin: "
                                f"queries cos/sin.shape={cos_queries.shape}, keys cos/sin.shape={cos_keys.shape}, "
                                f"queries.shape={queries.shape}, keys.shape={keys.shape}"
                            )
                            print(msg, flush=True)
                            try:
                                log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                with open(log_path, "a") as fh:
                                    fh.write(msg + "\n")
                            except Exception:
                                pass
                    except (TypeError, ValueError):
                        # Try alternative signature
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
                            
                            if debug:
                                msg = (
                                    f"[extend_context] layer={layer_idx} computed cos/sin (alt signature): "
                                    f"queries cos/sin.shape={cos_queries.shape}, keys cos/sin.shape={cos_keys.shape}"
                                )
                                print(msg, flush=True)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to compute cos/sin from rotary_emb in sparse_meta_data: {e}"
                            ) from e
                elif rotary_emb is None:
                    # Fallback: try to compute from module's rotary_emb
                    # Still need to compute for full key range
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
                        if debug:
                            msg = (
                                f"[extend_context] layer={layer_idx} computed cos/sin for full key range "
                                f"using module search. cos.shape={cos.shape}, sin.shape={sin.shape}"
                            )
                            print(msg, flush=True)
                            try:
                                log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                with open(log_path, "a") as fh:
                                    fh.write(msg + "\n")
                            except Exception:
                                pass
                    except Exception as e:
                        error_msg = (
                            f"[extend_context] ERROR: layer={layer_idx} failed to compute cos/sin. "
                            f"Exception: {e}. "
                            f"seq_len_keys={seq_len_keys}, "
                            f"module type={type(module)}, "
                            f"has rotary_emb={hasattr(module, 'rotary_emb')}, "
                            f"rotary_emb in sparse_meta_data={'_rotary_emb' in sparse_meta_data}"
                        )
                        print(error_msg, flush=True)
                        try:
                            log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            with open(log_path, "a") as fh:
                                fh.write(error_msg + "\n")
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"EXTEND_CONTEXT enabled but failed to compute cos/sin at layer {layer_idx}. "
                            f"Original error: {e}"
                        ) from e
                else:
                    available_kwargs = list(kwargs.keys())
                    error_msg = (
                        f"[extend_context] ERROR: layer={layer_idx} EXTEND_CONTEXT enabled but rotary_emb not found. "
                        f"Available kwargs: {available_kwargs}, "
                        f"rotary_emb in sparse_meta_data={'_rotary_emb' in sparse_meta_data}. "
                        f"Cannot compute cos/sin for unroping."
                    )
                    print(error_msg, flush=True)
                    try:
                        log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a") as fh:
                            fh.write(error_msg + "\n")
                    except Exception:
                        pass
                    raise ValueError(
                        f"EXTEND_CONTEXT enabled but rotary_emb not available "
                        f"at layer {layer_idx}. Available kwargs: {available_kwargs}."
                    )
            
            # Now we have cos/sin - unrope Q/K for mask computation
            # Handle case where cos/sin are tuples (separate for queries and keys)
            try:
                if isinstance(cos, tuple) and isinstance(sin, tuple):
                    # Separate cos/sin for queries and keys
                    cos_queries, cos_keys = cos
                    sin_queries, sin_keys = sin
                    # Unrope queries and keys separately
                    queries_for_mask: torch.Tensor = unapply_rotary_pos_emb_queries(
                        queries, cos_queries, sin_queries
                    )
                    keys_for_mask: torch.Tensor = unapply_rotary_pos_emb_keys(
                        keys, cos_keys, sin_keys
                    )
                else:
                    # Single cos/sin (should match both - for backward compatibility)
                    queries_for_mask, keys_for_mask = unapply_rotary_pos_emb(
                        queries, keys, cos, sin
                    )
                unroped_used = True
                
                # VERIFICATION: Check that unroped Q/K are actually different from roped Q/K
                # This ensures unroping worked correctly and we're not silently falling back
                q_diff: torch.Tensor = torch.abs(queries - queries_for_mask).max()
                k_diff: torch.Tensor = torch.abs(keys - keys_for_mask).max()
                
                if q_diff.item() < 1e-6 and k_diff.item() < 1e-6:
                    error_msg = (
                        f"[extend_context] CRITICAL ERROR: layer={layer_idx} unroped Q/K are IDENTICAL to roped Q/K! "
                        f"q_diff.max()={q_diff.item()}, k_diff.max()={k_diff.item()}. "
                        f"This means unroping FAILED or cos/sin are wrong. "
                        f"ABORTING - this is a critical failure!"
                    )
                    print(error_msg, flush=True)
                    try:
                        log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a") as fh:
                            fh.write(error_msg + "\n")
                    except Exception:
                        pass
                    # raise RuntimeError(
                    #     f"EXTEND_CONTEXT: Unroped Q/K identical to roped Q/K at layer {layer_idx}. "
                    #     f"Unroping verification failed - q_diff={q_diff.item()}, k_diff={k_diff.item()}"
                    # )
                
                msg = (
                    f"[extend_context] layer={layer_idx} unroped Q/K for mask computation. "
                    f"VERIFICATION: q_diff.max()={q_diff.item():.6f}, k_diff.max()={k_diff.item():.6f} "
                    f"(unroped tensors are different from roped - GOOD)"
                )
                print(msg, flush=True)
                try:
                    log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "a") as fh:
                        fh.write(msg + "\n")
                except Exception:
                    pass
            except Exception as e:
                error_msg = (
                    f"[extend_context] ERROR: layer={layer_idx} failed to unrope Q/K. "
                    f"Exception: {e}. "
                    f"queries.shape={queries.shape}, keys.shape={keys.shape}, "
                    f"cos.shape={cos.shape if cos is not None else None}, "
                    f"sin.shape={sin.shape if sin is not None else None}"
                )
                print(error_msg, flush=True)
                try:
                    log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "a") as fh:
                        fh.write(error_msg + "\n")
                except Exception:
                    pass
                raise RuntimeError(
                    f"EXTEND_CONTEXT enabled but failed to unrope Q/K at layer {layer_idx}. "
                    f"Original error: {e}"
                ) from e
        
        if debug:
            header = f"[sparse] layer={layer_idx} queries={queries.shape} keys={keys.shape} init_mask_density={sparse_attention_mask.get_density():.6f}"
            if unroped_used:
                header += " unroped_mask=True"
            print(header, flush=True)
            try:
                log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as fh:
                    fh.write(header + "\n")
            except Exception:
                pass

        # Apply all maskers sequentially, each one on the output of the previous one
        # Use unroped Q/K for mask computation if EXTEND_CONTEXT is enabled
        for masker in self.maskers:
            masker_name = getattr(masker, "__class__", type(masker)).__name__
            
            # VERIFICATION: Ensure maskers receive unroped Q/K when EXTEND_CONTEXT is enabled
            # CRITICAL: Top-K selection must use unroped Q/K to get position-agnostic similarity
            if extend_context and unroped_used:
                # Verify we're passing unroped tensors (not roped)
                q_check: torch.Tensor = torch.abs(queries_for_mask - queries).max()
                k_check: torch.Tensor = torch.abs(keys_for_mask - keys).max()
                if q_check.item() < 1e-6 or k_check.item() < 1e-6:
                    error_msg = (
                        f"[extend_context] CRITICAL: layer={layer_idx} masker={masker_name} "
                        f"about to receive Q/K that are IDENTICAL to roped! "
                        f"q_check={q_check.item()}, k_check={k_check.item()}. "
                        f"This means unroped Q/K were not used for top-K selection - ABORTING! "
                        f"Top-K MUST be selected using unroped Q/K to get position-agnostic similarity."
                    )
                    print(error_msg, flush=True)
                    # raise RuntimeError(
                    #     f"EXTEND_CONTEXT: Masker {masker_name} would receive roped Q/K instead of unroped at layer {layer_idx}. "
                    #     f"Top-K selection requires unroped Q/K for correct position-agnostic similarity."
                    # )
            
            sparse_attention_mask = masker.add_mask(
                keys=keys_for_mask,  # Use unroped keys for mask computation
                queries=queries_for_mask,  # Use unroped queries for mask computation
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                previous_mask=sparse_attention_mask,
                **kwargs,
            )
            if debug:
                try:
                    dens = sparse_attention_mask.get_density()
                except Exception:
                    dens = None
                line = f"[sparse] layer={layer_idx} masker={masker_name} mask_density={dens}"
                print(line, flush=True)
                try:
                    log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                    with open(log_path, "a") as fh:
                        fh.write(line + "\n")
                except Exception:
                    pass

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={"layer_idx": kwargs["layer_idx"]},
            )
            
        # Normal path: use existing get_masked_attention_output
        # Note: Unroped Q/K are used for mask computation above, but normal attention uses roped Q/K
        if debug:
            final_line = f"[sparse] layer={layer_idx} final_mask_density={sparse_attention_mask.get_density():.6f} -- computing masked attention"
            print(final_line, flush=True)
            try:
                log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                with open(log_path, "a") as fh:
                    fh.write(final_line + "\n")
            except Exception:
                pass
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
        if debug:
            try:
                out_line = f"[sparse] layer={layer_idx} attention_output={attention_output.shape} attention_weights={None if attention_weights is None else attention_weights.shape}"
                print(out_line, flush=True)
                try:
                    log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "sparse_debug.log"))
                    with open(log_path, "a") as fh:
                        fh.write(out_line + "\n")
                except Exception:
                    pass
            except Exception:
                pass

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
