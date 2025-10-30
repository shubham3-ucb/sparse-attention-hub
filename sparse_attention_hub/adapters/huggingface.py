"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import numpy as np
import os
from tqdm import tqdm
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse
from .model_servers.huggingface import ModelServerHF

INT_MAX = 2**31 - 1


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None
        # create model and tokenizer

        model_server = ModelServerHF()
        self.model = model_server.get_model(
            self.model_name, self.device, self.model_kwargs
        )
        self.tokenizer = model_server.get_tokenizer(
            self.model_name, self.tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    @staticmethod
    def _get_prefill_log_path() -> str:
        """Return the path to the prefill log file.

        Priority: SPARSE_LOG_PATH env var -> OUTPUT_DIR env var (hf_prefill.log) -> ./output_test_sparse/hf_prefill.log
        """
        base = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output_test_sparse"))
        return os.environ.get("SPARSE_LOG_PATH", os.path.join(base, "hf_prefill.log"))

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)
        max_new_tokens: int = generation_kwargs.get("max_new_tokens", INT_MAX)
        print(
            " Processing request with max_context_length: ",
            max_context_length,
            " and max_new_tokens: ",
            max_new_tokens,
            flush=True,
        )

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context
        answer_prefix: str = request.answer_prefix

        context, questions = self._preprocess_context_and_questions(
            context, questions, answer_prefix
        )

        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length
        if self.device is not None:
            context_tokens = context_tokens.to(self.device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                question_tokens = self.tokenizer.encode(question, return_tensors="pt")
                if self.device is not None:
                    question_tokens = question_tokens.to(self.device)

                # Optional prefill logging (enable with SPARSE_DEBUG=1)
                if os.environ.get("SPARSE_DEBUG"):
                    try:
                        before = f"[prefill] before model call context_tokens={context_tokens.shape} sparse_meta_keys={list(sparse_meta_data.keys())}"
                        print(before, flush=True)
                        log_path = self._get_prefill_log_path()
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a") as fh:
                            fh.write(before + "\n")
                    except Exception:
                        pass
                
                # import pdb; pdb.set_trace()
                # Prefill: support optional chunked prefill for dense mode (verify equality)
                prefill_chunk_size = request_kwargs.get("prefill_chunk_size") or os.environ.get("PREFILL_CHUNK_SIZE")
                assert_chunk_equals_full = request_kwargs.get("assert_chunked_equals_full", True)

                if self._sparse_attention_available:
                    # mark sparse prefill
                    if os.environ.get("SPARSE_DEBUG"):
                        marker = "%% working on sparse prefill"
                        print(marker, flush=True)
                        try:
                            log_path = self._get_prefill_log_path()
                            with open(log_path, "a") as fh:
                                fh.write(marker + "\n")
                        except Exception:
                            pass

                    with self.enable_sparse_mode():
                        # If a prefill_chunk_size is provided, perform chunked sparse prefill (opt-in)
                        if prefill_chunk_size is not None:
                            try:
                                chunk_size = int(prefill_chunk_size)
                            except Exception:
                                chunk_size = None
                        else:
                            chunk_size = None

                        if chunk_size is None or chunk_size <= 0:
                            # single-call sparse prefill (default)
                            context_outputs = self.model(
                                context_tokens,
                                past_key_values=None,
                                use_cache=True,
                                sparse_meta_data=sparse_meta_data,
                            )
                        else:
                            total_len = context_tokens.shape[1]
                            past = None
                            chunked_outputs = None
                            seq_lens = []
                            meta_keys_history = []
                            log_path = self._get_prefill_log_path()
                            for i in range(0, total_len, chunk_size):
                                chunk = context_tokens[:, i : i + chunk_size]
                                if os.environ.get("SPARSE_DEBUG"):
                                    print(f"[prefill] sparse chunk idx={i} chunk_shape={chunk.shape}", flush=True)
                                    try:
                                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                        with open(log_path, "a") as fh:
                                            fh.write(f"[prefill] sparse chunk idx={i} chunk_shape={chunk.shape}\n")
                                    except Exception:
                                        pass
                                try:
                                    pos_ids = torch.arange(i, i + chunk.shape[1], device=chunk.device).unsqueeze(0)
                                    chunked_outputs = self.model(
                                        chunk,
                                        past_key_values=(None if past is None else past),
                                        position_ids=pos_ids,
                                        use_cache=True,
                                        sparse_meta_data=sparse_meta_data,
                                    )
                                except Exception:
                                    chunked_outputs = self.model(
                                        chunk,
                                        past_key_values=(None if past is None else past),
                                        use_cache=True,
                                        sparse_meta_data=sparse_meta_data,
                                    )
                                past = getattr(chunked_outputs, "past_key_values", None)
                                # record seq len
                                seq_len = None
                                try:
                                    if past is not None:
                                        sample = past[0][0]
                                        seq_len = sample.shape[-2]
                                except Exception:
                                    seq_len = None
                                seq_lens.append(seq_len)
                                meta_keys_history.append(list(sparse_meta_data.keys()))

                            context_outputs = chunked_outputs

                            # Sanity checks: seq_lens should be non-decreasing and final == total_len
                            ok_seq = True
                            try:
                                # filter None
                                numeric = [s for s in seq_lens if s is not None]
                                if len(numeric) == 0:
                                    ok_seq = False
                                else:
                                    if numeric[-1] != total_len:
                                        ok_seq = False
                                    for a, b in zip(numeric, numeric[1:]):
                                        if b < a:
                                            ok_seq = False
                                            break
                            except Exception:
                                ok_seq = False

                            # log results
                            if os.environ.get("SPARSE_DEBUG"):
                                try:
                                    smsg = f"[prefill] sparse chunked seq_lens={seq_lens}"
                                    print(smsg, flush=True)
                                    with open(log_path, "a") as fh:
                                        fh.write(smsg + "\n")
                                    meta_msg = f"[prefill] sparse_meta_keys_progress={meta_keys_history}"
                                    print(meta_msg, flush=True)
                                    with open(log_path, "a") as fh:
                                        fh.write(meta_msg + "\n")
                                except Exception:
                                    pass

                            if not ok_seq:
                                msg = f"[prefill] SPARSE CHUNK SANITY FAILED: seq_lens={seq_lens} total_len={total_len}"
                                print(msg, flush=True)
                                try:
                                    with open(log_path, "a") as fh:
                                        fh.write(msg + "\n")
                                except Exception:
                                    pass
                                raise AssertionError(msg)
                            else:
                                msg = f"[prefill] SPARSE CHUNK SANITY OK: final_seq={seq_lens[-1]} total_len={total_len}"
                                if os.environ.get("SPARSE_DEBUG"):
                                    print(msg, flush=True)
                                    try:
                                        with open(log_path, "a") as fh:
                                            fh.write(msg + "\n")
                                    except Exception:
                                        pass
                else:
                    # mark dense prefill
                    if os.environ.get("SPARSE_DEBUG"):
                        marker = "%% working on dense prefill"
                        print(marker, flush=True)
                        try:
                            log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "hf_prefill.log"))
                            with open(log_path, "a") as fh:
                                fh.write(marker + "\n")
                        except Exception:
                            pass

                    # If a chunk size is provided, perform chunked prefill (dense-only) and optionally assert equality
                    if prefill_chunk_size is not None:
                        try:
                            chunk_size = int(prefill_chunk_size)
                        except Exception:
                            chunk_size = None

                    else:
                        chunk_size = None

                    def _get_seq_from_past(past_obj: Any) -> Optional[int]:
                        if past_obj is None:
                            return None
                        if hasattr(past_obj, "get_seq_length"):
                            try:
                                return past_obj.get_seq_length()
                            except Exception:
                                pass
                        try:
                            sample = past_obj[0][0]
                            return sample.shape[-2]
                        except Exception:
                            return None

                    def _compare_past(p1: Any, p2: Any) -> None:
                        # Compare structure and tensor equality of past_key_values
                        if (p1 is None) != (p2 is None):
                            raise AssertionError("past_key_values presence mismatch")
                        if p1 is None and p2 is None:
                            return
                        if len(p1) != len(p2):
                            raise AssertionError(f"num layers mismatch: {len(p1)} vs {len(p2)}")
                        for li, (l1, l2) in enumerate(zip(p1, p2)):
                            if len(l1) != len(l2):
                                raise AssertionError(f"layer {li} tuple length mismatch")
                            for ti, (t1, t2) in enumerate(zip(l1, l2)):
                                if t1.shape != t2.shape:
                                    raise AssertionError(f"tensor shape mismatch at layer {li} tensor {ti}: {t1.shape} vs {t2.shape}")

                                # cast to float for safe comparison (handles bfloat16/float16 kernels)
                                t1f = t1.float()
                                t2f = t2.float()
                                if not torch.allclose(t1f, t2f, atol=1e-3, rtol=1e-3):
                                    # compute helpful diagnostics
                                    try:
                                        diff = (t1f - t2f).abs()
                                        max_diff = float(diff.max())
                                        mean_diff = float(diff.mean())
                                        numel = diff.numel()
                                        n_diffs = int((diff > 1e-3).sum().item()) if numel > 0 else 0
                                        # argmax index
                                        argmax = int(diff.view(-1).argmax().item()) if numel > 0 else None
                                        if argmax is not None:
                                            try:
                                                idx = list(np.unravel_index(argmax, diff.shape))
                                            except Exception:
                                                idx = [argmax]
                                        else:
                                            idx = None
                                        # sample first few differing values
                                        flat_diff = diff.view(-1)
                                        topk = 8
                                        vals = flat_diff[flat_diff.argsort(descending=True)[:topk]].cpu().tolist() if numel > 0 else []
                                    except Exception:
                                        max_diff = None
                                        mean_diff = None
                                        n_diffs = None
                                        idx = None
                                        vals = []

                                    # print diagnostics to both stdout and prefill log if possible
                                    msg = (
                                        f"tensor values differ at layer {li} tensor {ti} | shape={t1.shape} | "
                                        f"max_diff={max_diff} mean_diff={mean_diff} num_diffs_gt_tol={n_diffs} argmax_index={idx} sample_top_diffs={vals}"
                                    )
                                    print("[prefill] CHUNK DIFF:", msg, flush=True)
                                    try:
                                        log_path = self._get_prefill_log_path()
                                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                                        with open(log_path, "a") as fh:
                                            fh.write("[prefill] CHUNK DIFF: " + msg + "\n")
                                    except Exception:
                                        pass

                                    # raise with short message; diagnostics are in log
                                    raise AssertionError(f"tensor values differ at layer {li} tensor {ti} (max_diff={max_diff})")

                    if chunk_size is None or chunk_size <= 0:
                        # normal single-call dense prefill
                        context_outputs = self.model(
                            context_tokens,
                            past_key_values=None,
                            use_cache=True,
                            sparse_meta_data=sparse_meta_data,
                        )
                    else:
                        # --- optional: compute full_outputs for assertion ---
                        full_outputs = None
                        if assert_chunk_equals_full:
                            # compute full outputs with explicit position_ids for determinism
                            total_len = context_tokens.shape[1]
                            try:
                                position_ids_full = torch.arange(0, total_len, device=context_tokens.device).unsqueeze(0)
                                full_outputs = self.model(
                                    context_tokens,
                                    past_key_values=None,
                                    position_ids=position_ids_full,
                                    use_cache=True,
                                    sparse_meta_data=sparse_meta_data,
                                )
                            except Exception:
                                # fallback if model doesn't accept explicit position_ids
                                full_outputs = self.model(
                                    context_tokens,
                                    past_key_values=None,
                                    use_cache=True,
                                    sparse_meta_data=sparse_meta_data,
                                )

                        # chunked prefill loop
                        total_len = context_tokens.shape[1]
                        past = None
                        chunked_outputs = None
                        if os.environ.get("SPARSE_DEBUG"):
                            print(f"[prefill] chunked prefill chunk_size={chunk_size} total_len={total_len}", flush=True)
                        for i in range(0, total_len, chunk_size):
                            chunk = context_tokens[:, i : i + chunk_size]
                            if os.environ.get("SPARSE_DEBUG"):
                                print(f"[prefill] chunk idx={i} chunk_shape={chunk.shape}", flush=True)
                            # pass explicit position_ids for each chunk to ensure absolute positions
                            try:
                                pos_ids = torch.arange(i, i + chunk.shape[1], device=chunk.device).unsqueeze(0)
                                chunked_outputs = self.model(
                                    chunk,
                                    past_key_values=(None if past is None else past),
                                    position_ids=pos_ids,
                                    use_cache=True,
                                    sparse_meta_data=sparse_meta_data,
                                )
                            except Exception:
                                chunked_outputs = self.model(
                                    chunk,
                                    past_key_values=(None if past is None else past),
                                    use_cache=True,
                                    sparse_meta_data=sparse_meta_data,
                                )
                            past = getattr(chunked_outputs, "past_key_values", None)

                        # optional assertion against full_outputs
                        if assert_chunk_equals_full and full_outputs is not None:
                            try:
                                _compare_past(getattr(full_outputs, "past_key_values", None), getattr(chunked_outputs, "past_key_values", None))
                                # also compare last-token logits with looser tolerance and cast to float
                                if hasattr(full_outputs, "logits") and hasattr(chunked_outputs, "logits"):
                                    f_last = full_outputs.logits[:, -1].float()
                                    c_last = chunked_outputs.logits[:, -1].float()
                                    if f_last.shape != c_last.shape or not torch.allclose(f_last, c_last, atol=1e-3, rtol=1e-3):
                                        raise AssertionError("final logits differ between full and chunked prefill")
                                # success: print/log OK for visibility when debug enabled
                                if os.environ.get("SPARSE_DEBUG"):
                                    ok_msg = f"[prefill] CHUNK ASSERT OK: chunk_size={chunk_size} total_len={total_len}"
                                    print(ok_msg, flush=True)
                                    try:
                                        log_path = self._get_prefill_log_path()
                                        with open(log_path, "a") as fh:
                                            fh.write(ok_msg + "\n")
                                    except Exception:
                                        pass
                            except AssertionError as e:
                                # If running in reduced precision (bfloat16/float16), convert this hard assert into a warning
                                try:
                                    dtype = getattr(self, "torch_dtype", None)
                                except Exception:
                                    dtype = None

                                is_low_precision = dtype in (torch.bfloat16, torch.float16)
                                if is_low_precision:
                                    warn_msg = f"[prefill] CHUNK ASSERT WARNING (non-fatal, low-precision): {e}"
                                    print(warn_msg, flush=True)
                                    try:
                                        log_path = os.environ.get("SPARSE_LOG_PATH", os.path.join(os.getcwd(), "output_test_sparse", "hf_prefill.log"))
                                        with open(log_path, "a") as fh:
                                            fh.write(warn_msg + "\n")
                                    except Exception:
                                        pass
                                    # continue without raising
                                else:
                                    print(f"[prefill] CHUNK ASSERT FAILED: {e}", flush=True)
                                    raise

                        context_outputs = chunked_outputs

                # import pdb; pdb.set_trace()
                if os.environ.get("SPARSE_DEBUG"):
                    try:
                        # try to get seq length from past_key_values
                        seq_len = None
                        pk = getattr(context_outputs, "past_key_values", None)
                        if pk is not None:
                            try:
                                sample = pk[0][0]
                                seq_len = sample.shape[-2] if hasattr(sample, "shape") and len(sample.shape) >= 2 else None
                            except Exception:
                                seq_len = None
                        after = f"[prefill] after model call past_kv_seq_len={seq_len} sparse_meta_keys={list(sparse_meta_data.keys())}"
                        print(after, flush=True)
                        with open(log_path, "a") as fh:
                            fh.write(after + "\n")
                    except Exception:
                        pass

                if self._sparse_attention_available:
                    with self.enable_sparse_mode():
                        response_text = self._generate_response(
                            question_tokens,
                            context_outputs,
                            sparse_meta_data,
                            generation_kwargs,
                            **kwargs,
                        )
                        responses.append(response_text)
                else:
                    # Dense-only mode: process questions with dense attention
                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )
                    responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str], answer_prefix: str
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        new_context = context.split(self.random_separator)[0]
        new_questions = [
            question + context.split(self.random_separator)[1] + answer_prefix
            for question in questions
        ]
        return new_context, new_questions

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ):
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response using greedy decoding based on kvpress pipeline approach.

        Args:
            question_tokens: The tokenized question
            context_outputs: The model outputs from processing the context
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text response

        TODO:
            move to huggingface genera`te() to leverage all possible generations
            pass generation_kwargs appropriately
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)  # type: ignore
        context_length: int = context_outputs.past_key_values.get_seq_length()

        position_ids = torch.arange(
            context_length,
            context_length + question_tokens.shape[1],
            device=self.model.device,
        ).unsqueeze(0)

        with torch.no_grad():
            question_outputs = self.model(
                input_ids=question_tokens,
                past_key_values=context_outputs.past_key_values,
                position_ids=position_ids,
                num_logits_to_keep=1,
                sparse_meta_data=sparse_meta_data,
            )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [question_outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in tqdm(range(max_new_tokens - 1), disable=(max_new_tokens < 1000)):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=context_outputs.past_key_values,
                    position_ids=position_ids + i,
                    sparse_meta_data=sparse_meta_data,
                )
                # TODO: support other forms of decoding
                new_id = outputs.logits[0, -1].argmax()
                generated_ids.append(new_id)

                if new_id.item() in should_stop_token_ids:
                    break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )
        return answer
