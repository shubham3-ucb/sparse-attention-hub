#!/usr/bin/env python3
"""Minimal benchmark runner: runs N HotpotQA samples (dense vs sparse) and
writes results under ``output_test_sparse/``. Designed to be small and
reproducible for local experiments.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))

from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.adapters.base import Request
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from benchmark.benchmark_registry import create_benchmark_instance


def run_example(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_context_length: Optional[int] = None,
    max_new_tokens: int = 8,
    num_samples: int = 1,
) -> None:
    """Run num_samples HotpotQA examples using dense and sparse attention.

    Outputs are written to ``./output_test_sparse/``.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output_test_sparse"))
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("Xnhyacinth/LongBench", "hotpotqa", split="test")
    ds = ds.select(list(range(min(num_samples, len(ds)))))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    # Use bfloat16 for memory efficiency; switch to float32 only for strict debugging
    model_kwargs = {"torch_dtype": torch.bfloat16}
    tokenizer_kwargs = {"padding_side": "left"}
    if hf_token:
        model_kwargs["use_auth_token"] = hf_token
        tokenizer_kwargs["use_auth_token"] = hf_token
        model_kwargs.setdefault("trust_remote_code", True)

    # Instantiate adapters
    adapter_dense = ModelAdapterHF(model_name, None, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs, device=device)
    # Sparse config: OracleTopK (top 5% attention scores) + Sink + Local
    # DEBUG: Set heavy_size=1.0 to create full mask and disable repositioning
    # This helps isolate if repositioning is causing the repetitive generation issue
    # When heavy_size >= seq_len_keys, full mask is created and repositioning is skipped
    heavy_size = float(os.environ.get("ORACLE_TOPK_HEAVY_SIZE", "0.1"))  # Default to 1.0 for debugging
    sparse_cfg = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=heavy_size),
        ]
    )
    print(f"[DEBUG] OracleTopK heavy_size={heavy_size} (set ORACLE_TOPK_HEAVY_SIZE env var to override)")
    adapter_sparse = ModelAdapterHF(model_name, sparse_cfg, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs, device=device)

    # Best-effort: enable FlashAttention hooks if present on model
    for mdl in (adapter_dense.model, adapter_sparse.model):
        try:
            cfg = getattr(mdl, "config", None)
            if cfg is not None:
                if hasattr(cfg, "use_flash_attention"):
                    cfg.use_flash_attention = True
                elif hasattr(cfg, "use_flash_attn"):
                    cfg.use_flash_attn = True
                elif hasattr(cfg, "attn_impl"):
                    cfg.attn_impl = "flash_attn"
            if hasattr(mdl, "enable_flash_attn"):
                mdl.enable_flash_attn()
            if hasattr(mdl, "enable_flash_attention"):
                mdl.enable_flash_attention()
        except Exception:
            pass

    generation_kwargs = {"max_new_tokens": max_new_tokens}
    # request_kwargs controls truncation; leave empty for full-context unless a cap is provided
    request_kwargs = {}
    if os.environ.get("MAX_CONTEXT_LENGTH"):
        try:
            request_kwargs["max_context_length"] = int(os.environ["MAX_CONTEXT_LENGTH"])
        except Exception:
            pass
    elif max_context_length is not None:
        request_kwargs["max_context_length"] = int(max_context_length)
    
    # Allow truncating from middle for faster experiments
    if os.environ.get("TRUNCATE_FROM_MIDDLE", "").lower() in ("1", "true", "yes"):
        request_kwargs["truncate_from_middle"] = True
        print(f"[DEBUG] Truncating from middle enabled (set TRUNCATE_FROM_MIDDLE=1)")

    # Header
    try:
        tokenizer_max = getattr(adapter_dense.tokenizer, "model_max_length", None)
        model_max = getattr(getattr(adapter_dense.model, "config", None), "max_position_embeddings", None)
    except Exception:
        tokenizer_max = model_max = None

    print(f"model={model_name} device={device} tokenizer_max={tokenizer_max} model_max_pos={model_max}")
    print(f"generation_kwargs={generation_kwargs} request_kwargs={request_kwargs}")

    results = {"dense": [], "sparse": []}
    rows = []

    for i, sample in enumerate(ds):
        context = sample.get("context", "")
        question = sample.get("question", "")
        if not context or not question:
            continue
        req = Request(context=context, questions=question, answer_prefix=sample.get("answer_prefix", "Answer: "))

        t0 = time.time()
        r_dense = adapter_dense.process_request(req, generation_kwargs, request_kwargs)
        t1 = time.time()
        dense_out = r_dense.responses
        results["dense"].append({"response": dense_out, "elapsed_s": t1 - t0})

        t0 = time.time()
        r_sparse = adapter_sparse.process_request(req, generation_kwargs, request_kwargs)
        t1 = time.time()
        sparse_out = r_sparse.responses
        results["sparse"].append({"response": sparse_out, "elapsed_s": t1 - t0})

        # import pdb; pdb.set_trace()
        # Free cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        rows.append({
            "context": context,
            "question": question,
            "predicted_answer": dense_out,
            "elapsed_s": results["dense"][-1]["elapsed_s"],
            "answers": sample.get("answers", None),
            "task": "hotpotqa",
            "method": "dense",
            "all_classes": sample.get("all_classes", []),
        })
        rows.append({
            "context": context,
            "question": question,
            "predicted_answer": sparse_out,
            "elapsed_s": results["sparse"][-1]["elapsed_s"],
            "answers": sample.get("answers", None),
            "task": "hotpotqa",
            "method": "sparse",
            "all_classes": sample.get("all_classes", []),
        })

    # Save outputs
    with open(os.path.join(out_dir, "test_sparse_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    # Eval
    try:
        longbench = create_benchmark_instance("longbench", subsets=["hotpotqa"])
        metrics = longbench.post_run_evaluate(df)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        metrics = None

    # Per-method evaluation and cross-check
    try:
        comp_dir = os.path.join(out_dir, "comparison_results")
        os.makedirs(comp_dir, exist_ok=True)

        df_dense = df[df.method == "dense"].copy()
        df_sparse = df[df.method == "sparse"].copy()

        dense_metrics = None
        sparse_metrics = None

        if not df_dense.empty:
            try:
                dense_metrics = longbench.post_run_evaluate(df_dense)
                with open(os.path.join(comp_dir, "dense_metrics.json"), "w") as f:
                    json.dump(dense_metrics, f, indent=2)
                print("Dense metrics:", dense_metrics)
            except Exception as e:
                print("Failed to compute dense metrics:", e)

        if not df_sparse.empty:
            try:
                sparse_metrics = longbench.post_run_evaluate(df_sparse)
                with open(os.path.join(comp_dir, "sparse_metrics.json"), "w") as f:
                    json.dump(sparse_metrics, f, indent=2)
                print("Sparse metrics:", sparse_metrics)
            except Exception as e:
                print("Failed to compute sparse metrics:", e)

        # Cross-check: weighted average of per-method overall_score should match combined overall_score
        try:
            if metrics is not None and dense_metrics is not None and sparse_metrics is not None:
                n_dense = int((df.method == "dense").sum())
                n_sparse = int((df.method == "sparse").sum())
                combined_weighted = (
                    dense_metrics.get("overall_score", 0) * n_dense
                    + sparse_metrics.get("overall_score", 0) * n_sparse
                ) / (n_dense + n_sparse)
                combined_actual = metrics.get("overall_score", None)
                ok = abs(combined_weighted - combined_actual) < 1e-6 if combined_actual is not None else False
                print(f"Combined overall_score from per-method weighted avg: {combined_weighted}")
                print(f"Combined overall_score reported: {combined_actual}")
                print("Per-method cross-check match:", ok)

        except Exception as e:
            print("Failed to perform cross-check:", e)

    except Exception as e:
        print("Per-method evaluation failed:", e)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model_name": model_name, "generation_kwargs": generation_kwargs, "request_kwargs": request_kwargs}, f, indent=2)

    # Comparison artifacts
    comp = os.path.join(out_dir, "comparison_results")
    os.makedirs(comp, exist_ok=True)
    df[df.method == "dense"].to_csv(os.path.join(comp, "dense_results.csv"), index=False)
    df[df.method == "sparse"].to_csv(os.path.join(comp, "sparse_results.csv"), index=False)
    summary = {
        "dense": {"n": int((df.method == "dense").sum()), "mean_elapsed_s": float(df[df.method == "dense"]["elapsed_s"].mean()) if (df.method == "dense").any() else None},
        "sparse": {"n": int((df.method == "sparse").sum()), "mean_elapsed_s": float(df[df.method == "sparse"]["elapsed_s"].mean()) if (df.method == "sparse").any() else None},
    }
    if metrics is not None:
        summary["metrics"] = metrics
    with open(os.path.join(comp, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Allow overriding number of samples via env var for quick experiments
    ns = os.environ.get("NUM_SAMPLES")
    try:
        num_samples = int(ns) if ns is not None else 10
    except Exception:
        num_samples = 10

    # Ensure OUTPUT_DIR exists and set SPARSE_LOG_PATH to a file inside it (always)
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output_test_sparse"))
    os.makedirs(out_dir, exist_ok=True)
    os.environ["SPARSE_LOG_PATH"] = os.environ.get("SPARSE_LOG_PATH", os.path.join(out_dir, "hf_prefill.log"))

    print(f"Running test_sparse_oracle with NUM_SAMPLES={num_samples}, OUTPUT_DIR={out_dir}, PREFILL_CHUNK_SIZE={os.environ.get('PREFILL_CHUNK_SIZE')}, SPARSE_LOG_PATH={os.environ.get('SPARSE_LOG_PATH')}")
    run_example(num_samples=num_samples)


