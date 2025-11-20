#!/usr/bin/env python3
"""Minimal benchmark runner for RULER 16k: runs N samples (dense vs sparse) and
writes results under ``output_test_sparse/``. Designed to be small and
reproducible for local experiments.

This script is adapted from test_sparse_oracle.py for RULER 16k benchmark.

ENV FLAGS CHEAT-SHEET (two-band remapping visibility)
-----------------------------------------------------
- Required to see two-band scaling logs in the sparse path:
  - SPARSE_DEBUG=1
  - ENABLE_POSITION_REASSIGNMENT=1
  - Ensure the mask is actually sparse (density < 1.0); otherwise the
    remapping branch is skipped by design. If your current maskers yield
    dense masks, reduce the oracle budget, e.g. ORACLE_TOPK_HEAVY_SIZE=0.05.

- Optional diagnostics:
  - EXTEND_CONTEXT=1               # compute unroped mask for analysis logs
  - COMPARE_MASK_ROPED_VS_UNROPED=1  # log mask similarity micrometrics
  - PREFILL_CHUNK_SIZE=4096        # chunked prefill to mirror prior runs

- Pack K+Chunk (translate-only, slope=1) toggle:
  - Controlled via code config (ResearchAttentionConfig.pack_k_chunk_translation=True),
    not an env var. When enabled, logs print one line:
      [Pack K+Chunk] M_end=..., Δ=..., K=[..], Chunk=[..]

- RULER 16k specific:
  - RULER_TASKS: Comma-separated list of task subsets to run (default: all 13 tasks)
    Available: cwe, fwe, niah_multikey_1, niah_multikey_2, niah_multikey_3,
               niah_multiquery, niah_multivalue, niah_single_1, niah_single_2,
               niah_single_3, qa_1, qa_2, vt
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

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
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from benchmark.benchmark_registry import create_benchmark_instance

# RULER 16k task-specific max_new_tokens requirements
# Source: https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
RULER_MAX_NEW_TOKENS = {
    "niah": 128,  # For all niah_* tasks
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa": 32,  # For qa_1, qa_2
}


def get_max_new_tokens_for_tasks(tasks: List[str]) -> int:
    """Determine max_new_tokens based on RULER tasks.
    
    Args:
        tasks: List of RULER task names (e.g., ["vt", "qa_1", "niah_single_1"])
    
    Returns:
        Maximum required max_new_tokens across all tasks
    """
    max_tokens = 8  # Minimum default
    for task in tasks:
        # Map task to its category
        if task.startswith("niah"):
            task_category = "niah"
        elif task.startswith("qa"):
            task_category = "qa"
        else:
            task_category = task  # vt, cwe, fwe
        
        if task_category in RULER_MAX_NEW_TOKENS:
            max_tokens = max(max_tokens, RULER_MAX_NEW_TOKENS[task_category])
    
    return max_tokens


def run_example(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_context_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,  # If None, will be determined from tasks
    num_samples: int = 1,
    ruler_tasks: Optional[List[str]] = None,
) -> None:
    """Run num_samples RULER 16k examples using dense and sparse attention.

    Args:
        model_name: HuggingFace model identifier
        max_context_length: Optional maximum context length to truncate to
        max_new_tokens: Maximum number of tokens to generate. If None, will be
            auto-determined from ruler_tasks based on RULER requirements:
            vt=30, cwe=120, fwe=50, qa=32, niah=128
        num_samples: Number of samples to process (total across all tasks)
        ruler_tasks: List of RULER task subsets to load (default: ["vt"])

    Outputs are written to ``./output_test_sparse/``.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output_test_sparse"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Configure MicroMetricLogger to enable attention weight difference logging
    metric_logger = MicroMetricLogger()
    enabled_metrics: List[str] = ["research_attention_weight_diff"]
    
    # Enable mask comparison metric if flag is set
    if os.environ.get("COMPARE_MASK_ROPED_VS_UNROPED", "0").lower() in ("1", "true", "yes"):
        enabled_metrics.append("research_mask_roped_vs_unroped")
    
    metric_logger.configure_logging(
        log_path=out_dir,
        enabled_metrics=enabled_metrics,
    )
    print(f"[MicroMetrics] Enabled metrics: {metric_logger.get_enabled_metrics()}")

    # Load RULER 16k dataset
    # Available tasks: cwe, fwe, niah_multikey_1, niah_multikey_2, niah_multikey_3,
    #                  niah_multiquery, niah_multivalue, niah_single_1, niah_single_2,
    #                  niah_single_3, qa_1, qa_2, vt
    all_ruler_tasks = [
        "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
        "niah_single_3", "qa_1", "qa_2", "vt"
    ]
    
    if ruler_tasks is None:
        # Get from env var or use only "vt" task by default
        tasks_env = os.environ.get("RULER_TASKS", "")
        if tasks_env:
            ruler_tasks = [t.strip() for t in tasks_env.split(",") if t.strip()]
        else:
            # Default to only "vt" task for all RULER experiments
            ruler_tasks = ["vt"]
    
    # Validate tasks
    invalid_tasks = [t for t in ruler_tasks if t not in all_ruler_tasks]
    if invalid_tasks:
        raise ValueError(f"Invalid RULER tasks: {invalid_tasks}. Valid tasks: {all_ruler_tasks}")
    
    # Determine max_new_tokens from tasks if not explicitly provided
    if max_new_tokens is None:
        max_new_tokens = get_max_new_tokens_for_tasks(ruler_tasks)
        print(f"[RULER 16k] Auto-determined max_new_tokens={max_new_tokens} for tasks: {ruler_tasks}")
    else:
        print(f"[RULER 16k] Using provided max_new_tokens={max_new_tokens}")
    
    print(f"[RULER 16k] Loading tasks: {ruler_tasks}")
    
    # Load datasets from all specified tasks
    all_samples = []
    samples_per_task = {}
    for task in ruler_tasks:
        try:
            ds_task = load_dataset("xAlg-AI/att-hub-ruler-16k", task, split=task)
            # Convert to list and add task name
            task_samples = []
            for sample in ds_task:
                sample["task"] = task
                task_samples.append(sample)
                all_samples.append(sample)
            samples_per_task[task] = len(task_samples)
            print(f"  ✓ Loaded {len(task_samples)} samples from {task}")
        except Exception as e:
            print(f"  ❌ Failed to load {task}: {e}")
            continue
    
    if not all_samples:
        raise RuntimeError("No RULER 16k samples could be loaded")
    
    total_loaded = len(all_samples)
    print(f"[RULER 16k] Loaded {total_loaded} total samples across {len(ruler_tasks)} tasks")
    print(f"[RULER 16k] Samples per task: {samples_per_task}")
    
    # Limit to num_samples total (takes first N samples across all tasks)
    # Note: This samples sequentially from tasks in order. For balanced sampling,
    # you could shuffle or use per-task limits, but this keeps it simple.
    if num_samples < len(all_samples):
        all_samples = all_samples[:num_samples]
        print(f"[RULER 16k] Limiting to {num_samples} samples (first {num_samples} from combined dataset)")
    
    print(f"[RULER 16k] Total samples to process: {len(all_samples)}")

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
    heavy_size = float(os.environ.get("ORACLE_TOPK_HEAVY_SIZE", "0.1"))
    sparse_cfg = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=heavy_size),
        ],
        # Ensure translate-pack K+Chunk (slope 1, pure translation)
        pack_k_chunk_translation=True,
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
    
    # Initialize CSV file with header (RULER format: answer instead of answers, task column)
    csv_path = os.path.join(out_dir, "raw_results.csv")
    df_header = pd.DataFrame(columns=["context", "question", "predicted_answer", "elapsed_s", "answer", "task", "method", "context_length"])
    df_header.to_csv(csv_path, index=False)

    for i, sample in enumerate(all_samples):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{len(all_samples)}")
        print(f"{'='*80}")
        
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")  # RULER uses "answer" (string), not "answers" (list)
        task = sample.get("task", "unknown")
        
        if not context or not question:
            print(f"  ⚠️  Skipping sample {i+1}: missing context or question")
            continue
        
        # RULER uses "Answer: " as default prefix
        answer_prefix = sample.get("answer_prefix", "Answer: ")
        req = Request(context=context, questions=question, answer_prefix=answer_prefix)

        print(f"  Task: {task}")
        print(f"  Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  Context length: {len(context)} chars")

        # Dense inference
        print(f"  [Dense] Processing...")
        t0 = time.time()
        r_dense = adapter_dense.process_request(req, generation_kwargs, request_kwargs)
        t1 = time.time()
        dense_out = r_dense.responses
        dense_elapsed = t1 - t0
        results["dense"].append({"response": dense_out, "elapsed_s": dense_elapsed})
        print(f"  [Dense] Response: {dense_out}")
        print(f"  [Dense] Elapsed: {dense_elapsed:.2f}s")

        # Sparse inference
        print(f"  [Sparse] Processing...")
        t0 = time.time()
        r_sparse = adapter_sparse.process_request(req, generation_kwargs, request_kwargs)
        t1 = time.time()
        sparse_out = r_sparse.responses
        sparse_elapsed = t1 - t0
        results["sparse"].append({"response": sparse_out, "elapsed_s": sparse_elapsed})
        print(f"  [Sparse] Response: {sparse_out}")
        print(f"  [Sparse] Elapsed: {sparse_elapsed:.2f}s")

        # Free cache and flush metrics periodically
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Flush metrics after each sample to ensure they're written
            metric_logger.flush()
        except Exception:
            pass

        # Add rows (RULER format)
        dense_row = {
            "context": context,
            "question": question,
            "predicted_answer": dense_out,
            "elapsed_s": dense_elapsed,
            "answer": answer,  # RULER uses "answer" (string)
            "task": task,
            "method": "dense",
            "context_length": 16384,  # RULER 16k fixed context length
        }
        sparse_row = {
            "context": context,
            "question": question,
            "predicted_answer": sparse_out,
            "elapsed_s": sparse_elapsed,
            "answer": answer,  # RULER uses "answer" (string)
            "task": task,
            "method": "sparse",
            "context_length": 16384,  # RULER 16k fixed context length
        }
        rows.append(dense_row)
        rows.append(sparse_row)
        
        # Save incrementally: append to CSV and update JSON
        df_new = pd.DataFrame([dense_row, sparse_row])
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
        
        # Update JSON incrementally
        with open(os.path.join(out_dir, "test_sparse_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Saved results for sample {i+1} (CSV + JSON updated)")
        
        # Try to compute and print metrics incrementally if possible
        try:
            df_current = pd.DataFrame(rows)
            ruler16k = create_benchmark_instance("ruler16k", subsets=ruler_tasks)
            metrics_current = ruler16k.post_run_evaluate(df_current)
            print(f"  [Current Metrics] Overall score: {metrics_current.get('overall_score', 'N/A')}")
            if "task_scores" in metrics_current:
                print(f"  [Current Metrics] Task scores: {metrics_current['task_scores']}")
            
            # Save metrics incrementally
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics_current, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Could not compute metrics yet: {e}")

    # Final summary (results already saved incrementally above)
    print(f"\n{'='*80}")
    print(f"COMPLETED: Processed {len(rows)//2} samples")
    print(f"{'='*80}")
    
    df = pd.DataFrame(rows)
    # CSV already saved incrementally, but ensure final version is correct
    df.to_csv(csv_path, index=False)

    # Final evaluation using RULER 16k benchmark
    try:
        ruler16k = create_benchmark_instance("ruler16k", subsets=ruler_tasks)
        metrics = ruler16k.post_run_evaluate(df)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Final metrics: {metrics}")
    except Exception as e:
        metrics = None
        print(f"Failed to compute final metrics: {e}")

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
                dense_metrics = ruler16k.post_run_evaluate(df_dense)
                with open(os.path.join(comp_dir, "dense_metrics.json"), "w") as f:
                    json.dump(dense_metrics, f, indent=2)
                print("Dense metrics:", dense_metrics)
            except Exception as e:
                print("Failed to compute dense metrics:", e)

        if not df_sparse.empty:
            try:
                sparse_metrics = ruler16k.post_run_evaluate(df_sparse)
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
        json.dump({
            "model_name": model_name,
            "generation_kwargs": generation_kwargs,
            "request_kwargs": request_kwargs,
            "ruler_tasks": ruler_tasks,
        }, f, indent=2)

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
    
    # Print final summary to terminal
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Dense: {summary['dense']['n']} samples, avg time: {summary['dense']['mean_elapsed_s']:.2f}s")
    print(f"Sparse: {summary['sparse']['n']} samples, avg time: {summary['sparse']['mean_elapsed_s']:.2f}s")
    if metrics is not None:
        print(f"Overall score: {metrics.get('overall_score', 'N/A')}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*80}\n")
    
    # Flush micro metrics to ensure all logs are written
    metric_logger.flush()
    micro_metrics_file = os.path.join(out_dir, 'micro_metrics.jsonl')
    if os.path.exists(micro_metrics_file):
        file_size = os.path.getsize(micro_metrics_file)
        print(f"[MicroMetrics] ✓ Metrics logged to: {micro_metrics_file} ({file_size} bytes)")
        # Count lines to show how many metrics were logged
        try:
            with open(micro_metrics_file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"[MicroMetrics] Total metric entries: {line_count}")
        except Exception:
            pass
    else:
        print(f"[MicroMetrics] ⚠️  Warning: micro_metrics.jsonl not found at {micro_metrics_file}")


if __name__ == "__main__":
    # Allow overriding number of samples via env var for quick experiments
    ns = os.environ.get("NUM_SAMPLES")
    try:
        num_samples = int(ns) if ns is not None else 10
    except Exception:
        num_samples = 10

    # Allow overriding model checkpoint via env var
    # Accept either MODEL_NAME or MODEL_ID (MODEL_NAME takes precedence)
    model_env: Optional[str] = os.environ.get("MODEL_NAME") or os.environ.get("MODEL_ID")
    if model_env is None or len(model_env.strip()) == 0:
        model_env = "meta-llama/Llama-3.2-1B-Instruct"

    # Parse RULER tasks from env var
    tasks_env = os.environ.get("RULER_TASKS", "")
    ruler_tasks: Optional[List[str]] = None
    if tasks_env:
        ruler_tasks = [t.strip() for t in tasks_env.split(",") if t.strip()]

    # Ensure OUTPUT_DIR exists and set SPARSE_LOG_PATH to a file inside it (always)
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output_test_sparse"))
    os.makedirs(out_dir, exist_ok=True)
    os.environ["SPARSE_LOG_PATH"] = os.environ.get("SPARSE_LOG_PATH", os.path.join(out_dir, "hf_prefill.log"))

    print(f"Running test_sparse_oracle_ruler16k with NUM_SAMPLES={num_samples}, OUTPUT_DIR={out_dir}, PREFILL_CHUNK_SIZE={os.environ.get('PREFILL_CHUNK_SIZE')}, SPARSE_LOG_PATH={os.environ.get('SPARSE_LOG_PATH')}, MODEL_NAME={model_env}, RULER_TASKS={ruler_tasks or 'all'}")
    run_example(model_name=model_env, num_samples=num_samples, ruler_tasks=ruler_tasks)

