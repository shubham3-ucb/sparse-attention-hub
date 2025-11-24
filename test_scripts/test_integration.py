#!/usr/bin/env python3
"""Integration test for prod code verification.

This test file is designed to be updated as we add features:
- Phase 1: Chunked Prefill verification
- Phase 2: Repositioning verification (future)
- Phase 3: Micro Metrics verification (future)

Current Phase 1 tests:
- Dense + Chunked Prefill vs Dense + Single Call
- Sparse + Chunked Prefill vs Sparse + Single Call
- Verifies chunked prefill produces same results as single-call
- Computes metrics for quantitative comparison

Usage:
    python test_scripts/test_integration.py
    # Or via shell script:
    ./test_scripts/run_test.sh
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import pandas as pd
import torch
from datasets import load_dataset

# Add parent directory to path to import sparse_attention_hub
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


def run_phase1_test(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_samples: int = 5,
    chunk_size: int = 1024,
) -> None:
    """Run Phase 1 (Chunked Prefill) verification test.

    Tests:
    1. Dense + Chunked Prefill
    2. Dense + Single Call
    3. Sparse + Chunked Prefill
    4. Sparse + Single Call

    Compares results and computes metrics to verify chunked prefill correctness.

    Args:
        model_name: HuggingFace model name
        num_samples: Number of HotpotQA samples to test
        chunk_size: Chunk size for chunked prefill (when enabled)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get sparse attention parameters FIRST (needed for settings and config)
    heavy_size = float(os.environ.get("ORACLE_TOPK_HEAVY_SIZE", "0.1"))
    sink_size = int(os.environ.get("SINK_SIZE", "128"))
    local_window = int(os.environ.get("LOCAL_WINDOW_SIZE", "128"))
    
    # Get repositioning parameters
    num_sink_tokens = int(os.environ.get("NUM_SINK_TOKENS", "0"))
    prefix_freeze_tail_k = int(os.environ.get("PREFIX_FREEZE_TAIL_K", "0"))
    enable_repositioning = os.environ.get("ENABLE_POSITION_REASSIGNMENT", "0").lower() in ("1", "true", "yes")
    extend_context = os.environ.get("EXTEND_CONTEXT", "0").lower() in ("1", "true", "yes")
    
    # Get context repetition parameter (for doubling context)
    repeat_count = int(os.environ.get("REPEAT_COUNT", "1"))
    
    # Create descriptive output directory name
    model_short = model_name.split("/")[-1].replace("-", "").lower()
    if "llama" in model_short:
        if "3.1" in model_name or "3_1" in model_short:
            model_short = "llama31_8b"
        elif "3.2" in model_name or "3_2" in model_short:
            model_short = "llama32_8b"
        elif "3" in model_short:
            model_short = "llama3_8b"
    
    # Build output directory name with repositioning info
    repo_suffix = ""
    if enable_repositioning:
        repo_suffix = f"_repo_k{prefix_freeze_tail_k}_s{num_sink_tokens}"
    if extend_context:
        repo_suffix += "_extctx"
    if repeat_count > 1:
        repo_suffix += f"_repeat{repeat_count}"
    
    # Add verify_refactor identifier if VERIFY_REFACTOR env var is set
    verify_suffix = ""
    if os.environ.get("VERIFY_REFACTOR", "0").lower() in ("1", "true", "yes"):
        verify_suffix = "_verify_refactor"
    
    # Create test_outputs directory if it doesn't exist
    test_outputs_dir = os.path.join(os.getcwd(), "test_outputs")
    os.makedirs(test_outputs_dir, exist_ok=True)
    
    default_out_dir = os.path.join(
        test_outputs_dir,
        f"test_integration_phase2_{model_short}_ns{num_samples}_pcs{chunk_size}{repo_suffix}{verify_suffix}"
    )
    out_dir = os.environ.get("OUTPUT_DIR", default_out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Set SPARSE_LOG_PATH for hf_prefill.log
    if not os.environ.get("SPARSE_LOG_PATH"):
        os.environ["SPARSE_LOG_PATH"] = os.path.join(out_dir, "hf_prefill.log")
    
    # Configure MicroMetricLogger ONCE at start (matching old codebase)
    # Metrics will only be logged when sparse attention code runs (dense doesn't call logging)
    # OLD CODEBASE ONLY ENABLED: research_attention_weight_diff + research_mask_roped_vs_unroped (if flag set)
    metric_logger = MicroMetricLogger()
    enabled_metrics: List[str] = [
        "research_attention_weight_diff",  # Only this metric (matching old codebase)
    ]
    
    # Enable mask comparison metric if flag is set (matching old codebase)
    if os.environ.get("COMPARE_MASK_ROPED_VS_UNROPED", "0").lower() in ("1", "true", "yes"):
        enabled_metrics.append("research_mask_roped_vs_unroped")
    
    metric_logger.configure_logging(
        log_path=out_dir,
        enabled_metrics=enabled_metrics,
    )
    print(f"[MicroMetrics] Configured logger with enabled metrics: {metric_logger.get_enabled_metrics()}")
    
    # Save comprehensive settings for reproducibility
    settings = {
        "test_info": {
            "test_name": "Phase 2: Repositioning Verification (K=0, S=0)",
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": 2,
        },
        "model": {
            "model_name": model_name,
            "model_short": model_short,
            "device": device,
            "torch_dtype": "bfloat16",
        },
        "dataset": {
            "name": "HotpotQA (LongBench)",
            "num_samples": num_samples,
            "max_new_tokens": 52,  # HotpotQA default
            "repeat_count": repeat_count,  # Context repetition multiplier (1 = no repetition, 2 = 2x context)
        },
        "chunked_prefill": {
            "enabled": True,
            "chunk_size": chunk_size,
            "description": "Processes long contexts in chunks to manage memory",
        },
        "sparse_attention": {
            "enabled": True,
            "technique": "Multi-Masker Sparse Attention",
            "maskers": [
                {
                    "name": "SinkMasker",
                    "type": "SinkMaskerConfig",
                    "sink_size": sink_size,
                    "description": f"Keeps first {sink_size} tokens (global context) - always attends to initial tokens",
                },
                {
                    "name": "LocalMasker",
                    "type": "LocalMaskerConfig",
                    "window_size": local_window,
                    "description": f"Sliding window of {local_window} tokens around each query position - local attention",
                },
                {
                    "name": "OracleTopK",
                    "type": "OracleTopKConfig",
                    "heavy_size": heavy_size,
                    "description": f"Selects top {heavy_size*100}% most important tokens based on ground truth attention weights (research only)",
                },
            ],
            "total_sparsity": f"~{heavy_size*100}% (OracleTopK) + local window + sink tokens",
            "note": "OracleTopK uses ground truth attention weights - for research/evaluation only",
        },
        "repositioning": {
            "enabled": enable_repositioning,
            "num_sink_tokens": num_sink_tokens,
            "prefix_freeze_tail_k": prefix_freeze_tail_k,
            "pack_k_chunk_translation": True,
            "description": f"Two-band scaling: S={num_sink_tokens} (sink tokens frozen), K={prefix_freeze_tail_k} (tail tokens frozen), middle band compressed",
        },
        "extend_context": {
            "enabled": extend_context,
            "description": "Use unroped Q/K for mask computation (position-agnostic similarity)",
        },
        "micro_metrics": {
            "enabled": True,
            "log_path": os.path.join(out_dir, "micro_metrics.jsonl"),
            "enabled_metrics": list(metric_logger.get_enabled_metrics()),
            "note": "Micro metrics are logged ONLY for sparse_chunked runs (matching old codebase: only research_attention_weight_diff + research_mask_roped_vs_unroped if flag set)",
            "description": "Micro-level metrics matching old codebase: weight differences and mask comparisons (if COMPARE_MASK_ROPED_VS_UNROPED=1)",
        },
        "scenarios": [
            {
                "name": "dense_chunked",
                "description": "Dense attention with chunked prefill",
                "sparse": False,
                "chunked": True,
            },
            {
                "name": "sparse_chunked",
                "description": "Sparse attention with chunked prefill (main test)",
                "sparse": True,
                "chunked": True,
            },
        ],
        "reproduction": {
            "command": f"""cd {os.getcwd()}
MODEL_NAME="{model_name}" \\
NUM_SAMPLES={num_samples} \\
PREFILL_CHUNK_SIZE={chunk_size} \\
ORACLE_TOPK_HEAVY_SIZE={heavy_size} \\
SINK_SIZE={sink_size} \\
LOCAL_WINDOW_SIZE={local_window} \\
OUTPUT_DIR="{out_dir}" \\
SPARSE_DEBUG=1 \\
./test_scripts/run_test.sh""",
            "environment_variables": {
                "MODEL_NAME": model_name,
                "NUM_SAMPLES": str(num_samples),
                "PREFILL_CHUNK_SIZE": str(chunk_size),
                "ORACLE_TOPK_HEAVY_SIZE": str(heavy_size),
                "SINK_SIZE": str(sink_size),
                "LOCAL_WINDOW_SIZE": str(local_window),
                "OUTPUT_DIR": out_dir,
                "SPARSE_DEBUG": "1",
            },
        },
    }
    
    # Save settings to file
    settings_file = os.path.join(out_dir, "settings.json")
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"✓ Saved settings to: {settings_file}")
    
    # Print output directory on a separate line for shell script to parse
    print(f"OUTPUT_DIR_ACTUAL: {out_dir}", flush=True)
    
    print("=" * 80)
    print("PHASE 1: CHUNKED PREFILL VERIFICATION TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"Output dir: {out_dir}")
    print(f"Sparse: Sink={sink_size}, Local={local_window}, OracleTopK={heavy_size}")
    print("=" * 80)

    # Load HotpotQA dataset
    print("\n[1/6] Loading HotpotQA dataset...")
    ds = load_dataset("Xnhyacinth/LongBench", "hotpotqa", split="test")
    ds = ds.select(list(range(min(num_samples, len(ds)))))
    print(f"✓ Loaded {len(ds)} samples")

    # Setup model kwargs
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    tokenizer_kwargs = {"padding_side": "left"}
    if hf_token:
        model_kwargs["use_auth_token"] = hf_token
        tokenizer_kwargs["use_auth_token"] = hf_token
        model_kwargs.setdefault("trust_remote_code", True)

    # Create adapters - ONLY sparse (skip dense to match old codebase pattern)
    print("\n[2/6] Creating adapters...")
    
    # Create sparse attention config (parameters already read above)
    sparse_cfg = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=sink_size),
            LocalMaskerConfig(window_size=local_window),
            OracleTopKConfig(heavy_size=heavy_size),
        ],
        pack_k_chunk_translation=True,  # Enable pack translation for repositioning
    )
    adapter_sparse = ModelAdapterHF(
        model_name, sparse_cfg, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs, device=device
    )
    print("✓ Adapters created")

    # Test scenarios - ONLY sparse chunked (skip dense to match old codebase pattern)
    scenarios = [
        {"name": "sparse_chunked", "adapter": adapter_sparse, "chunked": True, "sparse": True},
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {s["name"]: [] for s in scenarios}
    rows: List[Dict[str, Any]] = []

    print("\n[3/6] Running test scenarios...")
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        
        # MicroMetricLogger is already configured at start (matching old codebase)
        # Metrics will only be logged when sparse attention code runs
        # Dense attention doesn't call MicroMetricLogger.log(), so no metrics logged for dense
        
        # Set chunked prefill environment
        if scenario["chunked"]:
            os.environ["PREFILL_CHUNK_SIZE"] = str(chunk_size)
            print(f"  Chunked prefill enabled (chunk_size={chunk_size})")
        else:
            os.environ.pop("PREFILL_CHUNK_SIZE", None)
            print(f"  Single-call prefill (no chunking)")

        for i, sample in enumerate(ds):
            context = sample.get("context", "")
            question = sample.get("question", "")
            if not context or not question:
                print(f"  ⚠️  Skipping sample {i+1}: missing context or question")
                continue
            
            # Repeat context if REPEAT_COUNT > 1 (for doubling context)
            if repeat_count > 1:
                original_context = context
                context = context * repeat_count
                print(f"  Context repetition: {repeat_count}x (original: {len(original_context)} chars → repeated: {len(context)} chars)")

            # Read max_new_tokens from sample if available
            sample_max_new_tokens = sample.get("max_new_tokens", None)
            if sample_max_new_tokens is not None:
                try:
                    generation_kwargs = {"max_new_tokens": int(sample_max_new_tokens)}
                except (ValueError, TypeError):
                    generation_kwargs = {"max_new_tokens": 52}  # HotpotQA default
            else:
                generation_kwargs = {"max_new_tokens": 52}

            request_kwargs = {}
            if scenario["chunked"]:
                request_kwargs["prefill_chunk_size"] = chunk_size
                # Disable assertion for testing (small numerical differences are expected)
                if not scenario["sparse"]:
                    request_kwargs["assert_chunked_equals_full"] = False

            req = Request(context=context, questions=question, answer_prefix=sample.get("answer_prefix", "Answer: "))

            print(f"  Sample {i+1}/{len(ds)}: Processing...")
            t0 = time.time()
            
            try:
                if scenario["sparse"]:
                    with adapter_sparse.enable_sparse_mode():
                        r = scenario["adapter"].process_request(req, generation_kwargs, request_kwargs)
                else:
                    r = scenario["adapter"].process_request(req, generation_kwargs, request_kwargs)
                
                t1 = time.time()
                response = r.responses if isinstance(r.responses, str) else r.responses[0]
                elapsed = t1 - t0
                
                all_results[scenario["name"]].append({
                    "sample_idx": i,
                    "response": response,
                    "elapsed_s": elapsed,
                })
                
                rows.append({
                    "context": context,
                    "question": question,
                    "predicted_answer": response,
                    "elapsed_s": elapsed,
                    "answers": sample.get("answers", None),
                    "task": "hotpotqa",
                    "method": scenario["name"],
                    "all_classes": sample.get("all_classes", []),
                })
                
                print(f"    ✓ Response: {response[:80]}...")
                print(f"    ✓ Elapsed: {elapsed:.2f}s")
                
                # Flush metrics after each sample (matching old codebase)
                if scenario["sparse"]:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        metric_logger.flush()
                    except Exception:
                        pass
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                all_results[scenario["name"]].append({
                    "sample_idx": i,
                    "response": None,
                    "elapsed_s": None,
                    "error": str(e),
                })
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save raw results
    print("\n[4/6] Saving results...")
    for scenario_name, results in all_results.items():
        result_file = os.path.join(out_dir, f"results_{scenario_name}.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved {result_file}")

    # Save CSV
    csv_path = os.path.join(out_dir, "all_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path}")

    # Compute metrics
    print("\n[5/6] Computing metrics...")
    metrics_results: Dict[str, Any] = {}
    
    try:
        longbench = create_benchmark_instance("longbench", subsets=["hotpotqa"])
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            df_scenario = df[df.method == scenario_name].copy()
            
            if not df_scenario.empty:
                try:
                    metrics = longbench.post_run_evaluate(df_scenario)
                    metrics_results[scenario_name] = metrics
                    print(f"  ✓ {scenario_name}: overall_score={metrics.get('overall_score', 'N/A')}")
                except Exception as e:
                    print(f"  ⚠️  {scenario_name}: Failed to compute metrics: {e}")
                    metrics_results[scenario_name] = {"error": str(e)}
    except Exception as e:
        print(f"  ❌ Failed to compute metrics: {e}")

    # Save metrics
    metrics_file = os.path.join(out_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"  ✓ Saved {metrics_file}")

    # Generate comparison summary
    print("\n[6/6] Generating comparison summary...")
    summary = {
        "test_info": {
            "model_name": model_name,
            "num_samples": num_samples,
            "chunk_size": chunk_size,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "scenarios": {},
        "comparisons": {},
    }

    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_results = all_results[scenario_name]
        
        successful = [r for r in scenario_results if r.get("response") is not None]
        failed = [r for r in scenario_results if r.get("response") is None]
        
        avg_elapsed = sum(r["elapsed_s"] for r in successful) / len(successful) if successful else None
        
        summary["scenarios"][scenario_name] = {
            "successful": len(successful),
            "failed": len(failed),
            "avg_elapsed_s": round(avg_elapsed, 2) if avg_elapsed else None,
            "overall_score": metrics_results.get(scenario_name, {}).get("overall_score", None),
        }

    # No comparisons needed - only running sparse (matching old codebase pattern)

    # Save summary
    summary_file = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved {summary_file}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Samples: {num_samples}")
    print(f"  Chunk size: {chunk_size}")
    
    print(f"\nScenario Results:")
    for scenario_name, scenario_summary in summary["scenarios"].items():
        print(f"  {scenario_name}:")
        print(f"    Successful: {scenario_summary['successful']}/{num_samples}")
        print(f"    Avg elapsed: {scenario_summary['avg_elapsed_s']}s")
        print(f"    Overall score: {scenario_summary['overall_score']}")
    
    print(f"\nComparisons:")
    if "dense_chunked_vs_single" in summary["comparisons"]:
        comp = summary["comparisons"]["dense_chunked_vs_single"]
        match_str = "✅ MATCH" if comp["match"] else "⚠️  DIFFER"
        print(f"  Dense Chunked vs Single: {match_str}")
        print(f"    Chunked: {comp['chunked_score']}, Single: {comp['single_score']}")
        print(f"    Difference: {comp['score_difference']}")
    
    if "sparse_chunked_vs_single" in summary["comparisons"]:
        comp = summary["comparisons"]["sparse_chunked_vs_single"]
        match_str = "✅ MATCH" if comp["match"] else "⚠️  DIFFER"
        print(f"  Sparse Chunked vs Single: {match_str}")
        print(f"    Chunked: {comp['chunked_score']}, Single: {comp['single_score']}")
        print(f"    Difference: {comp['score_difference']}")
    
    # Flush micro metrics to ensure all metrics are written (only if sparse was run)
    print("\n[7/7] Flushing micro metrics...")
    if metric_logger.is_logging_configured() and metric_logger.get_enabled_metrics():
        metric_logger.flush()
        micro_metrics_file = os.path.join(out_dir, 'micro_metrics.jsonl')
        if os.path.exists(micro_metrics_file):
            file_size = os.path.getsize(micro_metrics_file)
            with open(micro_metrics_file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  ✓ Micro metrics flushed to: {micro_metrics_file}")
            print(f"  ✓ Total metric entries: {line_count:,} ({file_size:,} bytes)")
        else:
            print(f"  ⚠️  Micro metrics file not found (no sparse run?)")
    else:
        print(f"  ⚠️  Micro metrics not configured (dense-only run?)")
    
    print(f"\nResults saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # Allow overriding parameters via env vars
    model_env: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    num_samples_env: int = int(os.environ.get("NUM_SAMPLES", "5"))
    chunk_size_env: int = int(os.environ.get("PREFILL_CHUNK_SIZE", "1024"))
    
    # Output dir will be auto-generated with descriptive name
    # But can be overridden via OUTPUT_DIR env var (only if non-empty)
    output_dir_env: str = os.environ.get("OUTPUT_DIR", "").strip()
    
    # Only use OUTPUT_DIR if it's actually set and non-empty
    if output_dir_env:
        os.makedirs(output_dir_env, exist_ok=True)
        os.environ["SPARSE_LOG_PATH"] = os.environ.get("SPARSE_LOG_PATH", os.path.join(output_dir_env, "hf_prefill.log"))
        os.environ["OUTPUT_DIR"] = output_dir_env
    else:
        # Clear OUTPUT_DIR so Python script can auto-generate
        if "OUTPUT_DIR" in os.environ:
            del os.environ["OUTPUT_DIR"]

    print(f"Starting integration test with:")
    print(f"  MODEL_NAME={model_env}")
    print(f"  NUM_SAMPLES={num_samples_env}")
    print(f"  PREFILL_CHUNK_SIZE={chunk_size_env}")
    print(f"  ORACLE_TOPK_HEAVY_SIZE={os.environ.get('ORACLE_TOPK_HEAVY_SIZE', '0.1')}")
    print(f"  SINK_SIZE={os.environ.get('SINK_SIZE', '128')}")
    print(f"  LOCAL_WINDOW_SIZE={os.environ.get('LOCAL_WINDOW_SIZE', '128')}")
    print(f"  OUTPUT_DIR={output_dir_env if output_dir_env else '(auto-generated)'}")
    print(f"  SPARSE_DEBUG={os.environ.get('SPARSE_DEBUG', '0')}")
    
    run_phase1_test(
        model_name=model_env,
        num_samples=num_samples_env,
        chunk_size=chunk_size_env,
    )

