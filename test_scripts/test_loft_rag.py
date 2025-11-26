#!/usr/bin/env python3
"""Integration test for LOFT RAG benchmark verification.

This test follows the EXACT pattern from test_integration.py but uses LOFT RAG datasets.
It allows testing with different sparse attention settings via environment variables.

Usage:
    python test_scripts/test_loft_rag.py
    # Or via shell script:
    ./test_scripts/quick_test_loft_rag.sh

Environment variables:
    MODEL_NAME: HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)
    NUM_SAMPLES: Number of samples to test (default: 5)
    PREFILL_CHUNK_SIZE: Chunk size for chunked prefill (default: 1024)
    LOFT_DATASET: LOFT dataset to use (default: hotpotqa_32k)
    ORACLE_TOPK_HEAVY_SIZE: Sparse attention heavy size (default: 0.1)
    SINK_SIZE: Sink tokens (default: 128)
    LOCAL_WINDOW_SIZE: Local window size (default: 128)
    ENABLE_POSITION_REASSIGNMENT: Enable repositioning (default: 0)
    NUM_SINK_TOKENS: Number of sink tokens for repositioning (default: 0)
    PREFIX_FREEZE_TAIL_K: Prefix freeze tail K (default: 0)
    EXTEND_CONTEXT: Extend context (default: 0)
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


def run_loft_rag_test(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_samples: int = 5,
    chunk_size: int = 1024,
    loft_dataset: str = "hotpotqa_32k",
) -> None:
    """Run LOFT RAG benchmark test following exact pattern from test_integration.py.

    Tests:
    1. Sparse + Chunked Prefill

    Args:
        model_name: HuggingFace model name
        num_samples: Number of LOFT RAG samples to test
        chunk_size: Chunk size for chunked prefill (when enabled)
        loft_dataset: LOFT dataset name (e.g., "hotpotqa_32k", "nq_32k")
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
    
    # Parse LOFT dataset name (e.g., "hotpotqa_32k" -> dataset="hotpotqa", length="32k")
    loft_parts = loft_dataset.split("_")
    if len(loft_parts) < 2:
        raise ValueError(f"Invalid LOFT dataset format: {loft_dataset} (expected: dataset_length, e.g., hotpotqa_32k)")
    loft_length = loft_parts[-1]  # Last part is length
    loft_dataset_name = "_".join(loft_parts[:-1])  # Everything before last underscore
    
    # Create output directory name
    output_dir_base = os.environ.get("OUTPUT_DIR", "")
    if output_dir_base:
        out_dir = output_dir_base
        os.makedirs(out_dir, exist_ok=True)
    else:
        # Auto-generate output directory name
        out_dir = os.path.join(
            "test_outputs",
            f"test_loft_{loft_dataset}_{model_short}_ns{num_samples}_pcs{chunk_size}{repo_suffix}"
        )
        os.makedirs(out_dir, exist_ok=True)
    
    # Setup micro metric logger (same as test_integration.py)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=out_dir,  # log_path should be directory, not file path (flush() appends "micro_metrics.jsonl")
        enabled_metrics=["attention_mask_density", "kv_cache_size"],
    )
    
    # Save test settings
    settings = {
        "model_name": model_name,
        "num_samples": num_samples,
        "chunk_size": chunk_size,
        "loft_dataset": loft_dataset,
        "loft_dataset_name": loft_dataset_name,
        "loft_length": loft_length,
        "sparse_attention": {
            "heavy_size": heavy_size,
            "sink_size": sink_size,
            "local_window": local_window,
        },
        "repositioning": {
            "enabled": enable_repositioning,
            "num_sink_tokens": num_sink_tokens,
            "prefix_freeze_tail_k": prefix_freeze_tail_k,
            "extend_context": extend_context,
        },
        "repeat_count": repeat_count,
    }
    
    settings_file = os.path.join(out_dir, "test_settings.json")
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"✓ Saved settings to: {settings_file}")
    
    # Print output directory on a separate line for shell script to parse
    print(f"OUTPUT_DIR_ACTUAL: {out_dir}", flush=True)
    
    print("=" * 80)
    print("LOFT RAG BENCHMARK TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"LOFT Dataset: {loft_dataset} ({loft_dataset_name} {loft_length})")
    print(f"Output dir: {out_dir}")
    print(f"Sparse: Sink={sink_size}, Local={local_window}, OracleTopK={heavy_size}")
    print("=" * 80)

    # Load LOFT RAG dataset using benchmark's _load_datasets method
    print("\n[1/6] Loading LOFT RAG dataset...")
    try:
        loft_rag = create_benchmark_instance("loft_rag", subsets=[loft_dataset])
        df_full = loft_rag._load_datasets()
        
        # Limit to num_samples
        if len(df_full) > num_samples:
            df = df_full.head(num_samples).copy()
        else:
            df = df_full.copy()
        
        print(f"✓ Loaded {len(df)} samples from {loft_dataset}")
        print(f"  Total available: {len(df_full)} samples")
        
        # Verify required columns
        required_cols = ["context", "question", "answers", "answer_prefix", "task", "max_new_tokens"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
    except Exception as e:
        print(f"❌ Failed to load LOFT dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup model kwargs
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    tokenizer_kwargs = {"padding_side": "left"}
    if hf_token:
        model_kwargs["use_auth_token"] = hf_token
        tokenizer_kwargs["use_auth_token"] = hf_token
        model_kwargs.setdefault("trust_remote_code", True)

    # Create adapters - dense baseline + sparse chunked
    print("\n[2/6] Creating adapters...")
    
    # Create dense adapter (no sparse attention)
    adapter_dense = ModelAdapterHF(
        model_name, 
        sparse_attention_config=None,  # No sparse attention = dense mode
        model_kwargs=model_kwargs, 
        tokenizer_kwargs=tokenizer_kwargs, 
        device=device
    )
    print("  ✓ Dense adapter created")
    
    # Create sparse attention config
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
    print("  ✓ Sparse adapter created")
    print("✓ All adapters created")

    # Test scenarios - dense baseline + sparse chunked
    scenarios = [
        {"name": "dense", "adapter": adapter_dense, "chunked": False, "sparse": False},
        {"name": "sparse_chunked", "adapter": adapter_sparse, "chunked": True, "sparse": True},
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {scenario["name"]: [] for scenario in scenarios}
    rows: List[Dict[str, Any]] = []
    
    # Create separate responses JSONL file for real-time appending
    responses_file = os.path.join(out_dir, "responses.jsonl")
    print(f"  📝 Responses will be appended to: {responses_file}")

    # Process samples (following exact pattern from test_integration.py)
    print("\n[3/6] Processing samples...")
    for i, row in df.iterrows():
        context = row["context"]
        question = row["question"]
        answer_prefix = row.get("answer_prefix", "Final Answer: ")
        max_new_tokens = 8  # Fixed to 8 for testing
        task = row.get("task", loft_dataset)
        
        # TEMPORARY: Aggressive truncation for quick testing - keep only first 5000 and last 5000 chars
        # TODO: Remove this truncation for production runs
        # original_len = len(context)
        # if original_len > 100000:
        #     keep_size = 50000  # Keep first 5000 and last 5000 chars
        #     if original_len > keep_size * 2:
        #         context = context[:keep_size] + "..." + context[-keep_size:]
        #         print(f"    ⚠️  Context truncated: kept first/last {keep_size} chars (original: {original_len} chars, new: {len(context)} chars)")
        #     else:
        #         # If context is smaller, just truncate to max 10000 chars
        #         context = context[:10000]
        #         print(f"    ⚠️  Context truncated: limited to 10000 chars (original: {original_len} chars)")
        
        # Handle context repetition if needed
        if repeat_count > 1:
            context = context * repeat_count
        
        print(f"\n  Sample {i+1}/{len(df)}: {task}")
        print(f"    Context length: {len(context)} chars")
        print(f"    Question: {question[:80]}...")
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            adapter = scenario["adapter"]
            use_chunked = scenario["chunked"]
            
            try:
                # Create request
                request = Request(
                    context=context,
                    questions=question,
                    answer_prefix=answer_prefix,
                )
                
                # Generation kwargs
                generation_kwargs = {"max_new_tokens": max_new_tokens}
                
                # Request kwargs for chunked prefill
                request_kwargs = {}
                if use_chunked:
                    request_kwargs["prefill_chunk_size"] = chunk_size
                # For dense mode, disable chunked prefill validation to avoid assertion errors
                if not scenario.get("sparse", False):
                    request_kwargs["assert_chunked_equals_full"] = False
                
                # Process request
                start_time = time.time()
                response = adapter.process_request(request, generation_kwargs, request_kwargs)
                elapsed = time.time() - start_time
                
                # Extract response
                if isinstance(response.responses, list):
                    response_text = response.responses[0] if response.responses else ""
                else:
                    response_text = response.responses
                
                # Print response in logs
                print(f"    ✓ {scenario_name}: {elapsed:.2f}s, response length: {len(response_text)}")
                print(f"    📤 Response: {response_text}")
                
                # Append response to separate JSONL file immediately
                response_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario_name,
                    "sample_idx": i,
                    "task": task,
                    "response": response_text,
                    "elapsed_s": elapsed,
                }
                with open(responses_file, "a") as f:
                    f.write(json.dumps(response_entry) + "\n")
                
                # Store results
                result = {
                    "sample_idx": i,
                    "task": task,
                    "context": context,
                    "question": question,
                    "response": response_text,
                    "elapsed_s": elapsed,
                    "error": None,
                }
                all_results[scenario_name].append(result)
                
                rows.append({
                    "method": scenario_name,
                    "sample_idx": i,
                    "task": task,
                    "question": question[:100],  # Truncate for CSV
                    "response": response_text[:200],  # Truncate for CSV
                    "elapsed_s": elapsed,
                    "error": None,
                })
                
            except Exception as e:
                print(f"    ✗ {scenario_name}: Error - {e}")
                print(f"    📤 Response: ERROR - {str(e)}")
                
                # Append error to responses file
                error_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": scenario_name,
                    "sample_idx": i,
                    "task": task,
                    "response": None,
                    "elapsed_s": None,
                    "error": str(e),
                }
                with open(responses_file, "a") as f:
                    f.write(json.dumps(error_entry) + "\n")
                
                all_results[scenario_name].append({
                    "sample_idx": i,
                    "task": task,
                    "context": context,
                    "question": question,
                    "response": None,
                    "elapsed_s": None,
                    "error": str(e),
                })
                rows.append({
                    "method": scenario_name,
                    "sample_idx": i,
                    "task": task,
                    "question": question[:100],
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
    df_results = pd.DataFrame(rows)
    df_results.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {csv_path}")

    # Compute metrics using LOFT benchmark
    print("\n[5/6] Computing LOFT metrics...")
    metrics_results: Dict[str, Any] = {}
    
    try:
        # Create benchmark instance for metrics calculation
        loft_rag_metrics = create_benchmark_instance("loft_rag", subsets=[loft_dataset])
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            df_scenario = df_results[df_results.method == scenario_name].copy()
            
            if not df_scenario.empty:
                # Merge with original df to get full data for metrics
                # We need: context, question, predicted_answer, answers, task, answer_prefix
                df_for_metrics = df.copy()
                df_for_metrics["predicted_answer"] = None
                
                # Map responses back to original dataframe
                for idx, row in df_scenario.iterrows():
                    sample_idx = row["sample_idx"]
                    if sample_idx < len(df_for_metrics):
                        df_for_metrics.loc[df_for_metrics.index[sample_idx], "predicted_answer"] = row.get("response", "")
                
                # Filter to only rows with predictions
                df_for_metrics = df_for_metrics[df_for_metrics["predicted_answer"].notna()].copy()
                
                if not df_for_metrics.empty:
                    try:
                        metrics = loft_rag_metrics.post_run_evaluate(df_for_metrics)
                        metrics_results[scenario_name] = metrics
                        
                        # Print metrics summary
                        if "overall" in metrics:
                            overall = metrics["overall"]
                            print(f"  ✓ {scenario_name}:")
                            print(f"    EM: {overall.get('em', 'N/A'):.4f}" if isinstance(overall.get('em'), (int, float)) else f"    EM: {overall.get('em', 'N/A')}")
                            print(f"    Subspan EM: {overall.get('subspan_em', 'N/A'):.4f}" if isinstance(overall.get('subspan_em'), (int, float)) else f"    Subspan EM: {overall.get('subspan_em', 'N/A')}")
                            if "f1" in overall:
                                print(f"    F1: {overall.get('f1', 'N/A'):.4f}" if isinstance(overall.get('f1'), (int, float)) else f"    F1: {overall.get('f1', 'N/A')}")
                            if "coverage" in overall:
                                print(f"    Coverage: {overall.get('coverage', 'N/A'):.4f}" if isinstance(overall.get('coverage'), (int, float)) else f"    Coverage: {overall.get('coverage', 'N/A')}")
                        else:
                            print(f"  ✓ {scenario_name}: Metrics computed")
                    except Exception as e:
                        print(f"  ⚠️  {scenario_name}: Failed to compute metrics: {e}")
                        import traceback
                        traceback.print_exc()
                        metrics_results[scenario_name] = {"error": str(e)}
                else:
                    print(f"  ⚠️  {scenario_name}: No valid predictions for metrics")
                    metrics_results[scenario_name] = {"error": "No valid predictions"}
    except Exception as e:
        print(f"  ❌ Failed to compute metrics: {e}")
        import traceback
        traceback.print_exc()

    # Save metrics
    metrics_file = os.path.join(out_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"  ✓ Saved {metrics_file}")

    # Generate summary (following test_integration.py pattern)
    print("\n[6/6] Generating summary...")
    summary = {
        "test_info": {
            "model_name": model_name,
            "num_samples": num_samples,
            "chunk_size": chunk_size,
            "loft_dataset": loft_dataset,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "scenarios": {},
    }

    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_results = all_results[scenario_name]
        
        successful = [r for r in scenario_results if r.get("response") is not None]
        failed = [r for r in scenario_results if r.get("response") is None]
        
        avg_elapsed = sum(r["elapsed_s"] for r in successful) / len(successful) if successful else None
        
        # Extract metrics
        scenario_metrics = metrics_results.get(scenario_name, {})
        overall_metrics = scenario_metrics.get("overall", {})
        
        summary["scenarios"][scenario_name] = {
            "successful": len(successful),
            "failed": len(failed),
            "avg_elapsed_s": round(avg_elapsed, 2) if avg_elapsed else None,
            "metrics": overall_metrics,
        }

    # Save summary
    summary_file = os.path.join(out_dir, "summary.json")
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
    print(f"  LOFT Dataset: {loft_dataset}")
    
    print(f"\nScenario Results:")
    for scenario_name, scenario_summary in summary["scenarios"].items():
        print(f"  {scenario_name}:")
        print(f"    Successful: {scenario_summary['successful']}/{num_samples}")
        print(f"    Avg elapsed: {scenario_summary['avg_elapsed_s']}s")
        metrics = scenario_summary.get("metrics", {})
        if metrics:
            print(f"    Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"      {metric_name}: {metric_value:.4f}")
                else:
                    print(f"      {metric_name}: {metric_value}")
    
    # Flush micro metrics
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
            print(f"  ⚠️  Micro metrics file not found")
    else:
        print(f"  ⚠️  Micro metrics not configured")
    
    print(f"\nResults saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # Allow overriding parameters via env vars
    model_env: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    num_samples_env: int = int(os.environ.get("NUM_SAMPLES", "5"))
    chunk_size_env: int = int(os.environ.get("PREFILL_CHUNK_SIZE", "1024"))
    loft_dataset_env: str = os.environ.get("LOFT_DATASET", "hotpotqa_32k")
    
    # Output dir will be auto-generated with descriptive name
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

    print(f"Starting LOFT RAG test with:")
    print(f"  MODEL_NAME={model_env}")
    print(f"  NUM_SAMPLES={num_samples_env}")
    print(f"  PREFILL_CHUNK_SIZE={chunk_size_env}")
    print(f"  LOFT_DATASET={loft_dataset_env}")
    print(f"  ORACLE_TOPK_HEAVY_SIZE={os.environ.get('ORACLE_TOPK_HEAVY_SIZE', '0.1')}")
    print(f"  SINK_SIZE={os.environ.get('SINK_SIZE', '128')}")
    print(f"  LOCAL_WINDOW_SIZE={os.environ.get('LOCAL_WINDOW_SIZE', '128')}")
    print(f"  OUTPUT_DIR={output_dir_env if output_dir_env else '(auto-generated)'}")
    print(f"  SPARSE_DEBUG={os.environ.get('SPARSE_DEBUG', '0')}")
    
    run_loft_rag_test(
        model_name=model_env,
        num_samples=num_samples_env,
        chunk_size=chunk_size_env,
        loft_dataset=loft_dataset_env,
    )

