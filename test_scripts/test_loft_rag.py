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


def convert_loft_to_simple_format(context: str, question: str, answer_prefix: str) -> tuple[str, str, str]:
    """Convert LOFT format to simple HotpotQA-style format.
    
    Extracts query text and corpus documents, formats using HotpotQA template.
    This improves compatibility with chat templates and model understanding.
    
    Args:
        context: LOFT context (corpus docs + few-shot examples)
        question: LOFT question format ("====== Now let's start! ======\nquery: <text>")
        answer_prefix: Answer prefix (e.g., "Final Answer: ")
    
    Returns:
        Tuple of (converted_context, converted_question, answer_prefix)
        - converted_context: HotpotQA-style formatted context with corpus docs
        - converted_question: Simple "Question: <query>\n" format
        - answer_prefix: Unchanged (for LOFT metrics compatibility)
    """
    # Extract query text from LOFT question format
    # Format: "====== Now let's start! ======\nquery: <query_text>"
    query_text = ""
    
    # Strategy 1: Look for "query:" marker
    query_marker = "query:"
    query_marker_lower = query_marker.lower()
    question_lower = question.lower()
    
    if query_marker_lower in question_lower:
        # Find position of "query:" marker
        marker_idx = question_lower.find(query_marker_lower)
        # Extract text after "query:" (skip the marker and any whitespace)
        query_text = question[marker_idx + len(query_marker):].strip()
        # Remove any trailing newlines/whitespace
        query_text = query_text.strip()
    else:
        # Fallback: Look for separator and extract everything after it
        separator = "====== Now let's start! ======"
        if separator in question:
            separator_idx = question.find(separator)
            query_text = question[separator_idx + len(separator):].strip()
            # Remove "query:" if present
            if query_text.lower().startswith("query:"):
                query_text = query_text[6:].strip()
        else:
            # Last resort: use question as-is (shouldn't happen with proper LOFT format)
            query_text = question.strip()
    
    if not query_text:
        # If we couldn't extract query, use original question (fallback)
        query_text = question.strip()
    
    # Extract corpus documents from LOFT context
    # LOFT context structure:
    # 1. Corpus instruction ("You will be given a list of documents...")
    # 2. Formatting instruction ("Your final answer should be in a list...")
    # 3. Corpus documents (ID: 0 | TITLE: ... | CONTENT: ... | END ID: 0)
    # 4. Few-shot examples ("====== Example 1 ======" ... "====== Example 5 ======")
    #
    # We need to extract ONLY the corpus documents (part 3), removing:
    # - Corpus instruction
    # - Formatting instruction  
    # - Few-shot examples
    
    corpus_docs = ""
    
    # Strategy: Find where corpus documents start and end
    # Corpus documents start after the formatting instruction (which ends with "Final Answer: ['answer']")
    # Corpus documents end before the first few-shot example ("====== Example 1 ======")
    
    # Find the start of corpus documents
    # Look for the first "ID: 0 |" pattern (or "ID: " pattern)
    corpus_start_marker = "ID: "
    corpus_start_idx = context.find(corpus_start_marker)
    
    if corpus_start_idx == -1:
        # Fallback: if no ID marker found, try to find where documents might start
        # Look for common document patterns
        corpus_start_idx = 0
    
    # Find the end of corpus documents (start of few-shot examples)
    few_shot_marker = "====== Example 1 ======"
    corpus_end_idx = context.find(few_shot_marker)
    
    if corpus_end_idx == -1:
        # Fallback: look for other few-shot patterns
        few_shot_alt = "\n====== Example"
        corpus_end_idx = context.find(few_shot_alt)
    
    if corpus_end_idx == -1:
        # If no few-shot examples found, use entire context after corpus_start_idx
        # But warn - this means few-shot examples might be included
        print(f"  ⚠️  Warning: Few-shot marker not found, using context from ID marker onwards")
        corpus_docs = context[corpus_start_idx:].strip()
    else:
        # Extract corpus documents (between start and end)
        corpus_docs = context[corpus_start_idx:corpus_end_idx].strip()
    
    # Clean up: remove any trailing newlines/whitespace
    corpus_docs = corpus_docs.strip()
    
    # CRITICAL FIX: Remove ID markers to prevent model confusion
    # LOFT format: "ID: X | TITLE: ... | CONTENT: ... | END ID: X"
    # Model was outputting document IDs (like "430") instead of answers
    # Remove ID markers and convert to simple format: "TITLE: ...\nCONTENT: ..."
    import re
    # Pattern: "ID: X | TITLE: ... | CONTENT: ... | END ID: X"
    # Replace with: "TITLE: ...\nCONTENT: ...\n"
    corpus_docs_cleaned = re.sub(
        r'ID: \d+ \| ',  # Remove "ID: X | "
        '',
        corpus_docs
    )
    corpus_docs_cleaned = re.sub(
        r' \| END ID: \d+',  # Remove " | END ID: X"
        '',
        corpus_docs_cleaned
    )
    # Also clean up any remaining "|" separators between TITLE and CONTENT
    corpus_docs_cleaned = re.sub(
        r' \| TITLE: ',  # Replace " | TITLE: " with "\nTITLE: "
        '\nTITLE: ',
        corpus_docs_cleaned
    )
    corpus_docs_cleaned = re.sub(
        r' \| CONTENT: ',  # Replace " | CONTENT: " with "\nCONTENT: "
        '\nCONTENT: ',
        corpus_docs_cleaned
    )
    
    # CRITICAL: Remove TITLE:/CONTENT: markers to match HotpotQA format exactly
    # HotpotQA uses simple "Title: ...\n\n..." format without TITLE:/CONTENT: markers
    # Convert "TITLE: X\nCONTENT: Y" → "Title: X\n\nY"
    corpus_docs_cleaned = re.sub(
        r'TITLE: (.+?)\nCONTENT: (.+?)(?=\nTITLE:|\n*$)',  # Match TITLE: ...\nCONTENT: ...
        r'Title: \1\n\n\2',  # Replace with "Title: ...\n\n..."
        corpus_docs_cleaned,
        flags=re.DOTALL
    )
    # Handle last document if it doesn't have trailing newline
    corpus_docs_cleaned = re.sub(
        r'TITLE: (.+?)\nCONTENT: (.+?)$',  # Match last document
        r'Title: \1\n\n\2',
        corpus_docs_cleaned,
        flags=re.DOTALL
    )
    # Remove any remaining TITLE: or CONTENT: markers (fallback)
    corpus_docs_cleaned = re.sub(r'^TITLE: ', 'Title: ', corpus_docs_cleaned, flags=re.MULTILINE)
    corpus_docs_cleaned = re.sub(r'^CONTENT: ', '', corpus_docs_cleaned, flags=re.MULTILINE)
    
    corpus_docs = corpus_docs_cleaned.strip()
    
    # Verify extraction worked
    if "====== Example" in corpus_docs:
        print(f"  ⚠️  ERROR: Few-shot examples still in extracted corpus_docs!")
    if "You will be given a list of documents" in corpus_docs:
        print(f"  ⚠️  ERROR: Corpus instructions still in extracted corpus_docs!")
    
    # If we couldn't extract corpus docs, fallback to using context as-is
    # (but this shouldn't happen with proper LOFT format)
    if not corpus_docs:
        print(f"  ⚠️  Warning: Could not extract corpus documents, using full context")
        corpus_docs = context.strip()
    
    # Format context using HotpotQA-style template
    # HotpotQA template: "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    hotpotqa_context_template = (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{corpus_docs}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    )
    
    # Format question using HotpotQA-style template
    # HotpotQA format: "Question: {input}\n"
    hotpotqa_question_template = "Question: {query}\n"
    
    # Build converted format
    converted_context = hotpotqa_context_template.format(corpus_docs=corpus_docs)
    converted_question = hotpotqa_question_template.format(query=query_text)
    
    # OPTIMIZATION: Change answer prefix from "Final Answer:" to "Answer:" to match HotpotQA exactly
    # This improves model understanding and consistency
    if answer_prefix == "Final Answer: " or answer_prefix == "Final Answer:":
        answer_prefix = "Answer: "  # Match HotpotQA format exactly
    
    return converted_context, converted_question, answer_prefix


def run_loft_rag_test(
    model_name: str = "meta-llama",
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
        # Auto-generate output directory name with timestamp to avoid conflicts in parallel runs
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            "test_outputs",
            f"test_loft_{loft_dataset}_{model_short}_ns{num_samples}_pcs{chunk_size}{repo_suffix}_{timestamp}"
        )
        os.makedirs(out_dir, exist_ok=True)
    
    # Setup micro metric logger (same as test_integration.py)
    # Metrics will only be logged when sparse attention code runs (dense doesn't call logging)
    # Enable same metrics as HotpotQA runs: research_attention_weight_diff + research_mask_roped_vs_unroped (if flag set)
    metric_logger = MicroMetricLogger()
    enabled_metrics: List[str] = [
        "research_attention_weight_diff",  # Attention weight difference metrics
    ]
    
    # Enable mask comparison metric if flag is set (matching test_integration.py)
    if os.environ.get("COMPARE_MASK_ROPED_VS_UNROPED", "0").lower() in ("1", "true", "yes"):
        enabled_metrics.append("research_mask_roped_vs_unroped")
    
    metric_logger.configure_logging(
        log_path=out_dir,  # log_path should be directory, not file path (flush() appends "micro_metrics.jsonl")
        enabled_metrics=enabled_metrics,
    )
    print(f"[MicroMetrics] Configured logger with enabled metrics: {metric_logger.get_enabled_metrics()}")
    
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
    
    # Create LLM I/O JSONL file for exact prompts and responses
    llm_io_file = os.path.join(out_dir, "llm_io.jsonl")
    print(f"  📝 LLM I/O (exact prompts & responses) will be appended to: {llm_io_file}")

    # Process samples (following exact pattern from test_integration.py)
    print("\n[3/6] Processing samples...")
    for i, row in df.iterrows():
        # Load original LOFT format
        original_context = row["context"]
        original_question = row["question"]
        answer_prefix = row.get("answer_prefix", "Final Answer: ")
        # Use max_new_tokens = 52 (HotpotQA default) to allow complete answers
        max_new_tokens = 52
        task = row.get("task", loft_dataset)
        
        # Convert LOFT format to simple HotpotQA-style format
        # This improves compatibility with chat templates and model understanding
        context, question, answer_prefix = convert_loft_to_simple_format(
            original_context, original_question, answer_prefix
        )
        
        # Debug: Verify conversion worked
        if "====== Example 1 ======" in context:
            print(f"    ⚠️  WARNING: Few-shot examples still present in converted context!")
        if "You will be given a list of documents" in context:
            print(f"    ⚠️  WARNING: Corpus instructions still present in converted context!")
        if "Answer the question based on the given passages" not in context:
            print(f"    ⚠️  WARNING: HotpotQA template not found in converted context!")
        
        # Store original format for CSV logging (so we can see what was used)
        original_format = {
            "original_context": original_context,
            "original_question": original_question,
            "converted_context": context,
            "converted_question": question,
        }
        
        # Handle context repetition if needed (after conversion)
        if repeat_count > 1:
            context = context * repeat_count
        
        print(f"\n  Sample {i+1}/{len(df)}: {task}")
        print(f"    Original context length: {len(original_context)} chars")
        print(f"    Converted context length: {len(context)} chars")
        print(f"    Original question: {original_question[:80]}...")
        print(f"    Converted question: {question[:80]}...")
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            adapter = scenario["adapter"]
            use_chunked = scenario["chunked"]
            
            # Get exact prompt that will be sent to LLM (after preprocessing)
            # Compute this BEFORE processing so we can log it even if processing fails
            exact_prompt_to_llm = None
            try:
                preprocessed_context, preprocessed_questions = adapter._preprocess_context_and_questions(
                    context, [question] if isinstance(question, str) else question, answer_prefix
                )
                # The exact prompt sent to LLM is: preprocessed_context + preprocessed_question
                exact_prompt_to_llm = preprocessed_context + (preprocessed_questions[0] if preprocessed_questions else "")
            except Exception as prep_error:
                # If preprocessing fails, log the error but continue
                print(f"    ⚠️  Warning: Could not preprocess prompt: {prep_error}")
                exact_prompt_to_llm = f"[PREPROCESSING_ERROR: {str(prep_error)}] Context: {context[:200]}... Question: {question[:200]}..."
            
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
                
                # Clean chat template artifacts (remove "assistant\n\n" prefixes)
                # Some models output chat template artifacts that need to be removed
                response_text = response_text.strip()
                if response_text.startswith("assistant"):
                    # Remove "assistant" prefix and any following newlines/whitespace
                    response_text = response_text[len("assistant"):].strip()
                    # Remove leading newlines
                    while response_text.startswith("\n"):
                        response_text = response_text[1:].strip()
                
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
                
                # Log exact LLM I/O (prompt sent to LLM + response received)
                # exact_prompt_to_llm was computed before processing (line 389)
                llm_io_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # Include milliseconds
                    "scenario": scenario_name,
                    "tag": "dense" if not scenario.get("sparse", False) else "sparse",
                    "sample_idx": i,
                    "task": task,
                    "exact_prompt_to_llm": exact_prompt_to_llm,  # Exact prompt after all preprocessing (chat template, etc.)
                    "exact_response_from_llm": response_text,  # Exact response from LLM
                    "elapsed_s": elapsed,
                }
                with open(llm_io_file, "a") as f:
                    f.write(json.dumps(llm_io_entry) + "\n")
                
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
                    "context": context,  # Converted context (HotpotQA-style)
                    "question": question,  # Converted question (HotpotQA-style)
                    "predicted_answer": response_text,
                    "elapsed_s": elapsed,
                    "answers": row.get("answers", None),
                    "task": task,
                    "method": scenario_name,
                    "answer_prefix": answer_prefix,
                    # Store original LOFT format for reference
                    "original_context": original_context,
                    "original_question": original_question,
                })
                
                # Flush metrics after each sample/scenario (matching test_integration.py)
                if scenario.get("sparse", False):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        metric_logger.flush()
                    except Exception:
                        pass
                
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
                
                # Log error in LLM I/O file as well
                # Use exact_prompt_to_llm if we computed it, otherwise None
                llm_io_error_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "scenario": scenario_name,
                    "tag": "dense" if not scenario.get("sparse", False) else "sparse",
                    "sample_idx": i,
                    "task": task,
                    "exact_prompt_to_llm": exact_prompt_to_llm,  # May be None if preprocessing failed
                    "exact_response_from_llm": None,
                    "elapsed_s": None,
                    "error": str(e),
                }
                with open(llm_io_file, "a") as f:
                    f.write(json.dumps(llm_io_error_entry) + "\n")
                
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
                    "context": context,  # Converted context (HotpotQA-style)
                    "question": question,  # Converted question (HotpotQA-style)
                    "predicted_answer": None,
                    "elapsed_s": None,
                    "answers": row.get("answers", None),
                    "task": task,
                    "method": scenario_name,
                    "answer_prefix": answer_prefix,
                    "error": str(e),
                    # Store original LOFT format for reference
                    "original_context": original_context,
                    "original_question": original_question,
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
                # CRITICAL: Prepend answer_prefix to response for extract_prediction() to work
                # extract_prediction() expects prefix in response (e.g., "Answer: Mazda")
                # But model outputs just the answer (e.g., "Mazda")
                for idx, row in df_scenario.iterrows():
                    sample_idx = row["sample_idx"]
                    if sample_idx < len(df_for_metrics):
                        response = str(row.get("response", "")).strip()
                        answer_prefix = str(df_for_metrics.iloc[sample_idx].get("answer_prefix", "Answer: ")).strip()
                        # Prepend prefix to response for proper extraction
                        # Ensure prefix ends with space if it doesn't already
                        if answer_prefix and not answer_prefix.endswith(" "):
                            answer_prefix = answer_prefix + " "
                        predicted_answer_with_prefix = answer_prefix + response
                        df_for_metrics.loc[df_for_metrics.index[sample_idx], "predicted_answer"] = predicted_answer_with_prefix
                
                # Ensure answers column is in correct format (list of strings)
                # CSV stores answers as string like "['Mazda']", need to parse
                import ast
                def parse_answers(val):
                    if pd.isna(val):
                        return []
                    if isinstance(val, list):
                        return [str(a) for a in val]
                    if isinstance(val, str):
                        try:
                            parsed = ast.literal_eval(val)
                            if isinstance(parsed, list):
                                return [str(a) for a in parsed]
                            return [str(parsed)]
                        except:
                            return [str(val)]
                    return [str(val)]
                
                df_for_metrics["answers"] = df_for_metrics["answers"].apply(parse_answers)
                
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

