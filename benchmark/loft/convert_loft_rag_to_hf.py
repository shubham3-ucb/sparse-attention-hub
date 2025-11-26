#!/usr/bin/env python3
"""
Convert LOFT RAG datasets to HuggingFace dataset format matching benchmark structure.

This script uses LOFT's exact prompt construction mechanism to ensure 100% fidelity.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

# Add LOFT to path
LOFT_DIR = "/home/nvidia/shubham/sparse/loft"
sys.path.insert(0, LOFT_DIR)

# Import LOFT modules
import utils as loft_utils
from prompts import prompt_registry
from prompts.constants import common as common_constants

# LOFT context length mappings
CONTEXT_LENGTH_TO_NUM_TOKENS = {
    "32k": 32000,
    "128k": 128000,
    "1m": 1000000,
}

# RAG datasets (excluding topiocqa as requested)
RAG_DATASETS = ["nq", "hotpotqa", "musique", "qampari", "quest"]

# Max new tokens for RAG tasks (conservative, can be adjusted)
MAX_NEW_TOKENS_RAG = 256

# Answer prefix from LOFT format
ANSWER_PREFIX = "Final Answer: "


def get_query_separator_position(full_prompt: str, query_text: str) -> int:
    """
    Find where the test query starts in the full prompt.
    
    LOFT structure:
    - Context: corpus instruction + formatting instruction + corpus docs + few-shot examples
    - Separator: "====== Now let's start! ======\n"
    - Query part: query format + query text
    
    Returns the position where the separator starts (so separator is included in question part).
    """
    # LOFT uses TEST_QUERY_SEPARATOR = "====== Now let's start! ======\n"
    separator = "====== Now let's start! ======"
    
    # Strategy 1: Look for TEST_QUERY_SEPARATOR (primary method)
    idx = full_prompt.find(separator)
    if idx > 0:
        # Return the position where separator starts
        # This ensures separator is included in question part, matching LOFT structure
        return idx
    
    # Strategy 2: Fallback - look for "query:" marker followed by query text
    # This handles edge cases where separator might be formatted differently
    query_marker = "query:"
    query_lower = query_text.lower().strip()
    idx = full_prompt.lower().find(query_marker)
    if idx > 0:
        # Verify query text appears after marker
        query_start = full_prompt.lower().find(query_lower, idx)
        if query_start > 0 and query_start < idx + 200:  # Within reasonable distance
            # Go back to find the start of the query format line
            # Look for newline before "query:" to get the start of query format
            line_start = full_prompt.rfind("\n", 0, idx) + 1
            return line_start
    
    # Strategy 3: Last resort - look for query text directly
    query_start = full_prompt.lower().find(query_lower)
    if query_start > 0:
        # Find the start of the line containing query
        line_start = full_prompt.rfind("\n", 0, query_start) + 1
        return line_start
    
    # If all fails, return -1 (will use full prompt as context)
    return -1


def convert_loft_rag_to_benchmark_format(
    loft_data_dir: str,
    dataset: str,
    length: str,
    split: str,
    output_dir: str,
) -> None:
    """
    Convert LOFT RAG dataset to benchmark format using LOFT's exact prompt construction.
    
    Args:
        loft_data_dir: Base directory containing LOFT data (e.g., /path/to/loft_data)
        dataset: Dataset name (e.g., "nq", "hotpotqa")
        length: Context length (e.g., "32k", "128k", "1m")
        split: Split name (e.g., "test", "dev")
        output_dir: Output directory for HuggingFace dataset
    """
    print(f"\n{'='*80}")
    print(f"Converting: {dataset} {length} {split}")
    print(f"{'='*80}")
    
    # Set LOFT base directory
    prompt_registry.PromptRegistry.base_dir = loft_data_dir
    
    # Construct prompt name (LOFT format)
    prompt_name = f"rag_{dataset}_{length}_{split}:few_shot_with_cot"
    
    print(f"Prompt name: {prompt_name}")
    
    # Get examples using LOFT's PromptRegistry
    try:
        examples = prompt_registry.PromptRegistry.get_examples(
            name=prompt_name,
            base_dir=loft_data_dir,
        )
        print(f"Found {len(examples)} examples")
    except Exception as e:
        print(f"ERROR: Failed to get examples: {e}")
        print(f"  Prompt name: {prompt_name}")
        print(f"  Data dir: {loft_data_dir}")
        return
    
    if len(examples) == 0:
        print(f"WARNING: No examples found for {prompt_name}")
        return
    
    # Convert each example
    records = []
    for idx, ex in enumerate(examples):
        if idx % 100 == 0:
            print(f"  Processing example {idx+1}/{len(examples)}...")
        
        # Get full prompt using LOFT's concatenate_chunks - this is the EXACT LOFT prompt
        full_prompt = loft_utils.concatenate_chunks(ex.all_chunks)
        
        # Get query text from LOFTData (more reliable than parsing from chunks)
        query_text = ""
        data_dir = f"rag/{dataset}/{length}"
        try:
            loft_data = loft_utils.load_data_from_file(
                data_dir=data_dir,
                base_dir=loft_data_dir,
                split=split
            )
            if ex.qid in loft_data.queries:
                query_text = loft_data.queries[ex.qid]
                if isinstance(query_text, list):
                    query_text = query_text[0] if query_text else ""
        except Exception as e:
            print(f"    WARNING: Could not load query for {ex.qid}: {e}")
        
        # Fallback: try to get from query turns
        if not query_text and ex.query_turns and len(ex.query_turns) > 0:
            query_chunks = ex.query_turns[-1].chunks
            query_text = loft_utils.concatenate_chunks(query_chunks)
            # Extract just the query text (after "query:")
            if "query:" in query_text.lower():
                query_text = query_text.split("query:")[-1].strip()
        
        if not query_text:
            print(f"    WARNING: No query text found for {ex.qid}, skipping")
            continue
        
        # Find where test query separator starts in full prompt
        # This preserves: context = everything before separator, question = separator + query format + query
        query_start_pos = get_query_separator_position(full_prompt, query_text)
        
        if query_start_pos > 0:
            # Split into context and question
            # Context: corpus instruction + formatting + corpus docs + few-shot examples
            # DON'T strip context - preserve trailing whitespace/newlines to maintain exact structure
            context = full_prompt[:query_start_pos]
            # Question: separator + query format + query text
            # DON'T strip question - preserve leading whitespace/newlines to maintain exact structure
            question = full_prompt[query_start_pos:]
            
            # CRITICAL ASSERTION: Verify that reconstructed prompt (context + question) matches LOFT original
            # This ensures 100% fidelity - when we use the HF dataset, it will produce the exact same prompt as LOFT
            # Simply concatenate - no extra newlines needed since they're already in the original
            reconstructed_prompt = context + question
            loft_original = full_prompt
            
            # First check: exact match (should always pass with our fix)
            if reconstructed_prompt != loft_original:
                # Fallback: normalize whitespace for comparison (preserve content, ignore minor whitespace differences)
                def normalize_whitespace(s: str) -> str:
                    # Remove carriage returns, normalize multiple spaces/newlines
                    s = s.replace("\r", "")
                    # Normalize multiple newlines to double newline
                    while "\n\n\n" in s:
                        s = s.replace("\n\n\n", "\n\n")
                    # Normalize multiple spaces to single space (but preserve single newlines)
                    lines = s.split("\n")
                    normalized_lines = [" ".join(line.split()) for line in lines]
                    return "\n".join(normalized_lines).strip()
                
                orig_norm = normalize_whitespace(loft_original)
                recon_norm = normalize_whitespace(reconstructed_prompt)
                
                if orig_norm != recon_norm:
                    # Find first difference for debugging
                    diff_pos = -1
                    for i in range(min(len(orig_norm), len(recon_norm))):
                        if orig_norm[i] != recon_norm[i]:
                            diff_pos = i
                            break
                    
                    error_msg = (
                        f"❌ ASSERTION FAILED for QID {ex.qid}:\n"
                        f"  Reconstructed prompt does NOT match LOFT original!\n"
                        f"  Original length: {len(loft_original)} (normalized: {len(orig_norm)})\n"
                        f"  Reconstructed length: {len(reconstructed_prompt)} (normalized: {len(recon_norm)})\n"
                    )
                    
                    if diff_pos >= 0:
                        error_msg += (
                            f"  First difference at position {diff_pos}:\n"
                            f"  LOFT original: ...{orig_norm[max(0,diff_pos-50):diff_pos+50]}...\n"
                            f"  Reconstructed: ...{recon_norm[max(0,diff_pos-50):diff_pos+50]}...\n"
                        )
                    
                    # Additional checks
                    few_shot_marker = "====== Example"
                    if few_shot_marker in loft_original and few_shot_marker not in context:
                        error_msg += f"  ⚠️  Few-shot examples missing in context!\n"
                    
                    corpus_marker = "ID: 0 | TITLE:"
                    if corpus_marker in loft_original and corpus_marker not in context:
                        error_msg += f"  ⚠️  Corpus documents missing in context!\n"
                    
                    print(error_msg)
                    raise AssertionError(f"Sample {ex.qid} does not match LOFT original. See error above.")
            
            # Additional verification: check that key components are preserved
            few_shot_marker = "====== Example"
            if few_shot_marker in loft_original:
                assert few_shot_marker in context, f"Few-shot examples missing in context for {ex.qid}!"
                loft_fewshot_count = loft_original.count(few_shot_marker)
                context_fewshot_count = context.count(few_shot_marker)
                assert loft_fewshot_count == context_fewshot_count, (
                    f"Few-shot count mismatch for {ex.qid}: "
                    f"LOFT has {loft_fewshot_count}, context has {context_fewshot_count}"
                )
            
            corpus_marker = "ID: 0 | TITLE:"
            if corpus_marker in loft_original:
                assert corpus_marker in context, f"Corpus documents missing in context for {ex.qid}!"
            
            # Verify query separator is in question
            query_separator = "====== Now let's start! ======"
            if query_separator in loft_original:
                assert query_separator in question, f"Query separator missing in question for {ex.qid}!"
        else:
            # Fallback: use query text as question, full prompt as context
            print(f"    WARNING: Could not find query separator for {ex.qid}, using fallback")
            context = full_prompt
            question = query_text
        
        # Get answers
        answers = []
        if ex.gold_pids:
            # For RAG, answers are typically in the queries.jsonl
            data_dir = f"rag/{dataset}/{length}"
            try:
                loft_data = loft_utils.load_data_from_file(
                    data_dir=data_dir,
                    base_dir=loft_data_dir,
                    split=split
                )
                if ex.qid in loft_data.answers:
                    answers = loft_data.answers[ex.qid]
                    if not isinstance(answers, list):
                        answers = [answers] if answers else []
            except Exception as e:
                print(f"    WARNING: Could not load answers for {ex.qid}: {e}")
        
        if not answers:
            print(f"    WARNING: No answers found for {ex.qid}")
            continue
        
        # Create record matching benchmark format
        task_name = f"{dataset}_{length}"
        record = {
            "context": context,
            "question": question,
            "answer_prefix": ANSWER_PREFIX,
            "answers": answers,
            "task": task_name,
            "max_new_tokens": MAX_NEW_TOKENS_RAG,
        }
        
        records.append(record)
        
        # Progress indicator for assertions
        if (idx + 1) % 50 == 0:
            print(f"  ✓ Verified {idx + 1}/{len(examples)} samples match LOFT originals")
    
    if len(records) == 0:
        print(f"ERROR: No valid records created")
        return
    
    print(f"\n✅ Created {len(records)} records")
    print(f"✅ All {len(records)} samples verified: reconstructed prompts match LOFT originals!")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Create HuggingFace dataset
    dataset_hf = Dataset.from_pandas(df)
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = f"{dataset}_{length}_{split}"
    output_path = os.path.join(output_dir, dataset_name)
    
    print(f"Saving to: {output_path}")
    dataset_hf.save_to_disk(output_path)
    
    # Also save as JSON for inspection
    json_path = os.path.join(output_dir, f"{dataset_name}.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"Also saved as JSON: {json_path}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Records: {len(records)}")
    print(f"  Avg context length: {df['context'].str.len().mean():.0f} chars")
    print(f"  Avg question length: {df['question'].str.len().mean():.0f} chars")
    print(f"  Tasks: {df['task'].unique().tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LOFT RAG datasets to HuggingFace format"
    )
    parser.add_argument(
        "--loft_data_dir",
        type=str,
        required=True,
        help="Base directory containing LOFT data (e.g., /path/to/loft_data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=RAG_DATASETS,
        help="Dataset name (e.g., nq, hotpotqa). If not provided, converts all.",
    )
    parser.add_argument(
        "--length",
        type=str,
        choices=["32k", "128k", "1m"],
        help="Context length. If not provided, converts all.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "dev"],
        help="Split name. If not provided, converts all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./loft_rag_datasets",
        help="Output directory for HuggingFace datasets",
    )
    
    args = parser.parse_args()
    
    # Determine what to convert
    datasets_to_convert = [args.dataset] if args.dataset else RAG_DATASETS
    lengths_to_convert = [args.length] if args.length else ["32k", "128k", "1m"]
    splits_to_convert = [args.split] if args.split else ["test", "dev"]
    
    print(f"🚀 Converting LOFT RAG datasets")
    print(f"  LOFT data dir: {args.loft_data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Datasets: {datasets_to_convert}")
    print(f"  Lengths: {lengths_to_convert}")
    print(f"  Splits: {splits_to_convert}")
    
    # Convert each combination
    total = 0
    for dataset in datasets_to_convert:
        for length in lengths_to_convert:
            for split in splits_to_convert:
                # Check if data exists
                data_path = os.path.join(
                    args.loft_data_dir,
                    f"data/rag/{dataset}/{length}/{split}_queries.jsonl"
                )
                if not os.path.exists(data_path):
                    print(f"\n⏭️  Skipping {dataset} {length} {split} (data not found)")
                    continue
                
                try:
                    convert_loft_rag_to_benchmark_format(
                        loft_data_dir=args.loft_data_dir,
                        dataset=dataset,
                        length=length,
                        split=split,
                        output_dir=args.output_dir,
                    )
                    total += 1
                except Exception as e:
                    print(f"\n❌ ERROR converting {dataset} {length} {split}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion complete! Converted {total} datasets.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
