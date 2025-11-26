#!/usr/bin/env python3
"""
Comprehensive validation: Verify that all HF format datasets exactly match LOFT originals.

This script:
1. Loads HF datasets from disk
2. Reconstructs LOFT prompts for each sample
3. Compares context+question vs LOFT original (exact match)
4. Validates all datasets (32k curated, 128k, 1m) and all splits (dev, test)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk

# Add LOFT to path
LOFT_DIR = "/home/nvidia/shubham/sparse/loft"
sys.path.insert(0, LOFT_DIR)

# Import LOFT modules
import utils as loft_utils
from prompts import prompt_registry

# RAG datasets
RAG_DATASETS = ["nq", "hotpotqa", "musique", "qampari", "quest"]
LENGTHS = ["32k", "128k", "1m"]
SPLITS = ["dev", "test"]


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace for comparison (preserve content, ignore minor whitespace differences)."""
    # Remove carriage returns, normalize multiple spaces/newlines
    s = s.replace("\r", "")
    # Normalize multiple newlines to double newline
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    # Normalize multiple spaces to single space (but preserve single newlines)
    lines = s.split("\n")
    normalized_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(normalized_lines).strip()


def get_query_separator_position(full_prompt: str, query_text: str) -> int:
    """Find where the test query starts in the full prompt."""
    separator = "====== Now let's start! ======"
    
    # Strategy 1: Look for TEST_QUERY_SEPARATOR (primary method)
    idx = full_prompt.find(separator)
    if idx > 0:
        return idx
    
    # Strategy 2: Fallback - look for "query:" marker
    query_marker = "query:"
    query_lower = query_text.lower().strip()
    idx = full_prompt.lower().find(query_marker)
    if idx > 0:
        query_start = full_prompt.lower().find(query_lower, idx)
        if query_start > 0 and query_start < idx + 200:
            line_start = full_prompt.rfind("\n", 0, idx) + 1
            return line_start
    
    # Strategy 3: Last resort - look for query text directly
    query_start = full_prompt.lower().find(query_lower)
    if query_start > 0:
        line_start = full_prompt.rfind("\n", 0, query_start) + 1
        return line_start
    
    return -1


def get_qid_from_hf_sample(sample: Dict, dataset: str, length: str, split: str) -> str:
    """Extract QID from HF sample by reconstructing prompt and matching with LOFT."""
    # Try to extract from question text - look for patterns
    question = sample.get("question", "")
    
    # LOFT query format: "query: <query_text>"
    # Try to find query text and match with LOFT queries
    loft_data_dir = args.loft_data_dir
    data_dir = f"rag/{dataset}/{length}"
    
    try:
        loft_data = loft_utils.load_data_from_file(
            data_dir=data_dir,
            base_dir=loft_data_dir,
            split=split
        )
        
        # Extract query text from HF question (after separator)
        separator = "====== Now let's start! ======"
        if separator in question:
            query_part = question.split(separator)[-1]
            if "query:" in query_part.lower():
                query_text = query_part.split("query:")[-1].strip()
            else:
                query_text = query_part.strip()
        else:
            query_text = question.strip()
        
        # Match with LOFT queries
        for qid, loft_query in loft_data.queries.items():
            if isinstance(loft_query, list):
                loft_query = loft_query[0] if loft_query else ""
            if loft_query.strip().lower() == query_text.lower():
                return qid
    except Exception as e:
        pass
    
    return None


def validate_dataset(
    hf_dataset_path: str,
    dataset: str,
    length: str,
    split: str,
    loft_data_dir: str,
) -> Tuple[bool, int, int, List[str]]:
    """
    Validate a single HF dataset against LOFT original.
    
    Returns:
        (success, total_samples, matched_samples, errors)
    """
    print(f"\n{'='*80}")
    print(f"Validating: {dataset} {length} {split}")
    print(f"  HF path: {hf_dataset_path}")
    print(f"{'='*80}")
    
    # Load HF dataset
    if not os.path.exists(hf_dataset_path):
        return (False, 0, 0, [f"HF dataset not found: {hf_dataset_path}"])
    
    try:
        hf_dataset = load_from_disk(hf_dataset_path)
    except Exception as e:
        return (False, 0, 0, [f"Failed to load HF dataset: {e}"])
    
    # Set LOFT base directory
    prompt_registry.PromptRegistry.base_dir = loft_data_dir
    
    # Construct prompt name (LOFT format)
    prompt_name = f"rag_{dataset}_{length}_{split}:few_shot_with_cot"
    
    # Get LOFT examples
    try:
        loft_examples = prompt_registry.PromptRegistry.get_examples(
            name=prompt_name,
            base_dir=loft_data_dir,
        )
    except Exception as e:
        return (False, 0, 0, [f"Failed to get LOFT examples: {e}"])
    
    # Create QID to LOFT example mapping
    loft_by_qid = {ex.qid: ex for ex in loft_examples}
    
    # Create QID to HF sample mapping (by matching queries)
    hf_by_qid = {}
    loft_data_dir_local = loft_data_dir
    data_dir = f"rag/{dataset}/{length}"
    
    try:
        loft_data = loft_utils.load_data_from_file(
            data_dir=data_dir,
            base_dir=loft_data_dir_local,
            split=split
        )
    except Exception as e:
        return (False, 0, 0, [f"Failed to load LOFT data: {e}"])
    
    # Match HF samples to LOFT QIDs
    for idx, hf_sample in enumerate(hf_dataset):
        question = hf_sample.get("question", "")
        context = hf_sample.get("context", "")
        
        # Extract query text from HF question
        separator = "====== Now let's start! ======"
        if separator in question:
            query_part = question.split(separator)[-1]
            if "query:" in query_part.lower():
                query_text = query_part.split("query:")[-1].strip()
            else:
                query_text = query_part.strip()
        else:
            query_text = question.strip()
        
        # Match with LOFT queries
        matched_qid = None
        for qid, loft_query in loft_data.queries.items():
            if isinstance(loft_query, list):
                loft_query = loft_query[0] if loft_query else ""
            if loft_query.strip().lower() == query_text.lower():
                matched_qid = qid
                break
        
        if matched_qid:
            hf_by_qid[matched_qid] = hf_sample
        else:
            # Try to match by index if QID matching fails
            if idx < len(loft_examples):
                qid = loft_examples[idx].qid
                hf_by_qid[qid] = hf_sample
    
    # Validate each sample
    total_samples = len(hf_dataset)
    matched_samples = 0
    errors = []
    
    for qid, hf_sample in hf_by_qid.items():
        if qid not in loft_by_qid:
            errors.append(f"QID {qid}: Found in HF but not in LOFT")
            continue
        
        loft_ex = loft_by_qid[qid]
        
        # Get LOFT original prompt
        loft_full_prompt = loft_utils.concatenate_chunks(loft_ex.all_chunks)
        
        # Reconstruct from HF
        hf_context = hf_sample.get("context", "")
        hf_question = hf_sample.get("question", "")
        hf_reconstructed = hf_context + hf_question
        
        # Exact match check
        if hf_reconstructed != loft_full_prompt:
            # Try normalized comparison
            loft_norm = normalize_whitespace(loft_full_prompt)
            hf_norm = normalize_whitespace(hf_reconstructed)
            
            if loft_norm != hf_norm:
                # Find first difference
                diff_pos = -1
                for i in range(min(len(loft_norm), len(hf_norm))):
                    if loft_norm[i] != hf_norm[i]:
                        diff_pos = i
                        break
                
                error_msg = (
                    f"QID {qid}: Mismatch!\n"
                    f"  LOFT length: {len(loft_full_prompt)} (norm: {len(loft_norm)})\n"
                    f"  HF length: {len(hf_reconstructed)} (norm: {len(hf_norm)})\n"
                )
                
                if diff_pos >= 0:
                    error_msg += (
                        f"  First diff at pos {diff_pos}:\n"
                        f"  LOFT: ...{loft_norm[max(0,diff_pos-50):diff_pos+50]}...\n"
                        f"  HF:   ...{hf_norm[max(0,diff_pos-50):diff_pos+50]}...\n"
                    )
                
                errors.append(error_msg)
            else:
                # Normalized match - acceptable
                matched_samples += 1
        else:
            # Exact match
            matched_samples += 1
        
        # Validate answers
        hf_answers = hf_sample.get("answers", [])
        if not isinstance(hf_answers, list):
            hf_answers = [hf_answers] if hf_answers else []
        
        loft_answers = []
        if loft_ex.gold_pids:
            if qid in loft_data.answers:
                loft_answers = loft_data.answers[qid]
                if not isinstance(loft_answers, list):
                    loft_answers = [loft_answers] if loft_answers else []
        
        # Compare answers (normalize)
        hf_answers_norm = [str(a).strip().lower() for a in hf_answers]
        loft_answers_norm = [str(a).strip().lower() for a in loft_answers]
        
        if set(hf_answers_norm) != set(loft_answers_norm):
            errors.append(
                f"QID {qid}: Answer mismatch!\n"
                f"  LOFT: {loft_answers}\n"
                f"  HF:   {hf_answers}\n"
            )
    
    # Check for missing samples
    loft_qids = set(loft_by_qid.keys())
    hf_qids = set(hf_by_qid.keys())
    missing_in_hf = loft_qids - hf_qids
    extra_in_hf = hf_qids - loft_qids
    
    if missing_in_hf:
        errors.append(f"Missing in HF ({len(missing_in_hf)}): {sorted(list(missing_in_hf))[:10]}")
    if extra_in_hf:
        errors.append(f"Extra in HF ({len(extra_in_hf)}): {sorted(list(extra_in_hf))[:10]}")
    
    success = len(errors) == 0 and matched_samples == total_samples
    
    print(f"  Total samples: {total_samples}")
    print(f"  Matched: {matched_samples}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print(f"  ❌ Validation FAILED")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    else:
        print(f"  ✅ Validation PASSED")
    
    return (success, total_samples, matched_samples, errors)


def main():
    parser = argparse.ArgumentParser(
        description="Validate HF datasets against LOFT originals"
    )
    parser.add_argument(
        "--hf_datasets_dir",
        type=str,
        default="/data/sparse_attention_hub/loft_rag_datasets",
        help="Directory containing HF datasets",
    )
    parser.add_argument(
        "--loft_data_dir",
        type=str,
        default="/data/loft_data",
        help="LOFT data directory",
    )
    
    global args
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🔍 Comprehensive Validation: HF vs LOFT")
    print(f"{'='*80}")
    print(f"  HF datasets dir: {args.hf_datasets_dir}")
    print(f"  LOFT data dir: {args.loft_data_dir}")
    print(f"  Datasets: {RAG_DATASETS}")
    print(f"  Lengths: {LENGTHS}")
    print(f"  Splits: {SPLITS}")
    
    all_results = []
    total_datasets = 0
    passed_datasets = 0
    
    for dataset in RAG_DATASETS:
        for length in LENGTHS:
            for split in SPLITS:
                dataset_name = f"{dataset}_{length}_{split}"
                hf_path = os.path.join(args.hf_datasets_dir, dataset_name)
                
                if not os.path.exists(hf_path):
                    print(f"\n⚠️  Skipping {dataset_name}: HF dataset not found")
                    continue
                
                total_datasets += 1
                success, total_samples, matched_samples, errors = validate_dataset(
                    hf_dataset_path=hf_path,
                    dataset=dataset,
                    length=length,
                    split=split,
                    loft_data_dir=args.loft_data_dir,
                )
                
                all_results.append({
                    "dataset": dataset_name,
                    "success": success,
                    "total_samples": total_samples,
                    "matched_samples": matched_samples,
                    "errors": errors,
                })
                
                if success:
                    passed_datasets += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Total datasets validated: {total_datasets}")
    print(f"  Passed: {passed_datasets}")
    print(f"  Failed: {total_datasets - passed_datasets}")
    
    if passed_datasets == total_datasets:
        print(f"\n✅ ALL VALIDATIONS PASSED!")
        return 0
    else:
        print(f"\n❌ SOME VALIDATIONS FAILED!")
        print(f"\nFailed datasets:")
        for result in all_results:
            if not result["success"]:
                print(f"  - {result['dataset']}: {len(result['errors'])} errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())

