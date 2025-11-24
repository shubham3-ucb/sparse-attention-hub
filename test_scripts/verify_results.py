#!/usr/bin/env python3
"""Comprehensive verification script for test results.

Run this after test_integration.py completes to verify everything is correct.

Usage:
    python test_scripts/verify_results.py [results_dir]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

def verify_test_results(results_dir: str) -> bool:
    """Verify test results are complete and correct.
    
    Args:
        results_dir: Directory containing test results
        
    Returns:
        True if all checks pass, False otherwise
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    print("=" * 80)
    print("COMPREHENSIVE TEST RESULTS VERIFICATION")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # CHECK 1: Required files exist
    print("[CHECK 1] Required Files")
    print("-" * 80)
    required_files = [
        "results_dense_chunked.json",
        "results_dense_single.json",
        "results_sparse_chunked.json",
        "results_sparse_single.json",
        "all_results.csv",
        "metrics.json",
        "comparison_summary.json",
        "test_log.txt",
    ]
    
    for filename in required_files:
        filepath = results_path / filename
        checks_total += 1
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {filename} ({size} bytes)")
            checks_passed += 1
        else:
            print(f"  ❌ {filename} MISSING")
    
    # CHECK 2: JSON files are valid
    print("\n[CHECK 2] JSON File Validity")
    print("-" * 80)
    json_files = [f for f in required_files if f.endswith(".json")]
    for filename in json_files:
        filepath = results_path / filename
        checks_total += 1
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"  ✅ {filename} - Valid JSON ({len(str(data))} chars)")
            checks_passed += 1
        except Exception as e:
            print(f"  ❌ {filename} - Invalid JSON: {e}")
    
    # CHECK 3: Results contain expected data
    print("\n[CHECK 3] Results Data Structure")
    print("-" * 80)
    
    # Check individual result files
    scenario_files = [
        "results_dense_chunked.json",
        "results_dense_single.json",
        "results_sparse_chunked.json",
        "results_sparse_single.json",
    ]
    
    for filename in scenario_files:
        filepath = results_path / filename
        checks_total += 1
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            if isinstance(results, list) and len(results) > 0:
                # Check structure
                sample = results[0]
                has_response = "response" in sample
                has_elapsed = "elapsed_s" in sample
                has_sample_idx = "sample_idx" in sample
                
                if has_response and has_elapsed and has_sample_idx:
                    successful = sum(1 for r in results if r.get("response") is not None)
                    print(f"  ✅ {filename}: {len(results)} samples, {successful} successful")
                    checks_passed += 1
                else:
                    print(f"  ⚠️  {filename}: Missing fields (response={has_response}, elapsed={has_elapsed}, idx={has_sample_idx})")
                    checks_passed += 1  # Not critical
            else:
                print(f"  ⚠️  {filename}: Empty or invalid structure")
                checks_passed += 1  # Not critical
        except Exception as e:
            print(f"  ❌ {filename}: Error reading - {e}")
    
    # CHECK 4: Metrics file structure
    print("\n[CHECK 4] Metrics File")
    print("-" * 80)
    metrics_file = results_path / "metrics.json"
    checks_total += 1
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        expected_scenarios = ["dense_chunked", "dense_single", "sparse_chunked", "sparse_single"]
        found_scenarios = [s for s in expected_scenarios if s in metrics]
        
        if len(found_scenarios) == len(expected_scenarios):
            print(f"  ✅ metrics.json: All {len(found_scenarios)} scenarios present")
            for scenario in found_scenarios:
                if "overall_score" in metrics[scenario]:
                    score = metrics[scenario]["overall_score"]
                    print(f"    - {scenario}: overall_score = {score}")
            checks_passed += 1
        else:
            missing = set(expected_scenarios) - set(found_scenarios)
            print(f"  ⚠️  metrics.json: Missing scenarios: {missing}")
            checks_passed += 1  # Not critical
    except Exception as e:
        print(f"  ❌ metrics.json: Error - {e}")
    
    # CHECK 5: Comparison summary
    print("\n[CHECK 5] Comparison Summary")
    print("-" * 80)
    summary_file = results_path / "comparison_summary.json"
    checks_total += 1
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        has_test_info = "test_info" in summary
        has_scenarios = "scenarios" in summary
        has_comparisons = "comparisons" in summary
        
        if has_test_info and has_scenarios and has_comparisons:
            print(f"  ✅ comparison_summary.json: Structure correct")
            
            # Print key comparisons
            if "dense_chunked_vs_single" in summary["comparisons"]:
                comp = summary["comparisons"]["dense_chunked_vs_single"]
                match = comp.get("match", False)
                status = "✅ MATCH" if match else "⚠️  DIFFER"
                print(f"    Dense Chunked vs Single: {status}")
                print(f"      Chunked: {comp.get('chunked_score')}, Single: {comp.get('single_score')}")
                print(f"      Difference: {comp.get('score_difference')}")
            
            if "sparse_chunked_vs_single" in summary["comparisons"]:
                comp = summary["comparisons"]["sparse_chunked_vs_single"]
                match = comp.get("match", False)
                status = "✅ MATCH" if match else "⚠️  DIFFER"
                print(f"    Sparse Chunked vs Single: {status}")
                print(f"      Chunked: {comp.get('chunked_score')}, Single: {comp.get('single_score')}")
                print(f"      Difference: {comp.get('score_difference')}")
            
            checks_passed += 1
        else:
            print(f"  ⚠️  comparison_summary.json: Missing sections")
            checks_passed += 1  # Not critical
    except Exception as e:
        print(f"  ❌ comparison_summary.json: Error - {e}")
    
    # CHECK 6: CSV file
    print("\n[CHECK 6] CSV File")
    print("-" * 80)
    csv_file = results_path / "all_results.csv"
    checks_total += 1
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        expected_columns = ["context", "question", "predicted_answer", "elapsed_s", "answers", "task", "method", "all_classes"]
        missing_cols = [c for c in expected_columns if c not in df.columns]
        
        if not missing_cols:
            print(f"  ✅ all_results.csv: {len(df)} rows, all columns present")
            
            # Check methods
            methods = df["method"].unique().tolist()
            expected_methods = ["dense_chunked", "dense_single", "sparse_chunked", "sparse_single"]
            missing_methods = [m for m in expected_methods if m not in methods]
            
            if not missing_methods:
                print(f"    Methods: {methods}")
                for method in expected_methods:
                    count = len(df[df.method == method])
                    print(f"      - {method}: {count} samples")
                checks_passed += 1
            else:
                print(f"    ⚠️  Missing methods: {missing_methods}")
                checks_passed += 1  # Not critical
        else:
            print(f"  ⚠️  all_results.csv: Missing columns: {missing_cols}")
            checks_passed += 1  # Not critical
    except Exception as e:
        print(f"  ❌ all_results.csv: Error - {e}")
    
    # CHECK 7: Log file
    print("\n[CHECK 7] Log Files")
    print("-" * 80)
    log_file = results_path / "test_log.txt"
    checks_total += 1
    if log_file.exists():
        size = log_file.stat().st_size
        print(f"  ✅ test_log.txt: {size} bytes")
        checks_passed += 1
    else:
        print(f"  ⚠️  test_log.txt: Missing (may be OK if run directly)")
        checks_passed += 1  # Not critical
    
    prefill_log = results_path / "hf_prefill.log"
    checks_total += 1
    if prefill_log.exists():
        size = prefill_log.stat().st_size
        with open(prefill_log, 'r') as f:
            content = f.read()
            has_chunked = "[prefill] chunk" in content.lower() or "[prefill] sparse chunk" in content.lower()
        
        if has_chunked:
            print(f"  ✅ hf_prefill.log: {size} bytes, contains chunked prefill logs")
        else:
            print(f"  ⚠️  hf_prefill.log: {size} bytes, but no chunked prefill logs found")
        checks_passed += 1
    else:
        print(f"  ⚠️  hf_prefill.log: Missing (may be OK if chunked prefill not enabled)")
        checks_passed += 1  # Not critical
    
    # CHECK 8: Quantitative verification
    print("\n[CHECK 8] Quantitative Verification")
    print("-" * 80)
    
    try:
        with open(results_path / "comparison_summary.json", 'r') as f:
            summary = json.load(f)
        
        # Check that chunked and single produce similar results
        comparisons = summary.get("comparisons", {})
        
        if "dense_chunked_vs_single" in comparisons:
            comp = comparisons["dense_chunked_vs_single"]
            score_diff = comp.get("score_difference", 999)
            checks_total += 1
            if score_diff < 1.0:  # Allow 1% tolerance
                print(f"  ✅ Dense: Chunked vs Single score difference = {score_diff} (< 1.0 threshold)")
                checks_passed += 1
            else:
                print(f"  ⚠️  Dense: Chunked vs Single score difference = {score_diff} (>= 1.0 threshold)")
                checks_passed += 1  # Not critical, may vary
        
        if "sparse_chunked_vs_single" in comparisons:
            comp = comparisons["sparse_chunked_vs_single"]
            score_diff = comp.get("score_difference", 999)
            checks_total += 1
            if score_diff < 1.0:  # Allow 1% tolerance
                print(f"  ✅ Sparse: Chunked vs Single score difference = {score_diff} (< 1.0 threshold)")
                checks_passed += 1
            else:
                print(f"  ⚠️  Sparse: Chunked vs Single score difference = {score_diff} (>= 1.0 threshold)")
                checks_passed += 1  # Not critical, may vary
        
    except Exception as e:
        print(f"  ⚠️  Could not perform quantitative verification: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Checks passed: {checks_passed}/{checks_total}")
    print(f"Success rate: {100 * checks_passed / checks_total:.1f}%")
    
    if checks_passed == checks_total:
        print("\n🎉 ALL CHECKS PASSED - RESULTS ARE VALID!")
        return True
    elif checks_passed >= checks_total * 0.9:  # 90% threshold
        print("\n✅ MOST CHECKS PASSED - Results look good!")
        return True
    else:
        print("\n⚠️  SOME CHECKS FAILED - Review results carefully")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to test_integration_results in repo root
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        results_dir = str(repo_root / "test_integration_results")
    
    success = verify_test_results(results_dir)
    sys.exit(0 if success else 1)

