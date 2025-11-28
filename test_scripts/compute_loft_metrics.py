#!/usr/bin/env python3
"""Standalone script to compute LOFT metrics from responses.jsonl or all_results.csv.

Usage:
    python test_scripts/compute_loft_metrics.py <output_dir>
    
Example:
    python test_scripts/compute_loft_metrics.py test_outputs/test_loft_hotpotqa_32k_llama3_8b_ns30_pcs1024_repo_k0_s0_extctx

This script:
- Reads responses.jsonl or all_results.csv from the output directory
- Extracts responses and ground truth answers
- Computes LOFT metrics (EM, Subspan EM, F1 for single-value; EM, Coverage, Subspan EM for multi-value)
- Saves metrics.json and prints a summary
- Uses the exact same metric computation as LOFT benchmark
"""

import json
import os
import sys
import pandas as pd
import ast
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from benchmark.loft.calculate_metrics import calculate_metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_loft_metrics.py <output_dir>")
        print("  output_dir: Directory containing responses.jsonl or all_results.csv")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Computing LOFT Metrics")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    responses_file = os.path.join(output_dir, "responses.jsonl")
    all_results_file = os.path.join(output_dir, "all_results.csv")
    
    if os.path.exists(all_results_file):
        print(f"Reading from: {all_results_file}")
        df = pd.read_csv(all_results_file)
        
        # Parse answers column (might be stored as string representation of list)
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
        
        df["answers"] = df["answers"].apply(parse_answers)
        
        # Ensure predicted_answer has answer_prefix prepended
        # If predicted_answer doesn't have prefix, prepend it
        def ensure_prefix(row):
            predicted = str(row.get("predicted_answer", "")).strip()
            prefix = str(row.get("answer_prefix", "Answer: ")).strip()
            
            if not predicted:
                return ""
            
            # Check if prefix is already in predicted_answer
            if prefix.lower() in predicted.lower():
                return predicted
            
            # Prepend prefix
            if prefix and not prefix.endswith(" "):
                prefix = prefix + " "
            return prefix + predicted
        
        df["predicted_answer"] = df.apply(ensure_prefix, axis=1)
        
        # Filter to rows with predictions
        df = df[df["predicted_answer"].notna() & (df["predicted_answer"] != "")].copy()
        
        # Group by method (scenario) if method column exists
        if "method" in df.columns:
            scenarios = df["method"].unique()
        else:
            scenarios = ["all"]
            df["method"] = "all"
        
        metrics_results = {}
        
        for scenario in scenarios:
            df_scenario = df[df["method"] == scenario].copy()
            
            if df_scenario.empty:
                print(f"  ⚠️  {scenario}: No data")
                continue
            
            # Group by task
            if "task" not in df_scenario.columns:
                print(f"  ⚠️  {scenario}: Missing 'task' column")
                continue
            
            task_groups = df_scenario.groupby("task")
            task_metrics = {}
            all_em_scores = []
            all_subspan_em_scores = []
            all_f1_scores = []
            all_coverage_scores = []
            
            for task_name, task_df in task_groups:
                try:
                    metrics = calculate_metrics(task_df)
                    
                    if "error" in metrics:
                        print(f"  ❌ {scenario}/{task_name}: {metrics['error']}")
                        continue
                    
                    task_metrics[task_name] = metrics
                    
                    # Aggregate for overall
                    all_em_scores.append(metrics["em"])
                    all_subspan_em_scores.append(metrics["subspan_em"])
                    
                    if "f1" in metrics:
                        all_f1_scores.append(metrics["f1"])
                    if "coverage" in metrics:
                        all_coverage_scores.append(metrics["coverage"])
                    
                    print(f"  ✓ {scenario}/{task_name}: EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}", end="")
                    if "f1" in metrics:
                        print(f", F1={metrics['f1']:.4f}", end="")
                    if "coverage" in metrics:
                        print(f", Coverage={metrics['coverage']:.4f}", end="")
                    print()
                    
                except Exception as e:
                    print(f"  ❌ {scenario}/{task_name}: Error - {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Compute overall metrics for this scenario
            overall_metrics = {
                "overall": {
                    "em": float(sum(all_em_scores) / len(all_em_scores)) if all_em_scores else 0.0,
                    "subspan_em": float(sum(all_subspan_em_scores) / len(all_subspan_em_scores)) if all_subspan_em_scores else 0.0,
                },
                "task_metrics": {task: {k: round(v, 4) if isinstance(v, float) else v 
                                     for k, v in metrics.items()} 
                              for task, metrics in task_metrics.items()},
                "summary": {
                    "total_tasks": len(task_metrics),
                    "total_samples": len(df_scenario)
                }
            }
            
            if all_f1_scores:
                overall_metrics["overall"]["f1"] = float(sum(all_f1_scores) / len(all_f1_scores))
            if all_coverage_scores:
                overall_metrics["overall"]["coverage"] = float(sum(all_coverage_scores) / len(all_coverage_scores))
            
            # Round overall metrics
            overall_metrics["overall"] = {k: round(v, 4) if isinstance(v, float) else v 
                                          for k, v in overall_metrics["overall"].items()}
            
            metrics_results[scenario] = overall_metrics
    
    elif os.path.exists(responses_file):
        print(f"Reading from: {responses_file}")
        
        # Read responses.jsonl
        responses = []
        with open(responses_file, 'r') as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
        
        if not responses:
            print("Error: No responses found in responses.jsonl")
            sys.exit(1)
        
        # Need to load ground truth from all_results.csv or llm_io.jsonl
        # Try to find all_results.csv first
        if os.path.exists(all_results_file):
            df_gt = pd.read_csv(all_results_file)
            # Use first row's task and answer_prefix as defaults
            default_task = df_gt["task"].iloc[0] if "task" in df_gt.columns else "hotpotqa_32k"
            default_prefix = df_gt["answer_prefix"].iloc[0] if "answer_prefix" in df_gt.columns else "Answer: "
            
            # Create a mapping from sample_idx to ground truth
            gt_map = {}
            for idx, row in df_gt.iterrows():
                sample_idx = row.get("sample_idx", idx)
                answers = row.get("answers", [])
                if isinstance(answers, str):
                    try:
                        answers = ast.literal_eval(answers)
                    except:
                        answers = [answers]
                if not isinstance(answers, list):
                    answers = [answers]
                gt_map[sample_idx] = {
                    "answers": [str(a) for a in answers],
                    "task": row.get("task", default_task),
                    "answer_prefix": row.get("answer_prefix", default_prefix),
                }
        else:
            print("Warning: all_results.csv not found, using defaults")
            default_task = "hotpotqa_32k"
            default_prefix = "Answer: "
            gt_map = {}
        
        # Build DataFrame from responses
        rows = []
        for resp in responses:
            sample_idx = resp.get("sample_idx", len(rows))
            scenario = resp.get("scenario", "unknown")
            response_text = resp.get("response", "").strip()
            
            # Get ground truth
            gt = gt_map.get(sample_idx, {})
            answers = gt.get("answers", [])
            task = gt.get("task", default_task)
            answer_prefix = gt.get("answer_prefix", default_prefix)
            
            # Prepend prefix to response
            if answer_prefix and not answer_prefix.endswith(" "):
                answer_prefix = answer_prefix + " "
            predicted_answer = answer_prefix + response_text
            
            rows.append({
                "sample_idx": sample_idx,
                "method": scenario,
                "task": task,
                "answers": answers,
                "answer_prefix": answer_prefix.strip(),
                "predicted_answer": predicted_answer,
            })
        
        df = pd.DataFrame(rows)
        
        # Group by scenario
        scenarios = df["method"].unique()
        metrics_results = {}
        
        for scenario in scenarios:
            df_scenario = df[df["method"] == scenario].copy()
            
            if df_scenario.empty:
                continue
            
            # Group by task
            task_groups = df_scenario.groupby("task")
            task_metrics = {}
            all_em_scores = []
            all_subspan_em_scores = []
            all_f1_scores = []
            all_coverage_scores = []
            
            for task_name, task_df in task_groups:
                try:
                    metrics = calculate_metrics(task_df)
                    
                    if "error" in metrics:
                        print(f"  ❌ {scenario}/{task_name}: {metrics['error']}")
                        continue
                    
                    task_metrics[task_name] = metrics
                    
                    all_em_scores.append(metrics["em"])
                    all_subspan_em_scores.append(metrics["subspan_em"])
                    
                    if "f1" in metrics:
                        all_f1_scores.append(metrics["f1"])
                    if "coverage" in metrics:
                        all_coverage_scores.append(metrics["coverage"])
                    
                    print(f"  ✓ {scenario}/{task_name}: EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}", end="")
                    if "f1" in metrics:
                        print(f", F1={metrics['f1']:.4f}", end="")
                    if "coverage" in metrics:
                        print(f", Coverage={metrics['coverage']:.4f}", end="")
                    print()
                    
                except Exception as e:
                    print(f"  ❌ {scenario}/{task_name}: Error - {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Compute overall metrics
            overall_metrics = {
                "overall": {
                    "em": float(sum(all_em_scores) / len(all_em_scores)) if all_em_scores else 0.0,
                    "subspan_em": float(sum(all_subspan_em_scores) / len(all_subspan_em_scores)) if all_subspan_em_scores else 0.0,
                },
                "task_metrics": {task: {k: round(v, 4) if isinstance(v, float) else v 
                                     for k, v in metrics.items()} 
                              for task, metrics in task_metrics.items()},
                "summary": {
                    "total_tasks": len(task_metrics),
                    "total_samples": len(df_scenario)
                }
            }
            
            if all_f1_scores:
                overall_metrics["overall"]["f1"] = float(sum(all_f1_scores) / len(all_f1_scores))
            if all_coverage_scores:
                overall_metrics["overall"]["coverage"] = float(sum(all_coverage_scores) / len(all_coverage_scores))
            
            overall_metrics["overall"] = {k: round(v, 4) if isinstance(v, float) else v 
                                          for k, v in overall_metrics["overall"].items()}
            
            metrics_results[scenario] = overall_metrics
    
    else:
        print(f"Error: Neither responses.jsonl nor all_results.csv found in {output_dir}")
        sys.exit(1)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\n✓ Saved metrics to: {metrics_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    for scenario, metrics in metrics_results.items():
        print(f"\n{scenario}:")
        overall = metrics.get("overall", {})
        print(f"  EM: {overall.get('em', 'N/A'):.4f}" if isinstance(overall.get('em'), (int, float)) else f"  EM: {overall.get('em', 'N/A')}")
        print(f"  Subspan EM: {overall.get('subspan_em', 'N/A'):.4f}" if isinstance(overall.get('subspan_em'), (int, float)) else f"  Subspan EM: {overall.get('subspan_em', 'N/A')}")
        if "f1" in overall:
            print(f"  F1: {overall.get('f1', 'N/A'):.4f}" if isinstance(overall.get('f1'), (int, float)) else f"  F1: {overall.get('f1', 'N/A')}")
        if "coverage" in overall:
            print(f"  Coverage: {overall.get('coverage', 'N/A'):.4f}" if isinstance(overall.get('coverage'), (int, float)) else f"  Coverage: {overall.get('coverage', 'N/A')}")
        print(f"  Total samples: {metrics.get('summary', {}).get('total_samples', 'N/A')}")
    print("=" * 80)
    
    print("\n✅ Metrics computation complete!")


if __name__ == "__main__":
    main()

