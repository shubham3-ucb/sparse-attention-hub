#!/usr/bin/env python3
"""Standalone evaluation script for RULER 16k results.

This script reads results from a CSV file and computes metrics separately for
dense and sparse methods, reusing the evaluation code from test_sparse_oracle_ruler16k.py.

Usage:
    python evals_outside.py <output_directory>
    
Example:
    python evals_outside.py output_llama31_8b_ruler16k_naive_pcs1024_hs010
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

# Add current directory to path to import benchmark modules
sys.path.insert(0, os.path.dirname(__file__))

from benchmark.benchmark_registry import create_benchmark_instance


def parse_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse answer column from CSV - handles string representation of lists.
    
    The CSV stores answers as string representations of lists (e.g., "['A', 'B', 'C']").
    This function converts them to actual lists for proper evaluation.
    
    Args:
        df: DataFrame with 'answer' column that may contain string representations of lists
        
    Returns:
        DataFrame with 'answer' column converted to lists
    """
    df = df.copy()
    
    def parse_answer(ans: Any) -> list:
        """Parse answer value to list format."""
        try:
            if isinstance(ans, str):
                ans_stripped = ans.strip()
                if ans_stripped.startswith('['):
                    # String representation of list - parse it
                    return ast.literal_eval(ans_stripped)
                else:
                    # Plain string - wrap in list
                    return [ans_stripped]
            elif isinstance(ans, list):
                # Already a list
                return ans
            else:
                # Other type - convert to string and wrap
                return [str(ans)]
        except Exception:
            # If parsing fails, wrap as single-item list
            return [str(ans)]
    
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(parse_answer)
    
    return df


def evaluate_directory(output_dir: str) -> Dict[str, Any]:
    """Evaluate results in a given output directory.
    
    Args:
        output_dir: Path to directory containing raw_results.csv
        
    Returns:
        Dictionary with evaluation results
    """
    csv_path = os.path.join(output_dir, "raw_results.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"raw_results.csv not found in {output_dir}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Handle NaN/empty predicted_answer values
    df['predicted_answer'] = df['predicted_answer'].fillna('').astype(str)
    
    # Parse answer column (convert string representations of lists to actual lists)
    df = parse_answer_column(df)
    
    # Split dense and sparse
    df_dense = df[df['method'] == 'dense'].copy()
    df_sparse = df[df['method'] == 'sparse'].copy()
    
    # Create benchmark instance
    # Try to infer ruler_tasks from the data, or use default
    ruler_tasks = None
    if 'task' in df.columns:
        unique_tasks = df['task'].unique().tolist()
        # Filter to valid RULER tasks
        all_ruler_tasks = [
            "cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2",
            "niah_single_3", "qa_1", "qa_2", "vt"
        ]
        ruler_tasks = [t for t in unique_tasks if t in all_ruler_tasks]
        if not ruler_tasks:
            ruler_tasks = None  # Use default (all tasks)
    
    ruler16k = create_benchmark_instance("ruler16k", subsets=ruler_tasks)
    
    # Compute metrics for dense and sparse separately
    dense_metrics = None
    sparse_metrics = None
    
    if not df_dense.empty:
        try:
            dense_metrics = ruler16k.post_run_evaluate(df_dense)
        except Exception as e:
            print(f"Error computing dense metrics: {e}", file=sys.stderr)
            dense_metrics = None
    
    if not df_sparse.empty:
        try:
            sparse_metrics = ruler16k.post_run_evaluate(df_sparse)
        except Exception as e:
            print(f"Error computing sparse metrics: {e}", file=sys.stderr)
            sparse_metrics = None
    
    # Prepare results
    results = {
        'directory': output_dir,
        'dense': {
            'metrics': dense_metrics,
            'n_samples': len(df_dense),
        },
        'sparse': {
            'metrics': sparse_metrics,
            'n_samples': len(df_sparse),
        },
        'total_samples': len(df),
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results in a readable format.
    
    Args:
        results: Results dictionary from evaluate_directory
    """
    print("=" * 100)
    print(f"EVALUATION RESULTS: {results['directory']}")
    print("=" * 100)
    print()
    
    dense_metrics = results['dense']['metrics']
    sparse_metrics = results['sparse']['metrics']
    dense_n = results['dense']['n_samples']
    sparse_n = results['sparse']['n_samples']
    
    print(f"{'Method':<20} {'Overall Score':<15} {'Task Scores':<50} {'n':<10}")
    print("-" * 100)
    
    # Dense
    if dense_metrics:
        dense_score = dense_metrics.get('overall_score', 'N/A')
        task_scores = dense_metrics.get('task_scores', {})
        task_scores_str = ', '.join([f"{task}: {scores.get('string_match', 'N/A')}" 
                                     for task, scores in task_scores.items()])
        if len(task_scores_str) > 45:
            task_scores_str = task_scores_str[:42] + "..."
        print(f"{'DENSE':<20} {str(dense_score):<15} {task_scores_str:<50} {dense_n:<10}")
    else:
        print(f"{'DENSE':<20} {'N/A':<15} {'N/A':<50} {dense_n:<10}")
    
    # Sparse
    if sparse_metrics:
        sparse_score = sparse_metrics.get('overall_score', 'N/A')
        task_scores = sparse_metrics.get('task_scores', {})
        task_scores_str = ', '.join([f"{task}: {scores.get('string_match', 'N/A')}" 
                                     for task, scores in task_scores.items()])
        if len(task_scores_str) > 45:
            task_scores_str = task_scores_str[:42] + "..."
        print(f"{'SPARSE':<20} {str(sparse_score):<15} {task_scores_str:<50} {sparse_n:<10}")
    else:
        print(f"{'SPARSE':<20} {'N/A':<15} {'N/A':<50} {sparse_n:<10}")
    
    print()
    
    # Summary
    if dense_metrics and sparse_metrics:
        dense_score = dense_metrics.get('overall_score', 0)
        sparse_score = sparse_metrics.get('overall_score', 0)
        if isinstance(dense_score, (int, float)) and isinstance(sparse_score, (int, float)):
            diff = sparse_score - dense_score
            print(f"Difference (Sparse - Dense): {diff:+.2f}%")
    
    print()
    print("=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RULER 16k results from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evals_outside.py output_llama31_8b_ruler16k_naive_pcs1024_hs010
  python evals_outside.py exp_sweep_filter_ratios_ruler16k/sparsity_0.10_pcs1024_ns40_max2147483647
        """
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing raw_results.csv"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="OUTPUT_FILE",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Resolve directory path
    output_dir = os.path.abspath(args.directory)
    
    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        results = evaluate_directory(output_dir)
        
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_results(results)
        
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.save}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

