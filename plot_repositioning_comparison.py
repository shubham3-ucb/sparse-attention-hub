#!/usr/bin/env python3
"""
Plot comparison between sparse with repositioning vs sparse without repositioning.

Compares:
- output_llama31_8b_128k_longctx_hs010_pcs4096 (with repositioning)
- output_llama31_8b_128k_naive_pcs4096 (sparse only, no repositioning)

Plots L2 relative differences and Jaccard similarity for 5 sampled examples.
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
except ImportError as e:
    print(f"Error: Required packages not found. Please install: pip install matplotlib numpy")
    print(f"Missing: {e}")
    sys.exit(1)


def load_metrics(metrics_file: str) -> List[Dict]:
    """Load all metrics from a micro_metrics.jsonl file."""
    metrics: List[Dict] = []
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metric = json.loads(line)
                    metrics.append(metric)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: {metrics_file} not found")
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
    
    return metrics


def group_metrics_by_sample(metrics: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Group metrics by sample index.
    
    Samples are identified by tracking chunk progression:
    - First chunk of each sample has seq_len_q == seq_len_k (initial chunk)
    - Subsequent chunks have seq_len_k > seq_len_q (extended context)
    """
    samples: Dict[int, List[Dict]] = defaultdict(list)
    current_sample = 0
    prev_seq_len_k = 0
    
    for metric in metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        seq_len_k = metadata.get("seq_len_k", 0)
        
        # Detect new sample: if seq_len_k resets or decreases significantly
        if seq_len_k < prev_seq_len_k * 0.5 or (seq_len_q == seq_len_k and prev_seq_len_k > seq_len_q * 1.5):
            current_sample += 1
        
        samples[current_sample].append(metric)
        prev_seq_len_k = seq_len_k
    
    return samples


def extract_chunk_key(metadata: Dict) -> Tuple[int, int]:
    """Extract (seq_len_q, seq_len_k) as chunk identifier."""
    return (metadata.get("seq_len_q", 0), metadata.get("seq_len_k", 0))


def get_attention_metrics(sample_metrics: List[Dict]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Extract attention weight diff metrics grouped by chunk."""
    chunk_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    
    for metric in sample_metrics:
        if metric.get("metric") != "research_attention_weight_diff":
            continue
        
        metadata = metric.get("metadata", {})
        chunk_key = extract_chunk_key(metadata)
        value = metric.get("value", {})
        
        chunk_metrics[chunk_key] = {
            "l2_diff_relative": value.get("l2_diff_relative", 0.0),
            "max_diff": value.get("max_diff", 0.0),
            "mean_diff": value.get("mean_diff", 0.0),
            "l2_diff": value.get("l2_diff", 0.0),
        }
    
    return chunk_metrics


def get_mask_metrics(sample_metrics: List[Dict]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Extract mask comparison metrics grouped by chunk."""
    chunk_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    
    for metric in sample_metrics:
        if metric.get("metric") != "research_mask_roped_vs_unroped":
            continue
        
        metadata = metric.get("metadata", {})
        chunk_key = extract_chunk_key(metadata)
        value = metric.get("value", {})
        
        chunk_metrics[chunk_key] = {
            "jaccard_similarity": value.get("jaccard_similarity", 0.0),
            "overlap_percentage": value.get("overlap_percentage", 0.0),
            "diff_percentage": value.get("diff_percentage", 0.0),
        }
    
    return chunk_metrics


def plot_sample_comparison(
    sample_idx: int,
    repositioning_data: Dict[Tuple[int, int], Dict[str, float]],
    naive_data: Dict[Tuple[int, int], Dict[str, float]],
    output_dir: str,
    metric_type: str = "attention",
) -> None:
    """
    Plot comparison for a single sample.
    
    Args:
        sample_idx: Sample index
        repositioning_data: Metrics from run with repositioning
        naive_data: Metrics from run without repositioning
        output_dir: Output directory for plots
        metric_type: "attention" or "mask"
    """
    # Get all chunks (union of both runs)
    all_chunks = set(repositioning_data.keys()) | set(naive_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])  # Sort by seq_len_k
    
    if not all_chunks:
        print(f"  Warning: No chunks found for sample {sample_idx}")
        return
    
    # Create figure
    if metric_type == "attention":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Sample {sample_idx}: Attention Weight Differences\n(With Repositioning vs Naive Sparse)", 
                     fontsize=14, fontweight="bold")
        
        metrics = [
            ("l2_diff_relative", "Relative L2 Difference", True),
            ("max_diff", "Max Difference", False),
            ("mean_diff", "Mean Difference", True),
            ("l2_diff", "Absolute L2 Difference", True),
        ]
    else:  # mask
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Sample {sample_idx}: Mask Comparison (Jaccard Similarity)\n(With Repositioning vs Naive Sparse)", 
                     fontsize=14, fontweight="bold")
        
        metrics = [
            ("jaccard_similarity", "Jaccard Similarity", False),
            ("overlap_percentage", "Overlap Percentage", False),
            ("diff_percentage", "Difference Percentage", False),
        ]
        axes = axes.flatten()
    
    chunk_sizes = [k for (q, k) in all_chunks]
    chunk_labels = [f"Chunk {i+1}\n({k//1024}K)" if k >= 1024 else f"Chunk {i+1}\n({k})" 
                    for i, (q, k) in enumerate(all_chunks)]
    
    for idx, (metric_key, metric_title, use_log) in enumerate(metrics):
        ax = axes[idx] if metric_type == "mask" else axes[idx // 2, idx % 2]
        
        # Get values for both runs
        repositioning_values = [
            repositioning_data.get(chunk, {}).get(metric_key, 0.0) for chunk in all_chunks
        ]
        naive_values = [
            naive_data.get(chunk, {}).get(metric_key, 0.0) for chunk in all_chunks
        ]
        
        # Plot
        x = np.arange(len(all_chunks))
        width = 0.35
        
        ax.bar(x - width/2, repositioning_values, width, label="With Repositioning", 
               color="#2E86AB", alpha=0.8)
        ax.bar(x + width/2, naive_values, width, label="Naive Sparse", 
               color="#A23B72", alpha=0.8)
        
        ax.set_xlabel("Chunk (Context Size)", fontsize=11)
        ax.set_ylabel(metric_title, fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chunk_labels, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        
        if use_log:
            ax.set_yscale("log")
        
        if metric_type == "mask" and metric_key == "jaccard_similarity":
            ax.set_ylim([0, 1.05])
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    filename = f"sample_{sample_idx}_{metric_type}_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_comparison(
    all_samples_repositioning: Dict[int, Dict[Tuple[int, int], Dict[str, float]]],
    all_samples_naive: Dict[int, Dict[Tuple[int, int], Dict[str, float]]],
    sampled_indices: List[int],
    output_dir: str,
    metric_type: str = "attention",
) -> None:
    """Plot combined comparison across all sampled examples."""
    if metric_type == "attention":
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Combined Comparison: Attention Weight Differences\n(With Repositioning vs Naive Sparse)", 
                     fontsize=16, fontweight="bold")
        
        metrics = [
            ("l2_diff_relative", "Relative L2 Difference", True),
            ("max_diff", "Max Difference", False),
            ("mean_diff", "Mean Difference", True),
            ("l2_diff", "Absolute L2 Difference", True),
        ]
    else:  # mask
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Combined Comparison: Mask Metrics\n(With Repositioning vs Naive Sparse)", 
                     fontsize=16, fontweight="bold")
        
        metrics = [
            ("jaccard_similarity", "Jaccard Similarity", False),
            ("overlap_percentage", "Overlap Percentage", False),
            ("diff_percentage", "Difference Percentage", False),
        ]
        axes = axes.flatten()
    
    # Collect all chunks across all samples
    all_chunks = set()
    for sample_idx in sampled_indices:
        all_chunks.update(all_samples_repositioning.get(sample_idx, {}).keys())
        all_chunks.update(all_samples_naive.get(sample_idx, {}).keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])
    
    chunk_sizes = [k for (q, k) in all_chunks]
    
    for idx, (metric_key, metric_title, use_log) in enumerate(metrics):
        ax = axes[idx] if metric_type == "mask" else axes[idx // 2, idx % 2]
        
        # Collect values for each sample
        repositioning_all_values = []
        naive_all_values = []
        
        for sample_idx in sampled_indices:
            repositioning_data = all_samples_repositioning.get(sample_idx, {})
            naive_data = all_samples_naive.get(sample_idx, {})
            
            for chunk in all_chunks:
                repositioning_all_values.append(
                    repositioning_data.get(chunk, {}).get(metric_key, 0.0)
                )
                naive_all_values.append(
                    naive_data.get(chunk, {}).get(metric_key, 0.0)
                )
        
        # Scatter plot
        ax.scatter(repositioning_all_values, naive_all_values, alpha=0.6, s=50, 
                  color="#2E86AB", edgecolors="black", linewidths=0.5)
        
        # Add diagonal line (y=x)
        max_val = max(max(repositioning_all_values) if repositioning_all_values else [0],
                     max(naive_all_values) if naive_all_values else [0], 1e-6)
        min_val = min(min(repositioning_all_values) if repositioning_all_values else [1e6],
                     min(naive_all_values) if naive_all_values else [1e6], 0.0)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, 
               label="y=x (equal)")
        
        ax.set_xlabel(f"With Repositioning\n{metric_title}", fontsize=11)
        ax.set_ylabel(f"Naive Sparse\n{metric_title}", fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
    
    plt.tight_layout()
    
    filename = f"combined_{metric_type}_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main function to generate comparison plots."""
    # Paths
    repo_dir = "/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
    repositioning_dir = os.path.join(repo_dir, "output_llama31_8b_128k_longctx_hs010_pcs4096")
    naive_dir = os.path.join(repo_dir, "output_llama31_8b_128k_naive_pcs4096")
    output_dir = os.path.join(repo_dir, "plots_repositioning_comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Loading metrics...")
    print("=" * 80)
    
    # Load metrics
    repositioning_metrics = load_metrics(os.path.join(repositioning_dir, "micro_metrics.jsonl"))
    naive_metrics = load_metrics(os.path.join(naive_dir, "micro_metrics.jsonl"))
    
    print(f"Loaded {len(repositioning_metrics)} metrics from repositioning run")
    print(f"Loaded {len(naive_metrics)} metrics from naive run")
    
    # Group by sample
    print("\nGrouping metrics by sample...")
    repositioning_samples = group_metrics_by_sample(repositioning_metrics)
    naive_samples = group_metrics_by_sample(naive_metrics)
    
    print(f"Found {len(repositioning_samples)} samples in repositioning run")
    print(f"Found {len(naive_samples)} samples in naive run")
    
    # Sample 5 examples
    max_samples = min(len(repositioning_samples), len(naive_samples))
    num_to_sample = min(5, max_samples)
    sampled_indices = sorted(random.sample(range(max_samples), num_to_sample))
    
    print(f"\nSampling {num_to_sample} examples: {sampled_indices}")
    
    # Extract metrics for sampled examples
    print("\nExtracting attention and mask metrics...")
    all_samples_repositioning_attn: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    all_samples_naive_attn: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    all_samples_repositioning_mask: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    all_samples_naive_mask: Dict[int, Dict[Tuple[int, int], Dict[str, float]]] = {}
    
    for sample_idx in sampled_indices:
        all_samples_repositioning_attn[sample_idx] = get_attention_metrics(
            repositioning_samples.get(sample_idx, [])
        )
        all_samples_naive_attn[sample_idx] = get_attention_metrics(
            naive_samples.get(sample_idx, [])
        )
        all_samples_repositioning_mask[sample_idx] = get_mask_metrics(
            repositioning_samples.get(sample_idx, [])
        )
        all_samples_naive_mask[sample_idx] = get_mask_metrics(
            naive_samples.get(sample_idx, [])
        )
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    # Individual sample plots
    for sample_idx in sampled_indices:
        print(f"\nPlotting sample {sample_idx}...")
        plot_sample_comparison(
            sample_idx,
            all_samples_repositioning_attn[sample_idx],
            all_samples_naive_attn[sample_idx],
            output_dir,
            metric_type="attention",
        )
        plot_sample_comparison(
            sample_idx,
            all_samples_repositioning_mask[sample_idx],
            all_samples_naive_mask[sample_idx],
            output_dir,
            metric_type="mask",
        )
    
    # Combined comparison plots
    print("\nGenerating combined comparison plots...")
    plot_combined_comparison(
        all_samples_repositioning_attn,
        all_samples_naive_attn,
        sampled_indices,
        output_dir,
        metric_type="attention",
    )
    plot_combined_comparison(
        all_samples_repositioning_mask,
        all_samples_naive_mask,
        sampled_indices,
        output_dir,
        metric_type="mask",
    )
    
    print("\n" + "=" * 80)
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()




