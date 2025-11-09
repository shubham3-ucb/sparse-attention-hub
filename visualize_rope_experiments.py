#!/usr/bin/env python3
"""
Visualize attention weight differences (L2, max_diff, mean_diff) across RoPE repositioning experiments.

This script scans output directories for micro_metrics.jsonl files and creates clean visualizations
comparing attention weight differences across different experiments.
"""

import json
import os
import glob
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    # Use non-interactive backend
    matplotlib.use("Agg")
except ImportError as e:
    print(f"Error: Required packages not found. Please install: pip install matplotlib numpy")
    print(f"Missing: {e}")
    exit(1)


def find_experiment_dirs(base_dir: str = ".") -> List[Tuple[str, str]]:
    """
    Find all output directories containing micro_metrics.jsonl files.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of (experiment_name, full_path) tuples
    """
    experiments: List[Tuple[str, str]] = []
    
    # Look for output_* directories
    pattern = os.path.join(base_dir, "output_*", "micro_metrics.jsonl")
    for metrics_file in glob.glob(pattern):
        exp_dir = os.path.dirname(metrics_file)
        exp_name = os.path.basename(exp_dir)
        experiments.append((exp_name, exp_dir))
    
    return sorted(experiments)


def load_metrics(metrics_file: str) -> List[Dict]:
    """
    Load metrics from a micro_metrics.jsonl file.
    
    Args:
        metrics_file: Path to micro_metrics.jsonl
        
    Returns:
        List of metric dictionaries
    """
    metrics: List[Dict] = []
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metric = json.loads(line)
                    if metric.get("metric") == "research_attention_weight_diff":
                        metrics.append(metric)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: {metrics_file} not found")
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
    
    return metrics


def extract_chunk_info(metadata: Dict) -> Tuple[int, int]:
    """
    Extract chunk information from metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        (seq_len_q, seq_len_k) tuple representing chunk size
    """
    seq_len_q = metadata.get("seq_len_q", 0)
    seq_len_k = metadata.get("seq_len_k", 0)
    return (seq_len_q, seq_len_k)


def is_chunk(seq_len_q: int, chunk_threshold: int = 100) -> bool:
    """
    Determine if this is a chunk (prefill) or generation step.
    
    Args:
        seq_len_q: Query sequence length
        chunk_threshold: Threshold above which we consider it a chunk
        
    Returns:
        True if chunk, False if generation
    """
    return seq_len_q >= chunk_threshold


def group_metrics_by_sample(metrics: List[Dict], chunk_threshold: int = 100) -> List[List[Dict]]:
    """
    Group metrics by sample/example.
    
    Detects sample boundaries by:
    1. Large timestamp gaps (> 1 second)
    2. Reset to first chunk (seq_len_q == 4096 or large value after generation)
    
    Args:
        metrics: List of all metrics
        chunk_threshold: Threshold for distinguishing chunks from generation
        
    Returns:
        List of sample groups, each containing metrics for one sample
    """
    if not metrics:
        return []
    
    samples: List[List[Dict]] = []
    current_sample: List[Dict] = []
    
    prev_timestamp = None
    prev_was_generation = False
    
    for metric in metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        timestamp_str = metric.get("timestamp", "")
        
        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = None
        except:
            timestamp = None
        
        # Detect new sample:
        # 1. Large time gap (> 1 second)
        # 2. Reset to chunk after generation (seq_len_q >= chunk_threshold and prev was generation)
        is_new_sample = False
        
        if timestamp and prev_timestamp:
            time_diff = (timestamp - prev_timestamp).total_seconds()
            if time_diff > 1.0:  # Large gap indicates new sample
                is_new_sample = True
        
        is_current_chunk = is_chunk(seq_len_q, chunk_threshold)
        if prev_was_generation and is_current_chunk:
            # Reset to chunk after generation = new sample
            is_new_sample = True
        
        if is_new_sample and current_sample:
            samples.append(current_sample)
            current_sample = []
        
        current_sample.append(metric)
        prev_timestamp = timestamp
        prev_was_generation = not is_current_chunk
    
    # Add last sample
    if current_sample:
        samples.append(current_sample)
    
    return samples


def organize_metrics_by_chunk(metrics: List[Dict]) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Organize metrics by chunk (seq_len_q, seq_len_k).
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        Dictionary mapping (seq_len_q, seq_len_k) to list of metrics
    """
    organized: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    
    for metric in metrics:
        chunk_key = extract_chunk_info(metric.get("metadata", {}))
        organized[chunk_key].append(metric)
    
    return dict(organized)


def separate_chunks_and_generation(metrics: List[Dict], chunk_threshold: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate metrics into chunks (prefill) and generation steps.
    
    Args:
        metrics: List of metric dictionaries
        chunk_threshold: Threshold for distinguishing chunks from generation
        
    Returns:
        (chunks, generation) tuple
    """
    chunks: List[Dict] = []
    generation: List[Dict] = []
    
    for metric in metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        
        if is_chunk(seq_len_q, chunk_threshold):
            chunks.append(metric)
        else:
            generation.append(metric)
    
    return chunks, generation


def compute_chunk_statistics(metrics: List[Dict]) -> Dict[str, float]:
    """
    Compute statistics for a chunk (average across all metrics in chunk).
    
    Args:
        metrics: List of metric dictionaries for a chunk
        
    Returns:
        Dictionary with l2_diff, max_diff, mean_diff statistics
    """
    if not metrics:
        return {"l2_diff": 0.0, "max_diff": 0.0, "mean_diff": 0.0}
    
    l2_diffs = [m["value"]["l2_diff"] for m in metrics]
    max_diffs = [m["value"]["max_diff"] for m in metrics]
    mean_diffs = [m["value"]["mean_diff"] for m in metrics]
    
    return {
        "l2_diff": np.mean(l2_diffs),
        "max_diff": np.mean(max_diffs),
        "mean_diff": np.mean(mean_diffs),
    }


def clean_experiment_name(name: str) -> str:
    """
    Clean experiment name for display.
    
    Args:
        name: Raw experiment name
        
    Returns:
        Cleaned name
    """
    # Remove common prefixes
    name = name.replace("output_test_sparse_chunk_4096_", "")
    name = name.replace("output_test_sparse_chunk_4096", "baseline")
    name = name.replace("_", " ").title()
    return name


def plot_l2_comparison(
    experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]],
    output_dir: str,
) -> None:
    """
    Plot L2 differences across experiments and chunks.
    
    Args:
        experiments_data: Dictionary mapping experiment names to chunk statistics
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all unique chunks (sorted by seq_len_k)
    all_chunks = set()
    for exp_data in experiments_data.values():
        all_chunks.update(exp_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])  # Sort by seq_len_k
    
    # Create chunk labels
    chunk_labels = [f"Chunk {i+1}\n({k//1024}K keys)" for i, (q, k) in enumerate(all_chunks)]
    
    # Plot each experiment
    x = np.arange(len(all_chunks))
    width = 0.8 / len(experiments_data)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for idx, (exp_name, exp_data) in enumerate(experiments_data.items()):
        l2_values = [exp_data.get(chunk, {}).get("l2_diff", 0.0) for chunk in all_chunks]
        offset = (idx - len(experiments_data) / 2 + 0.5) * width
        ax.bar(x + offset, l2_values, width, label=clean_experiment_name(exp_name), color=colors[idx], alpha=0.8)
    
    ax.set_xlabel("Chunk", fontsize=12)
    ax.set_ylabel("L2 Difference", fontsize=12)
    ax.set_title("Attention Weight L2 Differences Across Experiments", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_labels, rotation=0, ha="center")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "l2_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_overview(
    experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]],
    output_dir: str,
) -> None:
    """
    Plot overview of all metrics (L2, max_diff, mean_diff) for first few chunks.
    
    Args:
        experiments_data: Dictionary mapping experiment names to chunk statistics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get all unique chunks (sorted by seq_len_k)
    all_chunks = set()
    for exp_data in experiments_data.values():
        all_chunks.update(exp_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])[:5]  # First 5 chunks
    
    chunk_labels = [f"Chunk {i+1}" for i in range(len(all_chunks))]
    x = np.arange(len(all_chunks))
    width = 0.8 / len(experiments_data)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    metrics_to_plot = ["l2_diff", "max_diff", "mean_diff"]
    metric_titles = ["L2 Difference", "Max Difference", "Mean Difference"]
    
    for metric_idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[metric_idx]
        
        for exp_idx, (exp_name, exp_data) in enumerate(experiments_data.items()):
            values = [exp_data.get(chunk, {}).get(metric, 0.0) for chunk in all_chunks]
            offset = (exp_idx - len(experiments_data) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=clean_experiment_name(exp_name), color=colors[exp_idx], alpha=0.8)
        
        ax.set_xlabel("Chunk", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chunk_labels)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "l2_diff":
            ax.set_yscale("log")
        if metric == "mean_diff":
            ax.set_yscale("log")
    
    # Add legend only to first subplot
    axes[0].legend(loc="upper left", fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "metrics_overview.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sample_progression(
    sample_metrics: List[Dict],
    exp_name: str,
    sample_idx: int,
    output_dir: str,
    chunk_threshold: int = 100,
) -> None:
    """
    Plot L2 progression for a single sample, clearly separating chunks and generation.
    
    Args:
        sample_metrics: List of metrics for one sample
        exp_name: Experiment name
        sample_idx: Sample index (0-based)
        output_dir: Directory to save plots
        chunk_threshold: Threshold for distinguishing chunks from generation
    """
    if not sample_metrics:
        return
    
    # Separate chunks and generation
    chunks, generation = separate_chunks_and_generation(sample_metrics, chunk_threshold)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot chunks
    chunk_l2 = [m["value"]["l2_diff"] for m in chunks]
    chunk_seq_k = [m["metadata"]["seq_len_k"] for m in chunks]
    chunk_indices = list(range(len(chunks)))
    
    if chunks:
        ax.plot(chunk_indices, chunk_l2, marker="o", label="Chunks (Prefill)", 
                color="steelblue", linewidth=2.5, markersize=8, alpha=0.8, zorder=3)
    
    # Plot generation
    gen_l2 = [m["value"]["l2_diff"] for m in generation]
    gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
    
    if generation:
        ax.plot(gen_indices, gen_l2, marker="s", label="Generation", 
                color="coral", linewidth=2, markersize=6, alpha=0.8, zorder=3)
    
    # Formatting
    ax.set_xlabel("Step Index", fontsize=12)
    ax.set_ylabel("L2 Difference", fontsize=12)
    clean_exp_name = clean_experiment_name(exp_name)
    ax.set_title(f"Sample {sample_idx + 1} - {clean_exp_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    
    # Add vertical line to separate chunks from generation (after setting scale)
    if chunks and generation:
        boundary_x = len(chunks) - 0.5
        ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=1.5, 
                  alpha=0.6, zorder=1)
        # Add text annotation (after ylim is set)
        ylim = ax.get_ylim()
        ax.text(boundary_x, ylim[1] * 0.95, "Chunk/Gen\nBoundary", 
               ha="center", va="top", fontsize=8, color="gray", alpha=0.8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"))
    
    # Add text annotation for chunk/gen counts
    if chunks:
        ax.text(0.02, 0.98, f"Chunks: {len(chunks)}", transform=ax.transAxes,
               fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", 
               facecolor="lightblue", alpha=0.5))
    if generation:
        ax.text(0.02, 0.90, f"Generation: {len(generation)}", transform=ax.transAxes,
               fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", 
               facecolor="lightcoral", alpha=0.5))
    
    plt.tight_layout()
    
    # Save with sample-specific filename
    safe_exp_name = exp_name.replace("output_", "").replace("/", "_")
    output_path = os.path.join(output_dir, f"sample_{sample_idx + 1:02d}_{safe_exp_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_chunk_progression(
    experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]],
    output_dir: str,
) -> None:
    """
    Plot how L2 changes as context grows (chunk progression).
    
    Args:
        experiments_data: Dictionary mapping experiment names to chunk statistics
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all unique chunks (sorted by seq_len_k)
    all_chunks = set()
    for exp_data in experiments_data.values():
        all_chunks.update(exp_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])  # Sort by seq_len_k
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for idx, (exp_name, exp_data) in enumerate(experiments_data.items()):
        chunk_sizes = [k for (q, k) in all_chunks]
        l2_values = [exp_data.get(chunk, {}).get("l2_diff", 0.0) for chunk in all_chunks]
        
        ax.plot(chunk_sizes, l2_values, marker="o", label=clean_experiment_name(exp_name), 
                color=colors[idx], linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel("Context Size (keys)", fontsize=12)
    ax.set_ylabel("L2 Difference", fontsize=12)
    ax.set_title("Attention Weight L2 Difference vs Context Size", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "chunk_progression.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]]) -> None:
    """
    Print a summary table of metrics.
    
    Args:
        experiments_data: Dictionary mapping experiment names to chunk statistics
    """
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Average L2 Differences by Experiment")
    print("=" * 80)
    
    # Get all chunks
    all_chunks = set()
    for exp_data in experiments_data.values():
        all_chunks.update(exp_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])
    
    # Print header
    print(f"{'Experiment':<40} {'Chunk 1':<12} {'Chunk 2':<12} {'Chunk 3':<12} {'Avg':<12}")
    print("-" * 80)
    
    # Print data for each experiment
    for exp_name, exp_data in experiments_data.items():
        clean_name = clean_experiment_name(exp_name)
        if len(clean_name) > 38:
            clean_name = clean_name[:35] + "..."
        
        # Get L2 values for first 3 chunks
        l2_values = []
        for chunk in all_chunks[:3]:
            l2 = exp_data.get(chunk, {}).get("l2_diff", 0.0)
            l2_values.append(l2)
        
        # Calculate average across all chunks
        all_l2 = [exp_data.get(chunk, {}).get("l2_diff", 0.0) for chunk in all_chunks]
        avg_l2 = np.mean(all_l2) if all_l2 else 0.0
        
        # Format values
        chunk_strs = [f"{v:.6f}" for v in l2_values]
        while len(chunk_strs) < 3:
            chunk_strs.append("N/A")
        
        print(f"{clean_name:<40} {chunk_strs[0]:<12} {chunk_strs[1]:<12} {chunk_strs[2]:<12} {avg_l2:.6f}")
    
    print("=" * 80 + "\n")


def main():
    """Main function to generate all visualizations."""
    base_dir = "."
    output_dir = "figures_rope_exps"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all experiments
    experiments = find_experiment_dirs(base_dir)
    print(f"Found {len(experiments)} experiments:")
    for exp_name, exp_path in experiments:
        print(f"  - {exp_name}")
    
    if not experiments:
        print("No experiments found. Looking for output_*/micro_metrics.jsonl files.")
        return
    
    # Load and organize data
    experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    experiments_samples: Dict[str, List[List[Dict]]] = {}  # Per-experiment, per-sample metrics
    
    for exp_name, exp_path in experiments:
        metrics_file = os.path.join(exp_path, "micro_metrics.jsonl")
        metrics = load_metrics(metrics_file)
        
        if not metrics:
            print(f"Warning: No metrics found for {exp_name}")
            continue
        
        # Group by sample
        samples = group_metrics_by_sample(metrics)
        experiments_samples[exp_name] = samples
        print(f"Loaded {len(metrics)} metrics from {exp_name} ({len(samples)} samples)")
        
        # Also organize by chunk for summary plots
        organized = organize_metrics_by_chunk(metrics)
        chunk_stats = {chunk: compute_chunk_statistics(chunk_metrics) 
                      for chunk, chunk_metrics in organized.items()}
        
        experiments_data[exp_name] = chunk_stats
    
    if not experiments_data:
        print("No data to plot.")
        return
    
    # Print summary table
    print_summary_table(experiments_data)
    
    # Generate per-sample plots
    print("\nGenerating per-sample plots...")
    for exp_name, samples in experiments_samples.items():
        for sample_idx, sample_metrics in enumerate(samples):
            plot_sample_progression(sample_metrics, exp_name, sample_idx, output_dir)
    
    # Generate summary plots
    print("\nGenerating summary plots...")
    plot_l2_comparison(experiments_data, output_dir)
    plot_metrics_overview(experiments_data, output_dir)
    plot_chunk_progression(experiments_data, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

