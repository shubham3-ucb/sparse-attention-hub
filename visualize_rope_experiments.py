#!/usr/bin/env python3
"""
Visualize attention weight differences (L2, max_diff, mean_diff) across RoPE repositioning experiments.

This script scans output directories for micro_metrics.jsonl files and creates clean visualizations
comparing attention weight differences across different experiments.
"""

import json
import os
import glob
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
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


def filter_duplicate_experiments(experiments: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Filter duplicate experiments, keeping only the latest one for each base type.
    For example, if there are multiple "sanity" experiments, keep only the latest.
    
    Args:
        experiments: List of (experiment_name, full_path) tuples
        
    Returns:
        Filtered list with duplicates removed (keeping latest)
    """
    # Group by base experiment type
    experiment_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    for exp_name, exp_path in experiments:
        # Extract base type (e.g., "sanity", "unrope_mask_no_reposition", etc.)
        base_type = exp_name
        # Remove common prefixes
        if "output_test_sparse_chunk_4096_" in base_type:
            base_type = base_type.replace("output_test_sparse_chunk_4096_", "")
        elif base_type == "output_test_sparse_chunk_4096":
            base_type = "sparse"
        
        # Group experiments by base type
        # 1. Sparse base (baseline) - group sparse_base and sanity together
        if "sparse_base" in base_type.lower() or "sanity" in base_type.lower() or base_type == "sparse":
            base_type = "sparse_base"  # Group sparse_base/sanity with sparse (same baseline)
        # 2. Unrope mask experiments - distinguish by repositioning
        elif "unrope_mask" in base_type.lower() and "no_unrope" not in base_type.lower():
            if "reposition_keys_all" in base_type.lower() or "extend_reposition_keys_all" in base_type.lower():
                base_type = "unrope_mask_reposition_all"
            elif "reposition_keys_only_prefix" in base_type.lower() or "extend_reposition_keys_only_prefix" in base_type.lower():
                base_type = "unrope_mask_reposition_prefix"
            elif "no_reposition" in base_type.lower():
                base_type = "unrope_mask_no_reposition"
            else:
                base_type = "unrope_mask"
        # 3. No unrope mask experiments (roped mask with repositioning)
        elif "no_unrope_mask" in base_type.lower():
            if "reposition_keys_all" in base_type.lower():
                base_type = "no_unrope_mask_reposition_all"
            elif "reposition_keys_only_prefix" in base_type.lower():
                base_type = "no_unrope_mask_reposition_prefix"
            elif "no_reposition" in base_type.lower():
                base_type = "no_unrope_mask_no_reposition"
            else:
                base_type = "no_unrope_mask"
        # 4. Legacy reposition experiments (without unrope/no_unrope prefix)
        elif "reposition" in base_type.lower() or "extend_reposition" in base_type.lower():
            if "only_prefix" in base_type.lower():
                base_type = "reposition_prefix"
            elif "all" in base_type.lower() or "keys_all" in base_type.lower():
                base_type = "reposition_all"
            else:
                base_type = "reposition"
        
        experiment_groups[base_type].append((exp_name, exp_path))
    
    # For each group, keep only the latest (by modification time)
    filtered: List[Tuple[str, str]] = []
    for base_type, group_experiments in experiment_groups.items():
        if len(group_experiments) > 1:
            # Sort by modification time of the metrics file (newest first)
            group_experiments.sort(
                key=lambda x: os.path.getmtime(os.path.join(x[1], "micro_metrics.jsonl")),
                reverse=True
            )
            # Keep only the latest
            filtered.append(group_experiments[0])
            print(f"Filtered duplicates for '{base_type}': kept {group_experiments[0][0]}, removed {len(group_experiments) - 1} others")
        else:
            filtered.extend(group_experiments)
    
    # Sort by original experiment name for consistency
    return sorted(filtered, key=lambda x: x[0])


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


def load_mask_metrics(metrics_file: str) -> List[Dict]:
    """
    Load mask comparison metrics from a micro_metrics.jsonl file.
    
    Args:
        metrics_file: Path to micro_metrics.jsonl
        
    Returns:
        List of mask metric dictionaries
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
                    if metric.get("metric") == "research_mask_roped_vs_unroped":
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


def group_mask_metrics_by_sample(metrics: List[Dict], chunk_threshold: int = 100, filter_layer: int = 15) -> List[List[Dict]]:
    """
    Group mask metrics by sample/example.
    Filters to only specified layer for consistency with other plots.
    
    Detects sample boundaries by:
    - Reset to first chunk (seq_len_q == seq_len_k and seq_len_q >= chunk_threshold) after generation (seq_len_q == 1)
    - Handles variable chunk sizes (e.g., 4096, 2258, etc.)
    
    Args:
        metrics: List of all mask metrics
        chunk_threshold: Threshold for distinguishing chunks from generation
        filter_layer: Only include metrics from this layer (default: 15, same as attention weight diff)
        
    Returns:
        List of sample groups, each containing mask metrics for one sample
    """
    if not metrics:
        return []
    
    # Filter to only specified layer first
    filtered_metrics = [
        m for m in metrics 
        if m.get("metadata", {}).get("layer_idx", -1) == filter_layer
    ]
    
    samples: List[List[Dict]] = []
    current_sample: List[Dict] = []
    
    prev_seq_len_q = None
    
    for metric in filtered_metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        seq_len_k = metadata.get("seq_len_k", 0)
        
        # Detect new sample: reset to first chunk (seq_len_q == seq_len_k and >= chunk_threshold) after generation (seq_len_q == 1)
        is_new_sample = False
        
        if prev_seq_len_q == 1 and seq_len_q == seq_len_k and seq_len_q >= chunk_threshold:
            # Reset to first chunk after generation = new sample
            # First chunk: seq_len_q == seq_len_k (no prefix yet)
            is_new_sample = True
        
        if is_new_sample and current_sample:
            samples.append(current_sample)
            current_sample = []
        
        current_sample.append(metric)
        prev_seq_len_q = seq_len_q
    
    # Add last sample
    if current_sample:
        samples.append(current_sample)
    
    return samples


def group_metrics_by_sample(metrics: List[Dict], chunk_threshold: int = 100) -> List[List[Dict]]:
    """
    Group metrics by sample/example.
    
    Detects sample boundaries by:
    - Reset to first chunk (seq_len_q == seq_len_k and seq_len_q >= chunk_threshold) after generation (seq_len_q == 1)
    - Handles variable chunk sizes (e.g., 4096, 2258, etc.)
    
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
    
    prev_seq_len_q = None
    
    for metric in metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        seq_len_k = metadata.get("seq_len_k", 0)
        
        # Detect new sample: reset to first chunk (seq_len_q == seq_len_k and >= chunk_threshold) after generation (seq_len_q == 1)
        is_new_sample = False
        
        if prev_seq_len_q == 1 and seq_len_q == seq_len_k and seq_len_q >= chunk_threshold:
            # Reset to first chunk after generation = new sample
            # First chunk: seq_len_q == seq_len_k (no prefix yet)
            is_new_sample = True
        
        if is_new_sample and current_sample:
            samples.append(current_sample)
            current_sample = []
        
        current_sample.append(metric)
        prev_seq_len_q = seq_len_q
    
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
        Dictionary with l2_diff, l2_diff_relative, max_diff, mean_diff statistics
    """
    if not metrics:
        return {"l2_diff": 0.0, "l2_diff_relative": 0.0, "max_diff": 0.0, "mean_diff": 0.0}
    
    l2_diffs = [m["value"]["l2_diff"] for m in metrics]
    l2_diff_relatives = [m["value"].get("l2_diff_relative", 0.0) for m in metrics]
    max_diffs = [m["value"]["max_diff"] for m in metrics]
    mean_diffs = [m["value"]["mean_diff"] for m in metrics]
    
    return {
        "l2_diff": np.mean(l2_diffs),
        "l2_diff_relative": np.mean(l2_diff_relatives),
        "max_diff": np.mean(max_diffs),
        "mean_diff": np.mean(mean_diffs),
    }


def organize_mask_metrics_by_chunk(metrics: List[Dict], filter_layer: int = 15) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Organize mask metrics by chunk (seq_len_q, seq_len_k).
    Filters to only specified layer for consistency with other plots.
    
    Args:
        metrics: List of mask metric dictionaries
        filter_layer: Only include metrics from this layer (default: 15, same as attention weight diff)
        
    Returns:
        Dictionary mapping (seq_len_q, seq_len_k) to list of metrics for that chunk
    """
    organized: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    
    for metric in metrics:
        # Filter to only specified layer for consistency
        layer_idx = metric.get("metadata", {}).get("layer_idx", -1)
        if layer_idx == filter_layer:
            chunk_key = extract_chunk_info(metric.get("metadata", {}))
            organized[chunk_key].append(metric)
    
    return dict(organized)


def compute_mask_chunk_statistics(metrics: List[Dict]) -> Dict[str, float]:
    """
    Compute statistics for a mask chunk (average across all metrics in chunk).
    
    Args:
        metrics: List of mask metric dictionaries for a chunk
        
    Returns:
        Dictionary with jaccard_similarity, overlap_percentage, diff_percentage statistics
    """
    if not metrics:
        return {"jaccard_similarity": 0.0, "overlap_percentage": 0.0, "diff_percentage": 0.0}
    
    jaccard_values = [m["value"].get("jaccard_similarity", 0.0) for m in metrics]
    overlap_values = [m["value"].get("overlap_percentage", 0.0) for m in metrics]
    diff_values = [m["value"].get("diff_percentage", 0.0) for m in metrics]
    
    return {
        "jaccard_similarity": np.mean(jaccard_values),
        "overlap_percentage": np.mean(overlap_values),
        "diff_percentage": np.mean(diff_values),
    }


def clean_experiment_name(name: str) -> str:
    """
    Clean experiment name for display.
    Uses the part after 'output_test_sparse_chunk_4096_' as the name.
    
    Args:
        name: Raw experiment name
        
    Returns:
        Cleaned name (part after prefix, with underscores replaced by spaces)
    """
    # Remove common prefixes
    if "output_test_sparse_chunk_4096_" in name:
        name = name.replace("output_test_sparse_chunk_4096_", "")
    elif name == "output_test_sparse_chunk_4096":
        name = "sparse_base"
    
    # Just replace underscores with spaces and title case
    name = name.replace("_", " ").title()
    return name


def plot_l2_comparison(
    experiments_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]],
    output_dir: str,
) -> None:
    """
    Plot relative L2 differences across experiments and chunks.
    
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
        l2_values = [exp_data.get(chunk, {}).get("l2_diff_relative", 0.0) for chunk in all_chunks]
        offset = (idx - len(experiments_data) / 2 + 0.5) * width
        ax.bar(x + offset, l2_values, width, label=clean_experiment_name(exp_name), color=colors[idx], alpha=0.8)
    
    ax.set_xlabel("Chunk", fontsize=12)
    ax.set_ylabel("Relative L2 Difference", fontsize=12)
    ax.set_title("Attention Weight Relative L2 Differences Across Experiments\n(Layer 15, Head 10)", fontsize=14, fontweight="bold")
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
    
    metrics_to_plot = ["l2_diff_relative", "max_diff", "mean_diff"]
    metric_titles = ["Relative L2 Difference", "Max Difference", "Mean Difference"]
    
    for metric_idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[metric_idx]
        
        for exp_idx, (exp_name, exp_data) in enumerate(experiments_data.items()):
            # For l2_diff_relative, fallback to l2_diff if not available
            if metric == "l2_diff_relative":
                values = [exp_data.get(chunk, {}).get("l2_diff_relative", exp_data.get(chunk, {}).get("l2_diff", 0.0)) for chunk in all_chunks]
            else:
                values = [exp_data.get(chunk, {}).get(metric, 0.0) for chunk in all_chunks]
            offset = (exp_idx - len(experiments_data) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=clean_experiment_name(exp_name), color=colors[exp_idx], alpha=0.8)
        
        ax.set_xlabel("Chunk", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chunk_labels)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "l2_diff_relative" or metric == "l2_diff":
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


def calculate_context_length(sample_metrics: List[Dict], chunk_threshold: int = 100) -> int:
    """
    Calculate total context length (max seq_len_k from chunks, since seq_len_k already includes prefix).
    
    Args:
        sample_metrics: List of metrics for one sample
        chunk_threshold: Threshold for distinguishing chunks from generation
        
    Returns:
        Total context length (max seq_len_k from all chunks)
    """
    chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
    
    if not chunks:
        return 0
    
    # seq_len_k already includes all previous keys (prefix + current chunk)
    # So we take the maximum, not the sum
    max_context = max(m["metadata"].get("seq_len_k", 0) for m in chunks)
    
    return max_context


def validate_sample_metrics(sample_metrics: List[Dict], chunk_threshold: int = 100) -> Dict[str, Any]:
    """
    Validate and check sample metrics for correctness.
    
    Args:
        sample_metrics: List of metrics for one sample
        chunk_threshold: Threshold for distinguishing chunks from generation
        
    Returns:
        Dictionary with validation results
    """
    if not sample_metrics:
        return {"valid": False, "errors": ["No metrics"]}
    
    chunks, generation = separate_chunks_and_generation(sample_metrics, chunk_threshold)
    
    errors = []
    warnings = []
    
    # Check that chunks come before generation
    if chunks and generation:
        chunk_indices = [i for i, m in enumerate(sample_metrics) if m in chunks]
        gen_indices = [i for i, m in enumerate(sample_metrics) if m in generation]
        
        if gen_indices and chunk_indices and min(gen_indices) < max(chunk_indices):
            errors.append("Generation steps found before chunks end")
    
    # Check that seq_len_k increases monotonically in chunks
    if len(chunks) > 1:
        seq_k_values = [m["metadata"].get("seq_len_k", 0) for m in chunks]
        if seq_k_values != sorted(seq_k_values):
            warnings.append("seq_len_k not monotonically increasing in chunks")
    
    # Check that first chunk has seq_len_q == seq_len_k (no prefix)
    if chunks:
        first_chunk = chunks[0]
        seq_q = first_chunk["metadata"].get("seq_len_q", 0)
        seq_k = first_chunk["metadata"].get("seq_len_k", 0)
        if seq_q != seq_k:
            warnings.append(f"First chunk has seq_len_q ({seq_q}) != seq_len_k ({seq_k})")
    
    # Check that generation has seq_len_q == 1 (mostly)
    if generation:
        gen_seq_q = [m["metadata"].get("seq_len_q", 0) for m in generation]
        if not all(q == 1 for q in gen_seq_q):
            warnings.append(f"Some generation steps have seq_len_q != 1: {set(gen_seq_q)}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_chunks": len(chunks),
        "num_generation": len(generation),
    }


def plot_all_samples_comparison(
    experiments_samples: Dict[str, List[List[Dict]]],
    output_dir: str,
    chunk_threshold: int = 100,
) -> None:
    """
    Plot all samples in a grid, comparing L2 across all experiments for each sample.
    
    Args:
        experiments_samples: Dictionary mapping experiment names to list of samples
        output_dir: Directory to save plots
        chunk_threshold: Threshold for distinguishing chunks from generation
    """
    if not experiments_samples:
        return
    
    # Find maximum number of samples across all experiments
    max_samples = max(len(samples) for samples in experiments_samples.values())
    
    if max_samples == 0:
        return
    
    # Create figure with subplots (one per sample + one empty for legend/notes)
    n_cols = min(3, max_samples + 1)  # +1 for legend/notes subplot
    n_rows = ((max_samples + 1) + n_cols - 1) // n_cols  # +1 for legend/notes subplot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    # Flatten axes array for easier indexing
    if (max_samples + 1) == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Colors and styles for different experiments
    exp_names = sorted(experiments_samples.keys())
    # Use a colormap with more distinct colors
    if len(exp_names) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    exp_colors = dict(zip(exp_names, colors))
    
    # Different line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    # Plot each sample
    for sample_idx in range(max_samples):
        ax = axes[sample_idx]
        
        # Calculate context length for this sample (use first experiment that has this sample)
        context_length = 0
        for exp_name in exp_names:
            samples = experiments_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                if sample_metrics:
                    context_length = calculate_context_length(sample_metrics, chunk_threshold)
                    break
        
        # Plot all experiments for this sample
        for exp_idx, exp_name in enumerate(exp_names):
            samples = experiments_samples[exp_name]
            if sample_idx >= len(samples):
                continue
            
            sample_metrics = samples[sample_idx]
            if not sample_metrics:
                continue
            
            # Separate chunks and generation
            chunks, generation = separate_chunks_and_generation(sample_metrics, chunk_threshold)
            
            # Get style for this experiment
            linestyle = line_styles[exp_idx % len(line_styles)]
            marker = markers[exp_idx % len(markers)]
            
            # Plot chunks - use relative L2 difference
            chunk_l2 = [m["value"].get("l2_diff_relative", m["value"].get("l2_diff", 0.0)) for m in chunks]
            chunk_indices = list(range(len(chunks)))
            
            if chunks:
                clean_exp_name = clean_experiment_name(exp_name)
                ax.plot(chunk_indices, chunk_l2, marker=marker, label=clean_exp_name, 
                       color=exp_colors[exp_name], linewidth=2.5, markersize=7, 
                       alpha=0.85, zorder=3, linestyle=linestyle, markevery=1)
            
            # Plot generation - use relative L2 difference
            gen_l2 = [m["value"].get("l2_diff_relative", m["value"].get("l2_diff", 0.0)) for m in generation]
            gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
            
            if generation:
                clean_exp_name = clean_experiment_name(exp_name)
                ax.plot(gen_indices, gen_l2, marker=marker, 
                       color=exp_colors[exp_name], linewidth=2, markersize=5, 
                       alpha=0.75, zorder=3, linestyle=linestyle, markevery=1)
        
        # Add vertical line to separate chunks from generation
        # Find max chunks across all experiments for this sample
        max_chunks = 0
        for exp_name in exp_names:
            samples = experiments_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
                max_chunks = max(max_chunks, len(chunks))
        
        if max_chunks > 0:
            boundary_x = max_chunks - 0.5
            ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=1, 
                      alpha=0.5, zorder=1)
        
        # Formatting
        ax.set_xlabel("Step Index (chunks: solid lines, generation: after dotted line)", fontsize=9)
        ax.set_ylabel("Relative L2 Difference", fontsize=10)
        
        # Title with context length
        title = f"Sample {sample_idx + 1}"
        if context_length > 0:
            title += f" (Context: {context_length:,} tokens)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")
    
    # Use last subplot for legend and notes
    legend_ax = axes[max_samples]
    legend_ax.axis('off')  # Hide axes
    
    # Collect all labels for legend
    all_labels = []
    all_handles = []
    for exp_name in sorted(exp_names):
        clean_exp_name = clean_experiment_name(exp_name)
        # Create a dummy line for legend
        line = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, marker="o", markersize=6)
        all_handles.append(line)
        all_labels.append(clean_exp_name)
    
    # Add legend
    legend_ax.legend(all_handles, all_labels, loc="upper left", fontsize=9, ncol=1, frameon=True)
    
    # Add notes
    note_text = (
        "Attention Weight Relative L2 Difference: ||sparse - dense|| / ||dense|| (normalized by dense weights norm).\n"
        "All experiments use chunked prefill with chunk size 4096.\n"
        "Metrics: Layer 15, Head 10 (sampled for memory efficiency).\n\n"
        "Step Index: Sequential processing steps.\n"
        "Chunks (solid lines with circles): Prefill phases where multiple tokens are processed together.\n"
        "Generation (dashed lines with squares): Token-by-token generation starts after dotted vertical line.\n\n"
        "Relative L2 Difference: Lower values indicate closer match to dense attention.\n"
        "0.0 = identical, 1.0 = difference magnitude equals dense weights magnitude.\n"
        "Log scale on y-axis for better visualization of differences across orders of magnitude."
    )
    legend_ax.text(0.05, 0.5, note_text, transform=legend_ax.transAxes,
                   fontsize=8, va="center", ha="left",
                   bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.8, edgecolor="gray"))
    
    # Hide unused subplots
    for idx in range(max_samples + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_samples_comparison.png")
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
    
    # Use a colormap with more distinct colors
    if len(experiments_data) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments_data)))
    
    # Different line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    for idx, (exp_name, exp_data) in enumerate(experiments_data.items()):
        chunk_sizes = [k for (q, k) in all_chunks]
        l2_values = [exp_data.get(chunk, {}).get("l2_diff_relative", 0.0) for chunk in all_chunks]
        
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        ax.plot(chunk_sizes, l2_values, marker=marker, label=clean_experiment_name(exp_name), 
                color=colors[idx], linewidth=2.5, markersize=7, alpha=0.85, linestyle=linestyle)
    
    ax.set_xlabel("Context Size (keys)", fontsize=12)
    ax.set_ylabel("Relative L2 Difference", fontsize=12)
    ax.set_title("Attention Weight Relative L2 Difference vs Context Size\n(Layer 15, Head 10)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "chunk_progression.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_mask_samples_comparison(
    experiments_mask_samples: Dict[str, List[List[Dict]]],
    output_dir: str,
    chunk_threshold: int = 100,
) -> None:
    """
    Plot mask comparison metrics for all samples, comparing across experiments.
    Similar to plot_all_samples_comparison but for mask metrics (jaccard similarity).
    
    Args:
        experiments_mask_samples: Dictionary mapping experiment names to list of samples (each sample is list of mask metrics)
        output_dir: Directory to save plots
        chunk_threshold: Threshold for distinguishing chunks from generation
    """
    if not experiments_mask_samples:
        return
    
    # Find maximum number of samples across all experiments
    max_samples = max(len(samples) for samples in experiments_mask_samples.values())
    
    if max_samples == 0:
        return
    
    # Create figure with subplots (one per sample + one empty for legend/notes)
    n_cols = min(3, max_samples + 1)  # +1 for legend/notes subplot
    n_rows = ((max_samples + 1) + n_cols - 1) // n_cols  # +1 for legend/notes subplot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    # Flatten axes array for easier indexing
    if (max_samples + 1) == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Colors and styles for different experiments
    exp_names = sorted(experiments_mask_samples.keys())
    # Use a colormap with more distinct colors
    if len(exp_names) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    exp_colors = dict(zip(exp_names, colors))
    
    # Different line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    # Plot each sample
    for sample_idx in range(max_samples):
        ax = axes[sample_idx]
        
        # Calculate context length for this sample (use first experiment that has this sample)
        context_length = 0
        for exp_name in exp_names:
            samples = experiments_mask_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                if sample_metrics:
                    # Calculate context length from mask metrics (same logic as attention metrics)
                    chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
                    if chunks:
                        context_length = max(m["metadata"].get("seq_len_k", 0) for m in chunks)
                    break
        
        # Plot all experiments for this sample
        for exp_idx, exp_name in enumerate(exp_names):
            samples = experiments_mask_samples[exp_name]
            if sample_idx >= len(samples):
                continue
            
            sample_metrics = samples[sample_idx]
            if not sample_metrics:
                continue
            
            # Separate chunks and generation
            chunks, generation = separate_chunks_and_generation(sample_metrics, chunk_threshold)
            
            # Get style for this experiment
            linestyle = line_styles[exp_idx % len(line_styles)]
            marker = markers[exp_idx % len(markers)]
            
            # Plot chunks - use jaccard similarity
            chunk_jaccard = [m["value"].get("jaccard_similarity", 0.0) for m in chunks]
            chunk_indices = list(range(len(chunks)))
            
            if chunks:
                clean_exp_name = clean_experiment_name(exp_name)
                ax.plot(chunk_indices, chunk_jaccard, marker=marker, label=clean_exp_name, 
                       color=exp_colors[exp_name], linewidth=2.5, markersize=7, 
                       alpha=0.85, zorder=3, linestyle=linestyle, markevery=1)
            
            # Plot generation (if any mask metrics exist for generation)
            gen_jaccard = [m["value"].get("jaccard_similarity", 0.0) for m in generation]
            gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
            
            if generation:
                clean_exp_name = clean_experiment_name(exp_name)
                ax.plot(gen_indices, gen_jaccard, marker=marker, 
                       color=exp_colors[exp_name], linewidth=2, markersize=5, 
                       alpha=0.75, zorder=3, linestyle=linestyle, markevery=1)
        
        # Add vertical line to separate chunks from generation
        max_chunks = 0
        for exp_name in exp_names:
            samples = experiments_mask_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
                max_chunks = max(max_chunks, len(chunks))
        
        if max_chunks > 0:
            boundary_x = max_chunks - 0.5
            ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=1, 
                      alpha=0.5, zorder=1)
        
        # Add horizontal line at 1.0 for perfect match
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
        
        # Formatting
        ax.set_xlabel("Step Index (chunks: solid lines, generation: after dotted line)", fontsize=9)
        ax.set_ylabel("Jaccard Similarity", fontsize=10)
        
        # Title with context length
        title = f"Sample {sample_idx + 1}"
        if context_length > 0:
            title += f" (Context: {context_length:,} tokens)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])
    
    # Use last subplot for legend and notes
    legend_ax = axes[max_samples]
    legend_ax.axis('off')  # Hide axes
    
    # Collect all labels for legend
    all_labels = []
    all_handles = []
    for exp_name in sorted(experiments_mask_samples.keys()):
        clean_exp_name = clean_experiment_name(exp_name)
        # Create a dummy line for legend
        line = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, marker="o", markersize=6)
        all_handles.append(line)
        all_labels.append(clean_exp_name)
    
    # Add legend
    legend_ax.legend(all_handles, all_labels, loc="upper left", fontsize=9, ncol=1, frameon=True)
    
    # Add notes and Jaccard explanation
    note_text = (
        "Mask Comparison: Jaccard similarity = |Intersection| / |Union| between masks\n"
        "(roped Q/K vs unroped Q/K). All experiments use chunked prefill with chunk size 4096.\n"
        "Metrics: Layer 15, Head 0 (sampled for memory efficiency).\n\n"
        "Jaccard Similarity = |A ∩ B| / |A ∪ B|\n"
        "Example: Mask A selects {1,2,3}, Mask B selects {2,3,4}\n"
        "  Intersection: {2,3} (2), Union: {1,2,3,4} (4)\n"
        "  Jaccard = 2/4 = 0.5 (50% overlap)\n"
        "  1.0 = identical masks, 0.0 = no overlap\n\n"
        "Step Index: Sequential processing steps.\n"
        "Chunks (solid lines with circles): Prefill phases.\n"
        "Generation (dashed lines with squares): Starts after dotted vertical line."
    )
    legend_ax.text(0.05, 0.5, note_text, transform=legend_ax.transAxes,
                   fontsize=8, va="center", ha="left",
                   bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.8, edgecolor="gray"))
    
    # Hide unused subplots
    for idx in range(max_samples + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "mask_samples_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_mask_comparison(experiments_mask_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]], output_dir: str) -> None:
    """
    Plot mask comparison metrics (jaccard similarity, overlap percentage, diff percentage).
    Compares masks computed with roped vs unroped Q/K.
    
    Args:
        experiments_mask_data: Dictionary mapping experiment names to mask metrics by chunk
        output_dir: Directory to save plots
    """
    if not experiments_mask_data:
        print("No mask comparison data to plot.")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get all unique chunks (sorted by seq_len_k)
    all_chunks = set()
    for exp_data in experiments_mask_data.values():
        all_chunks.update(exp_data.keys())
    all_chunks = sorted(all_chunks, key=lambda x: x[1])  # Sort by seq_len_k
    
    # Use a colormap with more distinct colors
    if len(experiments_mask_data) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_mask_data)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments_mask_data)))
    
    # Different line styles and markers for better distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    # Plot 1: Jaccard Similarity
    ax1 = axes[0]
    for idx, (exp_name, exp_data) in enumerate(experiments_mask_data.items()):
        chunk_sizes = [k for (q, k) in all_chunks]
        jaccard_values = [exp_data.get(chunk, {}).get("jaccard_similarity", 0.0) for chunk in all_chunks]
        
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        ax1.plot(chunk_sizes, jaccard_values, marker=marker, label=clean_experiment_name(exp_name), 
                color=colors[idx], linewidth=2.5, markersize=7, alpha=0.85, linestyle=linestyle)
    
    ax1.set_xlabel("Context Size (seq_len_k)", fontsize=11)
    ax1.set_ylabel("Jaccard Similarity", fontsize=11)
    ax1.set_title("Mask Jaccard Similarity vs Context Size\n(Layer 15, Head 0)", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label="Perfect Match")
    
    # Plot 2: Overlap Percentage
    ax2 = axes[1]
    for idx, (exp_name, exp_data) in enumerate(experiments_mask_data.items()):
        chunk_sizes = [k for (q, k) in all_chunks]
        overlap_values = [exp_data.get(chunk, {}).get("overlap_percentage", 0.0) for chunk in all_chunks]
        
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        ax2.plot(chunk_sizes, overlap_values, marker=marker, label=clean_experiment_name(exp_name), 
                color=colors[idx], linewidth=2.5, markersize=7, alpha=0.85, linestyle=linestyle)
    
    ax2.set_xlabel("Context Size (seq_len_k)", fontsize=11)
    ax2.set_ylabel("Overlap Percentage", fontsize=11)
    ax2.set_title("Mask Overlap Percentage vs Context Size", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 3: Difference Percentage
    ax3 = axes[2]
    for idx, (exp_name, exp_data) in enumerate(experiments_mask_data.items()):
        chunk_sizes = [k for (q, k) in all_chunks]
        diff_values = [exp_data.get(chunk, {}).get("diff_percentage", 0.0) for chunk in all_chunks]
        
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        ax3.plot(chunk_sizes, diff_values, marker=marker, label=clean_experiment_name(exp_name), 
                color=colors[idx], linewidth=2.5, markersize=7, alpha=0.85, linestyle=linestyle)
    
    ax3.set_xlabel("Context Size (seq_len_k)", fontsize=11)
    ax3.set_ylabel("Difference Percentage", fontsize=11)
    ax3.set_title("Mask Difference Percentage vs Context Size", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    ax3.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label="No Difference")
    
    # Add footer note with Jaccard explanation
    note_lines = [
        "Mask Comparison: Jaccard similarity = |Intersection| / |Union| between masks (roped Q/K vs unroped Q/K).",
        "Metrics: Layer 15, Head 0 (sampled for memory efficiency). Example: If Mask A selects keys {1,2,3} and Mask B selects {2,3,4}, then Jaccard = 2/4 = 0.5 (50% overlap).",
        "Only prefill chunks shown (generation steps excluded). Sanity check (roped vs roped) should show 100% similarity."
    ]
    for i, line in enumerate(note_lines):
        fig.text(0.5, 0.01 + i * 0.012, line, 
                 ha="center", fontsize=7.5, style="italic", color="gray")
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for footer (3 lines)
    output_path = os.path.join(output_dir, "mask_comparison.png")
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
    print("SUMMARY TABLE: Average Relative L2 Differences by Experiment")
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
        
        # Get relative L2 values for first 3 chunks
        l2_values = []
        for chunk in all_chunks[:3]:
            l2 = exp_data.get(chunk, {}).get("l2_diff_relative", 0.0)
            l2_values.append(l2)
        
        # Calculate average across all chunks
        all_l2 = [exp_data.get(chunk, {}).get("l2_diff_relative", 0.0) for chunk in all_chunks]
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
    all_experiments = find_experiment_dirs(base_dir)
    print(f"Found {len(all_experiments)} experiments:")
    for exp_name, exp_path in all_experiments:
        print(f"  - {exp_name}")
    
    # Filter duplicates (keep only latest sanity, rope, etc.)
    experiments = filter_duplicate_experiments(all_experiments)
    print(f"\nAfter filtering duplicates: {len(experiments)} experiments:")
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
        
        # Validate samples
        print(f"  Validating samples for {exp_name}...")
        for sample_idx, sample_metrics in enumerate(samples):
            validation = validate_sample_metrics(sample_metrics)
            if validation["errors"]:
                print(f"    Sample {sample_idx + 1}: ERRORS - {validation['errors']}")
            if validation["warnings"]:
                print(f"    Sample {sample_idx + 1}: Warnings - {validation['warnings']}")
        
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
    
    # Generate comparison plot: all samples, all experiments
    print("\nGenerating all-samples comparison plot...")
    plot_all_samples_comparison(experiments_samples, output_dir)
    
    # Generate summary plots
    print("\nGenerating summary plots...")
    plot_l2_comparison(experiments_data, output_dir)
    plot_metrics_overview(experiments_data, output_dir)
    plot_chunk_progression(experiments_data, output_dir)
    
    # Load and plot mask comparison metrics (if available)
    print("\nLoading mask comparison metrics...")
    experiments_mask_data: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    experiments_mask_samples: Dict[str, List[List[Dict]]] = {}  # Per-experiment, per-sample mask metrics
    
    for exp_name, exp_path in experiments:
        metrics_file = os.path.join(exp_path, "micro_metrics.jsonl")
        mask_metrics = load_mask_metrics(metrics_file)
        
        if mask_metrics:
            print(f"Loaded {len(mask_metrics)} mask metrics from {exp_name}")
            
            # Group by sample for per-sample plotting (filter to layer 15 for consistency)
            mask_samples = group_mask_metrics_by_sample(mask_metrics, chunk_threshold=100, filter_layer=15)
            experiments_mask_samples[exp_name] = mask_samples
            print(f"  Grouped into {len(mask_samples)} samples (layer 15 only)")
            
            # Also organize by chunk for summary plot (filter to layer 15 for consistency)
            organized = organize_mask_metrics_by_chunk(mask_metrics, filter_layer=15)
            # Filter to only chunks (seq_len_q >= 100) to avoid generation steps
            chunk_organized = {
                chunk: metrics for chunk, metrics in organized.items()
                if chunk[0] >= 100  # Only chunks, not generation
            }
            chunk_stats = {chunk: compute_mask_chunk_statistics(chunk_metrics) 
                          for chunk, chunk_metrics in chunk_organized.items()}
            
            if chunk_stats:
                experiments_mask_data[exp_name] = chunk_stats
    
    # Generate mask comparison plots if we have data
    if experiments_mask_data:
        print("\nGenerating mask comparison plots...")
        # Summary plot (aggregated by chunk)
        plot_mask_comparison(experiments_mask_data, output_dir)
        
        # Per-sample plot (similar to all_samples_comparison)
        if experiments_mask_samples:
            print("Generating per-sample mask comparison plot...")
            plot_mask_samples_comparison(experiments_mask_samples, output_dir)
    else:
        print("No mask comparison data found. Skipping mask comparison plots.")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

