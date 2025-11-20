#!/usr/bin/env python3
"""
Plot RULER 16k results similar to HotpotQA plotting.

Generates:
- all_samples_comparison.png: L2 relative difference across chunks and generation
- mask_samples_comparison.png: Jaccard similarity across chunks and generation

Usage:
    python plot_ruler16k.py <output_directory>
    
Example:
    python plot_ruler16k.py output_llama31_8b_ruler16k_8kctx_hs010_pcs1024_old
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
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


def is_chunk(seq_len_q: int, chunk_threshold: int = 100) -> bool:
    """Determine if this is a chunk (prefill) or generation step."""
    return seq_len_q >= chunk_threshold


def separate_chunks_and_generation(metrics: List[Dict], chunk_threshold: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """Separate metrics into chunks (prefill) and generation steps."""
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


def group_metrics_by_sample(metrics: List[Dict], chunk_threshold: int = 100) -> List[List[Dict]]:
    """
    Group metrics by sample/example.
    
    Detects sample boundaries by:
    - Reset to first chunk (seq_len_q == seq_len_k and seq_len_q >= chunk_threshold) after generation (seq_len_q == 1)
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
        
        # Detect new sample: reset to first chunk after generation
        is_new_sample = False
        
        if prev_seq_len_q == 1 and seq_len_q == seq_len_k and seq_len_q >= chunk_threshold:
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


def group_mask_metrics_by_sample(metrics: List[Dict], chunk_threshold: int = 100, filter_layer: int = 15) -> List[List[Dict]]:
    """Group mask metrics by sample, filtering to specified layer."""
    if not metrics:
        return []
    
    # Filter to only specified layer first
    filtered_metrics = [
        m for m in metrics 
        if m.get("metadata", {}).get("layer_idx", -1) == filter_layer
    ]
    
    return group_metrics_by_sample(filtered_metrics, chunk_threshold)


def calculate_context_length(sample_metrics: List[Dict], chunk_threshold: int = 100) -> int:
    """Calculate total context length (max seq_len_k from chunks)."""
    chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
    
    if not chunks:
        return 0
    
    max_context = max(m["metadata"].get("seq_len_k", 0) for m in chunks)
    return max_context


def plot_all_samples_comparison(
    experiments_samples: Dict[str, List[List[Dict]]],
    output_dir: str,
    chunk_threshold: int = 100,
    original_sample_indices: Optional[List[int]] = None,
) -> None:
    """
    Plot all samples in a grid, comparing L2 across all experiments for each sample.
    Matches format from plot_repositioning_chunkwise.py
    """
    if not experiments_samples:
        return
    
    # Find maximum number of samples across all experiments
    max_samples = max(len(samples) for samples in experiments_samples.values())
    
    if max_samples == 0:
        return
    
    # Create figure with subplots (one per sample + one empty for legend/notes)
    n_cols = min(3, max_samples + 1)
    n_rows = ((max_samples + 1) + n_cols - 1) // n_cols
    
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
    if len(exp_names) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    exp_colors = dict(zip(exp_names, colors))
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    # Plot each sample
    for sample_idx in range(max_samples):
        ax = axes[sample_idx]
        
        # Calculate context length for this sample
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
                clean_exp_name = exp_name.replace("_", " ").title()
                ax.plot(chunk_indices, chunk_l2, marker=marker, label=clean_exp_name, 
                       color=exp_colors[exp_name], linewidth=2.5, markersize=7, 
                       alpha=0.85, zorder=3, linestyle=linestyle, markevery=1)
            
            # Plot generation - use relative L2 difference
            gen_l2 = [m["value"].get("l2_diff_relative", m["value"].get("l2_diff", 0.0)) for m in generation]
            gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
            
            if generation:
                clean_exp_name = exp_name.replace("_", " ").title()
                ax.plot(gen_indices, gen_l2, marker=marker, 
                       color=exp_colors[exp_name], linewidth=2, markersize=5, 
                       alpha=0.75, zorder=3, linestyle=linestyle, markevery=1)
        
        # Add vertical line to separate chunks from generation
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
        if original_sample_indices and sample_idx < len(original_sample_indices):
            orig_num = original_sample_indices[sample_idx]
            title = f"Sample {orig_num} (plot {sample_idx + 1})"
        else:
            title = f"Sample {sample_idx + 1}"
        if context_length > 0:
            title += f" (Context: {context_length:,} tokens)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")
    
    # Use last subplot for legend and notes
    legend_ax = axes[max_samples]
    legend_ax.axis('off')
    
    # Collect all labels for legend
    all_labels = []
    all_handles = []
    for exp_name in sorted(exp_names):
        clean_exp_name = exp_name.replace("_", " ").title()
        line = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, marker="o", markersize=6)
        all_handles.append(line)
        all_labels.append(clean_exp_name)
    
    # Add legend
    legend_ax.legend(all_handles, all_labels, loc="upper left", fontsize=9, ncol=1, frameon=True)
    
    # Add notes
    note_text = (
        "Attention Weight Relative L2 Difference: ||sparse - dense|| / ||dense|| (normalized by dense weights norm).\n"
        "All experiments use chunked prefill with chunk size 1024.\n"
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


def plot_mask_samples_comparison(
    experiments_mask_samples: Dict[str, List[List[Dict]]],
    output_dir: str,
    chunk_threshold: int = 100,
    original_sample_indices: Optional[List[int]] = None,
) -> None:
    """
    Plot mask comparison metrics for all samples, comparing across experiments.
    Matches format from plot_repositioning_chunkwise.py
    """
    if not experiments_mask_samples:
        return
    
    # Find maximum number of samples across all experiments
    max_samples = max(len(samples) for samples in experiments_mask_samples.values())
    
    if max_samples == 0:
        return
    
    # Create figure with subplots (one per sample + one empty for legend/notes)
    n_cols = min(3, max_samples + 1)
    n_rows = ((max_samples + 1) + n_cols - 1) // n_cols
    
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
    if len(exp_names) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    exp_colors = dict(zip(exp_names, colors))
    
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    # Plot each sample
    for sample_idx in range(max_samples):
        ax = axes[sample_idx]
        
        # Calculate context length for this sample
        context_length = 0
        for exp_name in exp_names:
            samples = experiments_mask_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                if sample_metrics:
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
                clean_exp_name = exp_name.replace("_", " ").title()
                ax.plot(chunk_indices, chunk_jaccard, marker=marker, label=clean_exp_name, 
                       color=exp_colors[exp_name], linewidth=2.5, markersize=7, 
                       alpha=0.85, zorder=3, linestyle=linestyle, markevery=1)
            
            # Plot generation (if any mask metrics exist for generation)
            gen_jaccard = [m["value"].get("jaccard_similarity", 0.0) for m in generation]
            gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
            
            if generation:
                clean_exp_name = exp_name.replace("_", " ").title()
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
        if original_sample_indices and sample_idx < len(original_sample_indices):
            orig_num = original_sample_indices[sample_idx]
            title = f"Sample {orig_num} (plot {sample_idx + 1})"
        else:
            title = f"Sample {sample_idx + 1}"
        if context_length > 0:
            title += f" (Context: {context_length:,} tokens)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.05])
    
    # Use last subplot for legend and notes
    legend_ax = axes[max_samples]
    legend_ax.axis('off')
    
    # Collect all labels for legend
    all_labels = []
    all_handles = []
    for exp_name in sorted(exp_names):
        clean_exp_name = exp_name.replace("_", " ").title()
        line = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, marker="o", markersize=6)
        all_handles.append(line)
        all_labels.append(clean_exp_name)
    
    # Add legend
    legend_ax.legend(all_handles, all_labels, loc="upper left", fontsize=9, ncol=1, frameon=True)
    
    # Add notes
    note_text = (
        "Mask Comparison: Jaccard similarity = |Intersection| / |Union| between masks\n"
        "(roped Q/K vs unroped Q/K). All experiments use chunked prefill with chunk size 1024.\n"
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


def main():
    """Main function to generate plots for RULER 16k results."""
    parser = argparse.ArgumentParser(
        description="Plot RULER 16k results similar to HotpotQA plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python plot_ruler16k.py output_llama31_8b_ruler16k_8kctx_hs010_pcs1024_old
  
  # Compare two experiments
  python plot_ruler16k.py output_llama31_8b_ruler16k_8kctx_hs010_pcs1024 output_llama31_8b_ruler16k_naive_pcs1024_hs010
        """
    )
    parser.add_argument(
        "directory1",
        type=str,
        help="First directory containing micro_metrics.jsonl (or only directory if comparing)"
    )
    parser.add_argument(
        "directory2",
        type=str,
        nargs="?",
        default=None,
        help="Second directory for comparison (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <directory1>/plots or comparison_plots/)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to plot (default: all)"
    )
    parser.add_argument(
        "--chunk-threshold",
        type=int,
        default=100,
        help="Threshold to distinguish chunks from generation (default: 100)"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default=None,
        help="Name for first experiment (default: directory basename)"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Name for second experiment (default: directory basename)"
    )
    
    args = parser.parse_args()
    
    # Resolve directory paths
    dir1 = os.path.abspath(args.directory1)
    dir2 = os.path.abspath(args.directory2) if args.directory2 else None
    
    if not os.path.isdir(dir1):
        print(f"Error: Directory not found: {dir1}", file=sys.stderr)
        sys.exit(1)
    
    if dir2 and not os.path.isdir(dir2):
        print(f"Error: Directory not found: {dir2}", file=sys.stderr)
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif dir2:
        # Comparison mode - use comparison_plots directory
        output_dir = os.path.join(os.path.dirname(dir1), "comparison_plots")
    else:
        # Single experiment mode
        output_dir = os.path.join(dir1, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get experiment names
    name1 = args.name1 or os.path.basename(dir1.rstrip('/'))
    name2 = args.name2 or (os.path.basename(dir2.rstrip('/')) if dir2 else None)
    
    print("=" * 80)
    if dir2:
        print(f"COMPARISON MODE: {name1} vs {name2}")
        print(f"Loading metrics from:")
        print(f"  Dir1: {dir1}")
        print(f"  Dir2: {dir2}")
    else:
        print(f"SINGLE EXPERIMENT MODE: {name1}")
        print(f"Loading metrics from: {dir1}")
    print("=" * 80)
    
    # Load metrics from first directory
    metrics_file1 = os.path.join(dir1, "micro_metrics.jsonl")
    if not os.path.exists(metrics_file1):
        print(f"Error: micro_metrics.jsonl not found in {dir1}", file=sys.stderr)
        sys.exit(1)
    
    all_metrics1 = load_metrics(metrics_file1)
    print(f"Loaded {len(all_metrics1)} metrics from {name1}")
    
    # Load metrics from second directory if provided
    all_metrics2 = []
    if dir2:
        metrics_file2 = os.path.join(dir2, "micro_metrics.jsonl")
        if not os.path.exists(metrics_file2):
            print(f"Error: micro_metrics.jsonl not found in {dir2}", file=sys.stderr)
            sys.exit(1)
        all_metrics2 = load_metrics(metrics_file2)
        print(f"Loaded {len(all_metrics2)} metrics from {name2}")
    
    # Filter to attention weight diff metrics
    attn_metrics1 = [m for m in all_metrics1 if m.get("metric") == "research_attention_weight_diff"]
    attn_metrics2 = [m for m in all_metrics2 if m.get("metric") == "research_attention_weight_diff"] if all_metrics2 else []
    
    # Filter to mask metrics
    mask_metrics1 = [m for m in all_metrics1 if m.get("metric") == "research_mask_roped_vs_unroped"]
    mask_metrics2 = [m for m in all_metrics2 if m.get("metric") == "research_mask_roped_vs_unroped"] if all_metrics2 else []
    
    print(f"\nAttention metrics: {len(attn_metrics1)} ({name1})" + (f", {len(attn_metrics2)} ({name2})" if dir2 else ""))
    print(f"Mask metrics: {len(mask_metrics1)} ({name1})" + (f", {len(mask_metrics2)} ({name2})" if dir2 else ""))
    
    # Group by sample
    print("\nGrouping metrics by sample...")
    samples_attn1 = group_metrics_by_sample(attn_metrics1, chunk_threshold=args.chunk_threshold)
    samples_mask1 = group_mask_metrics_by_sample(mask_metrics1, chunk_threshold=args.chunk_threshold, filter_layer=15)
    
    samples_attn2 = group_metrics_by_sample(attn_metrics2, chunk_threshold=args.chunk_threshold) if attn_metrics2 else []
    samples_mask2 = group_mask_metrics_by_sample(mask_metrics2, chunk_threshold=args.chunk_threshold, filter_layer=15) if mask_metrics2 else []
    
    print(f"Found {len(samples_attn1)} samples (attention, {name1})" + (f", {len(samples_attn2)} ({name2})" if dir2 else ""))
    print(f"Found {len(samples_mask1)} samples (mask, {name1})" + (f", {len(samples_mask2)} ({name2})" if dir2 else ""))
    
    # For comparison mode, find common samples
    if dir2:
        # Find common sample count
        common_count = min(len(samples_attn1), len(samples_attn2))
        if args.max_samples:
            common_count = min(common_count, args.max_samples)
        
        print(f"\nUsing {common_count} common samples for comparison")
        
        samples_attn1 = samples_attn1[:common_count]
        samples_attn2 = samples_attn2[:common_count]
        samples_mask1 = samples_mask1[:common_count] if len(samples_mask1) >= common_count else samples_mask1
        samples_mask2 = samples_mask2[:common_count] if len(samples_mask2) >= common_count else samples_mask2
    else:
        # Single experiment mode
        if args.max_samples:
            samples_attn1 = samples_attn1[:args.max_samples]
            samples_mask1 = samples_mask1[:args.max_samples] if len(samples_mask1) >= args.max_samples else samples_mask1
    
    # Calculate context lengths
    print("\nCalculating context lengths...")
    max_samples_to_show = min(5, len(samples_attn1))
    for sample_idx in range(max_samples_to_show):
        if sample_idx < len(samples_attn1):
            context_length = calculate_context_length(samples_attn1[sample_idx], chunk_threshold=args.chunk_threshold)
            print(f"  Sample {sample_idx + 1}: Context = {context_length:,} tokens")
    
    # Organize for plotting
    if dir2:
        # Comparison mode
        experiments_samples_attn = {
            name1: samples_attn1,
            name2: samples_attn2,
        }
        experiments_samples_mask = {
            name1: samples_mask1,
            name2: samples_mask2,
        }
        selected_indices = list(range(len(samples_attn1)))
    else:
        # Single experiment mode - convert to dict format
        experiments_samples_attn = {
            name1: samples_attn1,
        }
        experiments_samples_mask = {
            name1: samples_mask1,
        }
        selected_indices = None
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    if experiments_samples_attn.get(name1):
        plot_all_samples_comparison(
            experiments_samples_attn, 
            output_dir, 
            chunk_threshold=args.chunk_threshold,
            original_sample_indices=selected_indices
        )
    else:
        print("No attention metrics to plot")
    
    if experiments_samples_mask.get(name1):
        plot_mask_samples_comparison(
            experiments_samples_mask, 
            output_dir, 
            chunk_threshold=args.chunk_threshold,
            original_sample_indices=selected_indices
        )
    else:
        print("No mask metrics to plot")
    
    print("\n" + "=" * 80)
    print(f"✓ All plots saved to: {output_dir}")
    if dir2:
        print(f"  Compared: {name1} vs {name2}")
    print("=" * 80)


if __name__ == "__main__":
    main()

