#!/usr/bin/env python3
"""
Visualize LOFT RAG micro metrics: L2 relative attention and Jaccard similarity.
Combined plots showing both metrics for the same samples.
"""

import json
import os
import glob
from typing import Dict, List, Tuple, Any
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
    exit(1)


def find_loft_experiments(base_dir: str = ".") -> List[Tuple[str, str]]:
    """
    Find all LOFT test output directories containing micro_metrics.jsonl files.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of (experiment_name, full_path) tuples
    """
    experiments: List[Tuple[str, str]] = []
    
    # Look for test_outputs/test_loft_*/micro_metrics.jsonl
    pattern = os.path.join(base_dir, "test_outputs", "test_loft_*", "micro_metrics.jsonl")
    for metrics_file in glob.glob(pattern):
        exp_dir = os.path.dirname(metrics_file)
        exp_name = os.path.basename(exp_dir)
        experiments.append((exp_name, exp_dir))
    
    return sorted(experiments)


def load_metrics(metrics_file: str) -> List[Dict]:
    """Load attention weight diff metrics from micro_metrics.jsonl."""
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
    """Load mask comparison metrics from micro_metrics.jsonl."""
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
    """Extract chunk information from metadata."""
    seq_len_q = metadata.get("seq_len_q", 0)
    seq_len_k = metadata.get("seq_len_k", 0)
    return (seq_len_q, seq_len_k)


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
    
    samples: List[List[Dict]] = []
    current_sample: List[Dict] = []
    
    prev_seq_len_q = None
    
    for metric in filtered_metrics:
        metadata = metric.get("metadata", {})
        seq_len_q = metadata.get("seq_len_q", 0)
        seq_len_k = metadata.get("seq_len_k", 0)
        
        is_new_sample = False
        
        if prev_seq_len_q == 1 and seq_len_q == seq_len_k and seq_len_q >= chunk_threshold:
            is_new_sample = True
        
        if is_new_sample and current_sample:
            samples.append(current_sample)
            current_sample = []
        
        current_sample.append(metric)
        prev_seq_len_q = seq_len_q
    
    if current_sample:
        samples.append(current_sample)
    
    return samples


def calculate_context_length(sample_metrics: List[Dict], chunk_threshold: int = 100) -> int:
    """Calculate total context length (max seq_len_k from chunks)."""
    chunks, _ = separate_chunks_and_generation(sample_metrics, chunk_threshold)
    
    if not chunks:
        return 0
    
    max_context = max(m["metadata"].get("seq_len_k", 0) for m in chunks)
    return max_context


def clean_experiment_name(name: str) -> str:
    """Clean experiment name for display."""
    # Remove test_outputs/test_loft_ prefix if present
    if name.startswith("test_loft_"):
        name = name.replace("test_loft_", "")
    
    # Replace underscores with spaces and title case
    name = name.replace("_", " ").title()
    return name


def plot_combined_l2_and_jaccard(
    experiments_samples: Dict[str, List[List[Dict]]],
    experiments_mask_samples: Dict[str, List[List[Dict]]],
    output_dir: str,
    chunk_threshold: int = 100,
) -> None:
    """
    Plot combined L2 relative attention and Jaccard similarity for all samples.
    Uses dual y-axis: left for L2 (log scale), right for Jaccard (linear, 0-1).
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    
    # Flatten axes array
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
        
        # Calculate context length
        context_length = 0
        for exp_name in exp_names:
            samples = experiments_samples[exp_name]
            if sample_idx < len(samples):
                sample_metrics = samples[sample_idx]
                if sample_metrics:
                    context_length = calculate_context_length(sample_metrics, chunk_threshold)
                    break
        
        # Create dual y-axis
        ax_l2 = ax  # Left axis for L2
        ax_jaccard = ax.twinx()  # Right axis for Jaccard
        
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
            color = exp_colors[exp_name]
            
            # Plot L2 on left axis
            chunk_l2 = [m["value"].get("l2_diff_relative", m["value"].get("l2_diff", 0.0)) for m in chunks]
            chunk_indices = list(range(len(chunks)))
            
            if chunks:
                clean_exp_name = clean_experiment_name(exp_name)
                ax_l2.plot(chunk_indices, chunk_l2, marker=marker, 
                          color=color, linewidth=2.5, markersize=7, 
                          alpha=0.85, zorder=3, linestyle=linestyle, markevery=1,
                          label=f"{clean_exp_name} (L2)")
            
            # Plot generation L2
            gen_l2 = [m["value"].get("l2_diff_relative", m["value"].get("l2_diff", 0.0)) for m in generation]
            gen_indices = list(range(len(chunks), len(chunks) + len(generation)))
            
            if generation:
                ax_l2.plot(gen_indices, gen_l2, marker=marker, 
                          color=color, linewidth=2, markersize=5, 
                          alpha=0.75, zorder=3, linestyle=linestyle, markevery=1)
            
            # Plot Jaccard on right axis (if mask metrics available)
            if exp_name in experiments_mask_samples:
                mask_samples = experiments_mask_samples[exp_name]
                if sample_idx < len(mask_samples):
                    mask_sample_metrics = mask_samples[sample_idx]
                    if mask_sample_metrics:
                        mask_chunks, mask_gen = separate_chunks_and_generation(mask_sample_metrics, chunk_threshold)
                        
                        # Plot chunks Jaccard
                        chunk_jaccard = [m["value"].get("jaccard_similarity", 0.0) for m in mask_chunks]
                        chunk_indices_j = list(range(len(mask_chunks)))
                        
                        if mask_chunks:
                            # Use slightly different style for Jaccard (dashed line, different marker)
                            ax_jaccard.plot(chunk_indices_j, chunk_jaccard, 
                                          marker='s', color=color, linewidth=2, 
                                          markersize=5, alpha=0.7, zorder=2, 
                                          linestyle=':', markevery=1,
                                          label=f"{clean_exp_name} (Jaccard)")
                        
                        # Plot generation Jaccard
                        gen_jaccard = [m["value"].get("jaccard_similarity", 0.0) for m in mask_gen]
                        gen_indices_j = list(range(len(mask_chunks), len(mask_chunks) + len(mask_gen)))
                        
                        if mask_gen:
                            ax_jaccard.plot(gen_indices_j, gen_jaccard, 
                                          marker='s', color=color, linewidth=1.5, 
                                          markersize=4, alpha=0.6, zorder=2, 
                                          linestyle=':', markevery=1)
        
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
            ax_l2.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=1, 
                         alpha=0.5, zorder=1)
        
        # Add horizontal line at 1.0 for perfect Jaccard match
        ax_jaccard.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
        
        # Formatting
        ax_l2.set_xlabel("Step Index (chunks: solid lines, generation: after dotted line)", fontsize=9)
        ax_l2.set_ylabel("Relative L2 Difference", fontsize=10, color='blue')
        ax_jaccard.set_ylabel("Jaccard Similarity", fontsize=10, color='red')
        
        # Color the y-axis labels
        ax_l2.tick_params(axis='y', labelcolor='blue')
        ax_jaccard.tick_params(axis='y', labelcolor='red')
        
        # Title with context length
        title = f"Sample {sample_idx + 1}"
        if context_length > 0:
            title += f" (Context: {context_length:,} tokens)"
        ax_l2.set_title(title, fontsize=11, fontweight="bold")
        ax_l2.grid(True, alpha=0.3, axis="y")
        ax_l2.set_yscale("log")
        ax_jaccard.set_ylim([0, 1.05])
    
    # Use last subplot for legend and notes
    legend_ax = axes[max_samples]
    legend_ax.axis('off')
    
    # Collect all labels for legend
    all_labels = []
    all_handles = []
    for exp_name in sorted(exp_names):
        clean_exp_name = clean_experiment_name(exp_name)
        # Create dummy lines for legend
        line_l2 = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, 
                        marker="o", markersize=6, linestyle='-', label=f"{clean_exp_name} (L2)")
        all_handles.append(line_l2)
        all_labels.append(f"{clean_exp_name} (L2)")
        
        if exp_name in experiments_mask_samples:
            line_jaccard = Line2D([0], [0], color=exp_colors[exp_name], linewidth=2, 
                                 marker="s", markersize=5, linestyle=':', 
                                 label=f"{clean_exp_name} (Jaccard)")
            all_handles.append(line_jaccard)
            all_labels.append(f"{clean_exp_name} (Jaccard)")
    
    # Add legend
    legend_ax.legend(all_handles, all_labels, loc="upper left", fontsize=8, ncol=1, frameon=True)
    
    # Add notes
    note_text = (
        "Combined Metrics Plot:\n"
        "• Left Y-axis (blue): Relative L2 Difference = ||sparse - dense|| / ||dense|| (log scale)\n"
        "• Right Y-axis (red): Jaccard Similarity = |Intersection| / |Union| between masks (0-1)\n\n"
        "Metrics: Layer 15, Head 10 (L2) and Head 0 (Jaccard).\n\n"
        "Step Index: Sequential processing steps.\n"
        "Chunks (solid lines): Prefill phases.\n"
        "Generation (after dotted line): Token-by-token generation.\n\n"
        "Expected: Jaccard ≈ 1.0 (perfect match) after re-roping.\n"
        "L2 shows attention weight differences (lower is better)."
    )
    legend_ax.text(0.05, 0.5, note_text, transform=legend_ax.transAxes,
                   fontsize=7.5, va="center", ha="left",
                   bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.8, edgecolor="gray"))
    
    # Hide unused subplots
    for idx in range(max_samples + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "loft_combined_l2_jaccard.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate LOFT visualizations."""
    base_dir = "."
    output_dir = "figures_loft_metrics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all LOFT experiments
    experiments = find_loft_experiments(base_dir)
    print(f"Found {len(experiments)} LOFT experiments:")
    for exp_name, exp_path in experiments:
        print(f"  - {exp_name}")
    
    if not experiments:
        print("No LOFT experiments found. Looking for test_outputs/test_loft_*/micro_metrics.jsonl files.")
        return
    
    # Load and organize data
    experiments_samples: Dict[str, List[List[Dict]]] = {}
    experiments_mask_samples: Dict[str, List[List[Dict]]] = {}
    
    for exp_name, exp_path in experiments:
        metrics_file = os.path.join(exp_path, "micro_metrics.jsonl")
        
        # Load attention weight diff metrics
        metrics = load_metrics(metrics_file)
        if metrics:
            samples = group_metrics_by_sample(metrics)
            experiments_samples[exp_name] = samples
            print(f"Loaded {len(metrics)} L2 metrics from {exp_name} ({len(samples)} samples)")
        
        # Load mask comparison metrics
        mask_metrics = load_mask_metrics(metrics_file)
        if mask_metrics:
            mask_samples = group_mask_metrics_by_sample(mask_metrics, chunk_threshold=100, filter_layer=15)
            experiments_mask_samples[exp_name] = mask_samples
            print(f"Loaded {len(mask_metrics)} mask metrics from {exp_name} ({len(mask_samples)} samples)")
    
    if not experiments_samples:
        print("No data to plot.")
        return
    
    # Generate combined plot
    print("\nGenerating combined L2 and Jaccard plot...")
    plot_combined_l2_and_jaccard(experiments_samples, experiments_mask_samples, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

