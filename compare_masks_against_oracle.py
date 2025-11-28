#!/usr/bin/env python3
"""
Compare all masks against Llama-3.1 roped mask (oracle).
Chunk-wise comparison with plots.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("Error: matplotlib not found. Install with: pip install matplotlib")
    exit(1)


# Hardcoded paths
BASE_DIR = Path("/data/masks")
LLAMA31_MODEL = "meta_llama_Llama_3.1_8B_Instruct"  # Note: dot in "3.1"
LLAMA3_MODEL = "meta_llama_Meta_Llama_3_8B_Instruct"
LAYER_IDX = 15
HEAD_IDX = 0
SAMPLE_IDX = 0


def load_mask(filepath: Path) -> Optional[np.ndarray]:
    """Load mask from .npy file."""
    if not filepath.exists():
        return None
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_metadata(filepath: Path) -> Optional[Dict]:
    """Load metadata from JSON file."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_mask_files(model_dir: Path, mask_type: str) -> Dict[Tuple[int, int], Path]:
    """Find all mask files for a model and mask type.
    
    Returns:
        Dictionary mapping (seq_len_q, seq_len_k) to filepath
    """
    masks: Dict[Tuple[int, int], Path] = {}
    
    layer_dir = model_dir / f"layer_{LAYER_IDX}" / f"head_{HEAD_IDX}"
    if not layer_dir.exists():
        print(f"Warning: Directory not found: {layer_dir}")
        return masks
    
    pattern = f"mask_{mask_type}_*.npy"
    mask_files = list(layer_dir.glob(pattern))
    print(f"  Found {len(mask_files)} files matching '{pattern}' in {layer_dir}")
    
    for mask_file in mask_files:
        # Parse filename: mask_{type}_l{layer}_h{head}_q{seq_q}_k{seq_k}_s{sample}.npy
        # Example: mask_roped_l15_h0_q1024_k1024_s0.npy
        # Parts: ['mask', 'roped', 'l15', 'h0', 'q1024', 'k1024', 's0']
        parts = mask_file.stem.split("_")
        try:
            # Find parts that start with 'q', 'k', 's'
            seq_len_q = None
            seq_len_k = None
            sample_idx = None
            
            for part in parts:
                if part.startswith("q"):
                    seq_len_q = int(part[1:])  # Remove 'q' prefix
                elif part.startswith("k"):
                    seq_len_k = int(part[1:])  # Remove 'k' prefix
                elif part.startswith("s"):
                    sample_idx = int(part[1:])  # Remove 's' prefix
            
            if seq_len_q is None or seq_len_k is None or sample_idx is None:
                print(f"  Warning: Could not parse {mask_file.name}")
                continue
            
            if sample_idx == SAMPLE_IDX:
                masks[(seq_len_q, seq_len_k)] = mask_file
        except (ValueError, IndexError) as e:
            print(f"  Warning: Error parsing {mask_file.name}: {e}")
            continue
    
    return masks


def compute_jaccard(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Jaccard similarity between two masks."""
    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0
    
    intersection = np.sum(mask1_bool & mask2_bool)
    union = np.sum(mask1_bool | mask2_bool)
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def compare_all_against_oracle():
    """Compare all masks against Llama-3.1 roped mask (oracle)."""
    
    # Paths
    llama31_dir = BASE_DIR / LLAMA31_MODEL
    llama3_dir = BASE_DIR / LLAMA3_MODEL
    
    if not llama31_dir.exists():
        print(f"Error: Llama-3.1 directory not found: {llama31_dir}")
        return
    
    if not llama3_dir.exists():
        print(f"Error: Llama-3 directory not found: {llama3_dir}")
        return
    
    # Load oracle (Llama-3.1 roped masks)
    oracle_masks = find_mask_files(llama31_dir, "roped")
    if not oracle_masks:
        print(f"Error: No oracle masks found in {llama31_dir}")
        return
    
    print(f"Found {len(oracle_masks)} oracle masks (Llama-3.1 roped)")
    
    # Load comparison masks
    llama31_unroped = find_mask_files(llama31_dir, "computed")
    llama3_unroped = find_mask_files(llama3_dir, "computed")
    llama3_roped = find_mask_files(llama3_dir, "roped")
    
    print(f"Found {len(llama31_unroped)} Llama-3.1 unroped masks")
    print(f"Found {len(llama3_unroped)} Llama-3 unroped masks")
    print(f"Found {len(llama3_roped)} Llama-3 roped masks")
    
    # Find common chunks (by seq_len_k)
    common_chunks = sorted(set(oracle_masks.keys()), key=lambda x: x[1])  # Sort by seq_len_k
    
    # Compute comparisons
    comparisons = {
        "Llama-3.1 unroped vs oracle": [],
        "Llama-3 unroped vs oracle": [],
        "Llama-3 roped vs oracle": [],
    }
    
    seq_len_ks = []
    
    for seq_len_q, seq_len_k in common_chunks:
        # Load oracle mask
        oracle_mask = load_mask(oracle_masks[(seq_len_q, seq_len_k)])
        if oracle_mask is None:
            continue
        
        seq_len_ks.append(seq_len_k)
        
        # Compare Llama-3.1 unroped vs oracle
        if (seq_len_q, seq_len_k) in llama31_unroped:
            mask31_unroped = load_mask(llama31_unroped[(seq_len_q, seq_len_k)])
            if mask31_unroped is not None and mask31_unroped.shape == oracle_mask.shape:
                jaccard = compute_jaccard(mask31_unroped, oracle_mask)
                comparisons["Llama-3.1 unroped vs oracle"].append(jaccard)
            else:
                comparisons["Llama-3.1 unroped vs oracle"].append(np.nan)
        else:
            comparisons["Llama-3.1 unroped vs oracle"].append(np.nan)
        
        # Compare Llama-3 unroped vs oracle
        if (seq_len_q, seq_len_k) in llama3_unroped:
            mask3_unroped = load_mask(llama3_unroped[(seq_len_q, seq_len_k)])
            if mask3_unroped is not None and mask3_unroped.shape == oracle_mask.shape:
                jaccard = compute_jaccard(mask3_unroped, oracle_mask)
                comparisons["Llama-3 unroped vs oracle"].append(jaccard)
            else:
                comparisons["Llama-3 unroped vs oracle"].append(np.nan)
        else:
            comparisons["Llama-3 unroped vs oracle"].append(np.nan)
        
        # Compare Llama-3 roped vs oracle
        if (seq_len_q, seq_len_k) in llama3_roped:
            mask3_roped = load_mask(llama3_roped[(seq_len_q, seq_len_k)])
            if mask3_roped is not None and mask3_roped.shape == oracle_mask.shape:
                jaccard = compute_jaccard(mask3_roped, oracle_mask)
                comparisons["Llama-3 roped vs oracle"].append(jaccard)
            else:
                comparisons["Llama-3 roped vs oracle"].append(np.nan)
        else:
            comparisons["Llama-3 roped vs oracle"].append(np.nan)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        "Llama-3.1 unroped vs oracle": "blue",
        "Llama-3 unroped vs oracle": "red",
        "Llama-3 roped vs oracle": "green",
    }
    
    markers = {
        "Llama-3.1 unroped vs oracle": "o",
        "Llama-3 unroped vs oracle": "s",
        "Llama-3 roped vs oracle": "^",
    }
    
    for comp_name, jaccards in comparisons.items():
        # Filter out NaN values
        valid_indices = [i for i, j in enumerate(jaccards) if not np.isnan(j)]
        valid_seq_k = [seq_len_ks[i] for i in valid_indices]
        valid_jaccards = [jaccards[i] for i in valid_indices]
        
        if valid_jaccards:
            ax.plot(
                valid_seq_k,
                valid_jaccards,
                marker=markers[comp_name],
                label=comp_name,
                color=colors[comp_name],
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )
    
    ax.set_xlabel("Context Size (seq_len_k)", fontsize=12)
    ax.set_ylabel("Jaccard Similarity vs Oracle", fontsize=12)
    ax.set_title("Mask Comparison: All Masks vs Llama-3.1 Roped (Oracle)\nLayer 15, Head 0, Sample 0", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label="Perfect Match")
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("figures_mask_comparison")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "mask_comparison_vs_oracle.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for comp_name, jaccards in comparisons.items():
        valid_jaccards = [j for j in jaccards if not np.isnan(j)]
        if valid_jaccards:
            mean_jaccard = np.mean(valid_jaccards)
            min_jaccard = np.min(valid_jaccards)
            max_jaccard = np.max(valid_jaccards)
            print(f"{comp_name}:")
            print(f"  Mean Jaccard: {mean_jaccard:.4f}")
            print(f"  Range: {min_jaccard:.4f} - {max_jaccard:.4f}")
            print(f"  Valid chunks: {len(valid_jaccards)}/{len(jaccards)}")
        else:
            print(f"{comp_name}: No valid comparisons")
        print()


if __name__ == "__main__":
    compare_all_against_oracle()

