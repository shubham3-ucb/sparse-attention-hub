#!/usr/bin/env bash
# Standalone script to compute LOFT metrics from responses.jsonl or all_results.csv
# Usage: ./compute_loft_metrics.sh <output_dir>
# Example: ./compute_loft_metrics.sh test_outputs/test_loft_hotpotqa_32k_llama3_8b_ns30_pcs1024_repo_k0_s0_extctx

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir>"
    echo "  output_dir: Directory containing responses.jsonl or all_results.csv"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory not found: $OUTPUT_DIR"
    exit 1
fi

cd "$(dirname "$0")/.."

python3 test_scripts/compute_loft_metrics.py "$OUTPUT_DIR"
