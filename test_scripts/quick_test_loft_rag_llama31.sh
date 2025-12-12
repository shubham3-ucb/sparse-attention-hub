#!/usr/bin/env bash
# Quick test: LOFT RAG HotpotQA 32k with Llama-3.1
set -euo pipefail

export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export NUM_SAMPLES=30
export PREFILL_CHUNK_SIZE=1024
export LOFT_DATASET="hotpotqa_32k"

# Sparse attention parameters
export ORACLE_TOPK_HEAVY_SIZE=0.10
export SINK_SIZE=128
export LOCAL_WINDOW_SIZE=256

# Repositioning parameters - K=0, S=0 (full repositioning)
export ENABLE_POSITION_REASSIGNMENT=1
export NUM_SINK_TOKENS=0
export PREFIX_FREEZE_TAIL_K=0
export EXTEND_CONTEXT=1
export COMPARE_MASK_ROPED_VS_UNROPED=1

# Mask saving (for comparison between models)
# export SAVE_MASKS=1

# Debug
export SPARSE_DEBUG=0

echo "=========================================="
echo "QUICK TEST: LOFT RAG HotpotQA 32k with Llama-3.1"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Chunk size: $PREFILL_CHUNK_SIZE"
echo "LOFT Dataset: $LOFT_DATASET"
echo "Repositioning: K=0, S=0 (8192 cap enforced)"
echo "=========================================="

cd "$(dirname "$0")/.."
python3 test_scripts/test_loft_rag.py 2>&1 | tee test_log_loft_llama31.txt

# Extract output directory from Python output
OUTPUT_DIR=$(grep "OUTPUT_DIR_ACTUAL:" test_log_loft_llama31.txt | cut -d' ' -f2)
if [ -n "${OUTPUT_DIR}" ]; then
    mv test_log_loft_llama31.txt "${OUTPUT_DIR}/test_log.txt"
    echo ""
    echo "✅ Test completed! Results saved to: ${OUTPUT_DIR}"
else
    echo "⚠️  Could not find output directory in logs"
fi

