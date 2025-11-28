#!/usr/bin/env bash
# Quick test: Llama-3 with 3 samples
set -euo pipefail

export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export NUM_SAMPLES=30
export PREFILL_CHUNK_SIZE=512

# Sparse attention parameters
export ORACLE_TOPK_HEAVY_SIZE=0.10
export SINK_SIZE=128
export LOCAL_WINDOW_SIZE=128

# Repositioning parameters - K=0, S=0 (full repositioning)
export ENABLE_POSITION_REASSIGNMENT=1
export NUM_SINK_TOKENS=0
export PREFIX_FREEZE_TAIL_K=0
export EXTEND_CONTEXT=1
export COMPARE_MASK_ROPED_VS_UNROPED=1
export REPEAT_COUNT=5

# Debug
export SPARSE_DEBUG=0

echo "=========================================="
echo "QUICK TEST: Llama-3 (3 samples)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Chunk size: $PREFILL_CHUNK_SIZE"
echo "Repositioning: K=0, S=0"
echo "Repeat count: $REPEAT_COUNT"
echo "=========================================="

cd "$(dirname "$0")/.."
python3 test_scripts/test_integration.py 2>&1 | tee test_log_temp.txt

# Extract output directory from Python output
OUTPUT_DIR=$(grep "OUTPUT_DIR_ACTUAL:" test_log_temp.txt | cut -d' ' -f2)
if [ -n "${OUTPUT_DIR}" ]; then
    mv test_log_temp.txt "${OUTPUT_DIR}/test_log.txt"
    echo ""
    echo "✅ Test completed! Results saved to: ${OUTPUT_DIR}"
else
    echo "⚠️  Could not find output directory in logs"
fi

