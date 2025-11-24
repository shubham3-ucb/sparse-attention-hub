#!/usr/bin/env bash
set -euo pipefail

# Test repositioning with K=0, S=0 for 2 samples
# This verifies exact same output as working codebase

export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export NUM_SAMPLES="${NUM_SAMPLES:-2}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-1024}"

# Sparse attention parameters
export ORACLE_TOPK_HEAVY_SIZE="${ORACLE_TOPK_HEAVY_SIZE:-0.10}"
export SINK_SIZE="${SINK_SIZE:-128}"
export LOCAL_WINDOW_SIZE="${LOCAL_WINDOW_SIZE:-128}"

# Repositioning parameters - K=0, S=0 (baseline)
export ENABLE_POSITION_REASSIGNMENT="${ENABLE_POSITION_REASSIGNMENT:-1}"
export NUM_SINK_TOKENS="${NUM_SINK_TOKENS:-0}"
export PREFIX_FREEZE_TAIL_K="${PREFIX_FREEZE_TAIL_K:-0}"
export EXTEND_CONTEXT="${EXTEND_CONTEXT:-1}"

# Context repetition (for doubling context - REPEAT_COUNT=2 means 2x context)
export REPEAT_COUNT="${REPEAT_COUNT:-1}"

# Optional: Debug flags
export SPARSE_DEBUG="${SPARSE_DEBUG:-0}"
export SPARSE_DEBUG_POSITIONS="${SPARSE_DEBUG_POSITIONS:-0}"
export COMPARE_MASK_ROPED_VS_UNROPED="${COMPARE_MASK_ROPED_VS_UNROPED:-1}"

echo "=========================================="
echo "TEST: Repositioning with K=0, S=0"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Chunk size: $PREFILL_CHUNK_SIZE"
echo "Repositioning: ENABLE_POSITION_REASSIGNMENT=$ENABLE_POSITION_REASSIGNMENT"
echo "  NUM_SINK_TOKENS (S)=$NUM_SINK_TOKENS"
echo "  PREFIX_FREEZE_TAIL_K (K)=$PREFIX_FREEZE_TAIL_K"
echo "EXTEND_CONTEXT=$EXTEND_CONTEXT"
if [ "${REPEAT_COUNT}" -gt 1 ]; then
    echo "REPEAT_COUNT=$REPEAT_COUNT (context will be ${REPEAT_COUNT}x longer)"
fi
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
