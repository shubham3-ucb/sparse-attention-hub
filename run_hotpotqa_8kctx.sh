#!/usr/bin/env bash
set -euo pipefail

# HotpotQA with Llama-3.1-8B and 8192 context cap for repositioning
# Model's actual max context remains 128k, but repositioning is capped at 8192
# Based on: run_ruler16k_8kctx.sh pattern

# Config
REPO="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"  # Context repetition multiplier (1 = no repetition, 2 = 2x context)
ORACLE_TOPK_HEAVY_SIZE="${ORACLE_TOPK_HEAVY_SIZE:-0.10}"  # 10% sparsity (match original)
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2147483647}"  # Model's actual max context (no truncation)
CAP_B_MAX="8192"  # Repositioning cap (DEFAULT_MAX_POSITION_ID in mask_attention_utils.py)
# Two-band scaling: S (sink tokens) and K (tail freeze tokens) - default to 0 (current working state)
NUM_SINK_TOKENS="${NUM_SINK_TOKENS:-0}"  # Number of sink tokens to freeze (default: 0)
PREFIX_FREEZE_TAIL_K="${PREFIX_FREEZE_TAIL_K:-0}"  # Number of tail tokens to freeze (default: 0)

# Settings with repositioning enabled
ENABLE_POSITION_REASSIGNMENT=1
COMPARE_MASK_ROPED_VS_UNROPED=1
EXTEND_CONTEXT=1
SPARSE_DEBUG=0
SPARSE_DEBUG_POSITIONS=1  # Enable position debugging to verify 8k cap

# Output directory - match pattern: output_llama31_8b_hotpotqa_8kctx_hs010_pcs1024[_repeat2][_k128_s128]
model_short=$(echo "${MODEL_NAME}" | sed 's|meta-llama/||' | sed 's|Meta-||' | tr '[:upper:]' '[:lower:]' | sed 's|llama-3.1-8b-instruct|llama31_8b|' | sed 's|llama-3-8b-instruct|llama3_8b|')
hs_formatted=$(printf "%03d" $(awk "BEGIN {printf \"%d\", ${ORACLE_TOPK_HEAVY_SIZE} * 100}"))
repeat_suffix=""
if [ "${REPEAT_COUNT}" -gt 1 ]; then
    repeat_suffix="_repeat${REPEAT_COUNT}"
fi
# Add K and S values to directory name if non-zero
ks_suffix=""
if [ "${NUM_SINK_TOKENS}" -gt 0 ] || [ "${PREFIX_FREEZE_TAIL_K}" -gt 0 ]; then
    ks_suffix="_k${PREFIX_FREEZE_TAIL_K}_s${NUM_SINK_TOKENS}"
fi
OUTDIR="${REPO}/output_${model_short}_hotpotqa_8kctx_hs${hs_formatted}_pcs${PREFILL_CHUNK_SIZE}${repeat_suffix}${ks_suffix}"

# Activate conda env
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
fi
conda activate sparse || {
  echo "[FATAL] Could not activate conda env: sparse"
  exit 1
}

# Create output directory
mkdir -p "${OUTDIR}" || {
  echo "[FATAL] Failed to create output directory: ${OUTDIR}"
  exit 1
}
echo "[INFO] Created output directory: ${OUTDIR}"

# Validation checks
echo "[INFO] HotpotQA with Llama-3.1-8B and 8k context cap for repositioning"
echo "[INFO] ================================================="
echo "[INFO] Output: ${OUTDIR}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] NUM_SAMPLES: ${NUM_SAMPLES}"
echo "[INFO] REPEAT_COUNT: ${REPEAT_COUNT} (context repetition multiplier)"
echo "[INFO] ORACLE_TOPK_HEAVY_SIZE: ${ORACLE_TOPK_HEAVY_SIZE} (sparsity)"
echo "[INFO] PREFILL_CHUNK_SIZE: ${PREFILL_CHUNK_SIZE}"
echo "[INFO] CAP_B_MAX (repositioning cap): ${CAP_B_MAX}"
echo "[INFO] MAX_CONTEXT_LENGTH (model max): ${MAX_CONTEXT_LENGTH}"
echo "[INFO] NUM_SINK_TOKENS (S): ${NUM_SINK_TOKENS} (sink tokens to freeze)"
echo "[INFO] PREFIX_FREEZE_TAIL_K (K): ${PREFIX_FREEZE_TAIL_K} (tail tokens to freeze)"
echo "[INFO] Settings: ENABLE_POSITION_REASSIGNMENT=1, EXTEND_CONTEXT=1, SPARSE_DEBUG=0"
echo "[INFO] Position debugging: SPARSE_DEBUG_POSITIONS=1 (to verify 8k cap)"
echo "[INFO] ================================================="

# Validate NUM_SAMPLES
if ! [[ "${NUM_SAMPLES}" =~ ^[0-9]+$ ]] || [ "${NUM_SAMPLES}" -le 0 ]; then
  echo "[FATAL] NUM_SAMPLES must be a positive integer, got: ${NUM_SAMPLES}"
  exit 1
fi

# Validate REPEAT_COUNT
if ! [[ "${REPEAT_COUNT}" =~ ^[0-9]+$ ]] || [ "${REPEAT_COUNT}" -le 0 ]; then
  echo "[FATAL] REPEAT_COUNT must be a positive integer, got: ${REPEAT_COUNT}"
  exit 1
fi

# Validate ORACLE_TOPK_HEAVY_SIZE
if ! awk "BEGIN {exit !(${ORACLE_TOPK_HEAVY_SIZE} >= 0 && ${ORACLE_TOPK_HEAVY_SIZE} <= 1)}"; then
  echo "[FATAL] ORACLE_TOPK_HEAVY_SIZE must be between 0 and 1, got: ${ORACLE_TOPK_HEAVY_SIZE}"
  exit 1
fi

# Validate PREFILL_CHUNK_SIZE
if ! [[ "${PREFILL_CHUNK_SIZE}" =~ ^[0-9]+$ ]] || [ "${PREFILL_CHUNK_SIZE}" -le 0 ]; then
  echo "[FATAL] PREFILL_CHUNK_SIZE must be a positive integer, got: ${PREFILL_CHUNK_SIZE}"
  exit 1
fi

# Validate CAP_B_MAX is 8192
if [ "${CAP_B_MAX}" != "8192" ]; then
  echo "[FATAL] CAP_B_MAX must be 8192 for 8k context cap, got: ${CAP_B_MAX}"
  exit 1
fi

# Check that REPO exists
if [ ! -d "${REPO}" ]; then
  echo "[FATAL] Repository directory does not exist: ${REPO}"
  exit 1
fi

# Check that Python script exists
if [ ! -f "${REPO}/test_sparse_oracle.py" ]; then
  echo "[FATAL] Python script not found: ${REPO}/test_sparse_oracle.py"
  exit 1
fi

echo "[INFO] All validation checks passed ✓"
echo ""

pushd "${REPO}" >/dev/null

# Save settings
cat > "${OUTDIR}/settings.json" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "oracle_topk_heavy_size": ${ORACLE_TOPK_HEAVY_SIZE},
  "prefill_chunk_size": ${PREFILL_CHUNK_SIZE},
  "num_samples": ${NUM_SAMPLES},
  "repeat_count": ${REPEAT_COUNT},
  "max_context_length": ${MAX_CONTEXT_LENGTH},
  "cap_b_max": ${CAP_B_MAX},
  "enable_position_reassignment": ${ENABLE_POSITION_REASSIGNMENT},
  "extend_context": ${EXTEND_CONTEXT},
  "compare_mask_roped_vs_unroped": ${COMPARE_MASK_ROPED_VS_UNROPED},
  "sparse_debug": ${SPARSE_DEBUG},
  "sparse_debug_positions": ${SPARSE_DEBUG_POSITIONS},
  "mode": "hotpotqa_8k_context_cap"
}
EOF

# Save command (MAX_POSITION_ID not needed - DEFAULT_MAX_POSITION_ID=8192 is hardcoded)
RUN_CMD="OUT=\"${OUTDIR}\" MODEL_NAME=\"${MODEL_NAME}\" PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE} EXTEND_CONTEXT=${EXTEND_CONTEXT} ENABLE_POSITION_REASSIGNMENT=${ENABLE_POSITION_REASSIGNMENT} COMPARE_MASK_ROPED_VS_UNROPED=${COMPARE_MASK_ROPED_VS_UNROPED} SPARSE_DEBUG=${SPARSE_DEBUG} SPARSE_DEBUG_POSITIONS=${SPARSE_DEBUG_POSITIONS} MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH} OUTPUT_DIR=\"${OUTDIR}\" SPARSE_LOG_PATH=\"${OUTDIR}/hf_prefill.log\" ORACLE_TOPK_HEAVY_SIZE=${ORACLE_TOPK_HEAVY_SIZE} NUM_SAMPLES=${NUM_SAMPLES} REPEAT_COUNT=${REPEAT_COUNT} NUM_SINK_TOKENS=${NUM_SINK_TOKENS} PREFIX_FREEZE_TAIL_K=${PREFIX_FREEZE_TAIL_K} python test_sparse_oracle.py > \"${OUTDIR}/log.txt\" 2>&1"
echo "${RUN_CMD}" > "${OUTDIR}/command.txt"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd \"${REPO}\""
  echo "${RUN_CMD}"
} > "${OUTDIR}/run_command.sh"
chmod +x "${OUTDIR}/run_command.sh"

echo "[RUN] Starting HotpotQA with Llama-3.1-8B and 8k context cap..."
if [ "${REPEAT_COUNT}" -gt 1 ]; then
  echo "[INFO] Context repetition: ${REPEAT_COUNT}x (context will be ${REPEAT_COUNT}x longer)"
fi
echo "[INFO] Repositioning will be capped at ${CAP_B_MAX} (DEFAULT_MAX_POSITION_ID=8192 hardcoded)"
echo "[INFO] Position debugging enabled (SPARSE_DEBUG_POSITIONS=1) to verify 8k cap"

# Run (MAX_POSITION_ID not needed - DEFAULT_MAX_POSITION_ID=8192 is hardcoded in mask_attention_utils.py)
OUT="${OUTDIR}" \
MODEL_NAME="${MODEL_NAME}" \
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE}" \
EXTEND_CONTEXT="${EXTEND_CONTEXT}" \
ENABLE_POSITION_REASSIGNMENT="${ENABLE_POSITION_REASSIGNMENT}" \
COMPARE_MASK_ROPED_VS_UNROPED="${COMPARE_MASK_ROPED_VS_UNROPED}" \
SPARSE_DEBUG="${SPARSE_DEBUG}" \
SPARSE_DEBUG_POSITIONS="${SPARSE_DEBUG_POSITIONS}" \
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH}" \
OUTPUT_DIR="${OUTDIR}" \
SPARSE_LOG_PATH="${OUTDIR}/hf_prefill.log" \
ORACLE_TOPK_HEAVY_SIZE="${ORACLE_TOPK_HEAVY_SIZE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
REPEAT_COUNT="${REPEAT_COUNT}" \
NUM_SINK_TOKENS="${NUM_SINK_TOKENS}" \
PREFIX_FREEZE_TAIL_K="${PREFIX_FREEZE_TAIL_K}" \
python test_sparse_oracle.py > "${OUTDIR}/log.txt" 2>&1

exit_code=$?

if [ ${exit_code} -eq 0 ]; then
  echo "[OK] Completed successfully → ${OUTDIR}"
  
  # Verify output files exist
  echo "[INFO] Verifying output files..."
  missing_files=0
  for file in "raw_results.csv" "metrics.json" "test_sparse_results.json" "config.json"; do
    if [ -f "${OUTDIR}/${file}" ]; then
      echo "  ✓ ${file}"
    else
      echo "  ✗ ${file} (missing)"
      missing_files=$((missing_files + 1))
    fi
  done
  
  if [ -d "${OUTDIR}/comparison_results" ]; then
    echo "  ✓ comparison_results/ directory"
  else
    echo "  ✗ comparison_results/ directory (missing)"
    missing_files=$((missing_files + 1))
  fi
  
  if [ ${missing_files} -eq 0 ]; then
    echo "[INFO] All expected output files present ✓"
  else
    echo "[WARNING] ${missing_files} expected file(s) missing"
  fi
  
  # Verify positions are within 8k cap
  echo ""
  echo "[VERIFY] Checking position IDs are within 8k cap..."
  if grep -q "DEBUG POSITIONS" "${OUTDIR}/log.txt" 2>/dev/null; then
    max_pos=$(grep "DEBUG POSITIONS" "${OUTDIR}/log.txt" | grep -o "Q range: \[.*\]" | grep -o "[0-9]*" | sort -n | tail -1)
    if [ -n "${max_pos}" ] && [ "${max_pos}" -le 8192 ]; then
      echo "  ✅ Max position ID: ${max_pos} (within 8k cap)"
    elif [ -n "${max_pos}" ]; then
      echo "  ⚠️  WARNING: Max position ID: ${max_pos} (exceeds 8k cap!)"
    else
      echo "  ⚠️  Could not extract max position from log"
    fi
  else
    echo "  ⚠️  No position debug output found in log"
  fi
  
  echo ""
  echo "[INFO] Results:"
  echo "  - CSV: ${OUTDIR}/raw_results.csv"
  echo "  - Metrics: ${OUTDIR}/metrics.json"
  echo "  - Comparison: ${OUTDIR}/comparison_results/"
  echo "  - Log: ${OUTDIR}/log.txt"
  echo "[INFO] Note: Repositioning is capped at ${CAP_B_MAX} (DEFAULT_MAX_POSITION_ID)"
else
  echo "[FATAL] Process exited with non-zero status (${exit_code})"
  echo "[INFO] Check logs: ${OUTDIR}/log.txt"
  tail -50 "${OUTDIR}/log.txt" || echo "Could not read log file"
  exit ${exit_code}
fi

popd >/dev/null

echo "[DONE] HotpotQA with Llama-3.1-8B and 8k context cap completed → ${OUTDIR}"

