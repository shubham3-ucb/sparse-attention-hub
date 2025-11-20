#!/usr/bin/env bash
set -euo pipefail

# RULER 16k with Llama-3.1-8B and 8192 context cap for repositioning
# Model's actual max context remains 128k, but repositioning is capped at 8192
# Based on: output_llama31_8b_128k_longctx_hs050_pcs1024

# Config
REPO="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
ORACLE_TOPK_HEAVY_SIZE="${ORACLE_TOPK_HEAVY_SIZE:-0.5}"  # 50% sparsity
RULER_TASKS="${RULER_TASKS:-}"  # Empty = all 13 tasks
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2147483647}"  # Model's actual max context (no truncation)
CAP_B_MAX="8192"  # Repositioning cap (DEFAULT_MAX_POSITION_ID in mask_attention_utils.py)

# Settings with repositioning enabled
ENABLE_POSITION_REASSIGNMENT=1
COMPARE_MASK_ROPED_VS_UNROPED=1
EXTEND_CONTEXT=1
SPARSE_DEBUG=0

# Output directory - match pattern: output_llama31_8b_ruler16k_8kctx_hs050_pcs1024
model_short=$(echo "${MODEL_NAME}" | sed 's|meta-llama/||' | sed 's|Meta-||' | tr '[:upper:]' '[:lower:]' | sed 's|llama-3.1-8b-instruct|llama31_8b|' | sed 's|llama-3.2-1b-instruct|llama32_1b|')
hs_formatted=$(printf "%03d" $(awk "BEGIN {printf \"%d\", ${ORACLE_TOPK_HEAVY_SIZE} * 100}"))
OUTDIR="${REPO}/output_${model_short}_ruler16k_8kctx_hs${hs_formatted}_pcs${PREFILL_CHUNK_SIZE}"

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
echo "[INFO] RULER 16k with Llama-3.1-8B and 8k context cap for repositioning"
echo "[INFO] ================================================="
echo "[INFO] Output: ${OUTDIR}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] NUM_SAMPLES: ${NUM_SAMPLES} (first ${NUM_SAMPLES} samples total across all tasks)"
echo "[INFO] ORACLE_TOPK_HEAVY_SIZE: ${ORACLE_TOPK_HEAVY_SIZE} (sparsity)"
echo "[INFO] PREFILL_CHUNK_SIZE: ${PREFILL_CHUNK_SIZE}"
echo "[INFO] RULER_TASKS: ${RULER_TASKS:-all 13 tasks}"
echo "[INFO] CAP_B_MAX (repositioning cap): ${CAP_B_MAX}"
echo "[INFO] MAX_CONTEXT_LENGTH (model max): ${MAX_CONTEXT_LENGTH}"
echo "[INFO] Settings: ENABLE_POSITION_REASSIGNMENT=1, EXTEND_CONTEXT=1, SPARSE_DEBUG=0"
echo "[INFO] ================================================="

# Validate NUM_SAMPLES
if ! [[ "${NUM_SAMPLES}" =~ ^[0-9]+$ ]] || [ "${NUM_SAMPLES}" -le 0 ]; then
  echo "[FATAL] NUM_SAMPLES must be a positive integer, got: ${NUM_SAMPLES}"
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

# Check that REPO exists
if [ ! -d "${REPO}" ]; then
  echo "[FATAL] Repository directory does not exist: ${REPO}"
  exit 1
fi

# Check that Python script exists
if [ ! -f "${REPO}/test_sparse_oracle_ruler16k.py" ]; then
  echo "[FATAL] Python script not found: ${REPO}/test_sparse_oracle_ruler16k.py"
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
  "max_context_length": ${MAX_CONTEXT_LENGTH},
  "ruler_tasks": "${RULER_TASKS:-all}",
  "cap_b_max": ${CAP_B_MAX},
  "enable_position_reassignment": ${ENABLE_POSITION_REASSIGNMENT},
  "extend_context": ${EXTEND_CONTEXT},
  "compare_mask_roped_vs_unroped": ${COMPARE_MASK_ROPED_VS_UNROPED},
  "sparse_debug": ${SPARSE_DEBUG},
  "mode": "ruler16k_8k_context_cap"
}
EOF

# Save command
RUN_CMD="OUT=\"${OUTDIR}\" MODEL_NAME=\"${MODEL_NAME}\" PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE} EXTEND_CONTEXT=${EXTEND_CONTEXT} ENABLE_POSITION_REASSIGNMENT=${ENABLE_POSITION_REASSIGNMENT} COMPARE_MASK_ROPED_VS_UNROPED=${COMPARE_MASK_ROPED_VS_UNROPED} SPARSE_DEBUG=${SPARSE_DEBUG} MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH} OUTPUT_DIR=\"${OUTDIR}\" SPARSE_LOG_PATH=\"${OUTDIR}/hf_prefill.log\" ORACLE_TOPK_HEAVY_SIZE=${ORACLE_TOPK_HEAVY_SIZE} NUM_SAMPLES=${NUM_SAMPLES} RULER_TASKS=\"${RULER_TASKS}\" python test_sparse_oracle_ruler16k.py > /dev/null 2>&1"
echo "${RUN_CMD}" > "${OUTDIR}/command.txt"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd \"${REPO}\""
  echo "${RUN_CMD}"
} > "${OUTDIR}/run_command.sh"
chmod +x "${OUTDIR}/run_command.sh"

echo "[RUN] Starting RULER 16k with Llama-3.1-8B and 8k context cap..."

# Run
OUT="${OUTDIR}" \
MODEL_NAME="${MODEL_NAME}" \
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE}" \
EXTEND_CONTEXT="${EXTEND_CONTEXT}" \
ENABLE_POSITION_REASSIGNMENT="${ENABLE_POSITION_REASSIGNMENT}" \
COMPARE_MASK_ROPED_VS_UNROPED="${COMPARE_MASK_ROPED_VS_UNROPED}" \
SPARSE_DEBUG="${SPARSE_DEBUG}" \
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH}" \
OUTPUT_DIR="${OUTDIR}" \
SPARSE_LOG_PATH="${OUTDIR}/hf_prefill.log" \
ORACLE_TOPK_HEAVY_SIZE="${ORACLE_TOPK_HEAVY_SIZE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
RULER_TASKS="${RULER_TASKS}" \
python test_sparse_oracle_ruler16k.py > /dev/null 2>&1

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
  
  echo "[INFO] Results:"
  echo "  - CSV: ${OUTDIR}/raw_results.csv"
  echo "  - Metrics: ${OUTDIR}/metrics.json"
  echo "  - Comparison: ${OUTDIR}/comparison_results/"
  echo "[INFO] Note: Repositioning is capped at ${CAP_B_MAX} (DEFAULT_MAX_POSITION_ID)"
else
  echo "[FATAL] Process exited with non-zero status (${exit_code})"
  echo "[INFO] Check logs: ${OUTDIR}/hf_prefill.log (log.txt disabled to reduce buffering)"
  exit ${exit_code}
fi

popd >/dev/null

echo "[DONE] RULER 16k with Llama-3.1-8B and 8k context cap completed → ${OUTDIR}"

