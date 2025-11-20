#!/usr/bin/env bash
set -euo pipefail

# RULER 16k sweep - EXACT same settings as HotpotQA (run_exp_1.sh)
# Only difference: uses test_sparse_oracle_ruler16k.py instead of test_sparse_oracle.py

# Config
REPO="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2147483647}"
CAP_B_MAX="8192"  # informational; enforced inside code
SWEEP=("0.10" "0.50" "0.90")
RULER_TASKS="${RULER_TASKS:-}"  # Empty = all 13 tasks
ROOT_OUT="${REPO}/exp_sweep_filter_ratios_ruler16k"

# Activate conda env
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
fi
conda activate sparse || {
  echo "[FATAL] Could not activate conda env: sparse"
  exit 1
}

mkdir -p "${ROOT_OUT}"
echo "[INFO] Writing results to: ${ROOT_OUT}"

run_one() {
  local heavy_size="$1"
  local hs_fmt
  hs_fmt="$(printf '%0.2f' "${heavy_size}")"
  local outdir="${ROOT_OUT}/sparsity_${hs_fmt}_pcs${PREFILL_CHUNK_SIZE}_ns${NUM_SAMPLES}_max${MAX_CONTEXT_LENGTH}"
  mkdir -p "${outdir}"
  echo "[RUN] heavy_size=${heavy_size} → ${outdir}"

  pushd "${REPO}" >/dev/null
  # Persist run settings for quick inspection
  cat > "${outdir}/settings.json" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "oracle_topk_heavy_size": ${heavy_size},
  "prefill_chunk_size": ${PREFILL_CHUNK_SIZE},
  "num_samples": ${NUM_SAMPLES},
  "max_context_length": ${MAX_CONTEXT_LENGTH},
  "ruler_tasks": "${RULER_TASKS:-all}",
  "enable_position_reassignment": true,
  "extend_context": true,
  "compare_mask_roped_vs_unroped": true,
  "sparse_debug": false,
  "cap_b_max": ${CAP_B_MAX}
}
EOF
  # Persist exact run command
  RUN_CMD="OUT=\"${outdir}\" MODEL_NAME=\"${MODEL_NAME}\" PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE} EXTEND_CONTEXT=1 ENABLE_POSITION_REASSIGNMENT=1 COMPARE_MASK_ROPED_VS_UNROPED=1 SPARSE_DEBUG=0 MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH} OUTPUT_DIR=\"${outdir}\" SPARSE_LOG_PATH=\"${outdir}/hf_prefill.log\" ORACLE_TOPK_HEAVY_SIZE=${heavy_size} NUM_SAMPLES=${NUM_SAMPLES} RULER_TASKS=\"${RULER_TASKS}\" python test_sparse_oracle_ruler16k.py > /dev/null 2>&1"
  echo "${RUN_CMD}" > "${outdir}/command.txt"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd \"${REPO}\""
    echo "${RUN_CMD}"
  } > "${outdir}/run_command.sh"
  chmod +x "${outdir}/run_command.sh"

  # Start run in background
  OUT="${outdir}" \
  MODEL_NAME="${MODEL_NAME}" \
  PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE}" \
  EXTEND_CONTEXT=1 \
  ENABLE_POSITION_REASSIGNMENT=1 \
  COMPARE_MASK_ROPED_VS_UNROPED=1 \
  SPARSE_DEBUG=0 \
  MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH}" \
  OUTPUT_DIR="${outdir}" \
  SPARSE_LOG_PATH="${outdir}/hf_prefill.log" \
  ORACLE_TOPK_HEAVY_SIZE="${heavy_size}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  RULER_TASKS="${RULER_TASKS}" \
  python test_sparse_oracle_ruler16k.py > /dev/null 2>&1 &

  local pid=$!
  echo "[INFO] PID=${pid} — running..."
  if ! wait "${pid}"; then
    echo "[FATAL] Process exited with non-zero status for heavy_size=${heavy_size}"; exit 1
  fi
  echo "[OK] Completed heavy_size=${heavy_size} → ${outdir}"
  popd >/dev/null
}

for s in "${SWEEP[@]}"; do
  run_one "${s}"
done

echo "[DONE] All runs completed successfully → ${ROOT_OUT}"

