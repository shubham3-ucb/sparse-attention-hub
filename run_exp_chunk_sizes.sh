#!/usr/bin/env bash
set -euo pipefail

# Config
REPO="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SAMPLES="40"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-2147483647}"
SIZES=("256" "1024" "4096")
ROOT_OUT="${REPO}/exp_sweep_chunk_sizes"

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
  local pcs="$1"
  local outdir="${ROOT_OUT}/pcs${pcs}_ns${NUM_SAMPLES}_max${MAX_CONTEXT_LENGTH}"
  mkdir -p "${outdir}"
  echo "[RUN] prefill_chunk_size=${pcs} → ${outdir}"

  pushd "${REPO}" >/dev/null
  # Persist run settings
  cat > "${outdir}/settings.json" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "prefill_chunk_size": ${pcs},
  "num_samples": ${NUM_SAMPLES},
  "max_context_length": ${MAX_CONTEXT_LENGTH},
  "enable_position_reassignment": true,
  "extend_context": true,
  "compare_mask_roped_vs_unroped": true,
  "sparse_debug": true
}
EOF
  # Persist exact run command (keep OracleTopK at default via test script)
  RUN_CMD="OUT=\"${outdir}\" MODEL_NAME=\"${MODEL_NAME}\" PREFILL_CHUNK_SIZE=${pcs} EXTEND_CONTEXT=1 ENABLE_POSITION_REASSIGNMENT=1 COMPARE_MASK_ROPED_VS_UNROPED=1 SPARSE_DEBUG=1 MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH} OUTPUT_DIR=\"${outdir}\" SPARSE_LOG_PATH=\"${outdir}/hf_prefill.log\" NUM_SAMPLES=${NUM_SAMPLES} python test_sparse_oracle.py > \"${outdir}/log.txt\" 2>&1"
  echo "${RUN_CMD}" > "${outdir}/command.txt"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd \"${REPO}\""
    echo "${RUN_CMD}"
  } > "${outdir}/run_command.sh"
  chmod +x "${outdir}/run_command.sh"

  # Start run and wait
  OUT="${outdir}" \
  MODEL_NAME="${MODEL_NAME}" \
  PREFILL_CHUNK_SIZE="${pcs}" \
  EXTEND_CONTEXT=1 \
  ENABLE_POSITION_REASSIGNMENT=1 \
  COMPARE_MASK_ROPED_VS_UNROPED=1 \
  SPARSE_DEBUG=1 \
  MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH}" \
  OUTPUT_DIR="${outdir}" \
  SPARSE_LOG_PATH="${outdir}/hf_prefill.log" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  python test_sparse_oracle.py > "${outdir}/log.txt" 2>&1 &

  local pid=$!
  echo "[INFO] PID=${pid} — running..."
  if ! wait "${pid}"; then
    echo "[FATAL] Process exited with non-zero status for pcs=${pcs}"; exit 1
  fi
  echo "[OK] Completed pcs=${pcs} → ${outdir}"
  popd >/dev/null
}

for pcs in "${SIZES[@]}"; do
  run_one "${pcs}"
done

echo "[DONE] All runs completed successfully → ${ROOT_OUT}"


