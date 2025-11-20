#!/usr/bin/env bash
set -euo pipefail
cd "/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub"
OUT="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub/output_llama31_8b_ruler16k_8kctx_hs050_pcs1024" MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" PREFILL_CHUNK_SIZE=1024 EXTEND_CONTEXT=1 ENABLE_POSITION_REASSIGNMENT=1 COMPARE_MASK_ROPED_VS_UNROPED=1 SPARSE_DEBUG=0 MAX_CONTEXT_LENGTH=2147483647 OUTPUT_DIR="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub/output_llama31_8b_ruler16k_8kctx_hs050_pcs1024" SPARSE_LOG_PATH="/home/nvidia/shubham/sparse/new_sparse/sparse-attention-hub/output_llama31_8b_ruler16k_8kctx_hs050_pcs1024/hf_prefill.log" ORACLE_TOPK_HEAVY_SIZE=0.5 NUM_SAMPLES=40 RULER_TASKS="" python test_sparse_oracle_ruler16k.py > /dev/null 2>&1
