# Chunked Prefill and Sparse Repositioning Implementation

## Quick Reference

**What this does**: Enables processing of very long contexts (32k+ tokens) in models with limited windows (8k tokens) using:
1. **Chunked Prefill**: Process long contexts in chunks (e.g., 1024 tokens)
2. **Sparse Repositioning**: Compress middle band while preserving critical start/end tokens

**Key Files**:
- `adapters/huggingface.py` - Chunked prefill implementation
- `sparse_attention/utils/repositioning_utils.py` - Repositioning logic
- `sparse_attention/research_attention/rope_utils.py` - RoPE utilities
- `test_scripts/test_integration.py` - Integration tests

**Quick Start**:
```bash
export HF_TOKEN="your_token"
./test_scripts/test_repositioning.sh
```

**Quick Test Commands**:
```bash
# Llama-3.1 with 3 samples
./test_scripts/quick_test_llama31.sh

# Llama-3 with 3 samples
./test_scripts/quick_test_llama3.sh
```

---

## Overview

This document describes the implementation of **chunked prefill** and **sparse repositioning** features for handling long-context sequences in transformer models.

---

## The Idea (In Plain English)

### Problem
Transformer models have limited context windows (e.g., 8k tokens). When processing very long documents (e.g., 32k+ tokens), we need to:
1. **Manage memory**: Process long sequences without running out of GPU memory
2. **Preserve important information**: Keep critical tokens accessible even when compressing the context
3. **Maintain accuracy**: Ensure the model can still answer questions correctly despite compression

### Solution: Two Complementary Techniques

#### 1. Chunked Prefill
Instead of processing the entire long context in one pass, we split it into smaller chunks (e.g., 1024 tokens each) and process them sequentially. This allows us to handle contexts much longer than the model's native window size while staying within memory limits.

**How it works:**
- Split long context into chunks of size `PREFILL_CHUNK_SIZE` (default: 1024)
- Process each chunk sequentially through the model
- Accumulate hidden states from all chunks
- Use accumulated states for final generation

#### 2. Sparse Repositioning (Two-Band Scaling)
When the context exceeds the model's position window (e.g., 8k tokens), we compress the middle portion while preserving critical tokens at the beginning and end.

**How it works:**
- **Sink tokens (S)**: First `NUM_SINK_TOKENS` tokens are frozen at their original positions (always preserved)
- **Middle band**: Tokens between sink and tail are compressed proportionally to fit within the window
- **Tail tokens (K)**: Last `PREFIX_FREEZE_TAIL_K` tokens are frozen at their original positions (always preserved)
- **Pack translation**: Optionally moves tail K tokens and current chunk together as a single block, preserving relative distances

**Key insight**: We use **unroped queries/keys** (position-agnostic) to compute attention masks, then apply repositioning, then re-apply rotary position embeddings (RoPE) with new position IDs. This allows us to manipulate positions while maintaining attention quality.

---

## Code Changes Summary

### New Files Created

1. **`sparse_attention_hub/sparse_attention/research_attention/rope_utils.py`**
   - **Purpose**: Rotary Position Embedding (RoPE) utilities
   - **Key functions**:
     - `unapply_rotary_pos_emb_queries()`: Remove RoPE from queries
     - `unapply_rotary_pos_emb_keys()`: Remove RoPE from keys
     - `compute_rope_cos_sin()`: Compute RoPE embeddings for new positions
   - **Why**: Needed for repositioning (unrope → reposition → rerope)

2. **`sparse_attention_hub/sparse_attention/utils/repositioning_utils.py`**
   - **Purpose**: Modular repositioning logic
   - **Key function**: `_apply_position_reassignment()`
   - **Why**: Extracted from `mask_attention_utils.py` for clean separation

3. **`test_scripts/test_integration.py`**
   - **Purpose**: Comprehensive integration test suite
   - **Tests**: Chunked prefill, repositioning, micro metrics
   - **Output**: Results, metrics, comparison summaries in `test_outputs/`

4. **`test_scripts/test_repositioning.sh`**
   - **Purpose**: Shell script to run repositioning tests
   - **Usage**: `./test_scripts/test_repositioning.sh`

### Modified Files

1. **`sparse_attention_hub/adapters/huggingface.py`**
   - **Changes**:
     - Added `_chunked_prefill_dense()` method: Chunked prefill for dense attention
     - Added `_chunked_prefill_sparse()` method: Chunked prefill for sparse attention
     - Modified `process_request()`: Routes to chunked prefill when `PREFILL_CHUNK_SIZE` is set
     - Added `rotary_emb` storage in `sparse_meta_data` when `EXTEND_CONTEXT=1`
   - **Why**: Implements chunked prefill and passes rotary embeddings for repositioning

2. **`sparse_attention_hub/sparse_attention/research_attention/base.py`**
   - **Changes**:
     - Added `pack_k_chunk_translation` field to `ResearchAttentionConfig`
     - Added `_prepare_unroped_qk_for_mask()` method: Prepares unroped Q/K for mask computation
     - Modified `custom_attention()`: Uses unroped Q/K for maskers, enables repositioning
     - Added micro metric registrations: `research_attention_weight_diff`, `research_mask_roped_vs_unroped`
   - **Why**: Enables unroped mask computation and micro metric logging

3. **`sparse_attention_hub/sparse_attention/utils/mask_attention_utils.py`**
   - **Changes**:
     - Imports `_apply_position_reassignment` from `repositioning_utils.py`
     - Calls `_apply_position_reassignment()` when `ENABLE_POSITION_REASSIGNMENT=1`
     - Re-exports constants for backward compatibility
   - **Why**: Delegates repositioning logic to modular function

### File Structure

```
sparse_attention_hub/
├── adapters/
│   └── huggingface.py                    # Modified: Chunked prefill methods
├── sparse_attention/
│   ├── research_attention/
│   │   ├── base.py                       # Modified: Unroped Q/K, micro metrics
│   │   └── rope_utils.py                 # NEW: RoPE utilities
│   └── utils/
│       ├── mask_attention_utils.py       # Modified: Calls repositioning
│       └── repositioning_utils.py        # NEW: Repositioning logic
test_scripts/
├── test_integration.py                    # NEW: Integration tests
└── test_repositioning.sh                  # NEW: Test runner script
test_outputs/                              # NEW: All test outputs go here
docs/refactoring/                           # NEW: Documentation
```

---

## Environment Variables

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model name |
| `NUM_SAMPLES` | `5` | Number of test samples to run |
| `PREFILL_CHUNK_SIZE` | `1024` | Chunk size for chunked prefill (set to enable) |

### Sparse Attention Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_TOPK_HEAVY_SIZE` | `0.1` | Fraction of tokens to keep (OracleTopK masker) |
| `SINK_SIZE` | `128` | Number of sink tokens (always attended) |
| `LOCAL_WINDOW_SIZE` | `128` | Local attention window size |

### Repositioning Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_POSITION_REASSIGNMENT` | `0` | Enable repositioning (`1` = enabled) |
| `NUM_SINK_TOKENS` (S) | `0` | Number of sink tokens to freeze at start |
| `PREFIX_FREEZE_TAIL_K` (K) | `0` | Number of tail tokens to freeze at end |
| `EXTEND_CONTEXT` | `0` | Use unroped Q/K for mask computation (`1` = enabled) |
| `COMPARE_MASK_ROPED_VS_UNROPED` | `0` | Log mask comparison metric (`1` = enabled) |

### Context Repetition

| Variable | Default | Description |
|----------|---------|-------------|
| `REPEAT_COUNT` | `1` | Multiply context length (2 = double context) |

### Debug & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARSE_DEBUG` | `0` | Enable sparse attention debug logging |
| `SPARSE_DEBUG_POSITIONS` | `0` | Debug position reassignment |
| `SPARSE_LOG_PATH` | Auto | Path for HuggingFace prefill logs |
| `OUTPUT_DIR` | Auto | Output directory (auto-generated in `test_outputs/`) |
| `VERIFY_REFACTOR` | `0` | Tag output dir with `_verify_refactor` suffix |

### HuggingFace Token (Required)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` or `HF_HUB_TOKEN` | HuggingFace token for model access (required) |

---

## Quick Start Guide

### Prerequisites

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd sparse-attention-hub
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set HuggingFace token**
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   # OR
   export HF_HUB_TOKEN="your_huggingface_token_here"
   ```

### Running Tests

#### Basic Test (Chunked Prefill Only)
```bash
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export NUM_SAMPLES=2
export PREFILL_CHUNK_SIZE=1024
python test_scripts/test_integration.py
```

#### Full Repositioning Test (K=0, S=0)
```bash
./test_scripts/test_repositioning.sh
```

#### Custom Configuration
```bash
export MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
export NUM_SAMPLES=2
export PREFILL_CHUNK_SIZE=1024
export ENABLE_POSITION_REASSIGNMENT=1
export NUM_SINK_TOKENS=0
export PREFIX_FREEZE_TAIL_K=0
export EXTEND_CONTEXT=1
export COMPARE_MASK_ROPED_VS_UNROPED=1
python test_scripts/test_integration.py
```

#### With Context Repetition (2x context)
```bash
export REPEAT_COUNT=2
./test_scripts/test_repositioning.sh
```

### Output Location

All test outputs are automatically saved to:
```
test_outputs/test_integration_phase2_<model>_ns<num_samples>_pcs<chunk_size>_repo_k<K>_s<S>_extctx[_repeat<N>][_verify_refactor]/
```

Example:
```
test_outputs/test_integration_phase2_llama31_8b_ns2_pcs1024_repo_k0_s0_extctx/
├── results_sparse_chunked.json      # Test results
├── metrics.json                      # Task metrics
├── micro_metrics.jsonl              # Micro-level metrics (if enabled)
├── settings.json                     # Full configuration
├── comparison_summary.json          # Scenario comparison
└── test_log.txt                      # Full test log
```

---

## Key Implementation Details

### Chunked Prefill Flow

1. **Input**: Long context (e.g., 32k tokens)
2. **Split**: Divide into chunks of `PREFILL_CHUNK_SIZE` (e.g., 1024 tokens)
3. **Process**: For each chunk:
   - Run forward pass through model
   - Accumulate hidden states
   - Update KV cache
4. **Generate**: Use accumulated states for final generation

### Repositioning Flow

1. **Unrope**: Remove RoPE from queries and keys (position-agnostic)
2. **Compute Masks**: Use unroped Q/K to compute attention masks
3. **Reposition**: Apply two-band scaling:
   - Freeze sink tokens (S) at start
   - Compress middle band proportionally
   - Freeze tail tokens (K) at end
4. **Rerope**: Re-apply RoPE with new position IDs
5. **Compute Attention**: Use reroped Q/K for final attention computation

### Micro Metrics

When enabled, logs detailed metrics for analysis:
- **`research_attention_weight_diff`**: Difference between roped and reroped attention weights
- **`research_mask_roped_vs_unroped`**: Comparison of masks computed with roped vs unroped Q/K

Metrics are logged only for:
- Sparse attention runs (not dense)
- Layer 15 (configurable via `sample_layers` in `base.py`)

---

## Verification

The implementation has been verified to produce **identical results** to the pre-refactoring codebase:
- ✅ Llama-3.1: All responses, metrics, micro metrics match exactly
- ✅ Llama-3: First 2 samples match exactly (tested with 2 vs 10 samples)
- ✅ Zero errors in test logs
- ✅ All code changes are modular and backward compatible

---

## Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: No module named 'ray'`**
   - **Solution**: Install missing dependencies: `pip install -r requirements.txt`

2. **`HF_TOKEN not set`**
   - **Solution**: Export HuggingFace token: `export HF_TOKEN="your_token"`

3. **CUDA out of memory**
   - **Solution**: Reduce `PREFILL_CHUNK_SIZE` (e.g., 512 instead of 1024)

4. **Test outputs not in `test_outputs/`**
   - **Solution**: Ensure you're running the latest code (outputs auto-create `test_outputs/`)

### Debug Mode

Enable detailed logging:
```bash
export SPARSE_DEBUG=1
export SPARSE_DEBUG_POSITIONS=1
python test_scripts/test_integration.py
```

---

## References

- **Chunked Prefill**: Processes long contexts in manageable chunks
- **Two-Band Scaling**: Compresses middle band while preserving critical tokens
- **RoPE Unroping/Reroping**: Enables position manipulation for repositioning
- **Micro Metrics**: Detailed logging for analysis and debugging

For detailed refactoring plans and analysis, see:
- `docs/refactoring/FINAL_REFACTORING_PLAN.md`
- `docs/refactoring/REFACTORING_ANALYSIS.md`
- `docs/refactoring/SAFE_REFACTORING_PLAN.md`

