# Chunked Prefill and Unroped Q/K for Mask Computation

## Overview

This document describes the implementation of **chunked prefill** and **unroped Q/K for mask computation** in the sparse attention system. These features enable processing of long contexts efficiently by breaking them into chunks and using position-agnostic similarity for top-K selection.

---

## Table of Contents

1. [Chunked Prefill](#chunked-prefill)
2. [Unroped Q/K for Mask Computation](#unroped-qk-for-mask-computation)
3. [Code Locations](#code-locations)
4. [Usage](#usage)
5. [Verification](#verification)

---

## Chunked Prefill

### Intent

Chunked prefill processes long input sequences by breaking them into smaller chunks (`PREFILL_CHUNK_SIZE`), allowing the model to handle contexts longer than what fits in memory at once. This is essential for:
- **Memory efficiency**: Process long contexts without OOM errors
- **Scalability**: Handle contexts of arbitrary length (limited only by KV cache management)

### How It Works

1. **Input Processing**: Long input sequences are split into chunks of size `PREFILL_CHUNK_SIZE` (default: 256 tokens)
2. **Chunk-by-Chunk Processing**: Each chunk is processed sequentially:
   - Query tokens from current chunk attend to:
     - All previous keys in KV cache (from previous chunks)
     - Keys from current chunk (causal masking ensures no future tokens)
3. **KV Cache Accumulation**: Keys and values from each chunk are appended to the KV cache
4. **Final State**: After all chunks, the KV cache contains all keys/values, ready for generation

### Key Behavior

- **Causal Masking**: Each chunk only attends to previous tokens (no future tokens)
- **Sequential Processing**: Chunks are processed in order, maintaining causal structure
- **KV Cache**: Grows with each chunk, containing all previous context

---

## Unroped Q/K for Mask Computation

### Intent

When `EXTEND_CONTEXT=1`, the system uses **unroped** (position-agnostic) Query and Key tensors for mask computation (top-K selection), while still using **roped** Q/K for the actual attention computation. This ensures:

1. **Position-Agnostic Similarity**: Top-K selection is based on content similarity, not positional encoding
2. **Correct Top-K Selection**: The most semantically similar keys are selected, regardless of their absolute position
3. **Preserved RoPE for Attention**: The actual attention scores still use roped Q/K, maintaining the model's trained positional understanding

### Why Unrope for Masks?

Rotary Position Embeddings (RoPE) encode position information into Q and K tensors. When selecting top-K keys based on attention scores computed with roped Q/K:
- Keys at similar positions get artificially high scores due to positional encoding
- True semantic similarity is masked by positional bias
- This is especially problematic for extended contexts where we want to select relevant keys regardless of position

**Solution**: Unrope Q/K before computing attention scores for mask selection, then use the selected keys with roped Q/K for actual attention.

### How It Works

1. **Compute cos/sin**: Extract or compute RoPE cos/sin tensors for queries and keys
2. **Unrope Q/K**: Reverse the RoPE transformation to get position-agnostic Q/K
3. **Verify Unroping**: Ensure unroped tensors are different from roped (diff > 1e-6)
4. **Mask Computation**: Use unroped Q/K for top-K selection (maskers)
5. **Attention Computation**: Use original roped Q/K for actual attention scores

### Mathematical Background

**RoPE Forward Transformation**:
```
x_rot = x * cos + rotate_half(x) * sin
```

**RoPE Inverse Transformation (Unroping)**:
```
x = x_rot * cos + rotate_half(x_rot) * sin
```

This is the inverse of the 2D rotation matrix: `R^T @ [x1_rot, x2_rot] = [x1, x2]`

### Code Flow

```
1. Check EXTEND_CONTEXT flag → enabled?
2. Extract/compute cos/sin from rotary_emb
3. Unrope queries: queries_for_mask = unapply_rotary_pos_emb_queries(queries, cos_queries, sin_queries)
4. Unrope keys: keys_for_mask = unapply_rotary_pos_emb_keys(keys, cos_keys, sin_keys)
5. Verify: Check that unroped != roped (diff > 1e-6)
6. Pass unroped Q/K to maskers for top-K selection
7. Use original roped Q/K for get_masked_attention_output()
```

---

## Code Locations

### Main Implementation

**File**: `sparse_attention_hub/sparse_attention/research_attention/base.py`

- **Chunked Prefill Logic**: Handled by HuggingFace adapter (see `huggingface.py`)
- **Unroping Logic**: Lines 112-437
  - Flag check: Line 113
  - cos/sin extraction: Lines 121-285
  - Unroping: Lines 286-367
  - Verification: Lines 307-332
  - Mask computation with unroped Q/K: Lines 380-412
  - Normal attention with roped Q/K: Lines 437+

**Key Functions**:
```python
# Lines 112-437: custom_attention()
# - Checks EXTEND_CONTEXT flag
# - Computes/extracts cos/sin
# - Unropes Q/K for mask computation
# - Verifies unroping worked
# - Passes unroped Q/K to maskers
# - Uses roped Q/K for attention
```

### RoPE Utilities

**File**: `sparse_attention_hub/sparse_attention/research_attention/rope_utils.py`

**Core Functions**:
- `unapply_rotary_pos_emb_queries()` (Lines 75-94): Unropes queries only
- `unapply_rotary_pos_emb_keys()` (Lines 97-116): Unropes keys only
- `unapply_rotary_pos_emb()` (Lines 43-72): Unropes both queries and keys
- `compute_rope_cos_sin()` (Lines 119-179): Computes cos/sin from position_ids
- `rotate_half()` (Lines 10-21): Helper for RoPE rotation
- `apply_rotary_pos_emb()` (Lines 24-40): Forward RoPE (for reference)

### Adapter Integration

**File**: `sparse_attention_hub/adapters/huggingface.py`

- **rotary_emb Storage**: Lines 142-150
  - Stores `rotary_emb` in `sparse_meta_data["_rotary_emb"]` when `EXTEND_CONTEXT=1`
  - Searches `model.model.rotary_emb` or `model.rotary_emb`

---

## Usage

### Basic Chunked Prefill (No Unroping)

```bash
PREFILL_CHUNK_SIZE=256 SPARSE_DEBUG=1 OUTPUT_DIR=output_test_original NUM_SAMPLES=1 python test_sparse_oracle.py
```

- Uses normal roped Q/K for mask computation
- Processes input in chunks of 256 tokens
- Standard behavior

### Chunked Prefill with Unroped Masks

```bash
EXTEND_CONTEXT=1 PREFILL_CHUNK_SIZE=256 SPARSE_DEBUG=1 OUTPUT_DIR=output_test_unroped NUM_SAMPLES=1 python test_sparse_oracle.py
```

- Uses unroped Q/K for mask computation (top-K selection)
- Uses roped Q/K for actual attention computation
- Processes input in chunks of 256 tokens

### Environment Variables

- `EXTEND_CONTEXT`: Enable unroped Q/K for mask computation (`1`, `true`, `yes`)
- `PREFILL_CHUNK_SIZE`: Size of each chunk for prefill (default: 256)
- `SPARSE_DEBUG`: Enable debug logging (`1`)
- `OUTPUT_DIR`: Directory for output files
- `NUM_SAMPLES`: Number of samples to process

---

## Verification

### Check Unroping is Happening

```bash
grep 'unroped Q/K for mask computation' output_test_unroped/hf_prefill.log | head -5
```

Should show messages like:
```
[extend_context] layer=0 unroped Q/K for mask computation. VERIFICATION: q_diff.max()=8.666034, k_diff.max()=17.702446 (unroped tensors are different from roped - GOOD)
```

### Verify Unroped Q/K are Different

```bash
grep 'q_diff.max()' output_test_unroped/hf_prefill.log | head -5
```

Should show `q_diff.max()` and `k_diff.max()` values **> 0.001** (typically > 1.0).

### Check No Unroping in Baseline

```bash
grep -c 'unroped_mask=True' output_test_original/hf_prefill.log
```

Should return `0` (no unroping in baseline).

### Verify Mask Computation Used Unroped Q/K

```bash
grep 'masker.*unroped' output_test_unroped/hf_prefill.log | head -5
```

Should show maskers receiving unroped Q/K.

### Compare Mask Densities

```bash
grep 'final_mask_density' output_test_original/hf_prefill.log | head -5
grep 'final_mask_density' output_test_unroped/hf_prefill.log | head -5
```

Mask densities should be similar (top-K selection should find similar keys, just using unroped similarity).

---

## Implementation Details

### Error Handling

The implementation **does not silently fall back**. If unroping fails:
- **ValueError**: If `rotary_emb` is not available
- **RuntimeError**: If cos/sin computation fails
- **RuntimeError**: If unroping fails
- **RuntimeError**: If unroped Q/K are identical to roped (verification failure)

All errors include detailed context for debugging.

### Verification Checks

1. **Unroping Verification** (Lines 307-332 in `base.py`):
   - Computes `q_diff = (queries - queries_for_mask).abs().max()`
   - Computes `k_diff = (keys - keys_for_mask).abs().max()`
   - **Requires**: `q_diff > 1e-6` and `k_diff > 1e-6`
   - Raises `RuntimeError` if verification fails

2. **Masker Input Verification** (Lines 386-404 in `base.py`):
   - Ensures maskers receive unroped Q/K when `EXTEND_CONTEXT=1`
   - Raises `RuntimeError` if roped Q/K would be passed to maskers

### Logging

When `SPARSE_DEBUG=1` and `EXTEND_CONTEXT=1`, the system logs:
- Cos/sin computation status
- Unroping success/failure
- Verification results (q_diff, k_diff)
- Mask computation status

Logs are written to:
- Terminal (stdout)
- `{OUTPUT_DIR}/hf_prefill.log`

---

## Key Design Decisions

1. **No Silent Fallbacks**: All failures raise explicit errors with context
2. **Separate Queries/Keys Unroping**: Handles different sequence lengths correctly
3. **Verification at Multiple Points**: Ensures correctness throughout the pipeline
4. **Clean Separation**: Unroped Q/K for masks, roped Q/K for attention
5. **No Repositioning**: Current implementation only handles unroped masks (no RoPE repositioning)

---

## Notes

- **Repositioning**: Repositioning logic has been removed. The current implementation only handles unroped Q/K for mask computation.
- **Chunked Prefill**: Works with or without `EXTEND_CONTEXT`. Chunked prefill is independent of unroping.
- **Compatibility**: Both features work together seamlessly - chunked prefill processes long contexts, unroped masks ensure correct top-K selection.

---

## Future Work

- Potential extension: RoPE repositioning for extended contexts (currently not implemented)
- Further optimization: Batch processing of chunks
- Enhanced verification: More comprehensive checks for correctness

---

## References

- **RoPE Paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Chunked Prefill**: Standard technique for processing long sequences in transformers

