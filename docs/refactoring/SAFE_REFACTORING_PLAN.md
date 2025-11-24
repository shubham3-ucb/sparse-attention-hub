# Safe Refactoring Plan - NO BREAKING CHANGES

## Goal
Extract ONLY our new repositioning code into a separate file while maintaining **100% functional equivalence** and **zero breaking changes**.

---

## Strategy: Create New File, Keep Constants Together

### ✅ **SAFE APPROACH**: Create `repositioning_utils.py`

**Key Principle**: Move repositioning logic + its constants together into a new file, import back into `mask_attention_utils.py`.

---

## Step-by-Step Plan

### Step 1: Create `repositioning_utils.py`

**Location**: `sparse_attention_hub/sparse_attention/utils/repositioning_utils.py`

**Contents**:
1. **Constants** (move from `mask_attention_utils.py`):
   ```python
   DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING: bool = True
   DEFAULT_PREFIX_SCALE_ALPHA: float = 1.0
   DEFAULT_USE_TWO_BAND_PREFIX_SCALING: bool = True
   DEFAULT_PACK_K_CHUNK_TRANSLATION: bool = True
   DEFAULT_MAX_POSITION_ID: int = 8192
   ```

2. **Helper function** (move from `mask_attention_utils.py`):
   - `_scale_prefix_positions()` - already exists, keep it

3. **New repositioning functions** (extract from inline block):
   - `_unrope_queries_and_keys()`
   - `_compute_repositioned_position_ids()`
   - `_compute_reroped_cos_sin()`
   - `_apply_reroping()`
   - `_apply_position_reassignment()` (main orchestrator)

### Step 2: Update `mask_attention_utils.py`

**Changes**:
1. **Remove constants** (they're now in `repositioning_utils.py`)
2. **Remove repositioning inline block** (lines 554-1174)
3. **Add import**:
   ```python
   from .repositioning_utils import (
       _apply_position_reassignment,
       DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING,
       DEFAULT_PREFIX_SCALE_ALPHA,
       DEFAULT_USE_TWO_BAND_PREFIX_SCALING,
       DEFAULT_PACK_K_CHUNK_TRANSLATION,
       DEFAULT_MAX_POSITION_ID,
   )
   ```
4. **Replace inline block with function call**:
   ```python
   if (sparse_attention_mask.get_density() < 1.0) and (rotary_emb is not None and position_ids_q is not None) and enable_position_reassignment:
       reroped_queries, reroped_keys = _apply_position_reassignment(
           queries=queries,
           keys=keys,
           rotary_emb=rotary_emb,
           position_ids_q=position_ids_q,
           position_ids_k=position_ids_k,
           sparse_attention_mask=sparse_attention_mask,
           kwargs=kwargs,
       )
       # Recompute attention weights with reroped Q/K
       exp_attention_weights = _compute_masked_exp_attention_weights(
           queries=reroped_queries,
           keys=reroped_keys,
           attention_mask=attention_mask,
           scaling=scaling,
           sparse_attention_mask=sparse_attention_mask,
           dropout=dropout,
           training=training,
       )
   ```

### Step 3: Ensure Constants Are Accessible

**Option A (Recommended)**: Import constants back into `mask_attention_utils.py`
- ✅ **No breaking changes** - constants still accessible via `mask_attention_utils.DEFAULT_*`
- ✅ **Backward compatible** - any code importing from `mask_attention_utils` still works

**Option B**: Keep constants in both files (not recommended - duplication)

**We'll use Option A** - import and re-export constants.

---

## Function Signatures (Exact)

### `_apply_position_reassignment()`
```python
def _apply_position_reassignment(
    queries: torch.Tensor,
    keys: torch.Tensor,
    rotary_emb: Any,
    position_ids_q: torch.Tensor,
    position_ids_k: torch.Tensor,
    sparse_attention_mask: Mask,
    kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply position reassignment and return reroped Q/K.
    
    This function extracts the repositioning logic from get_masked_attention_output().
    It maintains 100% functional equivalence with the inline implementation.
    
    Args:
        queries: Query tensor (batch, num_heads, seq_len_q, head_dim)
        keys: Key tensor (batch, num_kv_heads, seq_len_k, head_dim)
        rotary_emb: Rotary embedding module
        position_ids_q: Position IDs for queries (batch, seq_len_q)
        position_ids_k: Position IDs for keys (batch, seq_len_k)
        sparse_attention_mask: Sparse attention mask object
        kwargs: Additional keyword arguments including:
            - use_proportional_prefix_scaling (default: DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING)
            - prefix_scale_alpha (default: DEFAULT_PREFIX_SCALE_ALPHA)
            - use_two_band_prefix_scaling (default: DEFAULT_USE_TWO_BAND_PREFIX_SCALING)
            - pack_k_chunk_translation (default: DEFAULT_PACK_K_CHUNK_TRANSLATION)
            - max_position_id (default: DEFAULT_MAX_POSITION_ID)
            - num_sink_tokens (default: 0)
            - prefix_freeze_tail_k (default: 0)
            - monotone_prefix (default: True)
            - layer_idx (optional, for debugging)
    
    Returns:
        Tuple of (reroped_queries, reroped_keys) with same shapes as input
    
    Raises:
        AssertionError: If position reassignment fails or produces invalid results
    """
```

---

## Constants Handling Strategy

### ✅ **SAFE APPROACH**: Import and Re-export

In `mask_attention_utils.py`, after imports:
```python
# Import repositioning utilities
from .repositioning_utils import (
    _apply_position_reassignment,
    DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING,
    DEFAULT_PREFIX_SCALE_ALPHA,
    DEFAULT_USE_TWO_BAND_PREFIX_SCALING,
    DEFAULT_PACK_K_CHUNK_TRANSLATION,
    DEFAULT_MAX_POSITION_ID,
)

# Re-export constants for backward compatibility
# This ensures any code importing DEFAULT_* from mask_attention_utils still works
__all__ = [
    # ... existing exports ...
    "DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING",
    "DEFAULT_PREFIX_SCALE_ALPHA",
    "DEFAULT_USE_TWO_BAND_PREFIX_SCALING",
    "DEFAULT_PACK_K_CHUNK_TRANSLATION",
    "DEFAULT_MAX_POSITION_ID",
]
```

---

## Verification Checklist

### ✅ **Before Refactoring**:
- [x] Identify all constants used by repositioning code
- [x] Identify all imports needed by repositioning code
- [x] Identify all functions called by repositioning code

### ✅ **During Refactoring**:
- [ ] Copy code **verbatim** (no logic changes)
- [ ] Move constants **with** the functions that use them
- [ ] Import constants back for backward compatibility
- [ ] Test after each step

### ✅ **After Refactoring**:
- [ ] Run full test suite
- [ ] Compare outputs byte-for-byte (if possible)
- [ ] Verify constants are accessible
- [ ] Verify no import errors
- [ ] Verify no breaking changes

---

## Dependencies Analysis

### Imports Needed in `repositioning_utils.py`:
```python
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .mask import Mask
from ..research_attention.rope_utils import (
    unapply_rotary_pos_emb_queries,
    unapply_rotary_pos_emb_keys,
    rotate_half,
)
```

### Functions Called:
- `_scale_prefix_positions()` - move this function too
- `sparse_attention_mask.get_dense_mask()` - Mask method
- `rotary_emb()` - rotary embedding module call
- `_compute_masked_exp_attention_weights()` - stays in `mask_attention_utils.py` (called after repositioning)

---

## Critical Requirements

### ✅ **MUST PRESERVE**:
1. **Exact same behavior** - no logic changes
2. **Same error messages** - assertions stay identical
3. **Same timing/debugging** - all `os.environ.get()` calls preserved
4. **Same constants** - values unchanged, accessible same way
5. **Same function signatures** - no parameter changes

### ✅ **MUST AVOID**:
1. ❌ Changing constant values
2. ❌ Changing function behavior
3. ❌ Breaking imports
4. ❌ Changing error messages
5. ❌ Removing any code paths

---

## Implementation Order

1. **Create `repositioning_utils.py`** with constants + function stubs
2. **Copy `_scale_prefix_positions()`** verbatim
3. **Extract `_unrope_queries_and_keys()`** verbatim
4. **Extract `_apply_reroping()`** verbatim
5. **Extract `_compute_reroped_cos_sin()`** verbatim
6. **Extract `_compute_repositioned_position_ids()`** verbatim
7. **Create `_apply_position_reassignment()`** orchestrator
8. **Update `mask_attention_utils.py`** to import and use
9. **Test** - run full test suite
10. **Verify** - compare outputs

---

## Risk Mitigation

### Low Risk ✅
- Constants are module-level, easy to move
- Functions are self-contained
- No external dependencies on global state

### Medium Risk ⚠️
- Large function extraction (~390 lines for `_compute_repositioned_position_ids`)
- Many local variables need to be passed correctly

### Mitigation
- Extract incrementally (one function at a time)
- Test after each extraction
- Use type hints to catch errors early
- Keep original code in git for rollback

---

## Conclusion

✅ **REFACTORING IS SAFE** - We can extract repositioning code into a separate file while maintaining 100% functional equivalence and zero breaking changes.

**Key Safety Measures**:
1. Move constants WITH functions
2. Import constants back for backward compatibility
3. Copy code verbatim (no logic changes)
4. Test after each step
5. Keep git history for rollback

