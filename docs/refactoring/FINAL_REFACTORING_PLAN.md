# FINAL REFACTORING PLAN - EXACT SAME FUNCTIONALITY, ZERO BREAKING CHANGES

## ✅ GUARANTEE: EXACT SAME CODE, JUST ORGANIZED

**CRITICAL**: This refactoring will:
- ✅ Copy code **VERBATIM** (no logic changes)
- ✅ Preserve **EXACT** same behavior
- ✅ Keep constants accessible (re-export)
- ✅ No breaking changes
- ✅ No silent failures
- ✅ No fallbacks

---

## What We're Extracting

### Code Block to Extract: Lines 554-1174 in `mask_attention_utils.py`
**EXACTLY** 621 lines of repositioning logic that:
- Only executes when `ENABLE_POSITION_REASSIGNMENT=1`
- Uses constants: `DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING`, `DEFAULT_PREFIX_SCALE_ALPHA`, `DEFAULT_USE_TWO_BAND_PREFIX_SCALING`, `DEFAULT_PACK_K_CHUNK_TRANSLATION`, `DEFAULT_MAX_POSITION_ID`
- Uses helper: `_scale_prefix_positions()` (lines 26-61)
- Uses imports: `random`, `rope_utils` functions

### Code Block to KEEP: Lines 518-553 (setup code)
**CRITICAL**: This code runs BEFORE repositioning block and sets up `unroped_queries`/`unroped_keys`. Must stay in place.

### Code Block to KEEP: Lines 1177-1193 (final computation)
**CRITICAL**: This code runs AFTER repositioning block. Must stay in place.

---

## Step-by-Step Implementation

### Step 1: Create `repositioning_utils.py`

**File**: `sparse_attention_hub/sparse_attention/utils/repositioning_utils.py`

**Contents**:
1. **Imports** (EXACT same as used in block):
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

2. **Constants** (EXACT same values):
   ```python
   DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING: bool = True
   DEFAULT_PREFIX_SCALE_ALPHA: float = 1.0
   DEFAULT_USE_TWO_BAND_PREFIX_SCALING: bool = True
   DEFAULT_PACK_K_CHUNK_TRANSLATION: bool = True
   DEFAULT_MAX_POSITION_ID: int = 8192
   ```

3. **Helper function** (copy VERBATIM from lines 26-61):
   ```python
   def _scale_prefix_positions(...):
       # EXACT copy from lines 26-61
   ```

4. **Main function** (copy VERBATIM from lines 554-1174):
   ```python
   def _apply_position_reassignment(
       queries: torch.Tensor,
       keys: torch.Tensor,
       rotary_emb: Any,
       position_ids_q: torch.Tensor,
       position_ids_k: torch.Tensor,
       sparse_attention_mask: Mask,
       exp_attention_weights: torch.Tensor,  # Pre-computed weights
       attention_mask: Optional[torch.Tensor],
       scaling: float,
       dropout: float,
       training: bool,
       kwargs: Dict[str, Any],
   ) -> torch.Tensor:
       """
       Apply position reassignment and return reroped attention weights.
       
       This is EXACT copy of lines 554-1174 from mask_attention_utils.py.
       NO LOGIC CHANGES - just extracted for modularity.
       """
       # EXACT copy of lines 554-1174
       # Returns: exp_attention_weights_reroped
   ```

### Step 2: Update `mask_attention_utils.py`

**Changes**:

1. **Add import** (at top, after existing imports):
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

2. **Remove constants** (lines 13-23):
   - Delete: `DEFAULT_USE_PROPORTIONAL_PREFIX_SCALING`
   - Delete: `DEFAULT_PREFIX_SCALE_ALPHA`
   - Delete: `DEFAULT_USE_TWO_BAND_PREFIX_SCALING`
   - Delete: `DEFAULT_PACK_K_CHUNK_TRANSLATION`
   - Delete: `DEFAULT_MAX_POSITION_ID`

3. **Remove helper function** (lines 26-61):
   - Delete: `_scale_prefix_positions()` (moved to repositioning_utils.py)

4. **Remove repositioning block** (lines 554-1174):
   - Delete: Entire `if enable_position_reassignment:` block

5. **Replace with function call** (at line 554):
   ```python
   if (sparse_attention_mask.get_density() < 1.0) and (rotary_emb is not None and position_ids_q is not None) and enable_position_reassignment:
       exp_attention_weights = _apply_position_reassignment(
           queries=queries,
           keys=keys,
           rotary_emb=rotary_emb,
           position_ids_q=position_ids_q,
           position_ids_k=position_ids_k,
           sparse_attention_mask=sparse_attention_mask,
           exp_attention_weights=exp_attention_weights,
           attention_mask=attention_mask,
           scaling=scaling,
           dropout=dropout,
           training=training,
           kwargs=kwargs,
       )
   ```

6. **Remove unused imports** (if any):
   - Check: `random` import (line 518) - only used in repositioning block
   - Check: `rope_utils` imports (line 519-523) - only used in repositioning block
   - Remove if not used elsewhere

### Step 3: Verify No Breaking Changes

**Check**:
- ✅ Constants still accessible via `mask_attention_utils.DEFAULT_*`
- ✅ Function signature matches exactly
- ✅ All assertions preserved
- ✅ All error messages preserved
- ✅ All timing/debugging preserved

---

## Critical Requirements

### ✅ MUST PRESERVE:
1. **Exact same code execution** - line-by-line identical behavior
2. **Same error messages** - all assertions stay identical
3. **Same timing** - all `time.time()` calls preserved
4. **Same debugging** - all `os.environ.get()` calls preserved
5. **Same constants** - values unchanged, accessible same way

### ❌ MUST NOT:
1. Change any logic
2. Change any constants
3. Change any error messages
4. Remove any assertions
5. Change any function signatures (except adding new function)

---

## Verification Checklist

### Before Refactoring:
- [x] Read entire repositioning block (lines 554-1174)
- [x] Identify all constants used
- [x] Identify all imports needed
- [x] Identify all functions called
- [x] Identify all variables used

### During Refactoring:
- [ ] Copy code VERBATIM (no changes)
- [ ] Move constants WITH functions
- [ ] Import constants back for backward compatibility
- [ ] Test after each step

### After Refactoring:
- [ ] Run full test suite
- [ ] Compare outputs byte-for-byte
- [ ] Verify constants accessible
- [ ] Verify no import errors
- [ ] Verify no breaking changes

---

## File Structure After Refactoring

```
sparse_attention_hub/sparse_attention/utils/
├── mask_attention_utils.py (reduced from 1193 to ~570 lines)
│   ├── Imports (including repositioning_utils)
│   ├── Other functions (unchanged)
│   └── get_masked_attention_output() (calls _apply_position_reassignment)
│
└── repositioning_utils.py (NEW, ~650 lines)
    ├── Constants (DEFAULT_*)
    ├── _scale_prefix_positions()
    └── _apply_position_reassignment()
```

---

## Safety Guarantees

### ✅ **NO BREAKING CHANGES**:
- Constants re-exported from `mask_attention_utils.py`
- Function signatures match exactly
- All imports preserved

### ✅ **EXACT SAME FUNCTIONALITY**:
- Code copied verbatim
- No logic changes
- Same execution path

### ✅ **NO SILENT FAILURES**:
- All assertions preserved
- All error messages preserved
- All verification logic preserved

---

## Implementation Order

1. **Create `repositioning_utils.py`** with constants + function stub
2. **Copy `_scale_prefix_positions()`** verbatim
3. **Copy repositioning block** verbatim (lines 554-1174)
4. **Update `mask_attention_utils.py`** to import and call
5. **Test** - run full test suite
6. **Verify** - compare outputs

---

## Conclusion

✅ **REFACTORING IS SAFE** - We extract ONLY the repositioning block (lines 554-1174) into a separate file, keeping everything else EXACTLY the same.

**Key Safety Measures**:
1. Copy code verbatim (no logic changes)
2. Move constants with functions
3. Re-export constants for backward compatibility
4. Test after each step
5. Keep git history for rollback

