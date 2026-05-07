# MoGe Inference Scripts Comparison

## Files
- `moge_inference.py` (Feb 5 13:35) - **PARTIALLY FIXED**
- `moge_inference_v1.py` (Feb 5 11:40) - **FULLY FIXED**

## Bug: Normal/Points Transpose Issue

### Root Cause
MoGe v2 outputs normal and points in `(B, H, W, 3)` format (already in HWC).
Incorrect `.transpose(1, 2, 0)` operations were converting `(H, W, 3)` → `(W, 3, H)`, causing dimension mismatches.

See `BUG_NORMAL_TRANSPOSE.md` for full details.

## Status of Each File

### moge_inference.py (PARTIALLY FIXED)
**Fixed:**
- ✅ `predict_batch_fast()` method (lines 453-458)
  - Has correct comment: "MoGe v2 already outputs (B, H, W, 3), no transpose needed"
  - No transpose operations

**Still Buggy:**
- ❌ `predict()` method (lines 260-268)
  - Line 263: Still has `normal.transpose(1, 2, 0)` ❌
  - Line 267: Missing points handling (but probably not used)

### moge_inference_v1.py (FULLY FIXED)
**Fixed:**
- ✅ `predict()` method (lines 264-274)
  - Lines 265-268: Proper comment and no transpose
  - Lines 271-274: Points also fixed
- ✅ `predict_batch_fast()` method (lines 459-464)
  - Lines 459-461: Normal fixed
  - Lines 463-465: Points fixed

## Recommendation: CONSOLIDATE TO v1

**v1 is the correct, fully fixed version.** The main file was only partially updated.

### Why Consolidate?
1. **v1 is complete** - Both methods fixed
2. **Eliminates confusion** - One source of truth
3. **Prevents regressions** - No one will accidentally use the buggy `predict()` method

### Action Plan
1. Replace `moge_inference.py` with `moge_inference_v1.py`
2. Delete `moge_inference_v1.py` (no longer needed)
3. Remove version suffix from docstring (it's now the default)

## Impact Analysis

### Which method is used where?
- `predict()` - Single image inference, visualization scripts
- `predict_batch_fast()` - Batch inference, H5 generation scripts (`inference_to_h5.py`)

**Critical:** `inference_to_h5.py` likely uses `predict_batch_fast()` (which is fixed in both versions), so current H5 generation should be fine. But any code using `predict()` with `return_all_heads=True` would get buggy normal maps from `moge_inference.py`.

## Testing After Consolidation

```bash
# Test single image prediction with all heads
python moge_inference.py --model_path <checkpoint> --input <image> --return_all_heads

# Verify normal map dimensions
# Should be (H, W, 3) not (W, 3, H)
```
