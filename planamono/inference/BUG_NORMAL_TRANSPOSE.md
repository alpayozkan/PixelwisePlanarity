# Bug: Normal Map Transpose Issue in `predict_batch_fast`

## Problem

When running `inference_to_h5.py`, the segmentation fails with:

```
RuntimeError: Given groups=3, weight of size [3, 1, 3, 3], expected input[1, 512, 1440, 1920] to have 3 channels, but got 512 channels instead
```

## Root Cause

**File:** `planamono/inference/planarity/moge_inference.py`
**Function:** `predict_batch_fast()` (lines 451-457)

The code incorrectly transposes the normal and points outputs:

```python
if "normal" in outputs:
    normal = outputs["normal"][i].cpu().numpy().transpose(1, 2, 0)  # BUG
    res["normal"] = normal
if "points" in outputs:
    points = outputs["points"][i].cpu().numpy().transpose(1, 2, 0)  # BUG
    res["points"] = points
```

## Why It's Wrong

MoGe v2 model (`planamono/moge/moge/model/v2.py`, lines 183-185) outputs normal in `(B, H, W, 3)` format:

```python
if normal is not None:
    normal = normal.permute(0, 2, 3, 1)  # Converts from (B,3,H,W) to (B,H,W,3)
    normal = F.normalize(normal, dim=-1)
```

So `outputs["normal"][i]` is already `(H, W, 3)` = `(512, 768, 3)`.

The buggy `.transpose(1, 2, 0)` converts `(512, 768, 3)` → `(768, 3, 512)`, which is wrong.

Later in `inference_to_h5.py`:
```python
normal = res["normal"].transpose(2, 0, 1)  # Expects (H,W,3) → (3,H,W)
```

With the buggy input `(768, 3, 512)`, this becomes `(512, 768, 3)` which then gets resized and passed to segmentation as `(512, 1440, 1920)` - hence "512 channels".

## Fix

In `moge_inference.py`, remove the transpose since MoGe v2 already outputs `(H, W, 3)`:

```python
if return_all_heads:
    if "normal" in outputs:
        # MoGe v2 already outputs (B, H, W, 3), no transpose needed
        normal = outputs["normal"][i].cpu().numpy()
        res["normal"] = normal
    if "points" in outputs:
        # MoGe v2 already outputs (B, H, W, 3), no transpose needed
        points = outputs["points"][i].cpu().numpy()
        res["points"] = points
```

## Files Affected

- `planamono/inference/planarity/moge_inference.py` - needs fix
- `planamono/inference/planarity/inference_to_h5.py` - uses the buggy function

## Date Found

2025-02-05

## Status: FIXED (2026-02-05)

The bug has been fixed in [moge_inference.py:453-458](planamono/inference/planarity/moge_inference.py#L453-L458).

**Changes made:**
```python
# BEFORE (buggy)
normal = outputs["normal"][i].cpu().numpy().transpose(1, 2, 0)  # ❌
points = outputs["points"][i].cpu().numpy().transpose(1, 2, 0)  # ❌

# AFTER (fixed)
normal = outputs["normal"][i].cpu().numpy()  # ✓ No transpose needed
points = outputs["points"][i].cpu().numpy()  # ✓ No transpose needed
```

The fix resolves the dimension mismatch error in `inference_to_h5.py` and enables proper end-to-end inference with segmentation.
