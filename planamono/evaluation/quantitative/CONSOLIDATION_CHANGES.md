# Evaluation Script Consolidation Changes

## Date: 2026-02-05

## Summary
Consolidated evaluation scripts to use a unified approach for handling different label conventions across baseline methods. The main improvement is proper handling of ZeroPlane's non-planar region labeling (label 20 → 0 remapping).

## Files Modified

### 1. `evaluate_all_baselines.py` (Replaced with nonp version)
**Changes:**
- Now the primary/default evaluation script
- Added `nonplanar_label` parameter to method configurations
- Updated `LazyH5SceneLoader` to support non-planar label remapping
- For ZeroPlane: `nonplanar_label=20`, remaps to 0 before evaluation
- For other methods: `nonplanar_label=None`, no remapping needed
- Updated `EXP_VER` from "v4" → "v5"
- More comprehensive aggregation (includes all thresholds in combined table)

**Key Improvement:**
```python
# Old approach (incorrect):
"label_offset": 1  # Would turn 1-20 into 2-21

# New approach (correct):
"nonplanar_label": 20  # Remap 20 → 0 first
"label_offset": 0      # Then no offset needed
```

### 2. `evaluate_all_baselines_nonp.py` (Deleted)
**Action:** Removed (functionality now in main file)

**Reason:** Eliminated code duplication, consolidated into single default script

### 3. `evaluate_all_baselines_old.py.bak` (Created)
**Action:** Backup of old `evaluate_all_baselines.py` before replacement

**Purpose:** Preserves original version for reference/comparison

### 4. `evaluate_scannetpp_zeroplane_fast.py` (Updated)
**Changes:**
- Updated `LazyH5SceneLoader.__init__()` to accept `nonplanar_label=20` parameter
- Replaced `pred += 1` with proper remapping: `pred[pred == 20] = 0`
- Added `.copy()` to avoid modifying cached data
- Updated docstring to clarify label handling

### 5. `evaluate_hypersim_all_baselines.py` (Already Correct)
**Status:** No changes needed - already uses nonplanar_label approach

## Impact on Dependent Scripts

These scripts import from `evaluate_all_baselines.py` and automatically benefit from the changes:

1. `visualize_scannetpp_all_baselines.py`
2. `visualize_scannetpp_all_baselines_v1.py`
3. `visualize_scannetpp_all_baselines_v2.py`
4. `agg_results_baselines.py`

**No changes needed** - they import `METHODS`, `THRESHOLDS`, and other constants from the updated file.

## Label Convention Handling

### Before (Inconsistent):
```python
# ZeroPlane - Old approach
"label_offset": 1  # Incorrect: 1-20 → 2-21

# Ours - Standard approach
"label_offset": 0  # Correct: 0, 1, 2, ...
```

### After (Unified):
```python
# ZeroPlane
"nonplanar_label": 20  # Map 20 → 0
"label_offset": 0      # No offset needed

# Ours
"nonplanar_label": None  # No remapping
"label_offset": 0        # Standard convention
```

## Backward Compatibility

- Old results saved under `eval/method_v4/`
- New results saved under `eval/method_v5/`
- Different experiment versions prevent accidental overwriting
- Can compare old vs new results side-by-side

## Benefits

1. **Correctness**: Proper handling of ZeroPlane's non-planar regions
2. **Maintainability**: Single source of truth, no code duplication
3. **Extensibility**: Easy to add new methods with custom label conventions
4. **Consistency**: All evaluation scripts use the same approach

## Testing Recommendations

Before running on full dataset:
1. Test on 1-2 scenes: `--max-scenes 2`
2. Compare v4 vs v5 results on same scenes
3. Verify ZeroPlane predictions look correct (background should be 0)
4. Check aggregated tables include all methods

## Usage

```bash
# Evaluate all methods (new default behavior)
python evaluate_all_baselines.py

# Evaluate specific methods
python evaluate_all_baselines.py --methods ours zeroplane

# Only aggregate existing results
python evaluate_all_baselines.py --aggregate-only

# Test on small subset
python evaluate_all_baselines.py --max-scenes 2
```
