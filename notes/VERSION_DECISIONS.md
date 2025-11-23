# Version Decisions

This document explains which version of each file was kept and why.

## Core Algorithms

### planes_from_mesh_labels_v1.py (Kept)
**Why:** v1 is the actively used version in all shell scripts. Contains the complete pipeline with:
- Label-strict region growing
- EM-based expansion
- Recursive large-label splitting
- IRLS plane fitting
- Multi-stage quality filtering

**What happened to others:**
- `planes_from_mesh_labels.py` (older) → Removed, v1 has improvements

### plan2seg compute_vectorized_planar_segments_v1 (Kept)
**Why:** v1 has the best performance based on code analysis:
- Includes neighbor match count threshold for robustness
- Better handling of edge cases
- Most referenced in evaluation scripts

**What happened to others:**
- v0 → Basic version, kept as fallback logic
- v2 → 24-connected variant, feature merged
- v3 → Gradient-based, experimental
- v4 → GPU version, requires PyTorch, kept as optional

**Decision:** Keep v1 as default, document v4 as GPU-accelerated alternative

### Plane Fitting Functions

#### backproject_v1 (Kept)
**Why:** Returns valid_idx which is needed for proper index mapping
- Signature: `backproject_v1(depth, K, T_cw, plane_seg) → (pts_world, labels, valid_idx)`
- More complete than backproject()

#### fit_planes_per_label_v1 (Kept)
**Why:** Returns comprehensive metrics including:
- RANSAC and refined inlier ratios
- plane_df DataFrame for easy analysis
- Better outlier handling

#### compute_precision_recall_v1 (Kept)
**Why:** More detailed metrics per plane and globally

## Dataset-Specific

### Hypersim Rendering

#### render_scene_h5_v2.py (Kept)
**Why:** v2 has improvements over v1:
- Better camera intrinsics computation
- Multi-camera support
- Optimized HDF5 writing

**What happened to others:**
- `render_scene_h5.py` (v1) → Removed
- `render_scene.py` (PNG version) → Removed, HDF5 is standard

### ScanNet++ Rendering

#### render_scene_depth_h5.py (Kept)
**Why:** HDF5 format is standard for efficiency
- Used in all evaluation pipelines
- PNG version was slower and larger

### Hypersim Plane Extraction

#### plane_gt_run_hypersim_v1.py (Kept)
**Why:** v1 has additional configuration parameters
- More flexible than base version
- Better error handling

**What happened to others:**
- `plane_gt_run_hypersim.py` → Older version

#### planes_from_mesh_hypersim_v1.py (Kept)
**Why:** Hypersim-specific adaptations in v1
- Handles Hypersim mesh structure better
- Better semantic label handling

## Segmentation Variants

### plan2seg_pred_moge.py (Kept as base for inference/segmentation/predict.py)
**Why:** Most complete MoGe-based segmentation pipeline
- Includes all preprocessing steps
- Well-tested parameter settings
- Used in latest evaluations

**What happened to others:**
- `plan2seg_pred_moge2.py` → Minor variant, features merged
- `plan2seg_pred_v1.py` → Older, superseded by moge version
- `plan2seg_pred_v3.py` → Experimental variant

### plan2seg_gt.py versions
**Decision:** Merged all into gt_creation modules
- v1 and v3 had minor differences in thresholds
- Consolidated into single configurable version

## Visualization

### visualize_planes_v1.py (Kept)
**Why:** Most complete visualization utilities
- Better color palette generation
- Semantic label handling
- Used by both GT and evaluation

**What happened to others:**
- `visualize_planes.py` → Base version, v1 has improvements

## Evaluation

### eval.py + eval_run.py (Kept both, split functionality)
**Why:**
- `eval.py` → Core evaluation logic (reusable functions)
- `eval_run.py` → Main runner (orchestration)
- Clean separation of concerns

### Baseline Comparison Scripts
**Decision:** Kept only MoGe versions
- `qual_compare_baseline_moge.py` → Most complete
- `quant_compare_baseline_moge.py` → Most complete
- Dice variants merged into main versions

## General Principles

1. **Prefer v1 over base** - v1 versions typically have bug fixes and improvements
2. **Prefer HDF5 over PNG** - HDF5 is more efficient for large datasets
3. **Prefer latest MoGe integration** - MoGe is the current inference method
4. **Keep complete pipelines** - Kept files that have full workflows, not partial
5. **Remove redundant variants** - Merged similar files with minor differences

## Edge Cases

### Kept Multiple Versions Where Justified
- `compute_vectorized_planar_segments_v1` AND `v4` (GPU variant)
- Different rendering scripts for ScanNet++ vs Hypersim (dataset-specific)

### Combined Multiple Files
- All qual_compare scripts → `visualize_comparison.py`
- All video generation scripts → `video_generation.py`
- Multiple utils files → Organized into shared/utils/

## Validation
All kept versions were validated by:
1. Checking which shell scripts reference them
2. Analyzing import statements in active code
3. Reviewing git history for latest modifications
4. Testing completeness of functionality
