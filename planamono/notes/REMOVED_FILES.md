# Removed Files

This document lists all files that were removed during restructuring and explains why.

## Version Consolidations

### Kept v1, Removed Others
- **planes_from_mesh_labels.py** → Kept v1 (planes_from_mesh_labels_v1.py is the actively used version)
- **plan2seg.py versions** → Kept v1 algorithm (compute_vectorized_planar_segments_v1) - best performance with neighbor matching
- **render_scene_h5.py** → Kept v2 for Hypersim (render_scene_h5_v2.py)
- **plane_gt_run_hypersim.py** → Kept v1 (plane_gt_run_hypersim_v1.py)
- **planes_from_mesh_hypersim.py** → Kept v1
- **visualize_planes.py** → Kept v1

### Removed Variants
- `planarity_2_segmentation/plan2seg_pred_v1.py` → Consolidated into predict.py
- `planarity_2_segmentation/plan2seg_pred_v3.py` → Consolidated into predict.py
- `planarity_2_segmentation/plan2seg_pred_moge2.py` → Merged with plan2seg_pred_moge.py
- `planarity_2_segmentation/plan2seg_gt_v1.py` → Consolidated
- `planarity_2_segmentation/plan2seg_gt_v3.py` → Consolidated
- `gt_gen/render_scene_depth.py` → Kept HDF5 version (render_scene_depth_h5.py)
- `gt_gen/hypersim/render_scene.py` → Kept HDF5 version
- `gt_gen/hypersim/render_scene_h5.py` → Kept v2
- `gt_gen/planes_from_mesh_labels_run.py` → Logic merged into scene_runner.py

## Empty/Minimal Files
- `preprocess.py` (0 bytes) → Empty file, removed
- `gt_gen/compress_data.py` (0 bytes) → Empty file, removed

## One-Off Utility Scripts
- `permission.sh` → Server-specific ACL setup, not needed in clean codebase
- `preprocess.sh` → One-time file renaming (*.gt.npy → *.planarity.npy), not needed
- `hipersim_download.bash` → Redundant with dataset/hypersim/download.sh

## Duplicate/Similar Scripts
- `plan2seg_gt.bash` → Functionality incorporated into gt_creation/scripts/
- Multiple `planeseg_scannet*.py` variants → Consolidated into scannetpp_runner.py
- `planeseg_scannetv2_baseline_video.py` → Merged into video_generation.py
- `planeseg_scannetv2_video.py` → Merged into video_generation.py

## Exploratory/Development Files
- `planarity_2_segmentation/plane_video.py` → Kept only plane_video_top_supp.py logic
- `planarity_2_segmentation/plane_video_top.py` → Merged into video utilities
- `evaluation/qualitative/qual_compare_baseline.py` → Kept _moge version
- `evaluation/qualitative/qual_compare_baseline_moge_dice.py` → Consolidated
- `evaluation/quantitative/quant_compare_baseline.py` → Consolidated
- `evaluation/quantitative/quant_compare_baseline_moge_dice.py` → Consolidated

## Notebook Scripts
- `dataset/hypersim/notebooks/render_hypersim.py` → Exploratory, moved to exploration/
- `dataset/hypersim/notebooks/visualize_planes_v2.py` → Functionality in shared/utils/visualization.py

## Dataset-Specific Files Consolidated
- `dataset/scannetpp/comp_planes.py` → Merged into evaluation utilities
- `dataset/scannetpp/comp_sem.py` → Merged into evaluation utilities
- `dataset/hypersim/scene_list.py` → Merged into dataset loaders
- `dataset/hypersim/split_scenes.py` → Merged into dataset loaders

## Homography Experiments
- `homography/homog.py` → Experimental, kept only homog_utils.py in shared/
- `homography/homog_warp.py` → Experimental, archived
- `homography/visualize.py` → Merged into shared/utils/visualization.py
- `homography/homog_vis.py` → Merged into shared/utils/visualization.py
- `homography/warp_v0/` → Old experiments, removed
- `homography/warp_v1/` → Old experiments, removed

## Planarity DBSCAN
- `planarity_dbscan/` directory → Old clustering experiments, not used in current pipeline

## Submit Scripts (Cluster-Specific)
- `gt_gen/submit_all_splits*.sh` → Cluster-specific batch submission, documented but not copied
- `gt_gen/hypersim/submit_all_splits.sh` → Cluster-specific
- `plane_fitting/eval_run_job*.sh` → SLURM-specific, documented in scripts/README.md

## Total Files Removed
Approximately 80+ files were either removed or consolidated into cleaner versions.

## Notes
- All removed files are still available in the original codebase
- Key functionality from removed files was preserved in consolidated modules
- Version decisions documented in VERSION_DECISIONS.md
- Function mappings documented in CONSOLIDATED_FUNCTIONS.md
