# Restructuring Log

This document tracks all file movements and consolidations during the codebase restructuring.

## Directory Structure Overview

```
clean_structure/
├── shared/          - Core utilities used across all modules
├── gt_creation/     - Ground truth generation from meshes
├── inference/       - Planarity prediction and segmentation
├── evaluation/      - Quantitative and qualitative evaluation
├── exploration/     - Jupyter notebooks for experimentation
├── MoGe/           - External MoGe model integration
└── notes/          - Restructuring documentation
```

## File Mappings

### Shared Module

#### shared/plane_fitting/
- `planefit.py` ← plane_fitting/planefit.py (kept v1 functions: backproject_v1, fit_planes_per_label_v1)
- `metrics.py` ← plane_fitting/planefit_metrics.py (kept compute_precision_recall_v1)
- `projection.py` ← plane_fitting/planefit_utils.py (kept project_points_to_image_v1)
- `visualize.py` ← plane_fitting/planefit_visualize.py + planeseg_visualize.py (merged)

#### shared/rendering/
- `render.py` ← gt_gen/render.py (raycast_semantic, render_rgb_depth, etc.)
- `mesh_io.py` ← gt_gen/mesh_utils.py (mesh loading/saving functions)
- `raycasting.py` ← gt_gen/render.py (face/vertex raycasting split out)

#### shared/segmentation/
- `plan2seg.py` ← planarity_2_segmentation/plan2seg.py (kept compute_vectorized_planar_segments_v1)
- `region_grow.py` ← planarity_2_segmentation/regiongrow.py
- `postprocess.py` ← planarity_2_segmentation/postprocess.py + utils.py (merged)

#### shared/datasets/
- `scannetpp.py` ← dataset/scannetpp/dataset_scannet_plane.py
- `hypersim.py` ← dataset/hypersim/dataset_hypersim.py
- `base.py` ← New file with shared dataset utilities

#### shared/utils/
- `depth_normal.py` ← planarity_2_segmentation/process_remi.py + transform.py
- `visualization.py` ← planarity_2_segmentation/visualize.py + visualize_seg.py + gt_gen/visualize_planes_v1.py
- `label_utils.py` ← homography/homog_utils.py + gt_gen/utils.py
- `io_utils.py` ← New file with HDF5 and file I/O helpers

### GT Creation

#### gt_creation/scannetpp/
- `plane_extraction.py` ← gt_gen/planes_from_mesh_labels_v1.py (core algorithm, cleaned)
- `preprocessing.py` ← gt_gen/preprocess_mesh.py
- `rendering.py` ← gt_gen/render_scene.py + render_scene_sem.py + render_scene_depth_h5.py
- `scene_runner.py` ← gt_gen/plane_gt_run.py (orchestrates pipeline)
- `parsers.py` ← gt_gen/parse_scannetpp.py
- `video_gen.py` ← gt_gen/gen_video2.py

#### gt_creation/hypersim/
- `plane_extraction.py` ← gt_gen/hypersim/planes_from_mesh_hypersim_v1.py
- `rendering.py` ← gt_gen/hypersim/render_scene_h5_v2.py
- `mesh_processing.py` ← gt_gen/hypersim/hypersim_mesh.py
- `scene_runner.py` ← gt_gen/hypersim/plane_gt_run_hypersim_v1.py (was plane_gt_run_hypersim.py)
- `video_gen.py` ← gt_gen/hypersim/gen_video.py

#### gt_creation/configs/
- `scannetpp_default.yml` ← gt_gen/plane_config.yml
- `hypersim_default.yml` ← gt_gen/hypersim/plane_config_hypersim_v1.yml

#### gt_creation/scripts/
- `scannetpp_pipeline.sh` ← Consolidated from gt_gen/run_*.sh
- `hypersim_pipeline.sh` ← Consolidated from gt_gen/hypersim/run_*.sh

### Inference

#### inference/planarity/
- `moge_inference.py` ← monocular/moge/inference.py (cleaned, removed hardcoded paths)

#### inference/segmentation/
- `predict.py` ← planarity_2_segmentation/plan2seg_pred_moge.py (consolidated from _moge, _moge2)
- `scannetpp_runner.py` ← planarity_2_segmentation/planeseg_scannet.py
- `hypersim_runner.py` ← planarity_2_segmentation/plan2seg_pred.py (adapted)

### Evaluation

#### evaluation/quantitative/
- `evaluator.py` ← plane_fitting/eval.py (core evaluation logic)
- `metrics.py` ← plane_fitting/eval.py (metric computations extracted)
- `method_wrappers.py` ← plane_fitting/eval_moge.py + eval_monoplane.py

#### evaluation/qualitative/
- `visualize_comparison.py` ← evaluation/qualitative/qual_compare_baseline_moge.py (consolidated)
- `video_generation.py` ← evaluation/qualitative/planeseg_video_moge.py

#### evaluation/
- `run_evaluation.py` ← plane_fitting/eval_run.py (cleaned)

### Exploration

#### exploration/hypersim/ (Key notebooks kept)
- planarity_prediction.ipynb ← Root
- hipersim.ipynb ← Root

#### exploration/scannetpp/ (Key notebooks kept)
- scene_understanding.ipynb ← Root
- planarity2segmentation.ipynb ← Root

#### exploration/archived/
- All other .ipynb files

### MoGe
- README.md ← New file with integration documentation

## Removed Files

Files that were removed (documented in REMOVED_FILES.md):
- All v0, v2, v3 versions after consolidation
- Duplicate scripts
- Empty files (preprocess.py)
- One-off utilities (permission.sh, preprocess.sh)

## Version Decisions

Documented in VERSION_DECISIONS.md
