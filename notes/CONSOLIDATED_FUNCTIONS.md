# Consolidated Functions

This document maps functions from old locations to new locations in the restructured codebase.

## Plane Fitting Functions

### From plane_fitting/planefit.py → shared/plane_fitting/planefit.py
- `backproject()` → Removed, use `backproject_v1()`
- `backproject_v1(depth, K, T_cw, plane_seg)` → **Kept** (returns pts_world, labels, valid_idx)
- `refine_plane_least_squares(P)` → **Kept**
- `fit_planes_per_label_v1(...)` → **Kept** (main RANSAC fitting function)
- `filter_planes_by_inlier_ratio(...)` → **Kept**
- `mark_planes_below_threshold_as_outliers(...)` → **Kept**

### From plane_fitting/planefit_metrics.py → shared/plane_fitting/metrics.py
- `compute_precision_recall()` → Removed, use v1
- `compute_precision_recall_v1(...)` → **Kept** (preferred version)

### From plane_fitting/planefit_utils.py → shared/plane_fitting/projection.py
- `project_points(...)` → Removed, use v1
- `project_labels_to_image(...)` → Removed, use v1
- `project_points_to_image(...)` → Removed, use v1
- `project_points_to_image_v1(...)` → **Kept** (handles visibility correctly)

### From plane_fitting/*visualize.py → shared/plane_fitting/visualize.py
- All visualization functions merged and cleaned

## Rendering Functions

### From gt_gen/render.py → shared/rendering/render.py
- `render_rgb(mesh, width, height, K, T_wc)` → **Kept**
- `render_rgb_depth(mesh, width, height, K, T_wc)` → **Kept**

### From gt_gen/render.py → shared/rendering/raycasting.py
- `raycast_semantic(mesh, labels, width, height, K, T_wc)` → **Kept**
- `raycast_semantic_face_labels(mesh, face_labels, width, height, K, T_wc)` → **Kept**

### From gt_gen/mesh_utils.py → shared/rendering/mesh_io.py
- `read_ply_faces_with_plane_ids(filepath)` → **Kept**
- `save_mesh_with_vertex_labels(...)` → **Kept**
- `load_mesh_with_vertex_labels(filepath)` → **Kept**
- `propagate_face_labels_to_vertices(mesh, face_labels)` → **Kept**

## Segmentation Functions

### From planarity_2_segmentation/plan2seg.py → shared/segmentation/plan2seg.py
- `compute_vectorized_planar_segments_v0(...)` → Moved to legacy
- `compute_vectorized_planar_segments_v1(...)` → **Kept** (recommended)
- `compute_vectorized_planar_segments_v2(...)` → Moved to legacy
- `compute_vectorized_planar_segments_v3(...)` → Moved to legacy
- `compute_vectorized_planar_segments_v4(...)` → **Kept** (GPU version)
- `filter_small_segments(segmentation, min_size)` → **Kept**

### From planarity_2_segmentation/utils.py → shared/segmentation/postprocess.py
- `remove_small_components(segmentation, min_size)` → **Kept**

### From planarity_2_segmentation/regiongrow.py → shared/segmentation/region_grow.py
- All region growing functions → **Kept**

## Depth & Normal Processing

### From planarity_2_segmentation/process_remi.py → shared/utils/depth_normal.py
- `depth_to_normal_remi(depth, fx, fy, cx, cy)` → **Kept**

### From planarity_2_segmentation/transform.py → shared/utils/depth_normal.py
- `extract_zdepth(depth)` → **Kept**

## Visualization Functions

### From planarity_2_segmentation/visualize.py → shared/utils/visualization.py
- `visualize_top_components_v1(...)` → **Kept**
- Color generation functions → **Kept**

### From planarity_2_segmentation/visualize_seg.py → shared/utils/visualization.py
- Segmentation visualization functions → **Merged**

### From gt_gen/visualize_planes_v1.py → shared/utils/visualization.py
- PLY visualization functions → **Merged**
- Semantic color mapping → **Merged**

### From homography/visualize.py + homog_vis.py → shared/utils/visualization.py
- Homography visualization → **Merged**

## Label Utilities

### From homography/homog_utils.py → shared/utils/label_utils.py
- `keep_top_k_planes(labels, k)` → **Kept**
- `remap_labels(labels)` → **Kept**
- `fill_holes_inpaint(...)` → **Kept**
- `match_planes_by_overlap(...)` → **Kept**
- `map_array(arr, mapping)` → **Kept**

### From gt_gen/utils.py → shared/utils/label_utils.py
- `save_label_image(...)` → **Moved to io_utils.py**
- `save_label_image_sem(...)` → **Moved to io_utils.py**
- `remap_semantic(...)` → **Kept**

## Dataset Loaders

### From dataset/scannetpp/dataset_scannet_plane.py → shared/datasets/scannetpp.py
- `ScanNetPPPlaneDataset` class → **Kept** (cleaned paths)

### From dataset/hypersim/dataset_hypersim.py → shared/datasets/hypersim.py
- `HypersimPlanarityDataset` class → **Kept** (cleaned paths)

## GT Generation Functions

### From gt_gen/planes_from_mesh_labels_v1.py → gt_creation/scannetpp/plane_extraction.py
Kept core functions (cleaned and reorganized):
- `build_vertex_labels_from_segments(...)` → **Kept**
- `fit_plane_svd(P)` → **Kept**
- `fit_plane_irls(P, ...)` → **Kept**
- Region growing functions → **Kept**
- EM sweep functions → **Kept**
- Plane quality filtering → **Kept**
- Merge and split functions → **Kept**
- Main `run(args)` function → **Kept**

### From gt_gen/plane_gt_run.py → gt_creation/scannetpp/scene_runner.py
- `build_args(scene_id, config_path)` → **Kept**
- `cast_config_types(cfg)` → **Kept**
- Main execution logic → **Kept**

### From gt_gen/parse_scannetpp.py → gt_creation/scannetpp/parsers.py
- `load_semantic_id_to_name_list(...)` → **Kept**
- Dataset parsing functions → **Kept**

### From gt_gen/render_scene.py → gt_creation/scannetpp/rendering.py
- Plane raycasting logic → **Kept** (calls shared/rendering/)
- Scene rendering orchestration → **Kept**

### From gt_gen/render_scene_sem.py → gt_creation/scannetpp/rendering.py
- Semantic raycasting logic → **Merged**

### From gt_gen/render_scene_depth_h5.py → gt_creation/scannetpp/rendering.py
- Depth rendering logic → **Merged**

### From gt_gen/gen_video2.py → gt_creation/scannetpp/video_gen.py
- Video generation → **Kept**

## Hypersim GT Functions

### From gt_gen/hypersim/planes_from_mesh_hypersim_v1.py → gt_creation/hypersim/plane_extraction.py
- Hypersim-specific plane extraction → **Kept**

### From gt_gen/hypersim/render_scene_h5_v2.py → gt_creation/hypersim/rendering.py
- `compute_intrinsics_from_proj(...)` → **Kept**
- Hypersim rendering logic → **Kept**

### From gt_gen/hypersim/hypersim_mesh.py → gt_creation/hypersim/mesh_processing.py
- Hypersim mesh utilities → **Kept**

### From gt_gen/hypersim/plane_gt_run_hypersim_v1.py → gt_creation/hypersim/scene_runner.py
- Scene runner for Hypersim → **Kept**

## Inference Functions

### From monocular/moge/inference.py → inference/planarity/moge_inference.py
- `MoGePlanarityInference` class → **Kept** (cleaned)
- `__init__(model_path, ...)` → **Kept**
- `predict(rgb_path, ...)` → **Kept**
- `predict_batch(rgb_paths, ...)` → **Kept**
- `visualize_prediction(...)` → **Kept**

### From planarity_2_segmentation/plan2seg_pred_moge.py → inference/segmentation/predict.py
- MoGe-based segmentation pipeline → **Kept** (cleaned)
- Consolidated from multiple pred variants

### From planarity_2_segmentation/planeseg_scannet.py → inference/segmentation/scannetpp_runner.py
- ScanNet++ inference runner → **Kept**

## Evaluation Functions

### From plane_fitting/eval.py → evaluation/quantitative/evaluator.py
- `evaluate_planarity(...)` → **Kept**
- `segmentation_covering(...)` → **Kept**
- `get_plane_seg_baseline_from_h5(...)` → **Kept**
- Core evaluation loop → **Kept**

### From plane_fitting/eval.py → evaluation/quantitative/metrics.py
- Metric computation functions → **Extracted and organized**

### From plane_fitting/eval_moge.py → evaluation/quantitative/method_wrappers.py
- `moge_planarity_infer(...)` → **Kept**

### From plane_fitting/eval_monoplane.py → evaluation/quantitative/method_wrappers.py
- Monoplane inference wrapper → **Kept**

### From plane_fitting/eval_run.py → evaluation/run_evaluation.py
- Main evaluation runner → **Kept** (cleaned)
- Method dispatch logic → **Kept**

### From evaluation/qualitative/qual_compare_baseline_moge.py → evaluation/qualitative/visualize_comparison.py
- Comparison visualization → **Kept** (consolidated from multiple variants)

### From evaluation/qualitative/planeseg_video_moge.py → evaluation/qualitative/video_generation.py
- Video generation for evaluation → **Kept**

## Postprocessing Functions

### From planarity_2_segmentation/postprocess.py → shared/segmentation/postprocess.py
- All postprocessing functions → **Kept**

## Evaluation Metrics

### From plane_fitting/eval.py
- `labelmap_to_tensor_masks(labelmap, ignore_label)` → Moved to metrics.py
- `evaluateMasksTensor(predMasks, gtMasks, valid_mask)` → Moved to metrics.py
- `merge_plane_masks(seg_pred)` → Moved to metrics.py
- Variation of Information, Rand Index → Kept in metrics.py

## Function Import Changes

### Old Import Pattern:
```python
from plan2seg import compute_vectorized_planar_segments_v1
from planefit import fit_planes_per_label_v1
from process_remi import depth_to_normal_remi
```

### New Import Pattern:
```python
from shared.segmentation.plan2seg import compute_vectorized_planar_segments_v1
from shared.plane_fitting.planefit import fit_planes_per_label_v1
from shared.utils.depth_normal import depth_to_normal_remi
```

## Notes
- All functions kept their signatures unless cleaning was necessary
- Hardcoded paths removed from all functions
- Functions now use relative imports within clean_structure/
- Type hints added where missing
- Docstrings added to all public functions
