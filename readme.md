# Planar Surface Detection and Segmentation

Clean, restructured codebase for planar surface detection and segmentation from RGB images.

## Structure

```
clean_structure/
├── shared/               # Core utilities used across all modules
│   ├── plane_fitting/   # RANSAC plane fitting, metrics, projection
│   ├── rendering/       # Mesh rendering and raycasting (Open3D)
│   ├── segmentation/    # Planar segmentation algorithms
│   ├── datasets/        # PyTorch dataset loaders (ScanNet++, Hypersim)
│   └── utils/           # Depth/normal processing, visualization, I/O
├── gt_creation/         # Ground truth generation from 3D meshes
│   ├── scannetpp/       # ScanNet++ plane extraction
│   ├── hypersim/        # Hypersim plane extraction
│   ├── configs/         # YAML configuration files
│   └── scripts/         # Shell scripts for batch processing
├── inference/           # Inference pipelines
│   ├── planarity/       # Planarity prediction (MoGe)
│   └── segmentation/    # Plane segmentation from predictions
├── evaluation/          # Evaluation tools
│   ├── quantitative/    # Metrics computation
│   └── qualitative/     # Visualization and comparisons
├── exploration/         # Jupyter notebooks
│   ├── hypersim/        # Hypersim experiments
│   └── scannetpp/       # ScanNet++ experiments
├── MoGe/                # External MoGe model integration
└── notes/               # Restructuring documentation
```

## Datasets

### ScanNet++
- **Mesh-based GT generation** from semantic meshes
- **Rendered labels**: Planes, semantics, depth → HDF5
- **iPhone RGB images** with camera poses

### Hypersim
- **Synthetic photorealistic scenes**
- **HDF5 format**: RGB (HDR), depth, normals, semantics
- **Plane extraction** from mesh geometry

## Quick Start

### Shared Modules

All core functionality is in `shared/`:

```python
# Plane fitting
from shared.plane_fitting import backproject_v1, fit_planes_per_label_v1

# Rendering
from shared.rendering import render_rgb_depth, raycast_semantic

# Segmentation
from shared.segmentation import compute_vectorized_planar_segments_v1

# Utilities
from shared.utils import depth_to_normal_remi, extract_zdepth

# Datasets
from shared.datasets import ScanNetPPPlaneDataset, HypersimPlanarityDataset
```

### Ground Truth Generation

**ScanNet++ (from `gt_creation/scannetpp/`):**
```bash
# Single scene
python scene_runner.py <scene_id> --config ../configs/scannetpp_default.yml

# Batch processing
bash scripts/scannetpp_pipeline.sh
```

**Hypersim (from `gt_creation/hypersim/`):**
```bash
# Single scene
python scene_runner.py <scene_id> --config ../configs/hypersim_default.yml

# Batch processing
bash scripts/hypersim_pipeline.sh
```

### Inference

```python
from inference.planarity import MoGePlanarityInference
from inference.segmentation import predict_plane_segmentation

# Planarity prediction
model = MoGePlanarityInference(model_path="path/to/moge.pth")
planarity = model.predict(rgb_image)

# Plane segmentation
segments = predict_plane_segmentation(
    planarity_mask=planarity,
    depth=depth_image,
    normal=normal_image
)
```

### Evaluation

```bash
# Run full evaluation
python evaluation/run_evaluation.py --method moge --split test

# Quantitative metrics
python evaluation/quantitative/evaluator.py

# Qualitative visualization
python evaluation/qualitative/visualize_comparison.py
```

## Key Algorithms

### Plane Fitting (RANSAC + LS Refinement)
1. Backproject depth to 3D points
2. RANSAC plane fitting per segment
3. Least-squares refinement on inliers
4. Quality filtering by inlier ratio

### Planar Segmentation
1. Binary planarity prediction
2. Depth → surface normals
3. 8-connected region growing with normal/depth thresholds
4. Connected component labeling
5. Small component removal

### GT Generation
1. **Region Growing**: Label-strict growth on mesh faces
2. **EM Sweep**: Expand planes with quality gates
3. **IRLS Fitting**: Robust plane parameter estimation
4. **Merge & Split**: Consolidate compatible planes
5. **Quality Filtering**: Multiple geometric checks
6. **Raycasting**: Project plane labels to 2D images

## Configuration

GT generation configured via YAML files in `gt_creation/configs/`:

**Key parameters:**
- `rg_theta_deg`: Region growing angular threshold (degrees)
- `rg_dist_m`: Region growing distance threshold (meters)
- `min_faces_patch`: Minimum faces per plane
- `inlier_frac_min`: Minimum inlier fraction
- `merge_theta_deg` / `merge_dist_m`: Plane merging thresholds

## Dependencies

- **Core**: `numpy`, `opencv-python`, `torch`, `pandas`
- **3D**: `open3d`, `trimesh`, `plyfile`
- **I/O**: `h5py`, `pyyaml`
- **Utilities**: `tqdm`, `natsort`, `matplotlib`
- **Segmentation**: `cc3d`, `scipy`

## Documentation

- `notes/RESTRUCTURING_LOG.md` - File mapping (old → new)
- `notes/REMOVED_FILES.md` - What was removed and why
- `notes/VERSION_DECISIONS.md` - Which file versions were kept
- `notes/CONSOLIDATED_FUNCTIONS.md` - Function-level mappings

## Status

 **Completed Modules:**
- shared/plane_fitting (RANSAC fitting, metrics, projection)
- shared/rendering (mesh rendering, raycasting)
- shared/segmentation (plan2seg algorithms)
- shared/utils (depth/normal, visualization, labels, I/O)
- shared/datasets (ScanNet++, Hypersim loaders)

 **In Progress:**
- gt_creation/ (plane extraction pipelines)
- inference/ (MoGe integration, prediction)
- evaluation/ (metrics, visualization)
- exploration/ (notebook organization)
- training/ (carry MoGe/train_moge_4heads_planarity_fixed.py to appropriate place)

## Notes

- All hardcoded paths have been removed
- Imports updated for new structure
- Comprehensive docstrings added
- Type hints added where appropriate
- Only latest/best versions of algorithms kept
- Extensive documentation in `notes/`
