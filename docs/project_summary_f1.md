# Pixelwise Planarity Prediction and Plane Segmentation

## Motivation

Existing plane segmentation methods (e.g., PlaneRCNN, ZeroPlane) rely on end-to-end architectures that jointly predict plane masks and parameters. These models are expensive, dataset-specific, and struggle to generalize across indoor/outdoor domains. We take a different approach: decouple planarity estimation from segmentation, leveraging a geometric foundation model to produce high-quality per-pixel signals, then apply a fast geometric grouping step to recover plane instances.

## Our Approach

We predict per-pixel planarity from a single RGB image and segment planar surfaces using a two-stage pipeline:

1. **Planarity prediction**: We extend MoGe (Monocular Geometry Estimation), a DINOv2 ViT-Large backbone, with a 4th output head that predicts per-pixel planarity probability alongside depth, surface normals, and 3D points. The planarity head is trained with BCE/Dice/Focal losses on automatically generated ground truth from 3D meshes.
<!-- Training uses a two-phase schedule: 2 epochs with a frozen backbone (head initialization), then 28 epochs of full fine-tuning with AdamW and cosine LR decay. -->

2. **Plane segmentation**: A GPU-accelerated vectorized region growing algorithm groups planar pixels into plane instances. It uses Sobel-based edge detection on normals and depth, a 5x5 neighborhood connectivity check, and `cc3d` connected components. The full segmentation pipeline runs in ~30ms per frame (mean), with MoGe inference taking ~126ms (mean), making segmentation only ~19% of total compute.

## Contributions

- **Foundation model adaptation**: Repurposing a pretrained depth/normal estimation model for planarity prediction, avoiding the cost of training a full segmentation network from scratch.
- **Automatic GT generation pipeline**: A multi-stage pipeline that extracts high-quality plane labels from 3D meshes via label-strict region growing, EM sweep, IRLS robust fitting, quality filtering, and raycasting to 2D. Applied to ScanNet++ and Hypersim.
- **3D geometric evaluation protocol**: Rather than evaluating only 2D mask overlap, we backproject predicted segments to 3D, fit planes via RANSAC at the evaluation threshold, and measure:
  - **Precision** = inlier points / total predicted plane points (how geometrically consistent are the predicted planes?)
  - **Recall** = inlier points / all scene points (how much of the scene is covered by valid planes?)
  - Evaluated at multiple distance thresholds $\tau$ (1mm, 5mm, 10mm for indoor; 2cm, 5cm, 10cm for outdoor).

## Results

Evaluated across 4 datasets spanning indoor/outdoor and real/synthetic domains (~36K frames total).

### ScanNet++ (indoor real)

| Method | Scenes | Frames | RI | VOI | SC | P@0.1cm | R@0.1cm | F1@0.1cm | P@0.5cm | R@0.5cm | F1@0.5cm | P@1.0cm | R@1.0cm | F1@1.0cm |
|--------|-------:|-------:|---:|----:|---:|--------:|--------:|---------:|--------:|--------:|---------:|--------:|--------:|---------:|
| GT (upper bound) | 42 | 14439 | 1.000 | 0.000 | 1.000 | 0.603 | 0.440 | 0.509 | 0.940 | 0.682 | 0.790 | 0.968 | 0.701 | 0.813 |
| MoGe (Ours) | 42 | 14439 | 0.807 | 1.740 | 0.647 | 0.660 | 0.390 | 0.490 | 0.903 | 0.541 | 0.677 | 0.935 | 0.560 | 0.700 |
| ZeroPlane (finetuned) | 42 | 14439 | 0.845 | 1.615 | 0.690 | 0.478 | 0.356 | 0.408 | 0.830 | 0.615 | 0.707 | 0.891 | 0.657 | 0.756 |
| ZeroPlane (released) | 42 | 14439 | 0.813 | 2.044 | 0.577 | 0.283 | 0.258 | 0.270 | 0.588 | 0.536 | 0.561 | 0.704 | 0.641 | 0.671 |

### Hypersim (indoor synthetic)

| Method | Scenes | Frames | RI | VOI | SC | P@0.1cm | R@0.1cm | F1@0.1cm | P@0.5cm | R@0.5cm | F1@0.5cm | P@1.0cm | R@1.0cm | F1@1.0cm |
|--------|-------:|-------:|---:|----:|---:|--------:|--------:|---------:|--------:|--------:|---------:|--------:|--------:|---------:|
| GT (upper bound) | 68 | 10646 | 1.000 | 0.000 | 1.000 | 0.997 | 0.786 | 0.879 | 0.999 | 0.787 | 0.880 | 1.000 | 0.787 | 0.881 |
| MoGe (Ours) | 68 | 10646 | 0.756 | 2.815 | 0.504 | 0.793 | 0.433 | 0.560 | 0.831 | 0.452 | 0.586 | 0.849 | 0.461 | 0.598 |
| ZeroPlane (finetuned) | 68 | 10646 | 0.841 | 2.170 | 0.620 | 0.660 | 0.464 | 0.545 | 0.704 | 0.493 | 0.580 | 0.742 | 0.516 | 0.609 |
| ZeroPlane (released) | 68 | 10646 | 0.863 | 2.491 | 0.520 | 0.354 | 0.311 | 0.331 | 0.391 | 0.343 | 0.365 | 0.432 | 0.379 | 0.404 |

### Synthia (outdoor synthetic)

| Method | Scenes | Frames | RI | VOI | SC | P@2.0cm | R@2.0cm | F1@2.0cm | P@5.0cm | R@5.0cm | F1@5.0cm | P@10.0cm | R@10.0cm | F1@10.0cm |
|--------|-------:|-------:|---:|----:|---:|--------:|--------:|---------:|--------:|--------:|---------:|---------:|---------:|----------:|
| GT (upper bound) | 89 | 8426 | 1.000 | 0.000 | 1.000 | 0.782 | 0.390 | 0.520 | 0.819 | 0.411 | 0.547 | 0.848 | 0.430 | 0.571 |
| MoGe (Ours) | 89 | 8426 | 0.760 | 1.705 | 0.640 | 0.958 | 0.320 | 0.480 | 0.971 | 0.325 | 0.487 | 0.977 | 0.327 | 0.490 |
| ZeroPlane (finetuned) | 89 | 8426 | 0.822 | 1.086 | 0.741 | 0.485 | 0.287 | 0.361 | 0.522 | 0.305 | 0.385 | 0.558 | 0.323 | 0.409 |
| ZeroPlane (released) | 89 | 8426 | 0.793 | 1.722 | 0.646 | 0.445 | 0.320 | 0.372 | 0.499 | 0.358 | 0.417 | 0.577 | 0.413 | 0.481 |

### VKITTI2 (outdoor synthetic)

| Method | Scenes | Frames | RI | VOI | SC | P@2.0cm | R@2.0cm | F1@2.0cm | P@5.0cm | R@5.0cm | F1@5.0cm | P@10.0cm | R@10.0cm | F1@10.0cm |
|--------|-------:|-------:|---:|----:|---:|--------:|--------:|---------:|--------:|--------:|---------:|---------:|---------:|----------:|
| GT (upper bound) | 20 | 2360 | 1.000 | 0.000 | 1.000 | 0.997 | 0.199 | 0.332 | 0.999 | 0.200 | 0.333 | 1.000 | 0.200 | 0.333 |
| MoGe (Ours) | 20 | 2360 | 0.790 | 0.937 | 0.808 | 0.967 | 0.242 | 0.387 | 0.989 | 0.248 | 0.397 | 0.998 | 0.250 | 0.400 |
| ZeroPlane (finetuned) | 20 | 2360 | 0.765 | 0.784 | 0.787 | 0.807 | 0.202 | 0.323 | 0.838 | 0.209 | 0.335 | 0.864 | 0.214 | 0.343 |
| ZeroPlane (released) | 20 | 2360 | 0.665 | 1.332 | 0.692 | 0.659 | 0.271 | 0.384 | 0.685 | 0.284 | 0.402 | 0.745 | 0.308 | 0.436 |

### Key Observations

- **3D precision**: Our method consistently achieves the highest precision across all datasets and thresholds.
<!-- - **Indoor vs outdoor**: Indoor datasets use tighter thresholds (0.1--1.0cm) reflecting smaller depth ranges, while outdoor datasets use looser thresholds (2--10cm) due to larger scenes and depth-dependent noise. -->
- **Segmentation metrics**: ZeroPlane (finetuned) achieves higher SC and RI on most datasets, reflecting better 2D boundary alignment from its mask-based architecture. Our method trades some 2D agreement for superior 3D geometric quality.
- **Generalization**: Our approach maintains strong performance across both indoor (ScanNet++, Hypersim) and outdoor (Synthia, VKITTI2) domains using a single model, whereas ZeroPlane was designed primarily for indoor scenes.

## Model Comparison

Measured on ScanNet++ (14,439 frames), single GPU.

| Method | Backbone | Params (M) | Mean (ms/frame) | Speedup |
|--------|----------|----------:|----------------:|--------:|
| MoGe (Ours) | DINOv2 ViT-L/14 | 335.6 | 155.7 | **2.2x** |
| ZeroPlane (finetuned) | DUSt3R ViT-L | ~600 | 338.3 | 1.0x |
