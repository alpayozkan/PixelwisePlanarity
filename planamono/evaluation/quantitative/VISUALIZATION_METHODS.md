# Visualization Methods Reference

## ScanNet++

**Experiment version:** `v5`
**Inference root:** `/cluster/scratch/aoezkan/planeseg/scannetpp/inference/`
**GT root:** `/cluster/scratch/aoezkan/planeseg/dataset/scannetpp/` (uses `rendered.h5` per scene)
**Output root:** `/cluster/scratch/aoezkan/planeseg/scannetpp/visualizations/inliers/`

| Method key | H5 folder | Display name | Description |
|---|---|---|---|
| `gt` | *(from dataset `rendered.h5`)* | GT (upper bound) | Ground truth plane labels, serves as upper bound and reference for diff row |
| `ours` | `moge_ours_v2_h5` | Ours (full) | Full pipeline: MoGe planarity prediction + our segmentation, trained on ScanNet++ |
| `zeroplane` | `zeroplane_h5` | ZeroPlane | ZeroPlane baseline (label 20 = non-planar, auto-remapped to 0) |
| `zeroplane_mixed` | `zeroplane_mixed_h5` | ZeroPlane (mixed) | ZeroPlane trained on mixed datasets |
| `zeroplane_mixed_dust3r` | `zeroplane_mixed_dust3r_h5` | ZeroPlane (mixed+dust3r) | ZeroPlane trained on mixed datasets with Dust3R encoder |
| `moge_mixed_bce` | `moge_mixed_bce_h5` | MoGe Mixed BCE | MoGe planarity with BCE loss, trained on mixed ScanNet++ + Hypersim |
| `gtseg` | `gtseg_v1_h5` | GT Seg (upper bound) | GT segmentation + our predicted depth (ablation: isolates depth quality) |
| `gtplanarity_ourseg` | `gtplanarity_ourseg_h5` | GT Planarity + Our Seg | GT planarity map + our segmentation (ablation: isolates segmentation quality) |
| `ourplanarity_gtseg` | `ourplanarity_gtseg_h5` | Our Planarity + GT Seg | Our planarity prediction + GT segmentation (ablation: isolates planarity quality) |

### Available inference folders on disk

```
moge_ours_v2_h5              ← used by "ours"
moge_mixed_bce_h5            ← used by "moge_mixed_bce"
zeroplane_h5                 ← used by "zeroplane"
zeroplane_mixed_h5           ← used by "zeroplane_mixed"
zeroplane_mixed_dust3r_h5    ← used by "zeroplane_mixed_dust3r"
gtseg_v1_h5                  ← used by "gtseg"
gtplanarity_ourseg_h5        ← used by "gtplanarity_ourseg"
ourplanarity_gtseg_h5        ← used by "ourplanarity_gtseg"
moge_ours_h5                 (not used — older version)
moge_ours_h5_only            (not used)
moge_ours_merged_h5          (not used)
moge_ours_v1_h5              (not used — older version)
gtseg_h5                     (not used — older version)
gtplanarity_ourseg_v1_h5     (not used — older version)
ourplanarity_gtseg_v1_h5     (not used — older version)
planercnn_correct_format      (not used)
```

---

## Hypersim

**Experiment version:** `v2`
**Inference root:** `/cluster/scratch/aoezkan/planeseg/hypersim/inference/`
**GT root:** *(no separate GT H5 — GT plane labels come from dataset `__getitem__` directly)*
**Output root:** `/cluster/scratch/aoezkan/planeseg/hypersim/visualizations/inliers/`

| Method key | H5 folder | Display name | Description |
|---|---|---|---|
| `gt` | *(from dataset directly)* | GT (upper bound) | Ground truth plane labels loaded from `rendered_planes_cam_XX.h5` via dataset loader |
| `moge_ours` | `moge_ours_h5` | MoGe Ours (ScanNet++) | MoGe planarity + our segmentation, trained on ScanNet++ only (cross-dataset transfer) |
| `moge_mixed_bce` | `moge_mixed_bce_h5` | MoGe Mixed BCE | MoGe planarity with BCE loss, trained on mixed ScanNet++ + Hypersim |
| `zeroplane_mixed_dust3r` | `zeroplane_mixed_dust3r_h5` | ZeroPlane (Mixed Dust3R) | ZeroPlane trained on mixed datasets with Dust3R encoder (label 20 = non-planar) |
| `zeroplane_mixed` | `zeroplane_mixed_h5` | ZeroPlane (Mixed) | ZeroPlane trained on mixed datasets (label 20 = non-planar) |

### Available inference folders on disk

```
moge_ours_h5                 ← used by "moge_ours"
moge_mixed_bce_h5            ← used by "moge_mixed_bce"
zeroplane_mixed_dust3r_h5    ← used by "zeroplane_mixed_dust3r"
zeroplane_mixed_h5           ← used by "zeroplane_mixed"
```

### Hypersim H5 structure

Hypersim uses **per-camera H5 files** (unlike ScanNet++ which uses one `planes.h5` per scene):
```
inference/{h5_folder}/{scene_id}/planes_cam_00.h5
inference/{h5_folder}/{scene_id}/planes_cam_01.h5
inference/{h5_folder}/{scene_id}/planes_cam_02.h5
```

---

## Evaluation Parameters (both datasets)

| Parameter | Value |
|---|---|
| Distance thresholds | 0.001, 0.005, 0.01 m (1mm, 5mm, 10mm) |
| RANSAC iterations | 200 |
| Inlier ratio gate | 0.9 |

## Commands

```bash
# ScanNet++ — all methods (10 columns)
python visualize_and_generate_pdf.py --dataset scannetpp --n-samples 20 --format both

# Hypersim — all methods (5 columns)
python visualize_and_generate_pdf.py --dataset hypersim --n-samples 20 --format both

# Subset of methods
python visualize_and_generate_pdf.py --dataset scannetpp --methods ours moge_mixed_bce zeroplane_mixed zeroplane_mixed_dust3r --n-samples 20 --format both
```
