# Planar Surface Detection and Segmentation

Planar surface detection and segmentation from single RGB images using a 4-head MoGe-2
backbone (planarity + metric depth + normal + mask), GPU-accelerated region growing, and an
H5-based evaluation suite. Includes the full ground-truth generation pipeline (semantic mesh →
2D plane labels) for ScanNet++, Hypersim, SYNTHIA, and VKITTI2.

## Pipeline overview

```
RGB → MoGe 4-head model → planarity, metric depth, normals, mask     (pxwplanar/inference/planarity/)
        ↓
   8-connected region growing on planarity + normal + depth          (pxwplanar/shared/segmentation/)
        ↓
   per-segment RANSAC plane fitting                                  (pxwplanar/shared/plane_fitting/)
        ↓
   2D metrics (SC, RI, VOI) + 3D precision/recall @ thresholds       (pxwplanar/evaluation/quantitative/)
```

## Repository structure

```
├── pxwplanar/            # The Python package (pip-installed editable)
│   ├── shared/           #   Core library used by everything else
│   │   ├── plane_fitting/    # RANSAC fitting, metrics, projection
│   │   ├── segmentation/     # Region growing, merging, postprocessing
│   │   ├── rendering/        # Open3D mesh rendering and raycasting
│   │   ├── datasets/         # Loaders: ScanNet++, Hypersim, NYU-v2, 7-Scenes, SYNTHIA, VKITTI2
│   │   ├── outdoor/          # Outdoor-specific plane fitting
│   │   └── utils/            # Depth/normal processing, visualization, labels, I/O
│   ├── gt_creation/      #   Semantic mesh → 2D plane-label GT (HDF5)
│   │   ├── scannetpp/  hypersim/  synthia/  vkitti2/
│   │   ├── configs/          # YAML configs per dataset
│   │   └── scripts/          # SLURM batch scripts (README in each scripts/ dir)
│   ├── inference/        #   RGB → signals → plane labels
│   │   ├── planarity/        # MoGe wrapper, signal export, segmentation from H5
│   │   └── segmentation/     # On-the-fly prediction pipeline
│   ├── evaluation/       #   Metrics + visualization
│   │   ├── quantitative/     # evaluate_all_baselines.py + outdoor variants
│   │   └── qualitative/      # Comparison videos, 3D visualization
│   └── paths.py          #   ALL dataset/checkpoint/output paths — edit before running anything
├── splits/               # Train/val/test scene lists per dataset
├── env/                  # Conda environment + setup script
└── MoGe/                 # Git submodule: MoGe fork with 4-head training code
```

## Environment setup

```bash
bash env/create_env.sh    # conda env `pxwplanar` from env/environment.yml,
                          # `pip install -e .`, MoGe submodule init
conda activate pxwplanar
```

**Before running anything**: edit `pxwplanar/paths.py`. Every dataset root, checkpoint path,
and output root resolves through it.

## Main pipeline (H5-based, three stages)

```bash
# 1. RGB → planarity/depth/normal signals (one moge_signals.h5 per scene)
python pxwplanar/inference/planarity/save_moge_signals_planarity.py \
    --dataset scannetpp --scenes splits/scannetpp/test.txt \
    --model_path <checkpoint.pt> --output_root <signals_root> \
    --frame_step 25 --batch_size 8 --num_tokens 1600
# --dataset also accepts nyuv2 / sevenscenes (ZeroPlane "_d2" NPZ) and hypersim

# 2. Signals → plane labels (one planes.h5 per scene); shardable across SLURM jobs
python pxwplanar/inference/planarity/segment_signals_to_planes.py \
    --input_root <signals_root> --output_root <planes_root> \
    [--part_id 0 --num_parts 15]

# 3. Evaluate methods against GT
python pxwplanar/evaluation/quantitative/evaluate_all_baselines.py --methods gt ours --max-scenes 5
python pxwplanar/evaluation/quantitative/evaluate_all_baselines.py --aggregate-only   # summary tables
```

The method registry is the `METHODS` dict at the top of `evaluate_all_baselines.py` and ships
with exactly two methods: `gt` (upper bound from rendered GT labels) and `ours`
(experiment name `moge_ours_ep1` — planes from the pipeline above with the 4-head MoGe
checkpoint, read from `paths.ours_planes_root`). Add new baselines to the dict as documented
there. Outdoor variants: `evaluate_synthia_all_baselines.py`, `evaluate_vkitti2_all_baselines.py`.

### Canonical segmentation parameters

Used identically in `segment_signals_to_planes.py` and the benchmark — keep in sync:
**planarity > 0.3, normal threshold 5.0°, relative depth threshold 0.025, ≥8 matching neighbors.**

### Reproducibility

3D metrics use seeded RANSAC (`RANSAC_SEED = 0`) in `evaluate_all_baselines.py`
(`--ransac-seed -1` restores legacy non-deterministic behavior). Keep the seed fixed when
comparing methods.

### Label conventions

- Ours / GT: label **0 = non-planar**. ZeroPlane uses label 20 (remapped via `nonplanar_label`
  in `METHODS`).
- Always resize label maps with `cv2.INTER_NEAREST`, never linear.

## Ground truth generation

From `pxwplanar/gt_creation/<dataset>/` (scannetpp, hypersim, synthia, vkitti2):

```bash
python scene_runner.py <scene_id> --config ../configs/<dataset>_default.yml
```

Rendering to 2D labels (ScanNet++): `render_scene.py` raycasts the extracted plane mesh into
every Nth iPhone frame and writes the per-scene `rendered.h5` consumed by training and
evaluation; `rendering.py` produces PNG label previews (`<scene>/rendered/`).

Batch SLURM scripts (arguments documented in `pxwplanar/gt_creation/scripts/README.md`):

```bash
bash pxwplanar/gt_creation/scripts/scannetpp_plane_extraction.sh <scene_list> [config]
bash pxwplanar/gt_creation/scripts/scannetpp_render_planes.sh <scene_list>   # -> rendered.h5 per scene
# hypersim_*, synthia_*, vkitti2_* analogues; hypersim_raycast_depth.sh for raycast z-depth
```

Algorithm: label-strict region growing on mesh faces → EM sweep with quality gates → IRLS
plane fitting → merge/split → quality filtering (8 geometric checks) → raycast to 2D (HDF5).

Key YAML parameters: `rg_theta_deg`/`rg_dist_m` (region growing), `min_faces_patch`/
`min_area_patch` (minimum plane size), `inlier_frac_min`/`p95_final_max` (quality gates),
`merge_theta_deg`/`merge_dist_m` (merging).

## Training

Entry points live in the `MoGe/` submodule; run from the repo root with `pxwplanar` active:

```bash
python MoGe/train_moge_4heads_planarity_scannetpp.py   # ScanNet++ (rendered plane GT)
python MoGe/train_moge_4heads_planarity_hypersim.py    # Hypersim
python MoGe/train_moge_4heads_planarity_mixed.py       # Mixed datasets
```

Training consumes the rendered plane-GT H5s from `pxwplanar/gt_creation/` plus the split lists in
`splits/`; dataset roots resolve through `pxwplanar/paths.py` or the trainers' `--dataset_dir` arguments.

## Evaluation metrics

- **2D segmentation**: Segmentation Covering (SC), Rand Index (RI), Variation of Information (VOI)
- **3D geometry**: precision/recall @ distance thresholds (RANSAC-fitted planes vs GT)
- **Depth**: REL, RMSE, δ < 1.25ⁿ — **Normals**: mean angle error, <11.25°/22.5°/30°

## HDF5 schemas

`moge_signals.h5` (per scene): `frame_ids (N,)`, `planarity (N,H,W) f16`,
`normal (N,H,W,3) f16`, `depth_metric (N,H,W) f16`, `mask (N,H,W) u8`,
`intrinsics (N,3,3) f32`, `metric_scale (N,) f32`.

`planes.h5` (per scene): `planes (N,H,W) uint16` (0 = non-planar), `frame_ids`.

Always read chunked (`f['planes'][idx, :]`) — never load full arrays.

## SLURM

Shell scripts carry commented `#SBATCH` directives (typical: `--time=8:00:00
--cpus-per-task=4 --mem-per-cpu=8G --gpus=1`). Uncomment, `mkdir -p logs`, `sbatch script.sh`.
Python-level sharding uses `--part_id / --num_parts` (contiguous slices of the sorted scene
list; all parts share one output root safely).

## Dependencies

`numpy`, `opencv-python`, `torch`, `pandas`, `open3d`, `trimesh`, `plyfile`, `h5py`,
`pyyaml`, `tqdm`, `natsort`, `matplotlib`, `scipy`, `cc3d` — pinned in `env/environment.yml`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{pixelwiseplanarity2026,
  title     = {Pixel-wise Planarity for High-Precision Monocular Plane Segmentation},
  author    = {Yavuz, Ahmetcan and Ozkan, Alpay and Pautrat, R{\'e}mi and Liu, Shaohui and Pollefeys, Marc},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The `MoGe/` submodule is a fork of Microsoft's
[MoGe](https://github.com/microsoft/MoGe) (MIT licensed).
