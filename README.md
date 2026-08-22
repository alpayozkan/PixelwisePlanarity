<h1 align="center">Pixel-wise Planarity for High-Precision<br>Monocular Plane Segmentation</h1>

<p align="center">
  Ahmetcan Yavuz &middot; Alpay Ozkan &middot; R&eacute;mi Pautrat &middot; Shaohui Liu &middot; Marc Pollefeys
</p>

<p align="center"><b>ECCV 2026</b></p>

<p align="center">
  <img src="demo/assets/demo.gif" alt="Demo: RGB | depth | normal | planarity | planes" width="100%">
</p>
<p align="center">
  <em>From a single RGB image, a 4-head MoGe-2 backbone predicts planarity, metric depth and
  normals; region growing turns them into a plane segmentation.<br>
  Panels: RGB &nbsp;|&nbsp; depth &nbsp;|&nbsp; normal &nbsp;|&nbsp; planarity &nbsp;|&nbsp; planes.</em>
</p>

Planar surface detection and segmentation from single RGB images using a 4-head MoGe-2
backbone (planarity + metric depth + normal + mask), GPU-accelerated region growing, and an
H5-based evaluation suite. Includes the full ground-truth generation pipeline (semantic mesh →
2D plane labels) for ScanNet++, Hypersim, SYNTHIA, and VKITTI2.

## Setup

```bash
bash env/create_env.sh    # conda env `pxwplanar` from env/environment.yml,
                          # `pip install -e .`, MoGe submodule init
conda activate pxwplanar
```

**Before running anything**: configure the paths in `pxwplanar/paths.py`. Every dataset root,
checkpoint path, and output root resolves through it from a single data root — either set the
`PXWPLANAR_DATA_ROOT` environment variable (default: `<repo>/data`), or create
`pxwplanar/paths_local.py` (gitignored) overriding any subset of the variables.

### Pretrained checkpoint

Download the 4-head planarity checkpoint (`model_epoch1.pt`) from
[Google Drive](https://drive.google.com/drive/folders/1SqsAtNbKMO6YPAQPFuzwn-Y2hhEufF83?usp=sharing)
and place it at the location configured in `pxwplanar/paths.py`
(`planarity_model_path`, default:
`<data_root>/checkpoints/moge_HIRES_4datasets/model_epoch1.pt`) — or pass
`--model_path` to the scripts directly. The MoGe-2 base weights
(`Ruicheng/moge-2-vitl-normal`) are pulled from HuggingFace automatically.

## Demo

```bash
python demo/run_demo.py        # runs on the example frames in demo/inputs/
python demo/make_gif.py        # assemble the montages into demo/assets/demo.gif
```

For each input image the script runs the full pipeline (MoGe 4-head inference +
plane segmentation with the canonical parameters) and writes
`demo/outputs/<frame>/{depth,normal,planarity,planeseg}.png` plus a combined
montage (`combined.png` — RGB | depth | normal | planarity | planes,
top-20 planes shown). The checkpoint defaults from `pxwplanar/paths.py`
(`planarity_model_path`); pass `--model_path` to override. MoGe base weights
are pulled from HuggingFace into the standard cache (`~/.cache/huggingface`;
override via `HF_HOME`).

Minimal API usage:

```python
from pxwplanar.inference.planarity.moge_inference import MoGePlanarityInference
from pxwplanar.shared.segmentation import compute_vectorized_planar_segments
import numpy as np

model = MoGePlanarityInference("<checkpoint.pt>")
res = model.predict_metric("image.jpg", num_tokens=1600, return_all_heads=True)
# res: planarity_probability, depth (m), normal, points, mask, intrinsics

labels, n = compute_vectorized_planar_segments(
    (res["planarity_probability"] > 0.3).astype(np.int16),
    res["normal"], res["depth"],
    np.deg2rad(5.0), 0.025, neighbor_match_count_thresh=8)
```

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

<details>
<summary><b>Repository structure</b></summary>

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
│   └── paths.py          #   ALL dataset/checkpoint/output paths — configure before running
├── demo/                 # Example frames + run_demo.py + make_gif.py
├── splits/               # Train/val/test scene lists per dataset
├── env/                  # Conda environment + setup script
└── MoGe/                 # Git submodule: MoGe fork with 4-head training code
```

</details>

## Evaluation

The benchmark runs in three H5-based stages:

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

Metrics: **2D segmentation** — Segmentation Covering (SC), Rand Index (RI), Variation of
Information (VOI); **3D geometry** — precision/recall @ distance thresholds (RANSAC-fitted
planes vs GT); **depth** — REL, RMSE, δ < 1.25ⁿ; **normals** — mean angle error,
<11.25°/22.5°/30°.

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

Semantic mesh → 2D plane-label GT: label-strict region growing on mesh faces → EM sweep with
quality gates → IRLS plane fitting → merge/split → quality filtering (8 geometric checks) →
raycast to 2D (HDF5).

<details>
<summary><b>Commands and parameters</b></summary>

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

Key YAML parameters: `rg_theta_deg`/`rg_dist_m` (region growing), `min_faces_patch`/
`min_area_patch` (minimum plane size), `inlier_frac_min`/`p95_final_max` (quality gates),
`merge_theta_deg`/`merge_dist_m` (merging).

</details>

## Training

Entry points live in the `MoGe/` submodule; run from the repo root with `pxwplanar` active:

```bash
python MoGe/train_moge_4heads_planarity_scannetpp.py   # ScanNet++ (rendered plane GT)
python MoGe/train_moge_4heads_planarity_hypersim.py \
    --hypersim_root <raw_hypersim> --hypersim_planes <plane_gt> --hypersim_split_csv <csv>
python MoGe/train_moge_4heads_planarity_mixed.py \
    --hypersim_root <raw_hypersim> --hypersim_planes <plane_gt> --hypersim_split_csv <csv>
```

Training consumes the rendered plane-GT H5s from `pxwplanar/gt_creation/` plus the split lists in
`splits/`. ScanNet++ roots default from `pxwplanar/paths.py` (override with `--rgb_root` /
`--plane_gt_root`); Hypersim roots are passed explicitly.

<details>
<summary><b>HDF5 schemas</b></summary>

`moge_signals.h5` (per scene): `frame_ids (N,)`, `planarity (N,H,W) f16`,
`normal (N,H,W,3) f16`, `depth_metric (N,H,W) f16`, `mask (N,H,W) u8`,
`intrinsics (N,3,3) f32`, `metric_scale (N,) f32`.

`planes.h5` (per scene): `planes (N,H,W) uint16` (0 = non-planar), `frame_ids`.

Always read chunked (`f['planes'][idx, :]`) — never load full arrays.

</details>

<details>
<summary><b>SLURM usage</b></summary>

Shell scripts carry commented `#SBATCH` directives (typical: `--time=8:00:00
--cpus-per-task=4 --mem-per-cpu=8G --gpus=1`). Uncomment, `mkdir -p logs`, `sbatch script.sh`.
Python-level sharding uses `--part_id / --num_parts` (contiguous slices of the sorted scene
list; all parts share one output root safely).

</details>

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
