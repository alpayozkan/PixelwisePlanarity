# Shell Scripts Reference

All shell scripts are organized in `scripts/` directories within each module.

## Directory Structure

```
clean_structure/
├── gt_creation/scripts/           # Ground truth generation
│   ├── scannetpp_plane_extraction.sh
│   ├── scannetpp_render_planes.sh
│   ├── scannetpp_video_gen.sh
│   ├── hypersim_plane_extraction.sh
│   ├── hypersim_render_planes.sh
│   └── batch_submit.sh
│
├── inference/scripts/             # Model inference
│   ├── run_planarity_inference.sh
│   └── run_segmentation.sh
│
├── evaluation/scripts/            # Evaluation
│   ├── run_evaluation.sh
│   ├── run_qualitative.sh
│   └── batch_evaluate.sh
│
└── run_tests.sh                   # Tests
```

---

## GT Creation Scripts

### ScanNet++

#### 1. Plane Extraction
Extract planes from meshes.
```bash
./scannetpp_plane_extraction.sh <scene_list> [config] [input_root] [output_root]
```

**Defaults:**
- `config`: `../configs/scannetpp_default.yml`
- `input_root`: `/path/to/scannetpp/data`
- `output_root`: `/path/to/output`

#### 2. Render Planes
Render extracted planes to PNG images.
```bash
./scannetpp_render_planes.sh <scene_list> [input_root] [plane_root] [output_root] [frame_skip]
```

**Defaults:**
- `input_root`: `/cluster/project/cvg/Shared_datasets/scannetpp_v2/data`
- `plane_root`: `/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt`
- `output_root`: `/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt`
- `frame_skip`: `25`

#### 3. Video Generation
Generate visualization videos from rendered planes.
```bash
./scannetpp_video_gen.sh <scene_list> [h5_root] [rgb_root] [output_root] [fps]
```

**Defaults:**
- `h5_root`: `/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt`
- `rgb_root`: `/cluster/project/cvg/Shared_datasets/scannet++/data`
- `output_root`: `/cluster/scratch/aoezkan/dataset/scannetpp/visual`
- `fps`: `5`

---

### Hypersim

#### 1. Plane Extraction
Extract planes from Hypersim meshes.
```bash
./hypersim_plane_extraction.sh <scene_list> [config] [input_root] [output_root]
```

#### 2. Render Planes
Render planes to HDF5 files.
```bash
./hypersim_render_planes.sh <scene_list> [input_root] [plane_root] [output_root] [frame_skip]
```

**Defaults:**
- `input_root`: `/cluster/scratch/ayavuz/dataset/Hypersim_params`
- `plane_root`: `/cluster/scratch/ayavuz/dataset/Hypersim_ours`
- `output_root`: `/cluster/scratch/ayavuz/dataset/Hypersim_rendered`
- `frame_skip`: `25`

---

## Inference Scripts

### 1. Planarity Inference
Run MoGe planarity prediction on images.
```bash
./run_planarity_inference.sh <model_path> <input_dir> <output_dir> [model_size] [cache_dir]
```

**Defaults:**
- `model_path`: `/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt`
- `model_size`: `large`
- `cache_dir`: `/cluster/scratch/aoezkan/MoGe/checkpoints`

**Environment Variables:**
- `MOGE_CACHE_DIR`: Set automatically from `cache_dir`

**Outputs:**
- `raw/` - Raw probability maps (`.npy`)
- `binary/` - Binary masks (`.png`)
- `vis/` - Visualizations (`.png`)

---

### 2. Segmentation Prediction
Full pipeline: RGB → MoGe → plane segmentation.
```bash
./run_segmentation.sh <model_path> <input_root> <output_root> [model_size] [cache_dir] [frame_skip]
```

**Defaults:**
- `model_path`: `/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt`
- `input_root`: `/cluster/scratch/aoezkan/dataset/scannet_new/scans`
- `output_root`: `/cluster/scratch/aoezkan/results/scannet/moge`
- `model_size`: `large`
- `cache_dir`: `/cluster/scratch/aoezkan/MoGe/checkpoints`
- `frame_skip`: `50`

**Outputs:**
- `seg_pred/{scene_id}/` - Segmentation arrays (`.npy`)
- `seg_vis/{scene_id}/` - Visualizations (`.png`)

---

## Evaluation Scripts

### 1. Quantitative Evaluation
Run metrics evaluation on ScanNet++.
```bash
./run_evaluation.sh <method> [model_path] [rgb_root] [dataset_root] [save_dir] [max_scenes] [model_size] [cache_dir]
```

**Methods:** `gt`, `moge`, `planercnn`, `zeroplane`, `monoplane`

**Defaults:**
- `method`: `moge`
- `model_path`: `/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt`
- `rgb_root`: `/cluster/project/cvg/Shared_datasets/scannet++/data`
- `dataset_root`: `/cluster/scratch/aoezkan/dataset/scannetpp`
- `save_dir`: `/cluster/scratch/aoezkan/dataset/scannetpp/results/metrics`
- `max_scenes`: `5`

**Output:** `eval_{method}_{max_scenes}.csv`

---

### 2. Qualitative Comparison
Generate side-by-side comparison videos.
```bash
./run_qualitative.sh [rgb_root] [results_root] [gt_root] [output_root] [frame_skip] [max_scenes]
```

**Defaults:**
- `rgb_root`: `/cluster/scratch/aoezkan/dataset/scannet_new/scans`
- `results_root`: `/cluster/scratch/aoezkan/results/scannet`
- `gt_root`: `/cluster/scratch/aoezkan/dataset/planercnn/scannet_planeseg`
- `output_root`: `/cluster/scratch/aoezkan/results/scannet`
- `frame_skip`: `50`

**Output:** `comparison_videos/{scene_id}_baseline.mp4`

---

## SLURM Configuration

All scripts include SLURM directives (commented by default):
```bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/{script_name}_%j.out
#SBATCH --error=logs/{script_name}_%j.err
```

To use with SLURM:
1. Uncomment the `#SBATCH` lines
2. Create `logs/` directory: `mkdir -p logs`
3. Submit: `sbatch script.sh [args]`

---

## Scene List Format

All scripts accept scene list files:
```txt
# Comments start with #
scene_id_1
scene_id_2
# Skipped scene:
# scene_id_3
scene_id_4
```

---

## Quick Reference

| Task | Script | Key Args |
|------|--------|----------|
| Extract ScanNet++ planes | `scannetpp_plane_extraction.sh` | scene_list, config |
| Render ScanNet++ planes | `scannetpp_render_planes.sh` | scene_list, frame_skip |
| Generate videos | `scannetpp_video_gen.sh` | scene_list, fps |
| Extract Hypersim planes | `hypersim_plane_extraction.sh` | scene_list, config |
| Render Hypersim planes | `hypersim_render_planes.sh` | scene_list, frame_skip |
| Planarity inference | `run_planarity_inference.sh` | model_path, input_dir |
| Segmentation | `run_segmentation.sh` | model_path, input_root |
| Evaluate metrics | `run_evaluation.sh` | method, model_path |
| Comparison videos | `run_qualitative.sh` | rgb_root, results_root |

---

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `MOGE_CACHE_DIR` | inference scripts | HuggingFace cache for MoGe weights |
| `SCANNETPP_RGB_ROOT` | evaluator.py | RGB images root (fallback) |
| `BASELINE_ROOT` | evaluator.py | Baseline methods root (fallback) |
