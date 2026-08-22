# Inference Scripts

Shell wrappers for standalone inference on image folders. The main benchmark
pipeline (signals → planes H5s) uses the Python entry points directly — see the
repo readme (`save_moge_signals_planarity.py`, `segment_signals_to_planes.py`).

Both scripts carry commented `#SBATCH` directives — uncomment, `mkdir -p logs`,
`sbatch <script> [args]`. The checkpoint default resolves from `paths.py`
(`planarity_model_path`); MoGe base weights come from the standard HuggingFace
cache (`~/.cache/huggingface`; override via `HF_HOME`).

## 1. Planarity inference

Per-image planarity prediction on a folder of images:

```bash
./run_planarity_inference.sh <model_path> <input_dir> <output_dir>
```

Outputs: `raw/` (probability maps, `.npy`), `binary/` (masks, `.png`),
`vis/` (visualizations).

## 2. Segmentation prediction

Full RGB → MoGe → plane-segmentation pipeline over scene folders:

```bash
./run_segmentation.sh <model_path> <input_root> <output_root> [frame_skip]
# frame_skip default: 50
```

Outputs: `seg_pred/<scene_id>/` (label arrays, `.npy`), `seg_vis/<scene_id>/` (`.png`).
