# Evaluation Scripts

Shell wrappers for the qualitative evaluation tools. The quantitative benchmark
is a Python entry point, not a shell script — see the repo readme:

```bash
python evaluation/quantitative/evaluate_all_baselines.py --methods gt ours
```

## 1. Qualitative Comparison Videos

Side-by-side method comparison videos:

```bash
./run_qualitative.sh [rgb_root] [results_root] [gt_root] [output_root] [frame_skip] [max_scenes]
```

Output: `comparison_videos/{scene_id}_baseline.mp4`

## 2. 3D Plane Comparison Renders

GT-vs-prediction 3D plane renders for ScanNet++ scenes:

```bash
./run_3d_comparison.sh [scene_list] [output_root] [checkpoint]
```

Defaults: `scene_list = evaluation/qualitative/scannetpp_vis_scenes.txt`,
`checkpoint = paths.planarity_model_path`.

## SLURM

Both scripts carry commented `#SBATCH` directives — uncomment, `mkdir -p logs`,
submit with `sbatch <script.sh>`.
