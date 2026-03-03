"""
Evaluate GT plane annotation quality on ScanNet (all frames).

This script measures how well ScanNet's plane annotations fit geometric planes,
using RANSAC plane fitting on GT labels + GT depth. No model inference is needed.

GT labels are passed as both `labels` and `gt_seg_np` since we're measuring
annotation quality, not prediction quality.

Outputs (saved to EVAL_ROOT/scannet_gt_quality/):
  - results.csv              Per-frame metrics (gate=0.9)
  - results_per_scene.csv    Per-scene aggregated metrics
  - results_dataset.csv      Dataset-level summary
  - results_multigates.csv   Per-frame metrics at all gate×threshold combos
  - results_multigates_summary.csv  Dataset-level means for all combos
  - runtime_breakdown.csv    Profiling info
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from planamono.shared.datasets.scannet import ScanNetPlaneDataset
from planamono.shared.plane_fitting import backproject_v1 as backproject

from eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    compute_plane_metrics,
    compute_plane_metrics_multigates,
)


# ============================================================
# CONFIGURATION
# ============================================================

SCANNET_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/scannet"
EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/scannet/eval"
EXP_NAME = "scannet_gt_quality"

THRESHOLDS = (0.001, 0.005, 0.01)  # 1mm, 5mm, 10mm
INLIER_RATIO_GATE = 0.9
INLIER_RATIO_GATES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
RANSAC_ITERATIONS = 200
MIN_SUPPORT = 100

N_JOBS = min(16, os.cpu_count())
MAX_SCENES = None  # None = all scenes


# ============================================================
# PER-FRAME EVALUATION
# ============================================================

def evaluate_gt_frame(sample_dict, thresholds, inlier_ratio_gate, gates, ransac_iterations):
    """
    Evaluate a single frame's GT quality.

    Backprojects GT depth using GT labels, fits RANSAC planes, and computes
    precision/recall at multiple thresholds and gates.

    Returns:
        (default_metrics, multigates_metrics)
    """
    scene_id = sample_dict["scene_id"]
    frame_idx = sample_dict["frame_idx"]
    depth_np = sample_dict["depth_np"]
    plane_seg = sample_dict["plane_seg"]
    K_np = sample_dict["K_np"]
    c2w_np = sample_dict["c2w_np"]

    # Backproject once — shared by both default and multigates evaluation
    pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, plane_seg)

    n_planes = len(np.unique(plane_seg[plane_seg > 0]))

    # --- Default metrics (single gate) ---
    default_metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "n_planes": n_planes,
    }
    if pts_world.shape[0] == 0:
        for thr in thresholds:
            default_metrics[f"prec@{thr*100:.1f}cm"] = 0.0
            default_metrics[f"rec@{thr*100:.1f}cm"] = 0.0
    else:
        plane_thr = compute_plane_metrics(
            pts_world, pt_labels, thresholds,
            num_iterations=ransac_iterations,
            min_support=MIN_SUPPORT,
            inlier_ratio_gate=inlier_ratio_gate,
        )
        default_metrics.update(plane_thr)

    # --- Multigates metrics ---
    mg_metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "n_planes": n_planes,
    }
    if pts_world.shape[0] == 0:
        for thr in thresholds:
            thresh_str = f"{thr*100:.1f}cm"
            for gate in gates:
                mg_metrics[f"prec@{thresh_str}_gate{gate}"] = 0.0
                mg_metrics[f"rec@{thresh_str}_gate{gate}"] = 0.0
    else:
        mg_thr = compute_plane_metrics_multigates(
            pts_world, pt_labels, thresholds,
            inlier_ratio_gates=gates,
            num_iterations=ransac_iterations,
            min_support=MIN_SUPPORT,
        )
        mg_metrics.update(mg_thr)

    return default_metrics, mg_metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"[CONFIG] ScanNet root: {SCANNET_ROOT}")
    print(f"[CONFIG] Output dir:   {EVAL_ROOT}/{EXP_NAME}")
    print(f"[CONFIG] Thresholds:   {THRESHOLDS}")
    print(f"[CONFIG] Default gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] All gates:    {INLIER_RATIO_GATES}")
    print(f"[CONFIG] RANSAC iters: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] N_JOBS:       {N_JOBS}")
    print(f"[CONFIG] Max scenes:   {MAX_SCENES}")

    timer = Timer()

    # --- Build split file listing all scenes ---
    scans_dir = os.path.join(SCANNET_ROOT, "scans")
    if not os.path.isdir(scans_dir):
        print(f"[ERROR] ScanNet scans dir not found: {scans_dir}")
        sys.exit(1)

    scene_ids = sorted(os.listdir(scans_dir))
    print(f"[DATA] Found {len(scene_ids)} ScanNet scenes")

    split_txt_path = os.path.join(tempfile.gettempdir(), "scannet_all_scenes.txt")
    with open(split_txt_path, "w") as f:
        f.write("\n".join(scene_ids) + "\n")

    # --- Load dataset ---
    with timer("dataset_load"):
        ds = ScanNetPlaneDataset(
            data_root=SCANNET_ROOT,
            split_txt=split_txt_path,
            split="all",
            max_scenes=MAX_SCENES,
        )

    print(f"[DATA] Loaded {len(ds)} frames from {len(ds.scene_ids)} scenes")

    if len(ds) == 0:
        print("[ERROR] No frames found. Check SCANNET_ROOT and scene directories.")
        sys.exit(1)

    # --- Pre-extract all samples (CPU-bound I/O, sequential) ---
    print("==> Loading all frames...")
    all_samples = []
    with timer("data_loading"):
        for idx in tqdm(range(len(ds)), desc="Loading frames"):
            sample = ds[idx]
            depth_np = sample["depth"].squeeze(0).numpy()
            plane_seg = sample["plane"].squeeze(0).numpy().astype(np.int32)
            K_np = sample["K"].numpy()
            c2w_np = sample["c2w"].numpy()

            all_samples.append({
                "scene_id": sample["scene_id"],
                "frame_idx": sample["frame_idx"],
                "depth_np": depth_np,
                "plane_seg": plane_seg,
                "K_np": K_np,
                "c2w_np": c2w_np,
            })

    # --- Parallel evaluation ---
    print(f"==> Evaluating {len(all_samples)} frames with {N_JOBS} workers...")
    with timer("evaluation"):
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(evaluate_gt_frame)(
                s, THRESHOLDS, INLIER_RATIO_GATE, INLIER_RATIO_GATES, RANSAC_ITERATIONS
            )
            for s in tqdm(all_samples, desc="Evaluating")
        )

    # --- Collect results ---
    default_results = {}
    mg_results = {}
    for default_m, mg_m in outputs:
        key = (default_m["scene_id"], default_m["frame_idx"])
        default_results[key] = default_m
        mg_results[key] = mg_m

    # --- Save default results (gate=0.9) ---
    csv_out_dir = os.path.join(EVAL_ROOT, EXP_NAME)
    os.makedirs(csv_out_dir, exist_ok=True)

    print(f"\n==> Saving default results (gate={INLIER_RATIO_GATE})...")
    df_frames, df_scenes, df_dataset = save_results_csv(default_results, csv_out_dir)

    # --- Save multigates results ---
    print("==> Saving multigates results...")
    df_mg = pd.DataFrame.from_records(list(mg_results.values()))
    df_mg = df_mg.set_index(["scene_id", "frame_idx"])
    mg_csv = os.path.join(csv_out_dir, "results_multigates.csv")
    df_mg.to_csv(mg_csv)
    print(f"[CSV] Saved multigates per-frame results to {mg_csv}")

    # Multigates summary: dataset-level means
    numeric_cols = df_mg.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols if c.startswith(("prec@", "rec@"))]
    mg_summary = {}
    mg_summary["num_scenes"] = len(df_scenes)
    mg_summary["num_frames"] = len(df_mg)
    for c in metric_cols:
        mg_summary[f"{c}_mean"] = df_mg[c].mean()
        mg_summary[f"{c}_std"] = df_mg[c].std()

    df_mg_summary = pd.DataFrame([mg_summary])
    mg_summary_csv = os.path.join(csv_out_dir, "results_multigates_summary.csv")
    df_mg_summary.to_csv(mg_summary_csv, index=False)
    print(f"[CSV] Saved multigates summary to {mg_summary_csv}")

    # --- Save runtime ---
    save_runtime(timer, csv_out_dir)

    # --- Print summary ---
    timer.print_summary(num_frames=len(all_samples))

    print("\n" + "=" * 80)
    print(f"SCANNET GT QUALITY EVALUATION SUMMARY")
    print(f"  Scenes:  {len(df_scenes)}")
    print(f"  Frames:  {len(df_frames)}")
    print("=" * 80)

    # Default gate summary
    print(f"\n--- Default Gate = {INLIER_RATIO_GATE} ---")
    for thr in THRESHOLDS:
        thr_str = f"{thr*100:.1f}cm"
        p_col = f"prec@{thr_str}"
        r_col = f"rec@{thr_str}"
        if p_col in df_frames.columns:
            print(f"  {thr_str}:  precision = {df_frames[p_col].mean():.4f}  "
                  f"recall = {df_frames[r_col].mean():.4f}")

    # Multigates summary table
    print(f"\n--- Gate Sweep (dataset means) ---")
    header = f"{'gate':>6s}"
    for thr in THRESHOLDS:
        thr_str = f"{thr*1000:.0f}mm"
        header += f"  prec@{thr_str:>4s}  rec@{thr_str:>4s}"
    print(header)
    print("-" * len(header))

    for gate in INLIER_RATIO_GATES:
        row = f"{gate:>6.2f}"
        for thr in THRESHOLDS:
            thr_str = f"{thr*100:.1f}cm"
            p_col = f"prec@{thr_str}_gate{gate}"
            r_col = f"rec@{thr_str}_gate{gate}"
            p_val = df_mg[p_col].mean() if p_col in df_mg.columns else float("nan")
            r_val = df_mg[r_col].mean() if r_col in df_mg.columns else float("nan")
            row += f"  {p_val:>10.4f}  {r_val:>9.4f}"
        print(row)

    print(f"\n[DONE] Results saved to {csv_out_dir}")
