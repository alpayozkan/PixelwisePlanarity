"""
Evaluate GT plane annotation quality across different GT sources.

Measures how well GT plane annotations fit geometric planes by:
1. Backprojecting GT depth using GT labels
2. Running RANSAC plane fitting per segment
3. Computing precision/recall at multiple distance thresholds

This evaluates annotation quality — not model prediction quality.

Supported methods (via --method):
  - our_gt_scannetpp:      Our GT on ScanNet++ test split (152 scenes)
  - planercnn_gt_scannetpp: PlaneRCNN GT on ScanNet++ test split (~42 scenes)
  - scannet_gt:             ScanNet GT (all available scenes)

Outputs (saved to EVAL_ROOT/<method>/):
  - results.csv                   Per-frame metrics (gate=0.9)
  - results_per_scene.csv         Per-scene aggregated metrics
  - results_dataset.csv           Dataset-level summary
  - results_multigates.csv        Per-frame metrics at all gate x threshold combos
  - results_multigates_summary.csv Dataset-level means for all combos
  - runtime_breakdown.csv         Profiling info

Aggregation (--aggregate-only):
  - gt_quality_comparison.csv     Combined table across all methods
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from planamono.shared.plane_fitting import backproject_v1 as backproject
from planamono.paths import (
    repo_path,
    scannetpp_path,
    scannetpp_rend_plane_path,
)

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

EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/eval_dataset"

THRESHOLDS = (0.001, 0.005, 0.01)  # 1mm, 5mm, 10mm
INLIER_RATIO_GATE = 0.9
INLIER_RATIO_GATES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
RANSAC_ITERATIONS = 200
MIN_SUPPORT = 100

N_JOBS = min(16, os.cpu_count())
MAX_SCENES = None  # None = all scenes

# Method configurations
METHODS = {
    "our_gt_scannetpp": {
        "dataset_type": "scannetpp",
        "plane_label_root": scannetpp_rend_plane_path,
        "split": "test",
        "display_name": "Our GT (ScanNet++)",
    },
    "planercnn_gt_scannetpp": {
        "dataset_type": "scannetpp",
        "plane_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn_v1",
        "split": "test",
        "display_name": "PlaneRCNN GT (ScanNet++)",
    },
    "scannet_gt": {
        "dataset_type": "scannet",
        "data_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannet",
        "display_name": "ScanNet GT",
    },
}

# ScanNet++ shared paths (used for depth/sem/rgb regardless of GT source)
SCANNETPP_RGB_ROOT = os.path.join(scannetpp_path, "data")
SCANNETPP_SEM_ROOT = scannetpp_rend_plane_path
SCANNETPP_DEPTH_ROOT = scannetpp_rend_plane_path
SCANNETPP_SPLIT_DIR = os.path.join(repo_path, "splits", "scannetpp")


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
# DATASET LOADING
# ============================================================

def load_scannetpp_dataset(method_cfg):
    """Load ScanNet++ dataset with the specified plane_label_root."""
    from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset

    ds = ScanNetPPPlaneDataset(
        rgb_root=SCANNETPP_RGB_ROOT,
        plane_label_root=method_cfg["plane_label_root"],
        sem_label_root=SCANNETPP_SEM_ROOT,
        depth_label_root=SCANNETPP_DEPTH_ROOT,
        split_txt_dir=SCANNETPP_SPLIT_DIR,
        split=method_cfg["split"],
        max_scenes=MAX_SCENES,
    )
    return ds


def load_scannet_dataset(method_cfg):
    """Load ScanNet dataset from scans directory."""
    from planamono.shared.datasets.scannet import ScanNetPlaneDataset

    data_root = method_cfg["data_root"]
    scans_dir = os.path.join(data_root, "scans")
    if not os.path.isdir(scans_dir):
        print(f"[ERROR] ScanNet scans dir not found: {scans_dir}")
        sys.exit(1)

    scene_ids = sorted(os.listdir(scans_dir))
    print(f"[DATA] Found {len(scene_ids)} ScanNet scenes in {scans_dir}")

    split_txt_path = os.path.join(tempfile.gettempdir(), "scannet_all_scenes.txt")
    with open(split_txt_path, "w") as f:
        f.write("\n".join(scene_ids) + "\n")

    ds = ScanNetPlaneDataset(
        data_root=data_root,
        split_txt=split_txt_path,
        split="all",
        max_scenes=MAX_SCENES,
    )
    return ds


def load_dataset(method_name):
    """Load dataset based on method configuration."""
    method_cfg = METHODS[method_name]
    dtype = method_cfg["dataset_type"]

    if dtype == "scannetpp":
        return load_scannetpp_dataset(method_cfg)
    elif dtype == "scannet":
        return load_scannet_dataset(method_cfg)
    else:
        raise ValueError(f"Unknown dataset_type: {dtype}")


# ============================================================
# SINGLE METHOD EVALUATION
# ============================================================

def run_evaluation(method_name):
    """Run full evaluation pipeline for a single method."""
    method_cfg = METHODS[method_name]
    exp_name = method_name
    csv_out_dir = os.path.join(EVAL_ROOT, exp_name)

    print("=" * 80)
    print(f"EVALUATING: {method_cfg['display_name']}")
    print("=" * 80)
    print(f"[CONFIG] Method:       {method_name}")
    print(f"[CONFIG] Output dir:   {csv_out_dir}")
    print(f"[CONFIG] Thresholds:   {THRESHOLDS}")
    print(f"[CONFIG] Default gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] All gates:    {INLIER_RATIO_GATES}")
    print(f"[CONFIG] RANSAC iters: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] N_JOBS:       {N_JOBS}")
    print(f"[CONFIG] Max scenes:   {MAX_SCENES}")

    timer = Timer()

    # --- Load dataset ---
    with timer("dataset_load"):
        ds = load_dataset(method_name)

    print(f"[DATA] Loaded {len(ds)} frames from {len(ds.scene_ids)} scenes")

    if len(ds) == 0:
        print("[ERROR] No frames found. Check dataset paths.")
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

    # --- Save default results (gate=INLIER_RATIO_GATE) ---
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
    print(f"{method_cfg['display_name'].upper()} — GT QUALITY EVALUATION SUMMARY")
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


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_results():
    """Aggregate results from all available methods into a comparison table."""
    print("=" * 80)
    print("AGGREGATING GT QUALITY RESULTS")
    print("=" * 80)

    rows = []
    for method_name, method_cfg in METHODS.items():
        csv_path = os.path.join(EVAL_ROOT, method_name, "results_dataset.csv")
        if not os.path.isfile(csv_path):
            print(f"[SKIP] No results for {method_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        row = {
            "method": method_name,
            "display_name": method_cfg["display_name"],
            "dataset": method_cfg["dataset_type"],
            "split": method_cfg.get("split", "all"),
            "num_scenes": int(df["num_scenes"].iloc[0]),
            "num_frames": int(df["num_frames_total"].iloc[0]),
        }

        for thr in THRESHOLDS:
            thr_str = f"{thr*100:.1f}cm"
            p_col = f"prec@{thr_str}_mean"
            r_col = f"rec@{thr_str}_mean"
            if p_col in df.columns:
                row[f"P@{thr*1000:.0f}mm"] = df[p_col].iloc[0]
            if r_col in df.columns:
                row[f"R@{thr*1000:.0f}mm"] = df[r_col].iloc[0]

        rows.append(row)
        print(f"[OK] {method_cfg['display_name']:30s}  "
              f"{row['num_scenes']:>4d} scenes, {row['num_frames']:>6d} frames")

    if not rows:
        print("[ERROR] No results found to aggregate.")
        return

    df_comp = pd.DataFrame(rows)
    comp_csv = os.path.join(EVAL_ROOT, "gt_quality_comparison.csv")
    df_comp.to_csv(comp_csv, index=False)
    print(f"\n[CSV] Saved comparison table to {comp_csv}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("GT QUALITY COMPARISON")
    print("=" * 100)

    header = f"{'Method':30s} {'Dataset':10s} {'Split':6s} {'Scenes':>6s} {'Frames':>7s}"
    for thr in THRESHOLDS:
        thr_str = f"{thr*1000:.0f}mm"
        header += f"  P@{thr_str:>4s}  R@{thr_str:>4s}"
    print(header)
    print("-" * len(header))

    for _, row in df_comp.iterrows():
        line = f"{row['display_name']:30s} {row['dataset']:10s} {row['split']:6s} "
        line += f"{row['num_scenes']:>6d} {row['num_frames']:>7d}"
        for thr in THRESHOLDS:
            p_key = f"P@{thr*1000:.0f}mm"
            r_key = f"R@{thr*1000:.0f}mm"
            p_val = row.get(p_key, float("nan"))
            r_val = row.get(r_key, float("nan"))
            line += f"  {p_val:>6.4f}  {r_val:>6.4f}"
        print(line)

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GT plane annotation quality across different sources."
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS.keys()),
        help="Which GT method to evaluate.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing results into comparison table.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit number of scenes (for debugging).",
    )

    args = parser.parse_args()

    if args.max_scenes is not None:
        MAX_SCENES = args.max_scenes

    if args.aggregate_only:
        aggregate_results()
    elif args.method:
        run_evaluation(args.method)
        # Also run aggregation if all methods have results
        aggregate_results()
    else:
        parser.error("Provide --method or --aggregate-only.")
