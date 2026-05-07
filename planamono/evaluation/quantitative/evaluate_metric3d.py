#!/usr/bin/env python3
"""
Evaluation of Metric3D plane segmentation predictions on ScanNet++ and Hypersim.

Metric3D H5 format (same structure for both datasets):
    depth:      (N_frames, H, W) float32 — z-depth in meters (Metric3D predicted)
    frame_ids:  (N_frames,) bytes — frame identifiers
    gt_planes:  (N_frames, H, W) uint16 — ground truth plane labels (0=non-planar)
    intrinsics: (N_frames, 3, 3) float32 — per-frame camera intrinsics (correctly
                scaled to 480×640 from original 1920×1440; fx≈474 = JSON_fx/3)
    mask:       (N_frames, H, W) float32 — validity mask (not applied; labels handle invalids)
    planes:     (N_frames, H, W) uint16 — predicted plane labels (0=non-planar)

File layout:
    ScanNet++: {METRIC3D_SCANNETPP}/<scene_id>/rendered_v2.h5
    Hypersim:  {METRIC3D_HYPERSIM}/<scene_id>/rendered_planes_cam_XX.h5

Depth source for RANSAC:
    ScanNet++: GT depth from rendered_depth.h5 (in mm, divided by 1000 → meters),
               same as other baselines (PlaneRCNN, ZeroPlane, etc.) for fair comparison.
               Falls back to Metric3D predicted depth if GT depth unavailable.
    Hypersim:  Metric3D predicted z-depth (GT depth is Euclidean in a different format).

K (intrinsics) source:
    Uses Metric3D H5 intrinsics (fx≈474, correctly scaled to 480×640).
    NOTE: Other baselines (PlaneRCNN etc.) use ScanNetPPPlaneDataset which returns
    the raw JSON K (fx≈1424, original 1920×1440 resolution) without scaling to 480×640.
    This means other baselines have a systematic K scale error. Metric3D uses the
    correct K from H5, so 3D metrics may not be directly comparable with other baselines
    until those evaluations are re-run with scaled K.

c2w is identity: planes are fit independently in camera space per frame.

Usage:
    python evaluate_metric3d.py                    # Both datasets
    python evaluate_metric3d.py --datasets scannetpp
    python evaluate_metric3d.py --max-scenes 2     # Quick test
    python evaluate_metric3d.py --aggregate-only   # Re-aggregate existing results
    python evaluate_metric3d.py --scene-start 0 --scene-end 20  # Parallel shard
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    evaluate_single_frame,
    save_results_csv,
    save_runtime,
)
from planamono.paths import scannetpp_rend_plane_path


# ============================================================
# CONFIGURATION
# ============================================================

METRIC3D_SCANNETPP = "/cluster/scratch/ayavuz/dataset/metric3d_scannetpp"
METRIC3D_HYPERSIM = "/cluster/scratch/ayavuz/dataset/metric3d_hypersim"

# GT depth root for ScanNet++ — same path used by evaluate_planercnn.py.
# rendered_depth.h5 stores depth in mm (uint16); divide by 1000 for meters.
GT_DEPTH_ROOT_SCANNETPP = scannetpp_rend_plane_path

EVAL_ROOT_SCANNETPP = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval")
EVAL_ROOT_HYPERSIM = Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval")

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.001, 0.005, 0.01)
N_JOBS = min(16, os.cpu_count())
EXP_VER = "v1"


# ============================================================
# H5 LOADER
# ============================================================

class Metric3DH5Loader:
    """Load Metric3D predictions + GT from a single H5 file.

    H5 structure (both datasets):
        planes:     (N, H, W) uint16 — predicted labels (0=non-planar)
        gt_planes:  (N, H, W) uint16 — GT labels (0=non-planar)
        depth:      (N, H, W) float32 — z-depth in meters
        intrinsics: (N, 3, 3) float32 — per-frame camera intrinsics
        frame_ids:  (N,) bytes — frame identifiers
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.num_frames = f["planes"].shape[0]
            self.frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]

    def load_frame(self, idx: int) -> Optional[Dict]:
        """Load all data for a single frame by index."""
        if idx < 0 or idx >= self.num_frames:
            return None
        with h5py.File(self.h5_path, "r") as f:
            return {
                "planes": f["planes"][idx].astype(np.int32),       # (H, W)
                "gt_planes": f["gt_planes"][idx].astype(np.int32), # (H, W)
                "depth": f["depth"][idx],                           # (H, W) float32
                "K": f["intrinsics"][idx],                          # (3, 3) float32
                "frame_id": self.frame_ids[idx],
            }


# ============================================================
# EVALUATION HELPER
# ============================================================

def _eval_frame(scene_id: str, frame_idx: str, depth_np: np.ndarray,
                gt_seg_np: np.ndarray, K_np: np.ndarray, labels: np.ndarray):
    """Evaluate a single frame using standard pinhole backprojection."""
    c2w = np.eye(4, dtype=np.float32)  # per-frame eval, camera space is fine
    return evaluate_single_frame(
        scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w, labels,
        THRESHOLDS,
        compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
        ransac_iterations=RANSAC_ITERATIONS,
        inlier_ratio_gate=INLIER_RATIO_GATE,
    )


# ============================================================
# PER-DATASET EVALUATION
# ============================================================

def _load_gt_depth_scannetpp(scene_id: str) -> Dict[str, np.ndarray]:
    """Load GT depth for all frames of a scene from rendered_depth.h5.

    Returns {frame_id: depth_array_meters} mapping. Empty dict if unavailable.
    Depth is stored as uint16 millimeters; we divide by 1000 to get float32 meters.
    """
    gt_depth_path = Path(GT_DEPTH_ROOT_SCANNETPP) / scene_id / "rendered_depth.h5"
    if not gt_depth_path.exists():
        return {}
    gt_depth_map = {}
    with h5py.File(str(gt_depth_path), "r") as f:
        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
        for idx, fid in enumerate(frame_ids):
            gt_depth_map[fid] = f["depth"][idx].astype(np.float32) / 1000.0
    return gt_depth_map


def evaluate_scannetpp(
    metric3d_root: str,
    eval_root: Path,
    max_scenes: int = None,
    scene_start: int = None,
    scene_end: int = None,
) -> Dict:
    """Evaluate Metric3D on ScanNet++.

    Uses GT depth from rendered_depth.h5 for RANSAC plane fitting (consistent with
    other baselines). Falls back to Metric3D predicted depth if GT depth unavailable.

    Processes one scene at a time to avoid loading all ~14k frames into memory
    simultaneously (which causes severe joblib memory pressure / thrashing).
    Each scene's frames are evaluated in parallel, then memory is freed.
    """
    root = Path(metric3d_root)
    scene_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    scene_ids = [d.name for d in scene_dirs if (d / "rendered_v2.h5").exists()]

    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    if scene_start is not None or scene_end is not None:
        scene_ids = scene_ids[scene_start:scene_end]

    print(f"[ScanNet++] Evaluating {len(scene_ids)} scenes"
          f" (slice [{scene_start}:{scene_end}))")

    n_gt_depth_missing = 0
    results = {}
    for scene_id in tqdm(scene_ids, desc="ScanNet++ scenes"):
        h5_path = root / scene_id / "rendered_v2.h5"
        loader = Metric3DH5Loader(str(h5_path))

        # Load GT depth for this scene (for RANSAC, same source as other baselines)
        gt_depth_map = _load_gt_depth_scannetpp(scene_id)
        if not gt_depth_map:
            print(f"  [WARN] GT depth unavailable for {scene_id}, using Metric3D depth")
            n_gt_depth_missing += 1

        # Load all frames for this scene
        eval_items = []
        for i in range(loader.num_frames):
            frame = loader.load_frame(i)
            if frame is None:
                continue
            frame_id = frame["frame_id"]
            # GT depth for fair RANSAC comparison; fall back to predicted depth
            depth = gt_depth_map.get(frame_id, frame["depth"])
            eval_items.append((
                scene_id, frame_id,
                depth, frame["gt_planes"],
                frame["K"], frame["planes"],
            ))

        # Evaluate this scene's frames in parallel
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_eval_frame)(sid, fid, depth, gt, K, labels)
            for sid, fid, depth, gt, K, labels in eval_items
        )
        for (metrics, _), (sid, fid, *_rest) in zip(outputs, eval_items):
            results[(sid, fid)] = metrics

    if n_gt_depth_missing:
        print(f"[ScanNet++] WARNING: {n_gt_depth_missing} scenes used predicted depth"
              f" (GT depth unavailable)")
    print(f"[ScanNet++] Done: {len(results)} frames evaluated")
    return results


def evaluate_hypersim(
    metric3d_root: str,
    eval_root: Path,
    max_scenes: int = None,
    scene_start: int = None,
    scene_end: int = None,
) -> Dict:
    """Evaluate Metric3D on Hypersim.

    Discovers per-camera H5 files (rendered_planes_cam_XX.h5) under each scene.
    Frame IDs are embedded in the H5; scene_id/cam_name used as composite scene key.
    """
    root = Path(metric3d_root)
    scene_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    scene_ids = [d.name for d in scene_dirs]

    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    if scene_start is not None or scene_end is not None:
        scene_ids = scene_ids[scene_start:scene_end]

    print(f"[Hypersim] Evaluating {len(scene_ids)} scenes"
          f" (slice [{scene_start}:{scene_end}))")

    # Process one scene (all its cameras) at a time to bound memory usage
    results = {}
    for scene_id in tqdm(scene_ids, desc="Hypersim scenes"):
        scene_dir = root / scene_id
        cam_files = sorted(scene_dir.glob("rendered_planes_cam_*.h5"))
        for cam_file in cam_files:
            cam_name = cam_file.stem.replace("rendered_planes_", "")  # "cam_00"
            loader = Metric3DH5Loader(str(cam_file))
            composite_scene_id = f"{scene_id}/{cam_name}"

            eval_items = []
            for i in range(loader.num_frames):
                frame = loader.load_frame(i)
                if frame is None:
                    continue
                eval_items.append((
                    composite_scene_id, frame["frame_id"],
                    frame["depth"], frame["gt_planes"],
                    frame["K"], frame["planes"],
                ))

            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_eval_frame)(sid, fid, depth, gt, K, labels)
                for sid, fid, depth, gt, K, labels in eval_items
            )
            for (metrics, _), (sid, fid, *_rest) in zip(outputs, eval_items):
                results[(sid, fid)] = metrics

    print(f"[Hypersim] Done: {len(results)} frames evaluated")
    return results


# ============================================================
# MAIN EVALUATION DISPATCHER
# ============================================================

def evaluate_dataset(dataset_key: str, max_scenes: int = None,
                     scene_start: int = None, scene_end: int = None) -> Dict:
    """Evaluate Metric3D on a single dataset, save results."""
    display = {"scannetpp": "ScanNet++", "hypersim": "Hypersim"}[dataset_key]
    eval_root = {"scannetpp": EVAL_ROOT_SCANNETPP,
                 "hypersim": EVAL_ROOT_HYPERSIM}[dataset_key]
    metric3d_root = {"scannetpp": METRIC3D_SCANNETPP,
                     "hypersim": METRIC3D_HYPERSIM}[dataset_key]

    print(f"\n{'='*60}")
    print(f"Evaluating Metric3D on {display}")
    if scene_start is not None or scene_end is not None:
        print(f"  Scene slice: [{scene_start}:{scene_end})")
    print(f"{'='*60}")

    timer = Timer()
    with timer("evaluation"):
        if dataset_key == "scannetpp":
            results = evaluate_scannetpp(metric3d_root, eval_root, max_scenes,
                                         scene_start, scene_end)
        else:
            results = evaluate_hypersim(metric3d_root, eval_root, max_scenes,
                                        scene_start, scene_end)

    if not results:
        print(f"[WARN] No results for {display}")
        return {}

    exp_name = f"metric3d_{EXP_VER}"
    if scene_start is not None or scene_end is not None:
        shard_tag = f"_shard_{scene_start or 0}_{scene_end or 'end'}"
        csv_out_dir = eval_root / (exp_name + shard_tag)
    else:
        csv_out_dir = eval_root / exp_name

    print(f"[SAVE] Saving {len(results)} frames to {csv_out_dir}")
    save_results_csv(results, str(csv_out_dir))
    save_runtime(timer, str(csv_out_dir))
    timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_all(dataset_keys: List[str]):
    """Print cross-dataset summary of Metric3D results."""
    eval_roots = {
        "scannetpp": EVAL_ROOT_SCANNETPP,
        "hypersim": EVAL_ROOT_HYPERSIM,
    }
    display_names = {"scannetpp": "ScanNet++", "hypersim": "Hypersim"}

    print(f"\n{'='*80}")
    print("METRIC3D CROSS-DATASET SUMMARY")
    print(f"{'='*80}")

    rows = []
    for dk in dataset_keys:
        exp_name = f"metric3d_{EXP_VER}"
        csv_path = eval_roots[dk] / exp_name / "results_dataset.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing results for {dk}: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Dataset": display_names[dk],
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }
            for col, label in [("rand_index_mean", "RI"),
                                ("voi_mean", "VOI"),
                                ("sc_mean", "SC")]:
                if col in df.index:
                    row[label] = df[col]
            for thr in THRESHOLDS:
                thresh_str = f"{thr*100:.1f}cm"
                for prefix, label in [("prec", "P"), ("rec", "R")]:
                    col = f"{prefix}@{thresh_str}_mean"
                    if col in df.index:
                        row[f"{label}@{thresh_str}"] = df[col]
            for bp_col in ["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"]:
                col = f"{bp_col}_mean"
                if col in df.index:
                    row[bp_col] = df[col]
            rows.append(row)
            print(f"[OK] {display_names[dk]}: {row['num_frames']} frames")
        except Exception as e:
            print(f"[ERROR] Failed to read {dk}: {e}")

    if not rows:
        print("[ERROR] No results to aggregate")
        return

    df_all = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df_all.to_string(index=False))
    print("=" * 80)
    return df_all


# ============================================================
# SHARD MERGING
# ============================================================

def merge_shards(dataset_keys: List[str]):
    """Merge per-shard results.csv files into the final metric3d_v1/ directory.

    Finds all directories matching metric3d_v1_shard_* under each dataset's
    eval_root, concatenates their results.csv files (deduplicating on
    scene_id + frame_idx), then saves combined results to metric3d_v1/.
    """
    eval_roots = {
        "scannetpp": EVAL_ROOT_SCANNETPP,
        "hypersim": EVAL_ROOT_HYPERSIM,
    }
    display_names = {"scannetpp": "ScanNet++", "hypersim": "Hypersim"}

    for dk in dataset_keys:
        eval_root = eval_roots[dk]
        exp_name = f"metric3d_{EXP_VER}"
        shard_dirs = sorted(eval_root.glob(f"{exp_name}_shard_*"))

        if not shard_dirs:
            print(f"[{display_names[dk]}] No shard directories found, skipping merge")
            continue

        print(f"[{display_names[dk]}] Merging {len(shard_dirs)} shards: "
              f"{[d.name for d in shard_dirs]}")

        dfs = []
        for shard_dir in shard_dirs:
            csv_path = shard_dir / "results.csv"
            if not csv_path.exists():
                print(f"  [WARN] Missing {csv_path}, skipping")
                continue
            dfs.append(pd.read_csv(csv_path))

        if not dfs:
            print(f"  [ERROR] No shard CSVs found for {dk}")
            continue

        df_merged = pd.concat(dfs, ignore_index=True)
        # Deduplicate: keep last occurrence in case of overlap
        df_merged = df_merged.drop_duplicates(subset=["scene_id", "frame_idx"], keep="last")
        print(f"  Combined: {len(df_merged)} frames from {df_merged['scene_id'].nunique()} scenes")

        # Reconstruct results dict and re-save using standard save_results_csv
        results = {
            (row["scene_id"], row["frame_idx"]): row.to_dict()
            for _, row in df_merged.iterrows()
        }
        out_dir = eval_root / exp_name
        save_results_csv(results, str(out_dir))
        print(f"  Saved merged results to {out_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Metric3D predictions on ScanNet++ and Hypersim"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["scannetpp", "hypersim"],
        help="Datasets to evaluate (default: both)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results (reads from final metric3d_v1/ dir)",
    )
    parser.add_argument(
        "--merge-shards", action="store_true",
        help="Merge per-shard result CSVs into final metric3d_v1/ dir, then aggregate",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Limit scenes per dataset (for testing)",
    )
    parser.add_argument(
        "--scene-start", type=int, default=None,
        help="Start scene index (inclusive) for parallel SLURM sharding",
    )
    parser.add_argument(
        "--scene-end", type=int, default=None,
        help="End scene index (exclusive) for parallel SLURM sharding",
    )
    args = parser.parse_args()

    dataset_keys = args.datasets or ["scannetpp", "hypersim"]

    print(f"[CONFIG] Datasets: {dataset_keys}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] Thresholds: {THRESHOLDS}")
    print(f"[CONFIG] Experiment version: {EXP_VER}")

    if args.merge_shards:
        merge_shards(dataset_keys)
        aggregate_all(dataset_keys)
    elif args.aggregate_only:
        aggregate_all(dataset_keys)
    else:
        print(f"[CONFIG] Scene slice: [{args.scene_start}:{args.scene_end})")
        for dk in dataset_keys:
            evaluate_dataset(dk, args.max_scenes, args.scene_start, args.scene_end)
        # Only aggregate when running a full (non-sharded) job
        if args.scene_start is None and args.scene_end is None:
            aggregate_all(dataset_keys)

    print("\n[DONE] Metric3D evaluation complete!")


if __name__ == "__main__":
    main()
