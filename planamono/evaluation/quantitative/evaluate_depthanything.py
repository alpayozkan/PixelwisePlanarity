#!/usr/bin/env python3
"""
Unified evaluation of DepthAnything predictions across 4 datasets.

DepthAnything H5 format (self-contained per scene):
    ScanNet++: <root>/<scene_id>/rendered_v2.h5
    Hypersim:  <root>/<scene_id>/rendered_planes_cam_XX.h5
    Synthia:   <root>/<scene_id>/rendered_v2.h5
    VKITTI2:   <root>/<Scene>/<variant>/rendered_v2.h5

H5 keys:
    planes_dav2_normals  (N, H, W) uint16  — DepthAnything + DAV2 normals segmentation
    planes_moge_normals  (N, H, W) uint16  — DepthAnything + MoGe normals segmentation
    gt_planes            (N, H, W) uint16  — Ground truth plane labels (0 = non-planar)
    depth                (N, H, W) float32 — z-depth in CENTIMETERS (divide by 100 for meters)
    intrinsics           (N, 3, 3) float32 — Per-frame camera intrinsics (for the stored resolution)
    frame_ids            (N,)       bytes  — Frame identifiers

Label convention: 0 = non-planar (standard). Plane IDs >= 1.

Depth source for RANSAC:
    ScanNet++: GT depth from rendered_depth.h5 (mm → meters), same source as other
               baselines (PlaneRCNN, ZeroPlane, etc.) for fair comparison. Falls back
               to DA predicted depth if GT unavailable. Using DA predicted depth gives
               near-zero 3D metrics because small depth errors prevent RANSAC from
               aligning planes with GT thresholds (P@0.5cm: ~0.03 vs ~0.98 with GT).
    Others:    DA predicted depth (no standard GT depth H5 in same format).

Two variants evaluated:
    dav2_normals:  planes_dav2_normals vs gt_planes
    moge_normals:  planes_moge_normals vs gt_planes

Frame matching:
    ScanNet++: name-based — frame_ids like b'frame_000000'
    Hypersim:  sequential index per scene+camera file
    Synthia:   sequential index per scene file
    VKITTI2:   sequential index per Scene/variant file

Usage:
    python evaluate_depthanything.py                              # All datasets, both variants
    python evaluate_depthanything.py --datasets scannetpp hypersim
    python evaluate_depthanything.py --variants dav2_normals      # One variant only
    python evaluate_depthanything.py --max-scenes 2               # Quick test
    python evaluate_depthanything.py --aggregate-only             # Re-aggregate
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


# ============================================================
# CONFIGURATION
# ============================================================

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
N_JOBS = min(16, os.cpu_count())
EXP_VER = "v1"

# Depth unit: H5 stores depth in centimeters
DEPTH_SCALE = 0.01  # cm → m

VARIANTS = ["dav2_normals", "moge_normals"]
VARIANT_KEY = {
    "dav2_normals": "planes_dav2_normals",
    "moge_normals": "planes_moge_normals",
}

# Identity c2w — per-frame evaluation in camera space, pose is irrelevant
_C2W_IDENTITY = np.eye(4, dtype=np.float32)


# ============================================================
# DATASET ROOTS AND EVAL PATHS
# ============================================================

# GT depth root for ScanNet++ — same path used by evaluate_planercnn.py.
# rendered_depth.h5 stores depth in mm (uint16); divide by 1000 for meters.
GT_DEPTH_ROOT_SCANNETPP = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

_DA_ROOTS = {
    "scannetpp": Path("/cluster/scratch/ayavuz/dataset/depthanything_scannetpp"),
    "hypersim":  Path("/cluster/scratch/ayavuz/dataset/depthanything_hypersim"),
    "synthia":   Path("/cluster/scratch/ayavuz/dataset/depthanything_synthia"),
    "vkitti2":   Path("/cluster/scratch/ayavuz/dataset/depthanything_vkitti2"),
}

_EVAL_ROOTS = {
    "scannetpp": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval"),
    "hypersim":  Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval"),
    "synthia":   Path("/cluster/scratch/aoezkan/planeseg/synthia/eval"),
    "vkitti2":   Path("/cluster/scratch/aoezkan/planeseg/vkitti2/eval"),
}

_THRESHOLDS = {
    "scannetpp": (0.001, 0.005, 0.01),
    "hypersim":  (0.001, 0.005, 0.01),
    "synthia":   (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    "vkitti2":   (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
}

_DISPLAY_NAMES = {
    "scannetpp": "ScanNet++",
    "hypersim":  "Hypersim",
    "synthia":   "Synthia",
    "vkitti2":   "VKITTI2",
}


# ============================================================
# H5 FILE DISCOVERY
# ============================================================

def _discover_h5_files(dataset_key: str, da_root: Path) -> List[Tuple[str, str, Path]]:
    """Discover DepthAnything H5 files.

    Returns list of (scene_id, cam_tag, h5_path) tuples where:
      - scene_id: identifies the scene (e.g. '09c1414f1b' or 'Scene18/clone')
      - cam_tag:  camera identifier for Hypersim (e.g. 'cam_00'), empty string otherwise
    """
    entries = []

    if dataset_key == "hypersim":
        # <root>/<scene_id>/rendered_planes_cam_XX.h5
        for scene_dir in sorted(da_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            for h5_path in sorted(scene_dir.glob("rendered_planes_cam_*.h5")):
                cam_tag = h5_path.stem.replace("rendered_planes_", "")  # "cam_00"
                entries.append((scene_id, cam_tag, h5_path))

    elif dataset_key == "vkitti2":
        # <root>/<Scene>/<variant>/rendered_v2.h5
        for scene_dir in sorted(da_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            for variant_dir in sorted(scene_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                h5_path = variant_dir / "rendered_v2.h5"
                if h5_path.exists():
                    scene_id = f"{scene_dir.name}/{variant_dir.name}"
                    entries.append((scene_id, "", h5_path))

    else:
        # scannetpp / synthia: <root>/<scene_id>/rendered_v2.h5
        for scene_dir in sorted(da_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            h5_path = scene_dir / "rendered_v2.h5"
            if h5_path.exists():
                entries.append((scene_dir.name, "", h5_path))

    return entries


# ============================================================
# GT DEPTH LOADER (ScanNet++ only)
# ============================================================

def _load_gt_depth_scannetpp(scene_id: str) -> Dict[str, np.ndarray]:
    """Load GT depth for all frames of a ScanNet++ scene from rendered_depth.h5.

    Returns {frame_id: depth_array_meters}. Empty dict if file unavailable.
    Depth stored as uint16 millimeters; divide by 1000 to get float32 meters.
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


# ============================================================
# PER-FRAME EVALUATION WRAPPER
# ============================================================

def _eval_frame(scene_id, frame_idx, depth_m, gt_seg, K, labels, thresholds):
    """Thin wrapper around evaluate_single_frame."""
    return evaluate_single_frame(
        scene_id, frame_idx, depth_m, gt_seg, K, _C2W_IDENTITY,
        labels, thresholds,
        compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
        ransac_iterations=RANSAC_ITERATIONS,
        inlier_ratio_gate=INLIER_RATIO_GATE,
    )


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    dataset_key: str,
    variant: str,
    max_scenes: Optional[int] = None,
    part_id: int = 0,
    num_parts: int = 1,
) -> Dict:
    """Evaluate one variant on one dataset. Returns dict keyed by (scene_id, frame_idx)."""
    da_root = _DA_ROOTS[dataset_key]
    thresholds = _THRESHOLDS[dataset_key]
    display = _DISPLAY_NAMES[dataset_key]
    pred_key = VARIANT_KEY[variant]

    entries = _discover_h5_files(dataset_key, da_root)
    if not entries:
        print(f"[ERROR] No H5 files found in {da_root}")
        return {}

    if max_scenes is not None:
        # Group by scene_id to limit scenes (not individual camera files)
        seen_scenes = {}
        filtered = []
        for e in entries:
            sid = e[0]
            if sid not in seen_scenes:
                seen_scenes[sid] = 0
            if seen_scenes[sid] == 0:
                filtered.append(e)
                seen_scenes[sid] += 1
            if len(seen_scenes) >= max_scenes:
                break
        entries = filtered

    # Partition entries across workers
    if num_parts > 1:
        entries = entries[part_id::num_parts]

    print(f"[{display}] {variant}: {len(entries)} H5 file(s) to process (part {part_id+1}/{num_parts})")

    # Process scene-by-scene to avoid loading all frames into RAM at once.
    # Loading 14k+ frames at once (~65 GB) causes OOM; per-scene peak is ~1-3 GB.
    results = {}
    total_frames = 0
    n_gt_depth_missing = 0

    for scene_id, cam_tag, h5_path in tqdm(entries, desc=f"{display}/{variant}"):
        # ScanNet++: load GT depth for fair comparison with other baselines
        gt_depth_map: Dict[str, np.ndarray] = {}
        if dataset_key == "scannetpp":
            gt_depth_map = _load_gt_depth_scannetpp(scene_id)
            if not gt_depth_map:
                print(f"  [WARN] GT depth unavailable for {scene_id}, using DA depth")
                n_gt_depth_missing += 1

        try:
            with h5py.File(h5_path, "r") as f:
                n_frames = f["depth"].shape[0]
                frame_ids_raw = f["frame_ids"][:]
                scene_items = []
                for i in range(n_frames):
                    fid_bytes = frame_ids_raw[i]
                    fid = fid_bytes.decode("utf-8") if isinstance(fid_bytes, bytes) else str(fid_bytes)
                    # Use GT depth for ScanNet++ (fair comparison); DA predicted for others
                    if gt_depth_map and fid in gt_depth_map:
                        depth_m = gt_depth_map[fid]
                    else:
                        depth_m = f["depth"][i] * DEPTH_SCALE  # cm → m
                    gt_seg = f["gt_planes"][i].astype(np.int32)
                    pred = f[pred_key][i].astype(np.int32)
                    K = f["intrinsics"][i].astype(np.float64)
                    frame_idx = f"{cam_tag}/{fid}" if cam_tag else fid
                    scene_items.append((scene_id, frame_idx, depth_m, gt_seg, K, pred))
        except Exception as e:
            print(f"[WARN] Failed to read {h5_path}: {e}")
            continue

        if not scene_items:
            continue

        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_eval_frame)(sid, fidx, dm, gts, K, lbs, thresholds)
            for sid, fidx, dm, gts, K, lbs in scene_items
        )

        for (metrics, _), (sid, fidx, *_rest) in zip(outputs, scene_items):
            results[(sid, fidx)] = metrics

        total_frames += len(scene_items)

    if n_gt_depth_missing:
        print(f"[{display}] {variant}: WARNING: {n_gt_depth_missing} scenes used DA predicted"
              f" depth (GT depth unavailable)")
    print(f"[{display}] {variant}: evaluated {total_frames} frames")
    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_all(dataset_keys: List[str], variant_list: List[str]):
    """Print cross-dataset summary table."""
    print(f"\n{'='*80}")
    print("DEPTHANYTHING CROSS-DATASET SUMMARY")
    print(f"{'='*80}")

    rows = []
    for dk in dataset_keys:
        eval_root = _EVAL_ROOTS[dk]
        for var in variant_list:
            exp_name = f"depthanything_{var}_{EXP_VER}"
            csv_path = eval_root / exp_name / "results_dataset.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing: {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path).iloc[0]
                row = {
                    "Dataset": _DISPLAY_NAMES[dk],
                    "Variant": var,
                    "num_frames": int(df["num_frames_total"]),
                }
                for col, label in [("rand_index_mean", "RI"),
                                   ("voi_mean", "VOI"),
                                   ("sc_mean", "SC")]:
                    if col in df.index:
                        row[label] = df[col]
                for thr in _THRESHOLDS[dk]:
                    ts = f"{thr*100:.1f}cm"
                    for metric, prefix in [("prec", "P"), ("rec", "R")]:
                        col = f"{metric}@{ts}_mean"
                        if col in df.index:
                            row[f"{prefix}@{ts}"] = df[col]
                for bp in ["bp_f1", "bp_iou", "bp_recall", "bp_precision"]:
                    mc = f"{bp}_mean"
                    if mc in df.index:
                        row[bp] = df[mc]
                rows.append(row)
                print(f"[OK] {_DISPLAY_NAMES[dk]} / {var}: {row['num_frames']} frames")
            except Exception as e:
                print(f"[ERROR] {dk}/{var}: {e}")

    if not rows:
        return
    df_all = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df_all.to_string(index=False))
    print("=" * 80)
    return df_all


# ============================================================
# CLI
# ============================================================

def merge_parts(dataset_keys: List[str], variant_list: List[str], num_parts: int):
    """Merge per-part CSVs into final results, per-scene, and dataset CSVs."""
    for dk in dataset_keys:
        eval_root = _EVAL_ROOTS[dk]
        for var in variant_list:
            exp_name = f"depthanything_{var}_{EXP_VER}"
            out_dir = eval_root / exp_name

            part_dfs = []
            for pid in range(num_parts):
                part_dir = eval_root / f"{exp_name}_part{pid:02d}"
                csv = part_dir / "results.csv"
                if csv.exists():
                    part_dfs.append(pd.read_csv(csv))
                else:
                    print(f"[WARN] Missing part: {csv}")

            if not part_dfs:
                print(f"[WARN] No parts found for {dk}/{var}")
                continue

            df = pd.concat(part_dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["scene_id", "frame_idx"])
            out_dir.mkdir(parents=True, exist_ok=True)

            # Per-frame CSV
            df.to_csv(out_dir / "results.csv", index=False)
            print(f"[MERGE] {dk}/{var}: {len(df)} frames → {out_dir}/results.csv")

            # Per-scene CSV
            scene_group = df.groupby("scene_id")
            df_scene = scene_group.mean(numeric_only=True)
            df_scene["num_frames"] = scene_group.size()
            df_scene.to_csv(out_dir / "results_per_scene.csv")

            # Dataset CSV
            dataset_stats = {"num_scenes": len(df_scene), "num_frames_total": int(df_scene["num_frames"].sum())}
            for c in df_scene.select_dtypes(include="number").columns:
                if c != "num_frames":
                    dataset_stats[f"{c}_mean"] = df_scene[c].mean()
                    dataset_stats[f"{c}_std"] = df_scene[c].std()
            pd.DataFrame([dataset_stats]).to_csv(out_dir / "results_dataset.csv", index=False)
            print(f"[MERGE] {dk}/{var}: per-scene + dataset CSVs written")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DepthAnything predictions across multiple datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(_DA_ROOTS.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        choices=VARIANTS,
        help="Variants to evaluate (default: both dav2_normals moge_normals)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results, skip evaluation",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Limit number of scenes per dataset (for testing)",
    )
    parser.add_argument(
        "--part-id", type=int, default=0,
        help="Worker index (0-based) when splitting across multiple jobs",
    )
    parser.add_argument(
        "--num-parts", type=int, default=1,
        help="Total number of worker jobs (1 = no splitting)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge results from all parts into final output (run after all workers finish)",
    )
    args = parser.parse_args()

    dataset_keys = args.datasets or list(_DA_ROOTS.keys())
    variant_list = args.variants or VARIANTS

    print(f"[CONFIG] Datasets:   {dataset_keys}")
    print(f"[CONFIG] Variants:   {variant_list}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] EXP_VER:    {EXP_VER}")
    print(f"[CONFIG] RANSAC:     {RANSAC_ITERATIONS} iters, gate={INLIER_RATIO_GATE}")
    if args.num_parts > 1:
        print(f"[CONFIG] Part:       {args.part_id+1}/{args.num_parts}")

    if args.merge:
        merge_parts(dataset_keys, variant_list, args.num_parts)
        aggregate_all(dataset_keys, variant_list)
        print("\n[DONE] Merge complete!")
        return

    if not args.aggregate_only:
        for dk in dataset_keys:
            for var in variant_list:
                print(f"\n{'='*60}")
                print(f"Evaluating DepthAnything [{var}] on {_DISPLAY_NAMES[dk]}")
                print(f"{'='*60}")

                timer = Timer()
                with timer("evaluation"):
                    results = evaluate_dataset(
                        dk, var,
                        max_scenes=args.max_scenes,
                        part_id=args.part_id,
                        num_parts=args.num_parts,
                    )

                if not results:
                    print(f"[WARN] No results for {dk}/{var}")
                    continue

                exp_name = f"depthanything_{var}_{EXP_VER}"
                if args.num_parts > 1:
                    out_dir = _EVAL_ROOTS[dk] / f"{exp_name}_part{args.part_id:02d}"
                else:
                    out_dir = _EVAL_ROOTS[dk] / exp_name
                print(f"[SAVE] {len(results)} frames → {out_dir}")
                save_results_csv(results, str(out_dir))
                save_runtime(timer, str(out_dir))
                timer.print_summary(num_frames=len(results))

    if args.num_parts == 1:
        aggregate_all(dataset_keys, variant_list)
    print("\n[DONE] DepthAnything evaluation complete!")


if __name__ == "__main__":
    main()
