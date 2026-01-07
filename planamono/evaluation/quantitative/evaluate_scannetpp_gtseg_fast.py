"""
Evaluation script for: GT Segmentation quality evaluation.

Evaluates the quality of GT plane segmentation using plane fitting metrics.
No MoGe inference needed - just loads GT and evaluates.

FAST VERSION - Optimizations:
1. Single RANSAC pass with multi-threshold inlier counting (was 3x per frame!)
2. Vectorized segmentation_covering (~10x faster)
3. Reduced RANSAC iterations (200 instead of 2000)
4. Direct memory accumulation (no intermediate PNG files)
5. Fine-grained timing for profiling
6. Optional plane metrics flag
"""

import time
from contextlib import contextmanager
from collections import defaultdict

TIMINGS = defaultdict(float)
TIMING_COUNTS = defaultdict(int)
GLOBAL_START = time.perf_counter()

def format_time(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

@contextmanager
def timer(name, verbose=False):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMINGS[name] += elapsed
    TIMING_COUNTS[name] += 1
    if verbose and not name.startswith("_"):
        print(f"[TIMER] {name:30s} {format_time(elapsed)}")

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
)
from planamono.paths import *


# ============================================================
# CONFIGURATION
# ============================================================

# Set to False to skip RANSAC plane fitting (much faster, clustering metrics only)
COMPUTE_PLANE_METRICS = True

# RANSAC iterations (200 is sufficient for evaluation)
RANSAC_ITERATIONS = 200

exp_name = 'gtseg_v2'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"

dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

max_scenes_val = None
# max_scenes_val = 5  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VECTORIZED SEGMENTATION COVERING (OPTIMIZED)
# ============================================================

def segmentation_covering_fast(gt_mask, pred_mask, ignore_label=0):
    """
    Fully vectorized Segmentation Covering (SC).
    ~10x faster than loop-based version.
    """
    gt = gt_mask.ravel().astype(np.int64)
    pr = pred_mask.ravel().astype(np.int64)

    valid = gt != ignore_label
    if not np.any(valid):
        return 0.0

    gt = gt[valid]
    pr = pr[valid]

    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)

    n_gt = gt_labels.size
    n_pr = pr_labels.size

    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    contingency = counts.reshape((n_gt, n_pr))

    gt_areas = contingency.sum(axis=1)
    pr_areas = contingency.sum(axis=0)

    union = gt_areas[:, None] + pr_areas[None, :] - contingency

    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = np.where(union > 0, contingency / union, 0.0)

    best_iou = iou_matrix.max(axis=1)
    total_area = gt_areas.sum()
    sc = (best_iou * gt_areas).sum() / total_area if total_area > 0 else 0.0
    return float(sc)


# ============================================================
# OPTIMIZED PLANE FITTING
# ============================================================

def compute_inliers_at_threshold(pts_world, labels, plane_params, threshold):
    """Count inliers at threshold using pre-fitted plane params."""
    total_inliers = 0
    total_points = 0

    for pid, params in plane_params.items():
        mask = (labels == pid)
        pts_plane = pts_world[mask]
        n_pts = pts_plane.shape[0]

        if n_pts == 0:
            continue

        a, b, c, d = params
        distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
        n_inliers = np.sum(distances < threshold)

        if n_inliers / n_pts >= 0.5:
            total_inliers += n_inliers
            total_points += n_pts

    precision = total_inliers / total_points if total_points > 0 else 0
    recall = total_inliers / len(labels) if len(labels) > 0 else 0

    return {"precision": precision, "recall": recall}


def fit_planes_and_evaluate_multi_threshold(
    pts_world,
    labels,
    thresholds,
    base_threshold=0.02,
    num_iterations=200,
    min_support=100
):
    """
    OPTIMIZATION: Fit planes ONCE with base threshold, then evaluate at multiple thresholds.
    This is ~3x faster than running RANSAC for each threshold.
    """
    with timer("_ransac_fit"):
        results, df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=base_threshold,
            num_iterations=num_iterations,
            min_support=min_support
        )

    if df is None or len(df) == 0:
        return {thr: {"precision": 0.0, "recall": 0.0} for thr in thresholds}

    plane_params = {}
    for pid, data in results.items():
        if "plane_model_refined" in data:
            plane_params[pid] = data["plane_model_refined"]

    if not plane_params:
        return {thr: {"precision": 0.0, "recall": 0.0} for thr in thresholds}

    with timer("_threshold_eval"):
        metrics = {}
        for thr in thresholds:
            metrics[thr] = compute_inliers_at_threshold(pts_world, labels, plane_params, thr)

    return metrics


# ============================================================
# FRAME EVALUATION
# ============================================================

def evaluate_single_frame(
    scene_id,
    frame_idx,
    depth_np,
    gt_seg_np,
    K_np,
    c2w_np,
    thresholds,
    compute_plane_metrics=True
):
    """
    Evaluate a single frame's GT segmentation quality.

    For GT segmentation, pred = gt (we're measuring how well GT segments
    fit to actual planar surfaces in 3D).
    """
    # For GT evaluation, we compare GT with itself for clustering metrics (should be perfect)
    # and evaluate plane fitting quality on the GT segments
    labels = gt_seg_np

    metric_thr = {}

    if compute_plane_metrics:
        with timer("_backproject"):
            pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, labels)

        if pts_world.shape[0] == 0:
            metric_thr = {f"prec@{int(thr*100)}cm": 0.0 for thr in thresholds}
            metric_thr.update({f"rec@{int(thr*100)}cm": 0.0 for thr in thresholds})
        else:
            multi_metrics = fit_planes_and_evaluate_multi_threshold(
                pts_world, pt_labels, thresholds,
                base_threshold=0.02,
                num_iterations=RANSAC_ITERATIONS,
                min_support=100
            )
            for thr in thresholds:
                metric_thr[f"prec@{int(thr*100)}cm"] = multi_metrics[thr]["precision"]
                metric_thr[f"rec@{int(thr*100)}cm"] = multi_metrics[thr]["recall"]
    else:
        for thr in thresholds:
            metric_thr[f"prec@{int(thr*100)}cm"] = np.nan
            metric_thr[f"rec@{int(thr*100)}cm"] = np.nan

    # Clustering metrics: GT vs GT (should be perfect: RI=1, VOI=0, SC=1)
    # These serve as sanity check / upper bound
    with timer("_rand_score"):
        ri = rand_score(gt_seg_np.flatten(), labels.flatten())

    with timer("_voi"):
        Hs, Hm = variation_of_information(gt_seg_np, labels)

    with timer("_sc"):
        sc = segmentation_covering_fast(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "rand_index": ri,
        "voi": Hs + Hm,
        "sc": sc,
        **metric_thr
    }

    return metrics, labels


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")

    val_dataset = ScanNetPPPlaneDataset(
        rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=os.path.join(dataset_dir, ""),
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="val",
        max_scenes=max_scenes_val,
    )

    print(f"[DATA] Validation set: {len(val_dataset)} frames")

    BATCH_SIZE = 32
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ============================================================
    # EVALUATION PIPELINE (No GPU inference needed for GT-only)
    # ============================================================

    print("==> Running evaluation pipeline")

    thresholds = (0.01, 0.02, 0.05)
    N_JOBS = min(16, os.cpu_count())

    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, thresholds):
        return evaluate_single_frame(
            scene_id,
            frame_idx,
            depth_np,
            gt_seg_np,
            K_np,
            c2w_np,
            thresholds,
            compute_plane_metrics=COMPUTE_PLANE_METRICS
        )

    results = {}
    scene_predictions = {}

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc="Evaluating"):
            scene_ids = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            B = len(scene_ids)

            # Prepare batch data
            batch_items = []
            for i in range(B):
                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                batch_items.append({
                    "scene_id": scene_ids[i],
                    "frame_idx": frame_ids[i],
                    "depth_np": depth_np,
                    "gt_seg_np": gt_seg_np,
                    "K_np": Ks[i].numpy(),
                    "c2w_np": c2ws[i].numpy(),
                })

            # Parallel evaluation
            outputs = Parallel(
                n_jobs=N_JOBS,
                backend="loky",
            )(
                delayed(eval_frame_wrapper)(
                    item["scene_id"],
                    item["frame_idx"],
                    item["depth_np"],
                    item["gt_seg_np"],
                    item["K_np"],
                    item["c2w_np"],
                    thresholds
                )
                for item in batch_items
            )

            for (metrics, labels), item in zip(outputs, batch_items):
                scene_id = item["scene_id"]
                frame_id = item["frame_idx"]

                results[(scene_id, frame_id)] = metrics

                if scene_id not in scene_predictions:
                    scene_predictions[scene_id] = []
                scene_predictions[scene_id].append((frame_id, labels))

    print(f"[PIPELINE] Evaluated {len(results)} frames")

    # ============================================================
    # WRITE H5 FILES (GT labels for reference)
    # ============================================================

    print("==> Writing H5 files")

    os.makedirs(h5_root, exist_ok=True)

    with timer("h5_write"):
        for scene_id, frame_data in tqdm(scene_predictions.items(), desc="Writing H5"):
            frame_data.sort(key=lambda x: x[0])
            frame_ids_list = [fd[0] for fd in frame_data]
            planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

            scene_h5_dir = os.path.join(h5_root, scene_id)
            os.makedirs(scene_h5_dir, exist_ok=True)

            h5_path = os.path.join(scene_h5_dir, "planes.h5")
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
                f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    print(f"[H5] Written {len(scene_predictions)} scene files")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Saving results")

    os.makedirs(csv_out_dir, exist_ok=True)

    out_path = os.path.join(csv_out_dir, 'results.csv')
    df = pd.DataFrame.from_records(list(results.values()))
    df = df.set_index(["scene_id", "frame_idx"])
    df.to_csv(out_path)
    print(f"[CSV] Saved per-frame results to {out_path}")

    df_reset = df.reset_index()
    scene_group = df_reset.groupby("scene_id")
    df_scene = scene_group.mean(numeric_only=True)
    df_scene["num_frames"] = scene_group.size()
    cols = ["num_frames"] + [c for c in df_scene.columns if c != "num_frames"]
    df_scene = df_scene[cols]

    scene_csv = os.path.join(csv_out_dir, "results_per_scene.csv")
    df_scene.to_csv(scene_csv)
    print(f"[CSV] Saved per-scene results to {scene_csv}")

    # Dataset stats
    dataset_stats = {"num_scenes": len(df_scene), "num_frames_total": int(df_scene["num_frames"].sum())}
    numeric_cols = df_scene.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols if c != "num_frames"]
    for c in metric_cols:
        dataset_stats[f"{c}_mean"] = df_scene[c].mean()
        dataset_stats[f"{c}_std"] = df_scene[c].std()

    df_dataset = pd.DataFrame([dataset_stats])
    dataset_csv = os.path.join(csv_out_dir, "results_dataset.csv")
    df_dataset.to_csv(dataset_csv, index=False)

    # ============================================================
    # RUNTIME SUMMARY
    # ============================================================

    TOTAL = time.perf_counter() - GLOBAL_START

    print("\n" + "=" * 60)
    print("RUNTIME BREAKDOWN (aggregated)")
    print("=" * 60)
    for k, v in sorted(TIMINGS.items(), key=lambda x: -x[1]):
        count = TIMING_COUNTS[k]
        avg = v / count if count > 0 else 0
        print(f"{k:25s} {format_time(v):>15s} ({count:>6d} calls, {avg*1000:>8.2f}ms avg)")
    print("-" * 60)
    print(f"{'TOTAL WALL TIME':25s} {format_time(TOTAL):>15s}")
    print(f"{'Throughput':25s} {len(results) / TOTAL:>15.2f} fps")
    print("=" * 60)

    # Save runtime
    runtime_rows = []
    for name, seconds in TIMINGS.items():
        runtime_rows.append({
            "stage": name,
            "time_seconds": seconds,
            "time_hms": format_time(seconds),
            "calls": TIMING_COUNTS[name],
            "avg_ms": (seconds / TIMING_COUNTS[name] * 1000) if TIMING_COUNTS[name] > 0 else 0
        })

    df_runtime = pd.DataFrame(runtime_rows).sort_values(by="time_seconds", ascending=False)
    runtime_path = os.path.join(csv_out_dir, "runtime_breakdown.csv")
    df_runtime.to_csv(runtime_path, index=False)

    print(f"\n[DONE] Processed {len(results)} frames in {format_time(TOTAL)}")
