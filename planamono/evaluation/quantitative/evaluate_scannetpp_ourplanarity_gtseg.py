"""
Evaluation script for: Our Planarity + GT Segmentation ablation.

OPTIMIZED VERSION v2:
- Batch GPU inference (BATCH_SIZE=32 with predict_batch_fast)
- Parallel CPU evaluation (joblib)
- No PNG intermediate format (direct memory accumulation)
- Single RANSAC pass with multi-threshold inlier counting
- OPTIMIZATIONS:
  - Reduced RANSAC iterations (2000 → 500) for ~4x speedup
  - Vectorized segmentation_covering (removed Python loop)
  - Fine-grained timing for profiling
"""

import time
from contextlib import contextmanager

TIMINGS = {}
GLOBAL_START = time.perf_counter()

def format_time(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMINGS[name] = TIMINGS.get(name, 0.0) + elapsed
    if not name.startswith("_"):
        print(f"[TIMER] {name:30s} {format_time(elapsed)}")

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import h5py

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
)
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import *


# ============================================================
# VECTORIZED SEGMENTATION COVERING (OPTIMIZED)
# ============================================================

def segmentation_covering_fast(gt_mask, pred_mask, ignore_label=0):
    """
    OPTIMIZED: Fully vectorized Segmentation Covering (SC).

    Removed Python loop, uses numpy broadcasting for ~10x speedup.

    SC = (1 / sum_i |G_i|) * sum_i |G_i| * max_j IoU(G_i, P_j)
    """
    gt = gt_mask.ravel().astype(np.int64)
    pr = pred_mask.ravel().astype(np.int64)

    valid = gt != ignore_label
    if not np.any(valid):
        return 0.0

    gt = gt[valid]
    pr = pr[valid]

    # Remap labels to compact ranges
    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)

    n_gt = gt_labels.size
    n_pr = pr_labels.size

    # Build contingency matrix
    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    contingency = counts.reshape((n_gt, n_pr))

    gt_areas = contingency.sum(axis=1)  # (n_gt,)
    pr_areas = contingency.sum(axis=0)  # (n_pr,)

    # VECTORIZED: Compute IoU matrix and find best IoU per GT region
    # IoU(G_i, P_j) = contingency[i,j] / (gt_areas[i] + pr_areas[j] - contingency[i,j])
    union = gt_areas[:, None] + pr_areas[None, :] - contingency  # (n_gt, n_pr)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = np.where(union > 0, contingency / union, 0.0)

    # Best IoU per GT region (vectorized max across pred axis)
    best_iou = iou_matrix.max(axis=1)  # (n_gt,)

    # Weighted average by GT area
    total_area = gt_areas.sum()
    sc = (best_iou * gt_areas).sum() / total_area if total_area > 0 else 0.0
    return float(sc)


# ============================================================
# OPTIMIZED FUNCTIONS
# ============================================================

def compute_inliers_at_threshold(pts_world, labels, plane_params, threshold):
    """
    Given pre-fitted plane parameters, count inliers at a specific threshold.
    Much faster than running full RANSAC again.
    """
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

        # Only count if inlier ratio >= 0.5 (quality gate)
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
    num_iterations=2000,
    min_support=100
):
    """
    OPTIMIZATION: Fit planes ONCE with base threshold, then evaluate at multiple thresholds.
    """
    # Fit planes once with base threshold
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

    # Extract plane parameters
    plane_params = {}
    for pid, data in results.items():
        if "plane_model_refined" in data:
            plane_params[pid] = data["plane_model_refined"]

    if not plane_params:
        return {thr: {"precision": 0.0, "recall": 0.0} for thr in thresholds}

    # Evaluate at each threshold (fast - just counting)
    metrics = {}
    for thr in thresholds:
        metrics[thr] = compute_inliers_at_threshold(pts_world, labels, plane_params, thr)

    return metrics


def evaluate_single_frame(
    scene_id,
    frame_idx,
    depth_np,
    gt_seg_np,
    K_np,
    c2w_np,
    labels,
    thresholds
):
    """
    Evaluate a single frame with pre-computed planarity labels.
    Single RANSAC pass with multi-threshold inlier counting.
    """
    # Backproject to 3D
    pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, labels)

    if pts_world.shape[0] == 0:
        metric_thr = {f"prec@{int(thr*100)}cm": 0.0 for thr in thresholds}
        metric_thr.update({f"rec@{int(thr*100)}cm": 0.0 for thr in thresholds})
        return {
            "scene_id": scene_id,
            "frame_idx": frame_idx,
            "rand_index": 0.0,
            "voi": 0.0,
            "sc": 0.0,
            **metric_thr
        }, labels

    # Fit planes once, evaluate at multiple thresholds
    # OPTIMIZED: Reduced iterations 2000 → 500 for ~4x speedup
    multi_metrics = fit_planes_and_evaluate_multi_threshold(
        pts_world, pt_labels, thresholds,
        base_threshold=0.02,
        num_iterations=500,  # Was 2000, reduced for speed
        min_support=100
    )

    metric_thr = {}
    for thr in thresholds:
        metric_thr[f"prec@{int(thr*100)}cm"] = multi_metrics[thr]["precision"]
        metric_thr[f"rec@{int(thr*100)}cm"] = multi_metrics[thr]["recall"]

    # Clustering metrics (compare pred vs GT)
    # OPTIMIZED: Use vectorized segmentation_covering_fast
    ri = rand_score(gt_seg_np.flatten(), labels.flatten())
    Hs, Hm = variation_of_information(gt_seg_np, labels)
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


def process_batch_inference(
    rgb_paths,
    scene_ids,
    frame_ids,
    gt_segs,
    depths,
    inference_model,
    args
):
    """
    STAGE 1: Batch GPU inference for planarity.
    Returns list of (scene_id, frame_id, gt_seg_np, depth_np, K_np, c2w_np, labels).
    """
    results = inference_model.predict_batch_fast(
        rgb_paths,
        num_tokens=args.num_tokens,
        return_all_heads=True
    )

    batch_data = []
    for res, scene_id, frame_id, gt_seg, depth in zip(
        results, scene_ids, frame_ids, gt_segs, depths
    ):
        # Get GT segmentation at depth resolution
        if gt_seg.ndim == 3:
            gt_seg = gt_seg[0]
        gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
        H_depth, W_depth = gt_seg_np.shape

        depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

        # Get planarity and resize to depth resolution
        planarity = res["planarity_probability"]
        planarity = cv2.resize(planarity, (W_depth, H_depth), interpolation=cv2.INTER_LINEAR)

        # Apply our planarity mask to GT segmentation
        planarity_mask = (planarity > args.threshold_planarity).astype(np.int32)
        labels = gt_seg_np * planarity_mask
        labels, _ = remap_labels(labels)

        batch_data.append({
            "scene_id": scene_id,
            "frame_id": frame_id,
            "gt_seg_np": gt_seg_np,
            "depth_np": depth_np,
            "labels": labels,
        })

    return batch_data


# ============================================================
# CONFIGURATION
# ============================================================

exp_name = 'ourplanarity_gtseg_v1'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"

model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

max_scenes_val = None
# max_scenes_val = 1  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# SETUP
# ============================================================

print(f"[CONFIG] Experiment: {exp_name}")
print(f"[CONFIG] Device: {device}")
print(f"[CONFIG] Max scenes: {max_scenes_val}")

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

BATCH_SIZE = 32  # Batch GPU inference
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

from types import SimpleNamespace

args = SimpleNamespace(
    model_path=model_path,
    device=device,
    num_tokens=1024,
    threshold_planarity=0.6,
    normal_threshold_deg=10.0,
    depth_threshold=0.05,
    neighbor_match_count_thresh=24,
)

if not os.path.isfile(args.model_path):
    raise FileNotFoundError(f"Model not found: {args.model_path}")

inference_model = MoGePlanarityInference(args.model_path, device=args.device)
inference_model.model.encoder.use_memory_efficient_attention = False
torch.set_grad_enabled(False)
inference_model.model.eval()

# ============================================================
# STREAMING PIPELINE: GPU INFERENCE + CPU EVALUATION PER BATCH
# (Memory-efficient: processes and frees each batch before next)
# ============================================================

print("==> Running streaming pipeline (inference + evaluation per batch)")

thresholds = (0.01, 0.02, 0.05)
N_JOBS = min(16, os.cpu_count())

def eval_frame_wrapper(frame_data, thresholds):
    """Wrapper for joblib parallel execution."""
    return evaluate_single_frame(
        frame_data["scene_id"],
        frame_data["frame_id"],
        frame_data["depth_np"],
        frame_data["gt_seg_np"],
        frame_data["K_np"],
        frame_data["c2w_np"],
        frame_data["labels"],
        thresholds
    )

results = {}
scene_predictions = {}  # scene_id -> [(frame_id, labels), ...]

with timer("streaming_pipeline"):
    for batch in tqdm(val_loader, desc="Processing"):
        # --- GPU Inference for this batch ---
        rgb_paths = batch["rgb_path"]
        scene_ids = batch["scene_id"]
        frame_ids = batch["frame_idx"]
        gt_segs = batch["plane"]
        depths = batch["depth"]
        Ks = batch["K"]
        c2ws = batch["c2w"]

        batch_data = process_batch_inference(
            rgb_paths,
            scene_ids,
            frame_ids,
            gt_segs,
            depths,
            inference_model,
            args
        )

        # Add K and c2w to batch_data
        for i, data in enumerate(batch_data):
            data["K_np"] = Ks[i].numpy()
            data["c2w_np"] = c2ws[i].numpy()

        # --- CPU Evaluation for this batch (parallel within batch) ---
        outputs = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            batch_size=1
        )(
            delayed(eval_frame_wrapper)(frame_data, thresholds)
            for frame_data in batch_data
        )

        # --- Collect results (only keep metrics + labels for H5) ---
        for (metrics, labels), frame_data in zip(outputs, batch_data):
            scene_id = frame_data["scene_id"]
            frame_id = frame_data["frame_id"]

            results[(scene_id, frame_id)] = metrics

            if scene_id not in scene_predictions:
                scene_predictions[scene_id] = []
            scene_predictions[scene_id].append((frame_id, labels))

        # batch_data goes out of scope here, memory is freed each iteration

print(f"[PIPELINE] Processed {len(results)} frames")

# ============================================================
# STAGE 3: BATCH H5 WRITING (once per scene)
# ============================================================

print("==> Stage 3: Writing H5 files")

os.makedirs(h5_root, exist_ok=True)

with timer("stage3_h5_write"):
    for scene_id, frame_data in tqdm(scene_predictions.items(), desc="Writing H5"):
        # Sort by frame_id
        frame_data.sort(key=lambda x: x[0])

        frame_ids = [fd[0] for fd in frame_data]
        planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

        scene_h5_dir = os.path.join(h5_root, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        h5_path = os.path.join(scene_h5_dir, "planes.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
            f.create_dataset("frame_ids", data=np.array(frame_ids, dtype="S"))

print(f"[STAGE3] Written {len(scene_predictions)} scene files")

# ============================================================
# SAVE RESULTS
# ============================================================

print("==> Saving results")

os.makedirs(csv_out_dir, exist_ok=True)

# Per-frame results
out_path = os.path.join(csv_out_dir, 'moge_ours_merged_results.csv')
df = pd.DataFrame.from_records(list(results.values()))
df = df.set_index(["scene_id", "frame_idx"])
df.to_csv(out_path)
print(f"[CSV] Saved per-frame results to {out_path}")

# Per-scene aggregation
df_reset = df.reset_index()
scene_group = df_reset.groupby("scene_id")
df_scene = scene_group.mean(numeric_only=True)
df_scene["num_frames"] = scene_group.size()
cols = ["num_frames"] + [c for c in df_scene.columns if c != "num_frames"]
df_scene = df_scene[cols]

scene_csv = os.path.join(csv_out_dir, "moge_ours_merged_results_per_scene.csv")
df_scene.to_csv(scene_csv)
print(f"[CSV] Saved per-scene results to {scene_csv}")
print(f"[CSV] Number of scenes: {len(df_scene)}")

# Dataset-level stats
dataset_stats = {}
numeric_cols = df_scene.select_dtypes(include="number").columns
metric_cols = [c for c in numeric_cols if c != "num_frames"]

dataset_stats["num_scenes"] = len(df_scene)
dataset_stats["num_frames_total"] = int(df_scene["num_frames"].sum())

for c in metric_cols:
    dataset_stats[f"{c}_mean"] = df_scene[c].mean()
    dataset_stats[f"{c}_std"] = df_scene[c].std()

df_dataset = pd.DataFrame([dataset_stats])
dataset_csv = os.path.join(csv_out_dir, "moge_ours_merged_results_dataset.csv")
df_dataset.to_csv(dataset_csv, index=False)
print(f"[CSV] Saved dataset-level results to {dataset_csv}")

# ============================================================
# RUNTIME SUMMARY
# ============================================================

TOTAL = time.perf_counter() - GLOBAL_START

print("\n" + "=" * 50)
print("RUNTIME SUMMARY")
print("=" * 50)
for k, v in sorted(TIMINGS.items(), key=lambda x: -x[1]):
    if not k.startswith("_"):
        print(f"{k:35s} {format_time(v)}")
print("-" * 50)
print(f"{'TOTAL WALL TIME':35s} {format_time(TOTAL)}")
print("=" * 50)

# Save runtime CSVs
runtime_rows = []
for name, seconds in TIMINGS.items():
    if not name.startswith("_"):
        runtime_rows.append({
            "stage": name,
            "time_seconds": seconds,
            "time_hms": format_time(seconds),
        })

df_runtime = pd.DataFrame(runtime_rows).sort_values(by="time_seconds", ascending=False)
runtime_path = os.path.join(csv_out_dir, "runtime_breakdown.csv")
df_runtime.to_csv(runtime_path, index=False)

df_summary = pd.DataFrame([{
    "total_wall_time_seconds": TOTAL,
    "total_wall_time_hms": format_time(TOTAL),
    "num_frames": len(results),
    "fps": len(results) / TOTAL if TOTAL > 0 else 0,
}])
summary_path = os.path.join(csv_out_dir, "runtime_summary.csv")
df_summary.to_csv(summary_path, index=False)

print(f"\n[DONE] Processed {len(results)} frames in {format_time(TOTAL)}")
print(f"[DONE] Throughput: {len(results) / TOTAL:.2f} fps")
