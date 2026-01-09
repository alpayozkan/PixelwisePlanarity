"""
Evaluation-only script for ScanNet++.

Loads pre-computed predictions from H5 files and runs evaluation.
Use this when you have already saved predictions and want to re-evaluate
with different parameters (e.g., different RANSAC iterations, thresholds).

Usage:
    python evaluate_scannetpp_mogeours.py
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
from tqdm import tqdm
from collections import defaultdict

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_rend_plane_path

from eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Set to False to skip RANSAC plane fitting (much faster, clustering metrics only)
COMPUTE_PLANE_METRICS = True

# RANSAC iterations (200 is sufficient for evaluation)
RANSAC_ITERATIONS = 200

# Inlier ratio threshold for quality gate
INLIER_RATIO_GATE = 0.9

exp_name = 'moge_ours_v2'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"

dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

max_scenes_val = None  # Use all scenes
# max_scenes_val = 5  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PREDICTION LOADING
# ============================================================

def load_scene_predictions(h5_root: str) -> dict:
    """
    Load all predictions from H5 files.

    Returns:
        {scene_id: {frame_id: labels_np}}
    """
    predictions = {}

    scene_dirs = [d for d in os.listdir(h5_root) if os.path.isdir(os.path.join(h5_root, d))]

    for scene_id in tqdm(scene_dirs, desc="Loading predictions"):
        h5_path = os.path.join(h5_root, scene_id, "planes.h5")
        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, "r") as f:
            planes = f["planes"][:]  # (N, H, W)
            frame_ids = f["frame_ids"][:]  # array of bytes

        predictions[scene_id] = {}
        for i, frame_id in enumerate(frame_ids):
            # Decode bytes to string if needed
            if isinstance(frame_id, bytes):
                frame_id = frame_id.decode("utf-8")
            predictions[scene_id][frame_id] = planes[i]

    return predictions


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] H5 root: {h5_root}")
    print(f"[CONFIG] Output dir: {csv_out_dir}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    timer = Timer()

    # Load predictions
    print("==> Loading predictions from H5 files")
    with timer("load_predictions"):
        predictions = load_scene_predictions(h5_root)
    print(f"[DATA] Loaded predictions for {len(predictions)} scenes")

    # Load dataset for GT
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

    BATCH_SIZE = 64  # Larger batch since no GPU inference
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ============================================================
    # EVALUATION PIPELINE
    # ============================================================

    print("==> Running evaluation pipeline")

    thresholds = (0.01, 0.02, 0.05)
    N_JOBS = min(16, os.cpu_count())

    def eval_frame_wrapper(frame_data, thresholds):
        return evaluate_single_frame(
            frame_data["scene_id"],
            frame_data["frame_id"],
            frame_data["depth_np"],
            frame_data["gt_seg_np"],
            frame_data["K_np"],
            frame_data["c2w_np"],
            frame_data["labels"],
            thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc="Evaluating"):
            scene_ids = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_segs = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            batch_data = []
            for i in range(len(scene_ids)):
                scene_id = scene_ids[i]
                frame_id = frame_ids[i]

                # Convert tensor frame_id to string if needed
                if isinstance(frame_id, torch.Tensor):
                    frame_id = str(frame_id.item())

                # Check if we have predictions for this frame
                if scene_id not in predictions:
                    skipped_frames += 1
                    continue
                if frame_id not in predictions[scene_id]:
                    skipped_frames += 1
                    continue

                gt_seg = gt_segs[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                labels = predictions[scene_id][frame_id]

                batch_data.append({
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "gt_seg_np": gt_seg_np,
                    "depth_np": depth_np,
                    "K_np": Ks[i].numpy(),
                    "c2w_np": c2ws[i].numpy(),
                    "labels": labels,
                })

            if not batch_data:
                continue

            # Parallel evaluation
            outputs = Parallel(
                n_jobs=N_JOBS,
                backend="loky",
            )(
                delayed(eval_frame_wrapper)(frame_data, thresholds)
                for frame_data in batch_data
            )

            for (metrics, _), frame_data in zip(outputs, batch_data):
                scene_id = frame_data["scene_id"]
                frame_id = frame_data["frame_id"]
                results[(scene_id, frame_id)] = metrics

    print(f"[PIPELINE] Evaluated {len(results)} frames")
    if skipped_frames > 0:
        print(f"[WARNING] Skipped {skipped_frames} frames (no predictions found)")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Saving results")
    save_results_csv(results, csv_out_dir)
    save_runtime(timer, csv_out_dir)

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Evaluated {len(results)} frames in {timer.format_time(timer.total_elapsed())}")
