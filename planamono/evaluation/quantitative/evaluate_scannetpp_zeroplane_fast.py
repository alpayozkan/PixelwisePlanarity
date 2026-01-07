"""
Evaluation script for: ZeroPlane predictions.

Loads ZeroPlane predictions from H5 files and evaluates against GT.
No inference needed - predictions are pre-computed.

FAST VERSION - Optimizations:
1. Single RANSAC pass with multi-threshold inlier counting
2. Vectorized segmentation_covering (~10x faster)
3. Reduced RANSAC iterations (200 instead of 2000)
4. Parallel CPU evaluation with joblib
5. Fine-grained timing for profiling
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import h5py
from tqdm import tqdm

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

exp_name = 'zeroplane_v1'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"

# ZeroPlane predictions H5 root
zeroplane_h5_root = "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/zeroplane_h5"

dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

# max_scenes_val = None
max_scenes_val = 5  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# H5 LOADING
# ============================================================

def load_zeroplane_predictions(h5_root, scene_id):
    """
    Load ZeroPlane predictions from H5 file.

    Returns:
        planes: (N, H, W) array of plane labels
        frame_ids: list of frame IDs
    """
    h5_path = os.path.join(h5_root, scene_id, "planes.h5")

    if not os.path.exists(h5_path):
        return None, None

    with h5py.File(h5_path, "r") as f:
        planes = f["planes"][:]
        frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                     for fid in f["frame_ids"][:]]

    return planes, frame_ids


def get_zeroplane_prediction(planes, frame_ids, frame_idx, target_shape):
    """
    Get ZeroPlane prediction for a specific frame.

    Args:
        planes: (N, H, W) predictions array
        frame_ids: list of frame IDs
        frame_idx: target frame ID
        target_shape: (H, W) to resize to

    Returns:
        labels: (H, W) plane labels, or None if not found
    """
    if frame_idx not in frame_ids:
        return None

    idx = frame_ids.index(frame_idx)
    pred = planes[idx]

    # Resize to target shape if needed
    if pred.shape != target_shape:
        pred = cv2.resize(pred.astype(np.float32),
                         (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(np.int32)

    return pred.astype(np.int32)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] ZeroPlane H5 root: {zeroplane_h5_root}")

    timer = Timer()

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
    # LOAD ALL ZEROPLANE PREDICTIONS
    # ============================================================

    print("==> Loading ZeroPlane predictions")

    # Get unique scene IDs
    scene_ids = val_dataset.scene_ids

    zeroplane_cache = {}
    with timer("load_predictions"):
        for scene_id in tqdm(scene_ids, desc="Loading H5"):
            planes, frame_ids = load_zeroplane_predictions(zeroplane_h5_root, scene_id)
            if planes is not None:
                zeroplane_cache[scene_id] = (planes, frame_ids)
            else:
                print(f"[WARN] No ZeroPlane predictions for scene {scene_id}")

    print(f"[DATA] Loaded predictions for {len(zeroplane_cache)} scenes")

    # ============================================================
    # EVALUATION PIPELINE
    # ============================================================

    print("==> Running evaluation pipeline")

    thresholds = (0.01, 0.02, 0.05)
    N_JOBS = min(16, os.cpu_count())

    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds):
        return evaluate_single_frame(
            scene_id,
            frame_idx,
            depth_np,
            gt_seg_np,
            K_np,
            c2w_np,
            labels,
            thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc="Evaluating"):
            scene_ids_batch = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            B = len(scene_ids_batch)

            # Prepare batch data
            batch_items = []
            for i in range(B):
                scene_id = scene_ids_batch[i]
                frame_idx = frame_ids[i]

                # Get GT
                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
                H, W = gt_seg_np.shape

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                # Get ZeroPlane prediction
                if scene_id not in zeroplane_cache:
                    skipped_frames += 1
                    continue

                planes, frame_ids_scene = zeroplane_cache[scene_id]
                labels = get_zeroplane_prediction(planes, frame_ids_scene, frame_idx, (H, W))

                if labels is None:
                    skipped_frames += 1
                    continue

                batch_items.append({
                    "scene_id": scene_id,
                    "frame_idx": frame_idx,
                    "depth_np": depth_np,
                    "gt_seg_np": gt_seg_np,
                    "K_np": Ks[i].numpy(),
                    "c2w_np": c2ws[i].numpy(),
                    "labels": labels,
                })

            if not batch_items:
                continue

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
                    item["labels"],
                    thresholds
                )
                for item in batch_items
            )

            for (metrics, labels), item in zip(outputs, batch_items):
                scene_id = item["scene_id"]
                frame_id = item["frame_idx"]
                results[(scene_id, frame_id)] = metrics

    print(f"[PIPELINE] Evaluated {len(results)} frames (skipped {skipped_frames})")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Saving results")
    save_results_csv(results, csv_out_dir)
    save_runtime(timer, csv_out_dir)

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Processed {len(results)} frames in {timer.format_time(timer.total_elapsed())}")
