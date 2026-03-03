"""
Evaluation script for: Our Planarity + GT Segmentation ablation on Hypersim.

Uses our MoGe planarity prediction combined with GT segmentation labels.
The planarity mask filters which GT segments are considered planar.

FAST VERSION - Optimizations:
1. OPTIONAL plane fitting (skip RANSAC for clustering-only metrics)
2. Reduced RANSAC iterations (200 instead of 2000)
3. Vectorized segmentation_covering
4. Fine-grained timing for profiling
5. loky backend (safer, avoids segfaults)

Hypersim adaptations:
- HypersimPlaneDataset with raycasted Euclidean depth
- Image tensors → numpy uint8 for MoGe inference (no real file paths)
- evaluate_single_frame_hypersim with M_cam_from_uv + native_wh
- Per-camera H5 output: planes_cam_XX.h5
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import h5py
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from types import SimpleNamespace

from joblib import Parallel, delayed

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import repo_path

from eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame_hypersim,
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

EXP_VER = "v1"
EXP_TAG = "moge_mixed_bce_476644_ep6"  # Change this to match the model

exp_name = f'hypersim_ourplanarity_gtseg_{EXP_TAG}_{EXP_VER}'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/hypersim/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/hypersim/inference/{exp_name}_h5"

model_path = os.environ.get("MODEL_PATH",
    "/cluster/scratch/ayavuz/moge_mixed_output_bce_476644_fixed/model_epoch6.pt"
)

# Hypersim paths
HYPERSIM_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

num_workers = 4

max_scenes_val = None  # Use all scenes
# max_scenes_val = 2  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PER-CAMERA H5 SAVING
# ============================================================

def save_predictions_h5_percamera(scene_predictions, h5_root_dir):
    """
    Save predictions to per-camera H5 files (Hypersim format).

    Args:
        scene_predictions: {scene_id: [(cam_name, frame_id, labels), ...]}
        h5_root_dir: Root directory for H5 files
    """
    os.makedirs(h5_root_dir, exist_ok=True)

    # Group by scene and camera
    scene_cam_data = defaultdict(lambda: defaultdict(list))
    for scene_id, frame_list in scene_predictions.items():
        for cam_name, frame_id, labels in frame_list:
            scene_cam_data[scene_id][cam_name].append((frame_id, labels))

    for scene_id, cam_dict in tqdm(scene_cam_data.items(), desc="Writing H5"):
        scene_h5_dir = os.path.join(h5_root_dir, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        for cam_name, frame_data in cam_dict.items():
            frame_data.sort(key=lambda x: x[0])
            frame_ids_list = [fd[0] for fd in frame_data]
            planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

            h5_path = os.path.join(scene_h5_dir, f"planes_{cam_name}.h5")
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
                f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    print(f"[H5] Written {len(scene_predictions)} scene files to {h5_root_dir}")


# ============================================================
# BATCH INFERENCE
# ============================================================

def process_batch_inference(
    images,
    rgb_paths,
    scene_ids,
    frame_ids,
    gt_segs,
    depths,
    inference_model,
    args,
    timer
):
    """Batch GPU inference for planarity, then apply to GT segmentation.

    Args:
        images: list of numpy uint8 arrays (H, W, 3) from batch["image"] tensors
        rgb_paths: virtual paths like "scene_id/cam_name/fid"
    """
    with timer("_gpu_inference"):
        results = inference_model.predict_batch_fast(
            images,
            num_tokens=args.num_tokens,
            return_all_heads=True
        )

    batch_data = []
    with timer("_postprocess"):
        for res, rgb_path, scene_id, frame_id, gt_seg, depth in zip(
            results, rgb_paths, scene_ids, frame_ids, gt_segs, depths
        ):
            # Extract cam_name from virtual path "scene_id/cam_name/fid"
            cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"

            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
            H_depth, W_depth = gt_seg_np.shape

            depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

            # Get MoGe planarity prediction
            planarity = res["planarity_probability"]
            planarity = cv2.resize(planarity, (W_depth, H_depth), interpolation=cv2.INTER_LINEAR)

            # Apply our planarity mask to GT segmentation
            planarity_mask = (planarity > args.threshold_planarity).astype(np.int32)
            labels = gt_seg_np * planarity_mask
            labels, _ = remap_labels(labels)

            batch_data.append({
                "scene_id": scene_id,
                "frame_id": frame_id,
                "cam_name": cam_name,
                "gt_seg_np": gt_seg_np,
                "depth_np": depth_np,
                "labels": labels,
            })

    return batch_data


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-only", action="store_true",
                        help="Only produce H5 files, skip evaluation")
    cli_args = parser.parse_args()

    inference_only = cli_args.inference_only

    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")
    print(f"[CONFIG] Inference only: {inference_only}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")

    timer = Timer()

    val_dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=PLANE_LABEL_ROOT,
        params_root=PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split="test",
        max_scenes=max_scenes_val,
        use_raycasted_depth="euclidean",
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
    # METADATA FOR HYPERSIM EVALUATION
    # ============================================================

    metadata_csv = os.path.join(repo_path, "shared", "datasets", "metadata_camera_parameters.csv")
    _metadata_df = pd.read_csv(metadata_csv, index_col="scene_name")

    def _get_M_cam_from_uv(scene_id):
        row = _metadata_df.loc[scene_id]
        return np.array(
            [[row[f"M_cam_from_uv_{i}{j}"] for j in range(3)] for i in range(3)],
            dtype=np.float64,
        )

    def _get_native_wh(scene_id):
        row = _metadata_df.loc[scene_id]
        nw = int(row["settings_output_img_width"]) if "settings_output_img_width" in row.index else 1024
        nh = int(row["settings_output_img_height"]) if "settings_output_img_height" in row.index else 768
        return (nw, nh)

    # ============================================================
    # STREAMING PIPELINE
    # ============================================================

    print("==> Running streaming pipeline")

    thresholds = (0.001, 0.005, 0.01)
    N_JOBS = min(16, os.cpu_count())

    def eval_frame_wrapper(frame_data, thresholds):
        return evaluate_single_frame_hypersim(
            frame_data["scene_id"],
            frame_data["full_frame_id"],
            frame_data["depth_np"],
            frame_data["gt_seg_np"],
            frame_data["M_cam_from_uv"],
            frame_data["native_wh"],
            frame_data["c2w_np"],
            frame_data["labels"],
            thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    scene_predictions = {}

    with timer("streaming_pipeline"):
        for batch in tqdm(val_loader, desc="Processing"):
            rgb_paths = batch["rgb_path"]
            scene_ids = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_segs = batch["plane"]
            depths = batch["depth"]
            c2ws = batch["c2w"]

            # Convert image tensors to numpy uint8 for MoGe inference
            with timer("_image_convert"):
                images_np = []
                for i in range(len(rgb_paths)):
                    img_tensor = batch["image"][i]  # (3, H, W) float [0, 1]
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    images_np.append(img_np)

            batch_data = process_batch_inference(
                images_np, rgb_paths, scene_ids, frame_ids, gt_segs, depths,
                inference_model, args, timer
            )

            # Accumulate predictions for per-camera H5
            for frame_data in batch_data:
                scene_id = frame_data["scene_id"]
                cam_name = frame_data["cam_name"]
                frame_id = frame_data["frame_id"]
                if scene_id not in scene_predictions:
                    scene_predictions[scene_id] = []
                scene_predictions[scene_id].append((cam_name, frame_id, frame_data["labels"]))

            if not inference_only:
                for i, data in enumerate(batch_data):
                    data["c2w_np"] = c2ws[i].numpy()
                    sid = data["scene_id"]
                    data["M_cam_from_uv"] = _get_M_cam_from_uv(sid)
                    data["native_wh"] = _get_native_wh(sid)
                    data["full_frame_id"] = f"{data['cam_name']}/{data['frame_id']}"

                outputs = Parallel(
                    n_jobs=N_JOBS,
                    backend="loky",
                )(
                    delayed(eval_frame_wrapper)(frame_data, thresholds)
                    for frame_data in batch_data
                )

                for (metrics, labels), frame_data in zip(outputs, batch_data):
                    results[(frame_data["scene_id"], frame_data["full_frame_id"])] = metrics

    num_frames = sum(len(v) for v in scene_predictions.values())
    print(f"[PIPELINE] Processed {num_frames} frames")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Writing H5 files")
    with timer("h5_write"):
        save_predictions_h5_percamera(scene_predictions, h5_root)

    if not inference_only:
        print("==> Saving results")
        save_results_csv(results, csv_out_dir)
        save_runtime(timer, csv_out_dir)

    timer.print_summary(num_frames=num_frames)
    print(f"\n[DONE] Processed {num_frames} frames in {timer.format_time(timer.total_elapsed())}")
