"""
Evaluation script for Hypersim dataset: Our Planarity + Our Segmentation (full pipeline).

This script runs:
1. MoGe inference (planarity, depth, normals)
2. Vectorized segmentation
3. Plane fitting evaluation

FAST VERSION - Optimizations:
1. Batch GPU inference (BATCH_SIZE=32)
2. Single RANSAC pass with multi-threshold evaluation
3. Vectorized segmentation_covering (~10x faster)
4. Parallel CPU evaluation with joblib
5. Direct memory accumulation (no intermediate files)
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from types import SimpleNamespace

from joblib import Parallel, delayed

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v4
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import repo_path

from eval_utils import (
    Timer,
    save_results_csv,
    save_predictions_h5,
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

exp_name = 'hypersim_moge_ours_v1'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/hypersim/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/hypersim/inference/{exp_name}_h5"

# Update these paths according to your setup
model_path = "/cluster/scratch/aoezkan/moge_runs/hypersim/moge_hypersim_4heads_v1/final_planarity_4heads_model.pt"
rgb_depth_root = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
plane_label_root = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
intrinsics_root = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
# Old paths (buggy plane_id=0 collision in rendered labels)
# rgb_depth_root = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
# plane_label_root = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
# intrinsics_root = "/cluster/scratch/ayavuz/dataset/Hypersim_params"

num_workers = 4
max_scenes_val = None  # Use all scenes
# max_scenes_val = 5  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# BATCH INFERENCE
# ============================================================

def process_batch_inference(
    rgb_paths,
    scene_ids,
    frame_ids,
    gt_segs,
    depths,
    inference_model,
    args,
    timer
):
    """
    Batch GPU inference for planarity, depth, and normals.
    Then run vectorized segmentation on each frame.
    """
    with timer("_gpu_inference"):
        results = inference_model.predict_batch_fast(
            rgb_paths,
            num_tokens=args.num_tokens,
            return_all_heads=True
        )

    batch_data = []
    with timer("_postprocess"):
        for res, rgb_path, scene_id, frame_id, gt_seg, depth in zip(
            results, rgb_paths, scene_ids, frame_ids, gt_segs, depths
        ):
            # For Hypersim, rgb_path is a virtual path like "scene_id/cam_name/frame_id"
            # We need to load the actual RGB from the HDF5
            # But since we already have it from the dataset, we can skip this
            # Get dimensions from GT
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
            H_depth, W_depth = gt_seg_np.shape

            depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

            # Get MoGe outputs
            planarity = res["planarity_probability"]
            depth_moge = res["points"][:, :, 2]
            normal = res["normal"].transpose(2, 0, 1)

            H_moge, W_moge = planarity.shape

            # Resize to depth resolution for evaluation
            planarity_eval = cv2.resize(planarity, (W_depth, H_depth), interpolation=cv2.INTER_LINEAR)
            depth_moge_eval = cv2.resize(depth_moge, (W_depth, H_depth), interpolation=cv2.INTER_LINEAR)
            normal_eval = cv2.resize(normal.transpose(1, 2, 0), (W_depth, H_depth), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

            # Apply planarity threshold
            planarity_mask = (planarity_eval > args.threshold_planarity).astype(np.int32)

            # Run vectorized segmentation
            labels, _ = compute_vectorized_planar_segments_v4(
                planarity_mask,
                normal_eval.transpose(1, 2, 0),
                depth_moge_eval,
                np.deg2rad(args.normal_threshold_deg),
                args.depth_threshold,
                neighbor_match_count_thresh=args.neighbor_match_count_thresh
            )
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
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")

    timer = Timer()

    val_dataset = HypersimPlaneDataset(
        hypersim_root=rgb_depth_root,
        plane_label_root=plane_label_root,
        params_root=intrinsics_root,
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
    # STREAMING PIPELINE
    # ============================================================

    print("==> Running streaming pipeline")

    thresholds = (0.01, 0.02, 0.05)
    N_JOBS = min(16, os.cpu_count())

    # Load metadata for M_cam_from_uv backprojection
    import pandas as pd
    metadata_csv = os.path.join(repo_path, "shared", "datasets", "metadata_camera_parameters.csv")
    _metadata_df = pd.read_csv(metadata_csv, index_col="scene_name")

    def _get_M_cam_from_uv(scene_id):
        row = _metadata_df.loc[scene_id]
        return np.array(
            [[row[f"M_cam_from_uv_{i}{j}"] for j in range(3)] for i in range(3)],
            dtype=np.float64,
        )

    def eval_frame_wrapper(frame_data, thresholds):
        return evaluate_single_frame_hypersim(
            frame_data["scene_id"],
            frame_data["frame_id"],
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
            Ks = batch["K"]
            c2ws = batch["c2w"]

            batch_data = process_batch_inference(
                rgb_paths, scene_ids, frame_ids, gt_segs, depths,
                inference_model, args, timer
            )

            for i, data in enumerate(batch_data):
                data["c2w_np"] = c2ws[i].numpy()
                sid = data["scene_id"]
                data["M_cam_from_uv"] = _get_M_cam_from_uv(sid)
                # native_wh from metadata
                row = _metadata_df.loc[sid]
                nw = int(row["settings_output_img_width"]) if "settings_output_img_width" in row.index else 1024
                nh = int(row["settings_output_img_height"]) if "settings_output_img_height" in row.index else 768
                data["native_wh"] = (nw, nh)

            # Use loky backend (safer, avoids segfaults)
            outputs = Parallel(
                n_jobs=N_JOBS,
                backend="loky",
            )(
                delayed(eval_frame_wrapper)(frame_data, thresholds)
                for frame_data in batch_data
            )

            for (metrics, labels), frame_data in zip(outputs, batch_data):
                scene_id = frame_data["scene_id"]
                frame_id = frame_data["frame_id"]

                results[(scene_id, frame_id)] = metrics

                if scene_id not in scene_predictions:
                    scene_predictions[scene_id] = []
                scene_predictions[scene_id].append((frame_id, labels))

    print(f"[PIPELINE] Processed {len(results)} frames")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Writing H5 files")
    with timer("h5_write"):
        save_predictions_h5(scene_predictions, h5_root)

    print("==> Saving results")
    save_results_csv(results, csv_out_dir)
    save_runtime(timer, csv_out_dir)

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Processed {len(results)} frames in {timer.format_time(timer.total_elapsed())}")
