"""
Evaluation script for: GT Planarity + Our Segmentation ablation.

Uses GT planarity mask (from GT plane labels > 0) combined with
our segmentation algorithm (MoGe depth/normals).

FAST VERSION - Optimizations:
1. Vectorized segmentation_covering (~10x faster)
2. Reduced RANSAC iterations (200 instead of 2000)
3. Single RANSAC pass with multi-threshold inlier counting
4. Optional plane metrics flag
5. Fine-grained timing for profiling
6. loky backend (safer, avoids segfaults)
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

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v4
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import repo_path, scannetpp_rend_plane_path

from eval_utils import (
    Timer,
    save_results_csv,
    save_predictions_h5,
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

exp_name = 'gtplanarity_ourseg_v1'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"

model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

# max_scenes_val = None
max_scenes_val = 5  # For testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# BATCH INFERENCE
# ============================================================

def process_batch_inference(
    rgb_paths,
    scene_ids,
    frame_ids,
    gt_planes,
    depths_gt,
    inference_model,
    args,
    timer
):
    """
    Batch GPU inference for depth and normals.
    Use GT planarity mask + our segmentation algorithm.
    """
    with timer("_gpu_inference"):
        results = inference_model.predict_batch_fast(
            rgb_paths,
            num_tokens=args.num_tokens,
            return_all_heads=True
        )

    batch_data = []
    with timer("_postprocess"):
        for res, rgb_path, scene_id, frame_id, gt_plane, depth_gt in zip(
            results, rgb_paths, scene_ids, frame_ids, gt_planes, depths_gt
        ):
            # Get image dimensions
            img = Image.open(rgb_path).convert("RGB")
            img_np = np.array(img)
            H_rgb, W_rgb = img_np.shape[:2]

            # Get GT plane segmentation at depth resolution
            if gt_plane.ndim == 3:
                gt_plane = gt_plane[0]
            gt_seg_np = gt_plane.cpu().numpy().astype(np.int32)
            H_depth, W_depth = gt_seg_np.shape

            depth_gt_np = depth_gt[0].cpu().numpy() if depth_gt.ndim == 3 else depth_gt.cpu().numpy()

            # Get MoGe depth and normals at RGB resolution
            depth_moge = res["points"][:, :, 2]
            normal = res["normal"].transpose(2, 0, 1)

            # Create GT planarity mask (binary from GT plane labels)
            planarity = (gt_seg_np > 0).astype(np.int16)

            # Resize MoGe outputs to RGB resolution for segmentation
            depth_moge = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            normal = cv2.resize(normal.transpose(1, 2, 0), (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
            # Resize planarity to RGB resolution
            planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)

            # Run our segmentation algorithm at RGB resolution
            # normal is (3, H, W), transpose to (H, W, 3) for segmentation
            labels_rgb, _ = compute_vectorized_planar_segments_v4(
                planarity_rgb,
                normal.transpose(1, 2, 0),
                depth_moge,
                np.deg2rad(args.normal_threshold_deg),
                args.depth_threshold,
                neighbor_match_count_thresh=args.neighbor_match_count_thresh
            )
            labels_rgb, _ = remap_labels(labels_rgb)

            # Resize labels to depth resolution for evaluation
            labels = cv2.resize(labels_rgb, (W_depth, H_depth), interpolation=cv2.INTER_NEAREST)

            batch_data.append({
                "scene_id": scene_id,
                "frame_id": frame_id,
                "gt_seg_np": gt_seg_np,
                "depth_np": depth_gt_np,
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
    scene_predictions = {}

    with timer("streaming_pipeline"):
        for batch in tqdm(val_loader, desc="Processing"):
            rgb_paths = batch["rgb_path"]
            scene_ids = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            batch_data = process_batch_inference(
                rgb_paths, scene_ids, frame_ids, gt_planes, depths,
                inference_model, args, timer
            )

            for i, data in enumerate(batch_data):
                data["K_np"] = Ks[i].numpy()
                data["c2w_np"] = c2ws[i].numpy()

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
