"""
Evaluation script for: Our Planarity + GT Segmentation ablation.

Uses our MoGe planarity prediction combined with GT segmentation labels.
The planarity mask filters which GT segments are considered planar.

FAST VERSION - Optimizations:
1. OPTIONAL plane fitting (skip RANSAC for clustering-only metrics)
2. Reduced RANSAC iterations (200 instead of 2000)
3. Vectorized segmentation_covering
4. Fine-grained timing for profiling
5. loky backend (safer, avoids segfaults)
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from types import SimpleNamespace

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
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

EXP_VER = "v6"
EXP_TAG = "moge_mixed_bce_476644_ep6"  # Change this to match the model

exp_name = f'ourplanarity_gtseg_{EXP_TAG}_{EXP_VER}'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
h5_root = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"

model_path = os.environ.get("MODEL_PATH",
    "/cluster/scratch/ayavuz/moge_mixed_output_bce_476644_fixed/model_epoch6.pt"
)
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
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
    """Batch GPU inference for planarity, then apply to GT segmentation."""
    with timer("_gpu_inference"):
        results = inference_model.predict_batch_fast(
            rgb_paths,
            num_tokens=args.num_tokens,
            return_all_heads=True
        )

    batch_data = []
    with timer("_postprocess"):
        for res, scene_id, frame_id, gt_seg, depth in zip(
            results, scene_ids, frame_ids, gt_segs, depths
        ):
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

    val_dataset = ScanNetPPPlaneDataset(
        rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=os.path.join(dataset_dir, ""),
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
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

    thresholds = (0.001, 0.005, 0.01)
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
            gt_segs = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            batch_data = process_batch_inference(
                rgb_paths, scene_ids, frame_ids, gt_segs, depths,
                inference_model, args, timer
            )

            # Accumulate predictions for H5
            for frame_data in batch_data:
                scene_id = frame_data["scene_id"]
                frame_id = frame_data["frame_id"]
                if scene_id not in scene_predictions:
                    scene_predictions[scene_id] = []
                scene_predictions[scene_id].append((frame_id, frame_data["labels"]))

            if not inference_only:
                for i, data in enumerate(batch_data):
                    data["K_np"] = Ks[i].numpy()
                    data["c2w_np"] = c2ws[i].numpy()

                outputs = Parallel(
                    n_jobs=N_JOBS,
                    backend="loky",
                )(
                    delayed(eval_frame_wrapper)(frame_data, thresholds)
                    for frame_data in batch_data
                )

                for (metrics, labels), frame_data in zip(outputs, batch_data):
                    results[(frame_data["scene_id"], frame_data["frame_id"])] = metrics

    num_frames = sum(len(v) for v in scene_predictions.values())
    print(f"[PIPELINE] Processed {num_frames} frames")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("==> Writing H5 files")
    with timer("h5_write"):
        save_predictions_h5(scene_predictions, h5_root)

    if not inference_only:
        print("==> Saving results")
        save_results_csv(results, csv_out_dir)
        save_runtime(timer, csv_out_dir)

    timer.print_summary(num_frames=num_frames)
    print(f"\n[DONE] Processed {num_frames} frames in {timer.format_time(timer.total_elapsed())}")
