"""
Profiling script to diagnose bottlenecks in process_batch_inference.
Runs on 1 scene with detailed timing for each step.
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from types import SimpleNamespace
import time

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v4
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import repo_path, scannetpp_rend_plane_path


# ============================================================
# CONFIGURATION
# ============================================================

model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

max_scenes_val = 1  # Only 1 scene for profiling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetailedTimer:
    """Timer that tracks individual operations."""
    def __init__(self):
        self.times = {}
        self.counts = {}

    def add(self, name, elapsed):
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += elapsed
        self.counts[name] += 1

    def print_summary(self):
        print("\n" + "="*60)
        print("PROFILING RESULTS")
        print("="*60)

        # Sort by total time
        sorted_items = sorted(self.times.items(), key=lambda x: -x[1])

        total = sum(self.times.values())
        print(f"{'Operation':<40} {'Total':>10} {'Count':>6} {'Avg':>10} {'%':>6}")
        print("-"*60)

        for name, t in sorted_items:
            count = self.counts[name]
            avg = t / count if count > 0 else 0
            pct = 100 * t / total if total > 0 else 0
            print(f"{name:<40} {t:>9.2f}s {count:>6} {avg*1000:>9.1f}ms {pct:>5.1f}%")

        print("-"*60)
        print(f"{'TOTAL':<40} {total:>9.2f}s")


def process_batch_inference_profiled(
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
    Profiled version of batch inference.
    """
    # GPU inference
    t0 = time.perf_counter()
    results = inference_model.predict_batch_fast(
        rgb_paths,
        num_tokens=args.num_tokens,
        return_all_heads=True
    )
    timer.add("gpu_inference", time.perf_counter() - t0)

    batch_data = []

    for i, (res, rgb_path, scene_id, frame_id, gt_plane, depth_gt) in enumerate(zip(
        results, rgb_paths, scene_ids, frame_ids, gt_planes, depths_gt
    )):
        # 1. Image.open to get dimensions
        t0 = time.perf_counter()
        img = Image.open(rgb_path).convert("RGB")
        img_np = np.array(img)
        H_rgb, W_rgb = img_np.shape[:2]
        timer.add("image_open", time.perf_counter() - t0)

        # 2. GT plane processing
        t0 = time.perf_counter()
        if gt_plane.ndim == 3:
            gt_plane = gt_plane[0]
        gt_seg_np = gt_plane.cpu().numpy().astype(np.int32)
        H_depth, W_depth = gt_seg_np.shape
        depth_gt_np = depth_gt[0].cpu().numpy() if depth_gt.ndim == 3 else depth_gt.cpu().numpy()
        timer.add("gt_processing", time.perf_counter() - t0)

        # 3. Extract MoGe outputs
        t0 = time.perf_counter()
        depth_moge = res["points"][:, :, 2]
        normal = res["normal"].transpose(2, 0, 1)
        timer.add("moge_extract", time.perf_counter() - t0)

        # 4. Create planarity mask
        t0 = time.perf_counter()
        planarity = (gt_seg_np > 0).astype(np.int16)
        timer.add("planarity_mask", time.perf_counter() - t0)

        # 5. Resize MoGe outputs
        t0 = time.perf_counter()
        depth_moge = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
        normal = cv2.resize(normal.transpose(1, 2, 0), (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)
        timer.add("resize_moge", time.perf_counter() - t0)

        # 6. Segmentation (THE SUSPECTED BOTTLENECK)
        t0 = time.perf_counter()
        labels_rgb, _ = compute_vectorized_planar_segments_v4(
            planarity_rgb,
            normal.transpose(1, 2, 0),
            depth_moge,
            np.deg2rad(args.normal_threshold_deg),
            args.depth_threshold,
            neighbor_match_count_thresh=args.neighbor_match_count_thresh
        )
        timer.add("segmentation", time.perf_counter() - t0)

        # 7. Remap labels
        t0 = time.perf_counter()
        labels_rgb, _ = remap_labels(labels_rgb)
        timer.add("remap_labels", time.perf_counter() - t0)

        # 8. Resize labels to depth resolution
        t0 = time.perf_counter()
        labels = cv2.resize(labels_rgb, (W_depth, H_depth), interpolation=cv2.INTER_NEAREST)
        timer.add("resize_labels", time.perf_counter() - t0)

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
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Max scenes: {max_scenes_val}")

    timer = DetailedTimer()

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
    # RUN PROFILING
    # ============================================================

    print("==> Running profiled pipeline on 1 scene")

    for batch in tqdm(val_loader, desc="Profiling"):
        rgb_paths = batch["rgb_path"]
        scene_ids = batch["scene_id"]
        frame_ids = batch["frame_idx"]
        gt_planes = batch["plane"]
        depths = batch["depth"]

        batch_data = process_batch_inference_profiled(
            rgb_paths, scene_ids, frame_ids, gt_planes, depths,
            inference_model, args, timer
        )

    timer.print_summary()
    print(f"\n[DONE] Profiled {len(val_dataset)} frames")
