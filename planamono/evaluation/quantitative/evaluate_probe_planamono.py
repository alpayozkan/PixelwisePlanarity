"""
Evaluation script for probe_planamono_scannetpp experiment.

Loads pre-computed predictions from rendered_v2.h5 files at:
    /cluster/scratch/ayavuz/dataset/probe_planamono_scannetpp/<scene_id>/rendered_v2.h5

Saves results to:
    /cluster/scratch/aoezkan/planeseg/scannetpp/eval/probe_planamono_v6/

Usage:
    # Full evaluation (single job)
    python evaluate_probe_planamono.py

    # Shard evaluation (for parallel SLURM jobs)
    python evaluate_probe_planamono.py --scene-start 0 --scene-end 9
    python evaluate_probe_planamono.py --scene-start 9 --scene-end 18
    ...

    # Merge shards after all complete
    python evaluate_probe_planamono.py --merge-shards
"""

import os
import argparse
import glob as glob_mod
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

from eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)


# ============================================================
# CONFIGURATION
# ============================================================

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.001, 0.005, 0.01)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

exp_name = "probe_planamono_v6"
csv_out_dir = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval") / exp_name
h5_root = "/cluster/scratch/ayavuz/dataset/probe_planamono_scannetpp"
h5_filename = "rendered_v2.h5"

dataset_dir = scannetpp_rend_plane_path


# ============================================================
# LAZY H5 LOADER
# ============================================================

class LazyH5SceneLoader:
    """Memory-efficient loader: one scene in memory at a time."""

    def __init__(self, h5_root, h5_filename="rendered_v2.h5"):
        self.h5_root = h5_root
        self.h5_filename = h5_filename
        self._current_scene_id = None
        self._current_planes = None
        self._frame_id_to_idx = {}

    def _load_scene(self, scene_id):
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        if not os.path.exists(h5_path):
            return False

        self._current_planes = None
        self._frame_id_to_idx = {}

        with h5py.File(h5_path, "r") as f:
            self._current_planes = f["planes"][:]
            frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]

        self._frame_id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
        self._current_scene_id = scene_id
        return True

    def get_prediction(self, scene_id, frame_idx, target_shape):
        if not self._load_scene(scene_id):
            return None
        if frame_idx not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_idx]
        pred = self._current_planes[idx].copy()

        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        return pred.astype(np.int32)

    def has_scene(self, scene_id):
        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        return os.path.exists(h5_path)


# ============================================================
# MERGE SHARDS
# ============================================================

def merge_shards():
    """Merge shard CSV files into final results."""
    shard_files = sorted(glob_mod.glob(str(csv_out_dir / "results_shard_*.csv")))
    if not shard_files:
        print(f"[ERROR] No shard files found in {csv_out_dir}")
        return

    print(f"[MERGE] Found {len(shard_files)} shard files")
    dfs = []
    for f in shard_files:
        df = pd.read_csv(f)
        print(f"  {Path(f).name}: {len(df)} frames")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"[MERGE] Total: {len(merged)} frames")

    # Reconstruct results dict for save_results_csv
    results = {}
    for _, row in merged.iterrows():
        key = (row["scene_id"], row["frame_idx"])
        results[key] = row.to_dict()

    save_results_csv(results, str(csv_out_dir))
    print(f"[MERGE] Saved merged results to {csv_out_dir}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate probe_planamono_scannetpp")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Max scenes to evaluate (for testing)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--scene-start", type=int, default=None,
                        help="Start scene index (for distributed eval)")
    parser.add_argument("--scene-end", type=int, default=None,
                        help="End scene index exclusive (for distributed eval)")
    parser.add_argument("--merge-shards", action="store_true",
                        help="Only merge existing shard CSVs, skip evaluation")
    args = parser.parse_args()

    if args.merge_shards:
        merge_shards()
        exit(0)

    # Derive shard_id from scene_start
    shard_id = None
    if args.scene_start is not None:
        shard_id = args.scene_start

    print(f"[CONFIG] Experiment: {exp_name}")
    print(f"[CONFIG] H5 root: {h5_root}")
    print(f"[CONFIG] H5 filename: {h5_filename}")
    print(f"[CONFIG] Output dir: {csv_out_dir}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Scene range: [{args.scene_start}:{args.scene_end})")
    print(f"[CONFIG] Shard ID: {shard_id}")
    print(f"[CONFIG] Thresholds: {THRESHOLDS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    timer = Timer()

    # Load dataset for GT
    print("\n==> Loading dataset")
    val_dataset = ScanNetPPPlaneDataset(
        rgb_root=os.path.join(scannetpp_path, "data"),
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=os.path.join(dataset_dir, ""),
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
        max_scenes=args.max_scenes,
    )
    print(f"[DATA] Test set: {len(val_dataset)} frames from {len(val_dataset.scene_ids)} scenes")

    # Scene range slicing for distributed eval
    if args.scene_start is not None or args.scene_end is not None:
        all_scenes = val_dataset.scene_ids
        s = args.scene_start or 0
        e = args.scene_end or len(all_scenes)
        subset_scenes = set(all_scenes[s:e])
        val_dataset.valid_pairs = [
            p for p in val_dataset.valid_pairs
            if p[0].split("/")[-4] in subset_scenes
        ]
        val_dataset.scene_ids = [sid for sid in all_scenes if sid in subset_scenes]
        print(f"[DATA] Scene range [{s}:{e}) → {len(val_dataset.scene_ids)} scenes, {len(val_dataset)} frames")

    # Init prediction loader
    loader = LazyH5SceneLoader(h5_root, h5_filename)
    available = [s for s in val_dataset.scene_ids if loader.has_scene(s)]
    missing = set(val_dataset.scene_ids) - set(available)
    if missing:
        print(f"[WARN] Missing predictions for {len(missing)} scenes: {missing}")
    print(f"[DATA] Found predictions for {len(available)}/{len(val_dataset.scene_ids)} scenes")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluation wrapper
    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds):
        return evaluate_single_frame(
            scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE,
        )

    results = {}
    skipped_frames = 0

    print("\n==> Running evaluation")
    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc="Evaluating"):
            scene_ids_batch = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            batch_items = []
            for i in range(len(scene_ids_batch)):
                scene_id = scene_ids_batch[i]
                frame_idx = frame_ids[i]

                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
                H, W = gt_seg_np.shape

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                labels = loader.get_prediction(scene_id, frame_idx, (H, W))
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

            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(eval_frame_wrapper)(
                    item["scene_id"], item["frame_idx"],
                    item["depth_np"], item["gt_seg_np"],
                    item["K_np"], item["c2w_np"],
                    item["labels"], THRESHOLDS,
                )
                for item in batch_items
            )

            for (metrics, _), item in zip(outputs, batch_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics

    print(f"\n[PIPELINE] Evaluated {len(results)} frames")
    if skipped_frames > 0:
        print(f"[WARNING] Skipped {skipped_frames} frames (no predictions found)")

    # Save results
    print("\n==> Saving results")
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    if shard_id is not None:
        # Save as shard CSV
        df = pd.DataFrame.from_records(list(results.values()))
        shard_path = csv_out_dir / f"results_shard_{shard_id}.csv"
        df.to_csv(shard_path, index=False)
        print(f"[CSV] Saved shard {shard_id} ({len(results)} frames) to {shard_path}")
    else:
        save_results_csv(results, str(csv_out_dir))

    save_runtime(timer, str(csv_out_dir))
    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Evaluated {len(results)} frames in {timer.format_time(timer.total_elapsed())}")
