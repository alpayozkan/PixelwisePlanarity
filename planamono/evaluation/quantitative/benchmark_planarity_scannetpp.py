#!/usr/bin/env python3
"""
Benchmark binary planarity prediction on ScanNet++ test set.

For each method, loads H5 plane predictions and derives binary planarity
(labels > 0 = planar). Compares against GT planarity. Computes:
- Overall: accuracy, F1, IoU
- Planar class: precision, recall
- Non-planar class: precision, recall

For ZeroPlane: label 20 = non-planar, remapped to 0 before evaluation.
For MoGe: label 0 = non-planar (no remapping needed).

Usage:
    python benchmark_planarity_scannetpp.py
"""

import os
import sys
import csv
import numpy as np
import h5py
import cv2
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "planeseg", "PixelwisePlanarity"))
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_rend_plane_path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_ROOT = "/cluster/scratch/aoezkan/planeseg/scannetpp/inference"
EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/scannetpp/eval"
DATASET_DIR = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
RGB_ROOT = "/cluster/project/cvg/Shared_datasets/scannet++/data"

METHODS = OrderedDict([
    ("moge_ep6_v5seg", {
        "display_name": "MoGe ep6 (v5seg)",
        "h5_folder": "moge_mixed_bce_476644_ep6_h5",
        "nonplanar_label": None,
    }),
    ("moge_ep6_v6seg", {
        "display_name": "MoGe ep6 (v6seg)",
        "h5_folder": "moge_mixed_bce_476644_ep6_v6seg_h5",
        "nonplanar_label": None,
    }),
    ("zp_75k", {
        "display_name": "ZeroPlane 75k",
        "h5_folder": "zeroplane_mixed_h5_dust3r_75k_h5",
        "nonplanar_label": 20,
    }),
    ("zp_145k", {
        "display_name": "ZeroPlane 145k",
        "h5_folder": "zeroplane_mixed_h5_dust3r_145k_h5",
        "nonplanar_label": 20,
    }),
    ("hires_ep1", {
        "display_name": "HiRes ep1",
        "h5_folder": "moge_hires_ep1_h5",
        "nonplanar_label": None,
    }),
    ("hires_ep2", {
        "display_name": "HiRes ep2",
        "h5_folder": "moge_hires_ep2_h5",
        "nonplanar_label": None,
    }),
    ("hires_ep3", {
        "display_name": "HiRes ep3",
        "h5_folder": "moge_hires_ep3_h5",
        "nonplanar_label": None,
    }),
])


# ---------------------------------------------------------------------------
# H5 loader (keeps one scene in memory at a time)
# ---------------------------------------------------------------------------
class LazyH5Loader:
    def __init__(self, h5_root: str, h5_folder: str, nonplanar_label=None):
        self.base = os.path.join(h5_root, h5_folder)
        self.nonplanar_label = nonplanar_label
        self._scene_id = None
        self._planes = None
        self._frame_ids = None

    def _load_scene(self, scene_id: str):
        h5_path = os.path.join(self.base, scene_id, "planes.h5")
        if not os.path.exists(h5_path):
            self._scene_id = None
            self._planes = None
            self._frame_ids = None
            return
        with h5py.File(h5_path, "r") as f:
            self._planes = f["planes"][:]  # (N, H, W)
            self._frame_ids = [
                fid.decode() if isinstance(fid, bytes) else str(fid)
                for fid in f["frame_ids"][:]
            ]
        self._scene_id = scene_id

    def get(self, scene_id: str, frame_idx: str, target_hw=None):
        if self._scene_id != scene_id:
            self._load_scene(scene_id)
        if self._frame_ids is None or frame_idx not in self._frame_ids:
            return None
        idx = self._frame_ids.index(frame_idx)
        labels = self._planes[idx].astype(np.int32)

        # Remap non-planar label (ZeroPlane: 20 → 0)
        if self.nonplanar_label is not None:
            labels = np.where(labels == self.nonplanar_label, 0, labels)

        # Resize to GT resolution if needed
        if target_hw is not None:
            H, W = target_hw
            if labels.shape != (H, W):
                labels = cv2.resize(
                    labels.astype(np.uint16), (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

        return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_counts(gt_binary, pred_binary):
    """Return TP, TN, FP, FN counts."""
    tp = int(np.sum(gt_binary & pred_binary))
    tn = int(np.sum(~gt_binary & ~pred_binary))
    fp = int(np.sum(~gt_binary & pred_binary))
    fn = int(np.sum(gt_binary & ~pred_binary))
    return tp, tn, fp, fn


def aggregate_metrics(all_counts):
    """Compute metrics from list of (tp, tn, fp, fn) tuples.

    Uses micro-averaging (sum counts first, then compute metrics).
    """
    tp = sum(c[0] for c in all_counts)
    tn = sum(c[1] for c in all_counts)
    fp = sum(c[2] for c in all_counts)
    fn = sum(c[3] for c in all_counts)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0

    # Planar class
    planar_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    planar_rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Non-planar class
    nonplanar_prec = tn / (tn + fn) if (tn + fn) > 0 else 0
    nonplanar_rec = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = 2 * planar_prec * planar_rec / (planar_prec + planar_rec) if (planar_prec + planar_rec) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "planar_prec": planar_prec,
        "planar_rec": planar_rec,
        "nonplanar_prec": nonplanar_prec,
        "nonplanar_rec": nonplanar_rec,
        "f1": f1,
        "iou": iou,
    }


def macro_average_metrics(per_frame_metrics):
    """Compute macro-averaged metrics (mean over per-frame metrics)."""
    keys = per_frame_metrics[0].keys()
    return {k: np.mean([m[k] for m in per_frame_metrics]) for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load dataset
    dataset = ScanNetPPPlaneDataset(
        rgb_root=RGB_ROOT,
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=DATASET_DIR,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
        max_scenes=None,
    )
    print(f"Dataset: {len(dataset)} frames")

    # Build index: (scene_id, frame_idx) for each valid_pair
    frame_info = []
    for vp in dataset.valid_pairs:
        rgb_path = vp[0]
        scene_id = rgb_path.split("/")[-4]
        frame_idx = os.path.splitext(os.path.basename(rgb_path))[0]
        frame_info.append((scene_id, frame_idx))

    # Initialize loaders
    loaders = {}
    for mk, cfg in METHODS.items():
        loaders[mk] = LazyH5Loader(H5_ROOT, cfg["h5_folder"], cfg["nonplanar_label"])

    # Iterate through all frames
    all_counts = {mk: [] for mk in METHODS}
    per_frame_metrics = {mk: [] for mk in METHODS}
    n_missing = {mk: 0 for mk in METHODS}

    for idx in range(len(dataset)):
        scene_id, frame_idx = frame_info[idx]

        # Load GT (only load plane labels, not full sample — use H5 directly)
        sample = dataset[idx]
        gt_plane = sample["plane"][0].numpy().astype(np.int32)
        gt_binary = gt_plane > 0
        H_gt, W_gt = gt_binary.shape

        for mk in METHODS:
            pred_labels = loaders[mk].get(scene_id, frame_idx, target_hw=(H_gt, W_gt))
            if pred_labels is None:
                n_missing[mk] += 1
                continue

            pred_binary = pred_labels > 0
            tp, tn, fp, fn = compute_counts(gt_binary, pred_binary)
            all_counts[mk].append((tp, tn, fp, fn))

            # Per-frame metrics for macro averaging
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total > 0 else 0
            pp = tp / (tp + fp) if (tp + fp) > 0 else 0
            pr = tp / (tp + fn) if (tp + fn) > 0 else 0
            npp = tn / (tn + fn) if (tn + fn) > 0 else 0
            npr = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            per_frame_metrics[mk].append({
                "accuracy": acc, "planar_prec": pp, "planar_rec": pr,
                "nonplanar_prec": npp, "nonplanar_rec": npr, "f1": f1, "iou": iou,
            })

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} frames")

    # Print results
    print(f"\n{'=' * 130}")
    print("BINARY PLANARITY BENCHMARK — ScanNet++ test set (micro-averaged)")
    print(f"{'=' * 130}")

    header = f"{'Method':<25s} {'Frames':>6s}  {'Accuracy':>8s}  {'P_plan':>8s}  {'R_plan':>8s}  {'P_nonp':>8s}  {'R_nonp':>8s}  {'F1':>8s}  {'IoU':>8s}"
    print(header)
    print("-" * len(header))

    results_rows = []
    for mk, cfg in METHODS.items():
        if not all_counts[mk]:
            print(f"{cfg['display_name']:<25s} {'N/A':>6s}  (no predictions found)")
            continue
        m = aggregate_metrics(all_counts[mk])
        n_frames = len(all_counts[mk])
        line = (
            f"{cfg['display_name']:<25s} {n_frames:>6d}  "
            f"{m['accuracy']:>8.4f}  {m['planar_prec']:>8.4f}  {m['planar_rec']:>8.4f}  "
            f"{m['nonplanar_prec']:>8.4f}  {m['nonplanar_rec']:>8.4f}  "
            f"{m['f1']:>8.4f}  {m['iou']:>8.4f}"
        )
        print(line)
        results_rows.append({
            "method": cfg["display_name"],
            "n_frames": n_frames,
            **m,
        })

    print(f"{'=' * 130}")

    # Also print macro-averaged
    print(f"\n{'=' * 130}")
    print("BINARY PLANARITY BENCHMARK — ScanNet++ test set (macro-averaged / per-frame mean)")
    print(f"{'=' * 130}")
    print(header)
    print("-" * len(header))

    for mk, cfg in METHODS.items():
        if not per_frame_metrics[mk]:
            continue
        m = macro_average_metrics(per_frame_metrics[mk])
        n_frames = len(per_frame_metrics[mk])
        line = (
            f"{cfg['display_name']:<25s} {n_frames:>6d}  "
            f"{m['accuracy']:>8.4f}  {m['planar_prec']:>8.4f}  {m['planar_rec']:>8.4f}  "
            f"{m['nonplanar_prec']:>8.4f}  {m['nonplanar_rec']:>8.4f}  "
            f"{m['f1']:>8.4f}  {m['iou']:>8.4f}"
        )
        print(line)

    print(f"{'=' * 130}")

    # Missing frames
    for mk, cfg in METHODS.items():
        if n_missing[mk] > 0:
            print(f"[WARN] {cfg['display_name']}: {n_missing[mk]} frames missing")

    # Save CSV
    out_csv = os.path.join(EVAL_ROOT, "benchmark_planarity.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "n_frames", "accuracy",
            "planar_prec", "planar_rec", "nonplanar_prec", "nonplanar_rec",
            "f1", "iou",
        ])
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
