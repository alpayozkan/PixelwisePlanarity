#!/usr/bin/env python3
"""
Unified evaluation of PlaneRCNN predictions across 5 datasets.

PlaneRCNN H5 format (identical across all datasets):
    frames/<key>/{plane_segmentation (480,640) uint16,
                  plane_parameters (N,3),
                  camera_640x480 (6,),
                  intrinsic_3x3_original (3,3)}

Frame matching strategy:
    ScanNet++:  name-based  — strip "frame_" prefix from GT frame_idx
    Hypersim:   index-based — PlaneRCNN index i → GT index i (per scene+camera)
    Synthia:    index-based — PlaneRCNN index i → GT index i
    VKITTI2:    index-based — PlaneRCNN index i → GT index i
    PD:         index-based — PlaneRCNN index i → dataset[i]

Usage:
    python evaluate_planercnn.py                              # All 5 datasets
    python evaluate_planercnn.py --datasets scannetpp hypersim
    python evaluate_planercnn.py --max-scenes 1               # Quick test
    python evaluate_planercnn.py --aggregate-only              # Re-aggregate
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    evaluate_single_frame,
    evaluate_single_frame_hypersim,
    save_results_csv,
    save_runtime,
)
from planamono.paths import repo_path


# ============================================================
# CONFIGURATION
# ============================================================

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
N_JOBS = min(16, os.cpu_count())
EXP_VER = "v1"


# ============================================================
# PlaneRCNN H5 LOADER
# ============================================================

class PlaneRCNNH5Loader:
    """Load PlaneRCNN predictions from a single H5 file.

    H5 structure:
        frames/000000/plane_segmentation  (480, 640) uint16
        frames/000001/plane_segmentation  ...
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self._frame_keys = sorted(f["frames"].keys())
        self._key_to_idx = {k: i for i, k in enumerate(self._frame_keys)}

    @property
    def frame_keys(self) -> List[str]:
        return self._frame_keys

    @property
    def num_frames(self) -> int:
        return len(self._frame_keys)

    def get_segmentation(self, frame_key: str) -> Optional[np.ndarray]:
        """Get segmentation by exact frame key (e.g. '000000')."""
        if frame_key not in self._key_to_idx:
            return None
        with h5py.File(self.h5_path, "r") as f:
            seg = f[f"frames/{frame_key}/plane_segmentation"][:]
        return seg  # (480, 640) uint16

    def get_segmentation_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Get segmentation by sequential index."""
        if idx < 0 or idx >= len(self._frame_keys):
            return None
        return self.get_segmentation(self._frame_keys[idx])


def _remap_planercnn_labels(seg: np.ndarray) -> np.ndarray:
    """Remap PlaneRCNN segmentation to standard convention.

    PlaneRCNN: 0 = non-planar, plane instances start at 1.
    Already in standard convention — just cast to int32.
    """
    return seg.astype(np.int32)


def _resize_labels(labels: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize label map using nearest-neighbor interpolation."""
    if labels.shape[0] == target_h and labels.shape[1] == target_w:
        return labels
    return cv2.resize(
        labels.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

def _get_dataset_configs() -> Dict:
    """Return per-dataset configuration dicts."""
    splits_root = os.path.join(repo_path, "splits")

    return {
        "scannetpp": {
            "planercnn_root": "/cluster/scratch/ayavuz/dataset/planercnn_scannetpp_test",
            "dataset_class": "ScanNetPPPlaneDataset",
            "dataset_kwargs": {
                "rgb_root": "/cluster/project/cvg/Shared_datasets/scannet++/data",
                "plane_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
                "sem_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
                "depth_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
                "split_txt_dir": os.path.join(splits_root, "scannetpp"),
                "split": "test",
                "image_height": 480,
                "image_width": 640,
            },
            "eval_func": "standard",
            "thresholds": (0.001, 0.005, 0.01),
            "eval_root": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval"),
            "frame_match": "name",
            "display_name": "ScanNet++",
        },
        "hypersim": {
            "planercnn_root": "/cluster/scratch/ayavuz/dataset/planercnn_hypersim_test",
            "dataset_class": "HypersimPlaneDataset",
            "dataset_kwargs": {
                "hypersim_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
                "plane_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
                "params_root": "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params",
                "split_txt_dir": os.path.join(splits_root, "hypersim"),
                "split": "test",
                "image_height": 480,
                "image_width": 640,
                "use_raycasted_depth": "euclidean",
            },
            "eval_func": "hypersim",
            "thresholds": (0.001, 0.005, 0.01),
            "eval_root": Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval"),
            "frame_match": "index",
            "display_name": "Hypersim",
        },
        "synthia": {
            "planercnn_root": "/cluster/scratch/ayavuz/dataset/planercnn_synthia_test",
            "dataset_class": "SynthiaPlaneDataset",
            "dataset_kwargs": {
                "data_root": "/cluster/scratch/ayavuz/dataset/synthia_planes",
                "split": "test",
                "image_height": 480,
                "image_width": 640,
            },
            "eval_func": "standard",
            "thresholds": (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
            "eval_root": Path("/cluster/scratch/aoezkan/planeseg/synthia/eval"),
            "frame_match": "index",
            "display_name": "Synthia",
        },
        "vkitti2": {
            "planercnn_root": "/cluster/scratch/ayavuz/dataset/planercnn_vkitti2_test",
            "dataset_class": "VKITTI2PlaneDataset",
            "dataset_kwargs": {
                "data_root": "/cluster/scratch/ayavuz/dataset/vkitti2_planes",
                "split_txt_dir": os.path.join(splits_root, "vkitti2"),
                "split": "test",
                "image_height": 480,
                "image_width": 640,
            },
            "eval_func": "standard",
            "thresholds": (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
            "eval_root": Path("/cluster/scratch/aoezkan/planeseg/vkitti2/eval"),
            "frame_match": "index",
            "display_name": "VKITTI2",
        },
        "pd": {
            "planercnn_root": "/cluster/scratch/ayavuz/dataset/planercnn_parallel_domain_val",
            "dataset_class": "PDPlaneDataset",
            "dataset_kwargs": {
                "data_root": "/cluster/scratch/ayavuz/dataset/pd_zero/parallel_domain_plane",
                "split": "val",
            },
            "eval_func": "standard",
            "thresholds": (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
            "eval_root": Path("/cluster/scratch/aoezkan/planeseg/pd/eval"),
            "frame_match": "index",
            "display_name": "Parallel Domain",
        },
    }


def _load_dataset(cfg: Dict):
    """Instantiate the appropriate dataset class from config."""
    cls_name = cfg["dataset_class"]
    kwargs = cfg["dataset_kwargs"]

    if cls_name == "ScanNetPPPlaneDataset":
        from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
        return ScanNetPPPlaneDataset(**kwargs)
    elif cls_name == "HypersimPlaneDataset":
        from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
        return HypersimPlaneDataset(**kwargs)
    elif cls_name == "SynthiaPlaneDataset":
        from planamono.shared.datasets.synthia_plane_dataset import SynthiaPlaneDataset
        return SynthiaPlaneDataset(**kwargs)
    elif cls_name == "VKITTI2PlaneDataset":
        from planamono.shared.datasets.vkitti2_plane_dataset import VKITTI2PlaneDataset
        return VKITTI2PlaneDataset(**kwargs)
    elif cls_name == "PDPlaneDataset":
        from planamono.shared.datasets.pd_plane_dataset import PDPlaneDataset
        return PDPlaneDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset class: {cls_name}")


# ============================================================
# PlaneRCNN H5 DISCOVERY
# ============================================================

def _discover_planercnn_h5_files(planercnn_root: str) -> Dict[str, PlaneRCNNH5Loader]:
    """Discover all PlaneRCNN H5 files under a root directory.

    Returns {filename_stem: PlaneRCNNH5Loader} mapping.
    """
    loaders = {}
    root = Path(planercnn_root)
    if not root.exists():
        print(f"[ERROR] PlaneRCNN root does not exist: {root}")
        return loaders

    for h5_file in sorted(root.glob("*.h5")):
        stem = h5_file.stem  # e.g. "0a5c013435_planercnn" or "ai_001_004_cam_00_planercnn"
        loaders[stem] = PlaneRCNNH5Loader(str(h5_file))

    print(f"[PlaneRCNN] Found {len(loaders)} H5 files in {planercnn_root}")
    return loaders


# ============================================================
# PER-DATASET EVALUATION
# ============================================================

def _eval_frame_standard(
    scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np,
    labels, thresholds,
):
    """Wrapper for evaluate_single_frame (standard pinhole)."""
    return evaluate_single_frame(
        scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np,
        labels, thresholds,
        compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
        ransac_iterations=RANSAC_ITERATIONS,
        inlier_ratio_gate=INLIER_RATIO_GATE,
    )


def _eval_frame_hypersim(
    scene_id, frame_idx, depth_euc_np, gt_seg_np,
    M_cam_from_uv, native_wh, c2w_np,
    labels, thresholds,
):
    """Wrapper for evaluate_single_frame_hypersim (backproject_mcam)."""
    return evaluate_single_frame_hypersim(
        scene_id, frame_idx, depth_euc_np, gt_seg_np,
        M_cam_from_uv, native_wh, c2w_np,
        labels, thresholds,
        compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
        ransac_iterations=RANSAC_ITERATIONS,
        inlier_ratio_gate=INLIER_RATIO_GATE,
    )


def evaluate_scannetpp(cfg: Dict, loaders: Dict[str, PlaneRCNNH5Loader], max_scenes: int = None,
                       scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneRCNN on ScanNet++ (name-based frame matching)."""
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]

    # Build scene_id → loader mapping
    # PlaneRCNN files: <scene_id>_planercnn.h5
    scene_loaders = {}
    for stem, loader in loaders.items():
        if stem.endswith("_planercnn"):
            sid = stem[: -len("_planercnn")]
            scene_loaders[sid] = loader

    available = sorted([s for s in dataset.scene_ids if s in scene_loaders])
    # Apply scene slicing
    if scene_start is not None or scene_end is not None:
        available = available[scene_start:scene_end]
        scene_loaders = {k: v for k, v in scene_loaders.items() if k in available}
    print(f"[ScanNet++] Evaluating {len(available)} scenes (slice [{scene_start}:{scene_end}))")

    # Build per-scene GT frame lists: {scene_id: [(dataset_idx, gt_frame_key), ...]}
    scene_frames = {}
    for ds_idx in range(len(dataset)):
        pair = dataset.valid_pairs[ds_idx]
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        frame_idx_str = os.path.splitext(os.path.basename(rgb_path))[0]  # "frame_000000"
        if scene_id not in scene_loaders:
            continue
        scene_frames.setdefault(scene_id, []).append((ds_idx, frame_idx_str))

    # Build evaluation items
    eval_items = []
    skipped = 0
    for scene_id, frames in scene_frames.items():
        loader = scene_loaders[scene_id]
        for ds_idx, gt_frame_id in frames:
            # Strip "frame_" prefix to get PlaneRCNN key
            planercnn_key = gt_frame_id.replace("frame_", "")
            seg = loader.get_segmentation(planercnn_key)
            if seg is None:
                skipped += 1
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planercnn_labels(seg)
            labels = _resize_labels(labels, H, W)

            depth = sample["depth"].numpy()
            if depth.ndim == 3:
                depth = depth[0]

            eval_items.append({
                "scene_id": scene_id,
                "frame_idx": gt_frame_id,
                "depth_np": depth,
                "gt_seg_np": gt_seg,
                "K_np": sample["K"].numpy(),
                "c2w_np": sample["c2w"].numpy(),
                "labels": labels,
            })

    print(f"[ScanNet++] {len(eval_items)} frames to evaluate ({skipped} skipped)")

    # Parallel evaluation
    outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_eval_frame_standard)(
            item["scene_id"], item["frame_idx"],
            item["depth_np"], item["gt_seg_np"],
            item["K_np"], item["c2w_np"],
            item["labels"], thresholds,
        )
        for item in tqdm(eval_items, desc="ScanNet++")
    )

    results = {}
    for (metrics, _), item in zip(outputs, eval_items):
        results[(item["scene_id"], item["frame_idx"])] = metrics

    return results


def evaluate_hypersim(cfg: Dict, loaders: Dict[str, PlaneRCNNH5Loader], max_scenes: int = None,
                      scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneRCNN on Hypersim (index-based, per scene+camera)."""
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]

    # PlaneRCNN files: <scene_id>_<cam_name>_planercnn.h5
    # e.g. ai_001_004_cam_00_planercnn.h5
    # Extract (scene_id, cam_name) by splitting on "_cam_"
    cam_loaders = {}  # (scene_id, cam_name) → loader
    for stem, loader in loaders.items():
        if not stem.endswith("_planercnn"):
            continue
        base = stem[: -len("_planercnn")]  # "ai_001_004_cam_00"
        parts = base.split("_cam_")
        if len(parts) != 2:
            print(f"[WARN] Cannot parse Hypersim PlaneRCNN filename: {stem}")
            continue
        scene_id = parts[0]
        cam_name = f"cam_{parts[1]}"
        cam_loaders[(scene_id, cam_name)] = loader

    # Apply scene slicing on unique scene IDs
    if scene_start is not None or scene_end is not None:
        all_scene_ids = sorted(set(k[0] for k in cam_loaders.keys()))
        keep_scenes = set(all_scene_ids[scene_start:scene_end])
        cam_loaders = {k: v for k, v in cam_loaders.items() if k[0] in keep_scenes}

    print(f"[Hypersim] Found {len(cam_loaders)} scene+camera PlaneRCNN files"
          f" (slice [{scene_start}:{scene_end}))")

    # Group GT frames by (scene_id, cam_name)
    cam_frames = {}  # (scene_id, cam_name) → [(ds_idx, frame_order), ...]
    for ds_idx in range(len(dataset)):
        pair = dataset.valid_pairs[ds_idx]
        scene_id, cam_name = pair[0], pair[1]
        cam_frames.setdefault((scene_id, cam_name), []).append(ds_idx)

    # Build evaluation items (index-based matching)
    eval_items = []
    skipped = 0
    for (scene_id, cam_name), ds_indices in cam_frames.items():
        key = (scene_id, cam_name)
        if key not in cam_loaders:
            skipped += len(ds_indices)
            continue

        loader = cam_loaders[key]
        for order_idx, ds_idx in enumerate(ds_indices):
            seg = loader.get_segmentation_by_index(order_idx)
            if seg is None:
                skipped += 1
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planercnn_labels(seg)
            labels = _resize_labels(labels, H, W)

            depth_euc = sample["depth"].numpy()
            if depth_euc.ndim == 3:
                depth_euc = depth_euc[0]

            c2w = sample["c2w"].numpy()
            M_cam = dataset._get_M_cam_from_uv(scene_id)
            native_wh = dataset.valid_pairs[ds_idx][-1]

            full_frame_id = f"{cam_name}/{sample['frame_idx']}"

            eval_items.append({
                "scene_id": scene_id,
                "frame_idx": full_frame_id,
                "depth_euc_np": depth_euc,
                "gt_seg_np": gt_seg,
                "M_cam_from_uv": M_cam,
                "native_wh": native_wh,
                "c2w_np": c2w,
                "labels": labels,
            })

    print(f"[Hypersim] {len(eval_items)} frames to evaluate ({skipped} skipped)")

    outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_eval_frame_hypersim)(
            item["scene_id"], item["frame_idx"],
            item["depth_euc_np"], item["gt_seg_np"],
            item["M_cam_from_uv"], item["native_wh"],
            item["c2w_np"], item["labels"], thresholds,
        )
        for item in tqdm(eval_items, desc="Hypersim")
    )

    results = {}
    for (metrics, _), item in zip(outputs, eval_items):
        results[(item["scene_id"], item["frame_idx"])] = metrics

    return results


def evaluate_index_based(
    dataset_key: str, cfg: Dict,
    loaders: Dict[str, PlaneRCNNH5Loader],
    max_scenes: int = None,
    scene_start: int = None, scene_end: int = None,
):
    """Evaluate PlaneRCNN on Synthia / VKITTI2 / PD (index-based matching).

    For Synthia/VKITTI2: one H5 per scene, scene_id maps to filename.
    For PD: single H5 file for entire dataset.
    """
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        if dataset_key == "pd":
            kwargs["max_samples"] = max_scenes  # PD uses max_samples
        else:
            kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]
    display = cfg["display_name"]

    if dataset_key == "pd":
        # PD: single H5, flat index matching
        # Find the single H5 file
        if len(loaders) == 0:
            print(f"[{display}] No PlaneRCNN H5 files found")
            return {}
        # Use the first (and likely only) loader
        loader = list(loaders.values())[0]
        print(f"[{display}] Single H5 with {loader.num_frames} frames, "
              f"dataset has {len(dataset)} frames")

        eval_items = []
        skipped = 0
        for ds_idx in range(len(dataset)):
            seg = loader.get_segmentation_by_index(ds_idx)
            if seg is None:
                skipped += 1
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planercnn_labels(seg)
            labels = _resize_labels(labels, H, W)

            depth = sample["depth"].numpy()
            if depth.ndim == 3:
                depth = depth[0]

            eval_items.append({
                "scene_id": sample["scene_id"],
                "frame_idx": sample["frame_idx"],
                "depth_np": depth,
                "gt_seg_np": gt_seg,
                "K_np": sample["K"].numpy(),
                "c2w_np": sample["c2w"].numpy(),
                "labels": labels,
            })

    else:
        # Synthia / VKITTI2: one H5 per scene
        # Build scene_id → loader mapping
        scene_loaders = {}
        for stem, loader in loaders.items():
            if not stem.endswith("_planercnn"):
                continue
            scene_key = stem[: -len("_planercnn")]
            scene_loaders[scene_key] = loader

        # Group dataset frames by scene_id
        scene_frames = {}  # scene_id → [ds_idx, ...]
        for ds_idx in range(len(dataset)):
            sample_scene_id = dataset.valid_pairs[ds_idx][0] if dataset_key == "synthia" else None
            if dataset_key == "vkitti2":
                # valid_pairs: (h5_path, frame_idx_int, scene, variant, K, fid)
                scene = dataset.valid_pairs[ds_idx][2]
                variant = dataset.valid_pairs[ds_idx][3]
                sample_scene_id = f"{scene}/{variant}"
            elif dataset_key == "synthia":
                sample_scene_id = dataset.valid_pairs[ds_idx][0]  # scene_name
            scene_frames.setdefault(sample_scene_id, []).append(ds_idx)

        # Match scene_id to PlaneRCNN filename
        eval_items = []
        skipped = 0
        for scene_id, ds_indices in scene_frames.items():
            # Convert scene_id to PlaneRCNN filename stem
            if dataset_key == "vkitti2":
                # "Scene18/15-deg-left" → "Scene18_15-deg-left"
                planercnn_key = scene_id.replace("/", "_")
            else:
                planercnn_key = scene_id

            if planercnn_key not in scene_loaders:
                skipped += len(ds_indices)
                continue

            loader = scene_loaders[planercnn_key]
            for order_idx, ds_idx in enumerate(ds_indices):
                seg = loader.get_segmentation_by_index(order_idx)
                if seg is None:
                    skipped += 1
                    continue

                sample = dataset[ds_idx]
                gt_seg = sample["plane"].numpy()
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg = gt_seg.astype(np.int32)
                H, W = gt_seg.shape

                labels = _remap_planercnn_labels(seg)
                labels = _resize_labels(labels, H, W)

                depth = sample["depth"].numpy()
                if depth.ndim == 3:
                    depth = depth[0]

                eval_items.append({
                    "scene_id": sample["scene_id"],
                    "frame_idx": sample["frame_idx"],
                    "depth_np": depth,
                    "gt_seg_np": gt_seg,
                    "K_np": sample["K"].numpy(),
                    "c2w_np": sample["c2w"].numpy(),
                    "labels": labels,
                })

    print(f"[{display}] {len(eval_items)} frames to evaluate ({skipped} skipped)")

    if not eval_items:
        return {}

    outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_eval_frame_standard)(
            item["scene_id"], item["frame_idx"],
            item["depth_np"], item["gt_seg_np"],
            item["K_np"], item["c2w_np"],
            item["labels"], thresholds,
        )
        for item in tqdm(eval_items, desc=display)
    )

    results = {}
    for (metrics, _), item in zip(outputs, eval_items):
        results[(item["scene_id"], item["frame_idx"])] = metrics

    return results


# ============================================================
# MAIN EVALUATION DISPATCHER
# ============================================================

def evaluate_dataset(dataset_key: str, cfg: Dict, max_scenes: int = None,
                     scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneRCNN on a single dataset."""
    display = cfg["display_name"]
    print(f"\n{'='*60}")
    print(f"Evaluating PlaneRCNN on {display}")
    if scene_start is not None or scene_end is not None:
        print(f"  Scene slice: [{scene_start}:{scene_end})")
    print(f"{'='*60}")

    timer = Timer()

    # Discover PlaneRCNN H5 files
    loaders = _discover_planercnn_h5_files(cfg["planercnn_root"])
    if not loaders:
        print(f"[ERROR] No PlaneRCNN H5 files found for {display}")
        return {}

    with timer("evaluation"):
        if dataset_key == "scannetpp":
            results = evaluate_scannetpp(cfg, loaders, max_scenes, scene_start, scene_end)
        elif dataset_key == "hypersim":
            results = evaluate_hypersim(cfg, loaders, max_scenes, scene_start, scene_end)
        else:
            results = evaluate_index_based(dataset_key, cfg, loaders, max_scenes, scene_start, scene_end)

    if not results:
        print(f"[WARN] No results for {display}")
        return {}

    # Save results — use a shard suffix when scene slicing is active
    exp_name = f"planercnn_{EXP_VER}"
    if scene_start is not None or scene_end is not None:
        shard_tag = f"_shard_{scene_start or 0}_{scene_end or 'end'}"
        csv_out_dir = cfg["eval_root"] / (exp_name + shard_tag)
    else:
        csv_out_dir = cfg["eval_root"] / exp_name
    print(f"[SAVE] Saving {len(results)} frames to {csv_out_dir}")
    save_results_csv(results, str(csv_out_dir))
    save_runtime(timer, str(csv_out_dir))
    timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_all(dataset_keys: List[str], configs: Dict):
    """Print cross-dataset summary of PlaneRCNN results."""
    print(f"\n{'='*80}")
    print("PLANERCNN CROSS-DATASET SUMMARY")
    print(f"{'='*80}")

    rows = []
    for dk in dataset_keys:
        cfg = configs[dk]
        exp_name = f"planercnn_{EXP_VER}"
        csv_path = cfg["eval_root"] / exp_name / "results_dataset.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing results for {dk}: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Dataset": cfg["display_name"],
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }

            # Segmentation metrics
            for col, display in [("rand_index_mean", "RI"),
                                  ("voi_mean", "VOI"),
                                  ("sc_mean", "SC")]:
                if col in df.index:
                    row[display] = df[col]

            # Precision/recall at shared thresholds
            for thr in (0.001, 0.005, 0.01):
                thresh_str = f"{thr*100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]

            # Additional thresholds for outdoor datasets
            for thr in (0.02, 0.05, 0.1):
                thresh_str = f"{thr*100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]

            # Binary planarity
            for bp_col in ["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"]:
                mean_col = f"{bp_col}_mean"
                if mean_col in df.index:
                    row[bp_col] = df[mean_col]

            rows.append(row)
            print(f"[OK] {cfg['display_name']}: {row['num_frames']} frames")

        except Exception as e:
            print(f"[ERROR] Failed to read {dk}: {e}")

    if not rows:
        print("[ERROR] No results to aggregate")
        return

    df_all = pd.DataFrame(rows)

    # Print full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df_all.to_string(index=False))
    print("=" * 80)

    return df_all


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PlaneRCNN predictions across multiple datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to evaluate (default: all). "
             "Options: scannetpp, hypersim, synthia, vkitti2, pd",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results, skip evaluation",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Limit scenes per dataset (for testing)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of DataLoader workers (unused, kept for CLI compat)",
    )
    parser.add_argument(
        "--scene-start", type=int, default=None,
        help="Start scene index (inclusive) for parallel job splitting",
    )
    parser.add_argument(
        "--scene-end", type=int, default=None,
        help="End scene index (exclusive) for parallel job splitting",
    )
    args = parser.parse_args()

    all_configs = _get_dataset_configs()
    all_keys = list(all_configs.keys())

    if args.datasets is None:
        dataset_keys = all_keys
    else:
        invalid = set(args.datasets) - set(all_keys)
        if invalid:
            print(f"[ERROR] Invalid datasets: {invalid}")
            print(f"[ERROR] Valid options: {all_keys}")
            return
        dataset_keys = args.datasets

    print(f"[CONFIG] Datasets: {dataset_keys}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Scene slice: [{args.scene_start}:{args.scene_end})")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] Experiment version: {EXP_VER}")

    if not args.aggregate_only:
        for dk in dataset_keys:
            evaluate_dataset(dk, all_configs[dk], args.max_scenes,
                             args.scene_start, args.scene_end)

    # Only aggregate when running full evaluation (no scene slicing)
    if args.scene_start is None and args.scene_end is None:
        aggregate_all(dataset_keys, all_configs)
    print("\n[DONE] PlaneRCNN evaluation complete!")


if __name__ == "__main__":
    main()
