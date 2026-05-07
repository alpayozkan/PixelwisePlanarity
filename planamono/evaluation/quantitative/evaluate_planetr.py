#!/usr/bin/env python3
"""
Unified evaluation of PlaneTR predictions across 4 datasets.

PlaneTR H5 format (identical across all datasets):
    frame_ids   (N,)          object (byte strings, e.g. b'frame_000000')
    planes      (N, 480, 640) uint16

Label convention: 20 = non-planar (same as ZeroPlane), plane IDs 1-19.
Remapped to standard: 0 = non-planar.

File layout:
    ScanNet++:  <root>/scannetpp/<scene_id>/rendered_v2.h5
    Hypersim:   <root>/hypersim/<scene_id>/rendered_planes_cam_XX.h5
    Synthia:    <root>/synthia/<scene_name>/rendered_v2.h5
    VKITTI2:    <root>/vkitti2/<Scene>/<variant>/rendered_v2.h5

Frame matching:
    ScanNet++:  name-based — frame_ids contain 'frame_XXXXXX' matching GT frame_idx
    Hypersim:   index-based — PlaneTR index i → GT index i (per scene+camera)
    Synthia:    index-based — PlaneTR index i → GT index i
    VKITTI2:    index-based — PlaneTR index i → GT index i

Usage:
    python evaluate_planetr.py --resolution lowres              # All datasets, lowres
    python evaluate_planetr.py --resolution highres             # All datasets, highres
    python evaluate_planetr.py --datasets scannetpp hypersim    # Specific datasets
    python evaluate_planetr.py --max-scenes 1                   # Quick test
    python evaluate_planetr.py --aggregate-only                 # Re-aggregate
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
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path


# ============================================================
# CONFIGURATION
# ============================================================

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
N_JOBS = min(16, os.cpu_count())
EXP_VER = "v1"

NONPLANAR_LABEL = 20  # PlaneTR uses 20 for non-planar (same as ZeroPlane)

PLANETR_ROOTS = {
    "lowres": "/cluster/scratch/ayavuz/dataset/planrectr_lowres",
    "highres": "/cluster/scratch/ayavuz/dataset/planrectr_highres",
}


# ============================================================
# PlaneTR H5 LOADER
# ============================================================

class PlaneTRH5Loader:
    """Load PlaneTR predictions from a single H5 file.

    H5 structure:
        frame_ids   (N,)          object (byte strings)
        planes      (N, 480, 640) uint16
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            # Decode byte strings to regular strings
            self._frame_ids = [
                fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                for fid in f["frame_ids"][:]
            ]
            self._num_frames = f["planes"].shape[0]
        self._id_to_idx = {fid: i for i, fid in enumerate(self._frame_ids)}

    @property
    def frame_ids(self) -> List[str]:
        return self._frame_ids

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def get_segmentation(self, frame_id: str) -> Optional[np.ndarray]:
        """Get segmentation by exact frame ID (e.g. 'frame_000000')."""
        idx = self._id_to_idx.get(frame_id)
        if idx is None:
            return None
        return self.get_segmentation_by_index(idx)

    def get_segmentation_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Get segmentation by sequential index."""
        if idx < 0 or idx >= self._num_frames:
            return None
        with h5py.File(self.h5_path, "r") as f:
            seg = f["planes"][idx]
        return seg  # (480, 640) uint16


def _remap_planetr_labels(seg: np.ndarray) -> np.ndarray:
    """Remap PlaneTR segmentation to standard convention.

    PlaneTR: 20 = non-planar, 1-19 = plane instances, 0 = void/unlabeled.
    Standard: 0 = non-planar, 1+ = plane instances.
    """
    out = seg.astype(np.int32)
    out[out == NONPLANAR_LABEL] = 0
    return out


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

def _get_dataset_configs(planetr_root: str) -> Dict:
    """Return per-dataset configuration dicts."""
    splits_root = os.path.join(repo_path, "splits")

    return {
        "scannetpp": {
            "planetr_root": os.path.join(planetr_root, "scannetpp"),
            "dataset_class": "ScanNetPPPlaneDataset",
            "dataset_kwargs": {
                "rgb_root": os.path.join(scannetpp_path, "data"),
                "plane_label_root": scannetpp_rend_plane_path,
                "sem_label_root": scannetpp_rend_plane_path,
                "depth_label_root": scannetpp_rend_plane_path,
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
            "planetr_root": os.path.join(planetr_root, "hypersim"),
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
            "planetr_root": os.path.join(planetr_root, "synthia"),
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
            "planetr_root": os.path.join(planetr_root, "vkitti2"),
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
    else:
        raise ValueError(f"Unknown dataset class: {cls_name}")


# ============================================================
# PlaneTR H5 DISCOVERY
# ============================================================

def _discover_planetr_scannetpp(planetr_root: str) -> Dict[str, PlaneTRH5Loader]:
    """Discover PlaneTR H5 files for ScanNet++.

    Layout: <root>/<scene_id>/rendered_v2.h5
    Returns {scene_id: loader}.
    """
    loaders = {}
    root = Path(planetr_root)
    if not root.exists():
        print(f"[ERROR] PlaneTR root does not exist: {root}")
        return loaders

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        h5_path = scene_dir / "rendered_v2.h5"
        if h5_path.exists():
            loaders[scene_dir.name] = PlaneTRH5Loader(str(h5_path))

    print(f"[PlaneTR] Found {len(loaders)} ScanNet++ scenes in {planetr_root}")
    return loaders


def _discover_planetr_hypersim(planetr_root: str) -> Dict[Tuple[str, str], PlaneTRH5Loader]:
    """Discover PlaneTR H5 files for Hypersim.

    Layout: <root>/<scene_id>/rendered_planes_cam_XX.h5
    Returns {(scene_id, cam_name): loader}.
    """
    loaders = {}
    root = Path(planetr_root)
    if not root.exists():
        print(f"[ERROR] PlaneTR root does not exist: {root}")
        return loaders

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for h5_file in sorted(scene_dir.glob("rendered_planes_cam_*.h5")):
            # Extract cam name: "rendered_planes_cam_00.h5" → "cam_00"
            cam_name = h5_file.stem.replace("rendered_planes_", "")
            loaders[(scene_dir.name, cam_name)] = PlaneTRH5Loader(str(h5_file))

    print(f"[PlaneTR] Found {len(loaders)} Hypersim scene+camera files in {planetr_root}")
    return loaders


def _discover_planetr_synthia(planetr_root: str) -> Dict[str, PlaneTRH5Loader]:
    """Discover PlaneTR H5 files for Synthia.

    Layout: <root>/<scene_name>/rendered_v2.h5
    Returns {scene_name: loader}.
    """
    loaders = {}
    root = Path(planetr_root)
    if not root.exists():
        print(f"[ERROR] PlaneTR root does not exist: {root}")
        return loaders

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        h5_path = scene_dir / "rendered_v2.h5"
        if h5_path.exists():
            loaders[scene_dir.name] = PlaneTRH5Loader(str(h5_path))

    print(f"[PlaneTR] Found {len(loaders)} Synthia scenes in {planetr_root}")
    return loaders


def _discover_planetr_vkitti2(planetr_root: str) -> Dict[str, PlaneTRH5Loader]:
    """Discover PlaneTR H5 files for VKITTI2.

    Layout: <root>/<Scene>/<variant>/rendered_v2.h5
    Returns {"Scene/variant": loader}.
    """
    loaders = {}
    root = Path(planetr_root)
    if not root.exists():
        print(f"[ERROR] PlaneTR root does not exist: {root}")
        return loaders

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for variant_dir in sorted(scene_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            h5_path = variant_dir / "rendered_v2.h5"
            if h5_path.exists():
                key = f"{scene_dir.name}/{variant_dir.name}"
                loaders[key] = PlaneTRH5Loader(str(h5_path))

    print(f"[PlaneTR] Found {len(loaders)} VKITTI2 scene/variant combos in {planetr_root}")
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


def evaluate_scannetpp(cfg: Dict, max_scenes: int = None,
                       scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneTR on ScanNet++ (name-based frame matching).

    Processes scene-by-scene to avoid OOM from pre-loading all frames.
    """
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]

    # Discover PlaneTR H5 files: {scene_id: loader}
    scene_loaders = _discover_planetr_scannetpp(cfg["planetr_root"])

    available = sorted([s for s in dataset.scene_ids if s in scene_loaders])
    if scene_start is not None or scene_end is not None:
        available = available[scene_start:scene_end]
        scene_loaders = {k: v for k, v in scene_loaders.items() if k in available}
    print(f"[ScanNet++] Evaluating {len(available)} scenes (slice [{scene_start}:{scene_end}))")

    # Build per-scene GT frame lists: {scene_id: [(dataset_idx, gt_frame_id), ...]}
    scene_frames = {}
    for ds_idx in range(len(dataset)):
        pair = dataset.valid_pairs[ds_idx]
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        frame_idx_str = os.path.splitext(os.path.basename(rgb_path))[0]  # "frame_000000"
        if scene_id not in scene_loaders:
            continue
        scene_frames.setdefault(scene_id, []).append((ds_idx, frame_idx_str))

    total_frames = sum(len(v) for v in scene_frames.values())
    print(f"[ScanNet++] {total_frames} frames across {len(scene_frames)} scenes")

    # Process scene-by-scene to bound memory
    results = {}
    total_skipped = 0
    pbar = tqdm(total=total_frames, desc="ScanNet++")
    for scene_id in sorted(scene_frames.keys()):
        frames = scene_frames[scene_id]
        loader = scene_loaders[scene_id]

        eval_items = []
        for ds_idx, gt_frame_id in frames:
            seg = loader.get_segmentation(gt_frame_id)
            if seg is None:
                total_skipped += 1
                pbar.update(1)
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planetr_labels(seg)
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

        if eval_items:
            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_eval_frame_standard)(
                    item["scene_id"], item["frame_idx"],
                    item["depth_np"], item["gt_seg_np"],
                    item["K_np"], item["c2w_np"],
                    item["labels"], thresholds,
                )
                for item in eval_items
            )
            for (metrics, _), item in zip(outputs, eval_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics
            pbar.update(len(eval_items))
        del eval_items

    pbar.close()
    print(f"[ScanNet++] {len(results)} frames evaluated ({total_skipped} skipped)")

    return results


def evaluate_hypersim(cfg: Dict, max_scenes: int = None,
                      scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneTR on Hypersim (index-based, per scene+camera).

    Processes per scene+camera to avoid OOM from pre-loading all frames.
    """
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]

    # Discover: {(scene_id, cam_name): loader}
    cam_loaders = _discover_planetr_hypersim(cfg["planetr_root"])

    # Apply scene slicing on unique scene IDs
    if scene_start is not None or scene_end is not None:
        all_scene_ids = sorted(set(k[0] for k in cam_loaders.keys()))
        keep_scenes = set(all_scene_ids[scene_start:scene_end])
        cam_loaders = {k: v for k, v in cam_loaders.items() if k[0] in keep_scenes}

    print(f"[Hypersim] Found {len(cam_loaders)} scene+camera PlaneTR files"
          f" (slice [{scene_start}:{scene_end}))")

    # Group GT frames by (scene_id, cam_name)
    cam_frames = {}  # (scene_id, cam_name) → [ds_idx, ...]
    for ds_idx in range(len(dataset)):
        pair = dataset.valid_pairs[ds_idx]
        scene_id, cam_name = pair[0], pair[1]
        cam_frames.setdefault((scene_id, cam_name), []).append(ds_idx)

    total_frames = sum(
        len(v) for k, v in cam_frames.items() if k in cam_loaders
    )
    print(f"[Hypersim] {total_frames} matched frames across {len(cam_loaders)} scene+camera combos")

    # Process per scene+camera to bound memory
    results = {}
    total_skipped = 0
    pbar = tqdm(total=total_frames, desc="Hypersim")
    for (scene_id, cam_name), ds_indices in sorted(cam_frames.items()):
        key = (scene_id, cam_name)
        if key not in cam_loaders:
            continue

        loader = cam_loaders[key]
        eval_items = []

        for order_idx, ds_idx in enumerate(ds_indices):
            seg = loader.get_segmentation_by_index(order_idx)
            if seg is None:
                total_skipped += 1
                pbar.update(1)
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planetr_labels(seg)
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

        if eval_items:
            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_eval_frame_hypersim)(
                    item["scene_id"], item["frame_idx"],
                    item["depth_euc_np"], item["gt_seg_np"],
                    item["M_cam_from_uv"], item["native_wh"],
                    item["c2w_np"], item["labels"], thresholds,
                )
                for item in eval_items
            )
            for (metrics, _), item in zip(outputs, eval_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics
            pbar.update(len(eval_items))
        del eval_items

    pbar.close()
    print(f"[Hypersim] {len(results)} frames evaluated ({total_skipped} skipped)")

    return results


def evaluate_index_based(
    dataset_key: str, cfg: Dict,
    max_scenes: int = None,
    scene_start: int = None, scene_end: int = None,
):
    """Evaluate PlaneTR on Synthia / VKITTI2 (index-based matching)."""
    kwargs = dict(cfg["dataset_kwargs"])
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    cfg_copy = dict(cfg)
    cfg_copy["dataset_kwargs"] = kwargs
    dataset = _load_dataset(cfg_copy)
    thresholds = cfg["thresholds"]
    display = cfg["display_name"]

    # Discover loaders
    if dataset_key == "synthia":
        scene_loaders = _discover_planetr_synthia(cfg["planetr_root"])
    elif dataset_key == "vkitti2":
        scene_loaders = _discover_planetr_vkitti2(cfg["planetr_root"])
    else:
        raise ValueError(f"Unknown index-based dataset: {dataset_key}")

    # Group dataset frames by scene_id
    scene_frames = {}  # scene_key → [ds_idx, ...]
    for ds_idx in range(len(dataset)):
        if dataset_key == "vkitti2":
            # valid_pairs: (h5_path, frame_idx_int, scene, variant, K, fid)
            scene = dataset.valid_pairs[ds_idx][2]
            variant = dataset.valid_pairs[ds_idx][3]
            sample_scene_id = f"{scene}/{variant}"
        elif dataset_key == "synthia":
            sample_scene_id = dataset.valid_pairs[ds_idx][0]  # scene_name
        scene_frames.setdefault(sample_scene_id, []).append(ds_idx)

    # Process scene-by-scene to bound memory
    total_frames = sum(
        len(v) for k, v in scene_frames.items() if k in scene_loaders
    )
    print(f"[{display}] {total_frames} matched frames across {len(scene_loaders)} scenes")

    results = {}
    total_skipped = 0
    pbar = tqdm(total=total_frames, desc=display)
    for scene_id in sorted(scene_frames.keys()):
        ds_indices = scene_frames[scene_id]
        if scene_id not in scene_loaders:
            continue

        loader = scene_loaders[scene_id]
        eval_items = []

        for order_idx, ds_idx in enumerate(ds_indices):
            seg = loader.get_segmentation_by_index(order_idx)
            if seg is None:
                total_skipped += 1
                pbar.update(1)
                continue

            sample = dataset[ds_idx]
            gt_seg = sample["plane"].numpy()
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            gt_seg = gt_seg.astype(np.int32)
            H, W = gt_seg.shape

            labels = _remap_planetr_labels(seg)
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

        if eval_items:
            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_eval_frame_standard)(
                    item["scene_id"], item["frame_idx"],
                    item["depth_np"], item["gt_seg_np"],
                    item["K_np"], item["c2w_np"],
                    item["labels"], thresholds,
                )
                for item in eval_items
            )
            for (metrics, _), item in zip(outputs, eval_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics
            pbar.update(len(eval_items))
        del eval_items

    pbar.close()
    print(f"[{display}] {len(results)} frames evaluated ({total_skipped} skipped)")

    return results


# ============================================================
# MAIN EVALUATION DISPATCHER
# ============================================================

def evaluate_dataset(dataset_key: str, cfg: Dict, exp_name: str,
                     max_scenes: int = None,
                     scene_start: int = None, scene_end: int = None):
    """Evaluate PlaneTR on a single dataset."""
    display = cfg["display_name"]
    print(f"\n{'='*60}")
    print(f"Evaluating PlaneTR on {display}")
    if scene_start is not None or scene_end is not None:
        print(f"  Scene slice: [{scene_start}:{scene_end})")
    print(f"{'='*60}")

    timer = Timer()

    with timer("evaluation"):
        if dataset_key == "scannetpp":
            results = evaluate_scannetpp(cfg, max_scenes, scene_start, scene_end)
        elif dataset_key == "hypersim":
            results = evaluate_hypersim(cfg, max_scenes, scene_start, scene_end)
        else:
            results = evaluate_index_based(dataset_key, cfg, max_scenes, scene_start, scene_end)

    if not results:
        print(f"[WARN] No results for {display}")
        return {}

    # Save results — use a shard suffix when scene slicing is active
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

def aggregate_all(dataset_keys: List[str], configs: Dict, exp_name: str):
    """Print cross-dataset summary of PlaneTR results."""
    print(f"\n{'='*80}")
    print(f"PLANETR CROSS-DATASET SUMMARY ({exp_name})")
    print(f"{'='*80}")

    rows = []
    for dk in dataset_keys:
        cfg = configs[dk]
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
        description="Evaluate PlaneTR predictions across multiple datasets"
    )
    parser.add_argument(
        "--resolution", choices=["lowres", "highres"], required=True,
        help="PlaneTR resolution variant to evaluate",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to evaluate (default: all). "
             "Options: scannetpp, hypersim, synthia, vkitti2",
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
        "--scene-start", type=int, default=None,
        help="Start scene index (inclusive) for parallel job splitting",
    )
    parser.add_argument(
        "--scene-end", type=int, default=None,
        help="End scene index (exclusive) for parallel job splitting",
    )
    args = parser.parse_args()

    planetr_root = PLANETR_ROOTS[args.resolution]
    exp_name = f"planetr_{args.resolution}_{EXP_VER}"

    all_configs = _get_dataset_configs(planetr_root)
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

    print(f"[CONFIG] Resolution: {args.resolution}")
    print(f"[CONFIG] PlaneTR root: {planetr_root}")
    print(f"[CONFIG] Experiment name: {exp_name}")
    print(f"[CONFIG] Datasets: {dataset_keys}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Scene slice: [{args.scene_start}:{args.scene_end})")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    if not args.aggregate_only:
        for dk in dataset_keys:
            evaluate_dataset(dk, all_configs[dk], exp_name, args.max_scenes,
                             args.scene_start, args.scene_end)

    # Only aggregate when running full evaluation (no scene slicing)
    if args.scene_start is None and args.scene_end is None:
        aggregate_all(dataset_keys, all_configs, exp_name)
    print("\n[DONE] PlaneTR evaluation complete!")


if __name__ == "__main__":
    main()
