"""Generate qualitative comparison report: plane segmentation + inlier/outlier visualizations.

Produces per-sample PNGs and a combined PDF/PNG report across methods and datasets.

For each sample frame:
  Row 1: Plane segmentation (RGB | GT | Method1 | Method2 | ...)
  Row 2: Inlier/outlier overlay at middle threshold (green=inlier, red=outlier)

Middle thresholds:
  Indoor (ScanNet++, Hypersim): 0.5cm = 0.005m
  Outdoor (Synthia, VKITTI2): 5.0cm = 0.05m

Usage:
    python generate_qualitative_report.py --n-samples 5 --pdf --png
    python generate_qualitative_report.py --n-samples 10 --datasets scannetpp hypersim
    python generate_qualitative_report.py --n-samples 3 --datasets all --seed 123
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    backproject_mcam,
    fit_planes_per_label_v1,
)
from planamono.paths import repo_path

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("/cluster/scratch/aoezkan/planeseg/eval/qualitative")

# Middle thresholds for inlier visualization
INDOOR_THRESHOLD = 0.01    # 1.0cm
OUTDOOR_THRESHOLD = 0.05   # 5.0cm

RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

# --- ScanNet++ ---
H5_ROOT_SCANNETPP = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
SCANNETPP_RGB_ROOT = "/cluster/project/cvg/Shared_datasets/scannet++/data"
SCANNETPP_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

# Shared method definitions — H5 folders are consistent across all datasets.
# Display names match create_unified_tables.py (quantitative report).
SHARED_METHODS = {
    "gt": {
        "h5_folder": None, "display_name": "GT (upper bound)",
        "nonplanar_label": None, "uses_gt_h5": True,
    },
    "ours": {
        "h5_folder": "moge_hires_4ds_ep2_h5", "display_name": "MoGe (Ours)",
        "nonplanar_label": None, "uses_gt_h5": False,
    },
    "zeroplane_finetuned_dinov2_60k": {
        "h5_folder": "zeroplane_all_h5_dinov2_moge_60k_h5",
        "display_name": "ZeroPlane (finetuned, dinov2, 60k)",
        "nonplanar_label": 20, "uses_gt_h5": False,
    },
    "zeroplane_released": {
        "h5_folder": "zeroplane_default_dust3r_released_h5",
        "display_name": "ZeroPlane (released)",
        "nonplanar_label": 20, "uses_gt_h5": False,
    },
    "ours_indoors": {
        "h5_folder": "moge_hires_ep2_h5", "display_name": "MoGe (Ours Indoors)",
        "nonplanar_label": None, "uses_gt_h5": False,
    },
    "zeroplane_finetuned_indoors_dinov2_60k": {
        "h5_folder": "zeroplane_mixed_h5_dinov2_moge_60k_h5",
        "display_name": "ZeroPlane (finetuned Indoors, dinov2, 60k)",
        "nonplanar_label": 20, "uses_gt_h5": False,
    },
    "planeTR": {
        "h5_folder": "planeTR_lines_h5", "display_name": "PlaneTR",
        "nonplanar_label": 21, "uses_gt_h5": False,
    },
    "moge_4ds_ep2_v5_rel": {
        "h5_folder": "moge_hires_4ds_ep2_v5_relative_seg_h5",
        "display_name": "MoGe 4ds ep2 v5_rel",
        "nonplanar_label": None, "uses_gt_h5": False,
    },
    "moge_4ds_ep2_v5origparams_rel": {
        "h5_folder": "moge_hires_4ds_ep2_v5origparams_relative_seg_h5",
        "display_name": "MoGe 4ds ep2 v5orig_rel",
        "nonplanar_label": None, "uses_gt_h5": False,
    },
}

# --- Hypersim ---
H5_ROOT_HYPERSIM = Path("/cluster/scratch/aoezkan/planeseg/hypersim/inference")
HYPERSIM_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYPERSIM_PLANE_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYPERSIM_PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

# --- Synthia ---
H5_ROOT_SYNTHIA = Path("/cluster/scratch/aoezkan/planeseg/synthia/inference")
SYNTHIA_ROOT = "/cluster/scratch/ayavuz/dataset/synthia_planes"

# --- VKITTI2 ---
H5_ROOT_VKITTI2 = Path("/cluster/scratch/aoezkan/planeseg/vkitti2/inference")
VKITTI2_ROOT = "/cluster/scratch/ayavuz/dataset/vkitti2_planes"

# --- PlaneRCNN ---
PLANERCNN_SCANNETPP_ROOT = Path("/cluster/scratch/ayavuz/dataset/planercnn_scannetpp_test")
PLANERCNN_HYPERSIM_ROOT = Path("/cluster/scratch/ayavuz/dataset/planercnn_hypersim_test")
PLANERCNN_SYNTHIA_ROOT = Path("/cluster/scratch/ayavuz/dataset/planercnn_synthia_test")
PLANERCNN_VKITTI2_ROOT = Path("/cluster/scratch/ayavuz/dataset/planercnn_vkitti2_test")

ALL_DATASETS = ["scannetpp", "hypersim", "synthia", "vkitti2"]

# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def get_random_cmap(num_classes: int, seed: int = 42) -> np.ndarray:
    """Generate random colors for plane labels. Returns (num_classes, 3) uint8."""
    rng = np.random.RandomState(seed)
    colors = rng.randint(40, 240, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background = black
    return colors


def overlay_inliers(rgb: np.ndarray, inlier_mask: np.ndarray,
                    outlier_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay inliers (green) and outliers (red) on RGB image."""
    out = rgb.astype(np.float32).copy()
    if inlier_mask.any():
        out[inlier_mask] = out[inlier_mask] * (1 - alpha) + np.array([0, 200, 0], dtype=np.float32) * alpha
    if outlier_mask.any():
        out[outlier_mask] = out[outlier_mask] * (1 - alpha) + np.array([200, 0, 0], dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_inlier_mask(
    plane_seg: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    distance_threshold: float,
    inlier_ratio_gate: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute per-pixel inlier and outlier masks via RANSAC plane fitting."""
    H, W = depth.shape
    inlier_mask = np.zeros((H, W), dtype=bool)
    outlier_mask = np.zeros((H, W), dtype=bool)

    pts_world, labels, valid_idx = backproject(depth, K, c2w, plane_seg)
    if pts_world.shape[0] == 0:
        return inlier_mask, outlier_mask, {"precision": 0, "recall": 0, "num_inliers": 0}

    plane_results, _ = fit_planes_per_label_v1(
        pts_world, labels,
        distance_threshold=distance_threshold,
        num_iterations=RANSAC_ITERATIONS,
    )

    total_inliers = 0
    total_planar = 0

    for plane_id, result in plane_results.items():
        if plane_id == 0:
            continue
        plane_pts_mask = labels == plane_id
        n_pts = plane_pts_mask.sum()
        if n_pts == 0:
            continue

        total_planar += n_pts
        plane_model = result.get("plane_model_refined", result.get("plane_model"))
        if plane_model is None:
            pixel_idx = valid_idx[plane_pts_mask]
            rows, cols = np.unravel_index(pixel_idx, (H, W))
            outlier_mask[rows, cols] = True
            continue

        a, b, c, d = plane_model
        pts_plane = pts_world[plane_pts_mask]
        distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
        is_inlier = distances < distance_threshold

        inlier_ratio = is_inlier.sum() / n_pts
        pixel_idx = valid_idx[plane_pts_mask]
        rows, cols = np.unravel_index(pixel_idx, (H, W))

        if inlier_ratio >= inlier_ratio_gate:
            inlier_mask[rows[is_inlier], cols[is_inlier]] = True
            outlier_mask[rows[~is_inlier], cols[~is_inlier]] = True
            total_inliers += is_inlier.sum()
        else:
            outlier_mask[rows, cols] = True

    total_pts = (depth > 0).sum()
    precision = total_inliers / total_planar if total_planar > 0 else 0
    recall = total_inliers / total_pts if total_pts > 0 else 0

    return inlier_mask, outlier_mask, {
        "precision": precision, "recall": recall, "num_inliers": int(total_inliers),
    }


def compute_inlier_mask_hypersim(
    plane_seg: np.ndarray,
    depth_euc: np.ndarray,
    M_cam_from_uv: np.ndarray,
    native_wh: Tuple[int, int],
    c2w: np.ndarray,
    distance_threshold: float,
    inlier_ratio_gate: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute inlier/outlier masks for Hypersim using backproject_mcam."""
    H, W = depth_euc.shape
    inlier_mask = np.zeros((H, W), dtype=bool)
    outlier_mask = np.zeros((H, W), dtype=bool)

    pts_world, labels, valid_idx = backproject_mcam(
        depth_euc, M_cam_from_uv, native_wh[0], native_wh[1], c2w, plane_seg)
    if pts_world.shape[0] == 0:
        return inlier_mask, outlier_mask, {"precision": 0, "recall": 0, "num_inliers": 0}

    plane_results, _ = fit_planes_per_label_v1(
        pts_world, labels,
        distance_threshold=distance_threshold,
        num_iterations=RANSAC_ITERATIONS,
    )

    total_inliers = 0
    total_planar = 0

    for plane_id, result in plane_results.items():
        if plane_id == 0:
            continue
        plane_pts_mask = labels == plane_id
        n_pts = plane_pts_mask.sum()
        if n_pts == 0:
            continue

        total_planar += n_pts
        plane_model = result.get("plane_model_refined", result.get("plane_model"))
        if plane_model is None:
            pixel_idx = valid_idx[plane_pts_mask]
            rows, cols = np.unravel_index(pixel_idx, (H, W))
            outlier_mask[rows, cols] = True
            continue

        a, b, c, d = plane_model
        pts_plane = pts_world[plane_pts_mask]
        distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
        is_inlier = distances < distance_threshold

        inlier_ratio = is_inlier.sum() / n_pts
        pixel_idx = valid_idx[plane_pts_mask]
        rows, cols = np.unravel_index(pixel_idx, (H, W))

        if inlier_ratio >= inlier_ratio_gate:
            inlier_mask[rows[is_inlier], cols[is_inlier]] = True
            outlier_mask[rows[~is_inlier], cols[~is_inlier]] = True
            total_inliers += is_inlier.sum()
        else:
            outlier_mask[rows, cols] = True

    total_pts = (depth_euc > 0).sum()
    precision = total_inliers / total_planar if total_planar > 0 else 0
    recall = total_inliers / total_pts if total_pts > 0 else 0

    return inlier_mask, outlier_mask, {
        "precision": precision, "recall": recall, "num_inliers": int(total_inliers),
    }


# ============================================================
# DATA LOADING
# ============================================================

class LazyH5Loader:
    """Load plane labels from H5 prediction files, one scene at a time."""

    def __init__(self, h5_root: Path, h5_folder: str, nonplanar_label: Optional[int] = None,
                 h5_filename: str = "planes.h5"):
        self.h5_root = h5_root
        self.h5_folder = h5_folder
        self.nonplanar_label = nonplanar_label
        self.h5_filename = h5_filename
        self._cache_scene = None
        self._cache_fids = None
        self._cache_planes = None

    def load(self, scene_id: str, frame_idx: str) -> Optional[np.ndarray]:
        """Load plane labels for a given scene and frame."""
        if self._cache_scene != scene_id:
            if self.h5_folder:
                h5_path = self.h5_root / self.h5_folder / scene_id / self.h5_filename
            else:
                h5_path = self.h5_root / scene_id / self.h5_filename
            if not h5_path.exists():
                return None
            with h5py.File(str(h5_path), "r") as f:
                self._cache_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                                    for fid in f["frame_ids"][:]]
                self._cache_planes = f["planes"][:]
            self._cache_scene = scene_id

        try:
            idx = self._cache_fids.index(frame_idx)
        except ValueError:
            for i, fid in enumerate(self._cache_fids):
                if fid.lstrip("0") == frame_idx.lstrip("0") or fid == frame_idx:
                    idx = i
                    break
            else:
                return None

        plane_seg = self._cache_planes[idx].astype(np.int32)
        if self.nonplanar_label is not None:
            plane_seg[plane_seg == self.nonplanar_label] = 0
        return plane_seg


class LazyH5LoaderPerCamera:
    """Load plane labels from per-camera H5 files (Hypersim format)."""

    def __init__(self, h5_root: Path, h5_folder: str, nonplanar_label: Optional[int] = None,
                 h5_prefix: str = "planes"):
        self.h5_root = h5_root
        self.h5_folder = h5_folder
        self.nonplanar_label = nonplanar_label
        self.h5_prefix = h5_prefix
        self._cache_key = None
        self._cache_fids = None
        self._cache_planes = None

    def load(self, scene_id: str, cam_name: str, frame_idx: str) -> Optional[np.ndarray]:
        """Load plane labels for a given scene, camera, and frame."""
        cache_key = f"{scene_id}/{cam_name}"
        if self._cache_key != cache_key:
            if self.h5_folder:
                h5_path = self.h5_root / self.h5_folder / scene_id / f"{self.h5_prefix}_{cam_name}.h5"
            else:
                h5_path = self.h5_root / scene_id / f"{self.h5_prefix}_{cam_name}.h5"
            if not h5_path.exists():
                return None
            with h5py.File(str(h5_path), "r") as f:
                self._cache_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                                    for fid in f["frame_ids"][:]]
                self._cache_planes = f["planes"][:]
            self._cache_key = cache_key

        try:
            idx = self._cache_fids.index(frame_idx)
        except ValueError:
            for i, fid in enumerate(self._cache_fids):
                if fid.lstrip("0") == frame_idx.lstrip("0") or fid == frame_idx:
                    idx = i
                    break
            else:
                return None

        plane_seg = self._cache_planes[idx].astype(np.int32)
        if self.nonplanar_label is not None:
            plane_seg[plane_seg == self.nonplanar_label] = 0
        return plane_seg


class PlaneRCNNH5Loader:
    """Load plane labels from PlaneRCNN H5 files (frames/<fid>/plane_segmentation)."""

    def __init__(self, h5_root: Path, h5_folder: str = None, nonplanar_label=None):
        self.h5_root = h5_root
        self._cache_scene = None
        self._cache_h5 = None

    @staticmethod
    def _normalize_fid(frame_idx: str) -> str:
        """Strip 'frame_' prefix and zero-pad to 6 digits."""
        fid = frame_idx.replace("frame_", "")
        return fid.zfill(6)

    def load(self, scene_id: str, frame_idx: str) -> Optional[np.ndarray]:
        if self._cache_scene != scene_id:
            h5_path = self.h5_root / f"{scene_id}_planercnn.h5"
            if not h5_path.exists():
                return None
            if self._cache_h5 is not None:
                self._cache_h5.close()
            self._cache_h5 = h5py.File(str(h5_path), "r")
            self._cache_scene = scene_id

        fid = self._normalize_fid(frame_idx)
        if f"frames/{fid}" in self._cache_h5:
            return self._cache_h5[f"frames/{fid}/plane_segmentation"][:].astype(np.int32)
        return None


class PlaneRCNNH5LoaderPerCamera:
    """Load plane labels from PlaneRCNN per-camera H5 files."""

    def __init__(self, h5_root: Path, h5_folder: str = None, nonplanar_label=None):
        self.h5_root = h5_root
        self._cache_key = None
        self._cache_h5 = None

    def load(self, scene_id: str, cam_name: str, frame_idx: str) -> Optional[np.ndarray]:
        cache_key = f"{scene_id}/{cam_name}"
        if self._cache_key != cache_key:
            h5_path = self.h5_root / f"{scene_id}_{cam_name}_planercnn.h5"
            if not h5_path.exists():
                return None
            if self._cache_h5 is not None:
                self._cache_h5.close()
            self._cache_h5 = h5py.File(str(h5_path), "r")
            self._cache_key = cache_key

        fid = frame_idx.zfill(6)
        if f"frames/{fid}" in self._cache_h5:
            return self._cache_h5[f"frames/{fid}/plane_segmentation"][:].astype(np.int32)
        if f"frames/{frame_idx}" in self._cache_h5:
            return self._cache_h5[f"frames/{frame_idx}/plane_segmentation"][:].astype(np.int32)
        return None


class PlaneRCNNReindexedH5Loader:
    """PlaneRCNN loader for re-indexed datasets (Synthia, VKITTI2).

    PlaneRCNN re-indexes frames 0-based, so frame N in the PlaneRCNN H5
    corresponds to the Nth frame in the dataset's scene_data.h5.
    Uses scene_data.h5 frame_ids to map dataset frame_idx -> PlaneRCNN index.
    """

    def __init__(self, h5_root, h5_folder=None, nonplanar_label=None,
                 data_root=None, scene_id_to_filename=None):
        self.h5_root = Path(h5_root)
        self.data_root = Path(data_root) if data_root else None
        self.scene_id_to_filename = scene_id_to_filename  # e.g., lambda s: s.replace("/", "_")
        self._cache_scene = None
        self._cache_h5 = None
        self._fid_to_idx = {}

    def _filename_scene_id(self, scene_id: str) -> str:
        if self.scene_id_to_filename:
            return self.scene_id_to_filename(scene_id)
        return scene_id

    def load(self, scene_id: str, frame_idx: str) -> Optional[np.ndarray]:
        if self._cache_scene != scene_id:
            fname_id = self._filename_scene_id(scene_id)
            h5_path = self.h5_root / f"{fname_id}_planercnn.h5"
            if not h5_path.exists():
                return None
            if self._cache_h5 is not None:
                self._cache_h5.close()
            self._cache_h5 = h5py.File(str(h5_path), "r")
            self._cache_scene = scene_id

            # Build frame_idx -> PlaneRCNN 0-based index from scene_data.h5
            self._fid_to_idx = {}
            if self.data_root:
                scene_h5 = self.data_root / scene_id / "scene_data.h5"
                if scene_h5.exists():
                    with h5py.File(str(scene_h5), "r") as sf:
                        for i, fid in enumerate(sf["frame_ids"][:]):
                            fid_str = fid.decode() if isinstance(fid, bytes) else str(fid)
                            self._fid_to_idx[fid_str] = f"{i:06d}"

        # Map dataset frame_idx -> PlaneRCNN 0-based index
        planercnn_fid = self._fid_to_idx.get(frame_idx)
        if planercnn_fid is None:
            for k, v in self._fid_to_idx.items():
                if k.lstrip("0") == frame_idx.lstrip("0"):
                    planercnn_fid = v
                    break
        if planercnn_fid is None:
            return None

        key = f"frames/{planercnn_fid}/plane_segmentation"
        if key in self._cache_h5:
            return self._cache_h5[key][:].astype(np.int32)
        return None


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_sample_figure(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    gt_seg: np.ndarray,
    method_segs: Dict[str, np.ndarray],
    method_names: Dict[str, str],
    distance_threshold: float,
    scene_id: str,
    frame_idx: str,
    dataset_label: str = "",
    compute_inlier_fn=None,
) -> plt.Figure:
    """Generate a 2-row comparison figure for one sample.

    Row 1: Plane segmentation (RGB, GT, methods...)
    Row 2: Inlier/outlier overlay (RGB, GT, methods...)
    """
    n_methods = len(method_segs)
    n_cols = 2 + n_methods

    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    all_labels = set(np.unique(gt_seg))
    for seg in method_segs.values():
        all_labels.update(np.unique(seg))
    max_label = max(all_labels) + 1
    colors = get_random_cmap(max(max_label, 256))

    def colorize(seg):
        seg_clipped = np.clip(seg, 0, len(colors) - 1)
        return colors[seg_clipped]

    # Row 1: Segmentation
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(colorize(gt_seg))
    axes[0, 1].set_title("GT", fontsize=9, fontweight="bold")
    axes[0, 1].axis("off")

    for j, (method_key, seg) in enumerate(method_segs.items()):
        axes[0, 2 + j].imshow(colorize(seg))
        axes[0, 2 + j].set_title(method_names[method_key], fontsize=8)
        axes[0, 2 + j].axis("off")

    # Row 2: Inlier/outlier overlays
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title("RGB", fontsize=9, fontweight="bold")
    axes[1, 0].axis("off")

    _inlier_fn = compute_inlier_fn if compute_inlier_fn else \
        lambda seg, d, thr: compute_inlier_mask(seg, d, K, c2w, thr, INLIER_RATIO_GATE)

    gt_inlier, gt_outlier, gt_stats = _inlier_fn(gt_seg, depth, distance_threshold)
    gt_overlay = overlay_inliers(rgb, gt_inlier, gt_outlier)
    axes[1, 1].imshow(gt_overlay)
    axes[1, 1].set_title(f"GT  P={gt_stats['precision']:.2f} R={gt_stats['recall']:.2f}",
                         fontsize=8, fontweight="bold")
    axes[1, 1].axis("off")

    for j, (method_key, seg) in enumerate(method_segs.items()):
        inlier, outlier, stats = _inlier_fn(seg, depth, distance_threshold)
        method_overlay = overlay_inliers(rgb, inlier, outlier)
        axes[1, 2 + j].imshow(method_overlay)
        axes[1, 2 + j].set_title(
            f"{method_names[method_key]}  P={stats['precision']:.2f} R={stats['recall']:.2f}",
            fontsize=7)
        axes[1, 2 + j].axis("off")

    threshold_cm = distance_threshold * 100
    prefix = f"[{dataset_label}] " if dataset_label else ""
    fig.suptitle(f"{prefix}{scene_id} / {frame_idx}  (inlier threshold: {threshold_cm:.1f}cm)",
                 fontsize=10, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def save_individual_images(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    gt_seg: np.ndarray,
    method_segs: Dict[str, np.ndarray],
    method_names: Dict[str, str],
    distance_threshold: float,
    sample_prefix: str,
    seg_dir: Path,
    inlier_dir: Path,
    compute_inlier_fn=None,
):
    """Save each visualization panel as a separate PNG into per-method folders."""
    # Shared color map across GT and all methods
    all_labels = set(np.unique(gt_seg))
    for seg in method_segs.values():
        all_labels.update(np.unique(seg))
    max_label = max(all_labels) + 1
    colors = get_random_cmap(max(max_label, 256))

    def colorize(seg):
        seg_clipped = np.clip(seg, 0, len(colors) - 1)
        return colors[seg_clipped]

    fname = f"{sample_prefix}.png"

    # --- Segmentation PNGs ---
    (seg_dir / "rgb").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(seg_dir / "rgb" / fname))

    (seg_dir / "gt").mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(gt_seg)).save(str(seg_dir / "gt" / fname))

    for method_key, seg in method_segs.items():
        (seg_dir / method_key).mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorize(seg)).save(str(seg_dir / method_key / fname))

    # --- Inlier overlay PNGs ---
    _inlier_fn = compute_inlier_fn if compute_inlier_fn else \
        lambda seg, d, thr: compute_inlier_mask(seg, d, K, c2w, thr, INLIER_RATIO_GATE)

    (inlier_dir / "rgb").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(inlier_dir / "rgb" / fname))

    gt_inlier, gt_outlier, gt_stats = _inlier_fn(gt_seg, depth, distance_threshold)
    gt_overlay = overlay_inliers(rgb, gt_inlier, gt_outlier)
    p, r = gt_stats["precision"], gt_stats["recall"]
    (inlier_dir / "gt").mkdir(parents=True, exist_ok=True)
    Image.fromarray(gt_overlay).save(
        str(inlier_dir / "gt" / f"{sample_prefix}_P{p:.2f}_R{r:.2f}.png"))

    for method_key, seg in method_segs.items():
        inlier, outlier, stats = _inlier_fn(seg, depth, distance_threshold)
        overlay = overlay_inliers(rgb, inlier, outlier)
        p, r = stats["precision"], stats["recall"]
        (inlier_dir / method_key).mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(
            str(inlier_dir / method_key / f"{sample_prefix}_P{p:.2f}_R{r:.2f}.png"))


# ============================================================
# PER-DATASET RUNNERS
# ============================================================

def _run_generic(dataset, n_samples, seed, output_dir, methods, h5_root,
                 threshold, dataset_name, loader_cls=LazyH5Loader,
                 extract_ids_fn=None, compute_inlier_fn_factory=None):
    """Generic runner for datasets with standard H5 loading (single file per scene)."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))

    loaders = {}
    method_names = {}
    for key, cfg in methods.items():
        method_names[key] = cfg["display_name"]
        if not cfg["uses_gt_h5"]:
            _cls = cfg.get("loader_cls", loader_cls)
            _root = Path(cfg["h5_root_override"]) if cfg.get("h5_root_override") else h5_root
            _kwargs = cfg.get("loader_kwargs", {})
            loaders[key] = _cls(_root, cfg["h5_folder"], cfg.get("nonplanar_label"), **_kwargs)

    seg_dir = output_dir / dataset_name / "segmentation"
    inlier_dir = output_dir / dataset_name / "inliers"
    seg_dir.mkdir(parents=True, exist_ok=True)
    inlier_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i, idx in enumerate(tqdm(indices, desc=f"  [{dataset_name}] Generating samples")):
        sample = dataset[idx]

        rgb = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        depth = sample["depth"].squeeze(0).numpy()
        K = sample["K"].numpy()
        c2w = sample["c2w"].numpy()
        gt_seg = sample["plane"].squeeze(0).numpy().astype(np.int32)
        scene_id = sample["scene_id"]
        frame_idx = sample["frame_idx"]

        # Load predictions (skip individual missing methods, not the whole frame)
        method_segs = {}
        for key, cfg in methods.items():
            if cfg["uses_gt_h5"]:
                continue
            if extract_ids_fn:
                seg = extract_ids_fn(loaders[key], sample)
            else:
                seg = loaders[key].load(scene_id, frame_idx)
            if seg is None:
                print(f"    [SKIP] {scene_id}/{frame_idx}: missing {cfg['display_name']}")
                continue
            if seg.shape != depth.shape:
                seg = cv2.resize(seg, (depth.shape[1], depth.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            method_segs[key] = seg

        inlier_fn = compute_inlier_fn_factory(dataset, idx, sample) \
            if compute_inlier_fn_factory else None

        prefix = f"{i:03d}_{scene_id.replace('/', '_')}_{frame_idx}"
        save_individual_images(
            rgb, depth, K, c2w, gt_seg, method_segs, method_names,
            threshold, prefix, seg_dir, inlier_dir,
            compute_inlier_fn=inlier_fn)
        saved_count += 1

    print(f"  Saved {saved_count} samples:")
    print(f"    segmentation -> {seg_dir}")
    print(f"    inliers      -> {inlier_dir}")
    return saved_count, seg_dir, inlier_dir


def run_scannetpp(n_samples: int, seed: int, output_dir: Path, methods: dict,
                  threshold: float):
    """Generate qualitative samples for ScanNet++."""
    from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset

    print(f"\n--- ScanNet++ ({n_samples} samples, threshold={threshold*100:.1f}cm) ---")

    dataset = ScanNetPPPlaneDataset(
        rgb_root=SCANNETPP_RGB_ROOT,
        plane_label_root=SCANNETPP_LABEL_ROOT,
        sem_label_root=SCANNETPP_LABEL_ROOT,
        depth_label_root=SCANNETPP_LABEL_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
    )
    print(f"  Dataset: {len(dataset)} frames")

    return _run_generic(dataset, n_samples, seed, output_dir, methods,
                        H5_ROOT_SCANNETPP, threshold, "scannetpp")


def run_hypersim(n_samples: int, seed: int, output_dir: Path, methods: dict,
                 threshold: float):
    """Generate qualitative samples for Hypersim."""
    from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset

    print(f"\n--- Hypersim ({n_samples} samples, threshold={threshold*100:.1f}cm) ---")

    dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=HYPERSIM_PLANE_LABEL_ROOT,
        params_root=HYPERSIM_PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split="test",
        image_height=512,
        image_width=768,
        use_raycasted_depth="euclidean",
    )
    print(f"  Dataset: {len(dataset)} frames")

    # Hypersim uses per-camera H5 files and needs cam_name extraction
    def hypersim_load(loader, sample):
        rgb_path = sample["rgb_path"]
        cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"
        return loader.load(sample["scene_id"], cam_name, sample["frame_idx"])

    # Hypersim needs backproject_mcam with M_cam_from_uv (not pinhole K)
    def hypersim_inlier_fn_factory(ds, idx, sample):
        scene_id = sample["scene_id"]
        M_cam = ds._get_M_cam_from_uv(scene_id)
        native_wh = ds.valid_pairs[idx][-1]
        c2w = sample["c2w"].numpy()
        if M_cam is None:
            return None  # fall back to default pinhole
        def _fn(seg, depth_euc, thr):
            return compute_inlier_mask_hypersim(
                seg, depth_euc, M_cam, native_wh, c2w, thr, INLIER_RATIO_GATE)
        return _fn

    return _run_generic(dataset, n_samples, seed, output_dir, methods,
                        H5_ROOT_HYPERSIM, threshold, "hypersim",
                        loader_cls=LazyH5LoaderPerCamera,
                        extract_ids_fn=hypersim_load,
                        compute_inlier_fn_factory=hypersim_inlier_fn_factory)


def run_synthia(n_samples: int, seed: int, output_dir: Path, methods: dict,
                threshold: float):
    """Generate qualitative samples for Synthia."""
    from planamono.shared.datasets.synthia_plane_dataset import SynthiaPlaneDataset

    print(f"\n--- Synthia ({n_samples} samples, threshold={threshold*100:.1f}cm) ---")

    dataset = SynthiaPlaneDataset(
        data_root=SYNTHIA_ROOT,
        split="test",
    )
    print(f"  Dataset: {len(dataset)} frames")

    return _run_generic(dataset, n_samples, seed, output_dir, methods,
                        H5_ROOT_SYNTHIA, threshold, "synthia")


def run_vkitti2(n_samples: int, seed: int, output_dir: Path, methods: dict,
                threshold: float):
    """Generate qualitative samples for VKITTI2."""
    from planamono.shared.datasets.vkitti2_plane_dataset import VKITTI2PlaneDataset

    print(f"\n--- VKITTI2 ({n_samples} samples, threshold={threshold*100:.1f}cm) ---")

    dataset = VKITTI2PlaneDataset(
        data_root=VKITTI2_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "vkitti2"),
        split="test",
    )
    print(f"  Dataset: {len(dataset)} frames")

    return _run_generic(dataset, n_samples, seed, output_dir, methods,
                        H5_ROOT_VKITTI2, threshold, "vkitti2")


# ============================================================
# COMBINE OUTPUTS
# ============================================================

DATASET_LABELS = {
    "scannetpp": "ScanNet++ (indoor real)",
    "hypersim": "Hypersim (indoor synthetic)",
    "synthia": "Synthia (outdoor synthetic)",
    "vkitti2": "VKITTI2 (outdoor synthetic)",
}


def combine_to_png(sample_dir: Path, output_path: Path):
    """Stack all sample PNGs into one tall image."""
    png_files = sorted(sample_dir.glob("*.png"))
    if not png_files:
        return

    images = [np.array(Image.open(str(p))) for p in png_files]
    pad = 10
    total_h = sum(img.shape[0] for img in images) + pad * (len(images) - 1)
    max_w = max(img.shape[1] for img in images)

    combined = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
    y = 0
    for img in images:
        h, w = img.shape[:2]
        combined[y:y + h, :w] = img[:, :, :3]
        y += h + pad

    Image.fromarray(combined).save(str(output_path), dpi=(150, 150))
    print(f"  Combined PNG: {output_path}")


def combine_to_pdf(sample_dir: Path, output_path: Path):
    """Combine all sample PNGs into a multi-page PDF."""
    png_files = sorted(sample_dir.glob("*.png"))
    if not png_files:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        for png_path in png_files:
            img = Image.open(str(png_path))
            fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100))
            ax.imshow(np.array(img))
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"  Combined PDF: {output_path}")


def generate_combined_pdf(output_dir: Path, datasets: list, output_path: Path):
    """Generate a single PDF with all datasets, each section starting with a title page."""
    all_pngs = []
    for ds_name in datasets:
        sample_dir = output_dir / ds_name / "samples"
        pngs = sorted(sample_dir.glob("*.png"))
        if pngs:
            all_pngs.append((ds_name, pngs))

    if not all_pngs:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        for ds_name, pngs in all_pngs:
            label = DATASET_LABELS.get(ds_name, ds_name)
            threshold = DATASET_CONFIG[ds_name]["threshold"]
            threshold_cm = threshold * 100

            # Section title page
            fig, ax = plt.subplots(figsize=(16, 2))
            ax.text(0.5, 0.5, f"{label}\nInlier threshold: {threshold_cm:.1f}cm",
                    ha="center", va="center", fontsize=24, fontweight="bold")
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            # Sample pages
            for png_path in pngs:
                img = Image.open(str(png_path))
                fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100))
                ax.imshow(np.array(img))
                ax.axis("off")
                fig.tight_layout(pad=0)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    print(f"  Combined all-datasets PDF: {output_path}")


def generate_combined_png(output_dir: Path, datasets: list, output_path: Path):
    """Generate a single tall PNG with all datasets, separated by section headers."""
    all_images = []
    for ds_name in datasets:
        sample_dir = output_dir / ds_name / "samples"
        pngs = sorted(sample_dir.glob("*.png"))
        if not pngs:
            continue

        label = DATASET_LABELS.get(ds_name, ds_name)
        threshold = DATASET_CONFIG[ds_name]["threshold"]

        # Create section header image
        header_fig, header_ax = plt.subplots(figsize=(16, 0.6))
        header_ax.text(0.5, 0.5, f"{label}  (threshold: {threshold*100:.1f}cm)",
                       ha="center", va="center", fontsize=16, fontweight="bold")
        header_ax.set_facecolor("#e0e0e0")
        header_fig.set_facecolor("#e0e0e0")
        header_ax.axis("off")
        header_fig.tight_layout(pad=0.1)
        header_fig.canvas.draw()
        header_arr = np.array(header_fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(header_fig)
        all_images.append(header_arr)

        for png_path in pngs:
            all_images.append(np.array(Image.open(str(png_path)))[:, :, :3])

    if not all_images:
        return

    pad = 10
    max_w = max(img.shape[1] for img in all_images)
    total_h = sum(img.shape[0] for img in all_images) + pad * (len(all_images) - 1)

    combined = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
    y = 0
    for img in all_images:
        h, w = img.shape[:2]
        combined[y:y + h, :w] = img
        y += h + pad

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(str(output_path), dpi=(150, 150))
    print(f"  Combined all-datasets PNG: {output_path}")


def generate_markdown(output_dir: Path, datasets: list, methods: dict, output_path: Path):
    """Generate a markdown report with embedded images for all datasets."""
    lines = ["# Qualitative Results", ""]
    lines.append("For each sample: **Row 1** shows plane segmentation "
                 "(RGB | GT | predictions), **Row 2** shows inlier (green) / "
                 "outlier (red) overlays with precision and recall.")
    lines.append("")
    lines.append("Methods: " + ", ".join(
        cfg["display_name"] for cfg in methods.values() if not cfg["uses_gt_h5"]))
    lines.append("")

    for ds_name in datasets:
        label = DATASET_LABELS.get(ds_name, ds_name)
        threshold = DATASET_CONFIG[ds_name]["threshold"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"Inlier threshold: {threshold*100:.1f}cm")
        lines.append("")

        sample_dir = output_dir / ds_name / "samples"
        pngs = sorted(sample_dir.glob("*.png"))
        for png_path in pngs:
            rel_path = png_path.relative_to(output_dir)
            lines.append(f"![{png_path.stem}]({rel_path})")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  Markdown report: {output_path}")


# ============================================================
# MAIN
# ============================================================

# Per-dataset methods: shared methods + dataset-specific PlaneRCNN
SCANNETPP_METHODS = {
    **SHARED_METHODS,
    "planercnn": {
        "h5_folder": None, "display_name": "PlaneRCNN",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": str(PLANERCNN_SCANNETPP_ROOT),
        "loader_cls": PlaneRCNNH5Loader,
    },
    "pseudo_planamono": {
        "h5_folder": None, "display_name": "Pseudo-Planamono",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_scannetpp",
        "loader_cls": LazyH5Loader,
        "loader_kwargs": {"h5_filename": "rendered_v2.h5"},
    },
}
HYPERSIM_METHODS = {
    **SHARED_METHODS,
    "planercnn": {
        "h5_folder": None, "display_name": "PlaneRCNN",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": str(PLANERCNN_HYPERSIM_ROOT),
        "loader_cls": PlaneRCNNH5LoaderPerCamera,
    },
    "pseudo_planamono": {
        "h5_folder": None, "display_name": "Pseudo-Planamono",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_hypersim",
        "loader_cls": LazyH5LoaderPerCamera,
        "loader_kwargs": {"h5_prefix": "rendered_planes"},
    },
}
SYNTHIA_METHODS = {
    **SHARED_METHODS,
    "planercnn": {
        "h5_folder": None, "display_name": "PlaneRCNN",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": str(PLANERCNN_SYNTHIA_ROOT),
        "loader_cls": PlaneRCNNReindexedH5Loader,
        "loader_kwargs": {
            "data_root": str(SYNTHIA_ROOT) + "/test",
        },
    },
    "pseudo_planamono": {
        "h5_folder": None, "display_name": "Pseudo-Planamono",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_synthia",
        "loader_cls": LazyH5Loader,
        "loader_kwargs": {"h5_filename": "rendered_v2.h5"},
    },
}
VKITTI2_METHODS = {
    **SHARED_METHODS,
    "planercnn": {
        "h5_folder": None, "display_name": "PlaneRCNN",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": str(PLANERCNN_VKITTI2_ROOT),
        "loader_cls": PlaneRCNNReindexedH5Loader,
        "loader_kwargs": {
            "data_root": str(VKITTI2_ROOT),
            "scene_id_to_filename": lambda s: s.replace("/", "_"),
        },
    },
    "pseudo_planamono": {
        "h5_folder": None, "display_name": "Pseudo-Planamono",
        "nonplanar_label": None, "uses_gt_h5": False,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_vkitti2",
        "loader_cls": LazyH5Loader,
        "loader_kwargs": {"h5_filename": "rendered_v2.h5"},
    },
}

DATASET_CONFIG = {
    "scannetpp": {
        "runner": run_scannetpp,
        "methods": SCANNETPP_METHODS,
        "threshold": INDOOR_THRESHOLD,
    },
    "hypersim": {
        "runner": run_hypersim,
        "methods": HYPERSIM_METHODS,
        "threshold": INDOOR_THRESHOLD,
    },
    "synthia": {
        "runner": run_synthia,
        "methods": SYNTHIA_METHODS,
        "threshold": OUTDOOR_THRESHOLD,
    },
    "vkitti2": {
        "runner": run_vkitti2,
        "methods": VKITTI2_METHODS,
        "threshold": OUTDOOR_THRESHOLD,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison report (segmentation + inliers)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of sample frames per dataset (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=ALL_DATASETS + ["all"],
                        help="Datasets to include in combined output (default: all)")
    parser.add_argument("--regenerate", nargs="+", default=None,
                        choices=ALL_DATASETS + ["all"],
                        help="Only regenerate these datasets (reuse existing PNGs for the rest). "
                             "If not specified, regenerates all --datasets.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Prediction methods to visualize (default: all). "
                             "GT and RGB are always included.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    # Determine which datasets to regenerate vs reuse
    if args.regenerate is not None:
        regenerate = ALL_DATASETS if "all" in args.regenerate else args.regenerate
    else:
        regenerate = datasets  # default: regenerate everything in --datasets

    # Generate per-dataset samples (only for datasets in regenerate list)
    for ds_name in datasets:
        if ds_name not in regenerate:
            seg_dir = output_dir / ds_name / "segmentation"
            inlier_dir = output_dir / ds_name / "inliers"
            n_seg = len(list(seg_dir.glob("*.png"))) if seg_dir.exists() else 0
            n_inl = len(list(inlier_dir.glob("*.png"))) if inlier_dir.exists() else 0
            print(f"\n--- {ds_name}: reusing existing PNGs ({n_seg} seg, {n_inl} inlier) ---")
            continue

        cfg = DATASET_CONFIG[ds_name]
        methods = cfg["methods"]
        if args.methods:
            methods = {k: v for k, v in methods.items()
                       if k in args.methods or v["uses_gt_h5"]}
        cfg["runner"](
            args.n_samples, args.seed, output_dir, methods, cfg["threshold"])

    print(f"\nDone. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
