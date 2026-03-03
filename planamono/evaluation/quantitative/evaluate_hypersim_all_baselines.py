#!/usr/bin/env python3
"""
Unified evaluation script for all baseline methods on Hypersim dataset.

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Aggregated baseline comparison tables

Usage:
    python evaluate_hypersim_all_baselines.py                    # Evaluate all methods
    python evaluate_hypersim_all_baselines.py --methods ours_mixed zeroplane  # Evaluate specific methods
    python evaluate_hypersim_all_baselines.py --aggregate-only   # Only aggregate existing results
    python evaluate_hypersim_all_baselines.py --inlier-gates 0.5 0.7 0.8 0.9  # Multi-gate evaluation
    python evaluate_hypersim_all_baselines.py --inlier-gates 0.5 0.7 0.8 0.9 --aggregate-only  # Re-aggregate multigate
"""

import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
import cv2
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from joblib import Parallel, delayed

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.paths import repo_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame_hypersim,
    evaluate_single_frame_hypersim_multigates,
)



# ============================================================
# CONFIGURATION
# ============================================================

# Evaluation parameters
COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.001, 0.005, 0.01)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/hypersim/inference")
HYPERSIM_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
# Old paths (buggy plane_id=0 collision in rendered labels)
# HYPERSIM_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
# PLANE_LABEL_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
# PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_params"

# Experiment version (used in exp_name for all methods)
# v1: original (wrong K + Euclidean depth)
# v2: fixed K at native resolution + Euclidean→z-depth conversion + backproject_v1
# v3: raycasted Euclidean depth from planes.ply + backproject_mcam (M_cam_from_uv)
# EXP_VER = "v3"
EXP_VER = "v1"

# Method definitions: {method_key: {h5_folder, display_name, label_offset, nonplanar_label, uses_gt_h5}}
METHODS = {
    "gt": {
        "h5_folder": None,  # Use GT labels directly
        "exp_name": "gt_v3",
        "display_name": "GT (upper bound)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": True,
    },
    "moge_ours": {
        "h5_folder": "moge_ours_h5",
        "exp_name": f"moge_ours_{EXP_VER}",
        "display_name": "MoGe Ours (ScanNet++)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_mixed_bce": {
        "h5_folder": "moge_mixed_bce_h5",
        "exp_name": f"moge_mixed_bce_{EXP_VER}",
        "display_name": "MoGe Mixed BCE",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_dust3r": {
        "h5_folder": "zeroplane_mixed_dust3r_h5",
        "exp_name": f"zeroplane_mixed_dust3r_{EXP_VER}",
        "display_name": "ZeroPlane (Mixed Dust3R)",
        "label_offset": 0,
        "nonplanar_label": 20,  # ZeroPlane uses 20 for non-planar regions
        "uses_gt_h5": False,
    },
    "zeroplane_mixed": {
        "h5_folder": "zeroplane_mixed_h5",
        "exp_name": f"zeroplane_mixed_{EXP_VER}",
        "display_name": "ZeroPlane (Mixed)",
        "label_offset": 0,
        "nonplanar_label": 20,  # ZeroPlane uses 20 for non-planar regions
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_h5_dust3r": {
        "h5_folder": "zeroplane_mixed_h5_dust3r_h5",
        "exp_name": f"zeroplane_mixed_h5_dust3r_{EXP_VER}",
        "display_name": "ZeroPlane (mixed_h5+dust3r)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_h5_dust3r_75k": {
        "h5_folder": "zeroplane_mixed_h5_dust3r_75k_h5",
        "exp_name": f"zeroplane_mixed_h5_dust3r_75k_{EXP_VER}",
        "display_name": "ZeroPlane (mixed_h5+dust3r, 75k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_h5_dust3r_145k": {
        "h5_folder": "zeroplane_mixed_h5_dust3r_145k_h5",
        "exp_name": f"zeroplane_mixed_h5_dust3r_145k_{EXP_VER}",
        "display_name": "ZeroPlane (mixed_h5+dust3r, 145k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_default_dust3r_released": {
        "h5_folder": "zeroplane_default_dust3r_released_h5",
        "exp_name": f"zeroplane_default_dust3r_released_{EXP_VER}",
        "display_name": "ZeroPlane (default+dust3r released)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_all_h5_dust3r": {
        "h5_folder": "zeroplane_all_h5_dust3r_h5",
        "exp_name": f"zeroplane_all_h5_dust3r_{EXP_VER}",
        "display_name": "ZeroPlane (all_h5+dust3r)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_h5_dinov2_moge_60k": {
        "h5_folder": "zeroplane_mixed_h5_dinov2_moge_60k_h5",
        "exp_name": f"zeroplane_mixed_h5_dinov2_moge_60k_{EXP_VER}",
        "display_name": "ZeroPlane (mixed_h5+dinov2_moge 60k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_all_h5_dinov2_moge_60k": {
        "h5_folder": "zeroplane_all_h5_dinov2_moge_60k_h5",
        "exp_name": f"zeroplane_all_h5_dinov2_moge_60k_{EXP_VER}",
        "display_name": "ZeroPlane (all_h5+dinov2_moge 60k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_h5_dinov2_moge_165k": {
        "h5_folder": "zeroplane_mixed_h5_dinov2_moge_165k_h5",
        "exp_name": f"zeroplane_mixed_h5_dinov2_moge_165k_{EXP_VER}",
        "display_name": "ZeroPlane (mixed_h5+dinov2_moge 165k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "zeroplane_all_h5_dinov2_moge_165k": {
        "h5_folder": "zeroplane_all_h5_dinov2_moge_165k_h5",
        "exp_name": f"zeroplane_all_h5_dinov2_moge_165k_{EXP_VER}",
        "display_name": "ZeroPlane (all_h5+dinov2_moge 165k)",
        "label_offset": 0,
        "nonplanar_label": 20,
        "uses_gt_h5": False,
    },
    "moge_mixed_bce_476644": {
        "h5_folder": "moge_mixed_bce_476644_h5",
        "exp_name": f"moge_mixed_bce_476644_{EXP_VER}",
        "display_name": "MoGe Mixed BCE 476644",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_mixed_bce_476644_ep6": {
        "h5_folder": "moge_mixed_bce_476644_ep6_h5",
        "exp_name": f"moge_mixed_bce_476644_ep6_{EXP_VER}",
        "display_name": "MoGe Mixed BCE 476644 ep6",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_mixed_bce_476644_ep6_v6seg": {
        "h5_folder": "moge_mixed_bce_476644_ep6_v6seg_h5",
        "exp_name": f"moge_mixed_bce_476644_ep6_v6seg_{EXP_VER}",
        "display_name": "MoGe Mixed BCE 476644 ep6 v6seg",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "gtplanarity_ourseg_476644_ep6": {
        "h5_folder": f"hypersim_gtplanarity_ourseg_moge_mixed_bce_476644_ep6_{EXP_VER}_h5",
        "exp_name": f"hypersim_gtplanarity_ourseg_moge_mixed_bce_476644_ep6_{EXP_VER}",
        "display_name": "GT Planarity + Our Seg (476644 ep6)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ourplanarity_gtseg_476644_ep6": {
        "h5_folder": f"hypersim_ourplanarity_gtseg_moge_mixed_bce_476644_ep6_{EXP_VER}_h5",
        "exp_name": f"hypersim_ourplanarity_gtseg_moge_mixed_bce_476644_ep6_{EXP_VER}",
        "display_name": "Our Planarity + GT Seg (476644 ep6)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep1": {
        "h5_folder": "moge_hires_ep1_h5",
        "exp_name": f"moge_hires_ep1_{EXP_VER}",
        "display_name": "MoGe HiRes ep1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep2": {
        "h5_folder": "moge_hires_ep2_h5",
        "exp_name": f"moge_hires_ep2_{EXP_VER}",
        "display_name": "MoGe HiRes ep2",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3": {
        "h5_folder": "moge_hires_ep3_h5",
        "exp_name": f"moge_hires_ep3_{EXP_VER}",
        "display_name": "MoGe HiRes ep3",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v11seg_metric": {
        "h5_folder": "moge_hires_ep3_v11seg_metric_h5",
        "exp_name": f"moge_hires_ep3_v11seg_metric_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 v11seg metric",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v5_relative_seg": {
        "h5_folder": "moge_hires_ep3_v5_relative_seg_h5",
        "exp_name": f"moge_hires_ep3_v5_relative_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5_rel (Sobel+rel)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v5origparams_relative_seg": {
        "h5_folder": "moge_hires_ep3_v5origparams_relative_seg_h5",
        "exp_name": f"moge_hires_ep3_v5origparams_relative_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5orig_rel (Sobel+rel)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_neck_head_ep1": {
        "h5_folder": "moge_neck_head_ep1_h5",
        "exp_name": f"moge_neck_head_ep1_{EXP_VER}",
        "display_name": "MoGe Neck+Head ep1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_neck_head_ep2": {
        "h5_folder": "moge_neck_head_ep2_h5",
        "exp_name": f"moge_neck_head_ep2_{EXP_VER}",
        "display_name": "MoGe Neck+Head ep2",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_proj_neck_head_ep1": {
        "h5_folder": "moge_proj_neck_head_ep1_h5",
        "exp_name": f"moge_proj_neck_head_ep1_{EXP_VER}",
        "display_name": "MoGe Proj+Neck+Head ep1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_proj_neck_head_ep2": {
        "h5_folder": "moge_proj_neck_head_ep2_h5",
        "exp_name": f"moge_proj_neck_head_ep2_{EXP_VER}",
        "display_name": "MoGe Proj+Neck+Head ep2",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_focal_hires_ep1": {
        "h5_folder": "moge_focal_hires_ep1_h5",
        "exp_name": f"moge_focal_hires_ep1_{EXP_VER}",
        "display_name": "MoGe Focal HiRes ep1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep6": {
        "h5_folder": "moge_hires_ep6_h5",
        "exp_name": f"moge_hires_ep6_{EXP_VER}",
        "display_name": "MoGe HiRes ep6",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_4ds_ep2": {
        "h5_folder": "moge_hires_4ds_ep2_h5",
        "exp_name": f"moge_hires_4ds_ep2_{EXP_VER}",
        "display_name": "MoGe HiRes 4DS ep2",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_4ds_ep1": {
        "h5_folder": "moge_hires_4ds_ep1_h5",
        "exp_name": f"moge_hires_4ds_ep1_{EXP_VER}",
        "display_name": "MoGe HiRes 4DS ep1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "planeTR": {
        "h5_folder": "planeTR_h5",
        "exp_name": f"planeTR_{EXP_VER}",
        "display_name": "PlaneTR",
        "label_offset": 0,
        "nonplanar_label": 21,  # PlaneTR uses 21 for non-planar regions
        "uses_gt_h5": False,
    },
    "planeTR_lines": {
        "h5_folder": "planeTR_lines_h5",
        "exp_name": f"planeTR_lines_{EXP_VER}",
        "display_name": "PlaneTR (lines)",
        "label_offset": 0,
        "nonplanar_label": 21,
        "uses_gt_h5": False,
    },
    "planar_recon": {
        "h5_folder": "planar_recon_h5",
        "exp_name": f"planar_recon_{EXP_VER}",
        "display_name": "PlanarReconstruction",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "pseudo_planamono": {
        "h5_folder": None,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_hypersim",
        "h5_prefix": "rendered_planes",
        "exp_name": f"pseudo_planamono_{EXP_VER}",
        "display_name": "Pseudo-Planamono",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    # ── 4ds ep2 + v5_relative segmentation ──────────────────────────────
    "moge_hires_4ds_ep2_v5_relative_seg": {
        "h5_folder": "moge_hires_4ds_ep2_v5_relative_seg_h5",
        "exp_name": f"moge_hires_4ds_ep2_v5_relative_seg_{EXP_VER}",
        "display_name": "MoGe 4ds ep2 v5_rel (plan=0.3, norm=5°)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_4ds_ep2_v5origparams_relative_seg": {
        "h5_folder": "moge_hires_4ds_ep2_v5origparams_relative_seg_h5",
        "exp_name": f"moge_hires_4ds_ep2_v5origparams_relative_seg_{EXP_VER}",
        "display_name": "MoGe 4ds ep2 v5orig_rel (plan=0.6, norm=10°)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
}


# ============================================================
# AUTO-DISCOVERY
# ============================================================

def discover_zeroplane_methods(h5_root: Path, model_dirs: List[str] = None) -> Dict:
    """Auto-discover ZeroPlane experiments from H5_ROOT directory structure.

    Looks for: h5_root/{model_label}/thresh_*/{scene_id}/planes_cam_*.h5
    """
    methods = {}

    if model_dirs:
        candidates = [h5_root / d for d in model_dirs]
    else:
        candidates = [d for d in h5_root.iterdir() if d.is_dir()]

    for model_dir in sorted(candidates):
        if not model_dir.is_dir():
            continue
        model_label = model_dir.name

        for thresh_dir in sorted(model_dir.iterdir()):
            if not thresh_dir.is_dir() or not thresh_dir.name.startswith("thresh_"):
                continue
            thresh_label = thresh_dir.name

            # Verify at least one H5 file exists
            has_h5 = any(thresh_dir.rglob("planes_cam_*.h5"))
            if not has_h5:
                continue

            key = f"zeroplane_{model_label}_{thresh_label}"
            methods[key] = {
                "h5_folder": f"{model_label}/{thresh_label}",
                "exp_name": f"zeroplane_{model_label}_{thresh_label}_{EXP_VER}",
                "display_name": f"ZeroPlane {model_label} ({thresh_label})",
                "label_offset": 0,
                "nonplanar_label": 20,
                "uses_gt_h5": False,
            }

    return methods


# ============================================================
# H5 LOADING
# ============================================================

class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one camera in memory at a time.
    Adapted for Hypersim's per-camera H5 structure: scene_id/planes_cam_XX.h5
    """
    def __init__(self, h5_root: str, label_offset: int = 0, nonplanar_label: Optional[int] = None,
                 h5_prefix: str = "planes"):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self.nonplanar_label = nonplanar_label
        self.h5_prefix = h5_prefix
        self._current_scene_id = None
        self._current_cam_name = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_camera(self, scene_id: str, cam_name: str) -> bool:
        """Load a camera's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id and cam_name == self._current_cam_name:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, f"{self.h5_prefix}_{cam_name}.h5")
        if not os.path.exists(h5_path):
            return False

        # Clear previous camera to free memory
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

        with h5py.File(h5_path, "r") as f:
            self._current_planes = f["planes"][:]
            self._current_frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]

        # Build index for O(1) lookup
        self._frame_id_to_idx = {
            fid: idx for idx, fid in enumerate(self._current_frame_ids)
        }
        self._current_scene_id = scene_id
        self._current_cam_name = cam_name

        return True

    def get_pred_seg(self, scene_id: str, cam_name: str, frame_id: str) -> Optional[np.ndarray]:
        """Get predicted segmentation for a frame."""
        if not self._load_camera(scene_id, cam_name):
            return None

        # Frame ID is stored without cam_name prefix in per-camera H5 files
        if frame_id not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_id]
        labels = self._current_planes[idx].astype(np.int32)

        # Handle non-planar label remapping (e.g., ZeroPlane's 20 → 0)
        if self.nonplanar_label is not None:
            labels = np.where(labels == self.nonplanar_label, 0, labels)

        # Apply label offset
        if self.label_offset != 0:
            # Shift all non-zero labels
            labels = np.where(labels > 0, labels + self.label_offset, 0)

        return labels


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(method_key: str, method_config: Dict, args, split: str = "test",
                    inlier_gates: Optional[Tuple[float, ...]] = None,
                    shard_id: int = None):
    """Evaluate a single method.

    Args:
        inlier_gates: If provided, evaluate at multiple inlier ratio gates
            (fits RANSAC once, evaluates at each gate). If None, uses single
            INLIER_RATIO_GATE as before.
        shard_id: If set, save results as results_shard_{shard_id}.csv instead of
            results.csv (for distributed eval with SLURM array jobs).
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_config['display_name']}")
    if inlier_gates:
        print(f"Inlier gates: {inlier_gates}")
    print(f"{'='*60}")

    # Handle GT method (no H5 folder)
    if "h5_root_override" in method_config:
        h5_root = Path(method_config["h5_root_override"])
        print(f"H5 root:    {h5_root}")
    elif method_config["h5_folder"] is not None:
        h5_root = H5_ROOT / method_config["h5_folder"]
        print(f"H5 root:    {h5_root}")
    else:
        h5_root = None
        print(f"H5 root:    N/A (using GT labels)")

    # Use a different output dir for multigate to avoid overwriting
    exp_name = method_config["exp_name"]
    if inlier_gates:
        output_dir = EVAL_ROOT / f"{exp_name}_multigate"
    else:
        output_dir = EVAL_ROOT / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output:     {output_dir}")

    # Load dataset
    timer = Timer()
    print("\n[INFO] Loading dataset...")

    dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=PLANE_LABEL_ROOT,
        params_root=PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split=split,
        image_height=512,
        image_width=768,
        max_scenes=None,
        use_raycasted_depth="euclidean",
    )

    print(f"[INFO] Dataset size: {len(dataset)} frames")

    # Scene range slicing for distributed eval
    if hasattr(args, 'scene_start') and (args.scene_start is not None or args.scene_end is not None):
        all_scenes = dataset.scene_ids
        s = args.scene_start or 0
        e = args.scene_end or len(all_scenes)
        subset_scenes = set(all_scenes[s:e])
        dataset.valid_pairs = [
            p for p in dataset.valid_pairs
            if p[0] in subset_scenes  # valid_pairs[i][0] = scene_id for Hypersim
        ]
        dataset.scene_ids = [sid for sid in all_scenes if sid in subset_scenes]
        print(f"[DATA] Scene range [{s}:{e}] → {len(subset_scenes)} scenes, {len(dataset)} frames")

    # Initialize H5 loader (skip for GT method)
    h5_loader = None
    if not method_config.get("uses_gt_h5", False) and h5_root is not None:
        h5_prefix = method_config.get("h5_prefix", "planes")
        h5_loader = LazyH5SceneLoader(
            str(h5_root),
            label_offset=method_config["label_offset"],
            nonplanar_label=method_config.get("nonplanar_label"),
            h5_prefix=h5_prefix,
        )

    # Evaluation function
    def eval_frame(idx):
        sample = dataset[idx]
        scene_id = sample["scene_id"]
        frame_id = sample["frame_idx"]

        # Extract cam_name from rgb_path (format: "scene_id/cam_name/frame_id")
        rgb_path = sample["rgb_path"]
        cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"

        # Full frame identifier with camera name
        full_frame_id = f"{cam_name}/{frame_id}"

        # Load prediction or use GT
        if method_config.get("uses_gt_h5", False):
            # Use GT labels directly (upper bound)
            gt_seg = sample["plane"].numpy().astype(np.int32)
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            labels = gt_seg.copy()  # Use GT as prediction
        else:
            # Load prediction from H5
            labels = h5_loader.get_pred_seg(scene_id, cam_name, frame_id)
            if labels is None:
                return None

        # Get ground truth
        gt_seg = sample["plane"].numpy().astype(np.int32)
        if gt_seg.ndim == 3:
            gt_seg = gt_seg[0]

        # Depth = raycasted Euclidean from planes.ply (t_hit * MPAU)
        depth_euc = sample["depth"].numpy()
        if depth_euc.ndim == 3:
            depth_euc = depth_euc[0]

        c2w = sample["c2w"].numpy()

        # M_cam_from_uv and native resolution for backproject_mcam
        M_cam = dataset._get_M_cam_from_uv(scene_id)
        native_wh = dataset.valid_pairs[idx][-1]

        # Resize prediction to match GT if needed
        if labels.shape != gt_seg.shape:
            labels = cv2.resize(
                labels.astype(np.uint16),
                (gt_seg.shape[1], gt_seg.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Evaluate: multigate or single gate
        if inlier_gates:
            metrics, _ = evaluate_single_frame_hypersim_multigates(
                scene_id,
                full_frame_id,
                depth_euc,
                gt_seg,
                M_cam,
                native_wh,
                c2w,
                labels,
                THRESHOLDS,
                inlier_ratio_gates=inlier_gates,
                compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
                ransac_iterations=RANSAC_ITERATIONS,
            )
        else:
            metrics, _ = evaluate_single_frame_hypersim(
                scene_id,
                full_frame_id,
                depth_euc,
                gt_seg,
                M_cam,
                native_wh,
                c2w,
                labels,
                THRESHOLDS,
                compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
                ransac_iterations=RANSAC_ITERATIONS,
                inlier_ratio_gate=INLIER_RATIO_GATE,
            )

        return (scene_id, full_frame_id), metrics

    # Run evaluation in parallel
    print("\n[INFO] Running evaluation...")
    with timer("evaluation"):
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_frame)(idx) for idx in tqdm(range(len(dataset)), desc="Evaluating")
        )

    # Collect results
    results = {}
    skipped = 0
    for output in outputs:
        if output is None:
            skipped += 1
            continue
        (scene_id, frame_id), metrics = output
        results[(scene_id, frame_id)] = metrics

    print(f"\n[INFO] Processed {len(results)} frames ({skipped} skipped)")

    # Save results
    print("\n[INFO] Saving results...")
    if shard_id is not None:
        # Save as shard file (for distributed eval)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(list(results.values()))
        shard_path = output_dir / f"results_shard_{shard_id}.csv"
        df.to_csv(shard_path, index=False)
        print(f"[CSV] Saved shard {shard_id} ({len(results)} frames) to {shard_path}")
    else:
        save_results_csv(results, str(output_dir))
    save_runtime(timer, str(output_dir))

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Results saved to: {output_dir}")

    return results


def _merge_shards(exp_dir: Path):
    """Merge shard CSV files into results.csv, then produce per-scene and dataset CSVs."""
    import glob as glob_mod
    shard_files = sorted(glob_mod.glob(str(exp_dir / "results_shard_*.csv")))
    if not shard_files:
        return False

    print(f"[MERGE] Found {len(shard_files)} shard files in {exp_dir}")
    dfs = [pd.read_csv(f) for f in shard_files]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[MERGE] Total frames: {len(df_all)}")

    # Reconstruct results dict for save_results_csv
    results = {}
    for _, row in df_all.iterrows():
        key = (row["scene_id"], row["frame_idx"])
        results[key] = row.to_dict()
    save_results_csv(results, str(exp_dir))
    print(f"[MERGE] Saved merged results to {exp_dir}")
    return True


def aggregate_results(methods: list):
    """Aggregate results from multiple methods into comparison tables.
    Merges shard files first if present."""
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")

    # Merge shards for each method if needed
    for method_key in methods:
        if method_key not in METHODS:
            continue
        exp_dir = EVAL_ROOT / METHODS[method_key]["exp_name"]
        if exp_dir.exists():
            _merge_shards(exp_dir)

    all_results = []
    for method_key in methods:
        if method_key not in METHODS:
            print(f"[WARN] Unknown method: {method_key}")
            continue

        method_config = METHODS[method_key]
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        csv_path = EVAL_ROOT / exp_name / "results_dataset.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing results for {method_key}: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Method": display_name,
                "method_key": method_key,
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }

            # Segmentation metrics
            for col, display in [("rand_index_mean", "RI"),
                                  ("voi_mean", "VOI"),
                                  ("sc_mean", "SC")]:
                if col in df.index:
                    row[display] = df[col]

            # Precision/recall/F1 metrics
            for thr in THRESHOLDS:
                thresh_str = f"{thr*100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]
                p = row.get(f"P@{thresh_str}", 0)
                r = row.get(f"R@{thresh_str}", 0)
                row[f"F1@{thresh_str}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            all_results.append(row)
            print(f"[OK] Loaded results for {display_name}")

        except Exception as e:
            print(f"[ERROR] Could not read results for {method_key}: {e}")

    if not all_results:
        print("[ERROR] No results to aggregate")
        return

    # Create DataFrames
    df_all = pd.DataFrame(all_results)

    # Table 1: Precision/Recall
    prec_rec_cols = ["Method", "num_scenes", "num_frames"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        prec_rec_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"])
    df_pr = df_all[[c for c in prec_rec_cols if c in df_all.columns]]
    out_path = EVAL_ROOT / f"table_precision_recall_baselines_{EXP_VER}.csv"
    df_pr.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 2: Segmentation
    seg_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    df_seg = df_all[[c for c in seg_cols if c in df_all.columns]]
    out_path = EVAL_ROOT / f"table_segmentation_baselines_{EXP_VER}.csv"
    df_seg.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 3: Combined summary
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        combined_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"])
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]
    out_path = EVAL_ROOT / f"table_combined_baselines_{EXP_VER}.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_combined.to_string(index=False))
    print("=" * 100)


def aggregate_results_multigate(methods: list, inlier_gates: Tuple[float, ...]):
    """Aggregate multigate results into a single comparison table.

    Reads from {exp_name}_multigate/results_dataset.csv and produces
    table_precision_recall_baselines_{EXP_VER}_multigate.csv with columns:
    Method | P@0.1cm_gate0.5 | R@0.1cm_gate0.5 | ... | P@1.0cm_gate0.9 | R@1.0cm_gate0.9
    """
    print(f"\n{'='*60}")
    print(f"Aggregating Multigate Results (gates={inlier_gates})")
    print(f"{'='*60}")

    all_results = []
    for method_key in methods:
        if method_key not in METHODS:
            print(f"[WARN] Unknown method: {method_key}")
            continue

        method_config = METHODS[method_key]
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        csv_path = EVAL_ROOT / f"{exp_name}_multigate" / "results_dataset.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing multigate results for {method_key}: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Method": display_name,
                "method_key": method_key,
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }

            for thr in THRESHOLDS:
                thresh_str = f"{thr*100:.1f}cm"
                for gate in inlier_gates:
                    prec_col = f"prec@{thresh_str}_gate{gate}_mean"
                    rec_col = f"rec@{thresh_str}_gate{gate}_mean"
                    if prec_col in df.index:
                        row[f"P@{thresh_str}_gate{gate}"] = df[prec_col]
                    if rec_col in df.index:
                        row[f"R@{thresh_str}_gate{gate}"] = df[rec_col]

            all_results.append(row)
            print(f"[OK] Loaded multigate results for {display_name}")

        except Exception as e:
            print(f"[ERROR] Could not read multigate results for {method_key}: {e}")

    if not all_results:
        print("[ERROR] No multigate results to aggregate")
        return

    df_all = pd.DataFrame(all_results)

    # Build column order: Method, counts, then P/R for each (threshold, gate)
    cols = ["Method", "num_scenes", "num_frames"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        for gate in inlier_gates:
            cols.extend([f"P@{thresh_str}_gate{gate}", f"R@{thresh_str}_gate{gate}"])

    df_out = df_all[[c for c in cols if c in df_all.columns]]
    out_path = EVAL_ROOT / f"table_precision_recall_baselines_{EXP_VER}_multigate.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 120)
    print("MULTIGATE PRECISION/RECALL RESULTS")
    print("=" * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_out.to_string(index=False))
    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

INLIER_GATES_DEFAULT = (0.5, 0.7, 0.8, 0.9)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hypersim predictions for all baseline methods"
    )

    parser.add_argument("--methods", nargs="+", default=None,
                       help="Methods to evaluate (default: all)")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--inlier-gates", nargs="+", type=float, default=None,
                       help="Evaluate at multiple inlier ratio gates "
                            f"(default when flag used: {list(INLIER_GATES_DEFAULT)}). "
                            "Fits RANSAC once per threshold, evaluates at each gate.")
    parser.add_argument("--discover-zeroplane", nargs="*", default=None,
                       help="Auto-discover ZeroPlane experiments under H5_ROOT. "
                            "If model dirs given, scan only those. "
                            "If no args, scan all dirs with thresh_* subdirs.")
    parser.add_argument("--scene-start", type=int, default=None,
                       help="Start scene index (for distributed eval across SLURM array jobs)")
    parser.add_argument("--scene-end", type=int, default=None,
                       help="End scene index exclusive (for distributed eval)")

    args = parser.parse_args()

    # Auto-discover ZeroPlane experiments if requested
    if args.discover_zeroplane is not None:
        model_dirs = args.discover_zeroplane if args.discover_zeroplane else None
        discovered = discover_zeroplane_methods(H5_ROOT, model_dirs)
        METHODS.update(discovered)
        print(f"[DISCOVER] Found {len(discovered)} ZeroPlane experiments: {list(discovered.keys())}")
        if not args.methods:
            # When discovering without --methods, evaluate only discovered methods
            methods_to_eval = list(discovered.keys())

    # Determine which methods to evaluate
    if args.discover_zeroplane is None or args.methods:
        methods_to_eval = args.methods if args.methods else list(METHODS.keys())

    # Parse inlier gates
    inlier_gates = None
    if args.inlier_gates is not None:
        inlier_gates = tuple(sorted(args.inlier_gates))

    print(f"Methods to evaluate: {methods_to_eval}")
    if inlier_gates:
        print(f"Inlier ratio gates: {inlier_gates}")

    # Derive shard_id from scene_start (for distributed eval)
    shard_id = None
    if args.scene_start is not None:
        shard_id = args.scene_start

    # Run evaluation
    if not args.aggregate_only:
        for method_key in methods_to_eval:
            if method_key not in METHODS:
                print(f"[ERROR] Unknown method: {method_key}")
                continue

            evaluate_method(method_key, METHODS[method_key], args,
                          split=args.split, inlier_gates=inlier_gates,
                          shard_id=shard_id)

    # Aggregate results (merge shards if needed)
    # Skip aggregation when running as a shard job — let the dedicated --aggregate-only job handle it
    if args.scene_start is None:
        if inlier_gates:
            aggregate_results_multigate(methods_to_eval, inlier_gates)
        else:
            aggregate_results(methods_to_eval)


if __name__ == "__main__":
    main()
