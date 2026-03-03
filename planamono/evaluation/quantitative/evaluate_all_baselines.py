"""
Unified evaluation script for all baseline methods.

Properly handles different label conventions across methods:
- Standard methods (ours, gtseg, etc.): Use label 0 for non-planar/background
- ZeroPlane: Uses label 20 for non-planar regions (automatically remapped to 0)

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Aggregated baseline comparison tables

Usage:
    python evaluate_all_baselines.py                    # Evaluate all methods
    python evaluate_all_baselines.py --methods ours zeroplane  # Evaluate specific methods
    python evaluate_all_baselines.py --aggregate-only   # Only aggregate existing results
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
from typing import Dict, Optional, Tuple

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_rend_plane_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Evaluation parameters
COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
# THRESHOLDS = (0.01, 0.02, 0.05)
THRESHOLDS = (0.001, 0.005, 0.01)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
DATASET_DIR = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

# Experiment version (used in exp_name for all methods)
# EXP_VER = "v5"  # Unified version with proper non-planar label handling
EXP_VER = "v6"  # Unified version with proper non-planar label handling

# Method definitions: {method_key: {h5_folder, display_name, label_offset, nonplanar_label, uses_gt_h5}}
METHODS = {
    "gt": {
        "h5_folder": None,  # Use GT labels directly
        "exp_name": f"gt_{EXP_VER}",
        "display_name": "GT (upper bound)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": True,
    },
    "ours": {
        "h5_folder": "moge_ours_v2_h5",
        "exp_name": f"moge_ours_{EXP_VER}",
        "display_name": "Ours (full)",
        "label_offset": 0,
        "nonplanar_label": None,  # Our method uses 0 for background
        "uses_gt_h5": False,
    },
    "zeroplane": {
        "h5_folder": "zeroplane_h5",
        "exp_name": f"zeroplane_{EXP_VER}",
        "display_name": "ZeroPlane",
        "label_offset": 0,  # No offset after remapping label 20 → 0
        "nonplanar_label": 20,  # ZeroPlane uses 20 for non-planar regions
        "uses_gt_h5": False,
    },
    "zeroplane_mixed": {
        "h5_folder": "zeroplane_mixed_h5",
        "exp_name": f"zeroplane_mixed_{EXP_VER}",
        "display_name": "ZeroPlane (mixed)",
        "label_offset": 0,  # No offset after remapping label 20 → 0
        "nonplanar_label": 20,  # ZeroPlane uses 20 for non-planar regions
        "uses_gt_h5": False,
    },
    "zeroplane_mixed_dust3r": {
        "h5_folder": "zeroplane_mixed_dust3r_h5",
        "exp_name": f"zeroplane_mixed_dust3r_{EXP_VER}",
        "display_name": "ZeroPlane (mixed+dust3r)",
        "label_offset": 0,  # No offset after remapping label 20 → 0
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
    "zeroplane_default_dust3r_released": {
        "h5_folder": "zeroplane_default_dust3r_released_h5",
        "exp_name": f"zeroplane_default_dust3r_released_{EXP_VER}",
        "display_name": "ZeroPlane (default+dust3r released)",
        "label_offset": 0,
        "nonplanar_label": 20,
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
    "gtseg": {
        "h5_folder": "gtseg_v1_h5",
        "exp_name": f"gtseg_{EXP_VER}",
        "display_name": "GT Seg (upper bound)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "gtplanarity_ourseg": {
        "h5_folder": "gtplanarity_ourseg_h5",
        "exp_name": f"gtplanarity_ourseg_{EXP_VER}",
        "display_name": "GT Planarity + Our Seg",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ourplanarity_gtseg": {
        "h5_folder": "ourplanarity_gtseg_h5",
        "exp_name": f"ourplanarity_gtseg_{EXP_VER}",
        "display_name": "Our Planarity + GT Seg",
        "label_offset": 0,
        "nonplanar_label": None,
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
        "h5_folder": f"gtplanarity_ourseg_moge_mixed_bce_476644_ep6_{EXP_VER}_h5",
        "exp_name": f"gtplanarity_ourseg_moge_mixed_bce_476644_ep6_{EXP_VER}",
        "display_name": "GT Planarity + Our Seg (476644 ep6)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ourplanarity_gtseg_476644_ep6": {
        "h5_folder": f"ourplanarity_gtseg_moge_mixed_bce_476644_ep6_{EXP_VER}_h5",
        "exp_name": f"ourplanarity_gtseg_moge_mixed_bce_476644_ep6_{EXP_VER}",
        "display_name": "Our Planarity + GT Seg (476644 ep6)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "planercnn_gt": {
        "h5_folder": None,
        "h5_root_override": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn",
        "h5_filename": "rendered.h5",
        "exp_name": f"planercnn_gt_{EXP_VER}",
        "display_name": "PlaneRCNN GT",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "planercnn_gt_v1": {
        "h5_folder": None,
        "h5_root_override": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn_v1",
        "h5_filename": "rendered.h5",
        "exp_name": f"planercnn_gt_v1_{EXP_VER}",
        "display_name": "PlaneRCNN GT v1",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "pseudo_planamono": {
        "h5_folder": None,
        "h5_root_override": "/cluster/scratch/ayavuz/dataset/pseudo_planamono_scannetpp",
        "h5_filename": "rendered_v2.h5",
        "exp_name": f"pseudo_planamono_{EXP_VER}",
        "display_name": "Pseudo-Planamono",
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
    "moge_hires_ep3_v10seg": {
        "h5_folder": "moge_hires_ep3_v10seg_h5",
        "exp_name": f"moge_hires_ep3_v10seg_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 v10seg",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v10seg_metric": {
        "h5_folder": "moge_hires_ep3_v10seg_metric_h5",
        "exp_name": f"moge_hires_ep3_v10seg_metric_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 v10seg metric",
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
    "moge_hires_ep3_seg_v10": {
        "h5_folder": "moge_hires_ep3_seg_v10_h5",
        "exp_name": f"moge_hires_ep3_seg_v10_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 Seg v10",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_seg_v10_merge_v5": {
        "h5_folder": "moge_hires_ep3_seg_v10_merge_v5_h5",
        "exp_name": f"moge_hires_ep3_seg_v10_merge_v5_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 Seg v10 + Merge v5",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_seg_v9vote": {
        "h5_folder": "moge_hires_ep3_seg_v9vote_h5",
        "exp_name": f"moge_hires_ep3_seg_v9vote_{EXP_VER}",
        "display_name": "MoGe HiRes ep3 Seg v9vote",
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
        "nonplanar_label": None,  # Uses 0 for non-planar
        "uses_gt_h5": False,
    },
    # ---- Segmentation cue ablations (v5, shared raw from moge_hires_ep3) ----
    "ablation_only_normal": {
        "h5_folder": "ablation_only_normal_h5",
        "exp_name": f"ablation_only_normal_{EXP_VER}",
        "display_name": "Ablation: Normal only",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ablation_only_depth": {
        "h5_folder": "ablation_only_depth_h5",
        "exp_name": f"ablation_only_depth_{EXP_VER}",
        "display_name": "Ablation: Depth only",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ablation_normal_depth": {
        "h5_folder": "ablation_normal_depth_h5",
        "exp_name": f"ablation_normal_depth_{EXP_VER}",
        "display_name": "Ablation: Normal+Depth",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ablation_full": {
        "h5_folder": "ablation_full_h5",
        "exp_name": f"ablation_full_{EXP_VER}",
        "display_name": "Ablation: Normal+Depth+Planarity",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    # ---- v5 segmentation variant ablations (2x2: normal x depth) ----
    "moge_hires_ep3_v5seg": {
        "h5_folder": "moge_hires_ep3_v5seg_h5",
        "exp_name": f"moge_hires_ep3_v5seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5 (Sobel+abs)",
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
    "moge_hires_ep3_v5_no_sobel_seg": {
        "h5_folder": "moge_hires_ep3_v5_no_sobel_seg_h5",
        "exp_name": f"moge_hires_ep3_v5_no_sobel_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5_nosob (dot+abs)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v5_dotprod_relative_seg": {
        "h5_folder": "moge_hires_ep3_v5_dotprod_relative_seg_h5",
        "exp_name": f"moge_hires_ep3_v5_dotprod_relative_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5_dot_rel (dot+rel)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    # ---- v5 segmentation variant ablations — original v5 params (plan=0.6, norm=10°, match=24) ----
    "moge_hires_ep3_v5origparams_seg": {
        "h5_folder": "moge_hires_ep3_v5origparams_seg_h5",
        "exp_name": f"moge_hires_ep3_v5origparams_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5orig (Sobel+abs)",
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
    "moge_hires_ep3_v5origparams_no_sobel_seg": {
        "h5_folder": "moge_hires_ep3_v5origparams_no_sobel_seg_h5",
        "exp_name": f"moge_hires_ep3_v5origparams_no_sobel_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5orig_nosob (dot+abs)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "moge_hires_ep3_v5origparams_dotprod_relative_seg": {
        "h5_folder": "moge_hires_ep3_v5origparams_dotprod_relative_seg_h5",
        "exp_name": f"moge_hires_ep3_v5origparams_dotprod_relative_seg_{EXP_VER}",
        "display_name": "MoGe ep3 v5orig_dot_rel (dot+rel)",
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
# H5 LOADING
# ============================================================

class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one scene in memory at a time.

    Handles non-planar region remapping for methods like ZeroPlane.
    """
    def __init__(self, h5_root: str, label_offset: int = 0, nonplanar_label: Optional[int] = None, h5_filename: str = "planes.h5"):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self.nonplanar_label = nonplanar_label  # Label to remap to 0 (e.g., 20 for ZeroPlane)
        self.h5_filename = h5_filename
        self._current_scene_id = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_scene(self, scene_id: str) -> bool:
        """Load a scene's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        if not os.path.exists(h5_path):
            return False

        # Clear previous scene to free memory
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
        self._frame_id_to_idx = {fid: i for i, fid in enumerate(self._current_frame_ids)}
        self._current_scene_id = scene_id
        return True

    def get_prediction(self, scene_id: str, frame_idx: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get prediction for a specific frame, loading scene if needed.

        Returns:
            labels: (H, W) plane labels, or None if not found
        """
        if not self._load_scene(scene_id):
            return None

        if frame_idx not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_idx]
        pred = self._current_planes[idx].copy()  # Copy to avoid modifying cached data

        # Resize to target shape if needed
        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Remap non-planar label to 0 (e.g., ZeroPlane uses 20 for non-planar)
        if self.nonplanar_label is not None:
            pred[pred == self.nonplanar_label] = 0

        # Apply label offset (usually 0 after remapping)
        if self.label_offset != 0:
            pred = pred + self.label_offset

        return pred.astype(np.int32)

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene exists without loading it."""
        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        return os.path.exists(h5_path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
    method_key: str,
    method_config: Dict,
    val_dataset: ScanNetPPPlaneDataset,
    val_loader: DataLoader,
    shard_id: int = None,
) -> Dict:
    """
    Evaluate a single method and save results.

    Args:
        shard_id: If set, save results as results_shard_{shard_id}.csv instead of results.csv

    Returns:
        results: {(scene_id, frame_id): metrics_dict}
    """
    uses_gt_h5 = method_config.get("uses_gt_h5", False)
    exp_name = method_config["exp_name"]
    label_offset = method_config["label_offset"]
    csv_out_dir = EVAL_ROOT / exp_name

    if uses_gt_h5:
        h5_root = None
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print(f"H5 root: N/A (using GT labels)")
        print(f"Output: {csv_out_dir}")
        print(f"{'='*60}")
    else:
        if "h5_root_override" in method_config:
            h5_root = Path(method_config["h5_root_override"])
        else:
            h5_root = H5_ROOT / method_config["h5_folder"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print(f"H5 root: {h5_root}")
        print(f"Output: {csv_out_dir}")
        print(f"{'='*60}")

        if not h5_root.exists():
            print(f"[ERROR] H5 root does not exist: {h5_root}")
            return {}

    timer = Timer()

    # Initialize lazy loader (skip for GT method)
    loader = None
    if not uses_gt_h5:
        nonplanar_label = method_config.get("nonplanar_label", None)
        h5_filename = method_config.get("h5_filename", "planes.h5")
        loader = LazyH5SceneLoader(str(h5_root), label_offset=label_offset, nonplanar_label=nonplanar_label, h5_filename=h5_filename)

        # Check available scenes
        scene_ids = val_dataset.scene_ids
        available_scenes = [s for s in scene_ids if loader.has_scene(s)]
        missing_scenes = set(scene_ids) - set(available_scenes)
        if missing_scenes:
            print(f"[WARN] Missing predictions for {len(missing_scenes)} scenes")
        print(f"[DATA] Found predictions for {len(available_scenes)}/{len(scene_ids)} scenes")

        if len(available_scenes) == 0:
            print(f"[ERROR] No predictions found for {method_key}")
            return {}
    else:
        print(f"[DATA] Using GT labels as predictions for {len(val_dataset)} frames")

    # Evaluation wrapper (uses threshold-consistent RANSAC)
    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds):
        return evaluate_single_frame(
            scene_id,
            frame_idx,
            depth_np,
            gt_seg_np,
            K_np,
            c2w_np,
            labels,
            thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc=f"Evaluating {method_key}"):
            scene_ids_batch = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            B = len(scene_ids_batch)

            # Prepare batch data
            batch_items = []
            for i in range(B):
                scene_id = scene_ids_batch[i]
                frame_idx = frame_ids[i]

                # Get GT
                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
                H, W = gt_seg_np.shape

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                # Get prediction
                if uses_gt_h5:
                    # Use GT labels directly as prediction (upper bound)
                    labels = gt_seg_np.copy()
                else:
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

            # Parallel evaluation
            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(eval_frame_wrapper)(
                    item["scene_id"],
                    item["frame_idx"],
                    item["depth_np"],
                    item["gt_seg_np"],
                    item["K_np"],
                    item["c2w_np"],
                    item["labels"],
                    THRESHOLDS
                )
                for item in batch_items
            )

            for (metrics, labels), item in zip(outputs, batch_items):
                scene_id = item["scene_id"]
                frame_id = item["frame_idx"]
                results[(scene_id, frame_id)] = metrics

    print(f"[PIPELINE] Evaluated {len(results)} frames (skipped {skipped_frames})")

    # Save results
    if results:
        if shard_id is not None:
            # Save as shard file (for distributed eval)
            csv_out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame.from_records(list(results.values()))
            shard_path = csv_out_dir / f"results_shard_{shard_id}.csv"
            df.to_csv(shard_path, index=False)
            print(f"[CSV] Saved shard {shard_id} ({len(results)} frames) to {shard_path}")
        else:
            print("==> Saving results")
            save_results_csv(results, str(csv_out_dir))
        save_runtime(timer, str(csv_out_dir))
        timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================

def _merge_shards(exp_dir: Path):
    """Merge shard CSV files into results.csv, then produce per-scene and dataset CSVs.

    Looks for results_shard_*.csv in exp_dir, concatenates them, and calls
    save_results_csv() to produce the standard results.csv / results_per_scene.csv /
    results_dataset.csv files.
    """
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


def aggregate_results(methods: list, output_dir: Path = None):
    """
    Aggregate results from specified methods into summary tables.
    Merges shard files first if present.
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    # Merge shards for each method if needed
    for method_key in methods:
        if method_key not in METHODS:
            continue
        exp_dir = EVAL_ROOT / METHODS[method_key]["exp_name"]
        if exp_dir.exists():
            _merge_shards(exp_dir)

    # Collect results for specified methods
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

            # Precision/recall/F1 metrics (dynamically from THRESHOLDS)
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

            # Binary planarity metrics
            for bp_col in ["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"]:
                mean_col = f"{bp_col}_mean"
                if mean_col in df.index:
                    row[bp_col] = df[mean_col]

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
    out_path = output_dir / "table_precision_recall_baselines.csv"
    df_pr.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 2: Segmentation
    seg_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    df_seg = df_all[[c for c in seg_cols if c in df_all.columns]]
    out_path = output_dir / "table_segmentation_baselines.csv"
    df_seg.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 3: Combined summary (all thresholds for P/R + bp metrics)
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        combined_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"])
    combined_cols.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]
    out_path = output_dir / "table_combined_baselines.csv"
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

    return df_all


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate all baseline methods")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to evaluate (default: all). Options: {list(METHODS.keys())}")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate (for testing)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save aggregated tables")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--scene-start", type=int, default=None,
                        help="Start scene index (for distributed eval across SLURM array jobs)")
    parser.add_argument("--scene-end", type=int, default=None,
                        help="End scene index exclusive (for distributed eval)")
    args = parser.parse_args()

    # Determine which methods to evaluate
    if args.methods is None:
        methods_to_eval = list(METHODS.keys())
    else:
        methods_to_eval = args.methods
        invalid = set(methods_to_eval) - set(METHODS.keys())
        if invalid:
            print(f"[ERROR] Invalid methods: {invalid}")
            print(f"[ERROR] Valid options: {list(METHODS.keys())}")
            return

    print(f"[CONFIG] Methods to evaluate: {methods_to_eval}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    if not args.aggregate_only:
        # Load dataset once
        print("\n==> Loading dataset")
        val_dataset = ScanNetPPPlaneDataset(
            rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=os.path.join(DATASET_DIR, ""),
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split="test",
            max_scenes=args.max_scenes,
        )
        print(f"[DATA] Test set: {len(val_dataset)} frames")

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
            print(f"[DATA] Scene range [{s}:{e}] → {len(subset_scenes)} scenes, {len(val_dataset)} frames")

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Derive shard_id from scene_start (for distributed eval)
        shard_id = None
        if args.scene_start is not None:
            shard_id = args.scene_start

        # Evaluate each method
        for method_key in methods_to_eval:
            method_config = METHODS[method_key]
            evaluate_method(method_key, method_config, val_dataset, val_loader, shard_id=shard_id)

    # Aggregate results (merge shards if needed)
    # Skip aggregation when running as a shard job — let the dedicated --aggregate-only job handle it
    if args.scene_start is None:
        aggregate_results(methods_to_eval, Path(args.output_dir))

    print("\n[DONE] All evaluations complete!")


if __name__ == "__main__":
    main()
