# Centralized path configuration for datasets, GT data, and model checkpoints.
#
# Machine-specific locations resolve from a single data root:
#   1. set the PXWPLANAR_DATA_ROOT environment variable (default: <repo>/data), or
#   2. create pxwplanar/paths_local.py (gitignored) redefining any variables below.

import os

# This repository's root (split lists in splits/ resolve against it).
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root for all datasets, checkpoints, and outputs.
data_root = os.environ.get('PXWPLANAR_DATA_ROOT', os.path.join(repo_path, 'data'))

# ============ Dataset dirs (original) ============
hypersim_path = os.path.join(data_root, 'datasets', 'hypersim')
scannetpp_path = os.path.join(data_root, 'datasets', 'scannetpp')
scannetppv2_path = os.path.join(data_root, 'datasets', 'scannetpp_v2')

# ============ Our plane GT dataset dirs (3D mesh plane extraction) ============
hypersim_plane_path = os.path.join(data_root, 'plane_gt_mesh', 'hypersim')
scannetpp_plane_path = os.path.join(data_root, 'plane_gt_mesh', 'scannetpp')

# ============ Hypersim processed data ============
# Unified dataset (RGB + depth + rendered plane labels under one root)
hypersim_merged_path = os.path.join(data_root, 'dataset', 'hypersim')
hypersim_rendered_path = os.path.join(data_root, 'dataset', 'hypersim')
# camera parameters (not in the unified dataset, kept separately)
hypersim_params_path = os.path.join(data_root, 'hypersim_params')

# ============ ScanNet++ rendered data ============
scannetpp_merged_path = os.path.join(data_root, 'dataset', 'scannetpp')
scannetpp_rend_plane_path = os.path.join(data_root, 'plane_gt_rendered', 'scannetpp')

# ============ NYU-v2 and 7-Scenes (ZeroPlane "_d2" NPZ format) ============
nyuv2_path = os.path.join(data_root, 'dataset', 'nyuv2_plane')
sevenscenes_path = os.path.join(data_root, 'dataset', 'sevenscenes_plane')

# ============ Outdoor datasets: SYNTHIA and VKITTI2 ============
synthia_path = os.path.join(data_root, 'synthia', 'synthia_planes')
vkitti2_path = os.path.join(data_root, 'vkitti2', 'vkitti2_planes')
# raw SYNTHIA RGB/depth (used by gt_creation/synthia check utilities)
synthia_raw_path = os.path.join(data_root, 'synthia', 'raw')

# ============ Models / checkpoints ============
# Trained MoGe 4-head planarity checkpoint (HIRES, 4 datasets, epoch 1 — production)
planarity_model_path = os.path.join(data_root, 'checkpoints', 'moge_HIRES_4datasets', 'model_epoch1.pt')
# MoGe base weights are pulled from HuggingFace into the standard cache
# (~/.cache/huggingface; override via HF_HOME).

# ============ Evaluation / inference output roots ============
eval_root = os.path.join(data_root, 'eval', 'scannetpp')
synthia_eval_root = os.path.join(data_root, 'eval', 'synthia')
synthia_h5_root = os.path.join(data_root, 'inference', 'synthia')
vkitti2_eval_root = os.path.join(data_root, 'eval', 'vkitti2')
vkitti2_h5_root = os.path.join(data_root, 'inference', 'vkitti2')
# 3D qualitative comparison renders
vis3d_root = os.path.join(data_root, '3d_vis', 'scannetpp')
inference_h5_root = os.path.join(data_root, 'inference', 'scannetpp')
# Our method's predicted plane labels (per-scene planes.h5) evaluated as "ours"
# by evaluate_all_baselines.py; produced by the signals->planes pipeline
# (4-head MoGe moge_HIRES_4datasets ep1, 1440x1920).
ours_planes_root = os.path.join(data_root, 'inference', 'moge_ours_ep1')

# Optional site-specific overrides (create pxwplanar/paths_local.py; gitignored).
try:
    from pxwplanar.paths_local import *  # noqa: F401,F403
except ImportError:
    pass
