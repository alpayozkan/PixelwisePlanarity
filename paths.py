# Centralized path configuration for datasets, GT data, and model checkpoints.

import os

# This repository's root (split lists in splits/ resolve against it).
repo_path = os.path.dirname(os.path.abspath(__file__))

# ============ Dataset dirs (original) ============
hypersim_path = '/cluster/project/cvg/Shared_datasets/Hypersim'
scannetpp_path = '/cluster/project/cvg/Shared_datasets/scannet++'
scannetppv2_path = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'

# ============ Our plane GT dataset dirs (3D mesh plane extraction) ============
# NOTE: Hypersim mesh GT was purged from ayavuz's scratch — restore data here before use.
hypersim_plane_path = '/cluster/scratch/aoezkan/planeseg/hypersim/hypersim_mesh_ours'
scannetpp_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp/'

# ============ Hypersim processed data ============
# Unified dataset (RGB + depth + rendered plane labels under one root)
hypersim_merged_path = '/cluster/scratch/aoezkan/planeseg/dataset/hypersim'
hypersim_rendered_path = '/cluster/scratch/aoezkan/planeseg/dataset/hypersim'
# camera parameters (not in the unified dataset, kept separately)
# NOTE: purged from ayavuz's scratch — restore data here before use.
hypersim_params_path = '/cluster/scratch/aoezkan/planeseg/hypersim/Hypersim_params'

# ============ ScanNet++ rendered data ============
# Local copy of ayavuz's SCANNETPP_BACKUP (populate via scripts/copy_ayavuz_assets.sh)
scannetpp_merged_path = '/cluster/scratch/aoezkan/planeseg/dataset/scannetpp'
scannetpp_rend_plane_path = '/cluster/scratch/aoezkan/planeseg/scannetpp/plane_gt_rendered'

# ============ NYU-v2 and 7-Scenes (ZeroPlane "_d2" NPZ format) ============
nyuv2_path = '/cluster/scratch/aoezkan/planeseg/dataset/nyuv2_plane'
sevenscenes_path = '/cluster/scratch/aoezkan/planeseg/dataset/sevenscenes_plane'

# ============ Outdoor datasets: SYNTHIA and VKITTI2 ============
# NOTE: plane GT was purged from ayavuz's scratch — restore data here before use.
synthia_path = '/cluster/scratch/aoezkan/planeseg/synthia/synthia_planes'
vkitti2_path = '/cluster/scratch/aoezkan/planeseg/vkitti2/vkitti2_planes'
# raw SYNTHIA RGB/depth (used by gt_creation/synthia check utilities)
synthia_raw_path = '/cluster/scratch/aoezkan/planeseg/synthia/raw'

# ============ Models / checkpoints ============
# Trained MoGe 4-head planarity checkpoint (HIRES, 4 datasets, epoch 1 — production)
# Local copy of ayavuz's checkpoints (populate via scripts/copy_ayavuz_assets.sh)
planarity_model_path = '/cluster/scratch/aoezkan/planeseg/checkpoints/moge_HIRES_4datasets/model_epoch1.pt'
# HuggingFace cache for MoGe base weights (export as HF_HOME)
moge_cache_dir = '/cluster/scratch/aoezkan/cache/huggingface'

# ============ Evaluation / inference output roots ============
eval_root = '/cluster/scratch/aoezkan/planeseg/scannetpp/eval'
synthia_eval_root = '/cluster/scratch/aoezkan/planeseg/synthia/eval'
synthia_h5_root = '/cluster/scratch/aoezkan/planeseg/synthia/inference'
vkitti2_eval_root = '/cluster/scratch/aoezkan/planeseg/vkitti2/eval'
vkitti2_h5_root = '/cluster/scratch/aoezkan/planeseg/vkitti2/inference'
# 3D qualitative comparison renders
vis3d_root = '/cluster/scratch/aoezkan/planeseg/3d_vis/scannetpp'
inference_h5_root = '/cluster/scratch/aoezkan/planeseg/scannetpp/inference'
# Our method's predicted plane labels (per-scene planes.h5) evaluated as "ours"
# by evaluate_all_baselines.py; produced by the signals->planes pipeline
# (4-head MoGe moge_HIRES_4datasets ep1, 1440x1920).
ours_planes_root = '/cluster/scratch/aoezkan/planeseg/inference/moge_ours_ep1'
