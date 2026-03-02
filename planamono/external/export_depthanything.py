#!/usr/bin/env python3
"""
Export plane segmentation using Depth Anything V2 depth + our MoGe planarity head.

Pipeline: Our planarity + DAv2 depth → depth_to_normal → plan2seg → plane labels

Supports 4 datasets: scannetpp, hypersim, vkitti2, synthia (test splits).
Saves per-scene H5 files with depth, normals, planarity, mask, planes, gt_planes, intrinsics.

Usage:
    python planamono/external/export_depthanything.py \
        --dataset scannetpp \
        --checkpoint /path/to/moge_planarity.pt \
        --dav2_checkpoint /path/to/depth_anything_v2_vitl.pth

    # Quick test with 5 frames:
    python planamono/external/export_depthanything.py \
        --dataset scannetpp --max_frames 5 \
        --checkpoint /path/to/moge_planarity.pt \
        --dav2_checkpoint /path/to/depth_anything_v2_vitl.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as torchF
import cv2
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.shared.utils.depth_normal import depth_to_normal_remi
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hypersim_hdr_rgb(h5_path, percentile=90, target_max=0.8, gamma=2.2):
    """Load and tone-map Hypersim HDR image to uint8 RGB."""
    with h5py.File(h5_path, 'r') as f:
        hdr = f['dataset'][:].astype(np.float32)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=1e4, neginf=0.0)
    hdr = np.clip(hdr, 0, 1e4)
    brightness = hdr.mean(axis=2)
    scale_val = np.nanpercentile(brightness, percentile)
    scale_val = max(scale_val, 1e-6) if np.isfinite(scale_val) else 1.0
    img = hdr * (target_max / scale_val)
    img = np.clip(img, 0, None) ** (1.0 / gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def compute_K_from_M_cam_from_uv(M_cam_from_uv, H, W):
    """Compute CV-convention K from Hypersim's M_cam_from_uv."""
    M = np.array(M_cam_from_uv, dtype=np.float64)
    pixel_to_ndc = np.array([
        [2.0 / W, 0.0, (1.0 - W) / W],
        [0.0, -2.0 / H, (H - 1.0) / H],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    K_inv_opengl = M @ pixel_to_ndc
    K_inv_cv = K_inv_opengl.copy()
    K_inv_cv[1, :] *= -1.0
    K_inv_cv[2, :] *= -1.0
    K = np.linalg.inv(K_inv_cv)
    return K.astype(np.float32)


def scale_K(K, native_w, native_h, out_w, out_h):
    """Scale intrinsic matrix from native to output resolution."""
    K_out = K.copy()
    K_out[0, :] *= out_w / native_w
    K_out[1, :] *= out_h / native_h
    return K_out


def make_K(fx, fy, cx, cy):
    """Build 3x3 intrinsic matrix."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def resize_gt_planes(planes, out_h, out_w):
    """Resize GT plane labels with nearest interpolation."""
    return cv2.resize(planes, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)


def preprocess_for_moge(rgb_uint8, device):
    """Resize to 476x644 and convert to tensor for MoGe."""
    resized = cv2.resize(rgb_uint8, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


def run_moge_planarity(model, rgb_uint8, device, out_h=480, out_w=640,
                       return_normals=False):
    """Run our MoGe planarity model and return probability map at (out_h, out_w).

    If return_normals=True, also returns MoGe-predicted normals at (out_h, out_w, 3).
    """
    tensor = preprocess_for_moge(rgb_uint8, device)
    with torch.no_grad():
        output = model.model.forward(tensor.unsqueeze(0), num_tokens=1600)
    planarity = output['planarity'][0]  # (476, 644)
    planarity = torchF.interpolate(
        planarity[None, None], (out_h, out_w),
        mode='bilinear', align_corners=False
    )[0, 0]
    planarity_np = planarity.cpu().numpy().astype(np.float32)

    if not return_normals:
        return planarity_np

    normal = output['normal'][0]  # (476, 644, 3)
    normal = torchF.interpolate(
        normal.permute(2, 0, 1)[None], (out_h, out_w),
        mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0)
    # Re-normalize
    norm_mag = torch.norm(normal, dim=2, keepdim=True).clamp(min=1e-8)
    normal = normal / norm_mag
    normal_np = normal.cpu().numpy().astype(np.float32)

    return planarity_np, normal_np


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def random_label_colormap(num_labels):
    """Generate a random colormap for segment labels."""
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = 0  # background = black
    return colors


def save_vis_png(rgb, depth, depth_normals, moge_normals, planarity,
                 depth_labels, moge_labels, gt_planes,
                 out_path, scene_label, frame_id):
    """Save a single-frame visualization as a 3x3 grid PNG.

    Row 1: RGB, DAv2 Depth, Planarity
    Row 2: Depth-derived normals, segmentation, overlay
    Row 3: MoGe normals, segmentation, overlay
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"{scene_label} / {frame_id}", fontsize=14)

    # --- Row 1: RGB, Depth, Planarity ---
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    valid = depth > 0
    vmin = depth[valid].min() if valid.any() else 0
    vmax = depth[valid].max() if valid.any() else 1
    axes[0, 1].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("DAv2 Depth")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(planarity, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Planarity")
    axes[0, 2].axis("off")

    # --- Row 2: Depth-derived normals + segmentation ---
    dn_vis = (depth_normals * 0.5 + 0.5).clip(0, 1)
    axes[1, 0].imshow(dn_vis)
    axes[1, 0].set_title("Depth-derived Normals")
    axes[1, 0].axis("off")

    n_dl = int(depth_labels.max()) + 1
    cmap_dl = random_label_colormap(n_dl)
    seg_dl = cmap_dl[depth_labels.astype(np.int32)]
    axes[1, 1].imshow(seg_dl)
    axes[1, 1].set_title(f"Seg (depth normals, {n_dl - 1} planes)")
    axes[1, 1].axis("off")

    overlay_dl = rgb.copy().astype(np.float32)
    m_dl = depth_labels > 0
    overlay_dl[m_dl] = overlay_dl[m_dl] * 0.4 + seg_dl[m_dl].astype(np.float32) * 0.6
    axes[1, 2].imshow(overlay_dl.astype(np.uint8))
    axes[1, 2].set_title("Overlay (depth normals)")
    axes[1, 2].axis("off")

    # --- Row 3: MoGe normals + segmentation ---
    mn_vis = (moge_normals * 0.5 + 0.5).clip(0, 1)
    axes[2, 0].imshow(mn_vis)
    axes[2, 0].set_title("MoGe Normals")
    axes[2, 0].axis("off")

    n_ml = int(moge_labels.max()) + 1
    cmap_ml = random_label_colormap(n_ml)
    seg_ml = cmap_ml[moge_labels.astype(np.int32)]
    axes[2, 1].imshow(seg_ml)
    axes[2, 1].set_title(f"Seg (MoGe normals, {n_ml - 1} planes)")
    axes[2, 1].axis("off")

    overlay_ml = rgb.copy().astype(np.float32)
    m_ml = moge_labels > 0
    overlay_ml[m_ml] = overlay_ml[m_ml] * 0.4 + seg_ml[m_ml].astype(np.float32) * 0.6
    axes[2, 2].imshow(overlay_ml.astype(np.uint8))
    axes[2, 2].set_title("Overlay (MoGe normals)")
    axes[2, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scene iterators: yield (scene_label, h5_rel, frame_ids, rgbs, gt_planes, Ks)
#   rgbs: list of (H, W, 3) uint8 at output resolution
#   gt_planes: list of (H, W) uint16 at output resolution
#   Ks: list of (3, 3) float32 at output resolution
# ---------------------------------------------------------------------------

def iter_scannetpp_scenes(args):
    split_file = os.path.join(args.splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene_id in scenes:
        gt_h5 = os.path.join(args.scannetpp_gt_root, scene_id, "rendered.h5")
        if not os.path.exists(gt_h5):
            continue
        pose_file = os.path.join(args.scannetpp_rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")
        if not os.path.exists(pose_file):
            print(f"  Skipping {scene_id}: no pose_intrinsic_imu.json")
            continue
        with open(pose_file, "r") as f:
            pose_data = json.load(f)
        with h5py.File(gt_h5, "r") as hf:
            all_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in hf["frame_ids"][:]]
            gt_planes_all = hf["planes"][:]  # (N, 1440, 1920) uint16
        frame_ids, rgbs, gt_planes, Ks = [], [], [], []
        for idx, fid in enumerate(all_fids):
            rgb_path = os.path.join(args.scannetpp_rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
            if not os.path.exists(rgb_path):
                continue
            if fid not in pose_data:
                continue
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            native_h, native_w = rgb.shape[:2]
            K_native = np.array(pose_data[fid]["intrinsic"], dtype=np.float32)
            K_out = scale_K(K_native, native_w, native_h, args.width, args.height)
            rgbs.append(cv2.resize(rgb, (args.width, args.height)))
            gt_planes.append(resize_gt_planes(gt_planes_all[idx], args.height, args.width))
            Ks.append(K_out)
            frame_ids.append(fid)
        if frame_ids:
            yield scene_id, os.path.join(scene_id, "rendered_v2.h5"), frame_ids, rgbs, gt_planes, Ks


def load_hypersim_intrinsics_map(metadata_csv):
    """Load per-scene M_cam_from_uv from metadata CSV."""
    df = pd.read_csv(metadata_csv)
    intrinsics = {}
    for _, row in df.iterrows():
        scene = row['scene_name']
        M = np.array([
            [row['M_cam_from_uv_00'], row['M_cam_from_uv_01'], row['M_cam_from_uv_02']],
            [row['M_cam_from_uv_10'], row['M_cam_from_uv_11'], row['M_cam_from_uv_12']],
            [row['M_cam_from_uv_20'], row['M_cam_from_uv_21'], row['M_cam_from_uv_22']],
        ], dtype=np.float64)
        intrinsics[scene] = M
    return intrinsics


def iter_hypersim_scenes(args):
    split_csv = os.path.join(args.splits_root, "hypersim",
                             "metadata_images_split_with_planes_filtered.csv")
    df = pd.read_csv(split_csv)
    test_df = df[df["split_partition_name"] == "test"]
    groups = test_df.groupby(["scene_name", "camera_name"])

    metadata_csv = os.path.join(args.hypersim_data_root, "metadata_camera_parameters.csv")
    intrinsics_map = load_hypersim_intrinsics_map(metadata_csv)
    native_h, native_w = 768, 1024

    for (scene, cam), group_df in groups:
        M = intrinsics_map.get(scene)
        if M is None:
            print(f"  Skipping {scene}/{cam}: no intrinsics")
            continue
        K_native = compute_K_from_M_cam_from_uv(M, native_h, native_w)
        K_out = scale_K(K_native, native_w, native_h, args.width, args.height)

        gt_h5_path = os.path.join(args.hypersim_data_root, scene, f"rendered_planes_{cam}.h5")
        gt_fid_to_idx = {}
        gt_planes_all = None
        if os.path.exists(gt_h5_path):
            with h5py.File(gt_h5_path, "r") as gt_f:
                gt_fids_raw = gt_f["frame_ids"][:]
                gt_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                           for fid in gt_fids_raw]
                gt_fid_to_idx = {fid: i for i, fid in enumerate(gt_fids)}
                gt_planes_all = gt_f["planes"][:]

        frame_ids, rgbs, gt_planes, Ks = [], [], [], []
        for fid in sorted(group_df["frame_id"].tolist()):
            fid = int(fid)
            rgb_path = os.path.join(
                args.hypersim_data_root, scene, "images",
                f"scene_{cam}_final_hdf5", f"frame.{fid:04d}.color.hdf5")
            if not os.path.exists(rgb_path):
                continue
            rgb = load_hypersim_hdr_rgb(rgb_path)
            rgbs.append(cv2.resize(rgb, (args.width, args.height)))
            Ks.append(K_out)
            frame_ids.append(f"{fid:04d}")
            fid_str = f"{fid:04d}"
            gt_idx = gt_fid_to_idx.get(fid_str)
            if gt_idx is not None and gt_planes_all is not None:
                gt_planes.append(resize_gt_planes(gt_planes_all[gt_idx], args.height, args.width))
            else:
                gt_planes.append(np.zeros((args.height, args.width), dtype=np.uint16))
        if frame_ids:
            yield f"{scene}/{cam}", os.path.join(scene, f"rendered_planes_{cam}.h5"), \
                  frame_ids, rgbs, gt_planes, Ks


def iter_vkitti2_scenes(args):
    from planamono.shared.datasets.vkitti2 import FX, FY, CX, CY
    split_file = os.path.join(args.splits_root, "vkitti2", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.vkitti2_plane_root, scene, "clone", "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            native_h, native_w = hf["rgb"].shape[1], hf["rgb"].shape[2]
            K_native = make_K(FX, FY, CX, CY)
            K_out = scale_K(K_native, native_w, native_h, args.width, args.height)
            frame_ids = [f"{i:04d}" for i in range(n)]
            rgbs = [cv2.resize(hf["rgb"][i], (args.width, args.height)) for i in range(n)]
            if "planes" in hf:
                gt_planes = [resize_gt_planes(hf["planes"][i], args.height, args.width) for i in range(n)]
            else:
                gt_planes = [np.zeros((args.height, args.width), dtype=np.uint16)] * n
        Ks = [K_out] * n
        yield f"{scene}/clone", os.path.join(scene, "clone", "rendered_v2.h5"), \
              frame_ids, rgbs, gt_planes, Ks


def iter_synthia_scenes(args):
    from planamono.shared.datasets.synthia import FX, FY, CX, CY
    split_file = os.path.join(args.splits_root, "synthia", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.synthia_plane_root, "test", scene, "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            native_h, native_w = hf["rgb"].shape[1], hf["rgb"].shape[2]
            K_native = make_K(FX, FY, CX, CY)
            K_out = scale_K(K_native, native_w, native_h, args.width, args.height)
            frame_ids = [f"{i:04d}" for i in range(n)]
            rgbs = [cv2.resize(hf["rgb"][i], (args.width, args.height)) for i in range(n)]
            if "planes" in hf:
                gt_planes = [resize_gt_planes(hf["planes"][i], args.height, args.width) for i in range(n)]
            else:
                gt_planes = [np.zeros((args.height, args.width), dtype=np.uint16)] * n
        Ks = [K_out] * n
        yield scene, os.path.join(scene, "rendered_v2.h5"), \
              frame_ids, rgbs, gt_planes, Ks


DATASET_ITERS = {
    "scannetpp": iter_scannetpp_scenes,
    "hypersim": iter_hypersim_scenes,
    "vkitti2": iter_vkitti2_scenes,
    "synthia": iter_synthia_scenes,
}


# ---------------------------------------------------------------------------
# Depth Anything V2 model loading
# ---------------------------------------------------------------------------

DAV2_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def load_dav2_model(args):
    """Load Depth Anything V2 model.

    Metric and non-metric variants use different classes from different paths
    within the DAv2 repo.
    """
    dav2_repo = os.path.expanduser(args.dav2_repo)
    config = DAV2_CONFIGS[args.dav2_encoder].copy()

    if args.metric_depth:
        sys.path.insert(0, os.path.join(dav2_repo, 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        config['max_depth'] = args.max_depth
    else:
        sys.path.insert(0, dav2_repo)
        from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(args.dav2_checkpoint, map_location='cpu'))
    model = model.to(args.device).eval()
    return model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Export plane segmentation: DAv2 depth + our MoGe planarity")

    # Our planarity model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to MoGe planarity checkpoint (.pt)")

    # DAv2 model
    p.add_argument("--dav2_encoder", type=str, default="vitl",
                   choices=["vits", "vitb", "vitl"])
    p.add_argument("--dav2_checkpoint", type=str,
                   default="/cluster/scratch/ayavuz/checkpoints/depth_anything_v2_vitl.pth",
                   help="Path to DAv2 weights")
    p.add_argument("--dav2_repo", type=str, default="~/Depth-Anything-V2",
                   help="Path to Depth-Anything-V2 repo")
    p.add_argument("--metric_depth", action="store_true",
                   help="Use metric depth variant")
    p.add_argument("--max_depth", type=float, default=20.0,
                   help="Max depth for metric variant (20=indoor, 80=outdoor)")
    p.add_argument("--use_moge_normals", action="store_true",
                   help="Use MoGe-predicted normals instead of depth-derived normals")
    p.add_argument("--depth_blur_sigma", type=float, default=0.0,
                   help="Gaussian blur sigma on depth before normal computation (0=off, try 1.0-2.0 for metric depth)")

    # Dataset / output
    p.add_argument("--dataset", type=str, required=True,
                   choices=["scannetpp", "hypersim", "vkitti2", "synthia", "all"])
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output root (default: /cluster/scratch/ayavuz/dataset/depthanything_{dataset})")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--test_vis", action="store_true",
                   help="Test mode: run 5 frames and save visualization PNGs only (no H5)")

    # plan2seg thresholds
    p.add_argument("--planarity_threshold", type=float, default=0.5)
    p.add_argument("--normal_threshold_rad", type=float, default=0.15,
                   help="Normal similarity threshold in radians for plan2seg")
    p.add_argument("--depth_threshold", type=float, default=0.1,
                   help="Relative depth threshold (fraction of center depth) for plan2seg")

    # Dataset paths
    p.add_argument("--splits_root", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "splits"))
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_dataset(dataset_name, moge_model, dav2_model, args):
    if args.output_dir:
        ds_out = os.path.join(args.output_dir, dataset_name)
    else:
        ds_out = f"/cluster/scratch/ayavuz/dataset/depthanything_{dataset_name}"
    os.makedirs(ds_out, exist_ok=True)

    test_vis = getattr(args, 'test_vis', False)
    max_frames = args.max_frames

    total_frames = 0
    total_scenes = 0

    for scene_label, h5_rel, frame_ids, rgbs, gt_planes_list, Ks in \
            DATASET_ITERS[dataset_name](args):
        if test_vis and total_scenes >= 5:
            break
        if max_frames is not None and total_frames >= max_frames:
            break

        n = len(frame_ids)
        if test_vis:
            n = 1  # 1 frame per scene in test_vis mode
        elif max_frames is not None:
            n = min(n, max_frames - total_frames)
        frame_ids = frame_ids[:n]
        rgbs = rgbs[:n]
        gt_planes_list = gt_planes_list[:n]
        Ks = Ks[:n]

        # Preallocate
        depth_all = np.zeros((n, args.height, args.width), dtype=np.float32)
        normals_all = np.zeros((n, args.height, args.width, 3), dtype=np.float32)
        planarity_all = np.zeros((n, args.height, args.width), dtype=np.float32)
        mask_all = np.zeros((n, args.height, args.width), dtype=np.float32)
        planes_all = np.zeros((n, args.height, args.width), dtype=np.uint16)
        gt_planes_out = np.stack(gt_planes_list[:n], axis=0)
        intrinsics_all = np.stack(Ks[:n], axis=0)

        for i, rgb in enumerate(tqdm(rgbs, desc=f"  {scene_label}", leave=False)):
            # 1. Our MoGe planarity (always get normals in test_vis mode)
            need_moge_normals = args.use_moge_normals or test_vis
            if need_moge_normals:
                planarity, moge_normals = run_moge_planarity(
                    moge_model, rgb, args.device,
                    args.height, args.width, return_normals=True)
            else:
                planarity = run_moge_planarity(moge_model, rgb, args.device,
                                               args.height, args.width)

            # 2. DAv2 depth (expects BGR uint8)
            raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = dav2_model.infer_image(raw_bgr)  # (H, W) numpy
            depth = depth.astype(np.float32)
            if depth.shape != (args.height, args.width):
                depth = cv2.resize(depth, (args.width, args.height),
                                   interpolation=cv2.INTER_LINEAR)

            # 3. Compute depth-derived normals
            K = Ks[i]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if args.depth_blur_sigma > 0:
                ksize = int(np.ceil(args.depth_blur_sigma * 3) * 2 + 1)
                depth_for_normals = cv2.GaussianBlur(depth, (ksize, ksize), args.depth_blur_sigma)
            else:
                depth_for_normals = depth
            depth_normals = depth_to_normal_remi(depth_for_normals, fx, fy, cx, cy)

            # Pick which normals to use for the H5 output
            normals = moge_normals if args.use_moge_normals else depth_normals

            # 4. Validity mask
            valid_mask = (depth > 0).astype(np.float32)

            # 5. Plan2seg with chosen normals
            planarity_mask = (planarity > args.planarity_threshold).astype(np.uint8)
            labels, _ = compute_vectorized_planar_segments_v5_relative(
                planarity_mask, normals, depth,
                normal_threshold_rad=args.normal_threshold_rad,
                depth_threshold=args.depth_threshold,
                device=args.device,
            )

            depth_all[i] = depth
            normals_all[i] = normals
            planarity_all[i] = planarity
            mask_all[i] = valid_mask
            planes_all[i] = labels.astype(np.uint16)

            # Save visualization PNG in test_vis mode
            if test_vis:
                # Run plan2seg with both normal sources for comparison
                depth_labels, _ = compute_vectorized_planar_segments_v5_relative(
                    planarity_mask, depth_normals, depth,
                    normal_threshold_rad=args.normal_threshold_rad,
                    depth_threshold=args.depth_threshold,
                    device=args.device,
                )
                moge_labels, _ = compute_vectorized_planar_segments_v5_relative(
                    planarity_mask, moge_normals, depth,
                    normal_threshold_rad=args.normal_threshold_rad,
                    depth_threshold=args.depth_threshold,
                    device=args.device,
                )
                safe_scene = scene_label.replace("/", "_")
                vis_path = os.path.join(ds_out, "test_vis",
                                        f"{safe_scene}_{frame_ids[i]}.png")
                save_vis_png(rgb, depth, depth_normals, moge_normals, planarity,
                             depth_labels, moge_labels, gt_planes_list[i],
                             vis_path, scene_label, frame_ids[i])
                tqdm.write(f"    Vis: {vis_path}")

        # Save H5 (skip in test_vis mode)
        if not test_vis:
            out_h5 = os.path.join(ds_out, h5_rel)
            os.makedirs(os.path.dirname(out_h5), exist_ok=True)
            with h5py.File(out_h5, "w") as f:
                dt = h5py.string_dtype()
                f.create_dataset("frame_ids", data=frame_ids[:n], dtype=dt)
                f.create_dataset("depth", data=depth_all, dtype=np.float32)
                f.create_dataset("normals", data=normals_all, dtype=np.float32)
                f.create_dataset("planarity", data=planarity_all, dtype=np.float32)
                f.create_dataset("mask", data=mask_all, dtype=np.float32)
                f.create_dataset("planes", data=planes_all, dtype=np.uint16)
                f.create_dataset("gt_planes", data=gt_planes_out, dtype=np.uint16)
                f.create_dataset("intrinsics", data=intrinsics_all, dtype=np.float32)
            tqdm.write(f"  Saved: {out_h5} ({n} frames)")

        total_frames += n
        total_scenes += 1

    print(f"  {dataset_name}: {total_scenes} scene(s), {total_frames} frames -> {ds_out}")


def main():
    args = parse_args()
    datasets = list(DATASET_ITERS.keys()) if args.dataset == "all" else [args.dataset]

    # Load models
    moge_wrapper = MoGePlanarityInference(args.checkpoint, device=args.device)
    dav2_model = load_dav2_model(args)

    out_label = args.output_dir or "/cluster/scratch/ayavuz/dataset/depthanything_{dataset}"
    print("Depth Anything V2 + Our Planarity → Plane Segmentation")
    print("=" * 60)
    print(f"MoGe ckpt:  {args.checkpoint}")
    print(f"DAv2 ckpt:  {args.dav2_checkpoint}")
    print(f"DAv2 enc:   {args.dav2_encoder}")
    print(f"Metric:     {args.metric_depth} (max_depth={args.max_depth})")
    print(f"Normals:    {'MoGe' if args.use_moge_normals else 'depth-derived'}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Output:     {out_label}")
    print(f"Resolution: {args.height}x{args.width}")
    if args.test_vis:
        print(f"Mode:       TEST VIS (5 frames, PNGs only)")
    elif args.max_frames:
        print(f"Max frames: {args.max_frames} per dataset")
    print("=" * 60)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        export_dataset(ds, moge_wrapper, dav2_model, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
