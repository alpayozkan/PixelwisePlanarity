#!/usr/bin/env python3
"""
Export plane segmentation using Metric3D v2 depth+normals + our MoGe planarity head.

Pipeline: Our planarity + Metric3D depth & normals → plan2seg → plane labels

Supports 4 datasets: scannetpp, hypersim, vkitti2, synthia (test splits).
Saves per-scene H5 files with depth, normals, planarity, mask, planes, gt_planes, intrinsics.

Usage:
    python planamono/external/export_metric3d.py \
        --dataset scannetpp \
        --checkpoint /path/to/moge_planarity.pt

    # Quick test with 5 frames:
    python planamono/external/export_metric3d.py \
        --dataset scannetpp --max_frames 5 \
        --checkpoint /path/to/moge_planarity.pt
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
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative


# ---------------------------------------------------------------------------
# Helpers (shared with export_depthanything.py)
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


def run_moge_planarity(model, rgb_uint8, device, out_h=480, out_w=640):
    """Run our MoGe planarity model and return probability map at (out_h, out_w)."""
    tensor = preprocess_for_moge(rgb_uint8, device)
    with torch.no_grad():
        output = model.model.forward(tensor.unsqueeze(0), num_tokens=1600)
    planarity = output['planarity'][0]  # (476, 644)
    planarity = torchF.interpolate(
        planarity[None, None], (out_h, out_w),
        mode='bilinear', align_corners=False
    )[0, 0]
    return planarity.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def random_label_colormap(num_labels):
    """Generate a random colormap for segment labels."""
    rng = np.random.RandomState(42)
    colors = rng.randint(50, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = 0  # background = black
    return colors


def save_vis_png(rgb, depth, normals, planarity, labels, gt_planes,
                 out_path, scene_label, frame_id):
    """Save a single-frame visualization as a 2x3 grid PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{scene_label} / {frame_id}", fontsize=14)

    # 1. Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    # 2. Depth
    valid = depth > 0
    vmin = depth[valid].min() if valid.any() else 0
    vmax = depth[valid].max() if valid.any() else 1
    axes[0, 1].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Depth")
    axes[0, 1].axis("off")

    # 3. Normals (map [-1,1] -> [0,1] for display)
    normals_vis = (normals * 0.5 + 0.5).clip(0, 1)
    axes[0, 2].imshow(normals_vis)
    axes[0, 2].set_title("Normals")
    axes[0, 2].axis("off")

    # 4. Planarity probability
    axes[1, 0].imshow(planarity, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Planarity")
    axes[1, 0].axis("off")

    # 5. Plane segmentation (colored labels)
    num_labels = int(labels.max()) + 1
    cmap = random_label_colormap(num_labels)
    seg_vis = cmap[labels.astype(np.int32)]
    axes[1, 1].imshow(seg_vis)
    axes[1, 1].set_title(f"Segmentation ({num_labels - 1} planes)")
    axes[1, 1].axis("off")

    # 6. Segmentation overlay on RGB
    overlay = rgb.copy().astype(np.float32)
    mask = labels > 0
    overlay[mask] = overlay[mask] * 0.4 + seg_vis[mask].astype(np.float32) * 0.6
    axes[1, 2].imshow(overlay.astype(np.uint8))
    axes[1, 2].set_title("Overlay")
    axes[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scene iterators: yield (scene_label, h5_rel, frame_ids, rgbs, gt_planes, Ks)
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
            gt_planes_all = hf["planes"][:]
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
# Metric3D model loading and inference
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Metric3D canonical input size for ViT models
METRIC3D_INPUT_H = 616
METRIC3D_INPUT_W = 1064


def load_metric3d_model(args):
    """Load Metric3D v2 model via torch.hub."""
    metric3d_repo = os.path.expanduser(args.metric3d_repo)
    model = torch.hub.load(metric3d_repo, args.metric3d_model,
                           pretrain=True, source='local')
    model = model.to(args.device).eval()
    return model


def metric3d_infer(model, rgb_uint8, K, device, out_h=480, out_w=640):
    """
    Run Metric3D inference on a single frame.

    Follows the official Metric3D preprocessing:
    1. Resize keeping aspect ratio to fit (616, 1064)
    2. Scale intrinsics accordingly
    3. Pad to exact (616, 1064) with ImageNet mean
    4. Normalize with ImageNet mean/std
    5. Forward pass
    6. Un-pad, resize to output
    7. De-canonicalize depth: depth *= actual_fx / 1000.0

    Args:
        model: Metric3D model
        rgb_uint8: (H, W, 3) uint8 RGB image
        K: (3, 3) intrinsic matrix at the image resolution
        device: torch device
        out_h, out_w: output resolution

    Returns:
        depth: (out_h, out_w) float32 metric depth
        normals: (out_h, out_w, 3) float32 surface normals
    """
    h, w = rgb_uint8.shape[:2]
    fx = K[0, 0]

    # 1. Resize keeping aspect ratio
    scale = min(METRIC3D_INPUT_H / h, METRIC3D_INPUT_W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    rgb_resized = cv2.resize(rgb_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2. Scale intrinsics
    K_scaled = K.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale

    # 3. Pad to (616, 1064) with ImageNet mean
    pad_h = METRIC3D_INPUT_H - new_h
    pad_w = METRIC3D_INPUT_W - new_w
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left

    mean_pixel = (np.array(IMAGENET_MEAN) * 255).astype(np.uint8)
    padded = np.full((METRIC3D_INPUT_H, METRIC3D_INPUT_W, 3), mean_pixel, dtype=np.uint8)
    padded[pad_h_top:pad_h_top + new_h, pad_w_left:pad_w_left + new_w] = rgb_resized

    # 4. Normalize with ImageNet mean/std
    img_tensor = torch.tensor(padded / 255.0, dtype=torch.float32, device=device)
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 616, 1064)

    # 5. Forward pass
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': img_tensor})

    # 6. Un-pad
    pred_depth = pred_depth[0, 0]  # (616, 1064)
    pred_depth = pred_depth[pad_h_top:pad_h_top + new_h, pad_w_left:pad_w_left + new_w]

    # Resize to output resolution
    pred_depth = torchF.interpolate(
        pred_depth[None, None], (out_h, out_w),
        mode='bilinear', align_corners=False
    )[0, 0]

    # 7. De-canonicalize depth
    canonical_focal = 1000.0
    pred_depth = pred_depth * (fx / canonical_focal)
    depth = pred_depth.cpu().numpy().astype(np.float32)

    # Extract normals from model output (ViT models output normals)
    normals = None
    if 'prediction_normal' in output_dict:
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]  # (1, 3, 616, 1064)
        pred_normal = pred_normal[0]  # (3, 616, 1064)
        pred_normal = pred_normal[:, pad_h_top:pad_h_top + new_h, pad_w_left:pad_w_left + new_w]
        pred_normal = torchF.interpolate(
            pred_normal[None], (out_h, out_w),
            mode='bilinear', align_corners=False
        )[0]  # (3, out_h, out_w)
        # Normalize
        norm_mag = torch.norm(pred_normal, dim=0, keepdim=True).clamp(min=1e-8)
        pred_normal = pred_normal / norm_mag
        normals = pred_normal.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, 3)

    if normals is None:
        # Fallback: compute normals from depth
        from planamono.shared.utils.depth_normal import depth_to_normal_remi
        K_out = scale_K(K, w, h, out_w, out_h)
        normals = depth_to_normal_remi(
            depth, K_out[0, 0], K_out[1, 1], K_out[0, 2], K_out[1, 2])

    return depth, normals


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Export plane segmentation: Metric3D depth+normals + our MoGe planarity")

    # Our planarity model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to MoGe planarity checkpoint (.pt)")

    # Metric3D model
    p.add_argument("--metric3d_model", type=str, default="metric3d_vit_large",
                   choices=["metric3d_vit_small", "metric3d_vit_large", "metric3d_vit_giant2"])
    p.add_argument("--metric3d_repo", type=str, default="~/Metric3D",
                   help="Path to Metric3D repo")

    # Dataset / output
    p.add_argument("--dataset", type=str, required=True,
                   choices=["scannetpp", "hypersim", "vkitti2", "synthia", "all"])
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output root (default: /cluster/scratch/ayavuz/dataset/metric3d_{dataset})")
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

def export_dataset(dataset_name, moge_model, metric3d_model, args):
    if args.output_dir:
        ds_out = os.path.join(args.output_dir, dataset_name)
    else:
        ds_out = f"/cluster/scratch/ayavuz/dataset/metric3d_{dataset_name}"
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
            # 1. Our MoGe planarity
            planarity = run_moge_planarity(moge_model, rgb, args.device,
                                           args.height, args.width)

            # 2. Metric3D depth + normals
            K = Ks[i]
            depth, normals = metric3d_infer(
                metric3d_model, rgb, K, args.device, args.height, args.width)

            # 3. Validity mask
            valid_mask = (depth > 0).astype(np.float32)

            # 4. Plan2seg
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
                safe_scene = scene_label.replace("/", "_")
                vis_path = os.path.join(ds_out, "test_vis",
                                        f"{safe_scene}_{frame_ids[i]}.png")
                save_vis_png(rgb, depth, normals, planarity, labels,
                             gt_planes_list[i], vis_path, scene_label, frame_ids[i])
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
    metric3d_model = load_metric3d_model(args)

    out_label = args.output_dir or "/cluster/scratch/ayavuz/dataset/metric3d_{dataset}"
    print("Metric3D v2 + Our Planarity -> Plane Segmentation")
    print("=" * 60)
    print(f"MoGe ckpt:     {args.checkpoint}")
    print(f"Metric3D:      {args.metric3d_model}")
    print(f"Metric3D repo: {args.metric3d_repo}")
    print(f"Datasets:      {', '.join(datasets)}")
    print(f"Output:        {out_label}")
    print(f"Resolution:    {args.height}x{args.width}")
    if args.test_vis:
        print(f"Mode:          TEST VIS (5 frames, PNGs only)")
    elif args.max_frames:
        print(f"Max frames:    {args.max_frames} per dataset")
    print("=" * 60)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        export_dataset(ds, moge_wrapper, metric3d_model, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
