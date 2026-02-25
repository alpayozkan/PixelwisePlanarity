#!/usr/bin/env python3
"""
Export MoGe inference results on Hypersim validation frames to H5 files.

For each (scene, camera) group in the val split, runs MoGe forward on all
frames and saves:
  depth, normals, planarity, mask, gt_planes, intrinsics
into a single H5 file at 480x640 resolution.

Intrinsics come from Hypersim's ground-truth M_cam_from_uv matrices,
converted to standard CV-convention K and scaled to the output resolution.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add repo root to path so `planamono` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference


# ---------------------------------------------------------------------------
# Hypersim-specific helpers
# ---------------------------------------------------------------------------

def compute_K_from_M_cam_from_uv(M_cam_from_uv, H, W):
    """
    Compute a 3x3 CV-convention intrinsic matrix K from Hypersim's M_cam_from_uv.

    M_cam_from_uv maps NDC coordinates to camera-space ray directions (OpenGL convention):
        ndc_x = 2*(px+0.5)/W - 1
        ndc_y = 1 - 2*(py+0.5)/H   (OpenGL: Y up)
        dir_cam = M_cam_from_uv @ [ndc_x, ndc_y, 1]

    We compose pixel_to_ndc with M_cam_from_uv to get K_inv (OpenGL), then
    convert to CV convention (Y down, Z forward) by negating rows 1 and 2.
    """
    M = np.array(M_cam_from_uv, dtype=np.float64)

    pixel_to_ndc = np.array([
        [ 2.0 / W,      0.0,  (1.0 - W) / W],
        [     0.0, -2.0 / H,  (H - 1.0) / H],
        [     0.0,      0.0,            1.0 ],
    ], dtype=np.float64)

    K_inv_opengl = M @ pixel_to_ndc

    K_inv_cv = K_inv_opengl.copy()
    K_inv_cv[1, :] *= -1.0
    K_inv_cv[2, :] *= -1.0

    K = np.linalg.inv(K_inv_cv)
    return K.astype(np.float32)


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
    img = np.clip(img, 0, None)
    img = img ** (1.0 / gamma)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Shared helpers (adapted from export_scannetpp_val.py)
# ---------------------------------------------------------------------------

def preprocess_image(image_rgb, device):
    """Resize to 476x644, normalize, return tensor (1, 3, 476, 644)."""
    image_resized = cv2.resize(image_rgb, (644, 476))
    tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


def resize_predictions(depth, normals, planarity, mask, out_h, out_w):
    """Resize model outputs from (476, 644) to (out_h, out_w)."""
    depth_out = F.interpolate(depth[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    normals_out = F.interpolate(normals.permute(2, 0, 1)[None], (out_h, out_w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    planarity_out = F.interpolate(planarity[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    mask_out = F.interpolate(mask[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    return depth_out, normals_out, planarity_out, mask_out


def resize_gt_planes(planes, out_h, out_w):
    """Resize GT plane labels with nearest interpolation to preserve IDs."""
    planes_resized = cv2.resize(planes, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return planes_resized.astype(np.uint16)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Export MoGe inference on Hypersim val to H5")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained MoGe .pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save H5 files")
    parser.add_argument("--data_root", type=str,
                        default="/cluster/project/cvg/Shared_datasets/Hypersim",
                        help="Hypersim data root (contains scene dirs and metadata CSV)")
    parser.add_argument("--gt_root", type=str,
                        default="/cluster/scratch/aoezkan/planeseg/dataset/Hypersim",
                        help="Rendered GT H5 root")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Camera parameters CSV (default: {data_root}/metadata_camera_parameters.csv)")
    parser.add_argument("--split_csv", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "splits" / "hypersim" /
                                    "metadata_images_split_with_planes_filtered.csv"),
                        help="Split CSV with (scene_name, camera_name, frame_id, split_partition_name)")
    parser.add_argument("--num_tokens", type=int, default=1600,
                        help="DINOv2 tokens for inference")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size")
    parser.add_argument("--output_height", type=int, default=480,
                        help="Output height")
    parser.add_argument("--output_width", type=int, default=640,
                        help="Output width")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Intrinsics loading
# ---------------------------------------------------------------------------

def load_intrinsics_map(metadata_csv):
    """Load per-scene M_cam_from_uv matrices from the metadata CSV.

    Returns dict: scene_name -> 3x3 numpy array.
    """
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


# ---------------------------------------------------------------------------
# Per-(scene, camera) processing
# ---------------------------------------------------------------------------

def process_scene_camera(model, scene_name, camera_name, frame_ids,
                         data_root, gt_root, K_output, output_dir,
                         num_tokens, batch_size, out_h, out_w, device):
    """Process all frames of a single (scene, camera) group and write H5."""
    gt_h5_path = os.path.join(gt_root, scene_name, f"rendered_planes_{camera_name}.h5")
    if not os.path.exists(gt_h5_path):
        print(f"  Skipping {scene_name}/{camera_name}: GT file not found at {gt_h5_path}")
        return False

    # Read GT planes and frame_ids from rendered H5
    with h5py.File(gt_h5_path, "r") as gt_f:
        gt_frame_ids_raw = gt_f["frame_ids"][:]
        if isinstance(gt_frame_ids_raw[0], bytes):
            gt_frame_ids = [fid.decode("utf-8") for fid in gt_frame_ids_raw]
        else:
            gt_frame_ids = [str(fid) for fid in gt_frame_ids_raw]
        gt_planes_all = gt_f["planes"][:]  # (N_gt, 768, 1024) uint16

    # Build index from frame_id string to position in GT H5
    gt_id_to_idx = {fid: i for i, fid in enumerate(gt_frame_ids)}

    num_frames = len(frame_ids)
    label = f"{scene_name}/{camera_name}"
    print(f"  {label}: {num_frames} frames")

    # Preallocate output arrays
    intrinsics_all = np.tile(K_output, (num_frames, 1, 1))  # same K for all frames
    depths = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    normals_arr = np.zeros((num_frames, out_h, out_w, 3), dtype=np.float32)
    planarities = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    masks = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    gt_planes_out = np.zeros((num_frames, out_h, out_w), dtype=np.uint16)

    # Frame IDs as strings for H5 storage (zero-padded)
    frame_id_strs = [f"{fid:04d}" for fid in frame_ids]

    pbar = tqdm(total=num_frames, desc=f"  {label}", leave=False)
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_tensors = []
        batch_indices = []

        for idx in range(batch_start, batch_end):
            fid = frame_ids[idx]
            rgb_path = os.path.join(
                data_root, scene_name, "images",
                f"scene_{camera_name}_final_hdf5",
                f"frame.{fid:04d}.color.hdf5"
            )
            if not os.path.exists(rgb_path):
                print(f"    Warning: RGB not found: {rgb_path}")
                continue

            image_rgb = load_hypersim_hdr_rgb(rgb_path)
            tensor = preprocess_image(image_rgb, device)
            batch_tensors.append(tensor)
            batch_indices.append(idx)

        if not batch_tensors:
            pbar.update(batch_end - batch_start)
            continue

        batch_input = torch.stack(batch_tensors, dim=0)  # (B, 3, 476, 644)

        with torch.no_grad():
            output = model.model.forward(batch_input, num_tokens=num_tokens)

        for i, idx in enumerate(batch_indices):
            depth = output['points'][i, :, :, 2]
            normal = output['normal'][i]
            planarity = output['planarity'][i]
            mask = output['mask'][i]

            depth_r, normal_r, planarity_r, mask_r = resize_predictions(
                depth, normal, planarity, mask, out_h, out_w
            )

            depths[idx] = depth_r.cpu().numpy()
            normals_arr[idx] = normal_r.cpu().numpy()
            planarities[idx] = planarity_r.cpu().numpy()
            masks[idx] = mask_r.cpu().numpy()

            # Resize GT planes (look up by frame_id string)
            fid_str = frame_id_strs[idx]
            gt_idx = gt_id_to_idx.get(fid_str)
            if gt_idx is not None:
                gt_planes_out[idx] = resize_gt_planes(gt_planes_all[gt_idx], out_h, out_w)
            else:
                print(f"    Warning: frame {fid_str} not found in GT H5 for {label}")

        pbar.update(len(batch_indices))
    pbar.close()

    # Write H5
    scene_out_dir = os.path.join(output_dir, scene_name, camera_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    out_h5_path = os.path.join(scene_out_dir, "inference.h5")

    with h5py.File(out_h5_path, "w") as f:
        dt = h5py.string_dtype()
        f.create_dataset("frame_ids", data=frame_id_strs, dtype=dt)
        f.create_dataset("depth", data=depths, dtype=np.float32)
        f.create_dataset("normals", data=normals_arr, dtype=np.float32)
        f.create_dataset("planarity", data=planarities, dtype=np.float32)
        f.create_dataset("mask", data=masks, dtype=np.float32)
        f.create_dataset("gt_planes", data=gt_planes_out, dtype=np.uint16)
        f.create_dataset("intrinsics", data=intrinsics_all, dtype=np.float32)

    print(f"  Saved: {out_h5_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.metadata_csv is None:
        args.metadata_csv = os.path.join(args.data_root, "metadata_camera_parameters.csv")

    print("MoGe Hypersim Validation Export")
    print("=" * 50)
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Data root:    {args.data_root}")
    print(f"GT root:      {args.gt_root}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Split CSV:    {args.split_csv}")
    print(f"Resolution:   {args.output_height}x{args.output_width}")
    print(f"Num tokens:   {args.num_tokens}")
    print(f"Batch size:   {args.batch_size}")
    print("=" * 50)

    # Load model
    model = MoGePlanarityInference(args.checkpoint, device=args.device)

    # Load val samples from split CSV
    df = pd.read_csv(args.split_csv)
    val_df = df[df['split_partition_name'] == 'val']
    print(f"Found {len(val_df)} validation frames")

    # Load per-scene intrinsics
    intrinsics_map = load_intrinsics_map(args.metadata_csv)

    # Native Hypersim resolution
    native_h, native_w = 768, 1024
    out_h, out_w = args.output_height, args.output_width

    os.makedirs(args.output_dir, exist_ok=True)

    # Group by (scene_name, camera_name)
    groups = val_df.groupby(['scene_name', 'camera_name'])
    num_groups = len(groups)
    print(f"Processing {num_groups} (scene, camera) groups")

    success_count = 0
    for i, ((scene_name, camera_name), group_df) in enumerate(groups):
        print(f"\n[{i+1}/{num_groups}] {scene_name}/{camera_name}")

        # Compute K at output resolution
        M = intrinsics_map.get(scene_name)
        if M is None:
            print(f"  Skipping: no intrinsics found for {scene_name}")
            continue
        K_native = compute_K_from_M_cam_from_uv(M, native_h, native_w)
        # Scale K from native (768x1024) to output (480x640)
        sx = out_w / native_w
        sy = out_h / native_h
        K_output = K_native.copy()
        K_output[0, :] *= sx
        K_output[1, :] *= sy

        frame_ids = sorted(group_df['frame_id'].tolist())

        ok = process_scene_camera(
            model, scene_name, camera_name, frame_ids,
            args.data_root, args.gt_root, K_output, args.output_dir,
            args.num_tokens, args.batch_size, out_h, out_w, args.device
        )
        if ok:
            success_count += 1

    print(f"\nDone! Exported {success_count}/{num_groups} (scene, camera) groups to {args.output_dir}")


if __name__ == "__main__":
    main()
