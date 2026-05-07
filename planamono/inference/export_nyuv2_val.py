#!/usr/bin/env python3
"""
Export MoGe inference on the NYUv2 plane val split to a single H5 file.

Reads the JSON manifest (`nyuv2_plane_len654_test.json`) and the per-frame
npz files (each containing `raw_image (480,640,3)`, `raw_depth (192,256)`,
`high_res_raw_depth (480,640)`, `segmentation (192,256)`, `plane`, `intrinsic`).

Output H5 schema (480x640):

    image_ids       (N,)               string    "0_d2", "1_d2", ...
    depth           (N, H, W)          float32   MoGe Z-depth (points[..., 2])
    normals         (N, H, W, 3)       float32
    planarity       (N, H, W)          float32
    mask            (N, H, W)          float32
    intrinsics      (N, 3, 3)          float32   MoGe-recovered K, scaled to (H, W)
    gt_planes       (N, H, W)          uint16    npz "segmentation" (non-plane = 20)
    gt_depth        (N, H, W)          float32   npz "high_res_raw_depth"
    gt_intrinsics   (N, 3, 3)          float32   npz "intrinsic", scaled from 256x192 to (H, W)
    gt_planeparams  (N, P_max, 3)      float32   npz "plane" (per-plane n*offset), zero-padded
    num_planes      (N,)               int32

Usage:
    python planamono/inference/export_nyuv2_val.py \
        --checkpoint /path/to/moge_planarity.pt \
        --output_dir /cluster/scratch/ayavuz/inference/<run>/nyuv2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.moge.moge.utils.geometry_torch import recover_focal_shift
import utils3d


DEFAULT_NPZ_ROOT = "/cluster/scratch/ayavuz/dataset/nyuv2_plane"
DEFAULT_JSON = os.path.join(DEFAULT_NPZ_ROOT, "nyuv2_plane_len654_test.json")

# npz GT layouts
GT_LOW_W,  GT_LOW_H  = 256, 192
GT_HIGH_W, GT_HIGH_H = 640, 480

# Max plane count to allocate for gt_planeparams (nyuv2 appears to cap ~20)
P_MAX = 21


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--json_file", default=DEFAULT_JSON)
    ap.add_argument("--npz_root", default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--num_tokens", type=int, default=1600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output_height", type=int, default=480)
    ap.add_argument("--output_width", type=int, default=640)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def load_manifest(json_file, npz_root):
    with open(json_file) as f:
        info = json.load(f)
    out = []
    for ann in info["annotations"]:
        npz_path = os.path.join(npz_root, os.path.basename(ann["npz_file_name"]))
        out.append({
            "image_id": str(ann["image_id"]),
            "npz_path": npz_path,
        })
    return out


def preprocess_image(image_rgb, device):
    """Resize to 476x644, normalize, return tensor (3, 476, 644) on device."""
    image_resized = cv2.resize(image_rgb, (644, 476))
    tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


def resize_predictions(depth, normals, planarity, mask, out_h, out_w):
    depth_out = F.interpolate(depth[None, None], (out_h, out_w),
                              mode='bilinear', align_corners=False)[0, 0]
    normals_out = F.interpolate(normals.permute(2, 0, 1)[None], (out_h, out_w),
                                mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    planarity_out = F.interpolate(planarity[None, None], (out_h, out_w),
                                  mode='bilinear', align_corners=False)[0, 0]
    mask_out = F.interpolate(mask[None, None], (out_h, out_w),
                             mode='bilinear', align_corners=False)[0, 0]
    return depth_out, normals_out, planarity_out, mask_out


def scale_K(K, src_w, src_h, dst_w, dst_h):
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.astype(np.float32).copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return K2


def resize_gt_planes(planes, out_h, out_w):
    return cv2.resize(planes.astype(np.int32), (out_w, out_h),
                      interpolation=cv2.INTER_NEAREST).astype(np.uint16)


def resize_gt_depth(depth_m, out_h, out_w):
    valid = (depth_m > 0).astype(np.float32)
    d = cv2.resize(depth_m.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    d[valid_r < 0.5] = 0.0
    return d.astype(np.float32)


def main():
    args = parse_args()
    out_h, out_w = args.output_height, args.output_width

    print("MoGe NYUv2 Export")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output_dir}")
    print(f"JSON:       {args.json_file}")
    print(f"NPZ root:   {args.npz_root}")
    print(f"Resolution: {out_h}x{out_w}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    model = MoGePlanarityInference(args.checkpoint, device=args.device)
    entries = load_manifest(args.json_file, args.npz_root)
    print(f"Loaded {len(entries)} frames")

    N = len(entries)
    image_ids = []
    depths = np.zeros((N, out_h, out_w), dtype=np.float32)
    normals_arr = np.zeros((N, out_h, out_w, 3), dtype=np.float32)
    planarities = np.zeros((N, out_h, out_w), dtype=np.float32)
    masks = np.zeros((N, out_h, out_w), dtype=np.float32)
    intrinsics_all = np.zeros((N, 3, 3), dtype=np.float32)
    gt_planes_out = np.zeros((N, out_h, out_w), dtype=np.uint16)
    gt_depth_out = np.zeros((N, out_h, out_w), dtype=np.float32)
    gt_intrinsics_out = np.zeros((N, 3, 3), dtype=np.float32)
    gt_planeparams_out = np.zeros((N, P_MAX, 3), dtype=np.float32)
    num_planes_out = np.zeros((N,), dtype=np.int32)

    # Cache npz arrays per batch so we only open each file once
    pbar = tqdm(total=N, desc="NYUv2")
    for batch_start in range(0, N, args.batch_size):
        batch_end = min(batch_start + args.batch_size, N)
        batch_tensors = []
        batch_indices = []
        batch_npz = []

        for idx in range(batch_start, batch_end):
            entry = entries[idx]
            try:
                data = np.load(entry["npz_path"])
            except Exception as e:
                print(f"  [WARN] failed to load {entry['npz_path']}: {e}")
                continue
            rgb = np.asarray(data["raw_image"])  # (480, 640, 3) uint8 BGR per image_format=CV2_BGR
            # JSON says image_format=CV2_BGR; convert to RGB for MoGe
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            tensor = preprocess_image(rgb, args.device)
            batch_tensors.append(tensor)
            batch_indices.append(idx)
            batch_npz.append(data)

        if not batch_tensors:
            pbar.update(batch_end - batch_start)
            continue

        batch_input = torch.stack(batch_tensors, dim=0)  # (B, 3, 476, 644)

        with torch.no_grad():
            output = model.model.forward(batch_input, num_tokens=args.num_tokens)
            points_batch = output["points"].float()
            mask_binary = output["mask"] > 0.5
            focal, shift = recover_focal_shift(points_batch, mask_binary)
            H_model, W_model = points_batch.shape[1], points_batch.shape[2]
            aspect_ratio = W_model / H_model
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            K_batch = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

        for j, idx in enumerate(batch_indices):
            depth = output["points"][j, :, :, 2]
            normal = output["normal"][j]
            planarity = output["planarity"][j]
            mask_p = output["mask"][j]

            depth_r, normal_r, planarity_r, mask_r = resize_predictions(
                depth, normal, planarity, mask_p, out_h, out_w
            )

            entry = entries[idx]
            image_ids.append(entry["image_id"])
            depths[idx] = depth_r.cpu().numpy()
            normals_arr[idx] = normal_r.cpu().numpy()
            planarities[idx] = planarity_r.cpu().numpy()
            masks[idx] = mask_r.cpu().numpy()

            K = K_batch[j].cpu().numpy().copy()
            K[0, :] *= out_w
            K[1, :] *= out_h
            intrinsics_all[idx] = K

            # GT from npz
            data = batch_npz[j]
            gt_seg_src = np.asarray(data["segmentation"])
            gt_planes_out[idx] = resize_gt_planes(gt_seg_src, out_h, out_w)

            if "high_res_raw_depth" in data.files:
                gt_d_src = np.asarray(data["high_res_raw_depth"])
            else:
                gt_d_src = np.asarray(data["raw_depth"])
            gt_depth_out[idx] = resize_gt_depth(gt_d_src, out_h, out_w)

            K_npz = np.asarray(data["intrinsic"], dtype=np.float32)
            gt_intrinsics_out[idx] = scale_K(K_npz, GT_LOW_W, GT_LOW_H, out_w, out_h)

            params = np.asarray(data["plane"], dtype=np.float32)  # (Np, 3)
            num_planes_out[idx] = int(data["num_planes"])
            cap = min(params.shape[0], P_MAX)
            gt_planeparams_out[idx, :cap] = params[:cap]

        pbar.update(batch_end - batch_start)
    pbar.close()

    out_h5 = os.path.join(args.output_dir, "inference.h5")
    with h5py.File(out_h5, "w") as f:
        dt = h5py.string_dtype()
        f.create_dataset("image_ids", data=image_ids, dtype=dt)
        f.create_dataset("depth", data=depths, dtype=np.float32)
        f.create_dataset("normals", data=normals_arr, dtype=np.float32)
        f.create_dataset("planarity", data=planarities, dtype=np.float32)
        f.create_dataset("mask", data=masks, dtype=np.float32)
        f.create_dataset("intrinsics", data=intrinsics_all, dtype=np.float32)
        f.create_dataset("gt_planes", data=gt_planes_out, dtype=np.uint16)
        f.create_dataset("gt_depth", data=gt_depth_out, dtype=np.float32)
        f.create_dataset("gt_intrinsics", data=gt_intrinsics_out, dtype=np.float32)
        f.create_dataset("gt_planeparams", data=gt_planeparams_out, dtype=np.float32)
        f.create_dataset("num_planes", data=num_planes_out, dtype=np.int32)
    print(f"\nSaved: {out_h5}  ({len(image_ids)} frames)")


if __name__ == "__main__":
    main()
