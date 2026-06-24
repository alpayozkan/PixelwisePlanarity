#!/usr/bin/env python3
"""
Export MoGe inference results on ScanNet++ validation scenes to H5 files.

For each val scene, runs MoGe forward on all frames and saves:
  depth, normals, planarity, mask, gt_planes, intrinsics
into a single H5 file at 480x640 resolution.

Intrinsics are recovered from MoGe's predicted point map via recover_focal_shift(),
stored as (N, 3, 3) pixel-coordinate K matrices at the output resolution.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
from tqdm import tqdm
from pathlib import Path

# Add repo root to path so `planamono` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.moge.moge.utils.geometry_torch import recover_focal_shift
import utils3d


def parse_args():
    parser = argparse.ArgumentParser(description="Export MoGe inference on ScanNet++ val to H5")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained MoGe .pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save H5 files")
    parser.add_argument("--rgb_root", type=str,
                        default="/cluster/project/cvg/Shared_datasets/scannet++/data",
                        help="ScanNet++ data root")
    parser.add_argument("--gt_root", type=str,
                        default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
                        help="Rendered H5 root")
    parser.add_argument("--split_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "splits" / "scannetpp"),
                        help="Split file directory")
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
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Which split partition to export")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process scenes even if inference.h5 already exists "
                             "(default: skip existing, i.e. resume)")
    return parser.parse_args()


SPLIT_FILES = {
    "train": "nvs_sem_train_with_planes_fixed.txt",
    "val":   "nvs_sem_val_with_planes_fixed.txt",
    "test":  "nvs_sem_test_with_planes.txt",
}


def load_split_scenes(split_dir, split):
    """Read scene IDs from the split file for the given split."""
    split_file = os.path.join(split_dir, SPLIT_FILES[split])
    with open(split_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return scenes


def preprocess_image(image_bgr, device):
    """Resize to 476x644, normalize, return tensor (1, 3, 476, 644)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (644, 476))
    tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


def resize_predictions(depth, normals, planarity, mask, out_h, out_w):
    """Resize model outputs from (476, 644) to (out_h, out_w)."""
    # depth: (H, W) -> (1, 1, H, W) -> bilinear -> (out_h, out_w)
    depth_out = F.interpolate(depth[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    # normals: (H, W, 3) -> (1, 3, H, W) -> bilinear -> (out_h, out_w, 3)
    normals_out = F.interpolate(normals.permute(2, 0, 1)[None], (out_h, out_w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    # planarity: (H, W) -> bilinear
    planarity_out = F.interpolate(planarity[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    # mask: (H, W) -> bilinear
    mask_out = F.interpolate(mask[None, None], (out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    return depth_out, normals_out, planarity_out, mask_out


def resize_gt_planes(planes, out_h, out_w):
    """Resize GT plane labels with nearest interpolation to preserve IDs."""
    # planes: numpy (H, W) uint16
    planes_resized = cv2.resize(planes, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return planes_resized.astype(np.uint16)


def resize_gt_sem(sem, out_h, out_w):
    """Resize GT semantic labels with nearest interpolation."""
    sem_resized = cv2.resize(sem.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return sem_resized.astype(np.int32)


def resize_gt_depth_mm(depth_mm, out_h, out_w):
    """Resize GT depth (uint16 mm) to (out_h, out_w) and convert to float meters.

    Bilinear interpolation; 0 (invalid) pixels are preserved by zeroing any
    output pixel whose nearest-neighbor source is 0.
    """
    valid_mask = (depth_mm > 0).astype(np.float32)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_resized = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    valid_resized = cv2.resize(valid_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    depth_resized[valid_resized < 0.5] = 0.0
    return depth_resized.astype(np.float32)


def scale_intrinsics(K_native, native_w, native_h, out_w, out_h):
    """Scale 3x3 intrinsics from (native_w, native_h) to (out_w, out_h)."""
    sx = out_w / native_w
    sy = out_h / native_h
    K = K_native.astype(np.float32).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def process_scene(model, scene_id, rgb_root, gt_root, output_dir,
                  num_tokens, batch_size, out_h, out_w, device):
    """Process all frames of a single scene and write H5."""
    plane_h5_path = os.path.join(gt_root, scene_id, "rendered.h5")
    sem_h5_path = os.path.join(gt_root, scene_id, "rendered_sem.h5")
    depth_h5_path = os.path.join(gt_root, scene_id, "rendered_depth.h5")
    pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")

    for path, label in [(plane_h5_path, "rendered.h5"),
                        (sem_h5_path, "rendered_sem.h5"),
                        (depth_h5_path, "rendered_depth.h5"),
                        (pose_file, "pose_intrinsic_imu.json")]:
        if not os.path.exists(path):
            print(f"  Skipping {scene_id}: {label} not found at {path}")
            return False

    # Read frame_ids + GT planes / sem / depth from rendered H5s
    with h5py.File(plane_h5_path, "r") as gt_f:
        frame_ids_raw = gt_f["frame_ids"][:]
        if isinstance(frame_ids_raw[0], bytes):
            frame_ids = [fid.decode("utf-8") for fid in frame_ids_raw]
        else:
            frame_ids = list(frame_ids_raw)
        gt_planes_all = gt_f["planes"][:]  # (N, H_gt, W_gt) uint16

    with h5py.File(sem_h5_path, "r") as sem_f:
        gt_sem_all = sem_f["sem"][:]  # (N, H_gt, W_gt)

    with h5py.File(depth_h5_path, "r") as depth_f:
        gt_depth_all = depth_f["depth"][:]  # (N, H_gt, W_gt) uint16 millimeters

    # Per-frame GT intrinsics (native iPhone resolution) and poses
    with open(pose_file, "r") as f:
        pose_data = json.load(f)

    # ScanNet++ iPhone capture native resolution (used to scale K to output res).
    # Same constants used by gt_creation/scannetpp/render_scene_*.py.
    NATIVE_W, NATIVE_H = 1920, 1440

    num_frames = len(frame_ids)
    print(f"  {scene_id}: {num_frames} frames")

    # Preallocate output arrays
    rgbs = np.zeros((num_frames, out_h, out_w, 3), dtype=np.uint8)         # RGB at output res
    intrinsics_all = np.zeros((num_frames, 3, 3), dtype=np.float32)        # MoGe-recovered
    depths = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    normals = np.zeros((num_frames, out_h, out_w, 3), dtype=np.float32)
    planarities = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    masks = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    gt_planes_out = np.zeros((num_frames, out_h, out_w), dtype=np.uint16)
    gt_sem_out = np.zeros((num_frames, out_h, out_w), dtype=np.int32)
    gt_depth_out = np.zeros((num_frames, out_h, out_w), dtype=np.float32)
    gt_intrinsics_all = np.zeros((num_frames, 3, 3), dtype=np.float32)
    gt_pose_all = np.zeros((num_frames, 4, 4), dtype=np.float32)
    has_gt_pose = np.zeros(num_frames, dtype=bool)

    # Process frames in batches
    pbar = tqdm(total=num_frames, desc=f"  {scene_id}", leave=False)
    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_tensors = []
        batch_indices = []

        for idx in range(batch_start, batch_end):
            frame_id = frame_ids[idx]
            rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"{frame_id}.jpg")
            if not os.path.exists(rgb_path):
                print(f"    Warning: RGB not found: {rgb_path}")
                continue
            image_bgr = cv2.imread(rgb_path)
            if image_bgr is None:
                print(f"    Warning: Failed to read: {rgb_path}")
                continue
            tensor = preprocess_image(image_bgr, device)
            batch_tensors.append(tensor)
            batch_indices.append(idx)
            # Store RGB at output resolution (convert BGR -> RGB)
            rgb_out = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            rgbs[idx] = cv2.cvtColor(rgb_out, cv2.COLOR_BGR2RGB)

        if not batch_tensors:
            pbar.update(batch_end - batch_start)
            continue

        batch_input = torch.stack(batch_tensors, dim=0)  # (B, 3, 476, 644)

        with torch.no_grad():
            output = model.model.forward(batch_input, num_tokens=num_tokens)
            # output['points']: (B, 476, 644, 3)
            # output['normal']: (B, 476, 644, 3)
            # output['mask']:   (B, 476, 644)
            # output['planarity']: (B, 476, 644)

            # Recover intrinsics from MoGe's predicted point map
            points_batch = output['points'].float()
            mask_binary = output['mask'] > 0.5
            focal, shift = recover_focal_shift(points_batch, mask_binary)
            H_model, W_model = points_batch.shape[1], points_batch.shape[2]
            aspect_ratio = W_model / H_model
            fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            K_batch = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)  # (B, 3, 3) normalized

        for i, idx in enumerate(batch_indices):
            depth = output['points'][i, :, :, 2]     # (476, 644) z-depth
            normal = output['normal'][i]              # (476, 644, 3)
            planarity = output['planarity'][i]        # (476, 644)
            mask = output['mask'][i]                  # (476, 644)

            depth_r, normal_r, planarity_r, mask_r = resize_predictions(
                depth, normal, planarity, mask, out_h, out_w
            )

            depths[idx] = depth_r.cpu().numpy()
            normals[idx] = normal_r.cpu().numpy()
            planarities[idx] = planarity_r.cpu().numpy()
            masks[idx] = mask_r.cpu().numpy()

            # Store MoGe-predicted intrinsics scaled to output resolution
            K = K_batch[i].cpu().numpy()  # (3, 3) normalized [0, 1]
            K[0, :] *= out_w  # scale x-row to pixel coords
            K[1, :] *= out_h  # scale y-row to pixel coords
            intrinsics_all[idx] = K

            # Resize GT planes / sem / depth (per-frame index in rendered H5s)
            gt_planes_out[idx] = resize_gt_planes(gt_planes_all[idx], out_h, out_w)
            gt_sem_out[idx] = resize_gt_sem(gt_sem_all[idx], out_h, out_w)
            gt_depth_out[idx] = resize_gt_depth_mm(gt_depth_all[idx], out_h, out_w)

            # GT intrinsics (scaled from iPhone native to output) and pose
            fid = frame_ids[idx]
            entry = pose_data.get(fid)
            if entry is not None:
                K_native = np.array(entry["intrinsic"], dtype=np.float32)
                gt_intrinsics_all[idx] = scale_intrinsics(
                    K_native, NATIVE_W, NATIVE_H, out_w, out_h
                )
                c2w = np.array(entry["aligned_pose"], dtype=np.float32)
                gt_pose_all[idx] = c2w
                has_gt_pose[idx] = True
            else:
                print(f"    Warning: missing pose entry for {fid} in {scene_id}")

        pbar.update(len(batch_indices))
    pbar.close()

    # Write H5
    scene_out_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scene_out_dir, exist_ok=True)
    out_h5_path = os.path.join(scene_out_dir, "inference.h5")

    with h5py.File(out_h5_path, "w") as f:
        # Store frame_ids as variable-length strings
        dt = h5py.string_dtype()
        f.create_dataset("frame_ids", data=frame_ids, dtype=dt)
        f.create_dataset("rgb", data=rgbs, dtype=np.uint8, compression="gzip", compression_opts=4)
        f.create_dataset("depth", data=depths, dtype=np.float32)
        f.create_dataset("normals", data=normals, dtype=np.float32)
        f.create_dataset("planarity", data=planarities, dtype=np.float32)
        f.create_dataset("mask", data=masks, dtype=np.float32)
        f.create_dataset("gt_planes", data=gt_planes_out, dtype=np.uint16)
        f.create_dataset("gt_sem", data=gt_sem_out, dtype=np.int32)
        f.create_dataset("gt_depth", data=gt_depth_out, dtype=np.float32)
        f.create_dataset("gt_intrinsics", data=gt_intrinsics_all, dtype=np.float32)
        f.create_dataset("gt_pose", data=gt_pose_all, dtype=np.float32)
        f.create_dataset("has_gt_pose", data=has_gt_pose, dtype=np.bool_)
        f.create_dataset("intrinsics", data=intrinsics_all, dtype=np.float32)

    print(f"  Saved: {out_h5_path}")
    return True


def main():
    args = parse_args()

    print(f"MoGe ScanNet++ Export ({args.split})")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"RGB root:   {args.rgb_root}")
    print(f"GT root:    {args.gt_root}")
    print(f"Split:      {args.split}")
    print(f"Resolution: {args.output_height}x{args.output_width}")
    print(f"Num tokens: {args.num_tokens}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)

    # Load model
    model = MoGePlanarityInference(args.checkpoint, device=args.device)

    # Load scenes for the requested split
    scenes = load_split_scenes(args.split_dir, args.split)
    print(f"Found {len(scenes)} {args.split} scenes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each scene
    success_count = 0
    for i, scene_id in enumerate(scenes):
        out_h5_path = os.path.join(args.output_dir, scene_id, "inference.h5")
        if not args.overwrite and os.path.exists(out_h5_path):
            print(f"\n[{i+1}/{len(scenes)}] Skipping {scene_id} (already exists: {out_h5_path})")
            success_count += 1
            continue
        print(f"\n[{i+1}/{len(scenes)}] Processing {scene_id}")
        ok = process_scene(
            model, scene_id, args.rgb_root, args.gt_root, args.output_dir,
            args.num_tokens, args.batch_size, args.output_height, args.output_width,
            args.device
        )
        if ok:
            success_count += 1

    print(f"\nDone! Exported {success_count}/{len(scenes)} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
