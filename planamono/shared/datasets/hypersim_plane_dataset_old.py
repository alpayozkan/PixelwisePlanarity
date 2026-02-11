"""
Hypersim dataset for plane segmentation evaluation.

Similar to ScanNetPPPlaneDataset but adapted for Hypersim data structure.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from natsort import natsorted
import json


class HypersimPlaneDataset(Dataset):
    """
    Hypersim dataset for plane segmentation evaluation.

    Similar to ScanNetPPPlaneDataset but adapted for Hypersim data structure:
    - Multiple cameras per scene (cam_00, cam_01, etc.)
    - HDF5 files for RGB and depth merged together
    - Plane labels in separate HDF5 per camera
    - Intrinsics in separate directory

    Expected directory structure:
        rgb_depth_root/
            <scene_id>/
                <cam_name>_merged.h5           # Contains 'rgb' and 'depth' datasets
        plane_label_root/
            <scene_id>/
                rendered_planes_<cam_name>.h5  # Contains 'planes' and 'frame_ids'
        intrinsics_root/
            <scene_id>/
                <cam_name>_intrinsics.json     # Contains 'K' matrix

    Args:
        rgb_depth_root: Root directory containing RGB and depth data
        plane_label_root: Root directory containing plane labels
        intrinsics_root: Root directory containing camera intrinsics
        split_txt_dir: Directory containing split files (train.txt, val.txt, test.txt)
        split: 'train', 'val', or 'test'
        image_height: Target image height (default: 512)
        image_width: Target image width (default: 768)
        max_scenes: Maximum number of scenes to load (None = all)
    """
    def __init__(self,
                 rgb_depth_root,
                 plane_label_root,
                 intrinsics_root,
                 split_txt_dir,
                 split='train',
                 image_height=512,
                 image_width=768,
                 max_scenes=None):
        self.rgb_depth_root = rgb_depth_root
        self.plane_label_root = plane_label_root
        self.intrinsics_root = intrinsics_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load split file
        split_file = os.path.join(split_txt_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        # Build valid pairs: (scene_id, cam_name, frame_idx, fid, plane_h5, rgb_depth_h5, K)
        self.valid_pairs = []
        valid_scene_ids = []

        for scene_id in scene_ids:
            plane_scene_dir = os.path.join(plane_label_root, scene_id)
            rgb_depth_scene_dir = os.path.join(rgb_depth_root, scene_id)
            intrinsics_scene_dir = os.path.join(intrinsics_root, scene_id)

            if not os.path.exists(plane_scene_dir):
                print(f"[SKIP] Missing plane dir: {plane_scene_dir}")
                continue
            if not os.path.exists(rgb_depth_scene_dir):
                print(f"[SKIP] Missing RGB/depth dir: {rgb_depth_scene_dir}")
                continue
            if not os.path.exists(intrinsics_scene_dir):
                print(f"[SKIP] Missing intrinsics dir: {intrinsics_scene_dir}")
                continue

            # Find all plane HDF5 files (one per camera)
            plane_files = [f for f in os.listdir(plane_scene_dir)
                          if f.startswith("rendered_planes_") and f.endswith(".h5")]

            if len(plane_files) == 0:
                print(f"[SKIP] No plane files in {plane_scene_dir}")
                continue

            scene_frame_count = 0
            for plane_file in plane_files:
                # Extract camera name from filename: rendered_planes_cam_00.h5 -> cam_00
                cam_name = plane_file.replace("rendered_planes_", "").replace(".h5", "")
                plane_h5_path = os.path.join(plane_scene_dir, plane_file)
                rgb_depth_h5_path = os.path.join(rgb_depth_scene_dir, f"{cam_name}_merged.h5")
                intrinsics_json_path = os.path.join(intrinsics_scene_dir, f"{cam_name}_intrinsics.json")

                # Check if RGB/depth file exists
                if not os.path.exists(rgb_depth_h5_path):
                    print(f"[WARN] Missing RGB/depth for {scene_id}/{cam_name}: {rgb_depth_h5_path}")
                    continue

                # Check if intrinsics file exists
                if not os.path.exists(intrinsics_json_path):
                    print(f"[WARN] Missing intrinsics for {scene_id}/{cam_name}: {intrinsics_json_path}")
                    continue

                # Load intrinsics
                try:
                    with open(intrinsics_json_path, 'r') as f:
                        intrinsics_data = json.load(f)
                    K = np.array(intrinsics_data["K"], dtype=np.float32)  # (3, 3)
                except Exception as e:
                    print(f"[WARN] Failed to load intrinsics for {scene_id}/{cam_name}: {e}")
                    continue

                # Read frame_ids from plane HDF5
                try:
                    with h5py.File(plane_h5_path, "r") as f:
                        frame_ids = [fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                                    for fid in f["frame_ids"][:]]
                except Exception as e:
                    print(f"[SKIP] Error reading frame_ids from {plane_h5_path}: {e}")
                    continue

                # Validate that RGB/depth HDF5 has matching frames
                try:
                    with h5py.File(rgb_depth_h5_path, "r") as f:
                        n_frames_rgb = f["rgb"].shape[0]
                except Exception as e:
                    print(f"[WARN] Error reading RGB/depth {rgb_depth_h5_path}: {e}")
                    continue

                # Add valid pairs
                for idx, fid in enumerate(frame_ids):
                    if idx >= n_frames_rgb:
                        break
                    self.valid_pairs.append((
                        scene_id, cam_name, idx, fid,
                        plane_h5_path, rgb_depth_h5_path, K
                    ))
                    scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} frames")

        self.scene_ids = valid_scene_ids
        print(f"[Hypersim] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_id, cam_name, frame_idx, fid, plane_h5, rgb_depth_h5, K = self.valid_pairs[idx]

        # --- Load plane labels ---
        try:
            with h5py.File(plane_h5, "r") as f:
                plane = f["planes"][frame_idx]
            plane[plane < 0] = 0
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)  # [1, H, W]
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed plane label from {plane_h5} [{frame_idx}]: {e}")
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.int32)

        # --- Load RGB ---
        try:
            with h5py.File(rgb_depth_h5, "r") as f:
                rgb = f["rgb"][frame_idx]  # Assuming (H, W, 3) in [0, 255] or [0, 1]
            # Normalize to [0, 1] if needed
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            image = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        except Exception as e:
            print(f"[WARN] Failed RGB from {rgb_depth_h5} [{frame_idx}]: {e}")
            image = torch.zeros((3, H, W), dtype=torch.float32)

        # --- Load depth ---
        try:
            with h5py.File(rgb_depth_h5, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32)  # Already in meters
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed depth from {rgb_depth_h5} [{frame_idx}]: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic labels (optional, set to zeros if not available) ---
        sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- Camera pose (c2w) - set to identity if not available ---
        c2w = np.eye(4, dtype=np.float32)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "sem": sem,
            "rgb_path": f"{scene_id}/{cam_name}/{fid}",  # Virtual path for logging
            "K": torch.from_numpy(K),
            "c2w": torch.from_numpy(c2w),
            "scene_id": scene_id,
            "frame_idx": fid,
        }
