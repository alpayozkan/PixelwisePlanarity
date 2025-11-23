"""
ScanNet++ dataset loader for plane segmentation.

Loads RGB, depth, plane labels, semantic labels, camera intrinsics, and poses.
All data stored in HDF5 format for efficiency.
"""

import os
import json
import torch
import h5py
import cv2
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from typing import Optional, Tuple, List


class ScanNetPPPlaneDataset(Dataset):
    """
    ScanNet++ dataset with RGB images, depth, plane segmentation, semantics,
    camera intrinsics (K), and camera-to-world poses (c2w).
    """

    def __init__(
        self,
        rgb_root: str,
        plane_label_root: str,
        sem_label_root: str,
        depth_label_root: str,
        split_txt_dir: str,
        split: str = 'train',
        image_height: int = 512,
        image_width: int = 768,
        max_scenes: Optional[int] = None
    ):
        """
        Args:
            rgb_root: Root directory for RGB images
            plane_label_root: Root directory for rendered plane HDF5 files
            sem_label_root: Root directory for semantic label HDF5 files
            depth_label_root: Root directory for depth HDF5 files
            split_txt_dir: Directory containing split text files
            split: Dataset split ('train', 'val', or 'test')
            image_height: Target image height
            image_width: Target image width
            max_scenes: Optional limit on number of scenes
        """
        self.rgb_root = rgb_root
        self.plane_label_root = plane_label_root
        self.sem_label_root = sem_label_root
        self.depth_label_root = depth_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load scene list from split file
        split_file = os.path.join(split_txt_dir, f"nvs_sem_{split}_with_planes.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]

        scene_ids = natsorted(scene_ids)
        if max_scenes:
            scene_ids = scene_ids[:max_scenes]
        self.scene_ids = scene_ids

        # Build dataset: (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w)
        self.valid_pairs = []

        for scene_id in self.scene_ids:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")
            plane_h5 = os.path.join(plane_label_root, scene_id, "rendered_planes.h5")
            sem_h5 = os.path.join(sem_label_root, scene_id, "rendered_sem.h5")
            depth_h5 = os.path.join(depth_label_root, scene_id, "rendered_depth.h5")

            # Skip if missing files
            if not (os.path.isdir(rgb_dir) and os.path.exists(pose_file) and
                    os.path.exists(plane_h5) and os.path.exists(sem_h5) and
                    os.path.exists(depth_h5)):
                print(f"[SKIP] Scene {scene_id} missing files")
                continue

            # Load camera data
            try:
                with open(pose_file, "r") as f:
                    pose_data = json.load(f)
            except Exception as e:
                print(f"[SKIP] Failed loading pose for {scene_id}: {e}")
                continue

            # Get frame IDs from HDF5
            try:
                with h5py.File(plane_h5, "r") as f:
                    frame_ids = np.array(f["frame_ids"]).astype(str)
            except Exception as e:
                print(f"[SKIP] Failed reading frame_ids from {plane_h5}: {e}")
                continue

            # Add valid frame pairs
            for idx, fid in enumerate(frame_ids):
                if fid not in pose_data:
                    continue

                K = np.array(pose_data[fid]["intrinsic"], dtype=np.float32)
                c2w = np.array(pose_data[fid]["aligned_pose"], dtype=np.float32)
                rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")

                self.valid_pairs.append((rgb_path, plane_h5, sem_h5, depth_h5,
                                       idx, K, c2w, scene_id, fid))

        print(f"[ScanNet++] {split} split → {len(self.valid_pairs)} frames from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w, scene_id, frame_id = self.valid_pairs[idx]

        # Load plane mask
        try:
            with h5py.File(plane_h5, "r") as f:
                plane = f["rendered_planes"][frame_idx]
            plane[plane < 0] = 0  # Negative labels → background
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed loading plane: {plane_h5}[{frame_idx}]: {e}")
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.int32)

        # Load semantic labels
        try:
            with h5py.File(sem_h5, "r") as f:
                sem = f["rendered_sem"][frame_idx]
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed loading semantic: {sem_h5}[{frame_idx}]: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # Load depth
        try:
            with h5py.File(depth_h5, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32) / 1000.0  # mm → meters
            depth = torch.from_numpy(depth).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed loading depth: {depth_h5}[{frame_idx}]: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # Load RGB image
        image = cv2.imread(rgb_path)
        if image is None:
            raise RuntimeError(f"[ERROR] Cannot read RGB: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "pose": torch.from_numpy(c2w),
            "intrinsic": torch.from_numpy(K),
            "scene_id": scene_id,
            "frame_id": frame_id,
        }
