"""
Synthia Planes dataset for plane segmentation evaluation.

Dataset structure:
    data_root/
        train/
            <scene_name>/
                scene_data.h5   (rgb, depth, planes, semantic, frame_ids
                                 + root attrs with K)
                planes.json     (per-plane metadata: normals, distances,
                                 class names, p95)
        test/
            <scene_name>/
                ...

H5 datasets:
    rgb:       (N, 480, 640, 3) uint8
    depth:     (N, 480, 640)    float32  (meters)
    planes:    (N, 480, 640)    uint16   (instance labels, 0 = non-planar)
    semantic:  (N, 480, 640)    uint8
    frame_ids: (N,)             bytes

H5 root attrs:
    fx, fy, cx, cy, image_height, image_width, num_frames, scene_name, dataset

planes.json:
    List of dicts with keys: frame_id, plane_id, n (3-vec), d, num_pixels,
    p95, class_id, class_name

Returns the same dict format as ScanNetPPPlaneDataset / HypersimPlaneDataset:
    image:     (3, H, W) float32 [0, 1]
    depth:     (1, H, W) float32 meters
    plane:     (1, H, W) int32   (0 = non-planar)
    sem:       (1, H, W) int64
    K:         (3, 3)   float32
    c2w:       (4, 4)   float32  (identity — no poses in Synthia)
    rgb_path:  str      "<scene_name>/<frame_id>"
    scene_id:  str
    frame_idx: str
"""

import json
import os

import cv2
import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


class SynthiaPlaneDataset(Dataset):
    """Synthia Planes dataset for plane segmentation evaluation.

    Each scene contains a single ``scene_data.h5`` with all modalities and a
    ``planes.json`` with per-plane metadata (normals, offsets, class labels).

    Camera intrinsics are read from the H5 root attributes (constant across
    the dataset: fx=fy=895.692, cx=320, cy=240 at 640x480).
    """

    def __init__(
        self,
        data_root,
        split="train",
        image_height=480,
        image_width=640,
        max_scenes=None,
    ):
        """
        Args:
            data_root: Root directory containing train/ and test/
                subdirectories.
            split: 'train' or 'test'.
            image_height: Target image height for resizing
                (default 480 = native).
            image_width: Target image width for resizing (default 640 = native).
            max_scenes: Limit number of scenes to load (None = all).
        """
        self.data_root = data_root
        self.split = split
        self.image_height = image_height
        self.image_width = image_width

        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        scene_names = natsorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )
        if max_scenes is not None:
            scene_names = scene_names[:max_scenes]

        # Build (scene_path, frame_idx_int, frame_id_str, K) tuples
        self.valid_pairs = []
        valid_scene_ids = []

        for scene_name in scene_names:
            scene_path = os.path.join(split_dir, scene_name)
            h5_path = os.path.join(scene_path, "scene_data.h5")
            json_path = os.path.join(scene_path, "planes.json")

            if not os.path.exists(h5_path):
                print(f"[SKIP] Missing scene_data.h5: {h5_path}")
                continue

            try:
                with h5py.File(h5_path, "r") as f:
                    attrs = dict(f.attrs)
                    f["frame_ids"].shape[0]
                    frame_ids = [
                        fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in f["frame_ids"][:]
                    ]
            except Exception as e:
                print(f"[SKIP] Error reading {h5_path}: {e}")
                continue

            # Build intrinsics from H5 attrs
            fx = float(attrs.get("fx", 895.692))
            fy = float(attrs.get("fy", 895.692))
            cx = float(attrs.get("cx", 320.0))
            cy = float(attrs.get("cy", 240.0))
            K = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

            for idx, fid in enumerate(frame_ids):
                self.valid_pairs.append(
                    (scene_name, h5_path, json_path, idx, fid, K)
                )

            valid_scene_ids.append(scene_name)

        self.scene_ids = valid_scene_ids
        print(
            f"[Synthia] {split} split → {len(self.valid_pairs)} frames "
            f"from {len(self.scene_ids)} scenes"
        )

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_name, h5_path, json_path, frame_idx, frame_id, K = (
            self.valid_pairs[idx]
        )

        with h5py.File(h5_path, "r") as f:
            rgb_raw = f["rgb"][frame_idx]  # (H, W, 3) uint8
            depth_raw = f["depth"][frame_idx]  # (H, W) float32
            plane_raw = f["planes"][frame_idx]  # (H, W) uint16
            sem_raw = f["semantic"][frame_idx]  # (H, W) uint8

        H_native, W_native = rgb_raw.shape[:2]
        H, W = self.image_height, self.image_width
        need_resize = (H_native != H) or (W_native != W)

        # --- RGB ---
        if need_resize:
            rgb_raw = cv2.resize(
                rgb_raw, (W, H), interpolation=cv2.INTER_LINEAR
            )
        image = torch.from_numpy(rgb_raw.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )  # (3, H, W)

        # --- Depth ---
        if need_resize:
            depth_raw = cv2.resize(
                depth_raw, (W, H), interpolation=cv2.INTER_LINEAR
            )
        depth = torch.from_numpy(depth_raw.astype(np.float32)).unsqueeze(
            0
        )  # (1, H, W)

        # --- Plane labels ---
        if need_resize:
            plane_raw = cv2.resize(
                plane_raw, (W, H), interpolation=cv2.INTER_NEAREST
            )
        plane = torch.from_numpy(plane_raw.astype(np.int32)).unsqueeze(
            0
        )  # (1, H, W)

        # --- Semantic labels ---
        if need_resize:
            sem_raw = cv2.resize(
                sem_raw, (W, H), interpolation=cv2.INTER_NEAREST
            )
        sem = torch.from_numpy(sem_raw.astype(np.int64)).unsqueeze(
            0
        )  # (1, H, W)

        # --- Intrinsics (rescale if resized) ---
        K_out = K.copy()
        if need_resize:
            K_out[0, :] *= W / W_native  # fx, cx
            K_out[1, :] *= H / H_native  # fy, cy

        # --- Pose (not available — identity) ---
        c2w = np.eye(4, dtype=np.float32)

        return {
            "image": image,  # (3, H, W) float32 [0, 1]
            "depth": depth,  # (1, H, W) float32 meters
            "plane": plane,  # (1, H, W) int32, 0 = non-planar
            "sem": sem,  # (1, H, W) int64
            "K": torch.from_numpy(K_out),  # (3, 3) float32
            "c2w": torch.from_numpy(c2w),  # (4, 4) float32
            # virtual path (no file on disk)
            "rgb_path": f"{scene_name}/{frame_id}",
            "scene_id": scene_name,
            "frame_idx": frame_id,
        }

    def get_plane_metadata(self, idx):
        """Load per-plane metadata from planes.json for a given sample.

        Returns a list of dicts with keys:
            frame_id, plane_id, n (normal 3-vec), d (offset),
            num_pixels, p95, class_id, class_name
        Returns empty list if planes.json is missing.
        """
        scene_name, h5_path, json_path, frame_idx, frame_id, K = (
            self.valid_pairs[idx]
        )
        if not os.path.exists(json_path):
            return []
        with open(json_path) as f:
            all_planes = json.load(f)
        return [p for p in all_planes if p["frame_id"] == frame_id]
