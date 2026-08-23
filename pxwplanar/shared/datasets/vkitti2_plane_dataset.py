"""
Virtual KITTI 2 dataset for plane segmentation evaluation.

Data format:
    data_root/
        <scene>/              (Scene01, Scene02, Scene06, Scene18, Scene20)
            <variant>/        (clone, fog, morning, overcast, rain, sunset, 15-deg-left, ...)
                scene_data.h5
                planes.json   (optional, per-plane parameters)

Each scene_data.h5 contains:
    rgb:        (N, 375, 1242, 3) uint8
    depth:      (N, 375, 1242)    float32 in meters
    planes:     (N, 375, 1242)    uint16, 0 = non-planar
    semantic:   (N, 375, 1242)    int8
    c2w:        (N, 4, 4)         float32
    frame_ids:  (N,)              bytes
    Attributes: fx, fy, cx, cy, image_height, image_width, scene, variant, num_frames
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted


class VKITTI2PlaneDataset(Dataset):
    """
    Virtual KITTI 2 dataset for plane segmentation evaluation.

    Returns the same dict format as ScanNetPPPlaneDataset:
        image:     (3, H, W)  float32 [0, 1]
        plane:     (1, H, W)  int32, 0 = non-planar
        depth:     (1, H, W)  float32 in meters
        sem:       (1, H, W)  int64
        K:         (3, 3)     float32
        c2w:       (4, 4)     float32
        rgb_path:  str        "scene/variant/frame_id"
        scene_id:  str        "scene/variant"
        frame_idx: str        frame identifier

    Args:
        data_root: Root directory of VKITTI2 planes dataset.
        split_txt_dir: Directory containing split files (train.txt, val.txt, test.txt).
            Each line is a scene name (e.g., "Scene01"). All variants of that scene
            are included unless filtered by `variants`.
        split: 'train', 'val', or 'test'.
        variants: List of variant names to include. None = all variants found on disk.
        image_height: Target image height (None = native 375).
        image_width: Target image width (None = native 1242).
        max_scenes: Maximum number of scenes to load (None = all).
    """

    ALL_VARIANTS = [
        "clone", "fog", "morning", "overcast", "rain", "sunset",
        "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
    ]

    def __init__(
        self,
        data_root,
        split_txt_dir,
        split="train",
        variants=None,
        image_height=None,
        image_width=None,
        max_scenes=None,
    ):
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.variants = variants

        # Load split file
        split_file = os.path.join(split_txt_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, "r") as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        # Build valid pairs
        self.valid_pairs = []  # (h5_path, frame_idx_int, scene, variant, K, frame_id_str)
        valid_scene_ids = []

        for scene in scene_ids:
            scene_dir = os.path.join(data_root, scene)
            if not os.path.isdir(scene_dir):
                print(f"[SKIP] Missing scene dir: {scene_dir}")
                continue

            # Discover variants on disk
            available_variants = natsorted([
                d for d in os.listdir(scene_dir)
                if os.path.isdir(os.path.join(scene_dir, d))
            ])

            # Filter by requested variants
            if variants is not None:
                available_variants = [v for v in available_variants if v in variants]

            scene_frame_count = 0
            for variant in available_variants:
                h5_path = os.path.join(scene_dir, variant, "scene_data.h5")
                if not os.path.exists(h5_path):
                    print(f"[SKIP] Missing H5: {h5_path}")
                    continue

                try:
                    with h5py.File(h5_path, "r") as f:
                        num_frames = f.attrs.get("num_frames", f["rgb"].shape[0])
                        fx = float(f.attrs["fx"])
                        fy = float(f.attrs["fy"])
                        cx = float(f.attrs["cx"])
                        cy = float(f.attrs["cy"])
                        frame_ids = [
                            fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                            for fid in f["frame_ids"][:]
                        ]
                except Exception as e:
                    print(f"[SKIP] Error reading {h5_path}: {e}")
                    continue

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

                for idx in range(num_frames):
                    fid = frame_ids[idx] if idx < len(frame_ids) else str(idx)
                    self.valid_pairs.append((h5_path, idx, scene, variant, K, fid))
                    scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene)
                print(f"[DEBUG] Scene {scene} → {scene_frame_count} frames")

        self.scene_ids = valid_scene_ids
        print(f"[VKITTI2] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        h5_path, frame_idx, scene, variant, K, fid = self.valid_pairs[idx]

        with h5py.File(h5_path, "r") as f:
            # --- RGB ---
            rgb = f["rgb"][frame_idx]  # (H, W, 3) uint8
            image = torch.tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)

            # --- Plane labels ---
            plane = f["planes"][frame_idx]  # (H, W) uint16
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)  # (1, H, W)

            # --- Depth ---
            depth = f["depth"][frame_idx]  # (H, W) float32, already in meters
            depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)  # (1, H, W)

            # --- Semantic labels ---
            sem = f["semantic"][frame_idx]  # (H, W) int8
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)  # (1, H, W)

            # --- Camera-to-world pose ---
            # identity fallback: scene_data.h5 omits c2w when the extraction
            # ran without extrinsic.txt (per-frame fitting is pose-invariant)
            if "c2w" in f:
                c2w = f["c2w"][frame_idx]  # (4, 4) float32
            else:
                c2w = np.eye(4, dtype=np.float32)
            c2w = torch.from_numpy(c2w.astype(np.float32))

        # Resize if target dimensions specified
        H, W = image.shape[1], image.shape[2]
        target_h = self.image_height or H
        target_w = self.image_width or W

        if target_h != H or target_w != W:
            import cv2

            # Resize image (bilinear)
            img_np = image.permute(1, 2, 0).numpy()
            img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            image = torch.from_numpy(img_np).permute(2, 0, 1)

            # Resize labels (nearest)
            plane_np = plane.squeeze(0).numpy()
            plane_np = cv2.resize(plane_np.astype(np.float32), (target_w, target_h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
            plane = torch.from_numpy(plane_np).unsqueeze(0)

            depth_np = depth.squeeze(0).numpy()
            depth_np = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth = torch.from_numpy(depth_np).unsqueeze(0)

            sem_np = sem.squeeze(0).numpy()
            sem_np = cv2.resize(sem_np.astype(np.float32), (target_w, target_h),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            sem = torch.from_numpy(sem_np).unsqueeze(0)

            # Rescale intrinsics
            K = K.copy()
            K[0, :] *= target_w / W
            K[1, :] *= target_h / H

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "sem": sem,
            "rgb_path": f"{scene}/{variant}/{fid}",
            "K": torch.from_numpy(K),
            "c2w": c2w,
            "scene_id": f"{scene}/{variant}",
            "frame_idx": fid,
        }
