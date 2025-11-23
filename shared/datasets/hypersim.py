"""
Hypersim dataset loader for plane segmentation.

Loads RGB (with HDR tone mapping), depth, plane labels, and semantic labels.
All data stored in HDF5 format.
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from typing import Optional


class HypersimPlanarityDataset(Dataset):
    """
    Hypersim dataset with RGB images, depth, plane segmentation, and semantics.
    """

    def __init__(
        self,
        root_dir: str,
        plane_label_root: str,
        filtered_csv_path: str,
        split: str = 'train',
        image_height: int = 512,
        image_width: int = 768,
        max_samples: Optional[int] = None,
        preprocessed_rgb_dir: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory for Hypersim dataset
            plane_label_root: Root directory for plane label HDF5 files
            filtered_csv_path: Path to CSV with scene/frame metadata
            split: Dataset split ('train', 'val', or 'test')
            image_height: Target image height
            image_width: Target image width
            max_samples: Optional limit on number of samples
            preprocessed_rgb_dir: Optional directory with pre-tone-mapped RGB .npy files
        """
        self.root_dir = root_dir
        self.plane_label_root = plane_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.preprocessed_rgb_dir = preprocessed_rgb_dir

        # Load CSV metadata
        df = pd.read_csv(filtered_csv_path)
        df = df[df["split_partition_name"] == split]
        if max_samples is not None:
            df = df.iloc[:max_samples]
        self.df = df.reset_index(drop=True)

        # Build dataset entries
        self.valid_pairs = []
        for idx, row in self.df.iterrows():
            scene_id = row["scene_name"]
            cam_name = row["camera_name"]  # e.g., "cam_00"
            frame_id = f"{int(row['frame_id']):04d}"

            color_dir = os.path.join(root_dir, scene_id, "images",
                                    f"scene_{cam_name}_final_hdf5")
            geom_dir = os.path.join(root_dir, scene_id, "images",
                                   f"scene_{cam_name}_geometry_hdf5")

            rgb_path = os.path.join(color_dir, f"frame.{frame_id}.color.hdf5")
            sem_path = os.path.join(geom_dir, f"frame.{frame_id}.semantic.hdf5")
            depth_path = os.path.join(geom_dir, f"frame.{frame_id}.depth_meters.hdf5")
            plane_h5 = os.path.join(plane_label_root, scene_id,
                                   f"rendered_planes_{cam_name}.h5")

            self.valid_pairs.append((scene_id, frame_id, cam_name,
                                    rgb_path, sem_path, depth_path, plane_h5))

        print(f"[Hypersim] {split} split → {len(self.valid_pairs)} frames")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_id, frame_id, cam_name, rgb_path, sem_path, depth_path, plane_h5 = self.valid_pairs[idx]
        H, W = self.image_height, self.image_width

        # Load plane segmentation
        try:
            with h5py.File(plane_h5, "r") as f:
                frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
                if frame_id not in frame_ids:
                    raise ValueError(f"frame_id {frame_id} not in HDF5")
                frame_idx = frame_ids.index(frame_id)
                plane = f["planes"][frame_idx]
            plane = (plane > 0).astype(np.float32)  # Binary planarity mask
            plane = torch.from_numpy(plane).unsqueeze(0)
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed loading plane for {scene_id}/{frame_id}: {e}")
            plane = torch.zeros((1, H, W), dtype=torch.float32)

        # Load semantic labels
        try:
            with h5py.File(sem_path, "r") as f:
                sem = f["dataset"][:]
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed loading semantic for {scene_id}/{frame_id}: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # Load depth
        try:
            with h5py.File(depth_path, "r") as f:
                depth = f["dataset"][:].astype(np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed loading depth for {scene_id}/{frame_id}: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # Load RGB with tone mapping
        try:
            if self.preprocessed_rgb_dir is not None:
                # Load pre-tone-mapped RGB
                npy_path = os.path.join(self.preprocessed_rgb_dir,
                                       f"{scene_id}_{frame_id}.npy")
                rgb = np.load(npy_path)  # Already in [0,1]
            else:
                # Apply tone mapping on-the-fly
                rgb = self.load_hypersim_rgb_hdr(rgb_path)

            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            image = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        except Exception as e:
            print(f"[WARN] Failed loading RGB for {scene_id}/{frame_id}: {e}")
            image = torch.zeros((3, H, W), dtype=torch.float32)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "scene_id": scene_id,
            "frame_idx": int(frame_id),
        }

    def load_hypersim_rgb_hdr(
        self,
        h5_path: str,
        percentile: float = 90,
        target_max: float = 0.8,
        gamma: float = 2.2
    ) -> np.ndarray:
        """
        Load HDR RGB from Hypersim HDF5 and apply tone mapping.

        Args:
            h5_path: Path to .color.hdf5 file
            percentile: Brightness percentile for auto-exposure
            target_max: Target max brightness after scaling
            gamma: Gamma correction value

        Returns:
            rgb: (H,W,3) tone-mapped RGB in [0,1] as float32
        """
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            assert len(keys) == 1, f"Unexpected HDF5 structure: {h5_path}"
            hdr = f[keys[0]][:]  # (H, W, 3)

        # Auto-exposure based on brightness percentile
        brightness = hdr.mean(axis=2)
        scale_val = np.percentile(brightness, percentile)
        scale_val = max(scale_val, 1e-6)

        # Scale and gamma correct
        img = hdr * (target_max / scale_val)
        img = np.clip(img, 0, None)
        img = img ** (1.0 / gamma)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        return img
