import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class HypersimPlanarityDataset(Dataset):
    def __init__(self,
                 root_dir,
                 plane_label_root,
                 filtered_csv_path,
                 split='train',
                 image_height=512,
                 image_width=768,
                 max_samples=None,
                 preprocessed_rgb_dir=None):
        self.root_dir = root_dir
        self.plane_label_root = plane_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.preprocessed_rgb_dir = preprocessed_rgb_dir

        # === Load CSV ===
        df = pd.read_csv(filtered_csv_path)
        df = df[df["split_partition_name"] == split]
        if max_samples is not None:
            df = df.iloc[:max_samples]
        self.df = df.reset_index(drop=True)

        # === Prepare sample entries (lightweight) ===
        self.valid_pairs = []
        for idx, row in self.df.iterrows():
            scene_id = row["scene_name"]
            cam_name = row["camera_name"]  # e.g. "cam_00"
            frame_id = f"{int(row['frame_id']):04d}"

            color_dir = os.path.join(root_dir, scene_id, "images", f"scene_{cam_name}_final_hdf5")
            geom_dir  = os.path.join(root_dir, scene_id, "images", f"scene_{cam_name}_geometry_hdf5")

            rgb_path   = os.path.join(color_dir, f"frame.{frame_id}.color.hdf5")
            sem_path   = os.path.join(geom_dir,  f"frame.{frame_id}.semantic.hdf5")
            depth_path = os.path.join(geom_dir,  f"frame.{frame_id}.depth_meters.hdf5")
            plane_h5   = os.path.join(self.plane_label_root, scene_id, f"rendered_planes_{cam_name}.h5")

            self.valid_pairs.append((scene_id, frame_id, cam_name, rgb_path, sem_path, depth_path, plane_h5))

        print(f"[Hypersim] {split} split → {len(self.valid_pairs)} potential pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_id, frame_id, cam_name, rgb_path, sem_path, depth_path, plane_h5 = self.valid_pairs[idx]
        H, W = self.image_height, self.image_width

        # === Load plane segmentation ===
        try:
            with h5py.File(plane_h5, "r") as f:
                frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
                if frame_id not in frame_ids:
                    raise ValueError("frame_id not in HDF5 plane file")
                frame_idx = frame_ids.index(frame_id)
                plane = f["planes"][frame_idx]
            plane = (plane > 0).astype(np.float32)
            plane = torch.from_numpy(plane).unsqueeze(0)
            H, W = plane.shape[1:]  # update to true shape
        except Exception as e:
            plane = torch.zeros((1, H, W), dtype=torch.float32)

        # === Load semantic ===
        try:
            with h5py.File(sem_path, "r") as f:
                sem = f["dataset"][:]
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)
        except:
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # === Load depth ===
        try:
            with h5py.File(depth_path, "r") as f:
                depth = f["dataset"][:].astype(np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0)
        except:
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # === Load RGB ===
        try:
            if self.preprocessed_rgb_dir is not None:
                npy_path = os.path.join(self.preprocessed_rgb_dir, f"{scene_id}_{frame_id}.npy")
                rgb = np.load(npy_path)  # pre-tone-mapped RGB float32 in [0,1]
            else:
                rgb = self.load_hypersim_rgb(rgb_path)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            image = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
        except Exception as e:
            print(f"[WARN] Failed RGB for {scene_id}/{frame_id}: {e}")
            image = torch.zeros((3, H, W), dtype=torch.float32)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "scene_id": scene_id,
            "frame_idx": int(frame_id),
        }

    def load_hypersim_rgb(self, h5_path, percentile=90, target_max=0.8, gamma=2.2):
        """
        Load RGB from Hypersim HDR .color.hdf5 and apply tone mapping.

        Returns:
            np.ndarray: RGB image (H, W, 3) in [0,1] as float32
        """
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            assert len(keys) == 1, f"Unexpected HDF5 structure in {h5_path}"
            hdr = f[keys[0]][:]  # (H, W, 3), float16 or float32

        brightness = hdr.mean(axis=2)
        scale_val = np.percentile(brightness, percentile)
        scale_val = max(scale_val, 1e-6)

        img = hdr * (target_max / scale_val)
        img = np.clip(img, 0, None)
        img = img ** (1.0 / gamma)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return img
