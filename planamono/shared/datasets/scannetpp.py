import os
import cv2
import torch
import numpy as np
from natsort import natsorted
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import h5py
import json

class ScanNetPPPlanarityDataset(Dataset):
    """ScanNet++ dataset for planarity learning with RGB, depth, plane mask (H5), and semantic label (H5)."""

    def __init__(self,
                 rgb_root,
                 plane_label_root,
                 sem_label_root,
                 depth_label_root,
                 split_txt_dir,
                 split='train',
                 image_height=512,
                 image_width=768,
                 max_scenes=None):
        self.rgb_root = rgb_root
        self.plane_label_root = plane_label_root
        self.sem_label_root = sem_label_root
        self.depth_label_root = depth_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load split
        split_file = os.path.join(split_txt_dir, f"nvs_sem_{split}_with_planes.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]
        self.scene_ids = scene_ids

        self.valid_pairs = []  # (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx)

        for scene_id in self.scene_ids:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            # plane_h5 = os.path.join(plane_label_root, scene_id, "rendered_planes.h5")
            plane_h5 = os.path.join(plane_label_root, scene_id, "rendered.h5")
            sem_h5 = os.path.join(sem_label_root, scene_id, "rendered_sem.h5")
            depth_h5 = os.path.join(depth_label_root, scene_id, "rendered_depth.h5")

            if not os.path.isdir(rgb_dir):
                print(f"[SKIP] Missing RGB dir: {rgb_dir}")
                continue
            if not os.path.exists(plane_h5):
                print(f"[SKIP] Missing plane h5: {plane_h5}")
                continue
            if not os.path.exists(sem_h5):
                print(f"[SKIP] Missing semantic h5: {sem_h5}")
                continue
            if not os.path.exists(depth_h5):
                print(f"[SKIP] Missing depth h5: {depth_h5}")
                continue

            try:
                with h5py.File(plane_h5, "r") as f:
                    frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
            except Exception as e:
                print(f"[SKIP] Error reading frame_ids from {plane_h5}: {e}")
                continue

            for idx, fid in enumerate(frame_ids):
                rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
                if not os.path.exists(rgb_path):
                    continue
                self.valid_pairs.append((rgb_path, plane_h5, sem_h5, depth_h5, idx))

            print(f"[DEBUG] Scene {scene_id} → {len(frame_ids)} matched frames")

        print(f"[ScanNet++] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, plane_h5, sem_h5, depth_h5, frame_idx = self.valid_pairs[idx]

        # --- Plane label ---
        try:
            with h5py.File(plane_h5, "r") as f:
                # plane = f["rendered_planes"][frame_idx]
                plane = f["planes"][frame_idx]
            plane = (plane > 0).astype(np.float32)
            plane = torch.from_numpy(plane).unsqueeze(0)  # [1, H, W]
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed plane label from {plane_h5} [{frame_idx}]: {e}")
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic label ---
        try:
            with h5py.File(sem_h5, "r") as f:
                sem = f["rendered_sem"][frame_idx]
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed semantic label from {sem_h5} [{frame_idx}]: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- Depth ---
        try:
            with h5py.File(depth_h5, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32) / 1000.0  # convert mm → meters
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed depth from {depth_h5} [{frame_idx}]: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- RGB ---
        image = cv2.imread(rgb_path)
        if image is None:
            raise RuntimeError(f"[ERROR] Cannot read RGB: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]

        return image, depth, plane, sem, rgb_path

class ScanNetPPPlaneDataset(Dataset):
    def __init__(self,
                 rgb_root,
                 plane_label_root,
                 sem_label_root,
                 depth_label_root,
                 split_txt_dir,
                 split='train',
                 image_height=512,
                 image_width=768,
                 max_scenes=None):
        self.rgb_root = rgb_root
        self.plane_label_root = plane_label_root
        self.sem_label_root = sem_label_root
        self.depth_label_root = depth_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load split
        split_file = os.path.join(split_txt_dir, f"nvs_sem_{split}_with_planes.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]
        self.scene_ids = scene_ids

        self.valid_pairs = []  # (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K)

        for scene_id in self.scene_ids:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            # plane_h5 = os.path.join(plane_label_root, scene_id, "rendered_planes.h5")
            plane_h5 = os.path.join(plane_label_root, scene_id, "rendered.h5")
            sem_h5 = os.path.join(sem_label_root, scene_id, "rendered_sem.h5")
            depth_h5 = os.path.join(depth_label_root, scene_id, "rendered_depth.h5")
            pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")

            # Skip scenes if anything missing
            if not (os.path.isdir(rgb_dir) and os.path.exists(plane_h5)
                    and os.path.exists(sem_h5) and os.path.exists(depth_h5)
                    and os.path.exists(pose_file)):
                print(f"[SKIP] Missing one or more required files for scene: {scene_id}")
                continue

            # Load per-frame intrinsics from JSON
            with open(pose_file, "r") as f:
                pose_data = json.load(f)

            try:
                with h5py.File(plane_h5, "r") as f:
                    frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
            except Exception as e:
                print(f"[SKIP] Error reading frame_ids from {plane_h5}: {e}")
                continue

            for idx, fid in enumerate(frame_ids):
                rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
                if not os.path.exists(rgb_path):
                    continue
                if fid not in pose_data:
                    print(f"[WARN] Missing intrinsic for {fid} in {scene_id}")
                    continue

                K = np.array(pose_data[fid]["intrinsic"], dtype=np.float32)  # (3, 3)
                c2w = np.array(pose_data[fid]["aligned_pose"], dtype=np.float32)    # (4, 4)
                self.valid_pairs.append((rgb_path, plane_h5, sem_h5, depth_h5, idx, K, c2w))
                # self.valid_pairs.append((rgb_path, plane_h5, sem_h5, depth_h5, idx, K))

            print(f"[DEBUG] Scene {scene_id} → {len(frame_ids)} matched frames")

        print(f"[ScanNet++] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w = self.valid_pairs[idx]

        # --- Plane label ---
        try:
            with h5py.File(plane_h5, "r") as f:
                # plane = f["rendered_planes"][frame_idx]
                plane = f["planes"][frame_idx]
            plane[plane < 0] = 0
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)  # [1, H, W]
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed plane label from {plane_h5} [{frame_idx}]: {e}")
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic label ---
        try:
            with h5py.File(sem_h5, "r") as f:
                sem = f["rendered_sem"][frame_idx]
            sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed semantic label from {sem_h5} [{frame_idx}]: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- Depth ---
        try:
            with h5py.File(depth_h5, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32) / 1000.0
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed depth from {depth_h5} [{frame_idx}]: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- RGB ---
        image = cv2.imread(rgb_path)
        if image is None:
            raise RuntimeError(f"[ERROR] Cannot read RGB: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]

        # return image, depth, plane, sem, rgb_path, torch.from_numpy(K)
        return image, depth, plane, sem, rgb_path, torch.from_numpy(K), torch.from_numpy(c2w)


class ScanNetPPPlanarityDataset_v0(Dataset):
    """ScanNet++ dataset for planarity learning (RGB + rendered plane masks)."""

    def __init__(self,
                 rgb_root,
                 label_root,
                 split_txt_dir,
                 split='train',
                 image_height=512,
                 image_width=768,
                 max_scenes=None):
        """
        Args:
            rgb_root: Base directory for RGB images (contains <scene_id>/iphone/rgb/)
            label_root: Base directory for plane segmentation masks
            split_txt_dir: Directory containing split txt files (train/val lists)
            split: 'train' or 'val'
            image_height, image_width: Resize dimensions
            max_scenes: Limit number of scenes to load (None = all)
        """
        self.rgb_root = rgb_root
        self.label_root = label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # --- Load scene list from split file ---
        split_file = os.path.join(split_txt_dir, f"nvs_sem_{split}_with_planes.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]

        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]
        self.scene_ids = scene_ids

        # --- Build image-label pairs ---
        self.valid_pairs = []

        for scene_id in self.scene_ids:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            label_dir = os.path.join(label_root, scene_id, "rendered")

            if not os.path.isdir(rgb_dir):
                print(f"[SKIP] Missing RGB dir: {rgb_dir}")
                continue
            if not os.path.isdir(label_dir):
                print(f"[SKIP] Missing label dir: {label_dir}")
                continue

            # --- Step 1: collect valid frame basenames from label folder ---
            label_files = natsorted([
                f for f in os.listdir(label_dir)
                if f.endswith(".png")
            ])
            valid_fids = set(os.path.splitext(f)[0] for f in label_files)

            # --- Step 2: match RGBs that exist in valid_fids ---
            rgb_files = natsorted([
                f for f in os.listdir(rgb_dir)
                if f.endswith(".jpg") and os.path.splitext(f)[0] in valid_fids
            ])

            if len(rgb_files) == 0:
                print(f"[WARN] No matching frames for scene {scene_id}")
                continue

            print(f"[DEBUG] Scene {scene_id} → {len(rgb_files)} matched frames")

            for fname in rgb_files:
                fid = os.path.splitext(fname)[0]
                rgb_path = os.path.join(rgb_dir, fname)
                label_path = os.path.join(label_dir, f"{fid}.png")
                self.valid_pairs.append((rgb_path, label_path))

        print(f"[ScanNet++] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        rgb_path, label_path = self.valid_pairs[idx]
    
        # --- Load label first to get its shape ---
        try:
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if label is None:
                raise IOError(f"Cannot read label: {label_path}")
    
            # Planar mask: 1.0 for planar, 0.0 otherwise
            label = (label > 0).astype(np.float32)
            label = torch.from_numpy(label).unsqueeze(0)  # Shape: [1, H, W]
    
            label_H, label_W = label.shape[1], label.shape[2]
    
        except Exception as e:
            print(f"[WARN] Failed loading label {label_path}: {e}")
            label_H, label_W = self.image_height, self.image_width
            label = torch.zeros((1, label_H, label_W), dtype=torch.float32)
    
        # --- Load and resize RGB to match label size ---
        image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (label_W, label_H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Shape: [3, H, W]
    
        return image, label, rgb_path
    