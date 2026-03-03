"""
ScanNet plane datasets — same interface as ScanNetPPPlaneDataset.

Data format differences vs ScanNet++:
  - Individual files per frame (jpg/png/txt) instead of HDF5
  - Plane segmentation encoded as 3-channel RGB PNG:
        plane_id = (R*256*256 + G*256 + B) // 100 - 1
  - Intrinsics shared across all frames (intrinsic_depth.txt), not per-frame JSON
  - Poses stored as individual 4x4 .txt files, not JSON
  - Semantic labels in label-filt/ directory as uint16 PNGs

Expected layout:
    <data_root>/scans/<scene_id>/
        ├── frames/
        │   ├── color/<idx>.jpg
        │   ├── depth/<idx>.png           # uint16, /1000 → meters
        │   ├── pose/<idx>.txt            # 4x4 camera-to-world
        │   └── intrinsic/
        │       └── intrinsic_depth.txt   # 4x4 intrinsic matrix
        ├── annotation/
        │   ├── planes.npy                # (N, 3) global plane parameters
        │   ├── plane_info.npy            # (N, ...) plane metadata
        │   └── segmentation/<idx>.png    # 3-channel encoded plane masks
        └── label-filt/<idx>.png          # semantic labels (optional)
"""

import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


def _decode_scannet_segmentation(seg_img):
    """Decode ScanNet 3-channel PNG to plane indices.

    OpenCV loads as BGR, so seg_img[:,:,0]=B, [:,:,1]=G, [:,:,2]=R.
    Encoding: plane_id = (R*256^2 + G*256 + B) // 100 - 1
    Returns int32 array: -1 = unlabeled, 0..N = plane index.
    """
    seg = seg_img.astype(np.int32)
    return (seg[:, :, 2] * 256 * 256 + seg[:, :, 1] * 256 + seg[:, :, 0]) // 100 - 1


def _remap_plane_ids(raw_ids, num_global_planes, area_threshold=500):
    """Remap raw ScanNet plane indices to contiguous 1-based IDs.

    Filters out: invalid indices (-1, 167771), out-of-range, zero-norm, small area.
    Returns int32 array: 0 = non-planar, 1..M = valid plane instances.
    """
    result = np.zeros(raw_ids.shape, dtype=np.int32)
    segments, counts = np.unique(raw_ids, return_counts=True)

    new_id = 1
    for seg_id, count in sorted(zip(segments.tolist(), counts.tolist()), key=lambda x: -x[1]):
        if seg_id < 0 or seg_id == 167771:
            continue
        if seg_id >= num_global_planes:
            continue
        if count < area_threshold:
            continue
        result[raw_ids == seg_id] = new_id
        new_id += 1
    return result


def _load_intrinsic(scene_dir, scene_id):
    """Load 3x3 K from intrinsic_depth.txt, fallback to scene .txt."""
    intrinsic_path = os.path.join(scene_dir, 'frames', 'intrinsic', 'intrinsic_depth.txt')
    if os.path.isfile(intrinsic_path):
        mat = np.loadtxt(intrinsic_path, dtype=np.float32)
        return mat[:3, :3].copy()

    scene_txt = os.path.join(scene_dir, f"{scene_id}.txt")
    if os.path.isfile(scene_txt):
        params = {}
        with open(scene_txt) as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 3:
                    params[tokens[0]] = tokens[2]
        return np.array([
            [float(params.get('fx_depth', 0)), 0, float(params.get('mx_depth', 0))],
            [0, float(params.get('fy_depth', 0)), float(params.get('my_depth', 0))],
            [0, 0, 1],
        ], dtype=np.float32)

    return None


class ScanNetPlanarityDataset(Dataset):
    """ScanNet dataset for planarity learning — binary planar mask.

    Matches ScanNetPPPlaneDataset interface.
    Returns: image, depth, plane (binary float32), sem, rgb_path, scene_id, frame_idx.
    """

    def __init__(self,
                 data_root,
                 split_txt,
                 split='train',
                 image_height=480,
                 image_width=640,
                 max_scenes=None,
                 plane_area_threshold=500):
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.plane_area_threshold = plane_area_threshold

        with open(split_txt, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        self.valid_pairs = []
        valid_scene_ids = []

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_root, 'scans', scene_id)
            color_dir = os.path.join(scene_dir, 'frames', 'color')
            depth_dir = os.path.join(scene_dir, 'frames', 'depth')
            seg_dir = os.path.join(scene_dir, 'annotation', 'segmentation')
            planes_path = os.path.join(scene_dir, 'annotation', 'planes.npy')
            label_dir = os.path.join(scene_dir, 'label-filt')

            if not os.path.isdir(color_dir):
                print(f"[SKIP] Missing color dir: {color_dir}")
                continue
            if not os.path.isdir(seg_dir):
                print(f"[SKIP] Missing segmentation dir: {seg_dir}")
                continue
            if not os.path.isfile(planes_path):
                print(f"[SKIP] Missing planes.npy: {planes_path}")
                continue

            planes = np.load(planes_path, allow_pickle=True)
            has_semantics = os.path.isdir(label_dir)

            seg_indices = set(os.path.splitext(f)[0] for f in os.listdir(seg_dir) if f.endswith('.png'))

            scene_frame_count = 0
            for seg_idx in natsorted(seg_indices):
                rgb_path = os.path.join(color_dir, f'{seg_idx}.jpg')
                depth_path = os.path.join(depth_dir, f'{seg_idx}.png')
                seg_path = os.path.join(seg_dir, f'{seg_idx}.png')

                if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)
                        and os.path.isfile(seg_path)):
                    continue

                sem_path = os.path.join(label_dir, f'{seg_idx}.png') if has_semantics else None

                self.valid_pairs.append((rgb_path, depth_path, seg_path, sem_path,
                                         len(planes), scene_id, seg_idx))
                scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} matched frames")

        self.scene_ids = valid_scene_ids
        print(f"[ScanNet] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path, seg_path, sem_path, num_planes, scene_id, frame_idx = self.valid_pairs[idx]

        # --- Depth (determines H, W) ---
        try:
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise IOError(f"Cannot read depth: {depth_path}")
            depth = depth_raw.astype(np.float32) / 1000.0
            H, W = depth.shape
        except Exception as e:
            print(f"[WARN] Failed depth from {depth_path}: {e}")
            H, W = self.image_height, self.image_width
            depth = np.zeros((H, W), dtype=np.float32)

        # --- Plane label (binary: planar / non-planar) ---
        try:
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg_img is None:
                raise IOError(f"Cannot read segmentation: {seg_path}")
            raw_ids = _decode_scannet_segmentation(seg_img)
            remapped = _remap_plane_ids(raw_ids, num_planes, self.plane_area_threshold)
            plane = (remapped > 0).astype(np.float32)
        except Exception as e:
            print(f"[WARN] Failed segmentation from {seg_path}: {e}")
            plane = np.zeros((H, W), dtype=np.float32)

        if plane.shape != (H, W):
            plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)

        plane = torch.from_numpy(plane).unsqueeze(0)  # [1, H, W]
        depth = torch.from_numpy(depth).unsqueeze(0)   # [1, H, W]

        # --- Semantic label ---
        if sem_path and os.path.isfile(sem_path):
            try:
                sem = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
                sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
                sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)
            except Exception as e:
                print(f"[WARN] Failed semantic label from {sem_path}: {e}")
                sem = torch.zeros((1, H, W), dtype=torch.int64)
        else:
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- RGB ---
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
            "sem": sem,
            "rgb_path": rgb_path,
            "scene_id": scene_id,
            "frame_idx": frame_idx,
        }


class ScanNetPlaneDataset(Dataset):
    """ScanNet dataset with plane instance segmentation, intrinsics, and poses.

    Matches ScanNetPPPlaneDataset interface.
    Returns: image, depth, plane (int32 instance IDs), sem, rgb_path, K, c2w, scene_id, frame_idx.
    """

    def __init__(self,
                 data_root,
                 split_txt,
                 split='train',
                 image_height=480,
                 image_width=640,
                 max_scenes=None,
                 plane_area_threshold=500):
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.plane_area_threshold = plane_area_threshold

        with open(split_txt, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        self.valid_pairs = []
        valid_scene_ids = []

        for scene_id in scene_ids:
            scene_dir = os.path.join(data_root, 'scans', scene_id)
            color_dir = os.path.join(scene_dir, 'frames', 'color')
            depth_dir = os.path.join(scene_dir, 'frames', 'depth')
            pose_dir = os.path.join(scene_dir, 'frames', 'pose')
            seg_dir = os.path.join(scene_dir, 'annotation', 'segmentation')
            planes_path = os.path.join(scene_dir, 'annotation', 'planes.npy')
            label_dir = os.path.join(scene_dir, 'label-filt')

            if not os.path.isdir(color_dir):
                print(f"[SKIP] Missing color dir: {color_dir}")
                continue
            if not os.path.isdir(seg_dir):
                print(f"[SKIP] Missing segmentation dir: {seg_dir}")
                continue
            if not os.path.isfile(planes_path):
                print(f"[SKIP] Missing planes.npy: {planes_path}")
                continue

            K = _load_intrinsic(scene_dir, scene_id)
            if K is None:
                print(f"[SKIP] No intrinsics found for {scene_id}")
                continue

            planes = np.load(planes_path, allow_pickle=True)
            has_semantics = os.path.isdir(label_dir)

            seg_indices = set(os.path.splitext(f)[0] for f in os.listdir(seg_dir) if f.endswith('.png'))

            scene_frame_count = 0
            for seg_idx in natsorted(seg_indices):
                rgb_path = os.path.join(color_dir, f'{seg_idx}.jpg')
                depth_path = os.path.join(depth_dir, f'{seg_idx}.png')
                pose_path = os.path.join(pose_dir, f'{seg_idx}.txt')
                seg_path = os.path.join(seg_dir, f'{seg_idx}.png')

                if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)
                        and os.path.isfile(pose_path) and os.path.isfile(seg_path)):
                    continue

                sem_path = os.path.join(label_dir, f'{seg_idx}.png') if has_semantics else None

                self.valid_pairs.append((rgb_path, depth_path, pose_path, seg_path, sem_path,
                                         K.copy(), len(planes), scene_id, seg_idx))
                scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} matched frames")

        self.scene_ids = valid_scene_ids
        print(f"[ScanNet] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        (rgb_path, depth_path, pose_path, seg_path, sem_path,
         K, num_planes, scene_id, frame_idx) = self.valid_pairs[idx]

        # --- Depth (determines H, W) ---
        try:
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise IOError(f"Cannot read depth: {depth_path}")
            depth = depth_raw.astype(np.float32) / 1000.0
            H, W = depth.shape
        except Exception as e:
            print(f"[WARN] Failed depth from {depth_path}: {e}")
            H, W = self.image_height, self.image_width
            depth = np.zeros((H, W), dtype=np.float32)

        # --- Plane label (instance IDs: 0 = non-planar, 1..M = planes) ---
        try:
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            if seg_img is None:
                raise IOError(f"Cannot read segmentation: {seg_path}")
            raw_ids = _decode_scannet_segmentation(seg_img)
            plane = _remap_plane_ids(raw_ids, num_planes, self.plane_area_threshold)
        except Exception as e:
            print(f"[WARN] Failed segmentation from {seg_path}: {e}")
            plane = np.zeros((H, W), dtype=np.int32)

        if plane.shape != (H, W):
            plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)

        plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)  # [1, H, W]
        depth = torch.from_numpy(depth).unsqueeze(0)                    # [1, H, W]

        # --- Semantic label ---
        if sem_path and os.path.isfile(sem_path):
            try:
                sem = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
                sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
                sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)
            except Exception as e:
                print(f"[WARN] Failed semantic label from {sem_path}: {e}")
                sem = torch.zeros((1, H, W), dtype=torch.int64)
        else:
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- Camera pose (c2w) ---
        try:
            c2w = np.loadtxt(pose_path, dtype=np.float32)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                c2w = np.eye(4, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Failed pose from {pose_path}: {e}")
            c2w = np.eye(4, dtype=np.float32)

        # --- RGB ---
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
            "sem": sem,
            "rgb_path": rgb_path,
            "K": torch.from_numpy(K),
            "c2w": torch.from_numpy(c2w),
            "scene_id": scene_id,
            "frame_idx": frame_idx,
        }
