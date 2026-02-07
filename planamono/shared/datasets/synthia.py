import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# SYNTHIA-AL semantic classes
PLANAR = {1, 2, 3, 20}  # Ground, Sidewalk, Building, Lanemarking

CLASS_NAMES = {
    0: 'Void', 1: 'Ground', 2: 'Sidewalk', 3: 'Building',
    4: 'Sidewalk2', 5: 'Fence', 6: 'Pole', 7: 'TrafficLight',
    8: 'Car', 9: 'Vegetation', 10: 'Unknown', 11: 'Sky',
    12: 'Human', 14: 'Unknown2', 20: 'Lanemarking',
    21: 'Human', 25: 'Cyclist',
}

# Camera intrinsics
FX = FY = 895.692
CX, CY = 320.0, 240.0


def decode_depth_synthia(depth_png):
    """Decode SYNTHIA RGB-encoded depth to meters."""
    R = depth_png[:, :, 0].astype(np.float64)
    G = depth_png[:, :, 1].astype(np.float64)
    B = depth_png[:, :, 2].astype(np.float64)
    return (5000.0 * (R + G * 256 + B * 256 * 256) / (256**3 - 1)).astype(np.float32)


class SYNTHIAPlanarityDataset(Dataset):
    """
    SYNTHIA-AL dataset for planarity learning.

    Reads RGB from original SYNTHIA RGBA PNGs, depth from RGB-encoded PNGs,
    semantic segmentation from channel 0 of SemSeg PNGs, and plane labels
    from pre-computed H5 files.

    Args:
        data_root: Root of SYNTHIA test/ directory
        plane_label_root: Root of pre-computed plane H5 files
        split_file: Text file listing scene names (one per line)
        split: 'train', 'val', or 'test'
        image_height, image_width: Resize dimensions
    """

    def __init__(self,
                 data_root,
                 plane_label_root,
                 split_file,
                 split='train',
                 image_height=480,
                 image_width=640,
                 max_samples=None):
        self.data_root = data_root
        self.plane_label_root = plane_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load split: each line is a scene directory name
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        self.valid_pairs = []

        for scene_name in scene_names:
            scene_dir = os.path.join(data_root, scene_name)
            plane_h5 = os.path.join(plane_label_root, scene_name, "planes.h5")

            if not os.path.isdir(scene_dir):
                print(f"[SKIP] Missing scene dir: {scene_dir}")
                continue
            if not os.path.exists(plane_h5):
                print(f"[SKIP] Missing plane H5: {plane_h5}")
                continue

            # Find the timestamp subdirectory containing RGB/Depth/SemSeg
            timestamp_dirs = self._find_timestamp_dirs(scene_dir)
            if not timestamp_dirs:
                print(f"[SKIP] No data found in: {scene_dir}")
                continue

            # Read frame IDs from H5
            try:
                with h5py.File(plane_h5, 'r') as hf:
                    frame_ids = [fid.decode('utf-8') for fid in hf['frame_ids'][:]]
            except Exception as e:
                print(f"[SKIP] Error reading {plane_h5}: {e}")
                continue

            # Build frame_id -> file path mapping from all timestamp dirs
            frame_map = {}
            for ts_dir in timestamp_dirs:
                rgb_dir = os.path.join(ts_dir, "RGB")
                depth_dir = os.path.join(ts_dir, "Depth")
                seg_dir = os.path.join(ts_dir, "SemSeg")

                if not os.path.isdir(rgb_dir):
                    continue

                for fname in os.listdir(rgb_dir):
                    if not fname.endswith(".png"):
                        continue
                    fid = os.path.splitext(fname)[0]
                    rgb_path = os.path.join(rgb_dir, fname)
                    depth_path = os.path.join(depth_dir, fname)
                    seg_path = os.path.join(seg_dir, fname)

                    if os.path.exists(depth_path) and os.path.exists(seg_path):
                        frame_map[fid] = (rgb_path, depth_path, seg_path)

            for idx, fid in enumerate(frame_ids):
                if fid not in frame_map:
                    continue
                rgb_path, depth_path, seg_path = frame_map[fid]

                self.valid_pairs.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'seg_path': seg_path,
                    'plane_h5': plane_h5,
                    'frame_idx': idx,
                    'frame_id': fid,
                    'scene_name': scene_name,
                })

            print(f"[DEBUG] {scene_name}: {len(frame_ids)} frames in H5, "
                  f"{sum(1 for fid in frame_ids if fid in frame_map)} matched")

        if max_samples is not None and len(self.valid_pairs) > max_samples:
            indices = np.linspace(0, len(self.valid_pairs) - 1, max_samples, dtype=int)
            self.valid_pairs = [self.valid_pairs[i] for i in indices]

        print(f"[SYNTHIA] {split} split -> {len(self.valid_pairs)} pairs")

    @staticmethod
    def _find_timestamp_dirs(scene_dir):
        """Find timestamp subdirectories that contain RGB/Depth/SemSeg."""
        result = []
        # Check if scene_dir itself has RGB/
        if os.path.isdir(os.path.join(scene_dir, "RGB")):
            result.append(scene_dir)
            return result

        # Look for DD-MM-YYYY_HH-MM-SS style subdirectories
        for d in sorted(os.listdir(scene_dir)):
            sub = os.path.join(scene_dir, d)
            if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "RGB")):
                result.append(sub)
        return result

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        entry = self.valid_pairs[idx]
        H, W = self.image_height, self.image_width

        # --- Plane label ---
        try:
            with h5py.File(entry['plane_h5'], 'r') as hf:
                plane = hf['planes'][entry['frame_idx']]
            plane = (plane > 0).astype(np.float32)
            plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)
            plane = torch.tensor(plane.copy(), dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed plane label: {e}")
            plane = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Depth ---
        try:
            depth_png = cv2.imread(entry['depth_path'], cv2.IMREAD_UNCHANGED)
            depth = decode_depth_synthia(depth_png)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depth = torch.tensor(depth.copy(), dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed depth: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic segmentation ---
        try:
            semseg = cv2.imread(entry['seg_path'], cv2.IMREAD_UNCHANGED)
            class_ids = semseg[:, :, 0].astype(np.int64)  # channel 0 = class ID
            class_ids = cv2.resize(class_ids.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int64)
            sem = torch.tensor(class_ids.copy(), dtype=torch.int64).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed semantic: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- RGB ---
        image = cv2.imread(entry['rgb_path'], cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read RGB: {entry['rgb_path']}")
        # SYNTHIA has RGBA, take first 3 channels
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "scene_id": entry['scene_name'],
            "frame_idx": int(entry['frame_id']) if entry['frame_id'].isdigit() else hash(entry['frame_id']) % 100000,
        }
