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


def load_scene_list(split_file):
    """Load scene names from a split text file (one name per line)."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


class SYNTHIAPlanarityDataset(Dataset):
    """
    SYNTHIA-AL dataset for planarity learning.

    Reads from pre-computed scene_data.h5 files that contain RGB, depth,
    semantics, and plane labels all in one file per scene.

    H5 structure (per scene):
        rgb:       (N, H, W, 3) uint8
        depth:     (N, H, W) float32 meters
        semantic:  (N, H, W) uint8 class IDs
        planes:    (N, H, W) uint16 plane IDs (0=non-planar)
        frame_ids: (N,) string

    Args:
        data_root: Path to synthia/train or synthia/test (raw data, used to discover scene names)
        plane_label_root: Root of pre-computed H5 files (scene_data.h5 per scene)
        split: Label for logging ('train' or 'test')
        image_height, image_width: Resize dimensions
        max_samples: Limit number of samples (uniform subsampling)
        scene_list: Optional list of scene names to include. If provided,
                    overrides scanning data_root for scene discovery.
    """

    def __init__(self,
                 data_root,
                 plane_label_root,
                 split='train',
                 image_height=476,
                 image_width=644,
                 max_samples=None,
                 scene_list=None):
        self.plane_label_root = plane_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Discover scenes from data_root or use provided list
        if scene_list is not None:
            scene_names = sorted(scene_list)
        else:
            scene_names = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d)) and d.startswith('test5_')
            ])

        self.valid_pairs = []

        for scene_name in scene_names:
            h5_path = os.path.join(plane_label_root, scene_name, "scene_data.h5")

            if not os.path.exists(h5_path):
                continue

            try:
                with h5py.File(h5_path, 'r') as hf:
                    num_frames = hf.attrs['num_frames']
                    frame_ids = [fid.decode('utf-8') for fid in hf['frame_ids'][:]]
            except Exception as e:
                print(f"[SKIP] Error reading {h5_path}: {e}")
                continue

            for idx in range(num_frames):
                self.valid_pairs.append({
                    'h5_path': h5_path,
                    'frame_idx': idx,
                    'frame_id': frame_ids[idx],
                    'scene_name': scene_name,
                })

        if max_samples is not None and len(self.valid_pairs) > max_samples:
            indices = np.linspace(0, len(self.valid_pairs) - 1, max_samples, dtype=int)
            self.valid_pairs = [self.valid_pairs[i] for i in indices]

        print(f"[SYNTHIA] {split} split -> {len(self.valid_pairs)} pairs from {len(scene_names)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        entry = self.valid_pairs[idx]
        H, W = self.image_height, self.image_width
        fi = entry['frame_idx']

        with h5py.File(entry['h5_path'], 'r') as hf:
            rgb = hf['rgb'][fi]          # (h, w, 3) uint8
            depth = hf['depth'][fi]      # (h, w) float32
            semantic = hf['semantic'][fi] # (h, w) uint8
            plane = hf['planes'][fi]     # (h, w) uint16

        # Resize
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        semantic = cv2.resize(semantic.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
        plane = cv2.resize(plane.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
        depth = torch.tensor(depth.copy(), dtype=torch.float32).unsqueeze(0)       # (1, H, W)
        semantic = torch.tensor(semantic.copy(), dtype=torch.int64).unsqueeze(0)    # (1, H, W)
        plane = torch.tensor((plane > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": semantic,
            "scene_id": entry['scene_name'],
            "frame_idx": int(entry['frame_id']) if entry['frame_id'].isdigit() else hash(entry['frame_id']) % 100000,
        }
