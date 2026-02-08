import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# VKITTI2 semantic classes
CLASS_MAP = {
    (0, 0, 0):       ('undefined', 0),
    (210, 0, 200):   ('terrain', 1),
    (90, 200, 255):  ('sky', 2),
    (0, 199, 0):     ('tree', 3),
    (90, 240, 0):    ('vegetation', 4),
    (140, 140, 140): ('building', 5),
    (100, 60, 100):  ('road', 6),
    (250, 100, 255): ('guard_rail', 7),
    (255, 255, 0):   ('traffic_sign', 8),
    (200, 200, 0):   ('traffic_light', 9),
    (255, 130, 0):   ('pole', 10),
    (80, 80, 80):    ('misc', 11),
    (160, 60, 60):   ('truck', 12),
    (255, 127, 80):  ('car', 13),
    (0, 139, 139):   ('van', 14),
}

CLASS_NAMES = {v[1]: v[0] for v in CLASS_MAP.values()}

PLANAR = {5, 6, 8}  # building, road, traffic_sign

# Camera intrinsics
FX = FY = 725.0087
CX, CY = 620.5, 187.0


class VKITTI2PlanarityDataset(Dataset):
    """
    VKITTI2 dataset for planarity learning.

    Reads all data from self-contained scene_data.h5 files generated
    by the gt_creation pipeline.

    Args:
        data_root: Root directory containing SceneXX/variant/scene_data.h5
        image_height, image_width: Resize dimensions
        max_samples: Limit total samples
    """

    def __init__(self,
                 data_root,
                 image_height=512,
                 image_width=768,
                 max_samples=None):
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width

        # Discover all scene_data.h5 files
        self.entries = []
        for scene in sorted(os.listdir(data_root)):
            scene_dir = os.path.join(data_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for variant in sorted(os.listdir(scene_dir)):
                h5_path = os.path.join(scene_dir, variant, "scene_data.h5")
                if not os.path.exists(h5_path):
                    continue
                try:
                    with h5py.File(h5_path, 'r') as hf:
                        num_frames = hf.attrs['num_frames']
                except Exception as e:
                    print(f"[SKIP] Error reading {h5_path}: {e}")
                    continue

                for fi in range(num_frames):
                    self.entries.append({
                        'h5_path': h5_path,
                        'frame_idx': fi,
                        'scene': scene,
                        'variant': variant,
                    })

        if max_samples is not None and len(self.entries) > max_samples:
            indices = np.linspace(0, len(self.entries) - 1, max_samples, dtype=int)
            self.entries = [self.entries[i] for i in indices]

        print(f"[VKITTI2] {len(self.entries)} samples from {data_root}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        H, W = self.image_height, self.image_width
        fi = entry['frame_idx']

        with h5py.File(entry['h5_path'], 'r') as hf:
            rgb = hf['rgb'][fi]
            depth = hf['depth'][fi]
            semantic = hf['semantic'][fi]
            plane = hf['planes'][fi]

        # Plane: binary planarity mask
        plane = (plane > 0).astype(np.float32)
        plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)
        plane = torch.tensor(plane.copy(), dtype=torch.float32).unsqueeze(0)

        # Depth
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = torch.tensor(depth.copy(), dtype=torch.float32).unsqueeze(0)

        # Semantic
        sem = cv2.resize(semantic.astype(np.float32), (W, H),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        sem = torch.tensor(sem.copy(), dtype=torch.int64).unsqueeze(0)

        # RGB
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "scene_id": f"{entry['scene']}/{entry['variant']}",
            "frame_idx": fi,
        }
