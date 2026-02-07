import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# VKITTI2 semantic classes: RGB color -> (name, class_id)
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


def rgb_to_class_ids(seg_rgb):
    """Convert RGB segmentation image to integer class IDs."""
    class_ids = np.full(seg_rgb.shape[:2], -1, dtype=np.int32)
    for color, (name, cid) in CLASS_MAP.items():
        mask = np.all(seg_rgb == color, axis=-1)
        class_ids[mask] = cid
    return class_ids


class VKITTI2PlanarityDataset(Dataset):
    """
    VKITTI2 dataset for planarity learning.

    Reads RGB from original VKITTI2 images, depth from 16-bit PNGs,
    semantic segmentation from RGB-encoded PNGs, and plane labels
    from pre-computed H5 files.

    Args:
        rgb_root: Root of vkitti_2.0.3_rgb/
        depth_root: Root of vkitti_2.0.3_depth/
        semseg_root: Root of vkitti_2.0.3_classSegmentation/
        plane_label_root: Root of pre-computed plane H5 files
        split_file: Text file listing scene/variant pairs (one per line, e.g. "Scene01/clone")
        split: 'train', 'val', or 'test'
        image_height, image_width: Resize dimensions
        camera: Camera index (0=left, 1=right)
    """

    def __init__(self,
                 rgb_root,
                 depth_root,
                 semseg_root,
                 plane_label_root,
                 split_file,
                 split='train',
                 image_height=375,
                 image_width=1242,
                 camera=0,
                 max_samples=None):
        self.rgb_root = rgb_root
        self.depth_root = depth_root
        self.semseg_root = semseg_root
        self.plane_label_root = plane_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.camera = camera
        self.split = split

        # Load split: each line is "SceneXX/variant"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_variants = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        self.valid_pairs = []

        for sv in scene_variants:
            scene, variant = sv.split('/')
            cam_str = f"Camera_{camera}"

            rgb_dir = os.path.join(rgb_root, scene, variant, "frames", "rgb", cam_str)
            depth_dir = os.path.join(depth_root, scene, variant, "frames", "depth", cam_str)
            seg_dir = os.path.join(semseg_root, scene, variant, "frames",
                                   "classSegmentation", cam_str)
            plane_h5 = os.path.join(plane_label_root, scene, variant, "planes.h5")

            if not os.path.isdir(rgb_dir):
                print(f"[SKIP] Missing RGB dir: {rgb_dir}")
                continue
            if not os.path.exists(plane_h5):
                print(f"[SKIP] Missing plane H5: {plane_h5}")
                continue

            # Read frame IDs from H5
            try:
                with h5py.File(plane_h5, 'r') as hf:
                    frame_ids = [fid.decode('utf-8') for fid in hf['frame_ids'][:]]
            except Exception as e:
                print(f"[SKIP] Error reading {plane_h5}: {e}")
                continue

            for idx, fid in enumerate(frame_ids):
                rgb_path = os.path.join(rgb_dir, f"rgb_{fid}.jpg")
                depth_path = os.path.join(depth_dir, f"depth_{fid}.png")
                seg_path = os.path.join(seg_dir, f"classgt_{fid}.png")

                if not (os.path.exists(rgb_path) and os.path.exists(depth_path)
                        and os.path.exists(seg_path)):
                    continue

                self.valid_pairs.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'seg_path': seg_path,
                    'plane_h5': plane_h5,
                    'frame_idx': idx,
                    'frame_id': fid,
                    'scene': scene,
                    'variant': variant,
                })

            print(f"[DEBUG] {scene}/{variant}: {len(frame_ids)} frames")

        if max_samples is not None and len(self.valid_pairs) > max_samples:
            indices = np.linspace(0, len(self.valid_pairs) - 1, max_samples, dtype=int)
            self.valid_pairs = [self.valid_pairs[i] for i in indices]

        print(f"[VKITTI2] {split} split -> {len(self.valid_pairs)} pairs")

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
            depth = depth_png.astype(np.float32) / 100.0  # cm -> m
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depth = torch.tensor(depth.copy(), dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed depth: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic segmentation ---
        try:
            seg_rgb = cv2.imread(entry['seg_path'])
            seg_rgb = cv2.cvtColor(seg_rgb, cv2.COLOR_BGR2RGB)
            seg_rgb = cv2.resize(seg_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
            class_ids = rgb_to_class_ids(seg_rgb)
            sem = torch.tensor(class_ids.copy(), dtype=torch.int64).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Failed semantic: {e}")
            sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- RGB ---
        image = cv2.imread(entry['rgb_path'])
        if image is None:
            raise RuntimeError(f"Cannot read RGB: {entry['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "semantic": sem,
            "scene_id": f"{entry['scene']}/{entry['variant']}",
            "frame_idx": int(entry['frame_id']),
        }
