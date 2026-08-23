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
                 max_scenes=None,
                 require_gt=True):
        # require_gt=False: skip the rendered plane/sem/depth H5 existence checks and
        # read frame IDs from the pose JSON instead of from rendered.h5. Use this for
        # inference-only runs where ground truth is not available. __getitem__ already
        # handles missing GT via try/except and zero-fill.
        self.rgb_root = rgb_root
        self.plane_label_root = plane_label_root
        self.sem_label_root = sem_label_root
        self.depth_label_root = depth_label_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.require_gt = require_gt

        # Load split
        # split_file = os.path.join(split_txt_dir, f"nvs_sem_{split}_with_planes.txt")
        split_file = os.path.join(split_txt_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        self.valid_pairs = []  # (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K)
        valid_scene_ids = []  # Track scenes that actually have valid data

        for scene_id in scene_ids:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            # plane_h5 = os.path.join(plane_label_root, scene_id, "rendered_planes.h5")
            plane_h5 = os.path.join(plane_label_root, scene_id, "rendered.h5")
            sem_h5 = os.path.join(sem_label_root, scene_id, "rendered_sem.h5")
            depth_h5 = os.path.join(depth_label_root, scene_id, "rendered_depth.h5")
            pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")

            # RGB dir + pose JSON are always required
            if not (os.path.isdir(rgb_dir) and os.path.exists(pose_file)):
                print(f"[SKIP] Missing RGB or pose file for scene: {scene_id}")
                continue
            # Rendered GT files are required only when require_gt=True.
            # rendered_sem.h5 is optional — nothing downstream consumes the sem
            # tensor and __getitem__ zero-fills it when the file is absent.
            if require_gt and not (os.path.exists(plane_h5)
                                   and os.path.exists(depth_h5)):
                print(f"[SKIP] Missing rendered GT files for scene: {scene_id}")
                continue

            # Load per-frame intrinsics from JSON
            with open(pose_file, "r") as f:
                pose_data = json.load(f)

            # Frame IDs: prefer rendered.h5 ordering; fall back to pose JSON keys
            frame_ids = None
            if os.path.exists(plane_h5):
                try:
                    with h5py.File(plane_h5, "r") as f:
                        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
                except Exception as e:
                    print(f"[WARN] Error reading frame_ids from {plane_h5}: {e}")
            if frame_ids is None:
                if require_gt:
                    print(f"[SKIP] Could not read frame_ids and require_gt=True for {scene_id}")
                    continue
                frame_ids = natsorted(pose_data.keys())
                print(f"[INFO] Using {len(frame_ids)} frame_ids from pose JSON for {scene_id}")

            # rendered_depth.h5 is indexed positionally against rendered.h5's
            # frame order below — verify the two files were rendered over the
            # same frames (e.g. same --frame_skip), else depth pairs with the
            # wrong frame and the 3D metrics silently degrade.
            if require_gt and frame_ids is not None and os.path.exists(depth_h5):
                try:
                    with h5py.File(depth_h5, "r") as f:
                        depth_fids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]
                    if depth_fids != frame_ids:
                        print(f"[SKIP] {scene_id}: rendered_depth.h5 frame_ids do not "
                              f"match rendered.h5 ({len(depth_fids)} vs {len(frame_ids)} "
                              "frames) — re-render with the same frame_skip")
                        continue
                except KeyError:
                    pass  # legacy depth H5 without frame_ids: keep positional indexing

            scene_frame_count = 0
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
                scene_frame_count += 1

            # Only add to valid_scene_ids if we got at least one valid frame
            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} matched frames")

        # Update scene_ids to only include valid scenes
        self.scene_ids = valid_scene_ids
        print(f"[ScanNet++] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w = self.valid_pairs[idx]

        if not self.require_gt:
            # Inference-only: skip GT reads entirely (avoids per-frame [WARN] spam and h5py overhead).
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.float32)
            sem = torch.zeros((1, H, W), dtype=torch.int64)
            depth = torch.zeros((1, H, W), dtype=torch.float32)
        else:
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

            # --- Semantic label (optional; zero-filled when the file is absent) ---
            if os.path.exists(sem_h5):
                try:
                    with h5py.File(sem_h5, "r") as f:
                        sem = f["sem"][frame_idx]
                    sem = torch.from_numpy(sem.astype(np.int64)).unsqueeze(0)  # [1, H, W]
                except Exception as e:
                    print(f"[WARN] Failed semantic label from {sem_h5} [{frame_idx}]: {e}")
                    sem = torch.zeros((1, H, W), dtype=torch.int64)
            else:
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
        # return image, depth, plane, sem, rgb_path, torch.from_numpy(K), torch.from_numpy(c2w)
        fid = os.path.splitext(os.path.basename(rgb_path))[0]
        scene_id = rgb_path.split("/")[-4]

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "sem": sem,
            "rgb_path": rgb_path,
            "K": torch.from_numpy(K),
            "c2w": torch.from_numpy(c2w),
            "scene_id": scene_id,
            "frame_idx": fid,
        }

