"""
Hypersim dataset for plane segmentation evaluation - V2

Adapted for original Hypersim dataset format (not merged files).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from natsort import natsorted
import pandas as pd


class HypersimPlaneDataset(Dataset):
    """
    Hypersim dataset for plane segmentation evaluation.

    Works with original Hypersim dataset format:
    - Individual HDF5 files per frame for RGB, depth, semantic
    - Camera parameters in HDF5 files
    - Plane labels in rendered HDF5 per camera

    Expected directory structure:
        hypersim_root/  (your "Hypersim_merged")
            <scene_id>/
                images/
                    scene_cam_00_final_hdf5/
                        frame.XXXX.color.hdf5
                    scene_cam_00_geometry_hdf5/
                        frame.XXXX.depth_meters.hdf5
                        frame.XXXX.semantic.hdf5
                    scene_cam_01_final_hdf5/
                        ...
        plane_label_root/
            <scene_id>/
                rendered_planes_cam_00.h5
                rendered_planes_cam_01.h5
        params_root/
            <scene_id>/
                _detail/
                    cam_00/
                        camera_keyframe_positions.hdf5
                        camera_keyframe_orientations.hdf5

    Args:
        hypersim_root: Root directory of Hypersim dataset
        plane_label_root: Root directory containing plane labels
        params_root: Root directory containing camera parameters
        split_txt_dir: Directory containing split files
        split: 'train', 'val', or 'test'
        metadata_csv: Path to metadata_camera_parameters.csv (optional, for intrinsics)
        image_height: Target image height
        image_width: Target image width
        max_scenes: Maximum number of scenes to load
    """
    def __init__(self,
                 hypersim_root,
                 plane_label_root,
                 params_root,
                 split_txt_dir,
                 split='train',
                 metadata_csv=None,
                 image_height=768,
                 image_width=1024,
                 max_scenes=None):
        self.hypersim_root = hypersim_root
        self.plane_label_root = plane_label_root
        self.params_root = params_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split

        # Load metadata for intrinsics if available
        self.metadata = None
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv, index_col="scene_name")

        # Load split file
        split_file = os.path.join(split_txt_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        scene_ids = natsorted(scene_ids)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        # Build valid pairs
        self.valid_pairs = []
        valid_scene_ids = []

        for scene_id in scene_ids:
            scene_dir = os.path.join(hypersim_root, scene_id)
            images_dir = os.path.join(scene_dir, "images")
            plane_scene_dir = os.path.join(plane_label_root, scene_id)
            params_scene_dir = os.path.join(params_root, scene_id, "_detail")

            if not os.path.exists(images_dir):
                print(f"[SKIP] Missing images dir: {images_dir}")
                continue
            if not os.path.exists(plane_scene_dir):
                print(f"[SKIP] Missing plane dir: {plane_scene_dir}")
                continue

            # Find all plane HDF5 files (one per camera)
            plane_files = [f for f in os.listdir(plane_scene_dir)
                          if f.startswith("rendered_planes_") and f.endswith(".h5")]

            if len(plane_files) == 0:
                print(f"[SKIP] No plane files in {plane_scene_dir}")
                continue

            scene_frame_count = 0
            for plane_file in plane_files:
                # Extract camera name: rendered_planes_cam_00.h5 -> cam_00
                cam_name = plane_file.replace("rendered_planes_", "").replace(".h5", "")
                plane_h5_path = os.path.join(plane_scene_dir, plane_file)

                # Check camera directories exist
                rgb_dir = os.path.join(images_dir, f"scene_{cam_name}_final_hdf5")
                depth_dir = os.path.join(images_dir, f"scene_{cam_name}_geometry_hdf5")

                if not os.path.exists(rgb_dir):
                    print(f"[WARN] Missing RGB dir for {scene_id}/{cam_name}: {rgb_dir}")
                    continue
                if not os.path.exists(depth_dir):
                    print(f"[WARN] Missing depth dir for {scene_id}/{cam_name}: {depth_dir}")
                    continue

                # Get intrinsics
                try:
                    K = self._get_intrinsics(scene_id, cam_name, params_scene_dir)
                except Exception as e:
                    print(f"[WARN] Failed to get intrinsics for {scene_id}/{cam_name}: {e}")
                    continue

                # Read frame_ids from plane HDF5
                try:
                    with h5py.File(plane_h5_path, "r") as f:
                        frame_ids = [fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                                    for fid in f["frame_ids"][:]]
                except Exception as e:
                    print(f"[SKIP] Error reading frame_ids from {plane_h5_path}: {e}")
                    continue

                # Add valid pairs
                for idx, fid in enumerate(frame_ids):
                    rgb_path = os.path.join(rgb_dir, f"frame.{fid}.color.hdf5")
                    depth_path = os.path.join(depth_dir, f"frame.{fid}.depth_meters.hdf5")

                    if not os.path.exists(rgb_path):
                        continue
                    if not os.path.exists(depth_path):
                        continue

                    self.valid_pairs.append((
                        scene_id, cam_name, idx, fid,
                        rgb_path, depth_path, plane_h5_path, K
                    ))
                    scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} frames")

        self.scene_ids = valid_scene_ids
        print(f"[Hypersim] {split} split → {len(self.valid_pairs)} pairs from {len(self.scene_ids)} scenes")

    def _get_intrinsics(self, scene_id, cam_name, params_scene_dir):
        """Compute intrinsics matrix from metadata or use default."""
        # If we have metadata CSV
        if self.metadata is not None and scene_id in self.metadata.index:
            row = self.metadata.loc[scene_id]
            width = self.image_width
            height = self.image_height
            M_proj = np.array([[row[f"M_proj_{i}{j}"] for j in range(4)] for i in range(4)])

            # Convert projection matrix to intrinsics
            fx = M_proj[0, 0] * 0.5 * width
            fy = -M_proj[1, 1] * 0.5 * height
            cx = M_proj[0, 2] * 0.5 * width + 0.5 * width
            cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return K

        # Otherwise use default intrinsics (standard Hypersim)
        # These are typical values for Hypersim at 1024x768
        fx = fy = 886.81  # Approximate focal length for Hypersim
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        scene_id, cam_name, frame_idx, fid, rgb_path, depth_path, plane_h5, K = self.valid_pairs[idx]

        # --- Load plane labels ---
        try:
            with h5py.File(plane_h5, "r") as f:
                plane = f["planes"][frame_idx]
            plane[plane < 0] = 0
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(0)  # [1, H, W]
            H, W = plane.shape[1:]
        except Exception as e:
            print(f"[WARN] Failed plane label from {plane_h5} [{frame_idx}]: {e}")
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.int32)

        # --- Load RGB ---
        try:
            with h5py.File(rgb_path, "r") as f:
                key = list(f.keys())[0]  # Usually dataset name
                rgb = f[key][:]  # (H, W, 3)

            # Handle different dtypes
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            elif rgb.dtype == np.uint16:
                rgb = rgb.astype(np.float32) / 65535.0
            elif rgb.dtype in [np.float16, np.float32, np.float64]:
                # Hypersim HDR images - apply robust tone mapping
                rgb = self._tonemap_rgb_robust(rgb)
            else:
                # Unknown dtype, try tone mapping
                rgb = self._tonemap_rgb_robust(rgb.astype(np.float32))

            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            image = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        except Exception as e:
            print(f"[WARN] Failed RGB from {rgb_path}: {e}")
            import traceback
            traceback.print_exc()
            image = torch.zeros((3, H, W), dtype=torch.float32)

        # --- Load depth ---
        try:
            with h5py.File(depth_path, "r") as f:
                key = list(f.keys())[0]
                depth = f[key][:].astype(np.float32)  # Already in meters
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed depth from {depth_path}: {e}")
            depth = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Semantic labels (optional) ---
        sem = torch.zeros((1, H, W), dtype=torch.int64)

        # --- Camera pose (set to identity if not available) ---
        c2w = np.eye(4, dtype=np.float32)

        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "sem": sem,
            "rgb_path": f"{scene_id}/{cam_name}/{fid}",
            "K": torch.from_numpy(K),
            "c2w": torch.from_numpy(c2w),
            "scene_id": scene_id,
            "frame_idx": fid,
        }

    def _tonemap_rgb_robust(self, hdr, gamma=2.2):
        """
        Apply robust tone mapping for Hypersim HDR images.
        Uses simple normalization without percentiles to avoid overflow.
        """
        # Replace inf and nan
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure non-negative
        hdr = np.maximum(hdr, 0.0)

        # Find max value for normalization (use median of top values to avoid outliers)
        max_val = np.max(hdr)
        if max_val > 0:
            # Sort and take 99th percentile as max to avoid extreme outliers
            flat = hdr.flatten()
            flat_sorted = np.sort(flat[flat > 0])
            if len(flat_sorted) > 0:
                idx_99 = min(int(len(flat_sorted) * 0.99), len(flat_sorted) - 1)
                max_val = flat_sorted[idx_99]
                max_val = max(max_val, 1e-6)  # Avoid division by zero
            else:
                max_val = 1.0
        else:
            max_val = 1.0

        # Normalize to [0, 1]
        img = hdr / max_val
        img = np.clip(img, 0.0, 1.0)

        # Apply gamma correction
        img = img ** (1.0 / gamma)

        # Final safety
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        return img
