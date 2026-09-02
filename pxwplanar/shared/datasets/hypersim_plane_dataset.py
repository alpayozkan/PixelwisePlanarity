"""
Hypersim dataset for plane segmentation evaluation - V2

Adapted for original Hypersim dataset format (not merged files).
"""

import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch.utils.data import Dataset


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
        metadata_csv: Path to metadata_camera_parameters.csv
            (optional, for intrinsics)
        image_height: Target image height
        image_width: Target image width
        max_scenes: Maximum number of scenes to load
    """

    def __init__(
        self,
        hypersim_root,
        plane_label_root,
        params_root,
        split_txt_dir,
        split="train",
        metadata_csv=None,
        image_height=768,
        image_width=1024,
        max_scenes=None,
        use_raycasted_depth=False,
    ):
        """
        Args:
            use_raycasted_depth: Controls depth source.
                False (default): Use original V-Ray depth_meters.hdf5
                    (Euclidean, needs conversion).
                True or "zdepth": Use raycasted z-depth from *_raycast/
                    (no conversion needed).
                "euclidean": Use raycasted Euclidean depth from *_raycast_euc/
                    (no conversion needed, use with backproject_mcam).
        """
        self.hypersim_root = hypersim_root
        self.plane_label_root = plane_label_root
        self.params_root = params_root
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.use_raycasted_depth = use_raycasted_depth

        # Load metadata for intrinsics if available
        self.metadata = None
        if metadata_csv is None:
            metadata_csv = os.path.join(
                os.path.dirname(__file__), "metadata_camera_parameters.csv"
            )
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv, index_col="scene_name")

        # Load split file
        split_file = os.path.join(split_txt_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with open(split_file) as f:
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
            plane_files = [
                f
                for f in os.listdir(plane_scene_dir)
                if f.startswith("rendered_planes_") and f.endswith(".h5")
            ]

            if len(plane_files) == 0:
                print(f"[SKIP] No plane files in {plane_scene_dir}")
                continue

            scene_frame_count = 0
            for plane_file in plane_files:
                # Extract camera name: rendered_planes_cam_00.h5 -> cam_00
                cam_name = plane_file.replace("rendered_planes_", "").replace(
                    ".h5", ""
                )
                plane_h5_path = os.path.join(plane_scene_dir, plane_file)

                # Check camera directories exist
                rgb_dir = os.path.join(
                    images_dir, f"scene_{cam_name}_final_hdf5"
                )
                depth_dir = os.path.join(
                    images_dir, f"scene_{cam_name}_geometry_hdf5"
                )

                if not os.path.exists(rgb_dir):
                    print(
                        f"[WARN] Missing RGB dir for "
                        f"{scene_id}/{cam_name}: {rgb_dir}"
                    )
                    continue
                if not os.path.exists(depth_dir):
                    print(
                        f"[WARN] Missing depth dir for "
                        f"{scene_id}/{cam_name}: {depth_dir}"
                    )
                    continue

                # Get intrinsics (at native render resolution)
                try:
                    K, native_wh = self._get_intrinsics(
                        scene_id, cam_name, params_scene_dir
                    )
                except Exception as e:
                    print(
                        f"[WARN] Failed to get intrinsics for "
                        f"{scene_id}/{cam_name}: {e}"
                    )
                    continue

                # Read frame_ids from plane HDF5
                try:
                    with h5py.File(plane_h5_path, "r") as f:
                        frame_ids = [
                            fid.decode("utf-8")
                            if isinstance(fid, bytes)
                            else str(fid)
                            for fid in f["frame_ids"][:]
                        ]
                except Exception as e:
                    print(
                        f"[SKIP] Error reading frame_ids from "
                        f"{plane_h5_path}: {e}"
                    )
                    continue

                # Add valid pairs
                for idx, fid in enumerate(frame_ids):
                    rgb_path = os.path.join(rgb_dir, f"frame.{fid}.color.hdf5")
                    depth_path = os.path.join(
                        depth_dir, f"frame.{fid}.depth_meters.hdf5"
                    )

                    if not os.path.exists(rgb_path):
                        continue
                    if not os.path.exists(depth_path):
                        continue

                    self.valid_pairs.append(
                        (
                            scene_id,
                            cam_name,
                            idx,
                            fid,
                            rgb_path,
                            depth_path,
                            plane_h5_path,
                            K,
                            native_wh,
                        )
                    )
                    scene_frame_count += 1

            if scene_frame_count > 0:
                valid_scene_ids.append(scene_id)
                print(f"[DEBUG] Scene {scene_id} → {scene_frame_count} frames")

        self.scene_ids = valid_scene_ids
        print(
            f"[Hypersim] {split} split → {len(self.valid_pairs)} pairs "
            f"from {len(self.scene_ids)} scenes"
        )

    def _get_intrinsics(self, scene_id, cam_name, params_scene_dir):
        """Compute intrinsics matrix from metadata or use default.

        Intrinsics are computed at native Hypersim render resolution (typically
        1024x768).  In __getitem__ the K matrix is rescaled to match the actual
        plane-label dimensions so that cx/cy always correspond to the image that
        is returned.
        """
        # If we have metadata CSV, use per-scene projection matrix
        if self.metadata is not None and scene_id in self.metadata.index:
            row = self.metadata.loc[scene_id]
            # Use native render resolution from metadata
            native_w = (
                int(row["settings_output_img_width"])
                if "settings_output_img_width" in row.index
                else 1024
            )
            native_h = (
                int(row["settings_output_img_height"])
                if "settings_output_img_height" in row.index
                else 768
            )
            M_proj = np.array(
                [[row[f"M_proj_{i}{j}"] for j in range(4)] for i in range(4)]
            )

            W1 = native_w - 1  # M_screen_from_ndc uses (W-1), not W
            H1 = native_h - 1
            fx = M_proj[0, 0] * 0.5 * W1
            fy = -M_proj[1, 1] * 0.5 * H1
            cx = -M_proj[0, 2] * 0.5 * W1 + 0.5 * W1
            cy = M_proj[1, 2] * 0.5 * H1 + 0.5 * H1
            K = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            return K, (native_w, native_h)

        # Default intrinsics at native Hypersim resolution (1024x768)
        native_w, native_h = 1024, 768
        fx = fy = 886.81
        cx = (native_w - 1) / 2.0  # 511.5
        cy = (native_h - 1) / 2.0  # 383.5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, (native_w, native_h)

    def __len__(self):
        return len(self.valid_pairs)

    def _get_M_cam_from_uv(self, scene_id):
        """Load the 3x3 M_cam_from_uv matrix for a scene from metadata CSV.

        Returns None if metadata is not available.
        """
        if self.metadata is None or scene_id not in self.metadata.index:
            return None
        row = self.metadata.loc[scene_id]
        try:
            return np.array(
                [
                    [row[f"M_cam_from_uv_{i}{j}"] for j in range(3)]
                    for i in range(3)
                ],
                dtype=np.float64,
            )
        except KeyError:
            return None

    @staticmethod
    def _euclidean_to_zdepth(depth_euc, K):
        """Convert Euclidean ray distance to z-depth using pinhole K
        (APPROXIMATE).

        .. warning::

            This is a **pinhole approximation** of V-Ray's camera model.
            Hypersim rays are actually generated by ``M_cam_from_uv`` (a 3x3
            matrix that can have cross-terms and a non-unit z-component).
            The pinhole model assumes every ray has z-component = 1, which
            is not true for M_cam_from_uv.

            The approximation is *self-consistent* with ``backproject/v2``
            (which also use pinhole K), so errors partially cancel.
            However, the resulting 3D points are placed along pinhole ray
            directions instead of the true V-Ray ray directions, causing
            systematic position errors that grow toward image edges
            (~0.5-1 mm at tight thresholds).

            For exact conversion, use ``_euclidean_to_zdepth_mcam()`` instead.

        Conversion: ``Z = depth_euc / sqrt((x_n)^2 + (y_n)^2 + 1)``
        where ``x_n = (u - cx) / fx``, ``y_n = (v - cy) / fy``.
        """
        H, W = depth_euc.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u, v = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        )
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        ray_length = np.sqrt(x_n**2 + y_n**2 + 1.0)
        return depth_euc / ray_length

    @staticmethod
    def _euclidean_to_zdepth_mcam(depth_euc, M_cam_from_uv, native_w, native_h):
        """Convert Euclidean ray distance to z-depth using V-Ray's
        M_cam_from_uv.

        This is the **exact** conversion that matches V-Ray's camera model.

        ``depth_meters.hdf5`` stores the Euclidean distance along the V-Ray ray
        for each pixel.  The V-Ray ray direction in camera space is::

            d_cam = M_cam_from_uv @ [u_ndc, v_ndc, 1]

        The z-depth (perpendicular distance to the image plane) is::

            z = depth_euc * |d_cam.z| / |d_cam|

        This differs from the pinhole approximation because M_cam_from_uv can
        have cross-terms (e.g. row 2 depends on v_ndc), making the z-component
        vary per pixel instead of being a constant 1.

        Args:
            depth_euc: (H, W) Euclidean ray distance in meters.
            M_cam_from_uv: (3, 3) V-Ray camera matrix from metadata CSV.
            native_w: Native render width (for NDC grid computation).
            native_h: Native render height (for NDC grid computation).

        Returns:
            (H, W) float32 z-depth in meters.
        """
        H, W = depth_euc.shape

        # Build NDC grid matching V-Ray pixel sampling (same as render.py).
        # Each pixel center maps to an NDC coordinate in [-1, 1].
        # If depth has been resized (H != native_h), map through native coords.
        u_px = np.arange(W, dtype=np.float64)
        v_px = np.arange(H, dtype=np.float64)

        # Pixel -> native pixel -> NDC
        u_native = u_px * (native_w / W)
        v_native = v_px * (native_h / H)
        u_ndc = (2.0 * u_native + 1.0) / native_w - 1.0
        v_ndc = 1.0 - (2.0 * v_native + 1.0) / native_h

        uu, vv = np.meshgrid(u_ndc, v_ndc)
        uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)

        # Camera-space ray directions via M_cam_from_uv
        dirs_cam = uvs @ M_cam_from_uv.T  # (H, W, 3)

        # cos(theta) = |z_component| / |ray|
        # V-Ray camera looks along -Z, so d_cam.z is typically negative.
        ray_lengths = np.linalg.norm(dirs_cam, axis=-1)  # (H, W)
        z_abs = np.abs(dirs_cam[:, :, 2])  # (H, W)

        cos_theta = np.where(ray_lengths > 0, z_abs / ray_lengths, 1.0)

        return (depth_euc * cos_theta).astype(np.float32)

    def __getitem__(self, idx):
        (
            scene_id,
            cam_name,
            frame_idx,
            fid,
            rgb_path,
            depth_path,
            plane_h5,
            K,
            native_wh,
        ) = self.valid_pairs[idx]

        # --- Load plane labels ---
        try:
            with h5py.File(plane_h5, "r") as f:
                plane = f["planes"][frame_idx]
            plane[plane < 0] = 0
            plane = torch.from_numpy(plane.astype(np.int32)).unsqueeze(
                0
            )  # [1, H, W]
            H, W = plane.shape[1:]
        except Exception as e:
            print(
                f"[WARN] Failed plane label from {plane_h5} [{frame_idx}]: {e}"
            )
            H, W = self.image_height, self.image_width
            plane = torch.zeros((1, H, W), dtype=torch.int32)

        # --- Rescale intrinsics to match actual image dimensions ---
        # K was computed at native render resolution; scale to plane-label dims
        native_w, native_h = native_wh
        if native_w != W or native_h != H:
            scale_x = W / native_w
            scale_y = H / native_h
            K = K.copy()
            K[0, :] *= scale_x  # fx, cx
            K[1, :] *= scale_y  # fy, cy

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
            image = torch.tensor(rgb, dtype=torch.float32).permute(
                2, 0, 1
            )  # [3, H, W]
        except Exception as e:
            print(f"[WARN] Failed RGB from {rgb_path}: {e}")
            import traceback

            traceback.print_exc()
            image = torch.zeros((3, H, W), dtype=torch.float32)

        # --- Load depth ---
        try:
            # Determine depth source based on use_raycasted_depth:
            #   False        → original V-Ray depth (Euclidean, needs
            #                  conversion)
            #   True/zdepth  → raycasted z-depth from *_raycast/ (ready to use)
            #   "euclidean"  → raycasted Euclidean from *_raycast_euc/
            #                  (ready to use)
            actual_depth_path = depth_path
            is_raycasted = False
            urd = self.use_raycasted_depth
            if urd:
                dir_suffix = "raycast_euc" if urd == "euclidean" else "raycast"
                raycast_depth_path = depth_path.replace(
                    f"scene_{cam_name}_geometry_hdf5/",
                    f"scene_{cam_name}_geometry_hdf5_{dir_suffix}/",
                )
                if os.path.exists(raycast_depth_path):
                    actual_depth_path = raycast_depth_path
                    is_raycasted = True

            with h5py.File(actual_depth_path, "r") as f:
                key = list(f.keys())[0]
                depth_raw = f[key][:].astype(np.float32)
            depth_raw = cv2.resize(
                depth_raw, (W, H), interpolation=cv2.INTER_LINEAR
            )

            if is_raycasted:
                # Raycasted depth (both zdepth and euclidean) — use directly
                depth = depth_raw
            else:
                # V-Ray depth is Euclidean — convert to z-depth
                M_cam = self._get_M_cam_from_uv(scene_id)
                if M_cam is not None:
                    depth = self._euclidean_to_zdepth_mcam(
                        depth_raw, M_cam, native_w, native_h
                    )
                else:
                    depth = self._euclidean_to_zdepth(depth_raw, K)
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        except Exception as e:
            print(f"[WARN] Failed depth from {actual_depth_path}: {e}")
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

        # Find max value for normalization (use median of top values to
        # avoid outliers)
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
