"""
Parallel Domain plane dataset for plane segmentation evaluation.

Dataset structure (flat, no scene hierarchy):
    data_root/
        parallel_domain_plane_len356_train.json   — COCO-style metadata
        parallel_domain_plane_len356_val.json
        train_{0..N}_d2.npz                       — per-sample data
        val_{0..N}_d2.npz

NPZ contents per sample:
    image:              (192, 256, 3) uint8, BGR
    raw_image:          (480, 640, 3) uint8, BGR
    depth:              (1, 192, 256) float64
    raw_depth:          (192, 256)    float32
    high_res_depth:     (1, 480, 640) float64
    high_res_raw_depth: (480, 640)    float32
    segmentation:       (192, 256)    int64, plane IDs 0..N-1, label 20 = non-planar
    plane:              (N, 3)        float64, n/d format (n^T x = 1)
    num_planes:         scalar        int64
    intrinsic:          (3, 3)        float64, at 256x192 resolution
    origin_img_path:    scalar        str

Label convention:
    Original:  planes = 0..N-1, non-planar = 20
    Remapped:  non-planar = 0, planes = 1..N  (standard convention)

Returns the same dict format as ScanNetPPPlaneDataset / HypersimPlaneDataset:
    image:     (3, H, W) float32 [0, 1]
    depth:     (1, H, W) float32 meters
    plane:     (1, H, W) int32   (0 = non-planar, 1..N = plane instances)
    sem:       (1, H, W) int64   (zeros — no semantic labels available)
    K:         (3, 3)   float32
    c2w:       (4, 4)   float32  (identity — no poses available)
    rgb_path:  str      "pd/<split>_<idx>"
    scene_id:  str      scene name from origin_img_path
    frame_idx: str      sample index as string
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Original non-planar label in PD dataset (same as ZeroPlane convention)
_PD_NON_PLANAR = 20


class PDPlaneDataset(Dataset):
    """Parallel Domain plane dataset for plane segmentation evaluation.

    Per-sample NPZ files with RGB (BGR), depth, segmentation, plane
    parameters, and intrinsics.  Uses high-res (480x640) by default.

    Labels are remapped from the original convention (planes=0..N-1,
    non-planar=20) to the standard convention (non-planar=0, planes=1..N).
    """

    def __init__(
        self,
        data_root,
        split="train",
        image_height=480,
        image_width=640,
        max_samples=None,
    ):
        """
        Args:
            data_root: Directory containing NPZ files and JSON metadata.
            split: 'train' or 'val'.
            image_height: Target height (default 480 = high-res native).
            image_width: Target width (default 640 = high-res native).
            max_samples: Limit number of samples (None = all).
        """
        self.data_root = data_root
        self.split = split
        self.image_height = image_height
        self.image_width = image_width

        # Discover NPZ files
        npz_files = sorted(
            [f for f in os.listdir(data_root)
             if f.startswith(f"{split}_") and f.endswith("_d2.npz")],
            key=lambda f: int(f.split("_")[1]),
        )
        if not npz_files:
            raise FileNotFoundError(
                f"No {split}_*_d2.npz files found in {data_root}"
            )

        if max_samples is not None:
            npz_files = npz_files[:max_samples]

        # Load JSON metadata (optional — used by get_plane_metadata)
        json_name = None
        for fname in os.listdir(data_root):
            if fname.endswith(f"_{split}.json"):
                json_name = fname
                break
        self._json_path = os.path.join(data_root, json_name) if json_name else None
        self._json_cache = None

        # Read intrinsics from first sample (constant across dataset)
        first = np.load(os.path.join(data_root, npz_files[0]), allow_pickle=True)
        K_native = first["intrinsic"].astype(np.float32)  # at 256x192
        self._K_native = K_native
        self._native_h, self._native_w = 192, 256
        self._hires_h, self._hires_w = 480, 640

        # Build valid_pairs: (npz_path, sample_idx)
        self.valid_pairs = []
        scene_ids_seen = set()

        for npz_file in npz_files:
            idx = int(npz_file.split("_")[1])
            npz_path = os.path.join(data_root, npz_file)
            self.valid_pairs.append((npz_path, idx))

            # Extract scene from origin_img_path for scene_ids
            d = np.load(npz_path, allow_pickle=True)
            origin = str(d["origin_img_path"])
            scene = self._extract_scene(origin)
            scene_ids_seen.add(scene)

        self.scene_ids = sorted(scene_ids_seen)
        print(
            f"[PD] {split} split → {len(self.valid_pairs)} samples "
            f"from {len(self.scene_ids)} scenes"
        )

    @staticmethod
    def _extract_scene(origin_img_path):
        """Extract scene name from origin_img_path string."""
        for part in origin_img_path.split("/"):
            if part.startswith("scene_"):
                return part
        return "unknown"

    def _load_json(self):
        """Lazy-load and cache JSON metadata."""
        if self._json_cache is None and self._json_path is not None:
            with open(self._json_path, "r") as f:
                self._json_cache = json.load(f)
        return self._json_cache

    @staticmethod
    def _remap_labels(seg):
        """Remap PD labels (planes=0..N-1, non-planar=20) to standard
        convention (non-planar=0, planes=1..N).

        Returns int32 array.
        """
        out = np.empty_like(seg, dtype=np.int32)
        non_planar_mask = seg == _PD_NON_PLANAR
        out[~non_planar_mask] = seg[~non_planar_mask] + 1   # 0..N-1 → 1..N
        out[non_planar_mask] = 0
        return out

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        npz_path, sample_idx = self.valid_pairs[idx]
        d = np.load(npz_path, allow_pickle=True)

        H, W = self.image_height, self.image_width
        use_hires = (H >= self._hires_h) or (W >= self._hires_w)

        # --- RGB (BGR → RGB) ---
        if use_hires:
            rgb_raw = d["raw_image"][:, :, ::-1].copy()      # (480, 640, 3)
        else:
            rgb_raw = d["image"][:, :, ::-1].copy()           # (192, 256, 3)

        H_src, W_src = rgb_raw.shape[:2]
        need_resize = (H != H_src) or (W != W_src)

        if need_resize:
            rgb_raw = cv2.resize(rgb_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(
            rgb_raw.astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # --- Depth ---
        if use_hires:
            depth_raw = d["high_res_raw_depth"].astype(np.float32)  # (480, 640)
        else:
            depth_raw = d["raw_depth"].astype(np.float32)           # (192, 256)

        if need_resize:
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = torch.from_numpy(depth_raw).unsqueeze(0)  # (1, H, W)

        # --- Plane labels (remap to standard convention) ---
        seg_raw = d["segmentation"]  # (192, 256) int64, always at low-res
        if H != self._native_h or W != self._native_w:
            seg_raw = cv2.resize(
                seg_raw.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
        plane = torch.from_numpy(self._remap_labels(seg_raw)).unsqueeze(0)  # (1, H, W)

        # --- Semantic labels (not available — zeros) ---
        sem = torch.zeros(1, H, W, dtype=torch.int64)

        # --- Intrinsics (scale from native 256x192 to target) ---
        K = self._K_native.copy()
        K[0, :] *= W / self._native_w
        K[1, :] *= H / self._native_h

        # --- Pose (not available — identity) ---
        c2w = np.eye(4, dtype=np.float32)

        # --- Metadata ---
        origin = str(d["origin_img_path"])
        scene_id = self._extract_scene(origin)

        return {
            "image": image,                           # (3, H, W) float32 [0, 1]
            "depth": depth,                           # (1, H, W) float32 meters
            "plane": plane,                           # (1, H, W) int32, 0 = non-planar
            "sem": sem,                               # (1, H, W) int64
            "K": torch.from_numpy(K),                 # (3, 3) float32
            "c2w": torch.from_numpy(c2w),             # (4, 4) float32
            "rgb_path": f"pd/{self.split}_{sample_idx}",
            "scene_id": scene_id,
            "frame_idx": str(sample_idx),
        }

    def get_plane_metadata(self, idx):
        """Return per-plane metadata for a sample, decomposed from n/d format.

        Returns a list of dicts with keys:
            plane_id:   int (remapped: 1..N, matching output labels)
            n:          list[float] (unit normal, 3-vec)
            d:          float (plane distance in meters)
            nd:         list[float] (raw n/d params, 3-vec)
            num_pixels: int (at native 192x256 resolution)

        If JSON metadata is available, also includes:
            bbox:   [x, y, w, h]
            area:   int
            center: [cx, cy] (normalized)
        """
        npz_path, sample_idx = self.valid_pairs[idx]
        d = np.load(npz_path, allow_pickle=True)

        plane_params = d["plane"]           # (N, 3) n/d format
        num_planes = d["num_planes"].item()
        seg = d["segmentation"]

        # Load JSON segments_info if available
        json_info = {}
        meta = self._load_json()
        if meta is not None:
            annot = meta["annotations"][sample_idx]
            for s_info in annot["segments_info"]:
                json_info[s_info["id"]] = s_info

        result = []
        for orig_id in range(num_planes):
            nd = plane_params[orig_id]
            norm = np.linalg.norm(nd)
            if norm < 1e-12:
                continue
            distance = 1.0 / norm
            normal = (nd * distance).tolist()

            entry = {
                "plane_id": orig_id + 1,  # remapped ID (1-indexed)
                "n": normal,
                "d": float(distance),
                "nd": nd.tolist(),
                "num_pixels": int((seg == orig_id).sum()),
            }

            if orig_id in json_info:
                ji = json_info[orig_id]
                entry["bbox"] = ji.get("bbox")
                entry["area"] = ji.get("area")
                entry["center"] = ji.get("center")

            result.append(entry)

        return result
