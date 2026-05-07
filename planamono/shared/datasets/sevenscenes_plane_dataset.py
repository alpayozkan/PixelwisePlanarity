"""
7-Scenes plane dataset for plane segmentation evaluation.

Dataset structure (flat, single val split):
    data_root/
        sevenscenes_plane_len758_val.json   - COCO-style metadata
        val_0_d2.npz, val_1_d2.npz, ..., val_757_d2.npz

NPZ contents (identical to ParallelDomain "_d2" format):
    image:              (192, 256, 3) uint8, BGR
    raw_image:          (480, 640, 3) uint8, BGR
    depth:              (1, 192, 256) float64
    raw_depth:          (192, 256)    float64
    high_res_depth:     (1, 480, 640) float64
    high_res_raw_depth: (480, 640)    float64
    segmentation:       (192, 256)    int64, plane IDs 0..N-1, label 20 = non-planar
    plane:              (N, 3)        float64, n/d format (n^T x = 1)
    num_planes:         scalar        int64
    intrinsic:          (3, 3)        int64, at 256x192 resolution
    origin_img_path:    scalar str    e.g. ".../<scene>/seq-XX/frame-NNNNNN.color.png"

Scene names are extracted from origin_img_path: chess, fire, heads, office,
pumpkin, redkitchen, stairs.

Returns the same dict format as ScanNetPPPlaneDataset / PDPlaneDataset:
    image:     (3, H, W) float32 [0, 1]
    depth:     (1, H, W) float32 meters
    plane:     (1, H, W) int32   (0 = non-planar, 1..N = plane instances)
    sem:       (1, H, W) int64   (zeros - no semantic labels)
    K:         (3, 3)   float32
    c2w:       (4, 4)   float32  (identity - no poses)
    rgb_path:  str      "<scene>/<seq>/<frame>" parsed from origin_img_path
    scene_id:  str      scene name (chess/fire/.../stairs) or "unknown"
    frame_idx: str      sample index as string
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Original non-planar label in 7-Scenes / ZeroPlane "_d2" NPZ format
_NON_PLANAR = 20

# Canonical 7-Scenes scene names (used to robustly extract scene from path)
_SEVEN_SCENES = {
    "chess", "fire", "heads", "office",
    "pumpkin", "redkitchen", "stairs",
}


class SevenScenesPlaneDataset(Dataset):
    """7-Scenes plane dataset (ZeroPlane benchmark variant).

    Per-sample NPZ files with RGB (BGR), depth, segmentation, plane parameters,
    and intrinsics. Uses high-res (480x640) by default.

    Labels are remapped from the original convention (planes=0..N-1,
    non-planar=20) to the standard convention (non-planar=0, planes=1..N).

    ``scene_id`` is parsed from the NPZ ``origin_img_path`` (one of the seven
    canonical scene names) and falls back to ``"unknown"`` if no match.
    """

    def __init__(
        self,
        data_root,
        split="val",
        image_height=480,
        image_width=640,
        max_samples=None,
    ):
        """
        Args:
            data_root: Directory containing NPZ files and JSON metadata.
            split: Split name; only "val" is published.
            image_height: Target height (default 480 = high-res native).
            image_width: Target width (default 640 = high-res native).
            max_samples: Limit number of samples (None = all).
        """
        self.data_root = data_root
        self.split = split
        self.image_height = image_height
        self.image_width = image_width

        all_files = os.listdir(data_root)
        npz_candidates = sorted(
            [f for f in all_files
             if f.startswith(f"{split}_") and f.endswith("_d2.npz")],
            key=lambda f: int(f.split("_")[1]),
        )
        if not npz_candidates:
            raise FileNotFoundError(
                f"No {split}_*_d2.npz files found in {data_root}"
            )

        # Filter out 0-byte / unreadable NPZ files (the published 7-Scenes
        # release ships with several corrupted samples).
        npz_files = []
        n_skipped = 0
        for f in npz_candidates:
            if os.path.getsize(os.path.join(data_root, f)) == 0:
                n_skipped += 1
                continue
            npz_files.append(f)
        if n_skipped:
            print(f"[7Scenes] WARNING: skipped {n_skipped} zero-byte NPZ files "
                  f"(out of {len(npz_candidates)})")

        if max_samples is not None:
            npz_files = npz_files[:max_samples]

        # Optional COCO-style JSON metadata. The JSON is used for scene
        # extraction (much faster than opening every NPZ).
        json_name = None
        for fname in all_files:
            if fname.endswith(f"_{split}.json"):
                json_name = fname
                break
        self._json_path = os.path.join(data_root, json_name) if json_name else None
        self._json_cache = None

        # Build sample_idx -> origin_img_path map from JSON (if present).
        idx_to_origin = {}
        meta = self._load_json()
        if meta is not None:
            for ann in meta.get("annotations", []):
                # image_id like "0_d2" -> idx 0
                stem = str(ann.get("image_id", "")).split("_")[0]
                if stem.isdigit():
                    idx_to_origin[int(stem)] = ann.get("origin_img_path", "")

        # Read intrinsics from first valid sample (constant across dataset)
        first = np.load(os.path.join(data_root, npz_files[0]), allow_pickle=True)
        self._K_native = first["intrinsic"].astype(np.float32)  # at 256x192
        self._native_h, self._native_w = 192, 256
        self._hires_h, self._hires_w = 480, 640

        # Build valid_pairs: (npz_path, sample_idx, scene_id, origin_img_path)
        self.valid_pairs = []
        scene_ids_seen = set()

        for npz_file in npz_files:
            idx = int(npz_file.split("_")[1])
            npz_path = os.path.join(data_root, npz_file)

            # Prefer JSON-derived origin (avoids loading NPZ at init).
            origin = idx_to_origin.get(idx)
            if not origin:
                # Fallback: open NPZ. May raise on rare unexpected corruption.
                d = np.load(npz_path, allow_pickle=True)
                origin = str(d["origin_img_path"])

            scene = self._extract_scene(origin)
            scene_ids_seen.add(scene)
            self.valid_pairs.append((npz_path, idx, scene, origin))

        self.scene_ids = sorted(scene_ids_seen)
        print(
            f"[7Scenes] {split} split -> {len(self.valid_pairs)} samples "
            f"from {len(self.scene_ids)} scenes: {self.scene_ids}"
        )

    @staticmethod
    def _extract_scene(origin_img_path):
        """Extract scene name from origin_img_path string.

        Path looks like ``.../sevenscenes/<scene>/seq-XX/frame-NNNNNN.color.png``;
        we match on canonical 7-Scenes names to be robust to absolute-path drift.
        """
        for part in origin_img_path.split("/"):
            if part in _SEVEN_SCENES:
                return part
        return "unknown"

    @staticmethod
    def _parse_seq_frame(origin_img_path):
        """Parse ``"<seq>/<frame>"`` (e.g. ``"seq-03/frame-000498.color.png"``)
        from origin_img_path. Returns ``"unknown"`` if not parsable.
        """
        parts = origin_img_path.rstrip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return "unknown"

    def _load_json(self):
        """Lazy-load and cache JSON metadata."""
        if self._json_cache is None and self._json_path is not None:
            with open(self._json_path, "r") as f:
                self._json_cache = json.load(f)
        return self._json_cache

    @staticmethod
    def _remap_labels(seg):
        """Remap labels (planes=0..N-1, non-planar=20) to standard
        convention (non-planar=0, planes=1..N). Returns int32 array.
        """
        out = np.empty_like(seg, dtype=np.int32)
        non_planar_mask = seg == _NON_PLANAR
        out[~non_planar_mask] = seg[~non_planar_mask] + 1   # 0..N-1 -> 1..N
        out[non_planar_mask] = 0
        return out

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        npz_path, sample_idx, scene_id, origin = self.valid_pairs[idx]
        d = np.load(npz_path, allow_pickle=True)

        H, W = self.image_height, self.image_width
        use_hires = (H >= self._hires_h) or (W >= self._hires_w)

        # --- RGB (BGR -> RGB) ---
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
        seg_raw = d["segmentation"]  # (192, 256), always at low-res
        if H != self._native_h or W != self._native_w:
            seg_raw = cv2.resize(
                seg_raw.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
        plane = torch.from_numpy(self._remap_labels(seg_raw)).unsqueeze(0)  # (1, H, W)

        # --- Semantic labels (not available - zeros) ---
        sem = torch.zeros(1, H, W, dtype=torch.int64)

        # --- Intrinsics (scale from native 256x192 to target) ---
        K = self._K_native.copy()
        K[0, :] *= W / self._native_w
        K[1, :] *= H / self._native_h

        # --- Pose (not available in NPZ - identity) ---
        c2w = np.eye(4, dtype=np.float32)

        seq_frame = self._parse_seq_frame(origin)
        return {
            "image": image,
            "depth": depth,
            "plane": plane,
            "sem": sem,
            "K": torch.from_numpy(K),
            "c2w": torch.from_numpy(c2w),
            "rgb_path": f"{scene_id}/{seq_frame}",
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

        If JSON metadata is available, also includes bbox / area / center.
        """
        npz_path, sample_idx, _scene, _origin = self.valid_pairs[idx]
        d = np.load(npz_path, allow_pickle=True)

        plane_params = d["plane"]                           # (N, 3) n/d format
        num_planes = int(np.asarray(d["num_planes"]).flat[0])
        seg = d["segmentation"]

        json_info = {}
        meta = self._load_json()
        if meta is not None and sample_idx < len(meta.get("annotations", [])):
            annot = meta["annotations"][sample_idx]
            for s_info in annot.get("segments_info", []):
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
                "plane_id": orig_id + 1,
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
