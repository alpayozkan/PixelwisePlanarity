"""Add extra method visualizations to existing qualitative samples.

Reads the existing sample filenames from the qualitative output directory,
loads plane labels from H5 files, and saves segmentation + inlier PNGs
into the same directory structure under the new method name.

Supported methods:
  - planercnn_gt:    PlaneRCNN GT pipeline  (rendered.h5,    480x640)
  - planercnn_gt_v1: PlaneRCNN GT pipeline v1
  - planar_recon:    PlanarReconstruction   (planes.h5,      192x256)
  - planeTR:         PlaneTR (lowres)       (rendered_v2.h5, 480x640, nonplanar=20)

Only ScanNet++ is supported.

Usage:
    python add_planercnn_gt_qualitative.py                          # all methods
    python add_planercnn_gt_qualitative.py --methods planercnn_gt   # just one
    python add_planercnn_gt_qualitative.py --methods planeTR
    python add_planercnn_gt_qualitative.py --methods planeTR planar_recon
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
import h5py
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("/cluster/scratch/aoezkan/planeseg/eval/qualitative")

# ScanNet++ GT for loading depth + K + c2w (needed for inlier computation)
SCANNETPP_RGB_ROOT = "/cluster/project/cvg/Shared_datasets/scannet++/data"
SCANNETPP_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

INDOOR_THRESHOLD = 0.01  # 1.0cm (matches generate_qualitative_report.py)
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

# Method definitions: h5_root, h5_filename, method_key
# nonplanar_label: if set, pixels with this label are remapped to 0 (background) before
#                  colorization and inlier computation.
EXTRA_METHODS = {
    "planercnn_gt": {
        "h5_root": Path("/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn"),
        "h5_filename": "rendered.h5",
        "display_name": "PlaneRCNN GT",
        "nonplanar_label": None,
    },
    "planercnn_gt_v1": {
        "h5_root": Path("/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn_v1"),
        "h5_filename": "rendered.h5",
        "display_name": "PlaneRCNN GT v1",
        "nonplanar_label": None,
    },
    "planar_recon": {
        "h5_root": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference/planar_recon_h5"),
        "h5_filename": "planes.h5",
        "display_name": "PlanarReconstruction",
        "nonplanar_label": None,
    },
    "planeTR": {
        # PlaneTR lowres: 480x640, label 20 = non-planar (same convention as ZeroPlane)
        "h5_root": Path("/cluster/scratch/ayavuz/dataset/planrectr_lowres/scannetpp"),
        "h5_filename": "rendered_v2.h5",
        "display_name": "PlaneTR",
        "nonplanar_label": 20,
    },
}

# ============================================================
# HELPERS (copied from generate_qualitative_report.py)
# ============================================================

def get_random_cmap(num_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    colors = rng.randint(40, 240, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors


def overlay_inliers(rgb: np.ndarray, inlier_mask: np.ndarray,
                    outlier_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if inlier_mask.any():
        out[inlier_mask] = out[inlier_mask] * (1 - alpha) + np.array([0, 200, 0], dtype=np.float32) * alpha
    if outlier_mask.any():
        out[outlier_mask] = out[outlier_mask] * (1 - alpha) + np.array([200, 0, 0], dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_inlier_mask(
    plane_seg: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    distance_threshold: float,
    inlier_ratio_gate: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    H, W = depth.shape
    inlier_mask = np.zeros((H, W), dtype=bool)
    outlier_mask = np.zeros((H, W), dtype=bool)

    pts_world, labels, valid_idx = backproject(depth, K, c2w, plane_seg)
    if pts_world.shape[0] == 0:
        return inlier_mask, outlier_mask, {"precision": 0, "recall": 0, "num_inliers": 0}

    plane_results, _ = fit_planes_per_label_v1(
        pts_world, labels,
        distance_threshold=distance_threshold,
        num_iterations=RANSAC_ITERATIONS,
    )

    total_inliers = 0
    total_planar = 0

    for plane_id, result in plane_results.items():
        if plane_id == 0:
            continue
        plane_pts_mask = labels == plane_id
        n_pts = plane_pts_mask.sum()
        if n_pts == 0:
            continue

        total_planar += n_pts
        plane_model = result.get("plane_model_refined", result.get("plane_model"))
        if plane_model is None:
            pixel_idx = valid_idx[plane_pts_mask]
            rows, cols = np.unravel_index(pixel_idx, (H, W))
            outlier_mask[rows, cols] = True
            continue

        a, b, c, d = plane_model
        pts_plane = pts_world[plane_pts_mask]
        distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
        is_inlier = distances < distance_threshold

        inlier_ratio = is_inlier.sum() / n_pts
        pixel_idx = valid_idx[plane_pts_mask]
        rows, cols = np.unravel_index(pixel_idx, (H, W))

        if inlier_ratio >= inlier_ratio_gate:
            inlier_mask[rows[is_inlier], cols[is_inlier]] = True
            outlier_mask[rows[~is_inlier], cols[~is_inlier]] = True
            total_inliers += is_inlier.sum()
        else:
            outlier_mask[rows, cols] = True

    total_pts = (depth > 0).sum()
    precision = total_inliers / total_planar if total_planar > 0 else 0
    recall = total_inliers / total_pts if total_pts > 0 else 0

    return inlier_mask, outlier_mask, {
        "precision": precision, "recall": recall, "num_inliers": int(total_inliers),
    }


# ============================================================
# H5 LOADER
# ============================================================

class H5SceneLoader:
    """Load plane labels from per-scene H5 files (planes/frame_ids format)."""

    def __init__(self, root: Path, h5_filename: str = "planes.h5"):
        self.root = root
        self.h5_filename = h5_filename
        self._cache_scene = None
        self._cache_fids = None
        self._cache_planes = None

    def load(self, scene_id: str, frame_idx: str) -> Optional[np.ndarray]:
        if self._cache_scene != scene_id:
            h5_path = self.root / scene_id / self.h5_filename
            if not h5_path.exists():
                return None
            with h5py.File(str(h5_path), "r") as f:
                self._cache_fids = [
                    fid.decode() if isinstance(fid, bytes) else str(fid)
                    for fid in f["frame_ids"][:]
                ]
                self._cache_planes = f["planes"][:]
            self._cache_scene = scene_id

        # Try exact match first
        try:
            idx = self._cache_fids.index(frame_idx)
        except ValueError:
            # Try stripping zeros
            for i, fid in enumerate(self._cache_fids):
                if fid.lstrip("0") == frame_idx.lstrip("0") or fid == frame_idx:
                    idx = i
                    break
            else:
                return None

        return self._cache_planes[idx].astype(np.int32)


# ============================================================
# MAIN
# ============================================================

def parse_sample_filename(fname: str) -> Tuple[str, str]:
    """Extract scene_id and frame_idx from sample filename.

    Format: 000_bde1e479ad_frame_005650.png -> (bde1e479ad, frame_005650)
    """
    stem = Path(fname).stem
    # Pattern: NNN_<scene_id>_frame_NNNNNN
    m = re.match(r"\d+_([a-f0-9]+)_(frame_\d+)", stem)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse sample filename: {fname}")


def run_scannetpp(output_dir: Path, threshold: float, method_keys: list):
    from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
    from planamono.paths import repo_path

    ds_dir = output_dir / "scannetpp"
    seg_dir = ds_dir / "segmentation"
    inlier_dir = ds_dir / "inliers"

    # Discover existing samples from the rgb folder
    rgb_dir = seg_dir / "rgb"
    if not rgb_dir.exists():
        print(f"[ERROR] No existing samples found at {rgb_dir}")
        return

    sample_files = sorted(rgb_dir.glob("*.png"))
    print(f"Found {len(sample_files)} existing samples")

    # Parse scene/frame from filenames
    samples = []
    for f in sample_files:
        scene_id, frame_idx = parse_sample_filename(f.name)
        samples.append((f.stem, scene_id, frame_idx))

    # Load dataset for depth/K/c2w
    dataset = ScanNetPPPlaneDataset(
        rgb_root=SCANNETPP_RGB_ROOT,
        plane_label_root=SCANNETPP_LABEL_ROOT,
        sem_label_root=SCANNETPP_LABEL_ROOT,
        depth_label_root=SCANNETPP_LABEL_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
    )
    print(f"Dataset: {len(dataset)} frames")

    # Build scene+frame -> dataset index lookup
    frame_lookup = {}
    for idx in range(len(dataset)):
        sample = dataset.valid_pairs[idx]
        rgb_path = sample[0]
        s_id = rgb_path.split("/")[-4]
        f_idx = os.path.splitext(os.path.basename(rgb_path))[0]
        frame_lookup[(s_id, f_idx)] = idx

    colors = get_random_cmap(256)

    # Process each method
    for method_key in method_keys:
        cfg = EXTRA_METHODS[method_key]
        print(f"\n--- {cfg['display_name']} ({method_key}) ---")
        print(f"  H5 root: {cfg['h5_root']}")
        print(f"  H5 file: {cfg['h5_filename']}")

        loader = H5SceneLoader(cfg["h5_root"], cfg["h5_filename"])

        (seg_dir / method_key).mkdir(parents=True, exist_ok=True)
        (inlier_dir / method_key).mkdir(parents=True, exist_ok=True)

        saved = 0
        for prefix, scene_id, frame_idx in tqdm(samples, desc=cfg["display_name"]):
            # Load plane labels
            plane_seg = loader.load(scene_id, frame_idx)
            if plane_seg is None:
                print(f"  [SKIP] {scene_id}/{frame_idx}: missing")
                continue

            # Remap non-planar label to 0 if needed (e.g. PlaneTR uses 20)
            nonplanar_label = cfg.get("nonplanar_label")
            if nonplanar_label is not None:
                plane_seg[plane_seg == nonplanar_label] = 0

            # Find dataset index for depth/K/c2w
            ds_idx = frame_lookup.get((scene_id, frame_idx))
            if ds_idx is None:
                print(f"  [SKIP] {scene_id}/{frame_idx}: not in dataset")
                continue

            sample = dataset[ds_idx]
            rgb = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            depth = sample["depth"].squeeze(0).numpy()
            K = sample["K"].numpy()
            c2w = sample["c2w"].numpy()

            # Resize to match depth resolution if needed
            if plane_seg.shape != depth.shape:
                plane_seg = cv2.resize(
                    plane_seg, (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # --- Segmentation PNG ---
            seg_clipped = np.clip(plane_seg, 0, len(colors) - 1)
            seg_img = colors[seg_clipped]
            Image.fromarray(seg_img).save(str(seg_dir / method_key / f"{prefix}.png"))

            # --- Inlier overlay PNG ---
            inlier, outlier, stats = compute_inlier_mask(
                plane_seg, depth, K, c2w, threshold, INLIER_RATIO_GATE)
            overlay = overlay_inliers(rgb, inlier, outlier)
            p, r = stats["precision"], stats["recall"]
            Image.fromarray(overlay).save(
                str(inlier_dir / method_key / f"{prefix}_P{p:.2f}_R{r:.2f}.png"))

            saved += 1

        print(f"Saved {saved}/{len(samples)} {cfg['display_name']} samples")
        print(f"  segmentation -> {seg_dir / method_key}")
        print(f"  inliers      -> {inlier_dir / method_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Add extra method visuals to existing qualitative samples")
    parser.add_argument("--methods", nargs="+", default=list(EXTRA_METHODS.keys()),
                        choices=list(EXTRA_METHODS.keys()),
                        help=f"Methods to add (default: all = {list(EXTRA_METHODS.keys())})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    run_scannetpp(output_dir, INDOOR_THRESHOLD, args.methods)


if __name__ == "__main__":
    main()
