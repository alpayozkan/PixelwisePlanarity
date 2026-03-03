#!/usr/bin/env python3
"""
Hypersim Dataset Validity Report
=================================

Checks every component of the Hypersim dataset for every split (train/val/test)
and generates a comprehensive report covering:

1. Scene-level checks:
   - Scene directory exists in Hypersim_merged
   - GT plane label directory exists in Hypersim_rendered
   - Camera parameter directory exists in Hypersim_params

2. Camera-level checks:
   - GT H5 file exists (rendered_planes_cam_XX.h5)
   - RGB directory exists (scene_cam_XX_final_hdf5/)
   - Depth directory exists (scene_cam_XX_geometry_hdf5/)
   - Intrinsics available (from metadata CSV)

3. Frame-level checks:
   - RGB HDF5 file exists on disk
   - Depth HDF5 file exists on disk
   - Plane label readable from GT H5
   - RGB data quality (inf/nan/negative values)
   - Depth data quality (inf/nan/negative/zero values)
   - Plane label quality (all-zero frames, negative labels)
   - Depth range sanity (min/max)

4. Prediction H5 checks (for each inference method):
   - Prediction H5 exists per scene/camera
   - Frame count matches GT
   - Frame IDs match GT

Usage:
    python hypersim_dataset_validity_report.py                     # Full report (all splits)
    python hypersim_dataset_validity_report.py --splits val        # Val split only
    python hypersim_dataset_validity_report.py --splits val test   # Val + test
    python hypersim_dataset_validity_report.py --sample-frames 10  # Sample N frames per camera for data quality
    python hypersim_dataset_validity_report.py --skip-frame-content  # Skip reading frame data (fast mode)
    python hypersim_dataset_validity_report.py --check-predictions   # Also check prediction H5s
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ============================================================
# PATHS (same as evaluate_hypersim_all_baselines.py)
# ============================================================
HYPERSIM_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PLANE_LABEL_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"
# Old paths (buggy plane_id=0 collision in rendered labels)
# HYPERSIM_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
# PLANE_LABEL_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
# PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_params"
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "../../splits/hypersim")
METADATA_CSV = os.path.join(
    os.path.dirname(__file__), "../../shared/datasets/metadata_camera_parameters.csv"
)

# Prediction H5 roots (for --check-predictions)
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/hypersim/inference")
PREDICTION_FOLDERS = {
    "moge_ours": "moge_ours_h5",
    "moge_mixed_bce": "moge_mixed_bce_h5",
    "zeroplane_mixed_dust3r": "zeroplane_mixed_dust3r_h5",
    "zeroplane_mixed": "zeroplane_mixed_h5",
}


def load_split(split: str) -> List[str]:
    """Load scene IDs from a split file."""
    split_file = os.path.join(SPLIT_DIR, f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_metadata() -> Optional[pd.DataFrame]:
    """Load camera metadata CSV."""
    if os.path.exists(METADATA_CSV):
        return pd.read_csv(METADATA_CSV, index_col="scene_name")
    return None


# ============================================================
# SCENE-LEVEL CHECKS
# ============================================================

def check_scene_dirs(scene_id: str) -> Dict:
    """Check scene-level directory existence."""
    scene_dir = os.path.join(HYPERSIM_ROOT, scene_id)
    images_dir = os.path.join(scene_dir, "images")
    plane_dir = os.path.join(PLANE_LABEL_ROOT, scene_id)
    params_dir = os.path.join(PARAMS_ROOT, scene_id, "_detail")

    return {
        "scene_id": scene_id,
        "scene_dir_exists": os.path.isdir(scene_dir),
        "images_dir_exists": os.path.isdir(images_dir),
        "gt_plane_dir_exists": os.path.isdir(plane_dir),
        "params_dir_exists": os.path.isdir(params_dir),
    }


# ============================================================
# CAMERA-LEVEL CHECKS
# ============================================================

def discover_cameras(scene_id: str) -> Dict[str, Dict]:
    """Discover all cameras for a scene from GT plane H5 files."""
    plane_dir = os.path.join(PLANE_LABEL_ROOT, scene_id)
    cameras = {}

    if not os.path.isdir(plane_dir):
        return cameras

    for fname in sorted(os.listdir(plane_dir)):
        if fname.startswith("rendered_planes_") and fname.endswith(".h5"):
            cam_name = fname.replace("rendered_planes_", "").replace(".h5", "")
            cameras[cam_name] = {
                "gt_h5_path": os.path.join(plane_dir, fname),
            }

    return cameras


def check_camera(scene_id: str, cam_name: str, gt_h5_path: str,
                 metadata: Optional[pd.DataFrame]) -> Dict:
    """Check camera-level validity."""
    images_dir = os.path.join(HYPERSIM_ROOT, scene_id, "images")
    rgb_dir = os.path.join(images_dir, f"scene_{cam_name}_final_hdf5")
    depth_dir = os.path.join(images_dir, f"scene_{cam_name}_geometry_hdf5")

    result = {
        "scene_id": scene_id,
        "cam_name": cam_name,
        "gt_h5_exists": os.path.isfile(gt_h5_path),
        "gt_h5_readable": False,
        "gt_n_frames": 0,
        "gt_frame_ids": [],
        "gt_planes_shape": None,
        "rgb_dir_exists": os.path.isdir(rgb_dir),
        "depth_dir_exists": os.path.isdir(depth_dir),
        "intrinsics_available": False,
        "intrinsics_source": "none",
        "native_resolution": None,
    }

    # Check GT H5 readability
    if result["gt_h5_exists"]:
        try:
            with h5py.File(gt_h5_path, "r") as f:
                frame_ids = [
                    fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                    for fid in f["frame_ids"][:]
                ]
                planes_shape = f["planes"].shape
                result["gt_h5_readable"] = True
                result["gt_n_frames"] = len(frame_ids)
                result["gt_frame_ids"] = frame_ids
                result["gt_planes_shape"] = planes_shape
        except Exception as e:
            result["gt_h5_error"] = str(e)

    # Check intrinsics
    if metadata is not None and scene_id in metadata.index:
        row = metadata.loc[scene_id]
        has_proj = all(f"M_proj_{i}{j}" in row.index for i in range(4) for j in range(4))
        if has_proj:
            result["intrinsics_available"] = True
            result["intrinsics_source"] = "metadata_csv"
            native_w = int(row.get("settings_output_img_width", 1024))
            native_h = int(row.get("settings_output_img_height", 768))
            result["native_resolution"] = (native_w, native_h)
    if not result["intrinsics_available"]:
        # Falls back to default in the dataset class
        result["intrinsics_available"] = True
        result["intrinsics_source"] = "default_886.81"
        result["native_resolution"] = (1024, 768)

    return result


# ============================================================
# FRAME-LEVEL CHECKS
# ============================================================

def check_frame_existence(scene_id: str, cam_name: str,
                          frame_ids: List[str]) -> List[Dict]:
    """Check existence of RGB and depth files for each frame."""
    images_dir = os.path.join(HYPERSIM_ROOT, scene_id, "images")
    rgb_dir = os.path.join(images_dir, f"scene_{cam_name}_final_hdf5")
    depth_dir = os.path.join(images_dir, f"scene_{cam_name}_geometry_hdf5")

    results = []
    for fid in frame_ids:
        rgb_path = os.path.join(rgb_dir, f"frame.{fid}.color.hdf5")
        depth_path = os.path.join(depth_dir, f"frame.{fid}.depth_meters.hdf5")
        results.append({
            "scene_id": scene_id,
            "cam_name": cam_name,
            "frame_id": fid,
            "rgb_exists": os.path.isfile(rgb_path),
            "depth_exists": os.path.isfile(depth_path),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
        })
    return results


def check_frame_content(rgb_path: str, depth_path: str, gt_h5_path: str,
                        frame_idx: int) -> Dict:
    """Check data quality of a single frame (reads actual files)."""
    result = {
        "rgb_readable": False,
        "rgb_dtype": None,
        "rgb_shape": None,
        "rgb_has_inf": False,
        "rgb_has_nan": False,
        "rgb_has_negative": False,
        "rgb_pct_bad": 0.0,
        "depth_readable": False,
        "depth_dtype": None,
        "depth_shape": None,
        "depth_has_inf": False,
        "depth_has_nan": False,
        "depth_has_negative": False,
        "depth_has_zero": False,
        "depth_pct_invalid": 0.0,
        "depth_min": None,
        "depth_max": None,
        "depth_median": None,
        "plane_readable": False,
        "plane_shape": None,
        "plane_n_labels": 0,
        "plane_has_negative": False,
        "plane_all_zero": False,
        "plane_pct_planar": 0.0,
    }

    # Check RGB
    if os.path.isfile(rgb_path):
        try:
            with h5py.File(rgb_path, "r") as f:
                key = list(f.keys())[0]
                rgb = f[key][:]
            result["rgb_readable"] = True
            result["rgb_dtype"] = str(rgb.dtype)
            result["rgb_shape"] = rgb.shape

            rgb_f = rgb.astype(np.float64)
            n_pixels = rgb_f.size
            n_inf = np.isinf(rgb_f).sum()
            n_nan = np.isnan(rgb_f).sum()
            n_neg = (rgb_f < 0).sum()
            result["rgb_has_inf"] = bool(n_inf > 0)
            result["rgb_has_nan"] = bool(n_nan > 0)
            result["rgb_has_negative"] = bool(n_neg > 0)
            result["rgb_pct_bad"] = float((n_inf + n_nan + n_neg) / n_pixels * 100)
        except Exception as e:
            result["rgb_error"] = str(e)

    # Check depth
    if os.path.isfile(depth_path):
        try:
            with h5py.File(depth_path, "r") as f:
                key = list(f.keys())[0]
                depth = f[key][:].astype(np.float32)
            result["depth_readable"] = True
            result["depth_dtype"] = str(depth.dtype)
            result["depth_shape"] = depth.shape

            n_pixels = depth.size
            n_inf = np.isinf(depth).sum()
            n_nan = np.isnan(depth).sum()
            n_neg = (depth < 0).sum()
            n_zero = (depth == 0).sum()

            valid_mask = np.isfinite(depth) & (depth > 0)
            result["depth_has_inf"] = bool(n_inf > 0)
            result["depth_has_nan"] = bool(n_nan > 0)
            result["depth_has_negative"] = bool(n_neg > 0)
            result["depth_has_zero"] = bool(n_zero > 0)
            result["depth_pct_invalid"] = float(
                (n_inf + n_nan + n_neg + n_zero) / n_pixels * 100
            )
            if valid_mask.any():
                result["depth_min"] = float(depth[valid_mask].min())
                result["depth_max"] = float(depth[valid_mask].max())
                result["depth_median"] = float(np.median(depth[valid_mask]))
        except Exception as e:
            result["depth_error"] = str(e)

    # Check plane labels
    try:
        with h5py.File(gt_h5_path, "r") as f:
            plane = f["planes"][frame_idx]
        result["plane_readable"] = True
        result["plane_shape"] = plane.shape
        result["plane_n_labels"] = int(len(np.unique(plane[plane > 0])))
        result["plane_has_negative"] = bool((plane < 0).any())
        result["plane_all_zero"] = bool((plane <= 0).all())
        total = plane.size
        planar = (plane > 0).sum()
        result["plane_pct_planar"] = float(planar / total * 100) if total > 0 else 0.0
    except Exception as e:
        result["plane_error"] = str(e)

    return result


# ============================================================
# PREDICTION H5 CHECKS
# ============================================================

def check_predictions(scene_id: str, cam_name: str, gt_frame_ids: List[str],
                      method_name: str, h5_folder: str) -> Dict:
    """Check prediction H5 for a specific method/scene/camera."""
    h5_path = os.path.join(str(H5_ROOT), h5_folder, scene_id, f"planes_{cam_name}.h5")
    result = {
        "method": method_name,
        "scene_id": scene_id,
        "cam_name": cam_name,
        "pred_h5_exists": os.path.isfile(h5_path),
        "pred_h5_readable": False,
        "pred_n_frames": 0,
        "pred_frame_ids_match": False,
        "pred_missing_frames": [],
        "pred_extra_frames": [],
        "pred_planes_shape": None,
    }

    if result["pred_h5_exists"]:
        try:
            with h5py.File(h5_path, "r") as f:
                pred_frame_ids = [
                    fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                    for fid in f["frame_ids"][:]
                ]
                pred_shape = f["planes"].shape
            result["pred_h5_readable"] = True
            result["pred_n_frames"] = len(pred_frame_ids)
            result["pred_planes_shape"] = pred_shape

            gt_set = set(gt_frame_ids)
            pred_set = set(pred_frame_ids)
            result["pred_frame_ids_match"] = (gt_set == pred_set)
            result["pred_missing_frames"] = sorted(gt_set - pred_set)
            result["pred_extra_frames"] = sorted(pred_set - gt_set)
        except Exception as e:
            result["pred_h5_error"] = str(e)

    return result


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(splits: List[str], sample_frames: Optional[int],
                    skip_frame_content: bool, check_preds: bool) -> str:
    """Generate a full validity report."""
    metadata = load_metadata()
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 80)
    out("HYPERSIM DATASET VALIDITY REPORT")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 80)
    out()

    # --- Path verification ---
    out("## PATH VERIFICATION")
    out(f"  HYPERSIM_ROOT:    {HYPERSIM_ROOT}  {'[OK]' if os.path.isdir(HYPERSIM_ROOT) else '[MISSING]'}")
    out(f"  PLANE_LABEL_ROOT: {PLANE_LABEL_ROOT}  {'[OK]' if os.path.isdir(PLANE_LABEL_ROOT) else '[MISSING]'}")
    out(f"  PARAMS_ROOT:      {PARAMS_ROOT}  {'[OK]' if os.path.isdir(PARAMS_ROOT) else '[MISSING]'}")
    out(f"  METADATA_CSV:     {METADATA_CSV}  {'[OK]' if os.path.isfile(METADATA_CSV) else '[MISSING]'}")
    out(f"  SPLIT_DIR:        {SPLIT_DIR}  {'[OK]' if os.path.isdir(SPLIT_DIR) else '[MISSING]'}")
    if check_preds:
        out(f"  H5_ROOT:          {H5_ROOT}  {'[OK]' if H5_ROOT.is_dir() else '[MISSING]'}")
    out()

    # Grand totals
    grand_totals = {
        "scenes": 0,
        "scenes_missing_merged": 0,
        "scenes_missing_gt": 0,
        "scenes_missing_params": 0,
        "cameras": 0,
        "cameras_no_rgb_dir": 0,
        "cameras_no_depth_dir": 0,
        "frames_total": 0,
        "frames_rgb_missing": 0,
        "frames_depth_missing": 0,
        "frames_both_missing": 0,
        "frames_content_checked": 0,
        "frames_rgb_has_bad_pixels": 0,
        "frames_depth_has_invalid": 0,
        "frames_plane_all_zero": 0,
        "frames_plane_has_negative": 0,
    }

    for split in splits:
        out("=" * 80)
        out(f"SPLIT: {split.upper()}")
        out("=" * 80)
        out()

        scene_ids = load_split(split)
        out(f"  Scenes in split file: {len(scene_ids)}")
        out()

        split_totals = defaultdict(int)
        split_totals["n_scenes"] = len(scene_ids)

        # Per-scene issues
        scenes_missing_merged = []
        scenes_missing_gt = []
        scenes_missing_params = []
        scenes_no_intrinsics_csv = []

        # Per-camera issues
        cameras_no_rgb_dir = []
        cameras_no_depth_dir = []
        cameras_gt_unreadable = []

        # Per-frame issues
        frames_rgb_missing = []
        frames_depth_missing = []
        frames_both_missing = []
        frames_rgb_bad = []
        frames_depth_bad = []
        frames_plane_all_zero = []
        frames_plane_negative = []

        # Depth range stats
        depth_mins = []
        depth_maxs = []

        # Plane coverage stats
        plane_pct_planars = []

        for si, scene_id in enumerate(scene_ids):
            print(f"  [{split}] Checking scene {si+1}/{len(scene_ids)}: {scene_id}")

            # --- Scene-level ---
            scene_check = check_scene_dirs(scene_id)
            if not scene_check["scene_dir_exists"] or not scene_check["images_dir_exists"]:
                scenes_missing_merged.append(scene_id)
            if not scene_check["gt_plane_dir_exists"]:
                scenes_missing_gt.append(scene_id)
                continue  # Can't check cameras without GT
            if not scene_check["params_dir_exists"]:
                scenes_missing_params.append(scene_id)

            # --- Camera-level ---
            cameras = discover_cameras(scene_id)
            if not cameras:
                scenes_missing_gt.append(scene_id)
                continue

            split_totals["n_cameras"] += len(cameras)

            for cam_name, cam_info in cameras.items():
                cam_check = check_camera(
                    scene_id, cam_name, cam_info["gt_h5_path"], metadata
                )

                if not cam_check["gt_h5_readable"]:
                    cameras_gt_unreadable.append(f"{scene_id}/{cam_name}")
                    continue

                if not cam_check["rgb_dir_exists"]:
                    cameras_no_rgb_dir.append(f"{scene_id}/{cam_name}")
                if not cam_check["depth_dir_exists"]:
                    cameras_no_depth_dir.append(f"{scene_id}/{cam_name}")
                if cam_check["intrinsics_source"] == "default_886.81":
                    scenes_no_intrinsics_csv.append(scene_id)

                frame_ids = cam_check["gt_frame_ids"]
                n_frames = cam_check["gt_n_frames"]
                split_totals["n_frames_total"] += n_frames

                # --- Frame-level existence ---
                frame_existence = check_frame_existence(scene_id, cam_name, frame_ids)

                n_rgb_missing = 0
                n_depth_missing = 0
                for fe in frame_existence:
                    if not fe["rgb_exists"]:
                        n_rgb_missing += 1
                        frames_rgb_missing.append(
                            f"{scene_id}/{cam_name}/{fe['frame_id']}"
                        )
                    if not fe["depth_exists"]:
                        n_depth_missing += 1
                        frames_depth_missing.append(
                            f"{scene_id}/{cam_name}/{fe['frame_id']}"
                        )
                    if not fe["rgb_exists"] and not fe["depth_exists"]:
                        frames_both_missing.append(
                            f"{scene_id}/{cam_name}/{fe['frame_id']}"
                        )

                split_totals["n_rgb_missing"] += n_rgb_missing
                split_totals["n_depth_missing"] += n_depth_missing

                # --- Frame-level content (optional) ---
                if not skip_frame_content:
                    # Decide which frames to check
                    valid_frames = [
                        fe for fe in frame_existence
                        if fe["rgb_exists"] and fe["depth_exists"]
                    ]

                    if sample_frames is not None and len(valid_frames) > sample_frames:
                        # Deterministic sampling: evenly spaced
                        indices = np.linspace(
                            0, len(valid_frames) - 1, sample_frames, dtype=int
                        )
                        check_frames = [valid_frames[i] for i in indices]
                    else:
                        check_frames = valid_frames

                    for fi, fe in enumerate(check_frames):
                        # Find this frame's index in the GT H5
                        frame_h5_idx = frame_ids.index(fe["frame_id"])

                        content = check_frame_content(
                            fe["rgb_path"], fe["depth_path"],
                            cam_info["gt_h5_path"], frame_h5_idx,
                        )
                        split_totals["n_frames_content_checked"] += 1

                        fid_str = f"{scene_id}/{cam_name}/{fe['frame_id']}"

                        if content["rgb_pct_bad"] > 1.0:
                            frames_rgb_bad.append(
                                (fid_str, content["rgb_pct_bad"], content["rgb_dtype"])
                            )
                        if content["depth_pct_invalid"] > 5.0:
                            frames_depth_bad.append(
                                (fid_str, content["depth_pct_invalid"])
                            )
                        if content["plane_all_zero"]:
                            frames_plane_all_zero.append(fid_str)
                        if content["plane_has_negative"]:
                            frames_plane_negative.append(fid_str)

                        if content["depth_min"] is not None:
                            depth_mins.append(content["depth_min"])
                            depth_maxs.append(content["depth_max"])
                        if content["plane_pct_planar"] is not None:
                            plane_pct_planars.append(content["plane_pct_planar"])

                # --- Prediction checks ---
                if check_preds:
                    for method_name, h5_folder in PREDICTION_FOLDERS.items():
                        pred_check = check_predictions(
                            scene_id, cam_name, frame_ids, method_name, h5_folder
                        )
                        key_prefix = f"pred_{method_name}"
                        if not pred_check["pred_h5_exists"]:
                            split_totals[f"{key_prefix}_missing"] += 1
                        elif not pred_check["pred_h5_readable"]:
                            split_totals[f"{key_prefix}_unreadable"] += 1
                        else:
                            split_totals[f"{key_prefix}_ok"] += 1
                            if not pred_check["pred_frame_ids_match"]:
                                split_totals[f"{key_prefix}_frame_mismatch"] += 1

        # --- Report for this split ---
        out(f"  ### 1. SCENE-LEVEL SUMMARY")
        out(f"     Total scenes:                    {split_totals['n_scenes']}")
        out(f"     Scenes with Hypersim data:       {split_totals['n_scenes'] - len(scenes_missing_merged)}")
        out(f"     Scenes with GT plane labels:     {split_totals['n_scenes'] - len(scenes_missing_gt)}")
        out(f"     Scenes with camera params dir:   {split_totals['n_scenes'] - len(scenes_missing_params)}")
        out()

        if scenes_missing_merged:
            out(f"     [ISSUE] Scenes MISSING from Hypersim_merged ({len(scenes_missing_merged)}):")
            for s in scenes_missing_merged:
                out(f"       - {s}")
            out()

        if scenes_missing_gt:
            out(f"     [ISSUE] Scenes MISSING GT plane labels ({len(scenes_missing_gt)}):")
            for s in scenes_missing_gt:
                out(f"       - {s}")
            out()

        if scenes_missing_params:
            out(f"     [WARN] Scenes missing params dir ({len(scenes_missing_params)}):")
            for s in scenes_missing_params:
                out(f"       - {s}")
            out()

        out(f"  ### 2. CAMERA-LEVEL SUMMARY")
        out(f"     Total cameras (from GT H5):      {split_totals['n_cameras']}")
        if cameras_gt_unreadable:
            out(f"     [ISSUE] GT H5 unreadable ({len(cameras_gt_unreadable)}):")
            for c in cameras_gt_unreadable:
                out(f"       - {c}")
        if cameras_no_rgb_dir:
            out(f"     [ISSUE] Missing RGB directory ({len(cameras_no_rgb_dir)}):")
            for c in cameras_no_rgb_dir:
                out(f"       - {c}")
        if cameras_no_depth_dir:
            out(f"     [ISSUE] Missing depth directory ({len(cameras_no_depth_dir)}):")
            for c in cameras_no_depth_dir:
                out(f"       - {c}")
        out()

        # Intrinsics
        scenes_unique_no_csv = list(set(scenes_no_intrinsics_csv))
        if scenes_unique_no_csv:
            out(f"     [INFO] Scenes using default intrinsics (not in CSV): {len(scenes_unique_no_csv)}")
            if len(scenes_unique_no_csv) <= 10:
                for s in sorted(scenes_unique_no_csv):
                    out(f"       - {s}")
            else:
                out(f"       (showing first 10 of {len(scenes_unique_no_csv)})")
                for s in sorted(scenes_unique_no_csv)[:10]:
                    out(f"       - {s}")
        out()

        out(f"  ### 3. FRAME-LEVEL SUMMARY")
        out(f"     Total frames (from GT H5):       {split_totals['n_frames_total']}")
        n_valid = (
            split_totals["n_frames_total"]
            - max(split_totals["n_rgb_missing"], split_totals["n_depth_missing"])
        )
        out(f"     Frames with both RGB+depth:      {split_totals['n_frames_total'] - len(frames_both_missing) - len([f for f in frames_rgb_missing if f not in frames_both_missing]) - len([f for f in frames_depth_missing if f not in frames_both_missing])}")

        # Compute precise count of frames missing at least one
        all_missing = set(frames_rgb_missing) | set(frames_depth_missing)
        frames_fully_valid = split_totals["n_frames_total"] - len(all_missing)
        out(f"     Frames fully valid (RGB+depth):  {frames_fully_valid}")
        out(f"     Frames missing RGB:              {split_totals['n_rgb_missing']}")
        out(f"     Frames missing depth:            {split_totals['n_depth_missing']}")
        out(f"     Frames missing both:             {len(frames_both_missing)}")
        out()

        if frames_rgb_missing:
            # Group by scene/cam for compact display
            missing_by_scene_cam = defaultdict(list)
            for f in frames_rgb_missing:
                parts = f.split("/")
                missing_by_scene_cam[f"{parts[0]}/{parts[1]}"].append(parts[2])

            out(f"     [ISSUE] Frames missing RGB ({len(frames_rgb_missing)}):")
            for sc, fids in sorted(missing_by_scene_cam.items()):
                out(f"       {sc}: {len(fids)} frames missing (e.g., {', '.join(fids[:5])}{'...' if len(fids) > 5 else ''})")
            out()

        if frames_depth_missing and set(frames_depth_missing) != set(frames_rgb_missing):
            missing_by_scene_cam = defaultdict(list)
            for f in frames_depth_missing:
                parts = f.split("/")
                missing_by_scene_cam[f"{parts[0]}/{parts[1]}"].append(parts[2])

            out(f"     [ISSUE] Frames missing depth ({len(frames_depth_missing)}):")
            for sc, fids in sorted(missing_by_scene_cam.items()):
                out(f"       {sc}: {len(fids)} frames (e.g., {', '.join(fids[:5])}{'...' if len(fids) > 5 else ''})")
            out()
        elif frames_depth_missing:
            out(f"     [INFO] Depth-missing frames identical to RGB-missing ({len(frames_depth_missing)} frames)")
            out()

        # --- Content quality ---
        if not skip_frame_content:
            out(f"  ### 4. DATA QUALITY (content checks)")
            out(f"     Frames checked:                  {split_totals.get('n_frames_content_checked', 0)}")
            out()

            if frames_rgb_bad:
                out(f"     [WARN] RGB frames with >1% bad pixels ({len(frames_rgb_bad)}):")
                for fid_str, pct, dtype in sorted(frames_rgb_bad, key=lambda x: -x[1])[:20]:
                    out(f"       {fid_str}: {pct:.2f}% bad (dtype={dtype})")
                if len(frames_rgb_bad) > 20:
                    out(f"       ... and {len(frames_rgb_bad) - 20} more")
                out()
            else:
                out(f"     [OK] All checked RGB frames have <1% bad pixels")
                out()

            if frames_depth_bad:
                out(f"     [WARN] Depth frames with >5% invalid ({len(frames_depth_bad)}):")
                for fid_str, pct in sorted(frames_depth_bad, key=lambda x: -x[1])[:20]:
                    out(f"       {fid_str}: {pct:.2f}% invalid")
                if len(frames_depth_bad) > 20:
                    out(f"       ... and {len(frames_depth_bad) - 20} more")
                out()
            else:
                out(f"     [OK] All checked depth frames have <5% invalid pixels")
                out()

            if frames_plane_all_zero:
                out(f"     [WARN] Plane labels ALL ZERO ({len(frames_plane_all_zero)}):")
                for fid_str in frames_plane_all_zero[:20]:
                    out(f"       {fid_str}")
                if len(frames_plane_all_zero) > 20:
                    out(f"       ... and {len(frames_plane_all_zero) - 20} more")
                out()
            else:
                out(f"     [OK] No all-zero plane label frames found")
                out()

            if frames_plane_negative:
                out(f"     [WARN] Plane labels with NEGATIVE values ({len(frames_plane_negative)}):")
                for fid_str in frames_plane_negative[:20]:
                    out(f"       {fid_str}")
                out()
            else:
                out(f"     [OK] No negative plane label values found")
                out()

            # Depth range stats
            if depth_mins:
                out(f"     Depth range (across checked frames):")
                out(f"       Min depth:    {np.min(depth_mins):.4f} m")
                out(f"       Max depth:    {np.max(depth_maxs):.4f} m")
                out(f"       Median min:   {np.median(depth_mins):.4f} m")
                out(f"       Median max:   {np.median(depth_maxs):.4f} m")
                out()

            # Plane coverage stats
            if plane_pct_planars:
                out(f"     Plane coverage (% of pixels labeled planar):")
                out(f"       Mean:   {np.mean(plane_pct_planars):.1f}%")
                out(f"       Median: {np.median(plane_pct_planars):.1f}%")
                out(f"       Min:    {np.min(plane_pct_planars):.1f}%")
                out(f"       Max:    {np.max(plane_pct_planars):.1f}%")
                out(f"       Frames with <10% planar: {sum(1 for p in plane_pct_planars if p < 10)}")
                out()

        # --- Prediction checks ---
        if check_preds:
            out(f"  ### 5. PREDICTION H5 CHECKS")
            for method_name, h5_folder in PREDICTION_FOLDERS.items():
                n_ok = split_totals.get(f"pred_{method_name}_ok", 0)
                n_missing = split_totals.get(f"pred_{method_name}_missing", 0)
                n_unreadable = split_totals.get(f"pred_{method_name}_unreadable", 0)
                n_mismatch = split_totals.get(f"pred_{method_name}_frame_mismatch", 0)
                total_cams = split_totals["n_cameras"]

                status = "[OK]" if n_ok == total_cams else "[ISSUE]"
                out(f"     {status} {method_name} ({h5_folder}):")
                out(f"       Cameras with predictions: {n_ok}/{total_cams}")
                if n_missing > 0:
                    out(f"       Missing H5: {n_missing}")
                if n_unreadable > 0:
                    out(f"       Unreadable H5: {n_unreadable}")
                if n_mismatch > 0:
                    out(f"       Frame ID mismatches: {n_mismatch}")
                out()

        # Update grand totals
        grand_totals["scenes"] += split_totals["n_scenes"]
        grand_totals["scenes_missing_merged"] += len(scenes_missing_merged)
        grand_totals["scenes_missing_gt"] += len(scenes_missing_gt)
        grand_totals["scenes_missing_params"] += len(scenes_missing_params)
        grand_totals["cameras"] += split_totals["n_cameras"]
        grand_totals["cameras_no_rgb_dir"] += len(cameras_no_rgb_dir)
        grand_totals["cameras_no_depth_dir"] += len(cameras_no_depth_dir)
        grand_totals["frames_total"] += split_totals["n_frames_total"]
        grand_totals["frames_rgb_missing"] += split_totals["n_rgb_missing"]
        grand_totals["frames_depth_missing"] += split_totals["n_depth_missing"]
        grand_totals["frames_both_missing"] += len(frames_both_missing)
        grand_totals["frames_content_checked"] += split_totals.get("n_frames_content_checked", 0)
        grand_totals["frames_rgb_has_bad_pixels"] += len(frames_rgb_bad)
        grand_totals["frames_depth_has_invalid"] += len(frames_depth_bad)
        grand_totals["frames_plane_all_zero"] += len(frames_plane_all_zero)
        grand_totals["frames_plane_has_negative"] += len(frames_plane_negative)

        out()

    # ============================================================
    # GRAND SUMMARY
    # ============================================================
    out("=" * 80)
    out("GRAND SUMMARY (ALL SPLITS)")
    out("=" * 80)
    out()
    out(f"  Splits checked:        {', '.join(splits)}")
    out(f"  Total scenes:          {grand_totals['scenes']}")
    out(f"  Scenes missing data:   {grand_totals['scenes_missing_merged']} (Hypersim_merged)")
    out(f"  Scenes missing GT:     {grand_totals['scenes_missing_gt']} (Hypersim_rendered)")
    out(f"  Scenes missing params: {grand_totals['scenes_missing_params']} (Hypersim_params)")
    out()
    out(f"  Total cameras:         {grand_totals['cameras']}")
    out(f"  Cameras no RGB dir:    {grand_totals['cameras_no_rgb_dir']}")
    out(f"  Cameras no depth dir:  {grand_totals['cameras_no_depth_dir']}")
    out()
    out(f"  Total frames (GT H5):  {grand_totals['frames_total']}")
    out(f"  Frames missing RGB:    {grand_totals['frames_rgb_missing']}")
    out(f"  Frames missing depth:  {grand_totals['frames_depth_missing']}")
    out(f"  Frames missing both:   {grand_totals['frames_both_missing']}")
    usable = grand_totals["frames_total"] - max(
        grand_totals["frames_rgb_missing"], grand_totals["frames_depth_missing"]
    )
    pct_usable = (usable / grand_totals["frames_total"] * 100) if grand_totals["frames_total"] > 0 else 0
    out(f"  Frames usable:         {usable} ({pct_usable:.1f}%)")
    out()

    if not skip_frame_content:
        out(f"  Content-checked frames: {grand_totals['frames_content_checked']}")
        out(f"  RGB with >1% bad:       {grand_totals['frames_rgb_has_bad_pixels']}")
        out(f"  Depth with >5% invalid: {grand_totals['frames_depth_has_invalid']}")
        out(f"  Plane labels all-zero:  {grand_totals['frames_plane_all_zero']}")
        out(f"  Plane labels negative:  {grand_totals['frames_plane_has_negative']}")
        out()

    # Dataset class behavior
    out("=" * 80)
    out("NOTES ON DATASET CLASS BEHAVIOR (HypersimPlaneDataset)")
    out("=" * 80)
    out()
    out("  The HypersimPlaneDataset silently handles missing data:")
    out("    - Scenes without Hypersim_merged dir:  skipped (lines 104-106)")
    out("    - Scenes without GT plane dir:         skipped (lines 107-109)")
    out("    - Cameras with no plane H5 files:      skipped (lines 115-117)")
    out("    - Cameras with missing RGB dir:        skipped (line 129-131)")
    out("    - Cameras with missing depth dir:      skipped (line 132-134)")
    out("    - Cameras with bad intrinsics:         skipped (line 139-141)")
    out("    - Frames with missing RGB file:        skipped (line 157-158)")
    out("    - Frames with missing depth file:      skipped (line 159-160)")
    out()
    out("  Impact on evaluation:")
    out("    - Only frames with ALL of (GT H5 + RGB + depth) are included")
    out("    - Missing scenes/cameras/frames reduce evaluation set size")
    out("    - No error is raised; reduction is silent")
    out()

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Hypersim dataset validity report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to check (default: all)",
    )
    parser.add_argument(
        "--sample-frames", type=int, default=5,
        help="Number of frames to sample per camera for content checks "
             "(default: 5, use 0 or --skip-frame-content to disable)",
    )
    parser.add_argument(
        "--skip-frame-content", action="store_true",
        help="Skip reading frame data (fast mode, only checks file existence)",
    )
    parser.add_argument(
        "--check-predictions", action="store_true",
        help="Also check prediction H5 files for known methods",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: print to stdout + save to report_hypersim_validity.txt)",
    )
    args = parser.parse_args()

    if args.sample_frames == 0:
        args.skip_frame_content = True

    report = generate_report(
        splits=args.splits,
        sample_frames=args.sample_frames,
        skip_frame_content=args.skip_frame_content,
        check_preds=args.check_predictions,
    )

    # Print to stdout
    print(report)

    # Save to file
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "report_hypersim_validity.txt"
    )
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
