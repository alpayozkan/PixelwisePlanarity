#!/usr/bin/env python3
"""
Hypersim dataset verification checks.

Refactored from hypersim_dataset_validity_report.py with additional checks:
- Plane ID cross-check
- Split overlap detection
- Duplicate frame ID detection
- Per-camera intrinsics source tracking
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ============================================================
# Default paths (same as evaluate_hypersim_all_baselines.py)
# ============================================================
DEFAULT_PATHS = {
    "hypersim_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
    "plane_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
    "params_root": "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params",
    "split_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "../../splits/hypersim"
    ),
    "metadata_csv": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "../../shared/datasets/metadata_camera_parameters.csv"
    ),
    "h5_root": "/cluster/scratch/aoezkan/planeseg/hypersim/inference",
}

DEFAULT_PREDICTION_FOLDERS = {
    "moge_ours": "moge_ours_h5",
    "moge_mixed_bce": "moge_mixed_bce_h5",
    "zeroplane_mixed_dust3r": "zeroplane_mixed_dust3r_h5",
    "zeroplane_mixed": "zeroplane_mixed_h5",
}


# ============================================================
# Split loading & metadata
# ============================================================

def load_split(split: str, split_dir: str) -> List[str]:
    """Load scene IDs from a split file."""
    split_file = os.path.join(split_dir, f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_metadata(csv_path: str) -> Optional[pd.DataFrame]:
    """Load camera metadata CSV."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col="scene_name")
    return None


def inventory_split_files(split_dir: str) -> Dict[str, Dict]:
    """List all split files and their scene counts."""
    result = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(split_dir, f"{name}.txt")
        exists = os.path.isfile(path)
        n_scenes = 0
        if exists:
            with open(path) as f:
                n_scenes = sum(1 for line in f if line.strip())
        result[name] = {"path": path, "exists": exists, "n_scenes": n_scenes}
    return result


def check_split_overlap(split_dir: str, splits: List[str]) -> Dict[str, List[str]]:
    """Check for scene ID overlap between splits."""
    scene_sets = {}
    for split in splits:
        try:
            ids = load_split(split, split_dir)
            scene_sets[split] = set(ids)
        except FileNotFoundError:
            scene_sets[split] = set()

    overlap = {}
    split_list = list(scene_sets.keys())
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            a, b = split_list[i], split_list[j]
            common = sorted(scene_sets[a] & scene_sets[b])
            if common:
                overlap[f"{a}_{b}"] = common
    return overlap


# ============================================================
# Scene-level checks
# ============================================================

def check_scene_dirs(
    scene_id: str, hypersim_root: str, plane_label_root: str, params_root: str
) -> Dict:
    """Check scene-level directory existence."""
    scene_dir = os.path.join(hypersim_root, scene_id)
    images_dir = os.path.join(scene_dir, "images")
    plane_dir = os.path.join(plane_label_root, scene_id)
    params_dir = os.path.join(params_root, scene_id, "_detail")

    return {
        "scene_id": scene_id,
        "scene_dir_exists": os.path.isdir(scene_dir),
        "images_dir_exists": os.path.isdir(images_dir),
        "gt_plane_dir_exists": os.path.isdir(plane_dir),
        "params_dir_exists": os.path.isdir(params_dir),
        "plane_dir_path": plane_dir,
    }


# ============================================================
# Camera-level checks
# ============================================================

def discover_cameras(scene_id: str, plane_label_root: str) -> Dict[str, Dict]:
    """Discover all cameras for a scene from GT plane H5 files."""
    plane_dir = os.path.join(plane_label_root, scene_id)
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


def check_camera(
    scene_id: str, cam_name: str, gt_h5_path: str,
    hypersim_root: str, metadata: Optional[pd.DataFrame]
) -> Dict:
    """Check camera-level validity."""
    images_dir = os.path.join(hypersim_root, scene_id, "images")
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
        "gt_planes_dtype": None,
        "rgb_dir_exists": os.path.isdir(rgb_dir),
        "depth_dir_exists": os.path.isdir(depth_dir),
        "intrinsics_available": False,
        "intrinsics_source": "none",
        "native_resolution": None,
        "duplicate_frame_ids": [],
    }

    # H5 readability
    if result["gt_h5_exists"]:
        try:
            with h5py.File(gt_h5_path, "r") as f:
                raw_ids = f["frame_ids"][:]
                frame_ids = [
                    fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                    for fid in raw_ids
                ]
                planes_shape = f["planes"].shape
                planes_dtype = str(f["planes"].dtype)
            result["gt_h5_readable"] = True
            result["gt_n_frames"] = len(frame_ids)
            result["gt_frame_ids"] = frame_ids
            result["gt_planes_shape"] = planes_shape
            result["gt_planes_dtype"] = planes_dtype
            # Duplicate check
            if len(frame_ids) != len(set(frame_ids)):
                seen = set()
                dups = []
                for fid in frame_ids:
                    if fid in seen:
                        dups.append(fid)
                    seen.add(fid)
                result["duplicate_frame_ids"] = dups
        except Exception as e:
            result["gt_h5_error"] = str(e)

    # Intrinsics
    if metadata is not None and scene_id in metadata.index:
        row = metadata.loc[scene_id]
        has_proj = all(
            f"M_proj_{i}{j}" in row.index for i in range(4) for j in range(4)
        )
        if has_proj:
            result["intrinsics_available"] = True
            result["intrinsics_source"] = "metadata_csv"
            native_w = int(row.get("settings_output_img_width", 1024))
            native_h = int(row.get("settings_output_img_height", 768))
            result["native_resolution"] = (native_w, native_h)
    if not result["intrinsics_available"]:
        result["intrinsics_available"] = True
        result["intrinsics_source"] = "default_886.81"
        result["native_resolution"] = (1024, 768)

    return result


# ============================================================
# Frame-level checks
# ============================================================

def check_frame_existence(
    scene_id: str, cam_name: str, frame_ids: List[str], hypersim_root: str
) -> List[Dict]:
    """Check existence of RGB and depth files for each frame."""
    images_dir = os.path.join(hypersim_root, scene_id, "images")
    rgb_dir = os.path.join(images_dir, f"scene_{cam_name}_final_hdf5")
    depth_dir = os.path.join(images_dir, f"scene_{cam_name}_geometry_hdf5")

    results = []
    for fid in frame_ids:
        rgb_path = os.path.join(rgb_dir, f"frame.{fid}.color.hdf5")
        depth_path = os.path.join(depth_dir, f"frame.{fid}.depth_meters.hdf5")
        results.append({
            "frame_id": fid,
            "rgb_exists": os.path.isfile(rgb_path),
            "depth_exists": os.path.isfile(depth_path),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
        })
    return results


def check_frame_content(
    rgb_path: str, depth_path: str, gt_h5_path: str, frame_idx: int
) -> Dict:
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
        "plane_max_label": 0,
        "plane_has_negative": False,
        "plane_all_zero": False,
        "plane_pct_planar": 0.0,
    }

    # RGB (HDF5 format)
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
            n_inf = int(np.isinf(rgb_f).sum())
            n_nan = int(np.isnan(rgb_f).sum())
            n_neg = int((rgb_f < 0).sum())
            result["rgb_has_inf"] = n_inf > 0
            result["rgb_has_nan"] = n_nan > 0
            result["rgb_has_negative"] = n_neg > 0
            result["rgb_pct_bad"] = float((n_inf + n_nan + n_neg) / n_pixels * 100)
        except Exception as e:
            result["rgb_error"] = str(e)

    # Depth (HDF5 format, stored in meters)
    if os.path.isfile(depth_path):
        try:
            with h5py.File(depth_path, "r") as f:
                key = list(f.keys())[0]
                depth = f[key][:].astype(np.float32)
            result["depth_readable"] = True
            result["depth_dtype"] = str(depth.dtype)
            result["depth_shape"] = depth.shape
            n_pixels = depth.size
            n_inf = int(np.isinf(depth).sum())
            n_nan = int(np.isnan(depth).sum())
            n_neg = int((depth < 0).sum())
            n_zero = int((depth == 0).sum())
            result["depth_has_inf"] = n_inf > 0
            result["depth_has_nan"] = n_nan > 0
            result["depth_has_negative"] = n_neg > 0
            result["depth_has_zero"] = n_zero > 0
            result["depth_pct_invalid"] = float(
                (n_inf + n_nan + n_neg + n_zero) / n_pixels * 100
            )
            valid_mask = np.isfinite(depth) & (depth > 0)
            if valid_mask.any():
                result["depth_min"] = float(depth[valid_mask].min())
                result["depth_max"] = float(depth[valid_mask].max())
                result["depth_median"] = float(np.median(depth[valid_mask]))
        except Exception as e:
            result["depth_error"] = str(e)

    # Plane labels (from GT H5)
    try:
        with h5py.File(gt_h5_path, "r") as f:
            plane = f["planes"][frame_idx]
        result["plane_readable"] = True
        result["plane_shape"] = plane.shape
        unique_pos = np.unique(plane[plane > 0])
        result["plane_n_labels"] = int(len(unique_pos))
        result["plane_max_label"] = int(unique_pos.max()) if len(unique_pos) > 0 else 0
        result["plane_has_negative"] = bool((plane < 0).any())
        result["plane_all_zero"] = bool((plane <= 0).all())
        total = plane.size
        planar = int((plane > 0).sum())
        result["plane_pct_planar"] = float(planar / total * 100) if total > 0 else 0.0
    except Exception as e:
        result["plane_error"] = str(e)

    return result


# ============================================================
# Prediction H5 checks
# ============================================================

def check_predictions(
    scene_id: str, cam_name: str, gt_frame_ids: List[str],
    method_name: str, h5_root: str, h5_folder: str
) -> Dict:
    """Check prediction H5 for a specific method/scene/camera."""
    h5_path = os.path.join(h5_root, h5_folder, scene_id, f"planes_{cam_name}.h5")
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
# Plane ID cross-check
# ============================================================

def check_plane_ids(
    scene_id: str, cam_name: str, gt_h5_path: str, plane_label_root: str
) -> Dict:
    """Cross-check plane IDs in H5 labels for consistency.

    Checks max label across sampled frames to detect per-frame remap.
    (Hypersim doesn't have PLY meshes like ScanNet++, so we check
    cross-frame consistency only.)
    """
    result = {
        "scene_id": scene_id,
        "cam_name": cam_name,
        "max_h5": None,
        "frame_maxes": [],
        "frame_max_varies": False,
        "has_negative_labels": False,
        "diagnosis": "unknown",
    }

    try:
        with h5py.File(gt_h5_path, "r") as f:
            planes = f["planes"]
            n_frames = planes.shape[0]
            indices = np.linspace(0, n_frames - 1, min(20, n_frames), dtype=int)
            frame_maxes = []
            has_neg = False
            for idx in indices:
                frame_data = planes[idx]
                frame_maxes.append(int(frame_data.max()))
                if (frame_data < 0).any():
                    has_neg = True

        result["frame_maxes"] = frame_maxes
        result["max_h5"] = max(frame_maxes) if frame_maxes else 0
        result["frame_max_varies"] = len(set(frame_maxes)) > 1
        result["has_negative_labels"] = has_neg

        if result["frame_max_varies"]:
            result["diagnosis"] = "per_frame_remap"
        else:
            result["diagnosis"] = "consistent"
    except Exception as e:
        result["h5_error"] = str(e)
        result["diagnosis"] = "h5_read_error"

    return result


# ============================================================
# Main orchestrator
# ============================================================

def run_all_checks(config: Dict) -> Dict:
    """Run all Hypersim verification checks.

    Args:
        config: dict with keys:
            splits, sample_frames, skip_frame_content, check_predictions,
            check_plane_ids, max_scenes,
            hypersim_root, plane_label_root, params_root, split_dir,
            metadata_csv, h5_root, prediction_folders

    Returns:
        dict with dataset results, stats, and issues.
    """
    hypersim_root = config.get("hypersim_root", DEFAULT_PATHS["hypersim_root"])
    plane_label_root = config.get("plane_label_root", DEFAULT_PATHS["plane_label_root"])
    params_root = config.get("params_root", DEFAULT_PATHS["params_root"])
    split_dir = config.get("split_dir", DEFAULT_PATHS["split_dir"])
    metadata_csv = config.get("metadata_csv", DEFAULT_PATHS["metadata_csv"])
    h5_root = config.get("h5_root", DEFAULT_PATHS["h5_root"])
    prediction_folders = config.get("prediction_folders", DEFAULT_PREDICTION_FOLDERS)
    splits = config.get("splits", ["train", "val", "test"])
    sample_frames = config.get("sample_frames", 5)
    skip_frame_content = config.get("skip_frame_content", False)
    check_preds = config.get("check_predictions", False)
    check_plane_ids_flag = config.get("check_plane_ids", False)
    max_scenes = config.get("max_scenes")

    metadata = load_metadata(metadata_csv)

    result = {
        "dataset": "hypersim",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "hypersim_root": {"path": hypersim_root, "exists": os.path.isdir(hypersim_root)},
            "plane_label_root": {"path": plane_label_root, "exists": os.path.isdir(plane_label_root)},
            "params_root": {"path": params_root, "exists": os.path.isdir(params_root)},
            "metadata_csv": {"path": metadata_csv, "exists": os.path.isfile(metadata_csv)},
            "split_dir": {"path": split_dir, "exists": os.path.isdir(split_dir)},
        },
        "split_files": inventory_split_files(split_dir),
        "split_overlap": check_split_overlap(split_dir, splits),
        "splits": {},
    }

    if check_preds:
        result["paths"]["h5_root"] = {
            "path": h5_root, "exists": os.path.isdir(h5_root)
        }

    for split in splits:
        try:
            scene_ids = load_split(split, split_dir)
        except FileNotFoundError as e:
            result["splits"][split] = {
                "error": str(e), "n_scenes": 0, "issues": []
            }
            continue

        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        sr = _run_split_checks(
            split, scene_ids,
            hypersim_root, plane_label_root, params_root,
            h5_root, prediction_folders, metadata,
            sample_frames, skip_frame_content,
            check_preds, check_plane_ids_flag,
        )
        result["splits"][split] = sr

    return result


def _run_split_checks(
    split: str, scene_ids: List[str],
    hypersim_root: str, plane_label_root: str, params_root: str,
    h5_root: str, prediction_folders: Dict[str, str],
    metadata: Optional[pd.DataFrame],
    sample_frames: int, skip_frame_content: bool,
    check_preds: bool, check_plane_ids_flag: bool,
) -> Dict:
    """Run checks for a single Hypersim split."""
    issues = []
    sr = {
        "n_scenes": len(scene_ids),
        # Scene-level
        "n_scenes_with_data": 0,
        "n_scenes_with_gt": 0,
        "n_scenes_with_params": 0,
        "scenes_missing_data": [],
        "scenes_missing_gt": [],
        "scenes_missing_params": [],
        # Camera-level
        "total_cameras": 0,
        "cameras_gt_unreadable": [],
        "cameras_no_rgb_dir": [],
        "cameras_no_depth_dir": [],
        "cameras_default_intrinsics": [],
        "cameras_csv_intrinsics": [],
        "cameras_duplicate_frame_ids": [],
        # Frame-level
        "total_gt_frames": 0,
        "frames_with_rgb": 0,
        "frames_with_depth": 0,
        "frames_missing_rgb": [],
        "frames_missing_depth": [],
        "frames_missing_both": [],
        # Content quality
        "content_checked": 0,
        "content_rgb_bad": [],
        "content_depth_bad": [],
        "content_plane_all_zero": [],
        "content_plane_negative": [],
        "depth_mins": [],
        "depth_maxs": [],
        "plane_pct_planars": [],
        "plane_n_labels_list": [],
        # Predictions
        "predictions": {},
        # Plane ID cross-check
        "plane_id_checks": {},
        # Issues
        "issues": issues,
    }

    if check_preds:
        for method in prediction_folders:
            sr["predictions"][method] = {
                "n_with_h5": 0, "n_readable": 0, "n_frame_match": 0,
                "missing_cams": [], "unreadable_cams": [],
                "frame_mismatch_cams": [],
            }

    for si, scene_id in enumerate(scene_ids):
        print(f"  [hypersim/{split}] {si+1}/{len(scene_ids)}: {scene_id}")

        scene = check_scene_dirs(
            scene_id, hypersim_root, plane_label_root, params_root
        )

        # Scene-level
        if scene["scene_dir_exists"] and scene["images_dir_exists"]:
            sr["n_scenes_with_data"] += 1
        else:
            sr["scenes_missing_data"].append(scene_id)
            issues.append(("ISSUE", "scene", scene_id, "Missing Hypersim data dir"))

        if not scene["gt_plane_dir_exists"]:
            sr["scenes_missing_gt"].append(scene_id)
            issues.append(("ISSUE", "scene", scene_id, "Missing GT plane label dir"))
            continue

        sr["n_scenes_with_gt"] += 1

        if scene["params_dir_exists"]:
            sr["n_scenes_with_params"] += 1
        else:
            sr["scenes_missing_params"].append(scene_id)

        # Discover cameras
        cameras = discover_cameras(scene_id, plane_label_root)
        if not cameras:
            sr["scenes_missing_gt"].append(scene_id)
            issues.append(("ISSUE", "scene", scene_id, "No camera H5 files found"))
            continue

        sr["total_cameras"] += len(cameras)

        for cam_name, cam_info in cameras.items():
            cam = check_camera(
                scene_id, cam_name, cam_info["gt_h5_path"],
                hypersim_root, metadata
            )

            if not cam["gt_h5_readable"]:
                sr["cameras_gt_unreadable"].append(f"{scene_id}/{cam_name}")
                issues.append(("ISSUE", "camera", f"{scene_id}/{cam_name}",
                               "GT H5 unreadable"))
                continue

            if not cam["rgb_dir_exists"]:
                sr["cameras_no_rgb_dir"].append(f"{scene_id}/{cam_name}")
            if not cam["depth_dir_exists"]:
                sr["cameras_no_depth_dir"].append(f"{scene_id}/{cam_name}")

            if cam["intrinsics_source"] == "default_886.81":
                sr["cameras_default_intrinsics"].append(f"{scene_id}/{cam_name}")
            else:
                sr["cameras_csv_intrinsics"].append(f"{scene_id}/{cam_name}")

            if cam["duplicate_frame_ids"]:
                sr["cameras_duplicate_frame_ids"].append(
                    (f"{scene_id}/{cam_name}", cam["duplicate_frame_ids"])
                )
                issues.append(("ISSUE", "camera", f"{scene_id}/{cam_name}",
                               f"Duplicate frame IDs: {cam['duplicate_frame_ids'][:5]}"))

            frame_ids = cam["gt_frame_ids"]
            sr["total_gt_frames"] += cam["gt_n_frames"]

            # Frame existence
            frame_existence = check_frame_existence(
                scene_id, cam_name, frame_ids, hypersim_root
            )
            for fe in frame_existence:
                if fe["rgb_exists"]:
                    sr["frames_with_rgb"] += 1
                else:
                    sr["frames_missing_rgb"].append(
                        f"{scene_id}/{cam_name}/{fe['frame_id']}")
                if fe["depth_exists"]:
                    sr["frames_with_depth"] += 1
                else:
                    sr["frames_missing_depth"].append(
                        f"{scene_id}/{cam_name}/{fe['frame_id']}")
                if not fe["rgb_exists"] and not fe["depth_exists"]:
                    sr["frames_missing_both"].append(
                        f"{scene_id}/{cam_name}/{fe['frame_id']}")

            # Frame content (sampled)
            if not skip_frame_content:
                valid_frames = [
                    fe for fe in frame_existence
                    if fe["rgb_exists"] and fe["depth_exists"]
                ]
                if sample_frames is not None and len(valid_frames) > sample_frames:
                    indices = np.linspace(
                        0, len(valid_frames) - 1, sample_frames, dtype=int
                    )
                    check_list = [valid_frames[i] for i in indices]
                else:
                    check_list = valid_frames

                for fe in check_list:
                    frame_h5_idx = frame_ids.index(fe["frame_id"])
                    content = check_frame_content(
                        fe["rgb_path"], fe["depth_path"],
                        cam_info["gt_h5_path"], frame_h5_idx,
                    )
                    sr["content_checked"] += 1
                    fid_str = f"{scene_id}/{cam_name}/{fe['frame_id']}"

                    if content["rgb_pct_bad"] > 1.0:
                        sr["content_rgb_bad"].append(
                            (fid_str, content["rgb_pct_bad"], content["rgb_dtype"])
                        )
                    if content["depth_pct_invalid"] > 5.0:
                        sr["content_depth_bad"].append(
                            (fid_str, content["depth_pct_invalid"])
                        )
                    if content["plane_all_zero"]:
                        sr["content_plane_all_zero"].append(fid_str)
                    if content["plane_has_negative"]:
                        sr["content_plane_negative"].append(fid_str)
                    if content["depth_min"] is not None:
                        sr["depth_mins"].append(content["depth_min"])
                        sr["depth_maxs"].append(content["depth_max"])
                    if content["plane_pct_planar"] is not None:
                        sr["plane_pct_planars"].append(content["plane_pct_planar"])
                    if content["plane_n_labels"] > 0:
                        sr["plane_n_labels_list"].append(content["plane_n_labels"])

            # Prediction checks
            if check_preds:
                for method_name, h5_folder in prediction_folders.items():
                    pc = check_predictions(
                        scene_id, cam_name, frame_ids,
                        method_name, h5_root, h5_folder
                    )
                    pred_stats = sr["predictions"][method_name]
                    if pc["pred_h5_exists"]:
                        pred_stats["n_with_h5"] += 1
                        if pc["pred_h5_readable"]:
                            pred_stats["n_readable"] += 1
                            if pc["pred_frame_ids_match"]:
                                pred_stats["n_frame_match"] += 1
                            else:
                                pred_stats["frame_mismatch_cams"].append(
                                    f"{scene_id}/{cam_name}")
                        else:
                            pred_stats["unreadable_cams"].append(
                                f"{scene_id}/{cam_name}")
                    else:
                        pred_stats["missing_cams"].append(
                            f"{scene_id}/{cam_name}")

            # Plane ID cross-check
            if check_plane_ids_flag:
                pid = check_plane_ids(
                    scene_id, cam_name, cam_info["gt_h5_path"], plane_label_root
                )
                sr["plane_id_checks"][f"{scene_id}/{cam_name}"] = pid
                if pid["diagnosis"] == "per_frame_remap":
                    issues.append(("ISSUE", "plane_id",
                                   f"{scene_id}/{cam_name}",
                                   "Per-frame consecutive remap detected"))
                if pid["has_negative_labels"]:
                    issues.append(("WARN", "plane_id",
                                   f"{scene_id}/{cam_name}",
                                   "Negative plane labels found"))

    return sr
