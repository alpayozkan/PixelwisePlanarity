#!/usr/bin/env python3
"""
ScanNet++ dataset verification checks.

Refactored from scannetpp_dataset_validity_report.py with additional checks:
- Plane ID cross-check (PLY mesh vs H5 labels)
- Split overlap detection
- Duplicate frame ID detection
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ============================================================
# Default paths (same as evaluate_all_baselines.py)
# ============================================================
DEFAULT_PATHS = {
    "rgb_root": "/cluster/project/cvg/Shared_datasets/scannet++/data",
    "gt_root": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
    "mesh_root": "/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp/",
    "split_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "../../splits/scannetpp"
    ),
    "h5_root": "/cluster/scratch/aoezkan/planeseg/scannetpp/inference",
}

DEFAULT_PREDICTION_FOLDERS = {
    "ours": "moge_ours_v2_h5",
    "moge_mixed_bce": "moge_mixed_bce_h5",
    "zeroplane": "zeroplane_h5",
    "zeroplane_mixed": "zeroplane_mixed_h5",
    "zeroplane_mixed_dust3r": "zeroplane_mixed_dust3r_h5",
    "gtseg": "gtseg_v1_h5",
}


# ============================================================
# Split loading
# ============================================================

def load_split(split: str, split_dir: str) -> Tuple[List[str], str]:
    """Load scene IDs from a split file.

    Tries nvs_sem_{split}_with_planes.txt first (what the dataset class uses),
    then falls back to {split}.txt.

    Returns:
        (scene_ids, split_file_path)
    """
    primary = os.path.join(split_dir, f"nvs_sem_{split}_with_planes.txt")
    if os.path.exists(primary):
        with open(primary) as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids, primary

    fallback = os.path.join(split_dir, f"{split}.txt")
    if os.path.exists(fallback):
        with open(fallback) as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids, fallback

    raise FileNotFoundError(
        f"No split file found for '{split}'. Tried: {primary}, {fallback}"
    )


def inventory_split_files(split_dir: str) -> Dict[str, Dict]:
    """List all split files and their scene counts."""
    names = [
        "nvs_sem_train_with_planes", "nvs_sem_val_with_planes",
        "nvs_sem_test_with_planes",
        "nvs_sem_train_missing_planes", "nvs_sem_val_missing_planes",
        "train", "val", "test", "all_scenes",
    ]
    result = {}
    for name in names:
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
            ids, _ = load_split(split, split_dir)
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

def check_scene(scene_id: str, rgb_root: str, gt_root: str) -> Dict:
    """Check scene-level directory and file existence."""
    rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
    plane_h5 = os.path.join(gt_root, scene_id, "rendered.h5")
    sem_h5 = os.path.join(gt_root, scene_id, "rendered_sem.h5")
    depth_h5 = os.path.join(gt_root, scene_id, "rendered_depth.h5")
    pose_json = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")

    result = {
        "scene_id": scene_id,
        "rgb_dir_exists": os.path.isdir(rgb_dir),
        "plane_h5_exists": os.path.isfile(plane_h5),
        "sem_h5_exists": os.path.isfile(sem_h5),
        "depth_h5_exists": os.path.isfile(depth_h5),
        "pose_json_exists": os.path.isfile(pose_json),
        "plane_h5_path": plane_h5,
        "sem_h5_path": sem_h5,
        "depth_h5_path": depth_h5,
        "pose_json_path": pose_json,
        "rgb_dir_path": rgb_dir,
        "plane_h5_readable": False,
        "plane_n_frames": 0,
        "plane_frame_ids": [],
        "plane_shape": None,
        "plane_dtype": None,
        "depth_h5_readable": False,
        "depth_n_frames": 0,
        "sem_h5_readable": False,
        "sem_n_frames": 0,
        "pose_n_frames": 0,
        "duplicate_frame_ids": [],
    }

    # Check plane H5
    if result["plane_h5_exists"]:
        try:
            with h5py.File(plane_h5, "r") as f:
                raw_ids = f["frame_ids"][:]
                frame_ids = [
                    fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                    for fid in raw_ids
                ]
                planes_shape = f["planes"].shape
                planes_dtype = str(f["planes"].dtype)
            result["plane_h5_readable"] = True
            result["plane_n_frames"] = len(frame_ids)
            result["plane_frame_ids"] = frame_ids
            result["plane_shape"] = planes_shape
            result["plane_dtype"] = planes_dtype
            # Check for duplicates
            if len(frame_ids) != len(set(frame_ids)):
                seen = set()
                dups = []
                for fid in frame_ids:
                    if fid in seen:
                        dups.append(fid)
                    seen.add(fid)
                result["duplicate_frame_ids"] = dups
        except Exception as e:
            result["plane_h5_error"] = str(e)

    # Check depth H5
    if result["depth_h5_exists"]:
        try:
            with h5py.File(depth_h5, "r") as f:
                result["depth_h5_readable"] = True
                result["depth_n_frames"] = f["depth"].shape[0]
        except Exception as e:
            result["depth_h5_error"] = str(e)

    # Check sem H5
    if result["sem_h5_exists"]:
        try:
            with h5py.File(sem_h5, "r") as f:
                result["sem_h5_readable"] = True
                result["sem_n_frames"] = f["sem"].shape[0]
        except Exception as e:
            result["sem_h5_error"] = str(e)

    # Check pose JSON
    if result["pose_json_exists"]:
        try:
            with open(pose_json, "r") as f:
                pose_data = json.load(f)
            result["pose_n_frames"] = len(pose_data)
        except Exception as e:
            result["pose_json_error"] = str(e)

    return result


# ============================================================
# Frame-level checks
# ============================================================

def check_frame_existence(
    scene_id: str, frame_ids: List[str], rgb_root: str,
    pose_data: Optional[Dict]
) -> List[Dict]:
    """Check existence of RGB files and pose data for each frame."""
    rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
    results = []
    for fid in frame_ids:
        rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
        has_pose = pose_data is not None and fid in pose_data
        has_intrinsic = False
        if has_pose:
            has_intrinsic = "intrinsic" in pose_data.get(fid, {})
        results.append({
            "frame_id": fid,
            "rgb_exists": os.path.isfile(rgb_path),
            "rgb_path": rgb_path,
            "has_pose": has_pose,
            "has_intrinsic": has_intrinsic,
        })
    return results


def check_frame_content(
    frame_id: str, frame_idx: int,
    plane_h5_path: str, depth_h5_path: str, sem_h5_path: str, rgb_path: str
) -> Dict:
    """Check data quality of a single frame (reads actual files)."""
    import cv2

    result = {
        "frame_id": frame_id,
        "rgb_readable": False,
        "rgb_shape": None,
        "depth_readable": False,
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
        "sem_readable": False,
        "sem_shape": None,
        "sem_n_classes": 0,
    }

    # RGB
    if os.path.isfile(rgb_path):
        try:
            img = cv2.imread(rgb_path)
            if img is not None:
                result["rgb_readable"] = True
                result["rgb_shape"] = img.shape
            else:
                result["rgb_error"] = "cv2.imread returned None"
        except Exception as e:
            result["rgb_error"] = str(e)

    # Plane labels
    if os.path.isfile(plane_h5_path):
        try:
            with h5py.File(plane_h5_path, "r") as f:
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

    # Depth (stored in mm in ScanNet++ H5)
    if os.path.isfile(depth_h5_path):
        try:
            with h5py.File(depth_h5_path, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32)
            result["depth_readable"] = True
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
                result["depth_min"] = float(depth[valid_mask].min() / 1000.0)
                result["depth_max"] = float(depth[valid_mask].max() / 1000.0)
                result["depth_median"] = float(np.median(depth[valid_mask]) / 1000.0)
        except Exception as e:
            result["depth_error"] = str(e)

    # Semantics
    if os.path.isfile(sem_h5_path):
        try:
            with h5py.File(sem_h5_path, "r") as f:
                sem = f["sem"][frame_idx]
            result["sem_readable"] = True
            result["sem_shape"] = sem.shape
            result["sem_n_classes"] = int(len(np.unique(sem)))
        except Exception as e:
            result["sem_error"] = str(e)

    return result


# ============================================================
# Prediction H5 checks
# ============================================================

def check_predictions(
    scene_id: str, gt_frame_ids: List[str],
    method_name: str, h5_root: str, h5_folder: str
) -> Dict:
    """Check prediction H5 for a specific method/scene."""
    h5_path = os.path.join(h5_root, h5_folder, scene_id, "planes.h5")
    result = {
        "method": method_name,
        "scene_id": scene_id,
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
# Plane ID cross-check (PLY mesh vs H5 labels)
# ============================================================

def check_plane_ids(scene_id: str, plane_h5_path: str, mesh_root: str) -> Dict:
    """Cross-check plane IDs between PLY mesh and H5 labels.

    Checks:
    - max_h5 == max_ply + 1 (correct +1 shift)
    - Per-frame ID consistency (detect per-frame consecutive remap)
    - plane_id=0 collision
    """
    ply_path = os.path.join(mesh_root, scene_id, "planes.ply")
    result = {
        "scene_id": scene_id,
        "ply_exists": os.path.isfile(ply_path),
        "ply_readable": False,
        "max_ply": None,
        "max_h5": None,
        "frame_maxes": [],
        "frame_max_varies": False,
        "diagnosis": "unknown",
    }

    if not result["ply_exists"]:
        result["diagnosis"] = "no_ply"
        return result

    try:
        from planamono.shared.rendering.mesh_io import read_ply_faces_with_plane_ids
        _, _, plane_id_face, _ = read_ply_faces_with_plane_ids(ply_path)
        result["ply_readable"] = True
        result["max_ply"] = int(plane_id_face.max())
    except Exception as e:
        result["ply_error"] = str(e)
        result["diagnosis"] = "ply_read_error"
        return result

    try:
        with h5py.File(plane_h5_path, "r") as f:
            planes = f["planes"]
            n_frames = planes.shape[0]
            frame_maxes = []
            # Sample up to 20 frames to check consistency
            indices = np.linspace(0, n_frames - 1, min(20, n_frames), dtype=int)
            for idx in indices:
                frame_max = int(planes[idx].max())
                frame_maxes.append(frame_max)

        result["frame_maxes"] = frame_maxes
        result["max_h5"] = max(frame_maxes) if frame_maxes else 0
        result["frame_max_varies"] = len(set(frame_maxes)) > 1

        max_ply = result["max_ply"]
        max_h5 = result["max_h5"]

        if result["frame_max_varies"]:
            result["diagnosis"] = "per_frame_remap"
        elif max_h5 == max_ply + 1:
            result["diagnosis"] = "correct"
        elif max_h5 == max_ply:
            result["diagnosis"] = "collision"
        else:
            result["diagnosis"] = f"unexpected_shift_{max_h5 - max_ply}"
    except Exception as e:
        result["h5_error"] = str(e)
        result["diagnosis"] = "h5_read_error"

    return result


# ============================================================
# Main orchestrator
# ============================================================

def run_all_checks(config: Dict) -> Dict:
    """Run all ScanNet++ verification checks.

    Args:
        config: dict with keys:
            splits: list of split names
            sample_frames: int (frames per scene for content checks)
            skip_frame_content: bool
            check_predictions: bool
            check_plane_ids: bool
            max_scenes: Optional[int]
            rgb_root, gt_root, mesh_root, split_dir, h5_root: str paths
            prediction_folders: dict {method_name: h5_folder}

    Returns:
        dict with dataset results, stats, and issues.
    """
    rgb_root = config.get("rgb_root", DEFAULT_PATHS["rgb_root"])
    gt_root = config.get("gt_root", DEFAULT_PATHS["gt_root"])
    mesh_root = config.get("mesh_root", DEFAULT_PATHS["mesh_root"])
    split_dir = config.get("split_dir", DEFAULT_PATHS["split_dir"])
    h5_root = config.get("h5_root", DEFAULT_PATHS["h5_root"])
    prediction_folders = config.get("prediction_folders", DEFAULT_PREDICTION_FOLDERS)
    splits = config.get("splits", ["train", "val", "test"])
    sample_frames = config.get("sample_frames", 5)
    skip_frame_content = config.get("skip_frame_content", False)
    check_preds = config.get("check_predictions", False)
    check_plane_ids_flag = config.get("check_plane_ids", False)
    max_scenes = config.get("max_scenes")

    result = {
        "dataset": "scannetpp",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "rgb_root": {"path": rgb_root, "exists": os.path.isdir(rgb_root)},
            "gt_root": {"path": gt_root, "exists": os.path.isdir(gt_root)},
            "mesh_root": {"path": mesh_root, "exists": os.path.isdir(mesh_root)},
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
            scene_ids, split_file_used = load_split(split, split_dir)
        except FileNotFoundError as e:
            result["splits"][split] = {
                "error": str(e), "n_scenes": 0, "issues": []
            }
            continue

        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]

        sr = _run_split_checks(
            split, scene_ids, split_file_used,
            rgb_root, gt_root, mesh_root, h5_root,
            prediction_folders,
            sample_frames, skip_frame_content,
            check_preds, check_plane_ids_flag,
        )
        result["splits"][split] = sr

    return result


def _run_split_checks(
    split: str, scene_ids: List[str], split_file_used: str,
    rgb_root: str, gt_root: str, mesh_root: str, h5_root: str,
    prediction_folders: Dict[str, str],
    sample_frames: int, skip_frame_content: bool,
    check_preds: bool, check_plane_ids_flag: bool,
) -> Dict:
    """Run checks for a single split."""
    issues = []
    sr = {
        "split_file_used": os.path.basename(split_file_used),
        "n_scenes": len(scene_ids),
        # Scene-level counts
        "n_scenes_with_rgb_dir": 0,
        "n_scenes_with_gt_h5": 0,
        "n_scenes_with_sem_h5": 0,
        "n_scenes_with_depth_h5": 0,
        "n_scenes_with_pose_json": 0,
        "n_scenes_h5_readable": 0,
        "n_scenes_frame_count_match": 0,
        # Lists
        "scenes_missing_rgb_dir": [],
        "scenes_missing_gt_h5": [],
        "scenes_missing_sem_h5": [],
        "scenes_missing_depth_h5": [],
        "scenes_missing_pose_json": [],
        "scenes_h5_unreadable": [],
        "scenes_frame_count_mismatch": [],
        "scenes_duplicate_frame_ids": [],
        # Frame-level
        "total_gt_frames": 0,
        "frames_with_rgb": 0,
        "frames_with_pose": 0,
        "frames_missing_rgb": [],
        "frames_missing_pose": [],
        # Content quality
        "content_checked": 0,
        "content_rgb_unreadable": [],
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
                "missing_scenes": [], "unreadable_scenes": [],
                "frame_mismatch_scenes": [],
            }

    for si, scene_id in enumerate(scene_ids):
        print(f"  [scannetpp/{split}] {si+1}/{len(scene_ids)}: {scene_id}")

        scene = check_scene(scene_id, rgb_root, gt_root)

        # Scene-level tallying
        if scene["rgb_dir_exists"]:
            sr["n_scenes_with_rgb_dir"] += 1
        else:
            sr["scenes_missing_rgb_dir"].append(scene_id)
            issues.append(("WARN", "scene", scene_id, "Missing RGB dir"))

        if not scene["plane_h5_exists"]:
            sr["scenes_missing_gt_h5"].append(scene_id)
            issues.append(("ISSUE", "scene", scene_id, "Missing GT plane H5"))
            continue

        sr["n_scenes_with_gt_h5"] += 1

        if scene["sem_h5_exists"]:
            sr["n_scenes_with_sem_h5"] += 1
        else:
            sr["scenes_missing_sem_h5"].append(scene_id)

        if scene["depth_h5_exists"]:
            sr["n_scenes_with_depth_h5"] += 1
        else:
            sr["scenes_missing_depth_h5"].append(scene_id)

        if scene["pose_json_exists"]:
            sr["n_scenes_with_pose_json"] += 1
        else:
            sr["scenes_missing_pose_json"].append(scene_id)

        if not scene["plane_h5_readable"]:
            sr["scenes_h5_unreadable"].append(scene_id)
            issues.append(("ISSUE", "scene", scene_id, "Plane H5 unreadable"))
            continue

        sr["n_scenes_h5_readable"] += 1

        # Duplicate frame IDs
        if scene["duplicate_frame_ids"]:
            sr["scenes_duplicate_frame_ids"].append(
                (scene_id, scene["duplicate_frame_ids"])
            )
            issues.append(("ISSUE", "scene", scene_id,
                           f"Duplicate frame IDs: {scene['duplicate_frame_ids'][:5]}"))

        # Frame count consistency
        n_plane = scene["plane_n_frames"]
        n_depth = scene["depth_n_frames"]
        n_sem = scene["sem_n_frames"]
        mismatch = False
        if n_depth > 0 and n_depth != n_plane:
            sr["scenes_frame_count_mismatch"].append(
                f"{scene_id}: plane={n_plane}, depth={n_depth}")
            mismatch = True
        if n_sem > 0 and n_sem != n_plane:
            sr["scenes_frame_count_mismatch"].append(
                f"{scene_id}: plane={n_plane}, sem={n_sem}")
            mismatch = True
        if not mismatch:
            sr["n_scenes_frame_count_match"] += 1

        frame_ids = scene["plane_frame_ids"]
        sr["total_gt_frames"] += len(frame_ids)

        # Load pose data
        pose_data = None
        if scene["pose_json_exists"]:
            try:
                with open(scene["pose_json_path"]) as f:
                    pose_data = json.load(f)
            except Exception:
                pass

        # Frame existence checks
        frame_existence = check_frame_existence(
            scene_id, frame_ids, rgb_root, pose_data
        )
        for fe in frame_existence:
            if not fe["rgb_exists"]:
                sr["frames_missing_rgb"].append(f"{scene_id}/{fe['frame_id']}")
            else:
                sr["frames_with_rgb"] += 1
            if not fe["has_pose"]:
                sr["frames_missing_pose"].append(f"{scene_id}/{fe['frame_id']}")
            else:
                sr["frames_with_pose"] += 1

        # Frame content checks (sampled)
        if not skip_frame_content:
            valid_frames = [fe for fe in frame_existence if fe["rgb_exists"]]

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
                    fe["frame_id"], frame_h5_idx,
                    scene["plane_h5_path"], scene["depth_h5_path"],
                    scene["sem_h5_path"], fe["rgb_path"],
                )
                sr["content_checked"] += 1
                fid_str = f"{scene_id}/{fe['frame_id']}"

                if not content["rgb_readable"]:
                    sr["content_rgb_unreadable"].append(fid_str)
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
                    scene_id, frame_ids, method_name, h5_root, h5_folder
                )
                pred_stats = sr["predictions"][method_name]
                if pc["pred_h5_exists"]:
                    pred_stats["n_with_h5"] += 1
                    if pc["pred_h5_readable"]:
                        pred_stats["n_readable"] += 1
                        if pc["pred_frame_ids_match"]:
                            pred_stats["n_frame_match"] += 1
                        else:
                            pred_stats["frame_mismatch_scenes"].append(scene_id)
                    else:
                        pred_stats["unreadable_scenes"].append(scene_id)
                else:
                    pred_stats["missing_scenes"].append(scene_id)

        # Plane ID cross-check
        if check_plane_ids_flag:
            pid = check_plane_ids(scene_id, scene["plane_h5_path"], mesh_root)
            sr["plane_id_checks"][scene_id] = pid
            if pid["diagnosis"] == "per_frame_remap":
                issues.append(("ISSUE", "plane_id", scene_id,
                               "Per-frame consecutive remap detected"))
            elif pid["diagnosis"] == "collision":
                issues.append(("ISSUE", "plane_id", scene_id,
                               "plane_id=0 collision (max_h5 == max_ply)"))
            elif pid["diagnosis"] not in ("correct", "no_ply", "unknown"):
                issues.append(("WARN", "plane_id", scene_id,
                               f"Unexpected: {pid['diagnosis']}"))

    return sr
