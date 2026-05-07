#!/usr/bin/env python3
"""
ScanNet++ Dataset Validity Report
==================================

Checks every component of the ScanNet++ dataset for every split and generates
a comprehensive report covering:

1. Scene-level checks:
   - Scene RGB directory exists
   - GT plane H5 exists (rendered.h5)
   - Semantic H5 exists (rendered_sem.h5)
   - Depth H5 exists (rendered_depth.h5)
   - Pose/intrinsics JSON exists (pose_intrinsic_imu.json)

2. Frame-level checks:
   - RGB JPG file exists on disk
   - Frame has intrinsics/pose in JSON
   - Plane label readable from GT H5
   - Depth readable from GT H5
   - Semantic label readable from GT H5
   - RGB data quality (readable, dimensions)
   - Depth data quality (inf/nan/zero/negative)
   - Plane label quality (all-zero, negative labels)

3. Prediction H5 checks (for each inference method):
   - Prediction H5 exists per scene
   - Frame count matches GT
   - Frame IDs match GT

Usage:
    python scannetpp_dataset_validity_report.py                      # Full report
    python scannetpp_dataset_validity_report.py --splits val         # Val split only
    python scannetpp_dataset_validity_report.py --sample-frames 5    # Sample N frames/scene for content
    python scannetpp_dataset_validity_report.py --skip-frame-content # Fast mode (existence only)
    python scannetpp_dataset_validity_report.py --check-predictions  # Also check prediction H5s

Note on split files:
    ScanNet++ has TWO naming conventions for split files:
    - Dataset class uses: nvs_sem_{split}_with_planes.txt
    - Simple splits:      {split}.txt  (train/val/test)
    This script checks BOTH and reports which exist.
"""

import os
import sys
import argparse
import h5py
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from planamono.paths import scannetpp_path, scannetpp_rend_plane_path

# ============================================================
# PATHS (same as evaluate_all_baselines.py)
# ============================================================
RGB_ROOT = os.path.join(scannetpp_path, "data")
GT_ROOT = scannetpp_rend_plane_path  # plane + depth H5s
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "../../splits/scannetpp")

# Prediction H5 roots (for --check-predictions)
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
PREDICTION_FOLDERS = {
    "ours": "moge_ours_v2_h5",
    "moge_mixed_bce": "moge_mixed_bce_h5",
    "zeroplane": "zeroplane_h5",
    "zeroplane_mixed": "zeroplane_mixed_h5",
    "zeroplane_mixed_dust3r": "zeroplane_mixed_dust3r_h5",
    "gtseg": "gtseg_v1_h5",
}


def load_split(split: str) -> Tuple[List[str], str]:
    """Load scene IDs from a split file.

    Returns (scene_ids, split_file_used).
    Tries nvs_sem_{split}_with_planes.txt first (what the dataset class uses),
    then falls back to {split}.txt.
    """
    # Primary: what ScanNetPPPlaneDataset uses
    primary = os.path.join(SPLIT_DIR, f"nvs_sem_{split}_with_planes.txt")
    if os.path.exists(primary):
        with open(primary) as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids, primary

    # Fallback: simple split file
    fallback = os.path.join(SPLIT_DIR, f"{split}.txt")
    if os.path.exists(fallback):
        with open(fallback) as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids, fallback

    raise FileNotFoundError(
        f"No split file found for '{split}'. "
        f"Tried: {primary}, {fallback}"
    )


# ============================================================
# SCENE-LEVEL CHECKS
# ============================================================

def check_scene(scene_id: str) -> Dict:
    """Check scene-level directory and file existence."""
    rgb_dir = os.path.join(RGB_ROOT, scene_id, "iphone", "rgb")
    plane_h5 = os.path.join(GT_ROOT, scene_id, "rendered.h5")
    sem_h5 = os.path.join(GT_ROOT, scene_id, "rendered_sem.h5")
    depth_h5 = os.path.join(GT_ROOT, scene_id, "rendered_depth.h5")
    pose_json = os.path.join(RGB_ROOT, scene_id, "iphone", "pose_intrinsic_imu.json")

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
        # H5 metadata
        "plane_h5_readable": False,
        "plane_n_frames": 0,
        "plane_frame_ids": [],
        "plane_shape": None,
        "depth_h5_readable": False,
        "depth_n_frames": 0,
        "sem_h5_readable": False,
        "sem_n_frames": 0,
        # Pose metadata
        "pose_n_frames": 0,
    }

    # Check plane H5
    if result["plane_h5_exists"]:
        try:
            with h5py.File(plane_h5, "r") as f:
                frame_ids = [
                    fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                    for fid in f["frame_ids"][:]
                ]
                planes_shape = f["planes"].shape
            result["plane_h5_readable"] = True
            result["plane_n_frames"] = len(frame_ids)
            result["plane_frame_ids"] = frame_ids
            result["plane_shape"] = planes_shape
        except Exception as e:
            result["plane_h5_error"] = str(e)

    # Check depth H5
    if result["depth_h5_exists"]:
        try:
            with h5py.File(depth_h5, "r") as f:
                depth_shape = f["depth"].shape
            result["depth_h5_readable"] = True
            result["depth_n_frames"] = depth_shape[0]
        except Exception as e:
            result["depth_h5_error"] = str(e)

    # Check sem H5
    if result["sem_h5_exists"]:
        try:
            with h5py.File(sem_h5, "r") as f:
                sem_shape = f["sem"].shape
            result["sem_h5_readable"] = True
            result["sem_n_frames"] = sem_shape[0]
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
# FRAME-LEVEL CHECKS
# ============================================================

def check_frame_existence(scene_id: str, frame_ids: List[str],
                          pose_data: Optional[Dict]) -> List[Dict]:
    """Check existence of RGB files and pose data for each frame."""
    rgb_dir = os.path.join(RGB_ROOT, scene_id, "iphone", "rgb")
    results = []
    for fid in frame_ids:
        rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
        has_pose = pose_data is not None and fid in pose_data
        has_intrinsic = False
        if has_pose:
            has_intrinsic = "intrinsic" in pose_data.get(fid, {})
        results.append({
            "scene_id": scene_id,
            "frame_id": fid,
            "rgb_exists": os.path.isfile(rgb_path),
            "rgb_path": rgb_path,
            "has_pose": has_pose,
            "has_intrinsic": has_intrinsic,
        })
    return results


def check_frame_content(scene_id: str, frame_id: str, frame_idx: int,
                        plane_h5_path: str, depth_h5_path: str,
                        sem_h5_path: str, rgb_path: str) -> Dict:
    """Check data quality of a single frame (reads actual files)."""
    import cv2

    result = {
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
        "depth_unit": "mm",
        "plane_readable": False,
        "plane_shape": None,
        "plane_n_labels": 0,
        "plane_has_negative": False,
        "plane_all_zero": False,
        "plane_pct_planar": 0.0,
        "sem_readable": False,
        "sem_shape": None,
        "sem_n_classes": 0,
    }

    # Check RGB
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

    # Check plane labels
    if os.path.isfile(plane_h5_path):
        try:
            with h5py.File(plane_h5_path, "r") as f:
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

    # Check depth
    if os.path.isfile(depth_h5_path):
        try:
            with h5py.File(depth_h5_path, "r") as f:
                depth = f["depth"][frame_idx].astype(np.float32)
            # Depth is in mm in the H5
            result["depth_readable"] = True
            result["depth_shape"] = depth.shape

            n_pixels = depth.size
            n_inf = np.isinf(depth).sum()
            n_nan = np.isnan(depth).sum()
            n_neg = (depth < 0).sum()
            n_zero = (depth == 0).sum()

            result["depth_has_inf"] = bool(n_inf > 0)
            result["depth_has_nan"] = bool(n_nan > 0)
            result["depth_has_negative"] = bool(n_neg > 0)
            result["depth_has_zero"] = bool(n_zero > 0)
            result["depth_pct_invalid"] = float(
                (n_inf + n_nan + n_neg + n_zero) / n_pixels * 100
            )

            valid_mask = np.isfinite(depth) & (depth > 0)
            if valid_mask.any():
                # Convert mm → m for reporting
                result["depth_min"] = float(depth[valid_mask].min() / 1000.0)
                result["depth_max"] = float(depth[valid_mask].max() / 1000.0)
                result["depth_median"] = float(np.median(depth[valid_mask]) / 1000.0)
        except Exception as e:
            result["depth_error"] = str(e)

    # Check semantics
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
# PREDICTION H5 CHECKS
# ============================================================

def check_predictions(scene_id: str, gt_frame_ids: List[str],
                      method_name: str, h5_folder: str) -> Dict:
    """Check prediction H5 for a specific method/scene."""
    h5_path = os.path.join(str(H5_ROOT), h5_folder, scene_id, "planes.h5")
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
# REPORT GENERATION
# ============================================================

def generate_report(splits: List[str], sample_frames: Optional[int],
                    skip_frame_content: bool, check_preds: bool) -> str:
    """Generate a full validity report."""
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 80)
    out("SCANNET++ DATASET VALIDITY REPORT")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 80)
    out()

    # --- Path verification ---
    out("## PATH VERIFICATION")
    out(f"  RGB_ROOT:   {RGB_ROOT}  {'[OK]' if os.path.isdir(RGB_ROOT) else '[MISSING]'}")
    out(f"  GT_ROOT:    {GT_ROOT}  {'[OK]' if os.path.isdir(GT_ROOT) else '[MISSING]'}")
    out(f"  SPLIT_DIR:  {SPLIT_DIR}  {'[OK]' if os.path.isdir(SPLIT_DIR) else '[MISSING]'}")
    if check_preds:
        out(f"  H5_ROOT:    {H5_ROOT}  {'[OK]' if H5_ROOT.is_dir() else '[MISSING]'}")
    out()

    # --- Split file inventory ---
    out("## SPLIT FILE INVENTORY")
    for name in ["nvs_sem_train_with_planes", "nvs_sem_val_with_planes",
                 "nvs_sem_test_with_planes",
                 "nvs_sem_train_missing_planes", "nvs_sem_val_missing_planes",
                 "train", "val", "test", "all_scenes"]:
        path = os.path.join(SPLIT_DIR, f"{name}.txt")
        if os.path.isfile(path):
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
            out(f"  {name}.txt: {n} scenes [OK]")
        else:
            out(f"  {name}.txt: [MISSING]")
    out()

    grand_totals = {
        "scenes": 0,
        "scenes_missing_rgb_dir": 0,
        "scenes_missing_plane_h5": 0,
        "scenes_missing_sem_h5": 0,
        "scenes_missing_depth_h5": 0,
        "scenes_missing_pose_json": 0,
        "scenes_h5_frame_count_mismatch": 0,
        "frames_total": 0,
        "frames_rgb_missing": 0,
        "frames_pose_missing": 0,
        "frames_content_checked": 0,
        "frames_rgb_unreadable": 0,
        "frames_depth_bad": 0,
        "frames_plane_all_zero": 0,
        "frames_plane_negative": 0,
    }

    for split in splits:
        out("=" * 80)
        out(f"SPLIT: {split.upper()}")
        out("=" * 80)
        out()

        try:
            scene_ids, split_file_used = load_split(split)
        except FileNotFoundError as e:
            out(f"  [ERROR] {e}")
            out()
            continue

        out(f"  Split file used: {os.path.basename(split_file_used)}")
        out(f"  Scenes in split: {len(scene_ids)}")
        out()

        # Tracking
        scenes_missing_rgb = []
        scenes_missing_plane = []
        scenes_missing_sem = []
        scenes_missing_depth = []
        scenes_missing_pose = []
        scenes_h5_unreadable = []
        scenes_frame_count_mismatch = []

        frames_rgb_missing = []
        frames_pose_missing = []
        frames_rgb_unreadable = []
        frames_depth_bad = []
        frames_plane_all_zero = []
        frames_plane_negative = []

        depth_mins = []
        depth_maxs = []
        plane_pct_planars = []

        total_gt_frames = 0
        total_valid_frames = 0
        total_content_checked = 0

        for si, scene_id in enumerate(scene_ids):
            print(f"  [{split}] Checking scene {si+1}/{len(scene_ids)}: {scene_id}")

            scene_check = check_scene(scene_id)

            # --- Scene-level issues ---
            if not scene_check["rgb_dir_exists"]:
                scenes_missing_rgb.append(scene_id)
            if not scene_check["plane_h5_exists"]:
                scenes_missing_plane.append(scene_id)
                continue  # No GT → skip further checks
            if not scene_check["sem_h5_exists"]:
                scenes_missing_sem.append(scene_id)
            if not scene_check["depth_h5_exists"]:
                scenes_missing_depth.append(scene_id)
            if not scene_check["pose_json_exists"]:
                scenes_missing_pose.append(scene_id)

            if not scene_check["plane_h5_readable"]:
                scenes_h5_unreadable.append(scene_id)
                continue

            # Check frame count consistency across H5 files
            n_plane = scene_check["plane_n_frames"]
            n_depth = scene_check["depth_n_frames"]
            n_sem = scene_check["sem_n_frames"]
            if n_depth > 0 and n_depth != n_plane:
                scenes_frame_count_mismatch.append(
                    f"{scene_id}: plane={n_plane}, depth={n_depth}"
                )
            if n_sem > 0 and n_sem != n_plane:
                scenes_frame_count_mismatch.append(
                    f"{scene_id}: plane={n_plane}, sem={n_sem}"
                )

            frame_ids = scene_check["plane_frame_ids"]
            total_gt_frames += len(frame_ids)

            # --- Load pose JSON if available ---
            pose_data = None
            if scene_check["pose_json_exists"]:
                try:
                    with open(scene_check["pose_json_path"]) as f:
                        pose_data = json.load(f)
                except Exception:
                    pass

            # --- Frame-level existence ---
            frame_existence = check_frame_existence(scene_id, frame_ids, pose_data)

            for fe in frame_existence:
                if not fe["rgb_exists"]:
                    frames_rgb_missing.append(f"{scene_id}/{fe['frame_id']}")
                elif not fe["has_pose"]:
                    frames_pose_missing.append(f"{scene_id}/{fe['frame_id']}")
                else:
                    total_valid_frames += 1

            # --- Frame-level content (optional) ---
            if not skip_frame_content:
                valid_frames = [
                    fe for fe in frame_existence
                    if fe["rgb_exists"]
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
                        scene_id, fe["frame_id"], frame_h5_idx,
                        scene_check["plane_h5_path"],
                        scene_check["depth_h5_path"],
                        scene_check["sem_h5_path"],
                        fe["rgb_path"],
                    )
                    total_content_checked += 1

                    fid_str = f"{scene_id}/{fe['frame_id']}"

                    if not content["rgb_readable"]:
                        frames_rgb_unreadable.append(fid_str)

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
                        scene_id, frame_ids, method_name, h5_folder
                    )
                    key_prefix = f"pred_{method_name}"
                    if not pred_check["pred_h5_exists"]:
                        grand_totals.setdefault(f"{key_prefix}_missing", 0)
                        grand_totals[f"{key_prefix}_missing"] = grand_totals.get(f"{key_prefix}_missing", 0) + 1
                    elif not pred_check["pred_h5_readable"]:
                        grand_totals[f"{key_prefix}_unreadable"] = grand_totals.get(f"{key_prefix}_unreadable", 0) + 1
                    else:
                        grand_totals[f"{key_prefix}_ok"] = grand_totals.get(f"{key_prefix}_ok", 0) + 1
                        if not pred_check["pred_frame_ids_match"]:
                            grand_totals[f"{key_prefix}_frame_mismatch"] = grand_totals.get(f"{key_prefix}_frame_mismatch", 0) + 1

        # ========== REPORT FOR THIS SPLIT ==========

        n_scenes = len(scene_ids)
        n_with_gt = n_scenes - len(scenes_missing_plane)

        out(f"  ### 1. SCENE-LEVEL SUMMARY")
        out(f"     Total scenes:                    {n_scenes}")
        out(f"     Scenes with RGB dir:             {n_scenes - len(scenes_missing_rgb)}")
        out(f"     Scenes with GT plane H5:         {n_with_gt}")
        out(f"     Scenes with semantic H5:         {n_with_gt - len(scenes_missing_sem)}")
        out(f"     Scenes with depth H5:            {n_with_gt - len(scenes_missing_depth)}")
        out(f"     Scenes with pose JSON:           {n_scenes - len(scenes_missing_pose)}")
        out()

        if scenes_missing_rgb:
            out(f"     [ISSUE] Scenes MISSING RGB dir ({len(scenes_missing_rgb)}):")
            for s in scenes_missing_rgb[:20]:
                out(f"       - {s}")
            if len(scenes_missing_rgb) > 20:
                out(f"       ... and {len(scenes_missing_rgb) - 20} more")
            out()

        if scenes_missing_plane:
            out(f"     [ISSUE] Scenes MISSING GT plane H5 ({len(scenes_missing_plane)}):")
            for s in scenes_missing_plane[:20]:
                out(f"       - {s}")
            if len(scenes_missing_plane) > 20:
                out(f"       ... and {len(scenes_missing_plane) - 20} more")
            out()

        if scenes_missing_sem:
            out(f"     [WARN] Scenes missing semantic H5 ({len(scenes_missing_sem)}):")
            for s in scenes_missing_sem[:10]:
                out(f"       - {s}")
            if len(scenes_missing_sem) > 10:
                out(f"       ... and {len(scenes_missing_sem) - 10} more")
            out()

        if scenes_missing_depth:
            out(f"     [WARN] Scenes missing depth H5 ({len(scenes_missing_depth)}):")
            for s in scenes_missing_depth[:10]:
                out(f"       - {s}")
            if len(scenes_missing_depth) > 10:
                out(f"       ... and {len(scenes_missing_depth) - 10} more")
            out()

        if scenes_missing_pose:
            out(f"     [WARN] Scenes missing pose JSON ({len(scenes_missing_pose)}):")
            for s in scenes_missing_pose[:10]:
                out(f"       - {s}")
            if len(scenes_missing_pose) > 10:
                out(f"       ... and {len(scenes_missing_pose) - 10} more")
            out()

        if scenes_h5_unreadable:
            out(f"     [ISSUE] Scenes with unreadable plane H5 ({len(scenes_h5_unreadable)}):")
            for s in scenes_h5_unreadable:
                out(f"       - {s}")
            out()

        if scenes_frame_count_mismatch:
            out(f"     [WARN] H5 frame count mismatches ({len(scenes_frame_count_mismatch)}):")
            for s in scenes_frame_count_mismatch[:10]:
                out(f"       - {s}")
            if len(scenes_frame_count_mismatch) > 10:
                out(f"       ... and {len(scenes_frame_count_mismatch) - 10} more")
            out()

        out(f"  ### 2. FRAME-LEVEL SUMMARY")
        out(f"     Total frames (from GT H5):       {total_gt_frames}")
        out(f"     Frames with RGB on disk:          {total_gt_frames - len(frames_rgb_missing)}")
        out(f"     Frames with RGB + pose:           {total_valid_frames}")
        out(f"     Frames missing RGB:               {len(frames_rgb_missing)}")
        out(f"     Frames missing pose/intrinsics:   {len(frames_pose_missing)}")
        out()

        if frames_rgb_missing:
            missing_by_scene = defaultdict(list)
            for f in frames_rgb_missing:
                parts = f.split("/")
                missing_by_scene[parts[0]].append(parts[1])

            out(f"     [ISSUE] Frames missing RGB ({len(frames_rgb_missing)}):")
            for sc, fids in sorted(missing_by_scene.items()):
                out(f"       {sc}: {len(fids)} frames (e.g., {', '.join(fids[:5])}{'...' if len(fids) > 5 else ''})")
            out()

        if frames_pose_missing:
            missing_by_scene = defaultdict(list)
            for f in frames_pose_missing:
                parts = f.split("/")
                missing_by_scene[parts[0]].append(parts[1])

            out(f"     [WARN] Frames missing pose ({len(frames_pose_missing)}):")
            for sc, fids in sorted(missing_by_scene.items())[:15]:
                out(f"       {sc}: {len(fids)} frames")
            if len(missing_by_scene) > 15:
                out(f"       ... and {len(missing_by_scene) - 15} more scenes")
            out()

        # --- Content quality ---
        if not skip_frame_content:
            out(f"  ### 3. DATA QUALITY (content checks)")
            out(f"     Frames checked:                  {total_content_checked}")
            out()

            if frames_rgb_unreadable:
                out(f"     [ISSUE] RGB files unreadable ({len(frames_rgb_unreadable)}):")
                for fid_str in frames_rgb_unreadable[:10]:
                    out(f"       {fid_str}")
                out()
            else:
                out(f"     [OK] All checked RGB files are readable")
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
                for fid_str in frames_plane_negative[:10]:
                    out(f"       {fid_str}")
                out()
            else:
                out(f"     [OK] No negative plane label values found")
                out()

            if depth_mins:
                out(f"     Depth range (across checked frames, in meters):")
                out(f"       Min depth:    {np.min(depth_mins):.4f} m")
                out(f"       Max depth:    {np.max(depth_maxs):.4f} m")
                out(f"       Median min:   {np.median(depth_mins):.4f} m")
                out(f"       Median max:   {np.median(depth_maxs):.4f} m")
                out()

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
            out(f"  ### {'4' if not skip_frame_content else '3'}. PREDICTION H5 CHECKS")
            for method_name, h5_folder in PREDICTION_FOLDERS.items():
                n_ok = grand_totals.get(f"pred_{method_name}_ok", 0)
                n_missing = grand_totals.get(f"pred_{method_name}_missing", 0)
                n_unreadable = grand_totals.get(f"pred_{method_name}_unreadable", 0)
                n_mismatch = grand_totals.get(f"pred_{method_name}_frame_mismatch", 0)

                status = "[OK]" if n_missing == 0 and n_unreadable == 0 else "[ISSUE]"
                out(f"     {status} {method_name} ({h5_folder}):")
                out(f"       Scenes with predictions: {n_ok}/{n_with_gt}")
                if n_missing > 0:
                    out(f"       Missing H5: {n_missing}")
                if n_unreadable > 0:
                    out(f"       Unreadable H5: {n_unreadable}")
                if n_mismatch > 0:
                    out(f"       Frame ID mismatches: {n_mismatch}")
                out()

        # Update grand totals
        grand_totals["scenes"] += n_scenes
        grand_totals["scenes_missing_rgb_dir"] += len(scenes_missing_rgb)
        grand_totals["scenes_missing_plane_h5"] += len(scenes_missing_plane)
        grand_totals["scenes_missing_sem_h5"] += len(scenes_missing_sem)
        grand_totals["scenes_missing_depth_h5"] += len(scenes_missing_depth)
        grand_totals["scenes_missing_pose_json"] += len(scenes_missing_pose)
        grand_totals["scenes_h5_frame_count_mismatch"] += len(scenes_frame_count_mismatch)
        grand_totals["frames_total"] += total_gt_frames
        grand_totals["frames_rgb_missing"] += len(frames_rgb_missing)
        grand_totals["frames_pose_missing"] += len(frames_pose_missing)
        grand_totals["frames_content_checked"] += total_content_checked
        grand_totals["frames_rgb_unreadable"] += len(frames_rgb_unreadable)
        grand_totals["frames_depth_bad"] += len(frames_depth_bad)
        grand_totals["frames_plane_all_zero"] += len(frames_plane_all_zero)
        grand_totals["frames_plane_negative"] += len(frames_plane_negative)

        out()

    # ============================================================
    # GRAND SUMMARY
    # ============================================================
    out("=" * 80)
    out("GRAND SUMMARY (ALL SPLITS)")
    out("=" * 80)
    out()
    out(f"  Splits checked:             {', '.join(splits)}")
    out(f"  Total scenes:               {grand_totals['scenes']}")
    out(f"  Scenes missing RGB dir:     {grand_totals['scenes_missing_rgb_dir']}")
    out(f"  Scenes missing plane H5:    {grand_totals['scenes_missing_plane_h5']}")
    out(f"  Scenes missing sem H5:      {grand_totals['scenes_missing_sem_h5']}")
    out(f"  Scenes missing depth H5:    {grand_totals['scenes_missing_depth_h5']}")
    out(f"  Scenes missing pose JSON:   {grand_totals['scenes_missing_pose_json']}")
    out(f"  H5 frame count mismatches:  {grand_totals['scenes_h5_frame_count_mismatch']}")
    out()
    out(f"  Total frames (GT H5):       {grand_totals['frames_total']}")
    out(f"  Frames missing RGB:         {grand_totals['frames_rgb_missing']}")
    out(f"  Frames missing pose:        {grand_totals['frames_pose_missing']}")
    usable = grand_totals["frames_total"] - grand_totals["frames_rgb_missing"] - grand_totals["frames_pose_missing"]
    pct = (usable / grand_totals["frames_total"] * 100) if grand_totals["frames_total"] > 0 else 0
    out(f"  Frames usable (RGB+pose):   {usable} ({pct:.1f}%)")
    out()

    if not skip_frame_content:
        out(f"  Content-checked frames:     {grand_totals['frames_content_checked']}")
        out(f"  RGB unreadable:             {grand_totals['frames_rgb_unreadable']}")
        out(f"  Depth with >5% invalid:     {grand_totals['frames_depth_bad']}")
        out(f"  Plane labels all-zero:      {grand_totals['frames_plane_all_zero']}")
        out(f"  Plane labels negative:      {grand_totals['frames_plane_negative']}")
        out()

    # Dataset class behavior
    out("=" * 80)
    out("NOTES ON DATASET CLASS BEHAVIOR (ScanNetPPPlaneDataset)")
    out("=" * 80)
    out()
    out("  __init__ filtering (silent drops, frame never enters valid_pairs):")
    out("    - Missing RGB dir:          skipped (line 56-58 / 194-198)")
    out("    - Missing plane H5:         skipped (line 59-61 / 194-198)")
    out("    - Missing semantic H5:      skipped (line 62-64 / 194-198)")
    out("    - Missing depth H5:         skipped (line 65-67 / 194-198)")
    out("    - Missing pose JSON:        skipped (line 194-198, PlaneDataset only)")
    out("    - H5 frame_ids unreadable:  skipped (line 69-74 / 204-209)")
    out("    - RGB file missing:         skipped (line 79-80 / 214-215)")
    out("    - Pose missing for frame:   skipped (line 216-218, PlaneDataset only)")
    out()
    out("  __getitem__ fallbacks (returns zeros, sample looks valid):")
    out("    - Plane H5 read fails:  plane = zeros (line 248-251)")
    out("    - Sem H5 read fails:    sem = zeros (line 259-261)")
    out("    - Depth H5 read fails:  depth = zeros (line 268-270)")
    out("    - RGB read fails:       RAISES RuntimeError (line 274-275)")
    out()
    out("  Key difference from Hypersim: RGB failure RAISES instead of returning zeros.")
    out("  This means a corrupt JPG will crash the DataLoader, not silently corrupt metrics.")
    out()
    out("  Split file naming:")
    out("    - Dataset class uses: nvs_sem_{split}_with_planes.txt")
    out("    - Eval script uses split='test' → needs nvs_sem_test_with_planes.txt")
    out(f"    - nvs_sem_test_with_planes.txt exists: {os.path.isfile(os.path.join(SPLIT_DIR, 'nvs_sem_test_with_planes.txt'))}")
    out()

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ScanNet++ dataset validity report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Which splits to check (default: train val test). "
             "Uses nvs_sem_{split}_with_planes.txt if available, else {split}.txt",
    )
    parser.add_argument(
        "--sample-frames", type=int, default=5,
        help="Number of frames to sample per scene for content checks "
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
        help="Output file path (default: report_scannetpp_validity.txt)",
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
        os.path.dirname(__file__), "report_scannetpp_validity.txt"
    )
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
