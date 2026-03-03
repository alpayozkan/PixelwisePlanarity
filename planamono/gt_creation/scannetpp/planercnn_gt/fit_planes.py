"""
Stage 1: PlaneRCNN-style plane fitting on ScanNet++ meshes.

Produces:
  - planes.ply  (binary PLY with per-face plane_id + label_int)
  - planes.json (plane metadata)

Uses the same PlaneRCNN algorithm as original_planercnn_for_scannetpp.py
but outputs in the project's standard format (compatible with render_scene.py
and read_ply_faces_with_plane_ids).

Supports YAML configuration via --config flag (see planercnn_default.yml).
Without --config, uses the original hardcoded parameters from parse_scannetpp.
"""

import argparse
import os
import json
import numpy as np
import yaml
from tqdm import tqdm
from scipy import stats as sp_stats

from planamono.shared.parsers.parse_scannetpp import (
    readMesh_scannetpp,
    fitPlane,
    ransac_planes_for_segment,
    labelNumPlanes,
    nonPlanar,
    numPlanesPerSegment,
    planeAreaThreshold,
    fittingErrorThreshold,
    orthogonalThreshold,
)
from planamono.gt_creation.scannetpp.plane_extraction import (
    _write_ply_with_face_pid_binary,
)
from planamono.paths import scannetppv2_path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "planercnn_default.yml")

# Default output root (used when no config is provided)
DEFAULT_OUTPUT_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp_planercnn"


def load_config(config_path):
    """Load YAML config and return a dict with all parameters."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def fit_planes_planercnn(scene_id, root_dir, metadata_dir):
    """
    Run PlaneRCNN plane fitting on a ScanNet++ scene.

    Returns:
        points: (N, 3) vertex positions
        faces: (M, 3) face indices
        planeSeg: (N,) per-vertex plane IDs (-1 = non-planar, 0..K-1 = plane)
        planes: (K, 3) plane parameters in Hesse form (ax+by+cz=1)
        vertex_labels: (N,) per-vertex semantic labels
    """
    points, faces, segmentation, groupSegments, groupLabels, vertex_labels, classLabelMap = \
        readMesh_scannetpp(scene_id, root_dir, metadata_dir)

    planeSeg = np.full(len(points), -1, dtype=np.int32)
    all_planes = []    # list of (plane_vec_3,)
    all_indices = []   # list of vertex-index arrays

    for gi, (segments_in_group, raw_label) in tqdm(
        enumerate(zip(groupSegments, groupLabels)),
        total=len(groupSegments),
        desc="Fitting planes",
        ncols=100,
    ):
        label = raw_label if raw_label in labelNumPlanes else (
            'unannotated' if raw_label == '' else raw_label
        )
        minP, maxP = labelNumPlanes.get(label, [0, 5])
        if label in nonPlanar:
            minP, maxP = 0, 0

        seg_idx = np.asarray(segments_in_group, dtype=np.int32)
        if seg_idx.size < planeAreaThreshold or maxP == 0:
            # Too small or non-planar class → skip (stays -1)
            continue

        XYZ = points[seg_idx]

        # --- Try single LS fit ---
        try:
            p = fitPlane(XYZ)
            err = np.mean(np.abs(XYZ @ p - 1.0) / max(np.linalg.norm(p), 1e-6))
        except np.linalg.LinAlgError:
            p, err = np.zeros(3), 1e9

        group_planes = []
        group_indices = []

        if err < fittingErrorThreshold or maxP == 1:
            if err < 1e8:
                group_planes.append(p)
                group_indices.append(seg_idx)
            # else: stays non-planar
        else:
            # RANSAC fallback
            planes_r, idx_r, remainder = ransac_planes_for_segment(
                XYZ, seg_idx, K=numPlanesPerSegment
            )
            explained = sum(len(x) for x in idx_r)
            if explained >= 0.5 * len(seg_idx):
                group_planes.extend(planes_r)
                group_indices.extend(idx_r)
            # else: stays non-planar

        # --- Enforce min/max constraints ---
        num_real = sum(np.linalg.norm(pp) > 1e-4 for pp in group_planes)

        if minP == 1 and num_real == 0:
            # Force a plane from the largest subset
            if seg_idx.size >= planeAreaThreshold:
                try:
                    forced = fitPlane(XYZ)
                    group_planes = [forced]
                    group_indices = [seg_idx]
                except np.linalg.LinAlgError:
                    pass

        if minP == 1 and maxP == 1 and len(group_planes) > 1:
            # Collapse to single plane (e.g. floor)
            all_ids = np.concatenate(group_indices)
            try:
                one = fitPlane(points[all_ids])
                d = np.abs(points[all_ids] @ one - 1.0) / max(np.linalg.norm(one), 1e-6)
                scale = 3.0 if label == 'floor' else 1.0
                if d.mean() < fittingErrorThreshold * scale:
                    group_planes = [one]
                    group_indices = [all_ids]
            except np.linalg.LinAlgError:
                pass

        # Filter out dummy planes (zero-norm)
        for pp, ii in zip(group_planes, group_indices):
            if np.linalg.norm(pp) > 1e-4:
                all_planes.append(pp)
                all_indices.append(ii)

    # --- Assign global plane IDs ---
    planes = np.array(all_planes) if all_planes else np.zeros((0, 3))
    for k, idxs in enumerate(all_indices):
        planeSeg[idxs] = k

    print(f"[INFO] {len(all_planes)} planes fitted for scene {scene_id}")
    return points, faces, planeSeg, planes, vertex_labels


# ---------------------------------------------------------------------------
# Config-aware variants (used when --config is provided)
# ---------------------------------------------------------------------------

def ransac_planes_for_segment_cfg(XYZ, global_indices, K, cfg):
    """
    RANSAC plane extraction using parameters from YAML config.

    Same algorithm as ransac_planes_for_segment() in parse_scannetpp.py,
    but reads thresholds from the config dict instead of module globals.
    """
    _num_iterations = cfg["num_iterations"]
    _plane_diff_threshold = cfg["plane_diff_threshold"]
    _plane_area_threshold = cfg["plane_area_threshold"]

    planes = []
    plane_point_indices = []
    remaining_mask = np.ones(XYZ.shape[0], dtype=bool)

    for _ in range(K):
        pts = XYZ[remaining_mask]
        if pts.shape[0] < _plane_area_threshold:
            break

        best_num = 0
        best_inliers_mask_local = None
        best_plane = None

        for it in range(_num_iterations):
            choice = np.random.choice(np.arange(len(pts)), size=3, replace=False)
            try:
                cand = fitPlane(pts[choice])
            except np.linalg.LinAlgError:
                continue

            denom = max(np.linalg.norm(cand), 1e-6)
            d = np.abs(pts @ cand - 1.0) / denom
            inliers = d < _plane_diff_threshold
            num_inliers = int(inliers.sum())
            if num_inliers > best_num:
                best_num = num_inliers
                best_inliers_mask_local = inliers
                best_plane = cand

        if best_num < _plane_area_threshold or best_plane is None:
            break

        inlier_xyz_all = XYZ[remaining_mask][best_inliers_mask_local]
        refined = fitPlane(inlier_xyz_all)
        planes.append(refined)

        local2global = np.where(remaining_mask)[0][best_inliers_mask_local]
        plane_point_indices.append(global_indices[local2global])
        remaining_mask[np.where(remaining_mask)[0][best_inliers_mask_local]] = False

    remainder_indices = global_indices[remaining_mask]
    return planes, plane_point_indices, remainder_indices


def fit_planes_planercnn_cfg(scene_id, root_dir, metadata_dir, cfg):
    """
    Config-aware PlaneRCNN plane fitting.

    Same algorithm as fit_planes_planercnn(), but all thresholds and
    label mappings are read from the YAML config dict.
    """
    _num_planes_per_segment = cfg["num_planes_per_segment"]
    _plane_area_threshold = cfg["plane_area_threshold"]
    _fitting_error_threshold = cfg["fitting_error_threshold"]
    _coverage_gate = cfg["coverage_gate"]
    _floor_error_scale = cfg["floor_error_scale"]
    _label_num_planes = cfg["label_num_planes"]
    _non_planar_labels = set(cfg["non_planar_labels"])

    points, faces, segmentation, groupSegments, groupLabels, vertex_labels, classLabelMap = \
        readMesh_scannetpp(scene_id, root_dir, metadata_dir)

    planeSeg = np.full(len(points), -1, dtype=np.int32)
    all_planes = []
    all_indices = []

    for gi, (segments_in_group, raw_label) in tqdm(
        enumerate(zip(groupSegments, groupLabels)),
        total=len(groupSegments),
        desc="Fitting planes",
        ncols=100,
    ):
        label = raw_label if raw_label in _label_num_planes else (
            'unannotated' if raw_label == '' else raw_label
        )
        minP, maxP = _label_num_planes.get(label, [0, 5])
        if label in _non_planar_labels:
            minP, maxP = 0, 0

        seg_idx = np.asarray(segments_in_group, dtype=np.int32)
        if seg_idx.size < _plane_area_threshold or maxP == 0:
            continue

        XYZ = points[seg_idx]

        # --- Try single LS fit ---
        try:
            p = fitPlane(XYZ)
            err = np.mean(np.abs(XYZ @ p - 1.0) / max(np.linalg.norm(p), 1e-6))
        except np.linalg.LinAlgError:
            p, err = np.zeros(3), 1e9

        group_planes = []
        group_indices = []

        if err < _fitting_error_threshold or maxP == 1:
            if err < 1e8:
                group_planes.append(p)
                group_indices.append(seg_idx)
        else:
            planes_r, idx_r, remainder = ransac_planes_for_segment_cfg(
                XYZ, seg_idx, K=_num_planes_per_segment, cfg=cfg
            )
            explained = sum(len(x) for x in idx_r)
            if explained >= _coverage_gate * len(seg_idx):
                group_planes.extend(planes_r)
                group_indices.extend(idx_r)

        # --- Enforce min/max constraints ---
        num_real = sum(np.linalg.norm(pp) > 1e-4 for pp in group_planes)

        if minP == 1 and num_real == 0:
            if seg_idx.size >= _plane_area_threshold:
                try:
                    forced = fitPlane(XYZ)
                    group_planes = [forced]
                    group_indices = [seg_idx]
                except np.linalg.LinAlgError:
                    pass

        if minP == 1 and maxP == 1 and len(group_planes) > 1:
            all_ids = np.concatenate(group_indices)
            try:
                one = fitPlane(points[all_ids])
                d = np.abs(points[all_ids] @ one - 1.0) / max(np.linalg.norm(one), 1e-6)
                scale = _floor_error_scale if label == 'floor' else 1.0
                if d.mean() < _fitting_error_threshold * scale:
                    group_planes = [one]
                    group_indices = [all_ids]
            except np.linalg.LinAlgError:
                pass

        for pp, ii in zip(group_planes, group_indices):
            if np.linalg.norm(pp) > 1e-4:
                all_planes.append(pp)
                all_indices.append(ii)

    planes = np.array(all_planes) if all_planes else np.zeros((0, 3))
    for k, idxs in enumerate(all_indices):
        planeSeg[idxs] = k

    print(f"[INFO] {len(all_planes)} planes fitted for scene {scene_id}")
    return points, faces, planeSeg, planes, vertex_labels


def vertex_to_face_plane_ids(planeSeg, faces):
    """
    Convert per-vertex plane IDs to per-face plane IDs.
    A face is planar only if all 3 vertices agree on the same plane (>= 0).
    All faces are kept (non-planar faces get plane_id = -1).

    Returns:
        face_pid: (M,) int32, -1 = non-planar, 0..K-1 = plane
    """
    v0 = planeSeg[faces[:, 0]]
    v1 = planeSeg[faces[:, 1]]
    v2 = planeSeg[faces[:, 2]]
    agree = (v0 == v1) & (v1 == v2) & (v0 >= 0)
    face_pid = np.where(agree, v0, -1).astype(np.int32)
    return face_pid


def compute_face_semantic_labels(vertex_labels, faces):
    """
    Per-face semantic label via majority vote of vertex labels.

    Returns:
        labels_f: (M,) int32
    """
    l0 = vertex_labels[faces[:, 0]]
    l1 = vertex_labels[faces[:, 1]]
    l2 = vertex_labels[faces[:, 2]]
    labels_stack = np.stack([l0, l1, l2], axis=1)  # (M, 3)
    labels_f = sp_stats.mode(labels_stack, axis=1, keepdims=False).mode
    return labels_f.astype(np.int32)


def hesse_to_normal_form(planes):
    """
    Convert PlaneRCNN's Hesse form (ax+by+cz=1) to normal+distance.
    Returns list of dicts with 'n' (unit normal) and 'd' (distance from origin).
    """
    meta = []
    for i, p in enumerate(planes):
        norm = np.linalg.norm(p)
        if norm < 1e-8:
            meta.append({"plane_id": i, "n": [0, 0, 0], "d": 0.0})
        else:
            n = (p / norm).tolist()
            d = float(1.0 / norm)
            meta.append({"plane_id": i, "n": n, "d": d})
    return meta


def build_planes_meta(planes, face_pid, labels_f, faces, points):
    """
    Build planes_meta list for save_planes_mesh_and_json.
    """
    meta_list = hesse_to_normal_form(planes)

    # Compute per-plane area and semantic label
    for entry in meta_list:
        pid = entry["plane_id"]
        mask = face_pid == pid
        if mask.sum() == 0:
            entry["area"] = 0.0
            entry["label_int"] = -1
            entry["label_raw"] = ""
            continue

        # Triangle area
        tri_verts = faces[mask]
        v0 = points[tri_verts[:, 0]]
        v1 = points[tri_verts[:, 1]]
        v2 = points[tri_verts[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
        entry["area"] = area

        # Majority semantic label among this plane's faces
        plane_labels = labels_f[mask]
        lbl = int(sp_stats.mode(plane_labels, keepdims=False).mode)
        entry["label_int"] = lbl
        entry["label_raw"] = ""

    return meta_list


def main():
    parser = argparse.ArgumentParser(description="PlaneRCNN plane fitting → planes.ply + planes.json")
    parser.add_argument("scene_id", type=str, help="ScanNet++ scene ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: use hardcoded parameters)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root directory (overrides config)")
    args = parser.parse_args()

    scene_id = args.scene_id
    root_dir = os.path.join(scannetppv2_path, "data")
    metadata_dir = os.path.join(scannetppv2_path, "metadata")

    if args.config is not None:
        # Config-aware path
        cfg = load_config(args.config)
        output_root = args.output_root or cfg.get("mesh_root", DEFAULT_OUTPUT_ROOT)
        print(f"[INFO] Using config: {args.config}")
        points, faces, planeSeg, planes, vertex_labels = fit_planes_planercnn_cfg(
            scene_id, root_dir, metadata_dir, cfg
        )
    else:
        # Original hardcoded path
        output_root = args.output_root or DEFAULT_OUTPUT_ROOT
        points, faces, planeSeg, planes, vertex_labels = fit_planes_planercnn(
            scene_id, root_dir, metadata_dir
        )

    if planes.shape[0] == 0:
        print("[WARN] No planes found. Skipping output.")
        return

    # Convert vertex → face plane IDs (keep all faces)
    face_pid = vertex_to_face_plane_ids(planeSeg, faces)
    labels_f = compute_face_semantic_labels(vertex_labels, faces)

    # Re-index to consecutive IDs (some planes may have no faces)
    unique_pids = np.unique(face_pid[face_pid >= 0])
    if len(unique_pids) == 0:
        print("[WARN] No faces assigned to any plane. Skipping output.")
        return

    old_to_new = np.full(int(unique_pids.max()) + 1, -1, dtype=np.int32)
    for new, old in enumerate(unique_pids):
        old_to_new[old] = new
    face_pid_reindexed = np.where(
        face_pid >= 0,
        old_to_new[face_pid.clip(0)],
        -1,
    ).astype(np.int32)
    planes_reindexed = planes[unique_pids]

    print(f"[INFO] {len(unique_pids)} planes with face assignments (of {planes.shape[0]} total)")

    # Build metadata
    planes_meta = build_planes_meta(planes_reindexed, face_pid_reindexed, labels_f, faces, points)

    # Save
    out_dir = os.path.join(output_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    # Build planes_json with colors
    max_pid = int(np.max(face_pid_reindexed)) if face_pid_reindexed.size else -1
    rng = np.random.default_rng(1234)
    palette = (rng.random((max(max_pid + 1, 1), 3)) * 255).astype(np.uint8)
    planes_json = []
    for p in planes_meta:
        pid_i = int(p["plane_id"])
        color = [int(c) for c in palette[pid_i]] if 0 <= pid_i < len(palette) else [0, 0, 0]
        q = dict(p); q["color_rgb"] = color
        planes_json.append(q)

    json_path = os.path.join(out_dir, "planes.json")
    with open(json_path, "w") as f:
        json.dump(planes_json, f, indent=2)
    print(f"[OUT] planes.json")

    ply_path = os.path.join(out_dir, "planes.ply")
    _write_ply_with_face_pid_binary(
        points, faces, face_pid_reindexed, labels_f, ply_path, planes_json=planes_json
    )
    print(f"[OUT] planes.ply (binary)")

    print(f"[DONE] Saved to {out_dir}")
    print(f"  planes.ply: {len(points)} vertices, {len(faces)} faces, "
          f"{len(unique_pids)} planes")


if __name__ == "__main__":
    main()
