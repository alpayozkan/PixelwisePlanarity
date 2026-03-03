"""
Generate plane ground truth from ETH3D dense depth maps via LO-RANSAC.

Adapted from planamono/exploration/parallel_domain/ransac_outdoor.py.
Per-frame: backproject depth → 3D, compute normals, LO-RANSAC with removal.

Usage:
    python generate_planes_from_depth.py courtyard
    python generate_planes_from_depth.py courtyard --output_dir /tmp/eth3d_planes
    python generate_planes_from_depth.py courtyard --dist_thresh 0.01 --ransac_iters 500
"""

import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

from eth3d_utils import (
    DATASET_ROOT, SCENES_WITH_GT,
    load_scene_calibration, get_intrinsics_matrix, get_c2w,
    load_frame, list_frames,
)

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "ransac_iters": 300,
    "dist_thresh": 0.01,      # 1 cm
    "normal_cos": 0.985,      # ~10 degrees
    "min_inliers": 100,       # minimum pixels per plane
    "max_planes": 50,
    "lo_iters": 10,
    "merge_normal": 0.985,
    "merge_rel_dist": 0.02,
    "abs_merge_dist": 0.1,
    "z_ref": 10.0,            # depth-adaptive threshold reference
    "subsample_stride": 2,    # downsample factor for RANSAC (1 = no subsampling)
}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def depth_to_xyz(depth, fx, fy, cx, cy):
    """Depth map → (H, W, 3) point cloud."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    return np.stack([X, Y, depth], axis=-1).astype(np.float32)


def compute_normals(xyz, valid):
    """Surface normals via finite differences. Camera-facing (nz < 0)."""
    dx = np.zeros_like(xyz)
    dy = np.zeros_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dy[1:-1, :] = xyz[2:, :] - xyz[:-2, :]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    normals[normals[..., 2] > 0] *= -1
    normals[~valid] = 0
    return normals


def fit_plane_svd(points):
    """SVD plane fit. Returns (n, d) where n·x = d."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n) + 1e-12
    d = float(n @ centroid)
    if d < 0:
        n, d = -n, -d
    return n, d


# ── LO-RANSAC ─────────────────────────────────────────────────────────────────

def _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths, z_ref=10.0):
    """Depth-adaptive distance + normal gate."""
    adaptive = dist_thresh * np.maximum(1.0, depths / z_ref)
    nrm_len = np.linalg.norm(nrm, axis=-1)
    degenerate = nrm_len < 0.5
    return (dists < adaptive) & ((dots > normal_cos) | degenerate)


def local_optimize(n, d, points, nrm, dist_thresh, normal_cos, z_ref, lo_iters=10):
    depths = points[:, 2]
    for _ in range(lo_iters):
        dists = np.abs(points @ n - d)
        dots = np.abs(nrm @ n)
        inlier = _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths, z_ref)
        if inlier.sum() < 3:
            return n, d, inlier
        n, d = fit_plane_svd(points[inlier])
    dists = np.abs(points @ n - d)
    dots = np.abs(nrm @ n)
    inlier = _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths, z_ref)
    return n, d, inlier


def ransac_with_removal(points, nrm, indices, params):
    """LO-RANSAC with sequential plane removal.

    Returns list of plane dicts: n, d, pixel_indices, num_pixels, p95.
    """
    alive = np.ones(len(points), dtype=bool)
    planes = []

    for _ in range(params["max_planes"]):
        pool = np.where(alive)[0]
        if len(pool) < params["min_inliers"]:
            break

        pts = points[pool]
        nrm_pool = nrm[pool]

        best_count = 0
        best_n = best_d = best_inlier = None

        for _ in range(params["ransac_iters"]):
            idx3 = np.random.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx3]
            n = np.cross(p1 - p0, p2 - p0)
            nlen = np.linalg.norm(n)
            if nlen < 1e-8:
                continue
            n /= nlen
            d_val = float(n @ p0)

            dists = np.abs(pts @ n - d_val)
            dots = np.abs(nrm_pool @ n)
            inlier = _inlier_mask(dists, dots, nrm_pool,
                                  params["dist_thresh"], params["normal_cos"],
                                  pts[:, 2], params["z_ref"])
            count = inlier.sum()

            if count > best_count:
                n_lo, d_lo, inlier_lo = local_optimize(
                    n, d_val, pts, nrm_pool,
                    params["dist_thresh"], params["normal_cos"],
                    params["z_ref"], params["lo_iters"])
                count_lo = inlier_lo.sum()
                if count_lo > best_count:
                    best_count = count_lo
                    best_n, best_d, best_inlier = n_lo, d_lo, inlier_lo

        if best_count < params["min_inliers"]:
            break

        inlier_global = pool[best_inlier]
        residuals = np.abs(points[inlier_global] @ best_n - best_d)
        p95 = float(np.percentile(residuals, 95))

        planes.append({
            "n": best_n, "d": best_d,
            "pixel_indices": [indices[i] for i in inlier_global],
            "num_pixels": int(len(inlier_global)),
            "p95": p95,
        })
        alive[inlier_global] = False

    return planes


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_planes(planes, xyz, params):
    """Union-find merge of planes with similar parameters."""
    from collections import defaultdict

    if len(planes) < 2:
        return planes

    parent = list(range(len(planes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pi, pj = planes[i], planes[j]
            if abs(float(pi["n"] @ pj["n"])) < params["merge_normal"]:
                continue
            mean_d = (pi["d"] + pj["d"]) / 2.0
            thresh = max(params["abs_merge_dist"], params["merge_rel_dist"] * mean_d)

            pix_i = pi["pixel_indices"]
            pix_j = pj["pixel_indices"]
            sample_i = pix_i[::max(1, len(pix_i) // 500)]
            sample_j = pix_j[::max(1, len(pix_j) // 500)]
            pts_i = np.array([xyz[y, x] for y, x in sample_i])
            pts_j = np.array([xyz[y, x] for y, x in sample_j])

            d_i2j = np.median(np.abs(pts_i @ pj["n"] - pj["d"]))
            d_j2i = np.median(np.abs(pts_j @ pi["n"] - pi["d"]))
            if max(d_i2j, d_j2i) < thresh:
                union(i, j)

    groups = defaultdict(list)
    for i in range(len(planes)):
        groups[find(i)].append(i)

    merged = []
    for members in groups.values():
        all_pixels = []
        for m in members:
            all_pixels.extend(planes[m]["pixel_indices"])
        pts = np.array([xyz[y, x] for y, x in all_pixels])
        n, d = fit_plane_svd(pts)
        residuals = np.abs(pts @ n - d)
        merged.append({
            "n": n, "d": d,
            "pixel_indices": all_pixels,
            "num_pixels": len(all_pixels),
            "p95": float(np.percentile(residuals, 95)),
        })

    return merged


# ── Per-frame extraction ───────────────────────────────────────────────────────

def extract_planes_from_depth(depth, K, params):
    """Extract planes from a single depth frame.

    Uses spatial subsampling for RANSAC speed, then projects plane equations
    back to full resolution for dense labeling.

    Returns:
        plane_map: (H, W) int32, 0 = non-planar, 1+ = plane ID
        planes_info: list of plane dicts
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    stride = params.get("subsample_stride", 1)

    valid = np.isfinite(depth) & (depth > 0)
    xyz = depth_to_xyz(depth, fx, fy, cx, cy)
    normals = compute_normals(xyz, valid)

    if stride > 1:
        # Subsample for RANSAC
        valid_sub = valid[::stride, ::stride]
        xyz_sub = xyz[::stride, ::stride]
        normals_sub = normals[::stride, ::stride]

        yx_sub = np.argwhere(valid_sub)
        if len(yx_sub) < params["min_inliers"]:
            return np.zeros((H, W), dtype=np.int32), []

        pts_sub = xyz_sub[yx_sub[:, 0], yx_sub[:, 1]]
        nrm_sub = normals_sub[yx_sub[:, 0], yx_sub[:, 1]]
        indices_sub = [(int(y), int(x)) for y, x in yx_sub]

        # RANSAC on subsampled points to find plane equations
        planes_sub = ransac_with_removal(pts_sub, nrm_sub, indices_sub, params)

        # Project found planes back to full resolution
        plane_map = np.zeros((H, W), dtype=np.int32)
        planes_info = []
        for pid, p in enumerate(planes_sub):
            n, d = p["n"], p["d"]
            # Apply plane equation to all full-res valid points
            dists = np.abs(xyz[..., 0] * n[0] + xyz[..., 1] * n[1] + xyz[..., 2] * n[2] - d)
            adaptive = params["dist_thresh"] * np.maximum(1.0, depth / params["z_ref"])
            dot_normals = np.abs(normals @ n)
            nrm_len = np.linalg.norm(normals, axis=-1)
            degenerate = nrm_len < 0.5
            inlier_full = valid & (dists < adaptive) & ((dot_normals > params["normal_cos"]) | degenerate)
            # Don't overwrite already assigned pixels
            inlier_full = inlier_full & (plane_map == 0)
            if inlier_full.sum() < params["min_inliers"]:
                continue
            plane_map[inlier_full] = pid + 1
            residuals = dists[inlier_full]
            planes_info.append({
                "n": n, "d": d,
                "num_pixels": int(inlier_full.sum()),
                "p95": float(np.percentile(residuals, 95)),
            })

        return plane_map, planes_info

    # No subsampling path (original)
    yx = np.argwhere(valid)
    if len(yx) < params["min_inliers"]:
        return np.zeros((H, W), dtype=np.int32), []

    pts = xyz[yx[:, 0], yx[:, 1]]
    nrm = normals[yx[:, 0], yx[:, 1]]
    indices = [(int(y), int(x)) for y, x in yx]

    planes = ransac_with_removal(pts, nrm, indices, params)
    planes = merge_planes(planes, xyz, params)

    plane_map = np.zeros((H, W), dtype=np.int32)
    for pid, p in enumerate(planes):
        for y, x in p["pixel_indices"]:
            plane_map[y, x] = pid + 1

    return plane_map, planes


# ── Scene-level pipeline ───────────────────────────────────────────────────────

def process_scene(scene_name, output_dir, params=None, dataset_root=None, verbose=True):
    """Process all frames of a scene and save results as HDF5.

    Saves: {output_dir}/{scene_name}/planes_depth.h5
      - 'plane_labels': (N_frames, H, W) int32
      - 'frame_stems': (N_frames,) string
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if dataset_root is None:
        dataset_root = DATASET_ROOT

    cameras, images = load_scene_calibration(scene_name, root=dataset_root)
    cam = list(cameras.values())[0]
    K = get_intrinsics_matrix(cam)

    # Find frames with GT depth
    frame_stems = list_frames(scene_name, root=dataset_root)
    gt_dir = Path(dataset_root) / scene_name / "ground_truth_depth_dense" / "dslr_images_undistorted"
    if not gt_dir.exists():
        gt_dir = Path(dataset_root) / scene_name / "ground_truth_depth" / "dslr_images_undistorted"

    valid_stems = [s for s in frame_stems if (gt_dir / f"{s}.h5").exists()]

    if verbose:
        print(f"Scene: {scene_name}, {len(valid_stems)} frames with GT depth")

    all_labels = []
    all_stems = []

    for stem in tqdm(valid_stems, disable=not verbose, desc=scene_name):
        _, depth = load_frame(scene_name, stem, root=dataset_root)
        if depth is None:
            continue
        plane_map, planes = extract_planes_from_depth(depth, K, params)
        all_labels.append(plane_map)
        all_stems.append(stem)

    # Save
    out_dir = Path(output_dir) / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "planes_depth.h5"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("plane_labels", data=np.stack(all_labels), compression="gzip")
        dt = h5py.string_dtype()
        f.create_dataset("frame_stems", data=np.array(all_stems, dtype=object), dtype=dt)

    if verbose:
        print(f"Saved: {h5_path}")

    return h5_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate plane GT from ETH3D depth")
    parser.add_argument("scene", help="Scene name (e.g., courtyard)")
    parser.add_argument("--output_dir", default="/cluster/scratch/aoezkan/planeseg/eth3d_planes")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--dist_thresh", type=float, default=None)
    parser.add_argument("--ransac_iters", type=int, default=None)
    parser.add_argument("--min_inliers", type=int, default=None)
    parser.add_argument("--normal_cos", type=float, default=None)
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    for key in ["dist_thresh", "ransac_iters", "min_inliers", "normal_cos"]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    process_scene(args.scene, args.output_dir, params=params, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
