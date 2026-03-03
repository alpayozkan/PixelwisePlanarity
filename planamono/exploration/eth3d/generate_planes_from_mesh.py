"""
Generate plane ground truth from ETH3D surface meshes.

Adapted from planamono/gt_creation/scannetpp/plane_extraction.py.
ETH3D meshes have no semantic labels, so region growing is label-free
(all faces treated as one class).

Pipeline:
  1. Load mesh (Poisson surface reconstruction)
  2. Build face adjacency
  3. Region growing (BFS with normal + distance gates)
  4. IRLS plane fitting
  5. Quality filtering (residual P95, inlier fraction, thickness, width, fill)
  6. Merge compatible planes
  7. Raycast face labels to 2D images using COLMAP calibration

Usage:
    python generate_planes_from_mesh.py courtyard
    python generate_planes_from_mesh.py courtyard --output_dir /tmp/eth3d_planes
"""

import argparse
import json
import time
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
from collections import deque
from tqdm import tqdm

from eth3d_utils import (
    DATASET_ROOT, MESH_ROOT,
    load_scene_calibration, get_intrinsics_matrix, get_c2w,
    list_frames, raycast_face_labels,
)

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # Region growing
    "rg_theta_deg": 8.0,       # normal angle threshold (degrees)
    "rg_dist_m": 0.015,        # distance from plane (meters)
    "rg_refit_every": 20,      # refit plane every N faces
    "min_faces_patch": 100,    # minimum faces to keep a region
    "min_area_patch": 0.05,    # minimum area (m²)
    # Quality gates
    "p95_max": 0.03,           # max 95th percentile residual (m)
    "inlier_frac_min": 0.80,   # min inlier fraction
    "inlier_dist": 0.012,      # inlier distance threshold (m)
    "normal_p95_deg_max": 10.0, # max normal angle P95 (degrees)
    "thickness_max_m": 0.025,  # max slab thickness (m)
    "min_width_m": 0.05,       # min extent in any direction (m)
    "fill_frac_min": 0.10,     # min fill in OBB
    # Merge
    "merge_theta_deg": 10.0,   # merge normal threshold (degrees)
    "merge_dist_m": 0.02,      # merge distance threshold (m)
    # IRLS
    "irls_max_iters": 8,
    "irls_huber_k": 1.345,
}


# ── Plane fitting ──────────────────────────────────────────────────────────────

def fit_plane_irls(P, max_iters=8, huber_k=1.345):
    """IRLS robust plane fitting. Returns (n, d) where n·x + d = 0, ||n||=1."""
    centroid = P.mean(axis=0)
    P_c = P - centroid
    _, _, Vt = np.linalg.svd(P_c, full_matrices=False)
    n = Vt[-1].copy()
    n /= np.linalg.norm(n)
    d = -float(n @ centroid)
    if d < 0:
        n, d = -n, -d

    for _ in range(max_iters):
        r = P @ n + d
        mad = np.median(np.abs(r)) * 1.4826 + 1e-10
        c = huber_k * mad
        w = np.minimum(1.0, c / (np.abs(r) + 1e-12))

        w_sum = w.sum()
        centroid_w = (P * w[:, None]).sum(axis=0) / w_sum
        P_cw = (P - centroid_w) * np.sqrt(w[:, None])
        _, _, Vt = np.linalg.svd(P_cw, full_matrices=False)
        n_new = Vt[-1].copy()
        n_new /= np.linalg.norm(n_new)
        d_new = -float(n_new @ centroid_w)
        if d_new < 0:
            n_new, d_new = -n_new, -d_new

        if np.linalg.norm(n_new - n) < 1e-6 and abs(d_new - d) < 1e-6:
            n, d = n_new, d_new
            break
        n, d = n_new, d_new

    return n, d


# ── Face adjacency ─────────────────────────────────────────────────────────────

def build_face_adjacency(F, n_verts):
    """Build face adjacency via shared edges. Returns list of lists."""
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(F):
        for k in range(3):
            e = tuple(sorted((face[k], face[(k + 1) % 3])))
            edge_to_faces[e].append(fi)

    adj = [[] for _ in range(len(F))]
    for faces in edge_to_faces.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                adj[faces[i]].append(faces[j])
                adj[faces[j]].append(faces[i])
    return adj


# ── Region growing ─────────────────────────────────────────────────────────────

def region_growing(F, V, FN, FA, Cface, adj, params):
    """BFS region growing on mesh faces with normal + distance gates.

    Returns:
        regions: list of (face_indices_array, plane_n, plane_d)
    """
    n_faces = len(F)
    assigned = np.full(n_faces, -1, dtype=np.int32)

    cos_thresh = np.cos(np.radians(params["rg_theta_deg"]))
    dist_thresh = params["rg_dist_m"]
    refit_every = params["rg_refit_every"]
    min_faces = params["min_faces_patch"]
    min_area = params["min_area_patch"]

    # Sort faces by area (largest first → better seeds)
    order = np.argsort(-FA)

    regions = []
    for seed in order:
        if assigned[seed] >= 0:
            continue

        # Fit initial plane from seed face
        verts_seed = V[F[seed]]
        n, d = fit_plane_irls(verts_seed, max_iters=1)

        region = [seed]
        assigned[seed] = len(regions)
        queue = deque([seed])
        counter = 0

        while queue:
            fi = queue.popleft()
            for fj in adj[fi]:
                if assigned[fj] >= 0:
                    continue

                # Normal gate
                if abs(float(FN[fj] @ n)) < cos_thresh:
                    continue

                # Distance gate: all 3 vertices within threshold
                dists = np.abs(V[F[fj]] @ n + d)
                if dists.max() > dist_thresh:
                    continue

                assigned[fj] = len(regions)
                region.append(fj)
                queue.append(fj)
                counter += 1

                # Periodic refit
                if counter % refit_every == 0 and len(region) > 3:
                    all_verts = V[F[region].ravel()]
                    n, d = fit_plane_irls(all_verts, max_iters=params["irls_max_iters"])

        face_ids = np.array(region, dtype=np.int32)
        area = FA[face_ids].sum()

        if len(face_ids) >= min_faces and area >= min_area:
            # Final fit
            all_verts = V[F[face_ids].ravel()]
            n, d = fit_plane_irls(all_verts, max_iters=params["irls_max_iters"])
            regions.append((face_ids, n, d))
        else:
            assigned[face_ids] = -1  # release

    return regions


# ── Quality filtering ──────────────────────────────────────────────────────────

def filter_regions(regions, F, V, FN, FA, params):
    """Apply quality gates to regions. Returns filtered list."""
    filtered = []
    cos_normal_thresh = np.cos(np.radians(params["normal_p95_deg_max"]))

    for face_ids, n, d in regions:
        verts = V[F[face_ids].ravel()]  # (3K, 3)

        # Residual P95
        residuals = np.abs(verts @ n + d)
        p95 = np.percentile(residuals, 95)
        if p95 > params["p95_max"]:
            continue

        # Inlier fraction
        inlier_frac = (residuals < params["inlier_dist"]).mean()
        if inlier_frac < params["inlier_frac_min"]:
            continue

        # Thickness
        proj = verts @ n
        thickness = proj.max() - proj.min()
        if thickness > params["thickness_max_m"]:
            continue

        # Normal consistency P95
        cos_angles = np.abs(FN[face_ids] @ n)
        if np.percentile(cos_angles, 5) < cos_normal_thresh:
            continue

        # Minimum width (via OBB in plane-tangent frame)
        centroid = verts.mean(axis=0)
        P_c = verts - centroid
        # Build tangent frame
        t1 = np.cross(n, [1, 0, 0])
        if np.linalg.norm(t1) < 0.1:
            t1 = np.cross(n, [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        proj2d = P_c @ np.stack([t1, t2], axis=-1)  # (K, 2)
        extent = proj2d.max(axis=0) - proj2d.min(axis=0)
        if extent.min() < params["min_width_m"]:
            continue

        # Fill fraction
        area = FA[face_ids].sum()
        obb_area = extent[0] * extent[1] + 1e-10
        fill = area / obb_area
        if fill < params["fill_frac_min"]:
            continue

        filtered.append((face_ids, n, d))

    return filtered


# ── Merging ────────────────────────────────────────────────────────────────────

def merge_regions(regions, F, V, FA, params):
    """Merge regions with similar plane parameters."""
    if len(regions) < 2:
        return regions

    cos_thresh = np.cos(np.radians(params["merge_theta_deg"]))
    dist_thresh = params["merge_dist_m"]

    # Union-find
    parent = list(range(len(regions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            _, ni, di = regions[i]
            _, nj, dj = regions[j]

            # Normal similarity
            if abs(float(ni @ nj)) < cos_thresh:
                continue

            # Distance between planes: |di - dj| (same-sign normals) or |di + dj|
            if (ni @ nj) > 0:
                plane_dist = abs(di - dj)
            else:
                plane_dist = abs(di + dj)
            if plane_dist > dist_thresh:
                continue

            union(i, j)

    # Group and refit
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(regions)):
        groups[find(i)].append(i)

    merged = []
    for members in groups.values():
        all_faces = np.concatenate([regions[m][0] for m in members])
        all_verts = V[F[all_faces].ravel()]
        n, d = fit_plane_irls(all_verts, max_iters=params["irls_max_iters"])
        merged.append((all_faces, n, d))

    return merged


# ── Main pipeline ──────────────────────────────────────────────────────────────

def extract_planes_from_mesh(scene_name, params=None, mesh_root=None, verbose=True):
    """Run full mesh-based plane extraction pipeline.

    Returns:
        mesh: trimesh.Trimesh (original mesh)
        face_plane_ids: (N_faces,) int32, -1 = non-planar, 0+ = plane ID
        planes_meta: list of dicts with keys: plane_id, n, d, faces, area
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if mesh_root is None:
        mesh_root = MESH_ROOT

    mesh_path = Path(mesh_root) / scene_name / scene_name / "occlusion" / "surface_mesh.ply"
    if verbose:
        print(f"Loading mesh: {mesh_path}")

    t0 = time.time()
    mesh = trimesh.load(str(mesh_path), process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)

    # Compute face properties
    FN = np.asarray(mesh.face_normals, dtype=np.float64)
    FA = np.asarray(mesh.area_faces, dtype=np.float64)
    Cface = V[F].mean(axis=1)  # face centroids

    if verbose:
        print(f"  Vertices: {len(V):,}, Faces: {len(F):,}")
        print(f"  Loaded in {time.time()-t0:.1f}s")

    # Stage 1: Build adjacency
    t1 = time.time()
    if verbose:
        print("Building face adjacency...")
    adj = build_face_adjacency(F, len(V))
    if verbose:
        print(f"  Done in {time.time()-t1:.1f}s")

    # Stage 2: Region growing
    t2 = time.time()
    if verbose:
        print("Region growing...")
    regions = region_growing(F, V, FN, FA, Cface, adj, params)
    if verbose:
        print(f"  Found {len(regions)} raw regions in {time.time()-t2:.1f}s")

    # Stage 3: Quality filtering
    t3 = time.time()
    if verbose:
        print("Quality filtering...")
    regions = filter_regions(regions, F, V, FN, FA, params)
    if verbose:
        print(f"  Kept {len(regions)} regions after filtering in {time.time()-t3:.1f}s")

    # Stage 4: Merge
    t4 = time.time()
    if verbose:
        print("Merging compatible planes...")
    regions = merge_regions(regions, F, V, FA, params)
    if verbose:
        print(f"  {len(regions)} planes after merge in {time.time()-t4:.1f}s")

    # Sort by area (largest first)
    regions.sort(key=lambda r: FA[r[0]].sum(), reverse=True)

    # Build output
    face_plane_ids = np.full(len(F), -1, dtype=np.int32)
    planes_meta = []
    for pid, (face_ids, n, d) in enumerate(regions):
        face_plane_ids[face_ids] = pid
        verts = V[F[face_ids].ravel()]
        residuals = np.abs(verts @ n + d)
        inlier_frac = float((residuals < params["inlier_dist"]).mean())
        planes_meta.append({
            "plane_id": pid,
            "n": n.tolist(),
            "d": float(d),
            "num_faces": len(face_ids),
            "area_m2": float(FA[face_ids].sum()),
            "p95": float(np.percentile(residuals, 95)),
            "inlier_frac": inlier_frac,
        })

    total_planar = (face_plane_ids >= 0).sum()
    if verbose:
        print(f"\nResult: {len(planes_meta)} planes, "
              f"{total_planar}/{len(F)} faces planar ({total_planar/len(F)*100:.1f}%)")
        print(f"Total time: {time.time()-t0:.1f}s")

    return mesh, face_plane_ids, planes_meta


def render_planes_to_frames(scene_name, mesh, face_plane_ids, output_dir,
                            dataset_root=None, mesh_root=None, verbose=True):
    """Raycast mesh plane labels to all 2D frames and save as HDF5.

    Saves: {output_dir}/{scene_name}/planes_mesh.h5
      - 'plane_labels': (N_frames, H, W) int32, 0 = non-planar, 1+ = plane ID
      - 'depth': (N_frames, H, W) float32
      - 'frame_stems': (N_frames,) string
    """
    import h5py

    if dataset_root is None:
        dataset_root = DATASET_ROOT
    output_dir = Path(output_dir)

    cameras, images = load_scene_calibration(scene_name, root=dataset_root)
    cam = list(cameras.values())[0]
    K = get_intrinsics_matrix(cam)
    W, H = int(cam.width), int(cam.height)

    # Build Open3D mesh for raycasting
    o3d_mesh = o3d.geometry.TriangleMesh()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F_arr = np.asarray(mesh.faces, dtype=np.int32)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(V)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(F_arr)

    # Sort images by name
    sorted_images = sorted(images.values(), key=lambda x: x.name)

    # Filter to frames that exist on disk
    frame_stems = list_frames(scene_name, root=dataset_root)
    stem_set = set(frame_stems)

    frames_to_render = []
    for img in sorted_images:
        # img.name is like "dslr_images_undistorted/DSC_0286.png"
        stem = Path(img.name).stem
        if stem in stem_set:
            frames_to_render.append((stem, img))

    if verbose:
        print(f"\nRendering {len(frames_to_render)} frames for {scene_name} at {W}x{H}...")

    all_labels = []
    all_depths = []
    all_stems = []

    for stem, colmap_img in tqdm(frames_to_render, disable=not verbose):
        c2w = get_c2w(colmap_img)
        # Remap face_plane_ids: -1 → -1 (non-planar), 0+ → 0+ (plane ID)
        label_img, depth_img = raycast_face_labels(o3d_mesh, face_plane_ids, K, (W, H), c2w)
        # Remap to standard convention: -1/miss → 0, plane_id 0 → 1, etc.
        label_img_std = np.where(label_img >= 0, label_img + 1, 0).astype(np.int32)
        all_labels.append(label_img_std)
        all_depths.append(depth_img)
        all_stems.append(stem)

    # Save
    out_dir = output_dir / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "planes_mesh.h5"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("plane_labels", data=np.stack(all_labels), compression="gzip")
        f.create_dataset("depth", data=np.stack(all_depths), compression="gzip")
        dt = h5py.string_dtype()
        f.create_dataset("frame_stems", data=np.array(all_stems, dtype=object), dtype=dt)

    if verbose:
        print(f"Saved: {h5_path} ({len(all_stems)} frames)")

    return h5_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate plane GT from ETH3D mesh")
    parser.add_argument("scene", help="Scene name (e.g., courtyard)")
    parser.add_argument("--output_dir", default="/cluster/scratch/aoezkan/planeseg/eth3d_planes",
                        help="Output directory")
    parser.add_argument("--mesh_root", default=None, help="Mesh root (default: MESH_ROOT)")
    parser.add_argument("--dataset_root", default=None, help="Dataset root (default: DATASET_ROOT)")
    parser.add_argument("--render", action="store_true", help="Also render to 2D frames")
    parser.add_argument("--rg_theta_deg", type=float, default=None)
    parser.add_argument("--rg_dist_m", type=float, default=None)
    parser.add_argument("--min_faces_patch", type=int, default=None)
    parser.add_argument("--p95_max", type=float, default=None)
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    for key in ["rg_theta_deg", "rg_dist_m", "min_faces_patch", "p95_max"]:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    mesh, face_plane_ids, planes_meta = extract_planes_from_mesh(
        args.scene, params=params, mesh_root=args.mesh_root
    )

    # Save planes metadata
    out_dir = Path(args.output_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "planes_mesh.json"
    with open(json_path, "w") as f:
        json.dump(planes_meta, f, indent=2)
    print(f"Saved metadata: {json_path}")

    # Save face plane IDs
    np.save(out_dir / "face_plane_ids.npy", face_plane_ids)

    if args.render:
        render_planes_to_frames(
            args.scene, mesh, face_plane_ids, args.output_dir,
            dataset_root=args.dataset_root, mesh_root=args.mesh_root
        )


if __name__ == "__main__":
    main()
