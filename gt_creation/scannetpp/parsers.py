import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import open3d as o3d
from plyfile import PlyData
import numpy as np
import json
import os

import os, json, numpy as np

# --- thresholds (same spirit as original) ---
numPlanesPerSegment = 2
planeAreaThreshold = 100        # points min for a valid plane (raise a bit vs 10)
numIterations = 200             # RANSAC iters per plane
planeDiffThreshold = 0.02       # inlier dist (meters). 0.02–0.05 works
fittingErrorThreshold = planeDiffThreshold
orthogonalThreshold = np.cos(np.deg2rad(60))
parallelThreshold = np.cos(np.deg2rad(30))


def readMesh_scannetpp(scene_id, root_dir, metadata_dir):
    """
    Adapted ScanNet++ version of readMesh().
    Reads mesh, segmentation, and annotation to prepare plane fitting.
    """

    scan_dir = os.path.join(root_dir, scene_id, "scans")
    
    # --- Mesh and semantics ---
    mesh_path = os.path.join(scan_dir, "mesh_aligned_0.05.ply")
    sem_path = os.path.join(scan_dir, "mesh_aligned_0.05_semantic.ply")
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # --- Read vertex labels from PLY ---
    ply = PlyData.read(sem_path)
    vertex_data = ply["vertex"].data
    points = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1)
    vertex_labels = vertex_data["label"].astype(np.int32)

    # --- Faces (triangles) ---
    faces = np.stack(ply["face"].data["vertex_indices"], axis=0)
    
    # --- Per-vertex segment indices ---
    with open(os.path.join(scan_dir, "segments.json")) as f:
        seg_json = json.load(f)
        segmentation = np.array(seg_json["segIndices"], dtype=np.int32)

    # --- Segment annotation groups (object-level) ---
    with open(os.path.join(scan_dir, "segments_anno.json")) as f:
        segments_anno = json.load(f)
        groupSegments = [g["segments"] for g in segments_anno["segGroups"]]
        groupLabels = [g["label"] for g in segments_anno["segGroups"]]

    # --- Class mapping from metadata ---
    semantic_classes_path = os.path.join(metadata_dir, "semantic_classes.txt")
    with open(semantic_classes_path, "r") as f:
        id_to_name = [line.strip() for line in f if line.strip() != ""]

    classLabelMap = {name: [i, i] for i, name in enumerate(id_to_name)}
    classLabelMap["unannotated"] = [-1, len(id_to_name)]

    print(f"[INFO] Loaded scene {scene_id}: {len(points)} vertices, {len(faces)} faces, {len(groupSegments)} groups.")

    return points, faces, segmentation, groupSegments, groupLabels, vertex_labels, classLabelMap


def fitPlane(XYZ):
    A = XYZ
    b = np.ones((XYZ.shape[0], 1))
    plane, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return plane.squeeze()


def ransac_planes_for_segment(XYZ, global_indices, K=numPlanesPerSegment):
    """
    XYZ: (N,3) points of this segment
    global_indices: indices into the full vertex array
    Returns: planes[], plane_point_indices[], remainder_indices
    """
    planes = []
    plane_point_indices = []
    remaining_mask = np.ones(XYZ.shape[0], dtype=bool)

    for _ in range(K):
        pts = XYZ[remaining_mask]
        if pts.shape[0] < planeAreaThreshold:
            break

        best_num = 0
        best_inliers_mask_local = None
        best_plane = None

        # basic 3pt sampling
        # for it in range(min(len(pts), numIterations)):
        for it in range(numIterations):
            
            choice = np.random.choice(np.arange(len(pts)), size=3, replace=False)
            try:
                cand = fitPlane(pts[choice])
            except np.linalg.LinAlgError:
                continue

            denom = max(np.linalg.norm(cand), 1e-6)
            d = np.abs(pts @ cand - 1.0) / denom
            inliers = d < planeDiffThreshold
            num_inliers = int(inliers.sum())
            if num_inliers > best_num:
                best_num = num_inliers
                best_inliers_mask_local = inliers
                best_plane = cand

        if best_num < planeAreaThreshold or best_plane is None:
            break

        # refine with inliers
        inlier_xyz_all = XYZ[remaining_mask][best_inliers_mask_local]
        refined = fitPlane(inlier_xyz_all)
        planes.append(refined)

        # map inliers back to global indices
        local2global = np.where(remaining_mask)[0][best_inliers_mask_local]
        plane_point_indices.append(global_indices[local2global])

        # remove inliers from remaining
        remaining_mask[np.where(remaining_mask)[0][best_inliers_mask_local]] = False

    remainder_indices = global_indices[remaining_mask]
    return planes, plane_point_indices, remainder_indices


class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        base = np.array([
            [255, 0, 0],[0, 255, 0],[0, 0, 255],[80, 128, 255],[255, 230, 180],
            [255, 0, 255],[0, 255, 255],[100, 0, 0],[0, 100, 0],[255, 255, 0],
            [50, 150, 0],[200, 255, 255],[255, 200, 255],[128, 128, 80],
            [0, 50, 128],[0, 100, 100],[0, 255, 128],[0, 128, 255],
            [255, 0, 128],[128, 0, 255],[255, 128, 0],[128, 255, 0],
        ])
        if numColors > base.shape[0]:
            extra = np.random.randint(0, 255, size=(numColors - base.shape[0], 3))
            base = np.concatenate([base, extra], axis=0)
        self.colorMap = base

    def getColorMap(self): return self.colorMap



def writePointCloudFace(filename, points, faces):
    with open(filename, 'w') as f:
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {len(faces)}
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

def writePointCloudFace_with_id(filename, points, faces):
    with open(filename, 'w') as f:
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int plane_id
element face {len(faces)}
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])} {int(p[6])}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def build_segment_neighbors(faces, segmentation):
    edges = set()
    for a, b, c in faces:
        s1, s2, s3 = segmentation[a], segmentation[b], segmentation[c]
        if s1 != s2 and s1 != -1 and s2 != -1: edges.add((min(s1, s2), max(s1, s2)))
        if s1 != s3 and s1 != -1 and s3 != -1: edges.add((min(s1, s3), max(s1, s3)))
        if s2 != s3 and s2 != -1 and s3 != -1: edges.add((min(s2, s3), max(s2, s3)))
    neighbors = {}
    for u, v in edges:
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    return neighbors



labelNumPlanes = {
    'wall':[1,3], 'floor':[1,1], 'door':[1,2], 'picture':[1,1], 'entrance':[1,1], 'floor mat':[1,1],
    'cabinet':[0,5], 'bed':[0,5], 'chair':[0,5], 'sofa':[0,10], 'table':[0,5],
    'window':[0,2], 'bookshelf':[0,5], 'counter':[0,10], 'desk':[0,10],
    'shelf':[0,5], 'shelves':[0,5], 'ceiling':[0,5], 'whiteboard':[1,5],
    'night stand':[1,5], 'toilet':[0,5], 'sink':[0,5], 'bathtub':[0,5],
    'refridgerator':[0,5], 'otherprop':[0,5], 'otherstructure':[0,5], 'otherfurniture':[0,5],
    'unannotated':[0,5], '': [0,0]
}
nonPlanar = {'bicycle', 'bottle', 'water bottle', 'pillow', 'curtain', 'person', 'mirror', 'lamp', 'bag', 'book', 'books', 'paper', 'towel', 'shower curtain', 'box', 'clothes'}


def process_groups(points, faces, segmentation, groupSegments, groupLabels, classLabelMap, out_dir, debug=True):
    """
    ScanNet++-optimized: groupSegments[i] is a list of VERTEX INDICES (identity segmentation).
    We fit planes per OBJECT (one shot), with optional RANSAC fallback, and save outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # If your segmentation is identity (0..N-1), building segment neighbors is unnecessary.
    # Keep it only if you rely on it elsewhere; it's not used below.
    # segmentNeighbors = build_segment_neighbors(faces, segmentation)

    allXYZ = points  # (N,3)

    planeGroups = []            # per-group: list of (plane_vec, vertex_indices, neighbors)
    grouped_plane_segments = [] # not used for merging; kept for compatibility
    used_group_labels = []      # label per group

    print('start plane fitting')

    from tqdm import tqdm
    for gi, (segments_in_group, raw_label) in tqdm(
        enumerate(zip(groupSegments, groupLabels)),
        total=len(groupSegments),
        desc=f"Fitting planes ({len(groupSegments)} groups)",
        ncols=100
    ):
        # --- canonicalize label & constraints ---
        label = raw_label if raw_label in labelNumPlanes else ('unannotated' if raw_label == '' else raw_label)
        minP, maxP = labelNumPlanes.get(label, [0, 5])
        if label in nonPlanar:
            minP, maxP = 0, 0

        # --- collect ALL vertices for this object (segments_in_group are vertex indices) ---
        seg_idx = np.asarray(segments_in_group, dtype=np.int32)
        if seg_idx.size < planeAreaThreshold:
            # too small to fit anything
            group_planes = [np.zeros(3)]
            group_plane_indices = [seg_idx]
            neighbors_per_plane = [[]]
            planeGroups.append(list(zip(group_planes, group_plane_indices, neighbors_per_plane)))
            grouped_plane_segments.append([set(seg_idx)])  # not used later
            used_group_labels.append(label)
            continue

        XYZ = allXYZ[seg_idx]

        # --- downsample XYZ AND seg_idx together (critical to keep alignment) ---
        # MAX_OBJ_POINTS = 20000  # tune: 5k–20k typical
        # if XYZ.shape[0] > MAX_OBJ_POINTS:
        #     sample_idx = np.random.choice(XYZ.shape[0], MAX_OBJ_POINTS, replace=False)
        #     XYZ = XYZ[sample_idx]
        #     seg_idx = seg_idx[sample_idx]

        # --- fast least-squares fit; fallback to RANSAC if needed ---
        try:
            p = fitPlane(XYZ)
            err = np.mean(np.abs(XYZ @ p - 1.0) / max(np.linalg.norm(p), 1e-6))
        except np.linalg.LinAlgError:
            p, err = np.zeros(3), 1e9

        group_planes = []
        group_plane_indices = []

        if err < fittingErrorThreshold or maxP == 1:
            # Accept single LS plane, or forced single-plane class
            group_planes = [p if err < 1e8 else np.zeros(3)]
            group_plane_indices = [seg_idx]
        else:
            # RANSAC: up to K planes from the object's vertices
            planes, idx_lists, remainder = ransac_planes_for_segment(
                XYZ, seg_idx, K=numPlanesPerSegment
            )
            # Heuristic: if too few points explained, mark as non-planar
            if sum(len(x) for x in idx_lists) < 0.5 * len(seg_idx):
                group_planes = [np.zeros(3)]
                group_plane_indices = [seg_idx]
            else:
                group_planes.extend(planes)
                group_plane_indices.extend(idx_lists)
                if remainder.size > 0:
                    # leftover region marked non-planar (optional)
                    group_planes.append(np.zeros(3))
                    group_plane_indices.append(remainder)

        # --- enforce min/max plane constraints for certain classes ---
        num_real = sum(np.linalg.norm(pp) > 1e-4 for pp in group_planes)

        if minP == 1 and num_real == 0:
            # force a plane from the largest set of points
            areas = [len(ids) for ids in group_plane_indices]
            if len(areas):
                k = int(np.argmax(areas))
                forced = fitPlane(allXYZ[group_plane_indices[k]])
                group_planes[k] = forced
                num_real = 1

        if minP == 1 and maxP == 1 and num_real > 1:
            # collapse to one plane if class expects a single plane (e.g., floor)
            all_ids = np.concatenate(group_plane_indices, axis=0)
            one = fitPlane(allXYZ[all_ids])
            d = np.abs(allXYZ[all_ids] @ one - 1.0) / max(np.linalg.norm(one), 1e-6)
            if d.mean() < fittingErrorThreshold * (3.0 if label == 'floor' else 1.0):
                group_planes = [one]
                group_plane_indices = [all_ids]
                num_real = 1

        # --- OPTIONAL merging skipped ---
        # Since we're already per-object, merging within this object usually isn't needed.
        # If you still want merging, you'll need a segmentNeighbors defined on "segments".
        # Here we just compute neighbors by orthogonality among the group's planes.

        neighbors_per_plane = []
        for i in range(len(group_planes)):
            neigh = []
            for j in range(len(group_planes)):
                if i == j: continue
                pi, pj = group_planes[i], group_planes[j]
                if np.linalg.norm(pi) * np.linalg.norm(pj) < 1e-6: continue
                dp = abs(np.dot(pi, pj) / (np.linalg.norm(pi) * np.linalg.norm(pj)))
                if dp < orthogonalThreshold:  # orthogonal pair
                    neigh.append(j)
            neighbors_per_plane.append(neigh)

        planeGroups.append(list(zip(group_planes, group_plane_indices, neighbors_per_plane)))
        grouped_plane_segments.append([set(seg_idx)] * len(group_planes))  # placeholder
        used_group_labels.append(label)

    # ------------------ flatten & save outputs ------------------

    planes = []
    planePointIndices = []
    planeInfo = []  # per-plane: [(group_idx, nyuId, labelIdx), (structure_id, degree?)] like original

    structure_counter = 0
    for gi, group in enumerate(planeGroups):
        if len(group) == 0:
            continue
        gp_planes, gp_indices, gp_neighbors = zip(*group)

        # build "structure" groups: planes with >=2 orthogonal neighbors
        if len(gp_neighbors) > 0:
            A = np.eye(len(gp_neighbors))
            for r, neigh in enumerate(gp_neighbors):
                for c in neigh:
                    A[r, c] = 1
            participate = np.where(A.sum(-1) >= 2)[0].tolist()
        else:
            participate = []

        label = used_group_labels[gi]
        nyuId, labelIdx = classLabelMap.get(label, [-1, -1])

        used = set()
        structures = []
        for i in participate:
            if i in used:
                continue
            structure = (A[i] == 1).astype(int)
            ids = np.where(structure > 0)[0][:3]  # cap at 3 like original
            for k in ids:
                used.add(k)
            structures.append(ids)

        for pi, (pl, ids) in enumerate(zip(gp_planes, gp_indices)):
            info = [[(gi, nyuId, labelIdx)]]
            deg = None
            for ids_s in structures:
                if pi in ids_s:
                    info[0].append((structure_counter, len(ids_s)))
                    deg = len(ids_s)
                    break
            if deg is not None:
                structure_counter += 1

            planes.append(pl)
            planePointIndices.append(np.asarray(ids, dtype=np.int32))
            planeInfo.append(info[0])

    planes = np.array(planes) if len(planes) else np.zeros((0, 3))

    # per-vertex plane ID for visualization
    planeSeg = np.full(segmentation.shape, -1, dtype=np.int32)
    for k, idxs in enumerate(planePointIndices):
        planeSeg[idxs] = k

    # colorize & write PLY
    colors = ColorPalette(int(planeSeg.max()) + 2).getColorMap()
    colors[-1] = 0
    out_faces = faces.copy()
    keep = []
    for (a, b, c) in out_faces:
        s1, s2, s3 = planeSeg[a], planeSeg[b], planeSeg[c]
        keep.append(s1 == s2 == s3 and s1 != -1)
    out_faces = out_faces[np.array(keep, dtype=bool)]

    # vert_colors = colors[planeSeg]
    # pts_rgb = np.concatenate([points, vert_colors], axis=1)
    # writePointCloudFace(os.path.join(out_dir, 'planes.ply'), pts_rgb, out_faces)

    vert_colors = colors[planeSeg]
    plane_ids = np.maximum(planeSeg, 0).reshape(-1, 1)  # replace -1 with 0
    pts_rgb_id = np.concatenate([points, vert_colors, plane_ids], axis=1)
    
    writePointCloudFace_with_id(os.path.join(out_dir, 'planes.ply'), pts_rgb_id, out_faces)
    # writePointCloudFace_with_id(os.path.join(out_dir, 'planes_with_id.ply'), pts_rgb_id, out_faces)

    # match original scaling: planes *= (1/||p||)^2
    if planes.shape[0] > 0:
        planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-6)
        planes_scaled = planes * (planesD ** 2)
    else:
        planes_scaled = planes

    np.save(os.path.join(out_dir, 'planes.npy'), planes_scaled)
    np.save(os.path.join(out_dir, 'plane_info.npy'), np.array(planeInfo, dtype=object))

    if debug:
        seg_colors = ColorPalette(segmentation.max() + 2).getColorMap()
        seg_colors[-1] = 0
        writePointCloudFace(
            os.path.join(out_dir, 'segments.ply'),
            np.concatenate([points, seg_colors[segmentation]], axis=1),
            faces
        )

    print(f"[DONE] Saved plane annotations to: {out_dir}")
    


