import os
import json
import argparse
import numpy as np
import open3d as o3d
import h5py
from tqdm import tqdm

from planamono.shared.utils.utils import *
from planamono.shared.parsers.parse_scannetpp import *
# from planamono.shared.rendering.mesh_utils import *
from planamono.shared.rendering.mesh_io import *
from planamono.shared.rendering.render import *
from planamono.paths import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    # input mesh (planes_v2.ply) and intended scene
    # mesh_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/planes_v2.ply'
    # mesh_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/planes.ply'
    mesh_path = f'{scannetpp_plane_path}/{scene_id}/planes.ply'

    print(f"[INFO] Rendering scene: {scene_id}")
    # print("[INFO] Reading planes_v2.ply …")
    print("[INFO] Reading planes.ply …")

    # load mesh & per-vertex labels (your helper)
    # sem_mesh, vertex_labels = load_mesh_with_vertex_labels(mesh_path)
    V, F, plane_id_face, _ = read_ply_faces_with_plane_ids(mesh_path)

    # === Create mesh ===
    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V)
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()
    
    # --- Path to the ScanNet++ dataset ---
    # main_dir = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'
    main_dir = scannetppv2_path
    root_dir = f"{main_dir}/data"
    iphone_dir = os.path.join(root_dir, scene_id, "iphone")

    # --- Path to pose_intrinsic_imu.json ---
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")

    # --- Read the JSON file ---
    with open(pose_file, "r") as f:
        data = json.load(f)

    # segmentation files (kept for compatibility / reference)
    seg_path = os.path.join(root_dir, scene_id, "scans", "segments.json")
    with open(seg_path) as f:
        seg_json = json.load(f)
        segmentation = seg_json.get("segIndices", None)

    seg_anno_path = os.path.join(root_dir, scene_id, "scans", "segments_anno.json")
    with open(seg_anno_path) as f:
        segments_anno = json.load(f).get('segGroups', None)

    # metadata (semantic names) — optional, kept for completeness
    semantic_classes_path = os.path.join(main_dir, 'metadata', 'semantic_classes.txt')
    id_to_name = load_semantic_id_to_name_list(semantic_classes_path)

    # --- Intrinsics (scaled) ---
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    W_orig, H_orig = 1920, 1440
    W, H = 640, 480  # ← target resolution
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    # --- Output H5 (one per scene) ---
    # out_dir = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}'
    out_dir = f'{scannetpp_rend_plane_path}/{scene_id}'
    os.makedirs(out_dir, exist_ok=True)
    # h5_path = os.path.join(out_dir, "rendered_v2.h5")
    h5_path = os.path.join(out_dir, "rendered.h5")

    print("[INFO] Raycasting planes (face/vertex labels) …")
    frame_skip = 25
    frame_ids = []
    planes_list = []

    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % frame_skip != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])

        # raycast semantic/plane labels per pixel using existing helper
        # (expecting output shape H x W with integer plane IDs, and -1 for no-plane/non-planar)
        # semantic_img = raycast_semantic(sem_mesh, vertex_labels, K_scaled, (W, H), c2w)
        semantic_img = raycast_semantic_face_labels(sem_mesh, plane_id_face, K_scaled, (W, H), c2w)

        # --- Shift plane IDs: -1→0 (non-planar), 0→1, 1→2, … ---
        # The old remap had two bugs:
        #   1. remap_semantic() from utils.py does per-frame consecutive compaction,
        #      making plane IDs inconsistent across frames (label 2 = different plane per frame).
        #   2. np.where(x < 0, 0, x) causes plane_id=0 (largest plane) to collide with
        #      non-planar pixels — both become 0, losing the biggest plane.
        # The correct fix (matching Hypersim rendering.py) is a global +1 shift:
        #   non-planar (-1) → 0, plane 0 → 1, plane 1 → 2, …
        # semantic_img = remap_semantic(semantic_img)                   # BUGGY: per-frame compaction, IDs not consistent across frames
        # semantic_img = np.where(semantic_img < 0, 0, semantic_img)   # BUGGY: plane_id=0 collides with non-planar
        semantic_img = np.where(semantic_img < 0, 0, semantic_img + 1)
        semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)

        frame_ids.append(frame_id)
        planes_list.append(semantic_img)

    if len(planes_list) == 0:
        print("[WARN] No frames were rendered (check frame_skip / pose file). Exiting.")
    else:
        print(f"[SAVE] Writing {len(planes_list)} frames to {h5_path}")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("planes", data=np.stack(planes_list), compression="gzip")
            f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))
        print(f"[DONE] Saved H5 for scene: {scene_id}")
