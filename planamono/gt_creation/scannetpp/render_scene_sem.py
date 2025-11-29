import os
import json
import argparse
import numpy as np
import open3d as o3d
import h5py
from tqdm import tqdm
import imageio

from plyfile import PlyData

from planamono.shared.utils.utils import *
from planamono.shared.parsers.parse_scannetpp import *
from planamono.shared.rendering.mesh_utils import *
from planamono.shared.rendering.render import *
from planamono.paths import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    print(f"[INFO] Rendering semantic H5 for scene: {scene_id}")

    # === Paths ===
    # main_dir = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'
    main_dir = scannetppv2_path
    root_dir = f"{main_dir}/data"
    iphone_dir = os.path.join(root_dir, scene_id, "iphone")

    sem_mesh_path = os.path.join(root_dir, scene_id, "scans", "mesh_aligned_0.05_semantic.ply")
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")

    # --- Output H5 ---
    # sem_h5_dir = f'/cluster/scratch/aoezkan/dataset/scannetpp/semantic_gt/{scene_id}'
    sem_h5_dir = f'{scannetpp_rend_plane_path}/{scene_id}'
    os.makedirs(sem_h5_dir, exist_ok=True)
    sem_h5_path = os.path.join(sem_h5_dir, "rendered_sem.h5")

    # === Load pose & intrinsics ===
    with open(pose_file, "r") as f:
        data = json.load(f)

    # === Load semantic mesh ===
    print("[INFO] Reading semantic mesh …")
    sem_mesh, vertex_labels = load_mesh_with_vertex_labels(sem_mesh_path)
    sem_mesh.compute_vertex_normals()
    print(f"[INFO] Mesh stats: {len(sem_mesh.vertices)} vertices, {len(sem_mesh.triangles)} faces")

    # === Load metadata ===
    semantic_classes_path = os.path.join(main_dir, 'metadata', 'semantic_classes.txt')
    id_to_name = load_semantic_id_to_name_list(semantic_classes_path)

    # === Intrinsics (scaled) ===
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    W_orig, H_orig = 1920, 1440
    W, H = 640, 480  # target resolution

    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    # === Render settings ===
    # frame_skip = 10
    frame_skip = 25
    frame_ids = []
    sem_list = []

    print("[INFO] Raycasting semantics …")
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % frame_skip != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])
        semantic_img = raycast_semantic(sem_mesh, vertex_labels, K_scaled, (W, H), c2w)

        # ensure uint16 (class IDs)
        semantic_img = semantic_img.astype(np.uint16)

        frame_ids.append(frame_id)
        sem_list.append(semantic_img)

    # === Save all frames to .h5 ===
    print(f"[SAVE] SEMANTIC → {sem_h5_path}")
    with h5py.File(sem_h5_path, "w") as f_sem:
        f_sem.create_dataset("sem", data=np.stack(sem_list), compression="gzip")
        f_sem.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))

    print(f"[DONE] Finished scene: {scene_id}")
