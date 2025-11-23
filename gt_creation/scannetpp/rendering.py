from visualize_planes_v1 import *
from utils import *
from parse_scannetpp import *

from mesh_utils import *
from render import *

from plyfile import PlyData, PlyElement
import open3d as o3d
import imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    # mesh_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/planes.ply'
    mesh_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/planes_v2.ply'
    # render_save_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/rendered/'
    render_save_path = f'/cluster/scratch/aoezkan/dataset/scannetpp/plane_ours_gt/{scene_id}/rendered_v2/'
    
    print(f"[INFO] Rendering scene: {scene_id}")

    print("[INFO] Reading planes_v2.ply …")

    sem_mesh, vertex_labels = load_mesh_with_vertex_labels(mesh_path)

    # --- Path to the ScanNet++ dataset ---
    # root_dir = "/cluster/project/cvg/Shared_datasets/scannet++/data"
    main_dir = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'
    root_dir = f"{main_dir}/data"
    
    # Example: choose a scene (replace with the actual one)
    # scene_id = "0a5c013435"
    iphone_dir = os.path.join(root_dir, scene_id, "iphone")
    
    # --- Path to pose_intrinsic_imu.json ---
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")
    
    # --- Read the JSON file ---
    with open(pose_file, "r") as f:
        data = json.load(f)
    
    # mesh_path = os.path.join(root_dir, scene_id, "scans", "mesh_aligned_0.05.ply")
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # mesh.compute_vertex_normals()
    
    # 4. Segmentation (per-vertex segment index)
    seg_path = os.path.join(root_dir, scene_id, "scans", "segments.json")
    with open(seg_path) as f:
        seg_json = json.load(f)
        segmentation = seg_json["segIndices"]  # or possibly a .npy
    
    seg_path = os.path.join(root_dir, scene_id, "scans", "segments_anno.json")
    with open(seg_path) as f:
        segments_anno = json.load(f)
        segments_anno = segments_anno['segGroups']



    # metadata
    semantic_classes_path = os.path.join(main_dir, 'metadata', 'semantic_classes.txt')
    id_to_name = load_semantic_id_to_name_list(semantic_classes_path)

    
    # --- Intrinsics ---
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    # Native iPhone resolution for ScanNet++ (1920×1440)
    # W, H = 1920, 1440
    W_orig, H_orig = 1920, 1440
    W, H = 640, 480  # ← target resolution
    # Scale intrinsics to match target resolution
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    
    os.makedirs(render_save_path, exist_ok=True)

    print("[INFO] Raycasting …")
    frame_skip = 25  # save every 25th frame (0, 25, 50, ...)
    # frame_skip = 10
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        # if i >= 100:  # render only a subset for now
        #     break
        if i % frame_skip != 0:
            continue
            
        c2w = np.array(frame_data["aligned_pose"])
        # semantic_img_face = raycast_semantic_face_labels(sem_mesh, plane_id_face, K, (W, H), c2w)
        # semantic_img_vertex = raycast_semantic(sem_mesh, vertex_labels, K, (W, H), c2w)
        # semantic_img = raycast_semantic(sem_mesh, vertex_labels, K, (W, H), c2w)
        semantic_img = raycast_semantic(sem_mesh, vertex_labels, K_scaled, (W, H), c2w)
        semantic_img = remap_semantic(semantic_img)

        seg_path = os.path.join(render_save_path, f"{frame_id}.png")
        # imageio.imwrite(seg_path, semantic_img_vertex.astype(np.uint8))
        save_label_image(seg_path, semantic_img)

        # consideration if we have labels>255, saving will fail
        # also need to save -1 as 0, otherwise we get overflow nonplanar region becomes=255
    
    print(f"[DONE] Finished scene: {scene_id}")

