import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from plyfile import PlyData
import numpy as np

from scipy import stats


def render_rgb(mesh, K, img_res, c2w):
    W,H = img_res
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0, 0, 0, 1])

    # --- Material (lit for visibility) ---
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (0.8, 0.8, 0.8, 1.0)

    mesh.compute_vertex_normals()
    renderer.scene.add_geometry("mesh", mesh, mat)
    
    # --- World-to-camera (Open3D expects this) ---
    w2c = np.linalg.inv(c2w)
    # --- Set camera pose ---
    renderer.setup_camera(intr, w2c)
    # --- Render color ---
    color_o3d = renderer.render_to_image()
    color_np = np.asarray(color_o3d)
    # o3d.io.write_image(os.path.join(out_rgb, f"{frame_id}.png"), color_o3d)
    # --- Render depth ---
    # depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
    # depth_np = np.asarray(depth_o3d)
    # depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    return color_np

def render_rgb_depth(mesh, K, img_res, c2w):
    W,H = img_res
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0, 0, 0, 1])

    # --- Material (lit for visibility) ---
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (0.8, 0.8, 0.8, 1.0)

    mesh.compute_vertex_normals()
    renderer.scene.add_geometry("mesh", mesh, mat)
    
    # --- World-to-camera (Open3D expects this) ---
    w2c = np.linalg.inv(c2w)
    # --- Set camera pose ---
    renderer.setup_camera(intr, w2c)
    # --- Render color ---
    color_o3d = renderer.render_to_image()
    color_np = np.asarray(color_o3d)
    # o3d.io.write_image(os.path.join(out_rgb, f"{frame_id}.png"), color_o3d)
    # --- Render depth ---
    depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
    depth_np = np.asarray(depth_o3d)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    return color_np, depth_np


def raycast_semantic(sem_mesh, vertex_labels, K, img_res, c2w):
    W,H = img_res
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))
    triangles = np.asarray(sem_mesh.triangles)  # (M, 3)
    verts = np.asarray(sem_mesh.vertices)

    cx = K[0,2]
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]

    # flip_yz = np.diag([1, -1, -1, 1])
    flip_yz = np.diag([1, -1, -1, 1])
    c2w_gl = c2w @ flip_yz
    # Decompose
    R = c2w_gl[:3, :3]
    t = c2w_gl[:3, 3]
    # Ray origin (camera center in world)
    cam_origin = t

    # === Generate ray directions ===
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    v = H - 1 - v  # ✅ Flip Y to match OpenGL # Result: The rays go to the wrong places, vertically mirrored, because pixel (u,v) in OpenCV is not the same directionally as in OpenGL.
    # You also have to flip v in the image grid to reflect that the image origin has moved from top-left (OpenCV) → bottom-left (OpenGL).
    # dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, -np.ones_like(u)], axis=-1)

    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_world = dirs @ R.T
    rays_o = np.tile(cam_origin, (H, W, 1))

    # === Raycast ===
    rays = np.concatenate([rays_o, dirs_world], axis=-1).astype(np.float32)
    rays_o3d = o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_o3d)

    triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    semantic_img = np.zeros((H, W), dtype=np.uint8)
    hit_triangles = triangles[triangle_ids[hit_mask]]  # (N_hit, 3)

    # Step 1: Get hit triangle vertex indices
    hit_tri_ids = triangle_ids[hit_mask]
    hit_triangles = triangles[hit_tri_ids]  # (N_hit, 3)
    # Step 2: Get vertex positions
    v0 = verts[hit_triangles[:, 0]]
    v1 = verts[hit_triangles[:, 1]]
    v2 = verts[hit_triangles[:, 2]]
    # Step 3: Get ray hit points
    t_hit = ans['t_hit'].numpy().reshape(H, W)[hit_mask]
    dirs_flat = dirs_world[hit_mask].reshape(-1, 3)
    origins_flat = rays_o[hit_mask].reshape(-1, 3)
    hit_pts = origins_flat + t_hit[:, None] * dirs_flat  # (N_hit, 3)
    # Step 4: Compute distances to 3 vertices
    d0 = np.sum((hit_pts - v0) ** 2, axis=1)
    d1 = np.sum((hit_pts - v1) ** 2, axis=1)
    d2 = np.sum((hit_pts - v2) ** 2, axis=1)
    idx_min = np.argmin(np.stack([d0, d1, d2], axis=0), axis=0)
    # Step 5: Get vertex labels
    l0 = vertex_labels[hit_triangles[:, 0]]
    l1 = vertex_labels[hit_triangles[:, 1]]
    l2 = vertex_labels[hit_triangles[:, 2]]
    # Step 6: Choose the label of the closest vertex
    
    labels_hit = np.where(idx_min == 0, l0, np.where(idx_min == 1, l1, l2))
    # labels_stack = np.stack([l0, l1, l2], axis=1)  # (N_hit, 3)
    # labels_hit = stats.mode(labels_stack, axis=1).mode.squeeze()


    # Step 7: Write into image
    semantic_img = np.full((H, W), fill_value=-1, dtype=np.int32)
    semantic_img[hit_mask] = labels_hit

    return semantic_img



def raycast_semantic_face_labels(sem_mesh, face_labels, K, img_res, c2w):
    """
    Raycast per-face semantic labels using triangle IDs returned from raycast.
    """
    W, H = img_res
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))
    triangles = np.asarray(sem_mesh.triangles)  # (M, 3)

    verts = np.asarray(sem_mesh.vertices)
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # Transform to OpenGL-style world coordinates
    flip_yz = np.diag([1, -1, -1, 1])
    c2w_gl = c2w @ flip_yz
    R = c2w_gl[:3, :3]
    t = c2w_gl[:3, 3]
    cam_origin = t

    # Generate ray directions (OpenGL convention)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    v = H - 1 - v  # Flip Y to match OpenGL
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, -np.ones_like(u)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_world = dirs @ R.T
    rays_o = np.tile(cam_origin, (H, W, 1))

    # Flatten rays for raycasting
    rays = np.concatenate([rays_o, dirs_world], axis=-1).astype(np.float32)
    rays_o3d = o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_o3d)

    triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    # --- Per-face labels ---
    semantic_img = np.full((H, W), fill_value=-1, dtype=np.int32)
    semantic_img[hit_mask] = face_labels[triangle_ids[hit_mask]]

    return semantic_img


def raycast_semantic_face_labels_mcam(sem_mesh, face_labels, M_cam_from_uv,
                                      R_world_from_cam, cam_position, width, height):
    """Raycast per-face semantic labels using Hypersim's M_cam_from_uv convention.

    This matches V-Ray's pixel sampling exactly, avoiding the ~0.5px error
    of the pinhole-from-M_proj approach. See hypersim_intrinsics_bug.md for details.

    Args:
        sem_mesh: Open3D legacy TriangleMesh with plane geometry.
        face_labels: (N_faces,) int array of per-face plane IDs.
        M_cam_from_uv: (3, 3) matrix from metadata_camera_parameters.csv.
        R_world_from_cam: (3, 3) rotation matrix (camera-to-world, from
            camera_keyframe_orientations.hdf5).
        cam_position: (3,) camera centre in world space (from
            camera_keyframe_positions.hdf5).
        width, height: Image dimensions in pixels.

    Returns:
        (H, W) int32 array of plane IDs (-1 = no hit).
    """
    W, H = width, height
    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))

    # Official Hypersim UV grid: half-pixel offsets within [-1, 1]
    half_du = 1.0 / W
    half_dv = 1.0 / H
    u = np.linspace(-1 + half_du, 1 - half_du, W)
    v = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]  # top row = largest v
    uu, vv = np.meshgrid(u, v)

    # Camera-space ray directions via M_cam_from_uv
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    dirs_cam = uvs @ M_cam_from_uv.T                      # (H, W, 3)

    # World-space ray directions
    dirs_world = dirs_cam @ R_world_from_cam.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    # Raycast
    origins = np.broadcast_to(cam_position, dirs_world.shape).copy()
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    ans = scene_rc.cast_rays(
        o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    )

    triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    semantic_img = np.full((H, W), fill_value=-1, dtype=np.int32)
    semantic_img[hit_mask] = face_labels[triangle_ids[hit_mask]]
    return semantic_img  # NO flipud needed — y-flip built into UV grid

