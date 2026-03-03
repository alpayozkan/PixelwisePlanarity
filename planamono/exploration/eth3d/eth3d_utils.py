"""
ETH3D dataset utilities: COLMAP binary readers, data loading, raycasting.

Provides:
  - COLMAP binary format readers (cameras.bin, images.bin)
  - Scene data loading (RGB, depth, calibration)
  - OpenCV-convention raycasting for per-face plane labels
"""

import struct
import numpy as np
import h5py
import open3d as o3d
from pathlib import Path
from PIL import Image
from collections import namedtuple

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/cluster/project/cvg/Shared_datasets/ETH3D/ETH3D_undistorted_resizedx2")
MESH_ROOT = Path("/cluster/scratch/aoezkan/planeseg/dataset_mesh/eth3d")

# Scenes with GT dense depth (training split)
SCENES_WITH_GT = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "pipes", "playground", "relief",
    "relief_2", "terrace", "terrains",
]

# ── COLMAP binary readers ──────────────────────────────────────────────────────

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
}

ColmapCamera = namedtuple("ColmapCamera", ["id", "model", "width", "height", "params"])
ColmapImage = namedtuple("ColmapImage", ["id", "qvec", "tvec", "camera_id", "name"])


def read_cameras_binary(path):
    """Read COLMAP cameras.bin → dict[cam_id → ColmapCamera]."""
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = ColmapCamera(
                cam_id, CAMERA_MODELS[model_id].model_name, width, height, params
            )
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin → dict[img_id → ColmapImage]."""
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # skip 2D points
            images[img_id] = ColmapImage(img_id, np.array(qvec), np.array(tvec), camera_id, name)
    return images


def qvec_to_rotmat(qvec):
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def get_intrinsics_matrix(camera):
    """ColmapCamera → (3×3) intrinsics matrix K."""
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = f
    else:
        fx, fy, cx, cy = camera.params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def get_c2w(colmap_image):
    """ColmapImage → (4×4) camera-to-world matrix."""
    R = qvec_to_rotmat(colmap_image.qvec)
    t = colmap_image.tvec
    # COLMAP stores world-to-camera: p_cam = R @ p_world + t
    # Camera center: C = -R^T @ t
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


# ── Scene data loading ─────────────────────────────────────────────────────────

def load_scene_calibration(scene_name, root=None):
    """Load COLMAP calibration for a scene.

    Returns:
        cameras: dict[cam_id → ColmapCamera]
        images: dict[img_id → ColmapImage]
    """
    if root is None:
        root = DATASET_ROOT
    root = Path(root)

    # Try multiple locations
    for subdir in [
        root / scene_name / "dslr_calibration_undistorted",
        root / scene_name / "dslr_calibration_undistorted" / "reconstruction",
    ]:
        cam_path = subdir / "cameras.bin"
        img_path = subdir / "images.bin"
        if cam_path.exists() and img_path.exists():
            return read_cameras_binary(str(cam_path)), read_images_binary(str(img_path))
    raise FileNotFoundError(f"No calibration found for {scene_name}")


def load_frame(scene_name, frame_stem, root=None):
    """Load RGB image and dense GT depth for a single frame.

    Args:
        scene_name: e.g. "courtyard"
        frame_stem: e.g. "DSC_0286" (without extension)
        root: dataset root (default: DATASET_ROOT)

    Returns:
        rgb: (H, W, 3) uint8
        depth: (H, W) float32 in meters, 0 = invalid
    """
    if root is None:
        root = DATASET_ROOT
    root = Path(root)

    img_path = root / scene_name / "images" / "dslr_images_undistorted" / f"{frame_stem}.png"
    rgb = np.array(Image.open(img_path))

    depth = None
    for depth_dir_name in ["ground_truth_depth_dense", "ground_truth_depth"]:
        h5_path = root / scene_name / depth_dir_name / "dslr_images_undistorted" / f"{frame_stem}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                depth = f["depth"][:].astype(np.float32)
            break

    return rgb, depth


def list_frames(scene_name, root=None):
    """List all frame stems for a scene (sorted)."""
    if root is None:
        root = DATASET_ROOT
    root = Path(root)
    img_dir = root / scene_name / "images" / "dslr_images_undistorted"
    return sorted([p.stem for p in img_dir.glob("*.png")])


def load_mesh(scene_name, mesh_root=None):
    """Load ETH3D surface mesh as Open3D TriangleMesh.

    Returns:
        mesh: o3d.geometry.TriangleMesh (with vertex normals computed)
    """
    if mesh_root is None:
        mesh_root = MESH_ROOT
    mesh_root = Path(mesh_root)
    mesh_path = mesh_root / scene_name / scene_name / "occlusion" / "surface_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# ── Raycasting (OpenCV convention) ─────────────────────────────────────────────

def raycast_face_labels(mesh, face_labels, K, img_res, c2w):
    """Raycast per-face labels onto a 2D image using OpenCV camera convention.

    Args:
        mesh: o3d.geometry.TriangleMesh
        face_labels: (N_faces,) int32 array of per-face plane IDs
        K: (3,3) camera intrinsics
        img_res: (width, height) tuple
        c2w: (4,4) camera-to-world matrix (OpenCV convention)

    Returns:
        label_img: (H, W) int32, face_labels values at each pixel, -1 = no hit
        depth_img: (H, W) float32, z-depth in meters, 0 = no hit
    """
    W, H = img_res
    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(t_mesh)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera center and rotation in world coordinates
    R = c2w[:3, :3]
    cam_origin = c2w[:3, 3]

    # OpenCV convention: z forward, y down
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_world = dirs_cam @ R.T  # (H, W, 3)

    rays_o = np.broadcast_to(cam_origin, (H, W, 3))
    rays = np.concatenate([rays_o, dirs_world], axis=-1).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_tensor)

    triangle_ids = ans["primitive_ids"].numpy().reshape(H, W)
    t_hit = ans["t_hit"].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    # Per-face labels
    label_img = np.full((H, W), -1, dtype=np.int32)
    label_img[hit_mask] = face_labels[triangle_ids[hit_mask]]

    # Z-depth = t_hit * cos(angle_from_optical_axis)
    # For pinhole: z = t_hit * (dir_cam_z / |dir_cam|) = t_hit * dir_cam_z (already normalized)
    # Since dirs_cam was normalized, we need the original z component before normalization
    ray_z = np.ones((H, W), dtype=np.float64)
    ray_len = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    depth_img = np.zeros((H, W), dtype=np.float32)
    depth_img[hit_mask] = (t_hit[hit_mask] / ray_len[hit_mask]).astype(np.float32)

    return label_img, depth_img
