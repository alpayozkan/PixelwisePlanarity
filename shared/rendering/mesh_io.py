"""
Mesh I/O utilities for reading and writing PLY files with labels.

This module handles PLY files with per-vertex or per-face labels,
commonly used in semantic and plane segmentation.
"""

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from collections import Counter
from typing import Tuple


def propagate_face_labels_to_vertices(
    F: np.ndarray,
    face_labels: np.ndarray,
    num_vertices: int
) -> np.ndarray:
    """
    Convert face-level labels to vertex-level using majority voting.

    Each vertex is assigned the most common label among its adjacent faces.

    Args:
        F: (M,3) face vertex indices
        face_labels: (M,) face labels (e.g., plane IDs)
        num_vertices: Total number of vertices

    Returns:
        label_vertex: (N,) per-vertex labels
    """
    vertex_to_labels = [[] for _ in range(num_vertices)]

    # Collect labels from adjacent faces
    for face_idx, face in enumerate(F):
        label = face_labels[face_idx]
        for v in face:
            vertex_to_labels[v].append(label)

    # Assign majority label per vertex
    label_vertex = np.zeros(num_vertices, dtype=np.int32)
    for v, labels in enumerate(vertex_to_labels):
        if labels:
            label_vertex[v] = Counter(labels).most_common(1)[0][0]
        else:
            label_vertex[v] = -1  # Unlabeled

    return label_vertex


def save_mesh_with_vertex_labels(
    mesh: o3d.geometry.TriangleMesh,
    vertex_labels: np.ndarray,
    out_path: str
) -> None:
    """
    Save Open3D mesh to PLY with per-vertex labels.

    Args:
        mesh: Open3D TriangleMesh
        vertex_labels: (N,) int labels per vertex
        out_path: Output PLY file path
    """
    assert len(mesh.vertices) == len(vertex_labels), \
        f"Vertex count mismatch: {len(mesh.vertices)} vs {len(vertex_labels)}"

    # Prepare vertex data with labels
    vertices_np = np.asarray(mesh.vertices)
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')]
    vertex_array = np.empty(len(vertices_np), dtype=vertex_dtype)
    vertex_array['x'] = vertices_np[:, 0]
    vertex_array['y'] = vertices_np[:, 1]
    vertex_array['z'] = vertices_np[:, 2]
    vertex_array['label'] = vertex_labels

    # Prepare face data
    triangles_np = np.asarray(mesh.triangles)
    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_array = np.empty(len(triangles_np), dtype=face_dtype)
    face_array['vertex_indices'] = triangles_np

    # Write PLY
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')
    PlyData([vertex_element, face_element], text=True).write(out_path)

    print(f"[INFO] Mesh with vertex labels written to {out_path}")


def load_mesh_with_vertex_labels(ply_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Load PLY mesh with per-vertex labels.

    Args:
        ply_path: Path to PLY file with 'label' vertex property

    Returns:
        sem_mesh: Open3D TriangleMesh
        vertex_labels: (N,) int32 array of vertex labels
    """
    # Load PLY with labels
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data

    # Extract geometry and labels
    V = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    vertex_labels = vertex_data['label'].astype(np.int32)

    # Extract faces
    face_data = plydata['face'].data['vertex_indices']
    F = np.vstack(face_data).astype(np.int32)

    # Build Open3D mesh
    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V)
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

    return sem_mesh, vertex_labels


def read_ply_faces_with_plane_ids(filepath: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Read PLY with per-face plane_id labels.

    Args:
        filepath: Path to PLY file with 'plane_id' face property

    Returns:
        mesh: Open3D TriangleMesh
        face_plane_ids: (M,) int32 array of face plane IDs
    """
    plydata = PlyData.read(filepath)

    # Vertices
    V = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=-1)

    # Faces
    face_data = plydata['face'].data
    F = np.vstack(face_data['vertex_indices']).astype(np.int32)

    # Plane IDs
    if 'plane_id' in face_data.dtype.names:
        face_plane_ids = face_data['plane_id'].astype(np.int32)
    else:
        face_plane_ids = np.zeros(len(F), dtype=np.int32)

    # Build mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()

    return mesh, face_plane_ids
