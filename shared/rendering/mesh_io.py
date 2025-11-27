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


def read_ply_faces_with_plane_ids(ply_path: str):
    """
    Read our planes.ply (ASCII or binary).
    For binary:
      vertices: float32 (x,y,z)
      faces: [uchar count=3][int v0 v1 v2][int plane_id][int label_int]
    For ASCII:
      face line: "3 v0 v1 v2 plane_id label_int"

    Returns:
      V (N,3) float32
      F (M,3) int32
      plane_id_face (M,) int32
      label_int_face (M,) int32
    """
    import sys

    with open(ply_path, "rb") as f:
        # ---- header (bytes → ascii) ----
        def _readline():
            b = f.readline()
            if not b:
                raise SystemExit("[ERR] Unexpected EOF in header")
            try:
                return b.decode("ascii", errors="strict").rstrip("\r\n")
            except Exception:
                raise SystemExit("[ERR] Header is not ASCII text (binary body reached before end_header).")

        first = _readline()
        if not first.startswith("ply"):
            sys.exit("[ERR] Not a PLY file")

        fmt = None
        num_verts = None
        num_faces = None

        # Track face properties (we expect list+2 ints after)
        saw_vlist = False
        saw_plane = False
        saw_label = False

        while True:
            line = _readline()
            if line.startswith("format "):
                # ascii / binary_little_endian / binary_big_endian
                parts = line.split()
                fmt = parts[1].strip().lower()
            elif line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                num_faces = int(line.split()[-1])
            elif line.startswith("property list"):
                # property list uchar int vertex_indices
                if "vertex_indices" in line:
                    parts = line.split()
                    if len(parts) >= 5 and parts[-1] == "vertex_indices":
                        saw_vlist = True
            elif line.startswith("property int"):
                if "plane_id" in line:
                    saw_plane = True
                elif "label_int" in line:
                    saw_label = True
            elif line.startswith("end_header"):
                break
            # ignore comments/other props

        if fmt is None or num_verts is None or num_faces is None:
            sys.exit("[ERR] Missing format/vertex/face in header")
        if not saw_vlist or not saw_plane or not saw_label:
            sys.exit("[ERR] Face properties not as expected (need vertex_indices list + plane_id + label_int)")

        # ---- read data section ----
        if fmt == "ascii":
            # vertices
            V = np.zeros((num_verts, 3), dtype=np.float32)
            for i in range(num_verts):
                parts = _readline().split()
                if len(parts) < 3:
                    sys.exit(f"[ERR] Vertex line {i} malformed")
                V[i, 0] = float(parts[0])
                V[i, 1] = float(parts[1])
                V[i, 2] = float(parts[2])

            # faces
            F = np.zeros((num_faces, 3), dtype=np.int32)
            plane_id_face = np.zeros((num_faces,), dtype=np.int32)
            label_int_face = np.zeros((num_faces,), dtype=np.int32)
            for i in range(num_faces):
                parts = _readline().split()
                if len(parts) < 6:
                    sys.exit(f"[ERR] Face line {i} too short")
                if parts[0] != '3':
                    sys.exit(f"[ERR] Face {i} is not a triangle (count={parts[0]})")
                F[i, 0] = int(parts[1]); F[i, 1] = int(parts[2]); F[i, 2] = int(parts[3])
                plane_id_face[i] = int(parts[4])
                label_int_face[i] = int(parts[5])
            return V, F, plane_id_face, label_int_face

        # binary: choose endianness
        if fmt == "binary_little_endian":
            little = True
        elif fmt == "binary_big_endian":
            little = False
        else:
            sys.exit(f"[ERR] Unsupported PLY format: {fmt}")

        dt_v = "<f4" if little else ">f4"
        dt_u1 = np.dtype("<u1" if little else ">u1")
        dt_i4 = np.dtype("<i4" if little else ">i4")

        # vertices
        V = np.fromfile(f, dtype=dt_v, count=num_verts*3)
        if V.size != num_verts*3:
            sys.exit("[ERR] Could not read all vertex data")
        V = V.reshape(num_verts, 3).astype(np.float32, copy=False)

        # faces
        F = np.zeros((num_faces, 3), dtype=np.int32)
        plane_id_face = np.zeros((num_faces,), dtype=np.int32)
        label_int_face = np.zeros((num_faces,), dtype=np.int32)

        for i in range(num_faces):
            # list count
            c = np.fromfile(f, dtype=dt_u1, count=1)
            if c.size != 1:
                sys.exit(f"[ERR] Face {i}: failed to read list count")
            cnt = int(c[0])
            if cnt != 3:
                _ = np.fromfile(f, dtype=dt_i4, count=cnt)  # drain to keep file pointer aligned
                sys.exit(f"[ERR] Face {i} is not a triangle (count={cnt})")
            idx = np.fromfile(f, dtype=dt_i4, count=3)
            if idx.size != 3:
                sys.exit(f"[ERR] Face {i}: could not read 3 indices")
            F[i, :] = idx.astype(np.int32)
            pid_lab = np.fromfile(f, dtype=dt_i4, count=2)
            if pid_lab.size != 2:
                sys.exit(f"[ERR] Face {i}: could not read plane_id/label_int")
            plane_id_face[i] = int(pid_lab[0])
            label_int_face[i] = int(pid_lab[1])

        return V, F, plane_id_face, label_int_face
