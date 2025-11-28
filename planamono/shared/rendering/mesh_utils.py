import numpy as np
from collections import Counter
from plyfile import PlyData, PlyElement
import open3d as o3d

def propagate_face_labels_to_vertices(F, face_labels, num_vertices):
    """
    Args:
        F : (M, 3) array of face vertex indices
        face_labels : (M,) array of face-level labels (e.g., plane_id_face)
        num_vertices : total number of vertices (N)

    Returns:
        label_vertex : (N,) array of per-vertex labels
    """
    vertex_to_labels = [[] for _ in range(num_vertices)]

    # Collect labels per vertex from faces
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
            label_vertex[v] = -1  # or some default/unlabeled value

    return label_vertex


def save_mesh_with_vertex_labels(mesh, vertex_labels, out_path):
    """
    Save Open3D mesh to PLY file with per-vertex 'label' field using plyfile.
    """
    assert len(mesh.vertices) == len(vertex_labels), "Vertex count mismatch with labels"

    # Prepare vertex data
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

    # Create PlyElements
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')

    # Write to file
    PlyData([vertex_element, face_element], text=True).write(out_path)
    print(f"[INFO] Mesh with vertex labels written to {out_path}")

def load_mesh_with_vertex_labels(ply_path):
    """
    Loads a mesh with per-vertex labels saved in 'label' field.
    Returns:
        - Open3D TriangleMesh (sem_mesh)
        - vertex_labels: (N,) int32 numpy array
    """
    # Load the full PLY with labels
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data

    # Extract xyz and label
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
