"""
Mesh rendering and raycasting utilities.
"""

from .mesh_io import (
    load_mesh_with_vertex_labels,
    propagate_face_labels_to_vertices,
    read_ply_faces_with_plane_ids,
    save_mesh_with_vertex_labels,
)
from .render import (
    raycast_depth,
    raycast_semantic,
    raycast_semantic_face_labels,
    raycast_semantic_face_labels_mcam,
    render_rgb,
    render_rgb_depth,
)

__all__ = [
    "render_rgb",
    "render_rgb_depth",
    "raycast_depth",
    "raycast_semantic",
    "raycast_semantic_face_labels",
    "raycast_semantic_face_labels_mcam",
    "propagate_face_labels_to_vertices",
    "save_mesh_with_vertex_labels",
    "load_mesh_with_vertex_labels",
    "read_ply_faces_with_plane_ids",
]
