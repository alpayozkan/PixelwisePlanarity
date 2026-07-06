#!/usr/bin/env python3
"""
Hypersim Mesh Processing Script

Processes Hypersim mesh files to create semantic-colored PLY and GLB exports.

Usage:
    python mesh_processing.py --mesh_dir /path/to/scene/_detail/mesh --output_dir /path/to/output
"""
import os
import re
import h5py
import numpy as np
import pandas as pd
import trimesh
import argparse


def load_h5(path):
    """Load HDF5 file and return first dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with h5py.File(path, "r") as f:
        return next(iter(f.values()))[()]


def normalize_name(name: str) -> str:
    """
    Collapse instance names to semantic names.
    "floor_tile_obj_123" -> "floor_tile"
    "window2_obj_11"     -> "window2"
    "towel_03"           -> "towel"
    """
    name = re.sub(r"_obj_\d+", "", name)
    name = re.sub(r"_\d+$", "", name)
    return name


def process_mesh(mesh_dir, output_dir):
    """
    Process Hypersim mesh and create semantic exports.

    Args:
        mesh_dir: Directory containing mesh HDF5 files and metadata
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading mesh from: {mesh_dir}")

    # Load geometry
    verts = load_h5(os.path.join(mesh_dir, "mesh_vertices.hdf5"))
    faces = load_h5(os.path.join(mesh_dir, "mesh_faces_vi.hdf5"))
    if faces.min() == 1:
        faces = faces - 1

    print(f"[INFO] Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}")

    # Load per-face group ids & group names
    group_ids = load_h5(os.path.join(mesh_dir, "mesh_faces_gi.hdf5")).reshape(-1)
    groups_csv = os.path.join(mesh_dir, "metadata_groups.csv")
    if not os.path.exists(groups_csv):
        raise FileNotFoundError(f"metadata_groups.csv not found in {mesh_dir}")

    group_df = pd.read_csv(groups_csv)
    group_names = group_df["group_name"].tolist()

    # Clamp any out-of-range ids
    group_ids = np.clip(group_ids, 0, len(group_names) - 1)
    raw_names = [group_names[i] for i in group_ids]

    # Collapse instances to semantic names
    semantic_names = [normalize_name(n) for n in raw_names]
    unique_sem = sorted(set(semantic_names))
    sem_to_id = {n: i for i, n in enumerate(unique_sem)}

    print(f"[INFO] Found {len(unique_sem)} semantic categories")

    # Assign colors per semantic class
    rng = np.random.default_rng(42)  # stable colors across runs
    sem_to_color = {n: rng.integers(0, 256, size=3, dtype=np.uint8) for n in unique_sem}
    face_colors = np.array([sem_to_color[n] for n in semantic_names], dtype=np.uint8)

    # Export 1) PLY colored by semantic
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = face_colors
    ply_path = os.path.join(output_dir, "mesh_by_semantic.ply")
    mesh.export(ply_path)
    print(f"[SAVED] {ply_path}")

    # Export 2) Legend CSV
    face_counts = pd.Series(semantic_names).value_counts().to_dict()
    legend_rows = []
    for name in unique_sem:
        r, g, b = map(int, sem_to_color[name])
        legend_rows.append({
            "semantic_id": sem_to_id[name],
            "semantic_name": name,
            "R": r, "G": g, "B": b,
            "face_count": face_counts.get(name, 0)
        })
    legend_df = pd.DataFrame(legend_rows).sort_values("semantic_id")
    csv_path = os.path.join(output_dir, "semantic_legend.csv")
    legend_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    # Export 3) GLB with named materials per semantic
    parts = {}
    for name in unique_sem:
        mask = np.array(semantic_names) == name
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        sub_faces = faces[idx]

        submesh = trimesh.Trimesh(vertices=verts, faces=sub_faces, process=False)
        color = sem_to_color[name].astype(np.float32) / 255.0
        mat = trimesh.visual.material.SimpleMaterial(
            name=name, diffuse=color
        )
        submesh.visual.material = mat
        parts[name] = submesh

    scene = trimesh.Scene(parts)
    glb_path = os.path.join(output_dir, "mesh_by_semantic.glb")
    scene.export(glb_path)
    print(f"[SAVED] {glb_path}")

    print(f"[DONE] Processed mesh with {len(unique_sem)} semantic categories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Hypersim mesh to create semantic-colored exports"
    )
    parser.add_argument("--mesh_dir", type=str, required=True,
                        help="Directory containing mesh HDF5 files (mesh_vertices.hdf5, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output files")
    args = parser.parse_args()

    if not os.path.isdir(args.mesh_dir):
        print(f"[ERROR] Mesh directory not found: {args.mesh_dir}")
        exit(1)

    process_mesh(args.mesh_dir, args.output_dir)
