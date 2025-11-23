# save_semantic_with_names.py
import os, re, h5py, numpy as np, pandas as pd, trimesh

# --- Set your scene path ---
mesh_dir = "/Users/ahmetcanyavuz/scannetpp_gt/ai_003_007/_detail/mesh"

def load_h5(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with h5py.File(path, "r") as f:
        return next(iter(f.values()))[()]

# --- Load geometry ---
verts = load_h5(f"{mesh_dir}/mesh_vertices.hdf5")
faces = load_h5(f"{mesh_dir}/mesh_faces_vi.hdf5")
if faces.min() == 1:
    faces = faces - 1

# --- Load per-face group ids & group names ---
group_ids = load_h5(f"{mesh_dir}/mesh_faces_gi.hdf5").reshape(-1)
groups_csv = os.path.join(mesh_dir, "metadata_groups.csv")
if not os.path.exists(groups_csv):
    raise FileNotFoundError("metadata_groups.csv not found")
group_df = pd.read_csv(groups_csv)
group_names = group_df["group_name"].tolist()

# Clamp any out-of-range ids just in case
group_ids = np.clip(group_ids, 0, len(group_names) - 1)
raw_names = [group_names[i] for i in group_ids]

# --- Collapse instances to SEMANTIC names ---
def normalize_name(name: str) -> str:
    # "floor_tile_obj_123" -> "floor_tile"
    # "window2_obj_11"     -> "window2"
    # "towel_03"           -> "towel"
    name = re.sub(r"_obj_\d+", "", name)
    name = re.sub(r"_\d+$", "", name)
    return name

semantic_names = [normalize_name(n) for n in raw_names]
unique_sem = sorted(set(semantic_names))
sem_to_id = {n: i for i, n in enumerate(unique_sem)}

# --- Assign colors per semantic class ---
rng = np.random.default_rng(42)  # stable colors across runs
sem_to_color = {n: rng.integers(0, 256, size=3, dtype=np.uint8) for n in unique_sem}
face_colors = np.array([sem_to_color[n] for n in semantic_names], dtype=np.uint8)

# --- Export 1) PLY colored by semantic ---
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.visual.face_colors = face_colors
mesh.export("mesh_by_semantic.ply")
print(f"Saved mesh_by_semantic.ply with {len(unique_sem)} semantic categories")

# --- Export 2) Legend CSV (semantic name ↔ color ↔ id, plus face counts) ---
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
legend_df.to_csv("semantic_legend.csv", index=False)
print("Saved semantic_legend.csv")

# --- Export 3) GLB with named materials per semantic (names embedded) ---
parts = {}
for name in unique_sem:
    mask = np.array(semantic_names) == name
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        continue
    sub_faces = faces[idx]

    # Build a submesh per semantic
    submesh = trimesh.Trimesh(vertices=verts, faces=sub_faces, process=False)
    # Simple flat material with the class name
    color = sem_to_color[name].astype(np.float32) / 255.0
    mat = trimesh.visual.material.SimpleMaterial(
        name=name, diffuse=color
    )
    submesh.visual.material = mat
    parts[name] = submesh

# Create a scene from parts and export to GLB
scene = trimesh.Scene(parts)
scene.export("mesh_by_semantic.glb")
print("Saved mesh_by_semantic.glb with named materials (semantic names embedded)")
