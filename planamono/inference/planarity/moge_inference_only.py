#!/usr/bin/env python3
import os
import h5py
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from types import SimpleNamespace

from planamono.paths import *
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.inference.planarity.moge_inference import MoGePlanarityInference


# ============================================================
# HDF5 saving utility
# ============================================================

def save_moge_frame(h5_frames_group, frame_id, res):
    """Save one MoGe inference result into an HDF5 group."""
    grp = h5_frames_group.create_group(frame_id)

    def write(name, arr, dtype=None):
        if arr is None:
            return
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        grp.create_dataset(
            name,
            data=arr,
            compression="gzip",
            compression_opts=4,
            shuffle=True
        )

    write("planarity_probability",      res.get("planarity_probability"),      np.float32)
    write("planarity_binary",           res.get("planarity_binary"),           np.uint8)
    write("planarity_probability_full", res.get("planarity_probability_full"), np.float32)
    write("planarity_binary_full",      res.get("planarity_binary_full"),      np.uint8)
    write("mask",                       res.get("mask"),                       np.float32)
    write("normal",                     res.get("normal"),                     np.float32)
    write("points",                     res.get("points"),                     np.float32)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"

H5_ROOT = "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_ours_h5_only"

DATASET_DIR = scannetpp_rend_plane_path
RGB_ROOT = os.path.join(scannetpp_path, "data")

NUM_TOKENS = 1024
MAX_SCENES_VAL = None   # set to int for debugging

CACHE_DIR = os.path.join(H5_ROOT, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SCENE_INDEX_CACHE = os.path.join(
    CACHE_DIR,
    "scannetpp_val_scene_to_indices.pkl"
)


# ============================================================
# Safety checks
# ============================================================

assert torch.cuda.is_available(), "CUDA is required for MoGe inference"
assert os.path.isfile(MODEL_PATH), f"Model not found: {MODEL_PATH}"

device = torch.device("cuda")


# ============================================================
# Dataset (VAL split = source of truth)
# ============================================================

val_dataset = ScanNetPPPlaneDataset(
    rgb_root=RGB_ROOT,
    plane_label_root=scannetpp_rend_plane_path,
    sem_label_root=DATASET_DIR,
    depth_label_root=scannetpp_rend_plane_path,
    split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
    split="val",
    max_scenes=MAX_SCENES_VAL,
)

print(f"[INFO] Dataset size: {len(val_dataset)} frames")


# ============================================================
# Build scene → dataset indices map
# ============================================================

# scene_to_indices = defaultdict(list)

# print("[INFO] Building scene → frame index mapping...")
# for idx in tqdm(range(len(val_dataset)), desc="Indexing dataset"):
#     sample = val_dataset[idx]
#     scene_to_indices[sample["scene_id"]].append(idx)

# scene_ids = sorted(scene_to_indices.keys())
# print(f"[INFO] Val scenes: {len(scene_ids)}")

import pickle
from collections import defaultdict

# ------------------------------------------------------------
# Scene → frame index mapping (cached)
# ------------------------------------------------------------

if os.path.isfile(SCENE_INDEX_CACHE):
    print(f"[INFO] Loading scene index cache: {SCENE_INDEX_CACHE}")
    with open(SCENE_INDEX_CACHE, "rb") as f:
        scene_to_indices = pickle.load(f)
else:
    print("[INFO] Building scene → frame index mapping...")
    scene_to_indices = defaultdict(list)

    for idx in tqdm(range(len(val_dataset)), desc="Indexing dataset"):
        sample = val_dataset[idx]
        scene_to_indices[sample["scene_id"]].append(idx)

    # Convert to plain dict before saving
    scene_to_indices = dict(scene_to_indices)

    with open(SCENE_INDEX_CACHE, "wb") as f:
        pickle.dump(scene_to_indices, f)

    print(f"[INFO] Saved scene index cache to: {SCENE_INDEX_CACHE}")

scene_ids = sorted(scene_to_indices.keys())
print(f"[INFO] Val scenes: {len(scene_ids)}")



# ============================================================
# Load MoGe model
# ============================================================

args = SimpleNamespace(
    model_path=MODEL_PATH,
    num_tokens=NUM_TOKENS,
    device=device,
)

inference_model = MoGePlanarityInference(
    args.model_path,
    device=args.device
)

# Disable memory-efficient attention (stability)
inference_model.model.encoder.use_memory_efficient_attention = False


# ============================================================
# Scene-level MoGe → HDF5 generation
# ============================================================

os.makedirs(H5_ROOT, exist_ok=True)

for scene_id in tqdm(scene_ids, desc="Saving MoGe H5 (val)"):

    scene_h5_dir = os.path.join(H5_ROOT, scene_id)
    os.makedirs(scene_h5_dir, exist_ok=True)

    h5_path = os.path.join(scene_h5_dir, "moge.h5")

    # --------------------------------------------------------
    # Resume safety
    # --------------------------------------------------------
    if os.path.isfile(h5_path):
        print(f"[SKIP] {scene_id} already exists")
        continue

    indices = scene_to_indices[scene_id]

    # --------------------------------------------------------
    # Sort frames deterministically by frame index
    # --------------------------------------------------------
    indices = sorted(
        indices,
        key=lambda i: val_dataset[i]["frame_idx"]
    )

    frame_ids = []

    with h5py.File(h5_path, "w") as f:
        frames_grp = f.create_group("frames")

        for idx in tqdm(indices, leave=False):
            sample = val_dataset[idx]

            rgb_path = sample["rgb_path"]
            frame_id = sample["frame_idx"]

            res = inference_model.predict(
                rgb_path,
                num_tokens=NUM_TOKENS,
                return_all_heads=True
            )

            save_moge_frame(frames_grp, frame_id, res)
            frame_ids.append(frame_id)

        f.create_dataset(
            "frame_ids",
            data=np.array(frame_ids, dtype="S"),
            compression="gzip"
        )

        # Metadata (useful later)
        f.attrs["model"] = "MoGe-2-ViT-L + planarity head"
        f.attrs["num_tokens"] = NUM_TOKENS
        f.attrs["split"] = "val"

    print(f"[✓] {scene_id}: {len(frame_ids)} frames → {h5_path}")


print("[DONE] All scenes processed.")
