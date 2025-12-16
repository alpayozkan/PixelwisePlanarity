import os
import torch
from torch.utils.data import DataLoader

from planamono.shared.datasets.scannetpp import ScanNetPPPlanarityDataset, ScanNetPPPlaneDataset
from planamono.paths import *
# from planamono.moge.moge_planarity_scannet import MoGePlanarityHead

import os
import argparse
import numpy as np
import glob
import cv2
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from planamono.evaluation.quantitative.evaluator import segmentation_covering, process_single_frame
from planamono.shared.segmentation import compute_vectorized_planar_segments_v4
from planamono.shared.utils.label_utils import remap_labels
from planamono.shared.utils import visualize_top_components_v1
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
import imageio
import cv2
import numpy as np
import pandas as pd
import h5py
import os

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information
from tqdm import tqdm

from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
    mark_planes_below_threshold_as_outliers,
    compute_precision_recall_v1,
    project_labels_to_image
)
from planamono.paths import *
from planamono.evaluation.quantitative.evaluator import segmentation_covering

import os
import h5py
import numpy as np

import os
import glob
import h5py
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm


import cv2
import numpy as np
import pandas as pd
import h5py
import os

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information
from tqdm import tqdm

from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
    mark_planes_below_threshold_as_outliers,
    compute_precision_recall_v1,
    project_labels_to_image
)
from planamono.paths import *
from planamono.evaluation.quantitative.evaluator import segmentation_covering

import os
import h5py
import numpy as np

# Cache opened HDF5 files per scene
_MOGE_H5_CACHE = {}

def load_plane_pred_from_moge_h5(moge_h5_root, scene_id, frame_idx):
    """
    Load plane prediction for a given scene/frame from:
        moge_h5_root/<scene_id>/planes.h5

    Returns:
        plane_pred (H,W) int32, or None if not found
    """
    if scene_id not in _MOGE_H5_CACHE:
        h5_path = os.path.join(moge_h5_root, scene_id, "planes.h5")

        if not os.path.exists(h5_path):
            _MOGE_H5_CACHE[scene_id] = None
            return None

        f = h5py.File(h5_path, "r")
        frame_ids = f["frame_ids"][:].astype(str)   # shape (N,)
        planes = f["planes"]                        # shape (N,H,W)

        _MOGE_H5_CACHE[scene_id] = (f, frame_ids, planes)

    cached = _MOGE_H5_CACHE[scene_id]
    if cached is None:
        return None

    f, frame_ids, planes = cached

    key = str(frame_idx)
    if key not in frame_ids:
        return None

    idx = np.where(frame_ids == key)[0][0]
    return planes[idx].astype(np.int32)


csv_out_dir = "/cluster/scratch/aoezkan/planeseg/scannetpp/eval/"
# csv_out_dir = "/cluster/scratch/aoezkan/planeseg/eval/moge_results_v1.csv"
output_dir = '/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours'
h5_root  = "/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours_h5"

model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v2/last_model.pt"
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

# max_scenes = 1
# max_scenes = None
# max_scenes_val = None
max_scenes_val = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = ScanNetPPPlaneDataset(
    rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
    plane_label_root=scannetpp_rend_plane_path,
    sem_label_root=os.path.join(dataset_dir, ""),
    depth_label_root=scannetpp_rend_plane_path,
    split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
    split="val",
    max_scenes=max_scenes_val,
)

# print(len(train_dataset), len(val_dataset))
print(len(val_dataset))

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

from types import SimpleNamespace

args = SimpleNamespace(
    model_path=model_path,
    output_root=output_dir,
    model_size="large",
    device=device,
    cache_dir=None,
    num_tokens=1024,
    max_scenes=None,
    threshold_planarity=0.6,
    normal_threshold_deg=10.0,
    depth_threshold=0.05,
    neighbor_match_count_thresh=24,
)

if not os.path.isfile(args.model_path):
    print(f"[ERROR] Model not found: {args.model_path}")

inference_model = MoGePlanarityInference(
    args.model_path, device=args.device
)

inference_model.model.encoder.use_memory_efficient_attention = False

save_root = os.path.join(output_dir)
os.makedirs(save_root, exist_ok=True)

print("==> Running MoGe + segmentation on val set...")

for batch in tqdm(val_loader):

    image       = batch["image"]      # [B, 3, H, W]
    rgb_path    = batch["rgb_path"]   # list of strings
    scene_ids   = batch["scene_id"]   # list of strings
    frame_ids   = batch["frame_idx"]  # list of strings

    B = image.shape[0]

    for i in range(B):
        labels = \
        process_single_frame(
            rgb_path[i],
            scene_ids[i],
            frame_ids[i],
            inference_model,
            save_root,
            args
        )
        # break
    # break




# Create h5 from pngs
png_root = output_dir # "/cluster/scratch/aoezkan/planeseg/inference/moge"

os.makedirs(h5_root, exist_ok=True)

scene_ids = sorted([
    d for d in os.listdir(png_root)
    if os.path.isdir(os.path.join(png_root, d))
])

print(f"Found {len(scene_ids)} scenes")

for scene_id in tqdm(scene_ids, desc="Converting scenes"):

    scene_png_dir = os.path.join(png_root, scene_id)
    png_paths = sorted(glob.glob(os.path.join(scene_png_dir, "*.png")))

    if len(png_paths) == 0:
        print(f"[!] No PNGs found for {scene_id}, skipping")
        continue

    planes_list = []
    frame_ids   = []

    for p in png_paths:
        labels = imageio.imread(p).astype(np.uint16)  # (H,W)
        planes_list.append(labels)

        frame_id = os.path.splitext(os.path.basename(p))[0]
        frame_ids.append(frame_id)

    planes = np.stack(planes_list, axis=0)      # (N,H,W)
    frame_ids = np.array(frame_ids, dtype="S")  # bytes for HDF5

    # --------------------------------------------------
    # Create scene folder and save planes.h5 inside it
    # --------------------------------------------------
    scene_h5_dir = os.path.join(h5_root, scene_id)
    os.makedirs(scene_h5_dir, exist_ok=True)

    h5_path = os.path.join(scene_h5_dir, "planes.h5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "planes",
            data=planes,
            compression="gzip",
            compression_opts=4
        )
        f.create_dataset("frame_ids", data=frame_ids)

    print(f"[✓] {scene_id}: {planes.shape} → {h5_path}")


# metric-evaluation
moge_h5_root = h5_root # '/cluster/scratch/aoezkan/planeseg/inference/moge_ours'
results = {}
thresholds=(0.01, 0.02, 0.05)

idx=0
for batch in tqdm(val_loader):
    scene_id = batch["scene_id"][0]
    frame_idx = batch["frame_idx"][0]

    # --- GT from loader ---
    depths_np = batch["depth"][0][0].cpu().numpy()
    gt_plane_np = batch["plane"][0][0].cpu().numpy()
    K_np = batch["K"][0].cpu().numpy()
    c2w_np = batch["c2w"][0].cpu().numpy()

    # --- Load prediction from H5 ---
    plane_pred = load_plane_pred_from_moge_h5(
        moge_h5_root,
        scene_id,
        frame_idx
    )
    plane_pred = cv2.resize(
        plane_pred,
        (depths_np.shape[1], depths_np.shape[0]),  # (W,H)
        interpolation=cv2.INTER_NEAREST
    )

    if plane_pred is None:
        print('plane_pred None: ', scene_id, " | ", frame_idx)
        continue

    # --- Geometry evaluation ---
    pts_world, labels, valid_idx = backproject(
        depths_np, K_np, c2w_np, plane_pred
    )

    metric_per_threshold = {}
    for thr in thresholds:
        results_planefit, plane_df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=thr,
            ransac_n=3,
            num_iterations=2000,
            min_support=100
        )

        # if no plane fitted => 0 prec 0 rec
        if plane_df is None or len(plane_df) == 0:
            metric_per_threshold[f"prec@{int(thr*100)}cm"] = 0.0
            metric_per_threshold[f"rec@{int(thr*100)}cm"]  = 0.0
            continue
            
        results_planefit, plane_df = mark_planes_below_threshold_as_outliers(
            results_planefit,
            plane_df,
            inlier_ratio_threshold=0.5
        )

        metric_res = compute_precision_recall_v1(
            plane_df,
            total_scene_points=pts_world.shape[0]
        )

        metric_per_threshold[f"prec@{int(thr*100)}cm"] = float(metric_res["global_precision"])
        metric_per_threshold[f"rec@{int(thr*100)}cm"]  = float(metric_res["global_recall"])

    # --- Clustering metrics ---
    labels_true = gt_plane_np.astype(np.int32)
    labels_pred = plane_pred.astype(np.int32)

    ri = rand_score(labels_true.flatten(), labels_pred.flatten())
    H_split, H_merge = variation_of_information(labels_true, labels_pred)
    voi = H_split + H_merge
    sc = segmentation_covering(labels_true.flatten(), labels_pred.flatten())

    # --- Store results ---
    results[(scene_id, frame_idx)] = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "rand_index": ri,
        "voi": voi,
        "sc": sc,
        **metric_per_threshold
    }

out_path = os.path.join(csv_out_dir, 'moge_results.csv') # "/cluster/scratch/aoezkan/planeseg/eval/moge_results.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

df = pd.DataFrame.from_records(list(results.values()))
df = df.set_index(["scene_id", "frame_idx"])

df.to_csv(out_path)

print(f"Saved results to {out_path}")


os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Reset index for grouping
# -------------------------
df_reset = df.reset_index()  # scene_id, frame_idx become columns

# -------------------------
# Aggregate per scene
# -------------------------
scene_group = df_reset.groupby("scene_id")

df_scene = scene_group.mean(numeric_only=True)
df_scene["num_frames"] = scene_group.size()

# Optional: reorder columns (count first)
cols = ["num_frames"] + [c for c in df_scene.columns if c != "num_frames"]
df_scene = df_scene[cols]

# -------------------------
# Save
# -------------------------
scene_csv = os.path.join(csv_out_dir, "moge_results_per_scene.csv")
df_scene.to_csv(scene_csv)

print(f"Saved per-scene results to {scene_csv}")
print(f"Number of scenes: {len(df_scene)}")


# -------------------------
# Dataset-level stats
# -------------------------
dataset_stats = {}

numeric_cols = df_scene.select_dtypes(include="number").columns
metric_cols = [c for c in numeric_cols if c != "num_frames"]

# Mean & std across scenes
dataset_stats["num_scenes"] = len(df_scene)
dataset_stats["num_frames_total"] = int(df_scene["num_frames"].sum())

for c in metric_cols:
    dataset_stats[f"{c}_mean"] = df_scene[c].mean()
    dataset_stats[f"{c}_std"]  = df_scene[c].std()

df_dataset = pd.DataFrame([dataset_stats])

# -------------------------
# Save
# -------------------------
dataset_csv = os.path.join(csv_out_dir, "moge_results_dataset.csv")
df_dataset.to_csv(dataset_csv, index=False)

print(f"Saved dataset-level results to {dataset_csv}")