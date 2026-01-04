import time
from contextlib import contextmanager

TIMINGS = {}
GLOBAL_START = time.perf_counter()

def format_time(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMINGS[name] = TIMINGS.get(name, 0.0) + elapsed
    print(f"[TIMER] {name:30s} {format_time(elapsed)}")

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
# import imageio
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
import shutil

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

def evaluate_single_frame(
    scene_id,
    frame_idx,
    depths,
    gt_plane,
    K,
    c2w,
    h5_root,
    thresholds
):
    pred = load_plane_pred_from_moge_h5(h5_root, scene_id, frame_idx)
    if pred is None:
        return None

    pred = cv2.resize(
        pred,
        (depths.shape[1], depths.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    pts_world, labels, _ = backproject(depths, K, c2w, pred)

    metric_thr = {}
    for thr in thresholds:
        _, df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=thr,
            num_iterations=2000,
            min_support=100
        )

        if df is None or len(df) == 0:
            metric_thr[f"prec@{int(thr*100)}cm"] = 0.0
            metric_thr[f"rec@{int(thr*100)}cm"] = 0.0
            continue

        _, df = mark_planes_below_threshold_as_outliers(_, df, 0.5)
        res = compute_precision_recall_v1(df, pts_world.shape[0])

        metric_thr[f"prec@{int(thr*100)}cm"] = res["global_precision"]
        metric_thr[f"rec@{int(thr*100)}cm"] = res["global_recall"]

    ri = rand_score(gt_plane.flatten(), pred.flatten())
    Hs, Hm = variation_of_information(gt_plane, pred)
    sc = segmentation_covering(gt_plane.flatten(), pred.flatten())

    return {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "rand_index": ri,
        "voi": Hs + Hm,
        "sc": sc,
        **metric_thr
    }



def process_batch(
    rgb_paths,
    scene_ids,
    frame_ids,
    gt_planarity, # (B, 1, H, W) or (B, H, W)
    inference_model,
    png_root,
    args
):
    results = inference_model.predict_batch_fast(
        rgb_paths,
        num_tokens=args.num_tokens,
        return_all_heads=True
    )

    # for res, rgb_path, scene_id, frame_id in zip(
    #     results, rgb_paths, scene_ids, frame_ids
    # ):
    for res, rgb_path, scene_id, frame_id, gt_seg in zip(
        results, rgb_paths, scene_ids, frame_ids, gt_planarity
    ):
        img = Image.open(rgb_path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        # 🔁 REPLACEMENT: use GT planarity
        if gt_seg.ndim == 3:
            gt_seg = gt_seg[0]
        gt_seg = gt_seg.cpu().numpy().astype(np.int16)          # 🔑 convert to NumPy
        gt_seg = cv2.resize(gt_seg, (W, H))
        
        labels = gt_seg

   
        scene_dir = os.path.join(png_root, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        imageio.imwrite(
            os.path.join(scene_dir, f"{frame_id}.png"),
            labels.astype(np.uint16)
        )


# Cache opened HDF5 files per scene
# _MOGE_H5_CACHE = {}

# def load_plane_pred_from_moge_h5(moge_h5_root, scene_id, frame_idx):
#     """
#     Load plane prediction for a given scene/frame from:
#         moge_h5_root/<scene_id>/planes.h5

#     Returns:
#         plane_pred (H,W) int32, or None if not found
#     """
#     if scene_id not in _MOGE_H5_CACHE:
#         h5_path = os.path.join(moge_h5_root, scene_id, "planes.h5")

#         if not os.path.exists(h5_path):
#             _MOGE_H5_CACHE[scene_id] = None
#             return None

#         f = h5py.File(h5_path, "r")
#         frame_ids = f["frame_ids"][:].astype(str)   # shape (N,)
#         planes = f["planes"]                        # shape (N,H,W)

#         _MOGE_H5_CACHE[scene_id] = (f, frame_ids, planes)

#     cached = _MOGE_H5_CACHE[scene_id]
#     if cached is None:
#         return None

#     f, frame_ids, planes = cached

#     key = str(frame_idx)
#     if key not in frame_ids:
#         return None

#     idx = np.where(frame_ids == key)[0][0]
#     return planes[idx].astype(np.int32)

def load_plane_pred_from_moge_h5(moge_h5_root, scene_id, frame_idx):
    h5_path = os.path.join(moge_h5_root, scene_id, "planes.h5")
    if not os.path.exists(h5_path):
        return None

    key = str(frame_idx)

    with h5py.File(h5_path, "r") as f:
        frame_ids = f["frame_ids"][:].astype(str)
        planes = f["planes"]

        if key not in frame_ids:
            return None

        idx = np.where(frame_ids == key)[0][0]
        return planes[idx].astype(np.int32)


exp_name = 'gtseg'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
# csv_out_dir = "/cluster/scratch/aoezkan/planeseg/eval/moge_results_v1.csv"
output_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}"
h5_root  = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"
# output_dir = '/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours'
# h5_root  = "/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours_h5"

# model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v2/last_model.pt"
model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
dataset_dir = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
num_workers = 4

max_scenes_val = None
# max_scenes_val = 1

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

BATCH_SIZE = 32
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
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
torch.set_grad_enabled(False)
inference_model.model.eval()

save_root = os.path.join(output_dir)
os.makedirs(save_root, exist_ok=True)

print("==> Running MoGe inference")

with timer("moge_inference_total"):
    for batch in tqdm(val_loader, desc="GT-planarity inference"):
        rgb_paths = batch["rgb_path"]
        scene_ids = batch["scene_id"]
        frame_ids = batch["frame_idx"]
        gt_plane  = batch["plane"]      # ✅ ADD THIS

        with timer("moge_inference_batch"):
            process_batch(
                rgb_paths,
                scene_ids,
                frame_ids,
                gt_plane,              # ✅ PASS GT
                inference_model,        # still needed for depth+normal
                output_dir,
                args
            )

# ============================================================
# 2) PNG → H5 conversion
# ============================================================
os.makedirs(h5_root, exist_ok=True)

with timer("png_to_h5_total"):
    for scene_id in tqdm(os.listdir(output_dir), desc="PNG→H5"):
        scene_dir = os.path.join(output_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue

        pngs = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
        if not pngs:
            continue

        planes, frame_ids = [], []
        with timer("png_read"):
            for p in pngs:
                planes.append(imageio.imread(p).astype(np.uint16))
                frame_ids.append(os.path.splitext(os.path.basename(p))[0])

        planes = np.stack(planes, axis=0)
        frame_ids = np.array(frame_ids, dtype="S")

        with timer("h5_write"):
            os.makedirs(os.path.join(h5_root, scene_id), exist_ok=True)
            with h5py.File(os.path.join(h5_root, scene_id, "planes.h5"), "w") as f:
                f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
                f.create_dataset("frame_ids", data=frame_ids)

shutil.rmtree(output_dir)

# ============================================================
# 3) Metric evaluation
# ============================================================
results = {}
thresholds = (0.01, 0.02, 0.05)

tasks = []

for batch in val_loader:
    B = len(batch["scene_id"])
    for i in range(B):
        tasks.append((
            batch["scene_id"][i],
            batch["frame_idx"][i],
            batch["depth"][i][0].numpy(),
            batch["plane"][i][0].numpy(),
            batch["K"][i].numpy(),
            batch["c2w"][i].numpy(),
        ))

# N_JOBS = min(16, os.cpu_count())  # Euler: often 16–32
# N_JOBS = min(12, os.cpu_count())  # Euler: often 16–32
N_JOBS = min(48, os.cpu_count())  # Euler: often 16–32

with timer("evaluation_total"):
    with tqdm_joblib(tqdm(total=len(tasks), desc="Evaluation", unit="frame")):
        outputs = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            batch_size=1
        )(
            delayed(evaluate_single_frame)(
                scene_id,
                frame_idx,
                depths,
                gt_plane,
                K,
                c2w,
                h5_root,
                thresholds
            )
            for (
                scene_id,
                frame_idx,
                depths,
                gt_plane,
                K,
                c2w
            ) in tasks
        )

for out in outputs:
    if out is not None:
        results[(out["scene_id"], out["frame_idx"])] = out
        
# with timer("evaluation_total"):
#     for batch in tqdm(val_loader, desc="Evaluation"):
#         B = len(batch["scene_id"])
#         for i in range(B):
#             with timer("eval_per_frame"):
#                 scene_id = batch["scene_id"][i]
#                 frame_idx = batch["frame_idx"][i]

#                 depths = batch["depth"][i][0].numpy()
#                 gt_plane = batch["plane"][i][0].numpy()
#                 K = batch["K"][i].numpy()
#                 c2w = batch["c2w"][i].numpy()

#                 with timer("load_prediction"):
#                     pred = load_plane_pred_from_moge_h5(
#                         h5_root, scene_id, frame_idx
#                     )
#                     if pred is None:
#                         continue
#                     pred = cv2.resize(
#                         pred,
#                         (depths.shape[1], depths.shape[0]),
#                         interpolation=cv2.INTER_NEAREST
#                     )

#                 with timer("backproject"):
#                     pts_world, labels, _ = backproject(
#                         depths, K, c2w, pred
#                     )

#                 metric_thr = {}
#                 with timer("plane_fitting"):
#                     for thr in thresholds:
#                         _, df = fit_planes_per_label_v1(
#                             pts_world, labels,
#                             ignore_labels=(0,),
#                             distance_threshold=thr,
#                             num_iterations=2000,
#                             min_support=100
#                         )
#                         if df is None or len(df) == 0:
#                             metric_thr[f"prec@{int(thr*100)}cm"] = 0.0
#                             metric_thr[f"rec@{int(thr*100)}cm"] = 0.0
#                             continue

#                         _, df = mark_planes_below_threshold_as_outliers(_, df, 0.5)
#                         res = compute_precision_recall_v1(df, pts_world.shape[0])
#                         metric_thr[f"prec@{int(thr*100)}cm"] = res["global_precision"]
#                         metric_thr[f"rec@{int(thr*100)}cm"] = res["global_recall"]

#                 with timer("clustering_metrics"):
#                     ri = rand_score(gt_plane.flatten(), pred.flatten())
#                     Hs, Hm = variation_of_information(gt_plane, pred)
#                     sc = segmentation_covering(gt_plane.flatten(), pred.flatten())

#                 results[(scene_id, frame_idx)] = {
#                     "scene_id": scene_id,
#                     "frame_idx": frame_idx,
#                     "rand_index": ri,
#                     "voi": Hs + Hm,
#                     "sc": sc,
#                     **metric_thr
#                 }

# ============================================================
# Save CSVs (unchanged logic)
# ============================================================
# df = pd.DataFrame.from_records(results.values()).set_index(["scene_id", "frame_idx"])
os.makedirs(csv_out_dir, exist_ok=True)
out_path = os.path.join(csv_out_dir, 'moge_ours_merged_results.csv') # "/cluster/scratch/aoezkan/planeseg/eval/moge_results.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

df = pd.DataFrame.from_records(list(results.values()))
df = df.set_index(["scene_id", "frame_idx"])

df.to_csv(out_path)

print(f"Saved results to {out_path}")

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
scene_csv = os.path.join(csv_out_dir, "moge_ours_merged_results_per_scene.csv")
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
dataset_csv = os.path.join(csv_out_dir, "moge_ours_merged_results_dataset.csv")
df_dataset.to_csv(dataset_csv, index=False)

print(f"Saved dataset-level results to {dataset_csv}")
# ============================================================
# FINAL RUNTIME SUMMARY
# ============================================================
TOTAL = time.perf_counter() - GLOBAL_START

print("\n================ Runtime Summary ================")
for k, v in sorted(TIMINGS.items(), key=lambda x: -x[1]):
    print(f"{k:35s} {format_time(v)}")
print("-------------------------------------------------")
print(f"TOTAL WALL TIME                 {format_time(TOTAL)}")
print("=================================================\n")


# ============================================================
# SAVE RUNTIME CSVs
# ============================================================
os.makedirs(csv_out_dir, exist_ok=True)

# ---- Per-stage runtime breakdown ----
runtime_rows = []
for name, seconds in TIMINGS.items():
    runtime_rows.append({
        "stage": name,
        "time_seconds": seconds,
        "time_hms": format_time(seconds),
    })

df_runtime = pd.DataFrame(runtime_rows).sort_values(
    by="time_seconds", ascending=False
)

runtime_breakdown_path = os.path.join(
    csv_out_dir, "runtime_breakdown.csv"
)
df_runtime.to_csv(runtime_breakdown_path, index=False)

# ---- Dataset-level runtime summary ----
df_runtime_summary = pd.DataFrame([{
    "total_wall_time_seconds": TOTAL,
    "total_wall_time_hms": format_time(TOTAL),
}])

runtime_summary_path = os.path.join(
    csv_out_dir, "runtime_summary.csv"
)
df_runtime_summary.to_csv(runtime_summary_path, index=False)

print(f"[✓] Saved runtime breakdown to {runtime_breakdown_path}")
print(f"[✓] Saved runtime summary to   {runtime_summary_path}")