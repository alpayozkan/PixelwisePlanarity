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

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
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


exp_name = 'moge_ours_tmp3'
csv_out_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}"
# csv_out_dir = "/cluster/scratch/aoezkan/planeseg/eval/moge_results_v1.csv"
output_dir = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}"
h5_root  = f"/cluster/scratch/aoezkan/planeseg/scannetpp/inference/{exp_name}_h5"
# output_dir = '/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours'
# h5_root  = "/cluster/scratch/aoezkan/planeseg/inference/scannetpp/moge_ours_h5"

# model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v2/last_model.pt"
model_path = "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt"
dataset_dir = scannetpp_rend_plane_path
num_workers = 4

# max_scenes_val = None
max_scenes_val = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = ScanNetPPPlaneDataset(
    rgb_root=os.path.join(scannetpp_path, "data"),
    plane_label_root=scannetpp_rend_plane_path,
    sem_label_root=os.path.join(dataset_dir, ""),
    depth_label_root=scannetpp_rend_plane_path,
    split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
    split="test",
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
torch.set_grad_enabled(False)
inference_model.model.eval()

save_root = os.path.join(output_dir)
os.makedirs(save_root, exist_ok=True)

print("==> Running MoGe inference")

with timer("moge_inference_total"):
    for batch in tqdm(val_loader, desc="MoGe inference"):
        for i in range(batch["image"].shape[0]):
            with timer("moge_inference_per_frame"):
                process_single_frame(
                    batch["rgb_path"][i],
                    batch["scene_id"][i],
                    batch["frame_idx"][i],
                    inference_model,
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

with timer("evaluation_total"):
    for batch in tqdm(val_loader, desc="Evaluation"):
        with timer("eval_per_frame"):
            scene_id = batch["scene_id"][0]
            frame_idx = batch["frame_idx"][0]

            depths = batch["depth"][0][0].numpy()
            gt_plane = batch["plane"][0][0].numpy()
            K = batch["K"][0].numpy()
            c2w = batch["c2w"][0].numpy()

            with timer("load_prediction"):
                pred = load_plane_pred_from_moge_h5(h5_root, scene_id, frame_idx)
                if pred is None:
                    continue
                pred = cv2.resize(pred, (depths.shape[1], depths.shape[0]), interpolation=cv2.INTER_NEAREST)

            with timer("backproject"):
                pts_world, labels, _ = backproject(depths, K, c2w, pred)

            metric_thr = {}
            with timer("plane_fitting"):
                for thr in thresholds:
                    _, df = fit_planes_per_label_v1(
                        pts_world, labels,
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

            with timer("clustering_metrics"):
                ri = rand_score(gt_plane.flatten(), pred.flatten())
                Hs, Hm = variation_of_information(gt_plane, pred)
                sc = segmentation_covering(gt_plane.flatten(), pred.flatten())

            results[(scene_id, frame_idx)] = {
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "rand_index": ri,
                "voi": Hs + Hm,
                "sc": sc,
                **metric_thr
            }

# ============================================================
# Save CSVs (unchanged logic)
# ============================================================
# df = pd.DataFrame.from_records(results.values()).set_index(["scene_id", "frame_idx"])
df = pd.DataFrame.from_records(list(results.values())) \
       .set_index(["scene_id", "frame_idx"])
       
os.makedirs(csv_out_dir, exist_ok=True)
df.to_csv(os.path.join(csv_out_dir, "moge_ours_results.csv"))

df_scene = df.reset_index().groupby("scene_id").mean(numeric_only=True)
df_scene["num_frames"] = df.reset_index().groupby("scene_id").size()
df_scene.to_csv(os.path.join(csv_out_dir, "moge_ours_results_per_scene.csv"))

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