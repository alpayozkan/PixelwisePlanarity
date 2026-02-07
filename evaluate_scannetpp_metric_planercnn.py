import os
import glob
import shutil
import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import cv2
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.plane_fitting import (
    backproject_v1,
    fit_planes_per_label_v1,
    mark_planes_below_threshold_as_outliers,
    compute_precision_recall_v1,
)

from planamono.evaluation.quantitative.evaluator import segmentation_covering
from planamono.paths import (
    scannetpp_rend_plane_path,
    repo_path,
)

# ---------------------------------------------------------
# Paths / config
# ---------------------------------------------------------
# PlaneRCNN predictions in "flat" format:
#   {H5_ROOT}/{scene_id}/planes.h5
# with datasets:
#   - frame_ids: e.g. b'frame_000025'
#   - planes:    (N, H, W) int labels
H5_ROOT = "/cluster/scratch/ayavuz/dataset/planercnn_correct_format"

# Where to write evaluation CSVs
CSV_OUT_DIR = "/cluster/scratch/ayavuz/dataset/planercnn_eval"

# thresholds in meters (UNCHANGED)
THRESHOLDS = (0.01, 0.02, 0.05)

# ---------------------------------------------------------
# Dataset (UNCHANGED)
# ---------------------------------------------------------
dataset_dir = "/cluster/scratch/ayavuz/dataset/backup"
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

print(len(val_dataset))
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
print(f"[INFO] Validation frames: {len(val_dataset)}")


# ------------------------------------------------------------
# STEP 2: Cache for predictions (ONLY READING LOGIC CHANGED)
# ------------------------------------------------------------
_H5_CACHE = {}


def _decode_frame_ids(raw):
    """Decode H5 frame_ids into a list of Python strings."""
    out = []
    for x in raw:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

def _normalize_frame_idx(frame_idx):
    # frame_idx can be int-like, or a string like "frame_000000"
    if isinstance(frame_idx, (int, np.integer)):
        return int(frame_idx)
    s = str(frame_idx)
    if s.startswith("frame_"):
        return int(s.split("_")[1])
    return int(s)


def load_plane_pred_from_h5(h5_root, scene_id, frame_idx):
    """Load PlaneRCNN plane-instance segmentation for (scene_id, frame_idx)."""

    if scene_id not in _H5_CACHE:
        h5_path = os.path.join(h5_root, scene_id, "planes.h5")
        if not os.path.exists(h5_path):
            _H5_CACHE[scene_id] = None
            return None

        f = h5py.File(h5_path, "r")

        raw_ids = f["frame_ids"][:]
        frame_ids = _decode_frame_ids(raw_ids)

        # Build O(1) lookup: 'frame_000025' -> index
        index_map = {fid: i for i, fid in enumerate(frame_ids)}

        planes = f["planes"]
        _H5_CACHE[scene_id] = (f, index_map, planes)

    cached = _H5_CACHE[scene_id]
    if cached is None:
        return None

    f, index_map, planes = cached

    fi = _normalize_frame_idx(frame_idx)
    key = f"frame_{fi:06d}"

    idx = index_map.get(key, None)

    # Optional fallbacks (harmless; helps if some files were produced differently)
    if idx is None:
        idx = index_map.get(str(frame_idx), None)
    if idx is None:
        idx = index_map.get(f"{fi:06d}", None)

    if idx is None:
        return None

    return planes[idx]


# ------------------------------------------------------------
# STEP 3: Evaluation (UNCHANGED LOGIC)
# ------------------------------------------------------------
print("==> Evaluating PlaneRCNN predictions...")

results = {}

for batch in tqdm(val_loader):

    scene_id = batch["scene_id"][0]
    frame_idx = batch["frame_idx"][0]

    depth = batch["depth"][0][0].cpu().numpy()
    gt_plane = batch["plane"][0][0].cpu().numpy()
    K = batch["K"][0].cpu().numpy()
    c2w = batch["c2w"][0].cpu().numpy()

    plane_pred = load_plane_pred_from_h5(H5_ROOT, scene_id, frame_idx)
    if plane_pred is None:
        continue

    plane_pred = cv2.resize(
        plane_pred,
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    pts_world, labels, _ = backproject_v1(
        depth, K, c2w, plane_pred
    )

    metric_per_thr = {}
    for thr in THRESHOLDS:
        res, plane_df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=thr,
            ransac_n=3,
            num_iterations=2000,
            min_support=100,
        )

        if plane_df is None or len(plane_df) == 0:
            metric_per_thr[f"prec@{int(thr*100)}cm"] = 0.0
            metric_per_thr[f"rec@{int(thr*100)}cm"] = 0.0
            continue

        res, plane_df = mark_planes_below_threshold_as_outliers(
            res, plane_df, inlier_ratio_threshold=0.5
        )

        pr = compute_precision_recall_v1(
            plane_df, total_scene_points=pts_world.shape[0]
        )

        metric_per_thr[f"prec@{int(thr*100)}cm"] = float(pr["global_precision"])
        metric_per_thr[f"rec@{int(thr*100)}cm"] = float(pr["global_recall"])

    ri = rand_score(gt_plane.flatten(), plane_pred.flatten())
    voi = sum(variation_of_information(gt_plane, plane_pred))
    sc = segmentation_covering(gt_plane.flatten(), plane_pred.flatten())

    results[(scene_id, frame_idx)] = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "rand_index": ri,
        "voi": voi,
        "sc": sc,
        **metric_per_thr,
    }

# ------------------------------------------------------------
# STEP 4: CSVs (ONLY FILENAMES CHANGED)
# ------------------------------------------------------------
os.makedirs(CSV_OUT_DIR, exist_ok=True)

df = pd.DataFrame.from_records(list(results.values()))
df = df.set_index(["scene_id", "frame_idx"])
df.to_csv(os.path.join(CSV_OUT_DIR, "planercnn_results.csv"))

df_reset = df.reset_index()
scene_group = df_reset.groupby("scene_id")

df_scene = scene_group.mean(numeric_only=True)
df_scene["num_frames"] = scene_group.size()

df_scene.to_csv(os.path.join(CSV_OUT_DIR, "planercnn_results_per_scene.csv"))

dataset_stats = {
    "num_scenes": len(df_scene),
    "num_frames_total": int(df_scene["num_frames"].sum()),
}

for c in df_scene.columns:
    if c != "num_frames":
        dataset_stats[f"{c}_mean"] = df_scene[c].mean()
        dataset_stats[f"{c}_std"] = df_scene[c].std()

pd.DataFrame([dataset_stats]).to_csv(
    os.path.join(CSV_OUT_DIR, "planercnn_results_dataset.csv"),
    index=False,
)

print("[✓] PlaneRCNN ScanNet++ evaluation finished.")
print(f"[✓] Wrote CSVs to: {CSV_OUT_DIR}")
