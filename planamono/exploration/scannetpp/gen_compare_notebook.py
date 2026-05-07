#!/usr/bin/env python3
"""Generate the compare_segmentation_proposals.ipynb notebook."""
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# ── Cell 0: Title ──
cells.append(nbformat.v4.new_markdown_cell("""\
# Segmentation Algorithm Comparison

Compare existing (v5, v10) and proposed (v12: v10 + boundary merge) segmentation
algorithms against GT and ZeroPlane on ScanNet++.

**Methods:**
- **GT**: Ground truth plane segmentation
- **ZeroPlane**: Transformer-based baseline (loaded from H5)
- **MoGe+v5**: Sobel edge detection (current production default)
- **MoGe+v10**: Per-edge cosine + adaptive voting (best 3D precision)
- **MoGe+v12**: v10 + fast boundary-aware coplanar merge (reduces over-segmentation)"""))

# ── Cell 1: Imports ──
cells.append(nbformat.v4.new_code_cell("""\
import os, sys, time, random
import numpy as np
import torch
import cv2, h5py
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from PIL import Image

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.segmentation.plan2seg import (
    compute_vectorized_planar_segments_v5,
    compute_vectorized_planar_segments_v10,
    compute_vectorized_planar_segments_v12,
)
from planamono.shared.plane_fitting.planefit import backproject_v1, fit_planes_per_label_v1
from planamono.shared.utils.label_utils import remap_labels
from planamono.shared.utils.visualization import visualize_top_components_v2
from planamono.evaluation.quantitative.eval_utils import evaluate_single_frame
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

%matplotlib inline
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.max_open_warning"] = 50"""))

# ── Cell 2: Configuration ──
cells.append(nbformat.v4.new_code_cell("""\
# ── Paths ──
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
DATASET_DIR = scannetpp_rend_plane_path
RGB_ROOT = os.path.join(scannetpp_path, "data")
MODEL_PATH = "/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt"
NUM_TOKENS = 1600
H5_ZEROPLANE = "zeroplane_mixed_h5_dust3r_75k_h5"

# ── Evaluation ──
THRESHOLDS = (0.001, 0.005, 0.01)  # 1mm, 5mm, 10mm
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
VIS_THRESHOLD = 0.01  # for inlier visualization

# ── Scene selection ──
N_SCENES = 5
SEED = 42
SPLIT = "test"

# ── Per-method segmentation parameters ──
METHODS_CONFIG = {
    "v5": dict(
        threshold_planarity=0.6, normal_threshold_deg=10.0,
        depth_threshold=0.05, neighbor_match_count_thresh=24,
    ),
    "v10": dict(
        threshold_planarity=0.3, normal_threshold_deg=5.0,
        depth_threshold=0.025, adaptive_frac=0.75,
        min_valid_neighbors=3, min_segment_pixels=50,
    ),
    "v12": dict(
        normal_threshold_deg=5.0,
        depth_threshold=0.025, adaptive_frac=0.75,
        min_valid_neighbors=3, min_segment_pixels=50,
        mask_planarity_thresh=0.1, vote_planarity_thresh=0.3,
        merge_normal_deg=10.0, merge_depth_thresh=0.03,
        merge_min_boundary=5, merge_gap_px=3,
    ),
}

METHOD_NAMES = ["GT", "ZeroPlane", "MoGe+v5", "MoGe+v10", "MoGe+v12"]
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]"""))

# ── Cell 3: Helper functions ──
cells.append(nbformat.v4.new_markdown_cell("## Helper Functions"))

cells.append(nbformat.v4.new_code_cell("""\
def load_h5_predictions(h5_folder, scene_id, frame_idx, nonplanar_label=None):
    \"\"\"Load predicted segmentation from H5 file.\"\"\"
    h5_path = H5_ROOT / h5_folder / scene_id / "planes.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        frame_ids = [
            fid.decode() if isinstance(fid, bytes) else str(fid)
            for fid in f["frame_ids"][:]
        ]
        idx = None
        for i, fid in enumerate(frame_ids):
            if fid.lstrip("0") == str(frame_idx).lstrip("0"):
                idx = i
                break
        if idx is None:
            return None
        labels = f["planes"][idx].astype(np.int32)
    if nonplanar_label is not None:
        labels[labels == nonplanar_label] = 0
    return labels


def compute_inlier_map(depth, K, c2w, seg_labels, threshold=0.01, inlier_ratio_gate=0.9):
    \"\"\"Compute 2D inlier map: 0=bg, 1=inlier(green), -1=outlier(red), -2=rejected(orange).\"\"\"
    H, W = depth.shape
    inlier_map = np.zeros((H, W), dtype=np.int8)
    pts, pt_labels, valid_idx = backproject_v1(depth, K, c2w, seg_labels)
    if pts.shape[0] == 0:
        return inlier_map
    results, df = fit_planes_per_label_v1(
        pts, pt_labels, ignore_labels=(0,),
        distance_threshold=threshold,
        num_iterations=RANSAC_ITERATIONS, min_support=100,
    )
    if df is None or len(df) == 0:
        return inlier_map
    plane_params = {
        pid: d["plane_model_refined"]
        for pid, d in results.items()
        if "plane_model_refined" in d
    }
    for pid, (a, b, c_p, d) in plane_params.items():
        mask_3d = pt_labels == pid
        pts_p, ix = pts[mask_3d], valid_idx[mask_3d]
        if len(pts_p) == 0:
            continue
        dists = np.abs(pts_p @ np.array([a, b, c_p]) + d)
        ratio = np.sum(dists < threshold) / len(pts_p)
        for i, idx in enumerate(ix):
            r, c = idx // W, idx % W
            if ratio >= inlier_ratio_gate:
                inlier_map[r, c] = 1 if dists[i] < threshold else -1
            else:
                inlier_map[r, c] = -2
    return inlier_map


def colorize_inliers(inlier_map, rgb, alpha=0.6):
    \"\"\"Overlay inlier map on RGB: green=inlier, red=outlier, orange=rejected.\"\"\"
    vis = rgb.astype(np.float32)
    if vis.max() > 1:
        vis /= 255.0
    overlay = vis.copy()
    overlay[inlier_map == 1] = [0, 0.8, 0]
    overlay[inlier_map == -1] = [0.9, 0, 0]
    overlay[inlier_map == -2] = [0.9, 0.6, 0]
    mask = inlier_map != 0
    result = vis.copy()
    result[mask] = alpha * overlay[mask] + (1 - alpha) * vis[mask]
    return np.clip(result, 0, 1)"""))

# ── Cell 5: Dataset & Model header ──
cells.append(nbformat.v4.new_markdown_cell("## Dataset & Model"))

# ── Cell 6: Dataset + scene selection ──
cells.append(nbformat.v4.new_code_cell("""\
dataset = ScanNetPPPlaneDataset(
    rgb_root=RGB_ROOT, plane_label_root=scannetpp_rend_plane_path,
    sem_label_root=DATASET_DIR, depth_label_root=scannetpp_rend_plane_path,
    split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"), split=SPLIT,
)
print(f"Dataset: {len(dataset)} frames")

# Group frames by scene, select N random scenes + 1 random frame each
scene_to_indices = defaultdict(list)
for i in range(len(dataset)):
    rgb_path = dataset.valid_pairs[i][0]
    scene_to_indices[rgb_path.split("/")[-4]].append(i)

random.seed(SEED)
selected_scenes = random.sample(list(scene_to_indices.keys()), N_SCENES)
selected_frames = [random.choice(scene_to_indices[s]) for s in selected_scenes]

for i, (s, f) in enumerate(zip(selected_scenes, selected_frames)):
    print(f"  {i+1}. {s}  ({len(scene_to_indices[s])} frames)")"""))

# ── Cell 7: Model loading ──
cells.append(nbformat.v4.new_code_cell("""\
print(f"Loading MoGe model from: {MODEL_PATH}")
moge_model = MoGePlanarityInference(MODEL_PATH, device="cuda")
moge_model.model.eval()
torch.set_grad_enabled(False)
print("Model loaded.")"""))

# ── Cell 8: Processing header ──
cells.append(nbformat.v4.new_markdown_cell("## Process All Scenes"))

# ── Cell 9: Main processing loop ──
cells.append(nbformat.v4.new_code_cell("""\
all_results = []

for frame_i, (scene_id, dataset_idx) in enumerate(zip(selected_scenes, selected_frames)):
    print(f"\\n{'='*60}")
    print(f"Scene {frame_i+1}/{N_SCENES}: {scene_id}")
    print(f"{'='*60}")

    # ── Load GT ──
    sample = dataset[dataset_idx]
    gt_seg = sample["plane"].squeeze(0).numpy().astype(np.int32)
    depth_gt = sample["depth"].squeeze(0).numpy()
    K_gt = sample["K"].numpy()
    c2w = sample["c2w"].numpy()
    frame_idx = sample["frame_idx"]
    H_gt, W_gt = depth_gt.shape

    rgb_img = np.array(Image.open(sample["rgb_path"]).convert("RGB"))
    rgb_gt = cv2.resize(rgb_img, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)

    # ── ZeroPlane from H5 ──
    zp_labels = load_h5_predictions(H5_ZEROPLANE, scene_id, frame_idx, nonplanar_label=20)
    if zp_labels is not None and zp_labels.shape != (H_gt, W_gt):
        zp_labels = cv2.resize(zp_labels, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)

    # ── MoGe inference (shared for all our methods) ──
    t0 = time.time()
    moge_out = moge_model.predict_metric(
        sample["rgb_path"], num_tokens=NUM_TOKENS, return_all_heads=True
    )
    t_moge = time.time() - t0

    planarity = cv2.resize(moge_out["planarity_probability"], (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
    depth_moge = cv2.resize(moge_out["depth"], (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
    normal_moge = cv2.resize(moge_out["normal"], (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
    print(f"  MoGe inference: {t_moge:.2f}s, planarity coverage: {(planarity > 0.3).mean():.1%}")

    # ── Run segmentation methods ──
    seg_labels = {"GT": gt_seg, "ZeroPlane": zp_labels}
    timings = {"GT": 0, "ZeroPlane": 0}

    # v5
    p = METHODS_CONFIG["v5"]
    t0 = time.time()
    mask_v5 = (planarity > p["threshold_planarity"]).astype(np.int32)
    lbl, _ = compute_vectorized_planar_segments_v5(
        mask_v5, normal_moge, depth_moge,
        np.deg2rad(p["normal_threshold_deg"]), p["depth_threshold"],
        p["neighbor_match_count_thresh"],
    )
    lbl, _ = remap_labels(lbl)
    seg_labels["MoGe+v5"] = lbl
    timings["MoGe+v5"] = (time.time() - t0) * 1000

    # v10
    p = METHODS_CONFIG["v10"]
    t0 = time.time()
    mask_v10 = (planarity > p["threshold_planarity"]).astype(np.int32)
    lbl, _ = compute_vectorized_planar_segments_v10(
        mask_v10, normal_moge, depth_moge,
        np.deg2rad(p["normal_threshold_deg"]), p["depth_threshold"],
        adaptive_frac=p["adaptive_frac"],
        min_valid_neighbors=p["min_valid_neighbors"],
        min_segment_pixels=p["min_segment_pixels"],
    )
    lbl, _ = remap_labels(lbl)
    seg_labels["MoGe+v10"] = lbl
    timings["MoGe+v10"] = (time.time() - t0) * 1000

    # v12 (low threshold + voting + gap-bridging merge)
    p = METHODS_CONFIG["v12"]
    t0 = time.time()
    lbl, _ = compute_vectorized_planar_segments_v12(
        planarity, normal_moge, depth_moge,
        np.deg2rad(p["normal_threshold_deg"]), p["depth_threshold"],
        adaptive_frac=p["adaptive_frac"],
        min_valid_neighbors=p["min_valid_neighbors"],
        min_segment_pixels=p["min_segment_pixels"],
        mask_planarity_thresh=p["mask_planarity_thresh"],
        vote_planarity_thresh=p["vote_planarity_thresh"],
        merge_normal_deg=p["merge_normal_deg"],
        merge_depth_thresh=p["merge_depth_thresh"],
        merge_min_boundary=p["merge_min_boundary"],
        merge_gap_px=p["merge_gap_px"],
    )
    lbl, _ = remap_labels(lbl)
    seg_labels["MoGe+v12"] = lbl
    timings["MoGe+v12"] = (time.time() - t0) * 1000

    # ── Evaluate all methods ──
    frame_result = dict(
        scene_id=scene_id, frame_idx=frame_idx,
        rgb=rgb_gt, planarity=planarity,
        depth_gt=depth_gt, depth_moge=depth_moge,
        seg={}, inlier={}, metrics={}, timings=timings,
    )

    for name, labels in seg_labels.items():
        if labels is None:
            continue
        m, _ = evaluate_single_frame(
            scene_id, frame_idx, depth_gt, gt_seg, K_gt, c2w, labels,
            thresholds=THRESHOLDS, compute_plane_metrics_flag=True,
            ransac_iterations=RANSAC_ITERATIONS, inlier_ratio_gate=INLIER_RATIO_GATE,
        )
        frame_result["metrics"][name] = m
        frame_result["seg"][name] = labels
        frame_result["inlier"][name] = compute_inlier_map(
            depth_gt, K_gt, c2w, labels,
            threshold=VIS_THRESHOLD, inlier_ratio_gate=INLIER_RATIO_GATE,
        )
        n_seg = len(np.unique(labels)) - 1
        sc = m.get("sc", 0)
        p1 = m.get("prec@1.0cm", 0)
        r1 = m.get("rec@1.0cm", 0)
        t = timings.get(name, 0)
        print(f"  {name:18s}: {n_seg:3d} planes | SC={sc:.3f} P@1cm={p1:.3f} R@1cm={r1:.3f} | {t:.0f}ms")

    all_results.append(frame_result)

print(f"\\nDone. Processed {N_SCENES} scenes.")"""))

# ── Cell 10: Visualization header ──
cells.append(nbformat.v4.new_markdown_cell("## Qualitative Comparison"))

# ── Cell 11: Per-scene visualization ──
cells.append(nbformat.v4.new_code_cell("""\
for fr in all_results:
    active = [m for m in METHOD_NAMES if m in fr["seg"]]
    n_m = len(active)
    n_cols = n_m + 1  # first column: RGB / Planarity

    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 10))
    fig.suptitle(
        f"Scene: {fr['scene_id']}  |  Frame: {fr['frame_idx']}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Row 0: RGB + segmentation maps ──
    axes[0, 0].imshow(fr["rgb"])
    axes[0, 0].set_title("RGB", fontsize=9)
    axes[0, 0].axis("off")

    for j, method in enumerate(active):
        seg_rgb = visualize_top_components_v2(
            fr["seg"][method], k=20, ignore_label=0, return_colors=True
        )
        axes[0, j + 1].imshow(seg_rgb)
        n_planes = len(np.unique(fr["seg"][method])) - 1
        t_ms = fr["timings"].get(method, 0)
        axes[0, j + 1].set_title(f"{method}\\n({n_planes} seg, {t_ms:.0f}ms)", fontsize=8)
        axes[0, j + 1].axis("off")

    # ── Row 1: Planarity + inlier maps ──
    axes[1, 0].imshow(fr["planarity"], cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title("Planarity", fontsize=9)
    axes[1, 0].axis("off")

    for j, method in enumerate(active):
        inlier_rgb = colorize_inliers(fr["inlier"][method], fr["rgb"], alpha=0.6)
        axes[1, j + 1].imshow(inlier_rgb)
        im = fr["inlier"][method]
        n_in = int((im == 1).sum())
        n_out = int((im == -1).sum())
        n_rej = int((im == -2).sum())
        axes[1, j + 1].set_title(
            f"Inliers @1cm\\nG:{n_in} R:{n_out} O:{n_rej}", fontsize=8,
        )
        axes[1, j + 1].axis("off")

    # ── Row 2: Bar chart (full-width) ──
    for ax in axes[2, :]:
        ax.remove()
    ax_bar = fig.add_subplot(3, 1, 3)

    metric_keys = [
        "sc", "prec@0.1cm", "rec@0.1cm",
        "prec@0.5cm", "rec@0.5cm", "prec@1.0cm", "rec@1.0cm",
    ]
    metric_labels = ["SC", "P@1mm", "R@1mm", "P@5mm", "R@5mm", "P@1cm", "R@1cm"]

    x = np.arange(len(metric_keys))
    width = 0.8 / n_m
    for j, method in enumerate(active):
        if method not in fr["metrics"]:
            continue
        vals = [fr["metrics"][method].get(k, 0) for k in metric_keys]
        offset = (j - n_m / 2 + 0.5) * width
        ax_bar.bar(
            x + offset, vals, width,
            label=method, color=COLORS[j % len(COLORS)], alpha=0.85,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=9)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(loc="upper right", fontsize=7, ncol=3)
    ax_bar.set_title("Metrics Comparison", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()"""))

# ── Cell 12: Quantitative summary header ──
cells.append(nbformat.v4.new_markdown_cell("## Quantitative Summary"))

# ── Cell 13: Summary table + aggregated charts ──
cells.append(nbformat.v4.new_code_cell("""\
# ── Build DataFrame ──
rows = []
for fr in all_results:
    for method in METHOD_NAMES:
        if method not in fr["metrics"]:
            continue
        m = fr["metrics"][method]
        rows.append({
            "scene": fr["scene_id"], "method": method,
            "SC": m.get("sc", np.nan), "RI": m.get("rand_index", np.nan),
            "VOI": m.get("voi", np.nan),
            "P@1mm": m.get("prec@0.1cm", np.nan), "R@1mm": m.get("rec@0.1cm", np.nan),
            "P@5mm": m.get("prec@0.5cm", np.nan), "R@5mm": m.get("rec@0.5cm", np.nan),
            "P@1cm": m.get("prec@1.0cm", np.nan), "R@1cm": m.get("rec@1.0cm", np.nan),
        })
df = pd.DataFrame(rows)

# ── Aggregated table ──
print(f"{'='*110}")
print(f"AGGREGATED RESULTS (mean over {N_SCENES} scenes)")
print(f"{'='*110}")
agg = df.groupby("method").mean(numeric_only=True).reset_index()
agg["method"] = pd.Categorical(agg["method"], categories=METHOD_NAMES, ordered=True)
agg = agg.sort_values("method")
cols = ["method", "SC", "RI", "VOI", "P@1mm", "R@1mm", "P@5mm", "R@5mm", "P@1cm", "R@1cm"]
print(agg[cols].to_string(index=False, float_format="%.3f"))

# ── Runtime summary ──
print(f"\\n{'='*110}")
print(f"RUNTIME (mean ms, segmentation only — excludes MoGe inference)")
print(f"{'='*110}")
for method in METHOD_NAMES:
    times = [fr["timings"].get(method, np.nan) for fr in all_results]
    valid = [t for t in times if not np.isnan(t) and t > 0]
    if valid:
        print(f"  {method:18s}: {np.mean(valid):7.1f} ms")

# ── Aggregated bar charts ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

methods = agg["method"].tolist()
x = np.arange(len(methods))

# Left: 2D segmentation metrics
w2d = 0.25
ax1.bar(x - w2d, agg["SC"], w2d, label="SC", color="#2ecc71")
ax1.bar(x, agg["RI"], w2d, label="RI", color="#3498db")
ax1.bar(x + w2d, agg["VOI"] / agg["VOI"].max(), w2d, label="VOI (norm)", color="#e74c3c", alpha=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
ax1.set_title("2D Segmentation Metrics (higher=better, VOI normalized)")
ax1.legend()
ax1.set_ylim(0, 1.15)
ax1.grid(axis="y", alpha=0.3)

# Right: 3D precision/recall
w3d = 0.12
offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
colors_3d = ["#27ae60", "#2ecc71", "#2980b9", "#3498db", "#c0392b", "#e74c3c"]
for i, (pk, rk, lbl) in enumerate([
    ("P@1mm", "R@1mm", "1mm"), ("P@5mm", "R@5mm", "5mm"), ("P@1cm", "R@1cm", "1cm"),
]):
    ax2.bar(x + offsets[2*i] * w3d, agg[pk], w3d, label=f"P@{lbl}", color=colors_3d[2*i], alpha=0.85)
    ax2.bar(x + offsets[2*i+1] * w3d, agg[rk], w3d, label=f"R@{lbl}", color=colors_3d[2*i+1], alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
ax2.set_title("3D Plane Metrics (Precision / Recall)")
ax2.legend(fontsize=7, ncol=3)
ax2.set_ylim(0, 1.15)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ── Per-scene detail ──
print(f"\\n{'='*110}")
print("PER-SCENE BREAKDOWN")
print(f"{'='*110}")
for scene_id in [fr["scene_id"] for fr in all_results]:
    scene_df = df[df["scene"] == scene_id]
    print(f"\\n--- {scene_id} ---")
    print(scene_df[["method", "SC", "RI", "P@5mm", "R@5mm", "P@1cm", "R@1cm"]]
          .to_string(index=False, float_format="%.3f"))"""))

nb.cells = cells
out_path = "/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/exploration/scannetpp/compare_segmentation_proposals.ipynb"
nbformat.write(nb, out_path)
print(f"Notebook written to {out_path}")
