"""
Visualize inliers/outliers of 7-Scenes GT planes and dump full per-frame metrics.

For N random frames from SevenScenesPlaneDataset:
  - fit each GT plane via RANSAC (fit_planes_per_label_v1)
  - count inliers at multiple distance thresholds
  - render a per-frame PNG with inlier/outlier overlay
  - compute the full metric set via eval_utils.evaluate_single_frame

Outputs go to <output_root>/<exp_name>/:
  - frame_<k>_idx<idx>.png         per-frame visualization
  - per_frame_metrics.csv          one row per frame
  - summary.csv                    mean/std across frames
  - metrics.txt                    human-readable dump (per-frame + summary)
  - run_config.txt                 the configuration used

Usage:
    python planamono/exploration/visualize_inliers_sevenscenes_gt.py
    python planamono/exploration/visualize_inliers_sevenscenes_gt.py --n-samples 20 --seed 0
    python planamono/exploration/visualize_inliers_sevenscenes_gt.py --output-root /tmp/foo
"""

import argparse
import datetime
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from planamono.paths import sevenscenes_path
from planamono.shared.datasets.sevenscenes_plane_dataset import SevenScenesPlaneDataset
from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
)
from planamono.shared.plane_fitting.metrics import (
    compute_inliers_at_threshold_with_indices,
)
from planamono.evaluation.quantitative.eval_utils import evaluate_single_frame


# ============================================================
# CONFIG DEFAULTS
# ============================================================

# Match ScanNet++ evaluation setup (evaluate_all_baselines.py):
#   THRESHOLDS = (1mm, 5mm, 1cm), gate = 0.9, RANSAC = 200 iters,
#   RANSAC distance = evaluation threshold (refit per threshold via evaluate_single_frame v1).
THRESHOLDS = (0.001, 0.005, 0.01, 0.025, 0.05)  # 1mm, 5mm, 1cm, 2.5cm, 5cm
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
MIN_SUPPORT = 100
DEFAULT_OUTPUT_ROOT = Path(
    "/cluster/scratch/aoezkan/planeseg/exploration/sevenscenes_gt_inliers"
)


# ============================================================
# CORE: inlier computation (matches visualize_and_generate_pdf.py)
# ============================================================

def compute_gt_inliers(plane_seg, depth_np, K_np, c2w_np,
                       distance_threshold, inlier_ratio_gate, ransac_iterations):
    """RANSAC-fit each GT plane and return inlier/predicted masks + stats."""
    H, W = depth_np.shape
    pts_world, labels, valid_idx = backproject(depth_np, K_np, c2w_np, plane_seg)

    empty_stats = {
        "precision": 0.0, "recall": 0.0, "num_inliers": 0,
        "num_valid_planes": 0, "total_predicted_points": 0,
    }
    empty_mask = np.zeros((H, W), dtype=bool)
    if pts_world.shape[0] == 0:
        return empty_mask, empty_mask, empty_stats

    results, df = fit_planes_per_label_v1(
        pts_world, labels, ignore_labels=(0,),
        distance_threshold=distance_threshold,
        num_iterations=ransac_iterations,
        min_support=MIN_SUPPORT,
    )
    if df is None or len(df) == 0:
        return empty_mask, empty_mask, empty_stats

    plane_params = {pid: data["plane_model_refined"]
                    for pid, data in results.items()
                    if "plane_model_refined" in data}

    metrics = compute_inliers_at_threshold_with_indices(
        pts_world, labels, plane_params, distance_threshold, inlier_ratio_gate
    )

    inlier_mask = np.zeros(H * W, dtype=bool)
    if metrics["inlier_indices"]:
        inlier_mask[valid_idx[metrics["inlier_indices"]]] = True
    inlier_mask = inlier_mask.reshape(H, W)

    predicted_mask = np.zeros(H * W, dtype=bool)
    nonzero = labels != 0
    predicted_mask[valid_idx[nonzero]] = True
    predicted_mask = predicted_mask.reshape(H, W)

    stats = {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "num_inliers": int(metrics["num_inliers"]),
        "num_valid_planes": int(metrics["num_valid_planes"]),
        "total_predicted_points": int(metrics["total_predicted_points"]),
    }
    return inlier_mask, predicted_mask, stats


# ============================================================
# VISUALIZATION
# ============================================================

def colorize_segmentation(seg, seed=0):
    seg = seg.astype(np.int32)
    H, W = seg.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for pid in np.unique(seg):
        if pid == 0:
            continue
        out[seg == pid] = rng.random(3)
    return out


def overlay_inlier_outlier(rgb, predicted_mask, inlier_mask, alpha=0.55):
    rgb_f = rgb.astype(np.float32) / 255.0 if rgb.max() > 1 else rgb.astype(np.float32)
    out = rgb_f.copy()
    outlier_mask = predicted_mask & ~inlier_mask
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[inlier_mask] = (1 - alpha) * rgb_f[inlier_mask] + alpha * green
    out[outlier_mask] = (1 - alpha) * rgb_f[outlier_mask] + alpha * red
    return np.clip(out, 0, 1)


def render_frame_png(out_path, rgb_np, depth_np, gt_seg, inlier_masks_per_thr,
                     predicted_mask, frame_idx, stats_per_thr, scene_id=None):
    """Layout: RGB | depth | seg | one overlay panel per evaluation threshold."""
    sorted_thrs = sorted(inlier_masks_per_thr.keys())
    n_thr = len(sorted_thrs)
    fig, axes = plt.subplots(1, 3 + n_thr, figsize=(3.5 * (3 + n_thr), 3.6))

    title_idx = f"{scene_id}/{frame_idx}" if scene_id else f"idx={frame_idx}"
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB ({title_idx})")

    valid = depth_np > 0
    dvis = np.where(valid, depth_np, np.nan)
    axes[1].imshow(dvis, cmap="viridis")
    axes[1].set_title("GT depth")

    seed = int(frame_idx) if str(frame_idx).isdigit() else 0
    axes[2].imshow(colorize_segmentation(gt_seg, seed=seed))
    axes[2].set_title(f"GT segmentation ({int(gt_seg.max())} planes)")

    for i, thr in enumerate(sorted_thrs):
        s = stats_per_thr[thr]
        axes[3 + i].imshow(overlay_inlier_outlier(rgb_np, predicted_mask, inlier_masks_per_thr[thr]))
        axes[3 + i].set_title(
            f"Inliers (G) / Outliers (R) @ {thr*1000:.1f}mm\n"
            f"P={s['precision']:.3f} R={s['recall']:.3f} planes={s['num_valid_planes']}"
        )

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="7-Scenes GT inlier visualization + metrics")
    p.add_argument("--data-root", default=sevenscenes_path,
                   help=f"7-Scenes dataset root (default: {sevenscenes_path})")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of random frames to evaluate (ignored if --all-frames)")
    p.add_argument("--all-frames", action="store_true",
                   help="Evaluate every frame in the split (sequential 0..N-1, ignores --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Parent directory for the experiment subfolder")
    p.add_argument("--exp-name", default=None,
                   help="Subfolder name (overwritten on re-run). "
                        "Default: sevenscenes_gt_all OR sevenscenes_gt_n<N>_seed<S>")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS),
                   help="Distance thresholds in meters")
    p.add_argument("--inlier-ratio-gate", type=float, default=INLIER_RATIO_GATE)
    p.add_argument("--ransac-iters", type=int, default=RANSAC_ITERATIONS)
    p.add_argument("--max-pngs", type=int, default=None,
                   help="Cap on PNG visualizations saved (0 = no PNGs). "
                        "Default: save one PNG per evaluated frame.")
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--image-width", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()

    thresholds = tuple(sorted(args.thresholds))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        exp_name = args.exp_name
    elif args.all_frames:
        exp_name = "sevenscenes_gt_all"
    else:
        exp_name = f"sevenscenes_gt_n{args.n_samples}_seed{args.seed}"
    out_dir = args.output_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    ds = SevenScenesPlaneDataset(
        data_root=args.data_root,
        split="val",
        image_height=args.image_height,
        image_width=args.image_width,
    )

    if args.all_frames:
        sample_indices = list(range(len(ds)))
        print(f"Evaluating ALL {len(sample_indices)} frames", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(len(ds), size=args.n_samples, replace=False).tolist()
        print(f"Sampled {len(sample_indices)} frames (seed={args.seed}): {sample_indices}", flush=True)

    requested_max = args.max_pngs if args.max_pngs is not None else len(sample_indices)
    n_pngs = min(requested_max, len(sample_indices))
    print(f"Will save first {n_pngs} frames as PNG", flush=True)

    with open(out_dir / "run_config.txt", "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"effective_thresholds: {thresholds}\n")
        f.write(f"num_frames: {len(sample_indices)}\n")
        f.write(f"timestamp: {timestamp}\n")

    verbose = len(sample_indices) <= 20  # full per-frame logging only for short runs
    all_rows = []
    skipped = []  # (idx, reason) — corrupt NPZs etc.

    for k, idx in enumerate(tqdm(sample_indices, desc="frames")):
        try:
            sample = ds[idx]
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            npz_name = Path(ds.valid_pairs[idx][0]).name
            tqdm.write(f"[skip] ds idx={idx} ({npz_name}): {type(e).__name__}: {e}")
            skipped.append((idx, npz_name, f"{type(e).__name__}: {e}"))
            continue
        rgb_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        depth_np = sample["depth"].squeeze(0).numpy().astype(np.float32)
        gt_seg = sample["plane"].squeeze(0).numpy().astype(np.int32)
        K_np = sample["K"].numpy().astype(np.float32)
        c2w_np = sample["c2w"].numpy().astype(np.float32)
        frame_idx = sample["frame_idx"]
        scene_id = sample["scene_id"]

        if verbose:
            tqdm.write(f"\n=== Frame {k+1}/{len(sample_indices)} (sample idx={idx}, "
                       f"scene={scene_id}, frame_idx={frame_idx}) ===")
            tqdm.write(f"  RGB {rgb_np.shape}, depth {depth_np.shape}, "
                       f"planes={int(gt_seg.max())}, valid_depth={(depth_np > 0).mean():.1%}")

        # Per-threshold inlier mask + stats (also feeds the PNG)
        save_png = k < n_pngs
        inlier_masks_per_thr = {} if save_png else None
        stats_per_thr = {}
        predicted_mask = None
        for thr in thresholds:
            inlier_mask, pred_mask, stats = compute_gt_inliers(
                gt_seg, depth_np, K_np, c2w_np,
                distance_threshold=thr,
                inlier_ratio_gate=args.inlier_ratio_gate,
                ransac_iterations=args.ransac_iters,
            )
            stats_per_thr[thr] = stats
            predicted_mask = pred_mask
            if save_png:
                inlier_masks_per_thr[thr] = inlier_mask

            if verbose:
                tqdm.write(f"  @ {thr*1000:6.2f}mm  P={stats['precision']:.4f}  "
                           f"R={stats['recall']:.4f}  inliers={stats['num_inliers']:>7d}  "
                           f"valid_planes={stats['num_valid_planes']}  "
                           f"pts={stats['total_predicted_points']}")

        # Full metric set (clustering + binary planarity will be 1.0 since labels==gt)
        metrics, _ = evaluate_single_frame(
            scene_id=scene_id,
            frame_idx=frame_idx,
            depth_np=depth_np,
            gt_seg_np=gt_seg,
            K_np=K_np,
            c2w_np=c2w_np,
            labels=gt_seg,
            thresholds=thresholds,
            compute_plane_metrics_flag=True,
            ransac_iterations=args.ransac_iters,
            inlier_ratio_gate=args.inlier_ratio_gate,
        )

        # Enrich with extra per-frame info + per-threshold inlier counts
        metrics["sample_idx"] = idx
        metrics["num_planes"] = int(gt_seg.max())
        metrics["valid_depth_frac"] = float((depth_np > 0).mean())
        metrics["H"] = int(depth_np.shape[0])
        metrics["W"] = int(depth_np.shape[1])
        for thr, s in stats_per_thr.items():
            tag = f"{thr*100:.1f}cm"
            metrics[f"num_inliers@{tag}"] = s["num_inliers"]
            metrics[f"num_valid_planes@{tag}"] = s["num_valid_planes"]
            metrics[f"total_predicted_pts@{tag}"] = s["total_predicted_points"]
        all_rows.append(metrics)

        if save_png:
            png_path = out_dir / f"frame_{k:03d}_idx{idx}.png"
            render_frame_png(
                png_path, rgb_np, depth_np, gt_seg, inlier_masks_per_thr, predicted_mask,
                frame_idx=frame_idx, stats_per_thr=stats_per_thr, scene_id=scene_id,
            )
            if verbose:
                tqdm.write(f"  saved {png_path.name}")

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    front = ["sample_idx", "scene_id", "frame_idx", "num_planes", "valid_depth_frac", "H", "W"]
    metric_cols = [c for c in df.columns
                   if c.startswith(("prec@", "rec@", "bp_", "sc", "ri", "voi",
                                    "num_inliers@", "num_valid_planes@", "total_predicted_pts@"))]
    ordered = [c for c in front if c in df.columns] + [c for c in metric_cols if c in df.columns]
    leftover = [c for c in df.columns if c not in ordered]
    df_view = df[ordered + leftover]
    df_view.to_csv(out_dir / "per_frame_metrics.csv", index=False)

    numeric = df_view.select_dtypes(include=[np.number])
    summary = pd.concat([numeric.mean().rename("mean"),
                         numeric.std().rename("std"),
                         numeric.median().rename("median"),
                         numeric.min().rename("min"),
                         numeric.max().rename("max")], axis=1).T
    summary.to_csv(out_dir / "summary.csv")

    txt_lines = []
    txt_lines.append(f"7-Scenes GT inlier evaluation -- {timestamp}")
    txt_lines.append(f"frames: {len(sample_indices)}, thresholds: {thresholds} m, "
                     f"gate: {args.inlier_ratio_gate}, ransac_iters: {args.ransac_iters}")
    txt_lines.append("=" * 80)
    if len(sample_indices) <= 50:
        txt_lines.append("PER-FRAME METRICS")
        txt_lines.append("=" * 80)
        txt_lines.append(df_view.to_string(index=False))
        txt_lines.append("")
    txt_lines.append("=" * 80)
    txt_lines.append("SUMMARY (mean / std / median / min / max across frames)")
    txt_lines.append("=" * 80)
    txt_lines.append(summary.to_string())
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(txt_lines))

    if skipped:
        with open(out_dir / "skipped_frames.txt", "w") as f:
            f.write("ds_idx\tnpz_name\treason\n")
            for idx, name, reason in skipped:
                f.write(f"{idx}\t{name}\t{reason}\n")

    print("\n" + "=" * 80, flush=True)
    print(f"Evaluated {len(all_rows)} frames (requested {len(sample_indices)}, "
          f"skipped {len(skipped)} corrupt), saved {n_pngs} PNGs", flush=True)
    print(f"per_frame_metrics.csv  ({len(df_view)} rows, {len(df_view.columns)} cols)", flush=True)
    print(f"Output dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
