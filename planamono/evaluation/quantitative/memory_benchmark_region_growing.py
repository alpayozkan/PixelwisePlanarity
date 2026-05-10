"""
Memory-consumption ablation for the GPU region-growing segmentation
(``compute_vectorized_planar_segments_v5_relative``) at three resolution
levels: low (H/2, W/2), medium (native MoGe signal H, W), high (2H, 2W).

Mirrors ``runtime_benchmark_region_growing.py`` — same scene/shard flow, same
seg parameters, same resolution sweep — but reports **peak GPU memory** per
call instead of wall-time:

    peak_allocated_mb   = torch.cuda.max_memory_allocated() / 2^20
    peak_reserved_mb    = torch.cuda.max_memory_reserved()  / 2^20

Per timed call we ``empty_cache()`` + ``reset_peak_memory_stats()`` then run
the kernel once. The peak therefore captures GPU copies of the inputs
(planarity_mask, normal, depth), every intermediate tensor (5×5 unfold,
Sobel, propagation buffers, ...), and the output labels — i.e. the realistic
working set you would need to reserve in production.

For CPU runs we fall back to ``tracemalloc`` (peak Python-heap allocation,
which covers torch CPU tensor storage).

Output layout (mirrors the runtime script):
    <eval_root>/<exp>/scannetpp/<scene>/results.csv      (worker)
    <eval_root>/<exp>/scannetpp/aggregate_results.csv    (aggregator)
    <eval_root>/<exp>/scannetpp/aggregate_per_resolution.csv
    <eval_root>/<exp>/summary.csv

Sharding via --shard_id / --num_shards (round-robin over scenes).
Aggregate-only via --aggregate_only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.shared.segmentation.plan2seg import (                                            # noqa: E402
    compute_vectorized_planar_segments_v5_relative,
)


# ---------------------------------------------------------------------------
# Config defaults — matched to evaluate_gt_moge_zeroplane_benchmark.py and
# runtime_benchmark_region_growing.py.
# ---------------------------------------------------------------------------

SEG_THRESHOLD_PLANARITY = 0.3
SEG_NORMAL_THRESHOLD_DEG = 5.0
SEG_DEPTH_THRESHOLD_REL = 0.025
SEG_NEIGHBOR_MATCH_COUNT = 8

DEFAULT_MOGE_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/eval/moge_memory_benchmark"

RESOLUTION_FACTORS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}

_BYTES_PER_MB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Resolution helpers (verbatim from the runtime script)
# ---------------------------------------------------------------------------

def _resize_to(planarity: np.ndarray, normal: np.ndarray, depth: np.ndarray,
               H_out: int, W_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (planarity.shape[0], planarity.shape[1]) == (H_out, W_out):
        return (
            np.ascontiguousarray(planarity.astype(np.float32, copy=False)),
            np.ascontiguousarray(normal.astype(np.float32, copy=False)),
            np.ascontiguousarray(depth.astype(np.float32, copy=False)),
        )
    plan_r = cv2.resize(planarity.astype(np.float32, copy=False), (W_out, H_out),
                        interpolation=cv2.INTER_LINEAR)
    norm_r = cv2.resize(normal.astype(np.float32, copy=False), (W_out, H_out),
                        interpolation=cv2.INTER_LINEAR)
    n = np.linalg.norm(norm_r, axis=-1, keepdims=True)
    norm_r = np.divide(norm_r, np.clip(n, 1e-6, None), where=n > 1e-6).astype(np.float32)
    depth_r = cv2.resize(depth.astype(np.float32, copy=False), (W_out, H_out),
                         interpolation=cv2.INTER_LINEAR)
    return (
        np.ascontiguousarray(plan_r),
        np.ascontiguousarray(norm_r),
        np.ascontiguousarray(depth_r),
    )


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _measure_memory_segmentation(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    device: str,
    n_warmup: int,
    n_repeat: int,
) -> Tuple[float, float, float, float, int]:
    """Return (mean_alloc_mb, std_alloc_mb, mean_resv_mb, std_resv_mb, num_components).

    Per repeat we reset PyTorch's peak memory counters, run the kernel once,
    then read ``max_memory_allocated`` / ``max_memory_reserved``. CPU fallback
    uses ``tracemalloc`` (peak == reserved on that path).
    """
    mask = planarity > SEG_THRESHOLD_PLANARITY
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    normal_thr_rad = float(np.deg2rad(SEG_NORMAL_THRESHOLD_DEG))

    seg_kwargs = dict(
        planarity_mask=mask,
        normal=normal,
        depth=depth,
        normal_threshold_rad=normal_thr_rad,
        depth_threshold=SEG_DEPTH_THRESHOLD_REL,
        neighbor_match_count_thresh=SEG_NEIGHBOR_MATCH_COUNT,
        device=device,
    )

    # Warmup. The first call typically inflates peak (allocator cold-start,
    # autograd graph init, etc.) — we deliberately exclude it from stats.
    last_num = 0
    for _ in range(n_warmup):
        labels, last_num = compute_vectorized_planar_segments_v5_relative(**seg_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        del labels

    peaks_alloc_mb: List[float] = []
    peaks_resv_mb: List[float] = []

    for _ in range(n_repeat):
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            labels, last_num = compute_vectorized_planar_segments_v5_relative(**seg_kwargs)
            torch.cuda.synchronize()

            peak_alloc = float(torch.cuda.max_memory_allocated())
            peak_resv = float(torch.cuda.max_memory_reserved())
            del labels
        else:
            tracemalloc.start()
            labels, last_num = compute_vectorized_planar_segments_v5_relative(**seg_kwargs)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_alloc = float(peak)
            peak_resv = float(peak)
            del labels

        peaks_alloc_mb.append(peak_alloc / _BYTES_PER_MB)
        peaks_resv_mb.append(peak_resv / _BYTES_PER_MB)

    arr_a = np.asarray(peaks_alloc_mb, dtype=np.float64)
    arr_r = np.asarray(peaks_resv_mb, dtype=np.float64)
    return (
        float(arr_a.mean()), float(arr_a.std()),
        float(arr_r.mean()), float(arr_r.std()),
        int(last_num),
    )


# ---------------------------------------------------------------------------
# Per-scene worker
# ---------------------------------------------------------------------------

def _decode_frame_ids(arr) -> List[str]:
    return [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in np.asarray(arr).tolist()
    ]


def _process_scene(
    scene_id: str,
    h5_path: str,
    device: str,
    max_frames: Optional[int],
    n_warmup: int,
    n_repeat: int,
) -> List[Dict]:
    rows: List[Dict] = []
    with h5py.File(h5_path, "r") as f:
        frame_ids = _decode_frame_ids(f["frame_ids"][:])
        N = len(frame_ids)
        idxs: List[int] = list(range(N))
        if max_frames is not None and N > max_frames:
            idxs = np.linspace(0, N - 1, max_frames).astype(int).tolist()

        for i in tqdm(idxs, desc=f"  {scene_id}", unit="frame", leave=False):
            fid = frame_ids[i]
            planarity = f["planarity"][i].astype(np.float32)
            normal = f["normal"][i].astype(np.float32)
            depth = f["depth_metric"][i].astype(np.float32)

            H_med, W_med = planarity.shape

            for res_label, factor in RESOLUTION_FACTORS.items():
                H_out = max(2, int(round(H_med * factor)))
                W_out = max(2, int(round(W_med * factor)))
                plan_r, norm_r, depth_r = _resize_to(
                    planarity, normal, depth, H_out, W_out,
                )
                a_mean, a_std, r_mean, r_std, ncomp = _measure_memory_segmentation(
                    plan_r, norm_r, depth_r,
                    device=device,
                    n_warmup=n_warmup,
                    n_repeat=n_repeat,
                )
                rows.append({
                    "scene_id": scene_id,
                    "frame_id": fid,
                    "resolution": res_label,
                    "height": H_out,
                    "width": W_out,
                    "num_pixels": H_out * W_out,
                    "peak_allocated_mb_mean": a_mean,
                    "peak_allocated_mb_std": a_std,
                    "peak_reserved_mb_mean": r_mean,
                    "peak_reserved_mb_std": r_std,
                    "num_components": ncomp,
                    "device": device,
                    "n_repeat": n_repeat,
                })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read_scene_csvs_under(dataset_dir: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.isdir(dataset_dir):
        return out
    for entry in sorted(os.listdir(dataset_dir)):
        scene_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(scene_dir):
            continue
        results_csv = os.path.join(scene_dir, "results.csv")
        if not os.path.isfile(results_csv):
            continue
        try:
            df = pd.read_csv(results_csv)
            out.extend(df.to_dict(orient="records"))
        except Exception as e:
            print(f"  [warn] failed to read {results_csv}: {e}")
    return out


def _aggregate_dataset(dataset_dir: str) -> None:
    all_rows = _read_scene_csvs_under(dataset_dir)
    if not all_rows:
        print(f"  [agg] no rows under {dataset_dir}")
        return
    df = pd.DataFrame.from_records(all_rows)
    df.to_csv(os.path.join(dataset_dir, "aggregate_results.csv"), index=False)

    grp = df.groupby("resolution")
    per_res = grp.agg(
        height=("height", "first"),
        width=("width", "first"),
        num_pixels=("num_pixels", "first"),
        num_frames=("peak_allocated_mb_mean", "size"),
        peak_allocated_mb_mean=("peak_allocated_mb_mean", "mean"),
        peak_allocated_mb_median=("peak_allocated_mb_mean", "median"),
        peak_allocated_mb_std=("peak_allocated_mb_mean", "std"),
        peak_allocated_mb_p95=("peak_allocated_mb_mean", lambda s: float(np.percentile(s, 95))),
        peak_allocated_mb_min=("peak_allocated_mb_mean", "min"),
        peak_allocated_mb_max=("peak_allocated_mb_mean", "max"),
        peak_reserved_mb_mean=("peak_reserved_mb_mean", "mean"),
        peak_reserved_mb_median=("peak_reserved_mb_mean", "median"),
        peak_reserved_mb_std=("peak_reserved_mb_mean", "std"),
        peak_reserved_mb_p95=("peak_reserved_mb_mean", lambda s: float(np.percentile(s, 95))),
        peak_reserved_mb_min=("peak_reserved_mb_mean", "min"),
        peak_reserved_mb_max=("peak_reserved_mb_mean", "max"),
    ).reset_index()
    res_order = pd.Categorical(per_res["resolution"],
                               categories=["low", "medium", "high"], ordered=True)
    per_res = per_res.assign(_o=res_order).sort_values("_o").drop(columns="_o")
    per_res.to_csv(os.path.join(dataset_dir, "aggregate_per_resolution.csv"), index=False)

    print(f"\n[agg] per-resolution summary ({dataset_dir}):")
    print(per_res.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--moge_signals_root", default=DEFAULT_MOGE_ROOT)
    ap.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--device", default="cuda",
                    help="torch device for region growing (default: cuda).")
    ap.add_argument("--n_warmup", type=int, default=3,
                    help="Number of warmup iterations per (frame, resolution).")
    ap.add_argument("--n_repeat", type=int, default=10,
                    help="Number of timed iterations per (frame, resolution).")
    ap.add_argument("--max_frames_per_scene", type=int, default=None,
                    help="Limit to N evenly-spaced frames per scene (default: all).")
    ap.add_argument("--scene_ids", default=None,
                    help="Comma-separated list or path to a .txt with one scene per line.")
    ap.add_argument("--shard_id", type=int, default=None,
                    help="Shard index in [0, --num_shards).")
    ap.add_argument("--num_shards", type=int, default=None,
                    help="Total number of shards. Each shard processes 1/num_shards of scenes.")
    ap.add_argument("--skip_dataset_aggregates", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")

    args = ap.parse_args()

    out_root = os.path.join(args.eval_root, args.exp)
    dataset_dir = os.path.join(out_root, "scannetpp")
    os.makedirs(dataset_dir, exist_ok=True)

    if args.aggregate_only:
        print(f"[AGG] aggregating from {dataset_dir} ...")
        _aggregate_dataset(dataset_dir)
        print(f"[DONE] aggregate written under {dataset_dir}")
        return

    moge_dataset_root = os.path.join(args.moge_signals_root, "scannetpp")
    if not os.path.isdir(moge_dataset_root):
        ap.error(f"--moge_signals_root/scannetpp not found: {moge_dataset_root}")

    available_scenes = sorted([
        d for d in os.listdir(moge_dataset_root)
        if os.path.isfile(os.path.join(moge_dataset_root, d, "moge_signals.h5"))
    ])

    scene_filter: Optional[set] = None
    if args.scene_ids:
        if os.path.isfile(args.scene_ids):
            with open(args.scene_ids) as f:
                scene_filter = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        else:
            scene_filter = {s.strip() for s in args.scene_ids.split(",") if s.strip()}

    scene_ids = available_scenes
    if scene_filter is not None:
        scene_ids = [s for s in available_scenes if s in scene_filter]

    if args.shard_id is not None and args.num_shards is not None and args.num_shards > 1:
        n = len(scene_ids)
        if n == 0:
            print("[shard] empty scene list, nothing to do")
            return
        base = n // args.num_shards
        rem = n % args.num_shards
        start = args.shard_id * base + min(args.shard_id, rem)
        end = start + base + (1 if args.shard_id < rem else 0)
        before = n
        scene_ids = scene_ids[start:end]
        print(f"[shard] scannetpp: {len(scene_ids)}/{before} scenes "
              f"(shard {args.shard_id}/{args.num_shards}, slice [{start}:{end}])")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    print("================================================================")
    print(f" memory_benchmark_region_growing — exp={args.exp}")
    print("================================================================")
    print(f"  moge root:      {args.moge_signals_root}")
    print(f"  output root:    {out_root}")
    print(f"  device:         {args.device}")
    print(f"  warmup/repeat:  {args.n_warmup} / {args.n_repeat}")
    print(f"  resolutions:    {RESOLUTION_FACTORS}")
    print(f"  scenes:         {len(scene_ids)}")
    print(f"  max_frames/sc:  {args.max_frames_per_scene}")
    print(f"  seg params:     plan>{SEG_THRESHOLD_PLANARITY}, "
          f"normal<{SEG_NORMAL_THRESHOLD_DEG}°, depth_rel<{SEG_DEPTH_THRESHOLD_REL}, "
          f"match≥{SEG_NEIGHBOR_MATCH_COUNT}")
    print("================================================================")

    t0 = time.perf_counter()
    for sid in tqdm(scene_ids, desc="scenes", unit="scene"):
        h5_path = os.path.join(moge_dataset_root, sid, "moge_signals.h5")
        if not os.path.isfile(h5_path):
            tqdm.write(f"  [skip] {sid}: moge_signals.h5 missing")
            continue
        try:
            rows = _process_scene(
                scene_id=sid,
                h5_path=h5_path,
                device=args.device,
                max_frames=args.max_frames_per_scene,
                n_warmup=args.n_warmup,
                n_repeat=args.n_repeat,
            )
        except Exception as e:
            tqdm.write(f"  [error] {sid}: {e}")
            continue

        if not rows:
            continue
        scene_dir = os.path.join(dataset_dir, sid)
        os.makedirs(scene_dir, exist_ok=True)
        pd.DataFrame.from_records(rows).to_csv(
            os.path.join(scene_dir, "results.csv"), index=False,
        )

    wall = time.perf_counter() - t0
    print(f"\n[DONE worker] {len(scene_ids)} scenes in {wall:.1f}s")

    if args.skip_dataset_aggregates:
        return

    print("\n[AGG] aggregating ...")
    _aggregate_dataset(dataset_dir)
    print(f"[DONE] output under {out_root}")


if __name__ == "__main__":
    main()
