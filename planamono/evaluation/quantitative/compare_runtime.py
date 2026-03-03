#!/usr/bin/env python3
"""
Compare runtime benchmarks across ZeroPlane and MoGe methods.

Reads runtime_per_frame.csv from each method directory and produces
a unified comparison table (printed + saved as CSV).

Usage:
  python compare_runtime.py
  python compare_runtime.py --output runtime_comparison.csv
"""

import os
import argparse
import pandas as pd
import numpy as np

RUNTIME_ROOT = "/cluster/scratch/aoezkan/planeseg/scannetpp/runtime"

# Method definitions: name → (directory, column mappings)
# "compute" = total minus load_image (pure compute, no disk I/O)
METHODS = {
    "ZeroPlane 75k": {
        "dir": "zeroplane_mixed_h5_dust3r_75k_v1",
        "model_cols": ["model_forward"],
        "seg_cols": [],  # segmentation is inside model_forward
        "prepost_cols": ["preprocess", "postprocess"],
    },
    "ZeroPlane 145k": {
        "dir": "zeroplane_mixed_h5_dust3r_145k_v1",
        "model_cols": ["model_forward"],
        "seg_cols": [],
        "prepost_cols": ["preprocess", "postprocess"],
    },
    "MoGe v5 seg": {
        "dir": "moge_mixed_bce_476644_ep6_v6",
        "model_cols": ["moge_inference"],
        "seg_cols": ["seg_total"],
        "prepost_cols": ["resize_outputs", "threshold", "label_remap"],
    },
    "MoGe v6 seg": {
        "dir": "moge_mixed_bce_476644_ep6_v6seg_v6",
        "model_cols": ["moge_inference"],
        "seg_cols": ["seg_total"],
        "prepost_cols": ["resize_outputs", "threshold", "label_remap"],
    },
}


def load_per_frame(method_dir):
    """Load runtime_per_frame.csv, return DataFrame or None."""
    csv_path = os.path.join(RUNTIME_ROOT, method_dir, "runtime_per_frame.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def compute_stats(series):
    """Return mean, std, median in ms."""
    return {
        "mean_ms": series.mean() * 1000,
        "std_ms": series.std() * 1000,
        "median_ms": series.median() * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare runtime benchmarks")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: RUNTIME_ROOT/runtime_comparison.csv)")
    args = parser.parse_args()

    rows = []

    for method_name, cfg in METHODS.items():
        df = load_per_frame(cfg["dir"])
        if df is None:
            print(f"[SKIP] {method_name}: no data in {cfg['dir']}")
            continue

        row = {"method": method_name, "num_frames": len(df)}

        # Model inference (GPU)
        model_time = df[cfg["model_cols"]].sum(axis=1)
        stats = compute_stats(model_time)
        row["model_mean_ms"] = stats["mean_ms"]
        row["model_median_ms"] = stats["median_ms"]

        # Segmentation (separate step, if any)
        if cfg["seg_cols"]:
            seg_time = df[cfg["seg_cols"]].sum(axis=1)
            stats = compute_stats(seg_time)
            row["seg_mean_ms"] = stats["mean_ms"]
            row["seg_median_ms"] = stats["median_ms"]
        else:
            row["seg_mean_ms"] = 0.0
            row["seg_median_ms"] = 0.0

        # Pre/post processing
        prepost_time = df[cfg["prepost_cols"]].sum(axis=1)
        stats = compute_stats(prepost_time)
        row["prepost_mean_ms"] = stats["mean_ms"]
        row["prepost_median_ms"] = stats["median_ms"]

        # Compute total (everything except load_image)
        compute_time = model_time + prepost_time
        if cfg["seg_cols"]:
            compute_time = compute_time + df[cfg["seg_cols"]].sum(axis=1)
        stats = compute_stats(compute_time)
        row["compute_mean_ms"] = stats["mean_ms"]
        row["compute_median_ms"] = stats["median_ms"]

        # FPS (based on compute time, excluding I/O)
        row["fps_mean"] = 1000.0 / row["compute_mean_ms"] if row["compute_mean_ms"] > 0 else 0
        row["fps_median"] = 1000.0 / row["compute_median_ms"] if row["compute_median_ms"] > 0 else 0

        # Total including I/O
        stats = compute_stats(df["total"])
        row["total_mean_ms"] = stats["mean_ms"]
        row["total_median_ms"] = stats["median_ms"]

        rows.append(row)

    if not rows:
        print("No benchmark data found.")
        return

    df_cmp = pd.DataFrame(rows)

    # Print table
    print("\n" + "=" * 100)
    print("RUNTIME COMPARISON (ScanNet++ test set)")
    print("=" * 100)
    print(f"{'Method':<22s} {'Frames':>6s} "
          f"{'Model':>9s} {'Seg':>9s} {'Pre/Post':>9s} "
          f"{'Compute':>9s} {'Total':>9s} {'FPS':>7s}")
    print(f"{'':22s} {'':>6s} "
          f"{'(ms)':>9s} {'(ms)':>9s} {'(ms)':>9s} "
          f"{'(ms)':>9s} {'(ms)':>9s} {'':>7s}")
    print("-" * 100)
    for _, r in df_cmp.iterrows():
        seg_str = f"{r['seg_mean_ms']:.1f}" if r['seg_mean_ms'] > 0 else "—"
        print(f"{r['method']:<22s} {int(r['num_frames']):>6d} "
              f"{r['model_mean_ms']:>9.1f} {seg_str:>9s} {r['prepost_mean_ms']:>9.1f} "
              f"{r['compute_mean_ms']:>9.1f} {r['total_mean_ms']:>9.1f} {r['fps_mean']:>7.1f}")
    print("=" * 100)
    print("Compute = Model + Seg + Pre/Post (excludes image loading I/O)")
    print("Total   = Compute + load_image")

    # Save CSV to eval dir (alongside other table_*.csv files)
    eval_dir = "/cluster/scratch/aoezkan/planeseg/scannetpp/eval"
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(eval_dir, "table_runtime_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_cmp.to_csv(out_path, index=False)
    print(f"\n[CSV] Saved to: {out_path}")


if __name__ == "__main__":
    main()
