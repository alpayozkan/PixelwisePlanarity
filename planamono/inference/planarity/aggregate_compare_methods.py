"""Aggregate per-scene results.csv produced by compare_plane_param_methods.py.

Walks `<output_root>/<scene_id>/results.csv`, concatenates them, and writes
two summary files at the root of `--output_root`:

    aggregate_results.csv   one row per (scene, frame, method) — the union of
                            every per-scene results.csv.
    aggregate_summary.csv   one row per method with mean / std across all
                            (scene, frame) rows for every numeric column,
                            plus n_scenes, n_frames, n_rows totals.

Also prints the totals to stdout so the SLURM log carries the headline numbers.

Example
-------
python aggregate_compare_methods.py \\
    --output_root /cluster/scratch/.../compare_methods_test_ls_svd_ransac \\
    --methods least_squares svd ransac \\
    --scenes /cluster/.../planamono/splits/scannetpp/test.txt
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output_root", required=True,
                   help="Same dir passed to compare_plane_param_methods.py "
                        "(contains <scene_id>/results.csv).")
    p.add_argument("--methods", nargs="+", required=True,
                   help="Method order to use when reindexing the summary CSV.")
    p.add_argument("--scenes", default=None,
                   help="Optional path to scene list (e.g., test.txt) — used "
                        "only for reporting expected scene count in stdout.")
    args = p.parse_args()

    pattern = os.path.join(args.output_root, "*", "results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No results.csv under {args.output_root}", file=sys.stderr)
        return 1

    dfs = []
    skipped = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if not d.empty:
                dfs.append(d)
            else:
                skipped.append(f)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
            skipped.append(f)

    if not dfs:
        print("[ERROR] All results.csv files were empty or unreadable.",
              file=sys.stderr)
        return 1

    df = pd.concat(dfs, ignore_index=True)

    # Per-(scene, frame, method) concat: faithful join of all source CSVs.
    out_results = os.path.join(args.output_root, "aggregate_results.csv")
    df.to_csv(out_results, index=False)

    # Per-method aggregate across every (scene, frame) row.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = df.groupby("method")[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    counts = df.groupby("method").agg(
        n_scenes=("scene_id", "nunique"),
        n_frames=("frame_id", "count"),
        n_rows=("method", "size"),
    )
    agg = pd.concat([counts, agg], axis=1)

    method_order = [m for m in args.methods if m in agg.index]
    extra = [m for m in agg.index if m not in args.methods]
    agg = agg.reindex(method_order + extra).reset_index()

    out_summary = os.path.join(args.output_root, "aggregate_summary.csv")
    agg.to_csv(out_summary, index=False)

    # Totals for the SLURM log.
    expected = None
    if args.scenes is not None and os.path.exists(args.scenes):
        with open(args.scenes) as fh:
            expected = sum(1 for line in fh if line.strip())
    total_scenes = int(df["scene_id"].nunique())
    total_frame_method_rows = int(len(df))
    total_unique_frames = int(df.groupby("scene_id")["frame_id"].nunique().sum())
    methods_seen = sorted(df["method"].unique().tolist())

    print("==============================================")
    print(f"[AGG] sources read:        {len(dfs)} CSVs ({len(skipped)} skipped)")
    print(f"[AGG] scenes covered:      {total_scenes}"
          + (f" / {expected} expected" if expected else ""))
    print(f"[AGG] unique frames:       {total_unique_frames}")
    print(f"[AGG] (scene,frame,method) rows: {total_frame_method_rows}")
    print(f"[AGG] methods:             {methods_seen}")
    print(f"[AGG] wrote per-row:       {out_results}")
    print(f"[AGG] wrote per-method:    {out_summary}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
