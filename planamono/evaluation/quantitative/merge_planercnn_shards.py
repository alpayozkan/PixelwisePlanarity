#!/usr/bin/env python3
"""Merge PlaneRCNN evaluation shards into final planercnn_v1/ directory."""

import argparse
from pathlib import Path
import pandas as pd


def merge_dataset(eval_root: Path):
    """Merge all planercnn_v1_shard_* dirs into planercnn_v1/."""
    shard_dirs = sorted(eval_root.glob("planercnn_v1_shard_*"))
    print(f"Found {len(shard_dirs)} shard directories in {eval_root}")

    all_rows = []
    for sd in shard_dirs:
        csv_path = sd / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_rows.append(df)
            print(f"  {sd.name}: {len(df)} frames")
        else:
            print(f"  {sd.name}: MISSING results.csv")

    if not all_rows:
        print("ERROR: no shard results found")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    out_dir = eval_root / "planercnn_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame results
    merged.to_csv(out_dir / "results.csv", index=False)

    # Per-scene aggregation
    scene_agg = []
    for scene_id, grp in merged.groupby("scene_id"):
        row = {"scene_id": scene_id, "num_frames": len(grp)}
        for col in grp.columns:
            if col in ("scene_id", "frame_idx"):
                continue
            try:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_std"] = grp[col].std()
            except Exception:
                pass
        scene_agg.append(row)
    scene_df = pd.DataFrame(scene_agg)
    scene_df.to_csv(out_dir / "results_per_scene.csv", index=False)

    # Dataset-level aggregation
    ds_row = {"num_scenes": len(scene_df), "num_frames_total": len(merged)}
    for col in merged.columns:
        if col in ("scene_id", "frame_idx"):
            continue
        try:
            ds_row[f"{col}_mean"] = merged[col].mean()
            ds_row[f"{col}_std"] = merged[col].std()
        except Exception:
            pass
    pd.DataFrame([ds_row]).to_csv(out_dir / "results_dataset.csv", index=False)

    print(f"Merged: {len(merged)} frames from {len(scene_df)} scenes -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["scannetpp", "hypersim"],
        choices=["scannetpp", "hypersim", "synthia", "vkitti2", "pd"],
    )
    args = parser.parse_args()

    roots = {
        "scannetpp": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval"),
        "hypersim": Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval"),
        "synthia": Path("/cluster/scratch/aoezkan/planeseg/synthia/eval"),
        "vkitti2": Path("/cluster/scratch/aoezkan/planeseg/vkitti2/eval"),
        "pd": Path("/cluster/scratch/aoezkan/planeseg/pd/eval"),
    }

    for ds in args.datasets:
        print(f"\n=== {ds.upper()} ===")
        merge_dataset(roots[ds])


if __name__ == "__main__":
    main()
