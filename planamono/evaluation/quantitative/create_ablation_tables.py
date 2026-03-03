#!/usr/bin/env python3
"""
Create ablation tables for v5 segmentation variant experiments.

Generates table CSVs in the same format as create_unified_tables.py,
organized under /cluster/scratch/aoezkan/planeseg/eval/ablations/.

Usage:
    python create_ablation_tables.py              # both ablations
    python create_ablation_tables.py --ablations v5_seg_ablation
"""

import csv
import os
import argparse

# Reuse helpers from create_unified_tables
from planamono.evaluation.quantitative.create_unified_tables import (
    read_results, fmt, compute_f1,
)

EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/scannetpp/eval"
OUTPUT_BASE = "/cluster/scratch/aoezkan/planeseg/eval/ablations"
THRESHOLDS = ["0.1", "0.5", "1.0"]

# ---- Ablation definitions ----
ABLATIONS = {
    "v5_seg_ablation": {
        "description": "v5 seg variants, notebook params (plan=0.3, norm=5deg, match=8)",
        "methods": [
            {"display": "v5 (Sobel+abs)", "dir": "moge_hires_ep3_v5seg_v6"},
            {"display": "v5_rel (Sobel+rel)", "dir": "moge_hires_ep3_v5_relative_seg_v6"},
            {"display": "v5_nosob (dot+abs)", "dir": "moge_hires_ep3_v5_no_sobel_seg_v6"},
            {"display": "v5_dot_rel (dot+rel)", "dir": "moge_hires_ep3_v5_dotprod_relative_seg_v6"},
        ],
    },
    "v5_seg_ablation_origparams": {
        "description": "v5 seg variants, original v5 params (plan=0.6, norm=10deg, match=24)",
        "methods": [
            {"display": "v5orig (Sobel+abs)", "dir": "moge_hires_ep3_v5origparams_seg_v6"},
            {"display": "v5orig_rel (Sobel+rel)", "dir": "moge_hires_ep3_v5origparams_relative_seg_v6"},
            {"display": "v5orig_nosob (dot+abs)", "dir": "moge_hires_ep3_v5origparams_no_sobel_seg_v6"},
            {"display": "v5orig_dot_rel (dot+rel)", "dir": "moge_hires_ep3_v5origparams_dotprod_relative_seg_v6"},
        ],
    },
}


def build_ablation_table(ablation_name, ablation_cfg):
    """Build a single ablation table CSV."""
    methods = ablation_cfg["methods"]

    # Build header (same format as create_unified_tables)
    header = ["Method", "num_scenes", "num_frames"]
    header.extend(["RI", "VOI", "SC"])
    for t in THRESHOLDS:
        header.extend([f"P@{t}cm", f"R@{t}cm", f"F1@{t}cm"])
    header.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])

    rows = []
    for method in methods:
        display = method["display"]
        dir_name = method["dir"]

        data = read_results(EVAL_ROOT, dir_name)
        if data is None:
            print(f"  Skipping {display} (no data)")
            continue

        row = [display]
        row.append(data.get("num_scenes", ""))
        row.append(data.get("num_frames_total", ""))
        row.append(fmt(data.get("rand_index_mean", "")))
        row.append(fmt(data.get("voi_mean", "")))
        row.append(fmt(data.get("sc_mean", "")))

        for t in THRESHOLDS:
            p = data.get(f"prec@{t}cm_mean", "")
            r = data.get(f"rec@{t}cm_mean", "")
            f1 = compute_f1(p, r)
            row.extend([fmt(p), fmt(r), fmt(f1)])

        row.append(fmt(data.get("bp_accuracy_mean", "")))
        row.append(fmt(data.get("bp_precision_mean", "")))
        row.append(fmt(data.get("bp_recall_mean", "")))
        row.append(fmt(data.get("bp_f1_mean", "")))
        row.append(fmt(data.get("bp_iou_mean", "")))

        rows.append(row)
        print(f"  OK: {display} ({dir_name})")

    # Write CSV
    out_dir = os.path.join(OUTPUT_BASE, ablation_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table_scannetpp.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"  Written: {out_path} ({len(rows)} methods)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create v5 segmentation ablation tables")
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(ABLATIONS.keys()),
        choices=list(ABLATIONS.keys()),
        help="Which ablations to generate (default: all)",
    )
    args = parser.parse_args()

    for abl_name in args.ablations:
        abl_cfg = ABLATIONS[abl_name]
        print(f"\n=== {abl_name} ===")
        print(f"  {abl_cfg['description']}")
        build_ablation_table(abl_name, abl_cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()
