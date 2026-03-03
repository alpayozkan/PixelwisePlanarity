#!/usr/bin/env python3
"""
Create unified evaluation tables across all datasets with consistent method names.

Reads per-method results_dataset.csv from each dataset's eval directory and combines
them into a single CSV per dataset with F1 columns after each P@/R@ pair.

Output: /cluster/scratch/aoezkan/planeseg/eval/table_{dataset}.csv

Usage:
    python create_unified_tables.py                          # all datasets
    python create_unified_tables.py --datasets scannetpp hypersim
"""

import csv
import os
import argparse

# Method mapping: display_name -> {dataset: eval_dir_name}
METHODS = [
    {
        "display": "GT (upper bound)",
        "dirs": {
            "scannetpp": "gt_v6",
            "hypersim": "gt_v3",
            "synthia": "gt_v1",
            "vkitti2": "gt_v1",
        },
    },
    {
        "display": "MoGe (Ours)",
        "dirs": {
            "scannetpp": "moge_hires_4ds_ep2_v6",
            "hypersim": "moge_hires_4ds_ep2_v1",
            "synthia": "moge_hires_4ds_ep2_v1",
            "vkitti2": "moge_hires_4ds_ep2_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned)",
        "dirs": {
            "scannetpp": "zeroplane_all_h5_dust3r_v6",
            "hypersim": "zeroplane_all_h5_dust3r_v1",
            "synthia": "zeroplane_all_h5_dust3r_v1",
            "vkitti2": "zeroplane_all_h5_dust3r_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned, dinov2, 60k)",
        "dirs": {
            "scannetpp": "zeroplane_all_h5_dinov2_moge_60k_v6",
            "hypersim": "zeroplane_all_h5_dinov2_moge_60k_v1",
            "synthia": "zeroplane_all_h5_dinov2_moge_60k_v1",
            "vkitti2": "zeroplane_all_h5_dinov2_moge_60k_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned, dinov2, 165k)",
        "dirs": {
            "scannetpp": "zeroplane_all_h5_dinov2_moge_165k_v6",
            "hypersim": "zeroplane_all_h5_dinov2_moge_165k_v1",
            "synthia": "zeroplane_all_h5_dinov2_moge_165k_v1",
            "vkitti2": "zeroplane_all_h5_dinov2_moge_165k_v1",
        },
    },
    {
        "display": "ZeroPlane (released)",
        "dirs": {
            "scannetpp": "zeroplane_default_dust3r_released_v6",
            "hypersim": "zeroplane_default_dust3r_released_v1",
            "synthia": "zeroplane_default_dust3r_released_v1",
            "vkitti2": "zeroplane_default_dust3r_released_v1",
        },
    },
    {
        "display": "MoGe (Ours Indoors)",
        "dirs": {
            "scannetpp": "moge_hires_ep2_v6",
            "hypersim": "moge_hires_ep2_v1",
            "synthia": "moge_hires_ep2_v1",
            "vkitti2": "moge_hires_ep2_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned Indoors)",
        "dirs": {
            "scannetpp": "zeroplane_mixed_h5_dust3r_75k_v6",
            "hypersim": "zeroplane_mixed_h5_dust3r_75k_v1",
            "synthia": "zeroplane_mixed_h5_dust3r_75k_v1",
            "vkitti2": "zeroplane_mixed_h5_dust3r_75k_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned Indoors, dinov2, 60k)",
        "dirs": {
            "scannetpp": "zeroplane_mixed_h5_dinov2_moge_60k_v6",
            "hypersim": "zeroplane_mixed_h5_dinov2_moge_60k_v1",
            "synthia": "zeroplane_mixed_h5_dinov2_moge_60k_v1",
            "vkitti2": "zeroplane_mixed_h5_dinov2_moge_60k_v1",
        },
    },
    {
        "display": "ZeroPlane (finetuned Indoors, dinov2, 165k)",
        "dirs": {
            "scannetpp": "zeroplane_mixed_h5_dinov2_moge_165k_v6",
            "hypersim": "zeroplane_mixed_h5_dinov2_moge_165k_v1",
            "synthia": "zeroplane_mixed_h5_dinov2_moge_165k_v1",
            "vkitti2": "zeroplane_mixed_h5_dinov2_moge_165k_v1",
        },
    },
    {
        "display": "Pseudo-Planamono",
        "dirs": {
            "scannetpp": "pseudo_planamono_v6",
            "hypersim": "pseudo_planamono_v1",
            "synthia": "pseudo_planamono_v1",
            "vkitti2": "pseudo_planamono_v1",
        },
    },
    {
        "display": "PlaneRCNN",
        "dirs": {
            "scannetpp": "planercnn_v1",
            "hypersim": "planercnn_v1",
            "synthia": "planercnn_v1",
            "vkitti2": "planercnn_v1",
        },
    },
    {
        "display": "Metric3D",
        "dirs": {
            "scannetpp": "metric3d_v1",
            "hypersim": "metric3d_v1",
        },
    },
    {
        "display": "PlaneTR",
        "dirs": {
            "scannetpp": "planeTR_lines_v6",
            "hypersim": "planeTR_lines_v1",
            "synthia": "planeTR_lines_v1",
            "vkitti2": "planeTR_v1",
        },
    },
    {
        "display": "PlanRecTR (lowres)",
        "dirs": {
            "scannetpp": "planrectr_lowres_v1",
            "hypersim": "planrectr_lowres_v1",
            "synthia": "planrectr_lowres_v1",
            "vkitti2": "planrectr_lowres_v1",
        },
    },
    {
        "display": "PlanRecTR (highres)",
        "dirs": {
            "scannetpp": "planrectr_highres_v1",
            "hypersim": "planrectr_highres_v1",
            "synthia": "planrectr_highres_v1",
            "vkitti2": "planrectr_highres_v1",
        },
    },
    {
        "display": "PlanarRecon",
        "dirs": {
            "scannetpp": "planar_recon_v6",
            "hypersim": "planar_recon_v1",
            "synthia": "planar_recon_v1",
            "vkitti2": "planar_recon_v1",
        },
    },
    {
        "display": "MoGe ep3 v5_rel (plan=0.3, norm=5°, match=8, depth_rel=0.025)",
        "dirs": {
            "scannetpp": "moge_hires_ep3_v5_relative_seg_v6",
            "hypersim": "moge_hires_ep3_v5_relative_seg_v1",
            "synthia": "moge_hires_ep3_v5_relative_seg_v1",
            "vkitti2": "moge_hires_ep3_v5_relative_seg_v1",
        },
    },
    {
        "display": "MoGe ep3 v5origparams_rel (plan=0.6, norm=10°, match=24, depth_rel=0.025)",
        "dirs": {
            "scannetpp": "moge_hires_ep3_v5origparams_relative_seg_v6",
            "hypersim": "moge_hires_ep3_v5origparams_relative_seg_v1",
            "synthia": "moge_hires_ep3_v5origparams_relative_seg_v1",
            "vkitti2": "moge_hires_ep3_v5origparams_relative_seg_v1",
        },
    },
    {
        "display": "MoGe 4ds ep2 v5_rel (plan=0.3, norm=5°, match=8, depth_rel=0.025)",
        "dirs": {
            "scannetpp": "moge_hires_4ds_ep2_v5_relative_seg_v6",
            "hypersim": "moge_hires_4ds_ep2_v5_relative_seg_v1",
            "synthia": "moge_hires_4ds_ep2_v5_relative_seg_v1",
            "vkitti2": "moge_hires_4ds_ep2_v5_relative_seg_v1",
        },
    },
    {
        "display": "MoGe 4ds ep2 v5origparams_rel (plan=0.6, norm=10°, match=24, depth_rel=0.025)",
        "dirs": {
            "scannetpp": "moge_hires_4ds_ep2_v5origparams_relative_seg_v6",
            "hypersim": "moge_hires_4ds_ep2_v5origparams_relative_seg_v1",
            "synthia": "moge_hires_4ds_ep2_v5origparams_relative_seg_v1",
            "vkitti2": "moge_hires_4ds_ep2_v5origparams_relative_seg_v1",
        },
    },
    {
        "display": "DepthAnything (DAV2 normals)",
        "dirs": {
            "scannetpp": "depthanything_dav2_normals_v1",
            "hypersim": "depthanything_dav2_normals_v1",
            "synthia": "depthanything_dav2_normals_v1",
            "vkitti2": "depthanything_dav2_normals_v1",
        },
    },
    {
        "display": "DepthAnything (MoGe normals)",
        "dirs": {
            "scannetpp": "depthanything_moge_normals_v1",
            "hypersim": "depthanything_moge_normals_v1",
            "synthia": "depthanything_moge_normals_v1",
            "vkitti2": "depthanything_moge_normals_v1",
        },
    },
]

DATASET_ROOTS = {
    "scannetpp": "/cluster/scratch/aoezkan/planeseg/scannetpp/eval",
    "hypersim": "/cluster/scratch/aoezkan/planeseg/hypersim/eval",
    "synthia": "/cluster/scratch/aoezkan/planeseg/synthia/eval",
    "vkitti2": "/cluster/scratch/aoezkan/planeseg/vkitti2/eval",
}

# Thresholds per dataset
THRESHOLDS = {
    "scannetpp": ["0.1", "0.5", "1.0"],
    "hypersim": ["0.1", "0.5", "1.0"],
    "synthia": ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0"],
    "vkitti2": ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0"],
}

OUTPUT_DIR = "/cluster/scratch/aoezkan/planeseg/eval"


def read_results(eval_root, dir_name):
    """Read results_dataset.csv from an eval directory."""
    path = os.path.join(eval_root, dir_name, "results_dataset.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found!")
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        print(f"  WARNING: {path} is empty!")
        return None
    return rows[0]  # Single row for dataset-level results


def fmt(val, decimals=3):
    """Format a numeric value to fixed decimal places. Pass through empty strings."""
    if val == "" or val is None:
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return val


def compute_f1(p, r):
    """Compute F1 from precision and recall."""
    p = float(p) if p else 0.0
    r = float(r) if r else 0.0
    if (p + r) > 0:
        return 2 * p * r / (p + r)
    return 0.0


def build_table(dataset):
    """Build unified table for a dataset."""
    eval_root = DATASET_ROOTS[dataset]
    thresholds = THRESHOLDS[dataset]

    # Build header
    header = ["Method", "num_scenes", "num_frames"]
    header.extend(["RI", "VOI", "SC"])
    for t in thresholds:
        header.extend([f"P@{t}cm", f"R@{t}cm", f"F1@{t}cm"])
    header.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])

    rows = []
    for method in METHODS:
        display = method["display"]
        dir_name = method["dirs"].get(dataset)
        if not dir_name:
            print(f"  Skipping {display} for {dataset} (no dir)")
            continue

        data = read_results(eval_root, dir_name)
        if data is None:
            print(f"  Skipping {display} for {dataset} (no data)")
            continue

        row = [display]
        row.append(data.get("num_scenes", ""))
        row.append(data.get("num_frames_total", ""))
        row.append(fmt(data.get("rand_index_mean", "")))
        row.append(fmt(data.get("voi_mean", "")))
        row.append(fmt(data.get("sc_mean", "")))

        for t in thresholds:
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"table_{dataset}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"  Written: {out_path} ({len(rows)} methods)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create unified eval tables across datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scannetpp", "hypersim", "synthia", "vkitti2"],
        choices=["scannetpp", "hypersim", "synthia", "vkitti2"],
        help="Datasets to build tables for (default: all)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        build_table(dataset)

    print("\nDone!")


if __name__ == "__main__":
    main()
