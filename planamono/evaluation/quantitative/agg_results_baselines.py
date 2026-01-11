"""
Aggregate results from all baseline fast evaluation scripts.

This script looks for results in:
/cluster/scratch/aoezkan/planeseg/scannetpp/eval/{exp_name}/

And produces summary tables for precision/recall and segmentation metrics.
"""

import pandas as pd
from pathlib import Path

from planamono.evaluation.quantitative.evaluate_all_baselines import METHODS, THRESHOLDS, EVAL_ROOT

# Root directory where eval results are stored (imported from evaluate_all_baselines)
ROOT = EVAL_ROOT


def find_dataset_csv(folder: Path):
    """Find the results_dataset.csv file in a folder."""
    # First try exact match
    exact = folder / "results_dataset.csv"
    if exact.exists():
        return exact
    # Fall back to glob pattern
    files = list(folder.glob("*_results_dataset.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No results_dataset.csv in {folder}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple dataset CSVs in {folder}: {files}")
    return files[0]


def aggregate_results(root: Path = ROOT, output_dir: Path = None):
    """
    Aggregate results from all methods and save summary tables.

    Args:
        root: Root directory containing method folders
        output_dir: Directory to save output CSVs (defaults to current directory)
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- TABLE 1: Precision / Recall ----------
    rows_pr = []

    for method_key, method_config in METHODS.items():
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        folder = root / exp_name
        if not folder.exists():
            print(f"[WARN] Missing folder {folder}")
            continue

        try:
            csv_path = find_dataset_csv(folder)
            df = pd.read_csv(csv_path).iloc[0]
        except Exception as e:
            print(f"[ERROR] Could not read results for {method_key}: {e}")
            continue

        row = {
            "Method": display_name,
            "num_scenes": int(df["num_scenes"]),
            "num_frames": int(df["num_frames_total"]),
        }

        # Add precision/recall metrics if available
        for thr in THRESHOLDS:
            thresh_str = f"{thr*100:.1f}cm"
            prec_col = f"prec@{thresh_str}_mean"
            rec_col = f"rec@{thresh_str}_mean"
            if prec_col in df.index:
                row[f"prec@{thresh_str}"] = df[prec_col]
            if rec_col in df.index:
                row[f"recall@{thresh_str}"] = df[rec_col]

        rows_pr.append(row)

    if rows_pr:
        df_pr = pd.DataFrame(rows_pr)
        out_path = output_dir / "table_precision_recall_baselines.csv"
        df_pr.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("[WARN] No precision/recall results to aggregate")

    # ---------- TABLE 2: Segmentation ----------
    rows_seg = []

    for method_key, method_config in METHODS.items():
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        folder = root / exp_name
        if not folder.exists():
            continue

        try:
            csv_path = find_dataset_csv(folder)
            df = pd.read_csv(csv_path).iloc[0]
        except Exception as e:
            continue

        row = {
            "Method": display_name,
            "num_scenes": int(df["num_scenes"]),
            "num_frames": int(df["num_frames_total"]),
        }

        # Add segmentation metrics if available
        for col, display in [("rand_index_mean", "Rand Index"),
                              ("voi_mean", "VOI"),
                              ("sc_mean", "SC")]:
            if col in df.index:
                row[display] = df[col]

        rows_seg.append(row)

    if rows_seg:
        df_seg = pd.DataFrame(rows_seg)
        out_path = output_dir / "table_segmentation_baselines.csv"
        df_seg.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("[WARN] No segmentation results to aggregate")

    # ---------- TABLE 3: Combined Summary ----------
    rows_combined = []

    for method_key, method_config in METHODS.items():
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        folder = root / exp_name
        if not folder.exists():
            continue

        try:
            csv_path = find_dataset_csv(folder)
            df = pd.read_csv(csv_path).iloc[0]
        except Exception as e:
            continue

        row = {
            "Method": display_name,
            "num_scenes": int(df["num_scenes"]),
            "num_frames": int(df["num_frames_total"]),
        }

        # Segmentation metrics
        for col, display in [("rand_index_mean", "RI"),
                              ("voi_mean", "VOI"),
                              ("sc_mean", "SC")]:
            if col in df.index:
                row[display] = df[col]

        # Precision/recall at all thresholds
        for thr in THRESHOLDS:
            thresh_str = f"{thr*100:.1f}cm"
            prec_col = f"prec@{thresh_str}_mean"
            rec_col = f"rec@{thresh_str}_mean"
            if prec_col in df.index:
                row[f"P@{thresh_str}"] = df[prec_col]
            if rec_col in df.index:
                row[f"R@{thresh_str}"] = df[rec_col]

        rows_combined.append(row)

    if rows_combined:
        df_combined = pd.DataFrame(rows_combined)

        # Custom row order: GT Seg (upper bound), Ours, ZeroPlane, ablations
        order_priority = {
            "GT Seg (upper bound)": 0,
            "Ours (full)": 1,
            "ZeroPlane": 2,
            "GT Planarity + Our Seg": 3,
            "Our Planarity + GT Seg": 4,
        }
        df_combined["_sort_key"] = df_combined["Method"].map(
            lambda x: order_priority.get(x, 100)
        )
        df_combined = df_combined.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

        # Save CSV
        out_path = output_dir / "table_combined_baselines.csv"
        df_combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        # Save XLSX
        xlsx_path = output_dir / "table_combined_baselines.xlsx"
        df_combined.to_excel(xlsx_path, index=False, sheet_name="Combined Results")
        print(f"Saved: {xlsx_path}")

        # Also print a nice summary
        print("\n" + "=" * 80)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 80)
        print(df_combined.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate baseline evaluation results")
    parser.add_argument("--root", type=str, default=str(ROOT),
                        help="Root directory containing eval results")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output CSVs")
    args = parser.parse_args()

    aggregate_results(Path(args.root), Path(args.output_dir))
