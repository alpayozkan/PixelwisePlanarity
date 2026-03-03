"""
Aggregate evaluation results from multiple experiments into a single CSV table.

Supports two directory structures:
  1. Flat (baselines):  EVAL_ROOT/{exp_name}/results_dataset.csv
  2. Nested (ablations): EVAL_ROOT/zp_ablations/{exp}/{model}/{thresh}/results_dataset.csv

Experiments are specified as CLI arguments. Each argument is resolved as:
  - A flat experiment name:    "gt_v5"  ->  EVAL_ROOT/gt_v5/results_dataset.csv
  - A nested ablation path:    "mixed_dust3r/model_0074999/thresh_default"
                                ->  EVAL_ROOT/zp_ablations/mixed_dust3r/model_0074999/thresh_default/results_dataset.csv
  - A partial ablation path:   "mixed_dust3r/model_0074999"
                                ->  all thresh_* under that model
  - An experiment name with nested results: "default_dust3r_released"
                                ->  all model_*/thresh_* under it

Usage:
    # Combine specific experiments
    python aggregate_results.py gt_v5 moge_mixed_bce_v5 zeroplane_mixed_v5 \
        mixed_dust3r/model_0074999 default_dust3r_released/model_0000000 \
        -o table_combined_baselines_v6.csv

    # Include all thresholds from an experiment
    python aggregate_results.py gt_v5 moge_mixed_bce_v5 default_dust3r_released \
        -o table_combined.csv

    # List all available experiments
    python aggregate_results.py --list
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Paths
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval")
ZP_ABLATIONS_ROOT = EVAL_ROOT / "zp_ablations"

THRESHOLDS = (0.001, 0.005, 0.01)


def find_results_dataset(name: str) -> List[Tuple[str, Path]]:
    """
    Resolve an experiment name to (display_name, results_dataset.csv) pairs.

    Returns a list because partial paths can expand to multiple results.
    """
    results = []

    # 1. Try flat: EVAL_ROOT/{name}/results_dataset.csv
    flat_path = EVAL_ROOT / name / "results_dataset.csv"
    if flat_path.exists():
        results.append((name, flat_path))
        return results

    # 2. Try full nested path: zp_ablations/{name}/results_dataset.csv
    nested_path = ZP_ABLATIONS_ROOT / name / "results_dataset.csv"
    if nested_path.exists():
        results.append((name, nested_path))
        return results

    # 3. Try partial: zp_ablations/{name}/thresh_*/results_dataset.csv  (exp/model given)
    partial_dir = ZP_ABLATIONS_ROOT / name
    if partial_dir.exists():
        for thresh_dir in sorted(partial_dir.iterdir()):
            if thresh_dir.is_dir() and thresh_dir.name.startswith("thresh_"):
                csv = thresh_dir / "results_dataset.csv"
                if csv.exists():
                    rel = f"{name}/{thresh_dir.name}"
                    results.append((rel, csv))
        if results:
            return results

    # 4. Try experiment-level: zp_ablations/{name}/model_*/thresh_*/results_dataset.csv
    exp_dir = ZP_ABLATIONS_ROOT / name
    if exp_dir.exists():
        for model_dir in sorted(exp_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name.startswith("model_"):
                for thresh_dir in sorted(model_dir.iterdir()):
                    if thresh_dir.is_dir() and thresh_dir.name.startswith("thresh_"):
                        csv = thresh_dir / "results_dataset.csv"
                        if csv.exists():
                            rel = f"{name}/{model_dir.name}/{thresh_dir.name}"
                            results.append((rel, csv))

    return results


def load_row(display_name: str, csv_path: Path) -> Dict:
    """Load a results_dataset.csv and return a row dict."""
    df_row = pd.read_csv(csv_path).iloc[0]

    row = {
        "Method": display_name,
        "num_scenes": int(df_row["num_scenes"]),
        "num_frames": int(df_row["num_frames_total"]),
    }

    for col, display in [("rand_index_mean", "RI"),
                          ("voi_mean", "VOI"),
                          ("sc_mean", "SC")]:
        if col in df_row.index:
            row[display] = df_row[col]

    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        prec_col = f"prec@{thresh_str}_mean"
        rec_col = f"rec@{thresh_str}_mean"
        if prec_col in df_row.index:
            row[f"P@{thresh_str}"] = df_row[prec_col]
        if rec_col in df_row.index:
            row[f"R@{thresh_str}"] = df_row[rec_col]

    for col, display in [("bp_accuracy_mean", "bp_accuracy"),
                          ("bp_precision_mean", "bp_precision"),
                          ("bp_recall_mean", "bp_recall"),
                          ("bp_f1_mean", "bp_f1"),
                          ("bp_iou_mean", "bp_iou")]:
        if col in df_row.index:
            row[display] = df_row[col]

    return row


def list_available():
    """List all available experiments."""
    print("=== Flat experiments (EVAL_ROOT/{name}/) ===")
    for d in sorted(EVAL_ROOT.iterdir()):
        if d.is_dir() and (d / "results_dataset.csv").exists():
            print(f"  {d.name}")

    print("\n=== Ablation experiments (zp_ablations/{exp}/{model}/{thresh}/) ===")
    if ZP_ABLATIONS_ROOT.exists():
        for exp_dir in sorted(ZP_ABLATIONS_ROOT.iterdir()):
            if not exp_dir.is_dir():
                continue
            for model_dir in sorted(exp_dir.iterdir()):
                if not model_dir.is_dir() or not model_dir.name.startswith("model_"):
                    continue
                for thresh_dir in sorted(model_dir.iterdir()):
                    if thresh_dir.is_dir() and (thresh_dir / "results_dataset.csv").exists():
                        print(f"  {exp_dir.name}/{model_dir.name}/{thresh_dir.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results into a single CSV table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python aggregate_results.py --list
  python aggregate_results.py gt_v5 moge_mixed_bce_v5 mixed_dust3r/model_0074999 -o table_v6.csv
  python aggregate_results.py gt_v5 default_dust3r_released -o table.csv
""",
    )
    parser.add_argument("experiments", nargs="*",
                        help="Experiment names or paths to include")
    parser.add_argument("-o", "--output", type=str, default="table_combined.csv",
                        help="Output CSV filename (saved in EVAL_ROOT)")
    parser.add_argument("--list", action="store_true",
                        help="List all available experiments and exit")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.experiments:
        parser.error("Specify experiments to aggregate, or use --list")

    rows = []
    for exp_name in args.experiments:
        found = find_results_dataset(exp_name)
        if not found:
            print(f"[WARN] Not found: {exp_name}")
            continue
        for display_name, csv_path in found:
            try:
                row = load_row(display_name, csv_path)
                rows.append(row)
                print(f"[OK] {display_name} ({row['num_scenes']} scenes, {row['num_frames']} frames)")
            except Exception as e:
                print(f"[ERROR] {display_name}: {e}")

    if not rows:
        print("[ERROR] No results loaded")
        return

    df = pd.DataFrame(rows)

    # Column order
    metric_cols = ["RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        metric_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}"])
    metric_cols.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])
    all_cols = ["Method", "num_scenes", "num_frames"] + metric_cols
    all_cols = [c for c in all_cols if c in df.columns]
    df = df[all_cols]

    # Save
    out_path = EVAL_ROOT / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    try:
        xlsx_path = out_path.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Saved: {xlsx_path}")
    except ImportError:
        pass

    # Print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(f"\n{'='*120}")
    print(df.to_string(index=False))
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
