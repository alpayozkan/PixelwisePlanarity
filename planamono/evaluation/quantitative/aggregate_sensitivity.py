#!/usr/bin/env python3
"""
Aggregate sensitivity analysis results into formatted tables.

Reads per-parameter summary CSVs produced by sensitivity_analysis_v5rel.py
and generates:
  1. Per-parameter tables (printed + CSV)
  2. Combined LaTeX table
  3. Best-config summary

Usage:
    python aggregate_sensitivity.py \
        --results_dir /path/to/sensitivity_results

    # Only specific parameters
    python aggregate_sensitivity.py \
        --results_dir /path/to/sensitivity_results \
        --params threshold_planarity normal_threshold_deg
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

PARAM_ORDER = [
    "threshold_planarity",
    "normal_threshold_deg",
    "depth_threshold",
    "neighbor_match_count_thresh",
]

PARAM_LABELS = {
    "threshold_planarity": "Planarity Thr.",
    "normal_threshold_deg": "Normal Thr. (°)",
    "depth_threshold": "Rel. Depth Thr.",
    "neighbor_match_count_thresh": "Min Neighbors",
}

PARAM_FORMAT = {
    "threshold_planarity": ".1f",
    "normal_threshold_deg": ".1f",
    "depth_threshold": ".3f",
    "neighbor_match_count_thresh": "d",
}

BASELINE = {
    "threshold_planarity": 0.3,
    "normal_threshold_deg": 5.0,
    "depth_threshold": 0.025,
    "neighbor_match_count_thresh": 8,
}

# Metrics to show: (column, display_name, higher_is_better, format)
METRIC_COLS = [
    ("sc",          "SC",          True,  ".4f"),
    ("rand_index",  "RI",          True,  ".4f"),
    ("voi",         "VOI",         False, ".4f"),
    ("bp_f1",       "BP-F1",       True,  ".4f"),
    ("prec@0.1cm",  "P@1mm",       True,  ".4f"),
    ("rec@0.1cm",   "R@1mm",       True,  ".4f"),
    ("prec@0.5cm",  "P@5mm",       True,  ".4f"),
    ("rec@0.5cm",   "R@5mm",       True,  ".4f"),
    ("prec@1.0cm",  "P@10mm",      True,  ".4f"),
    ("rec@1.0cm",   "R@10mm",      True,  ".4f"),
]


# ============================================================
# TABLE FORMATTING
# ============================================================

def format_val(val, fmt, is_best=False, is_baseline=False):
    """Format a metric value, optionally bolding the best."""
    s = f"{val:{fmt}}"
    if is_best and is_baseline:
        return f"**{s}**"
    elif is_best:
        return f"*{s}*"
    elif is_baseline:
        return f"[{s}]"
    return s


def print_table(param_name, df):
    """Print a formatted table for one parameter sweep."""
    print(f"\n{'='*100}")
    print(f"  {PARAM_LABELS[param_name]} Sensitivity  (baseline = {BASELINE[param_name]})")
    print(f"{'='*100}")

    # Filter to available metrics
    available = [(col, name, hb, fmt) for col, name, hb, fmt in METRIC_COLS if col in df.columns]

    # Find best per metric
    bests = {}
    for col, _, hb, _ in available:
        bests[col] = df[col].idxmax() if hb else df[col].idxmin()

    # Header
    pfmt = PARAM_FORMAT[param_name]
    hdr = f"{'Value':>10s}"
    for _, name, _, _ in available:
        hdr += f"  {name:>8s}"
    print(hdr)
    print("-" * len(hdr))

    # Rows
    for idx, row in df.iterrows():
        val = row["param_value"]
        is_bl = (abs(float(val) - BASELINE[param_name]) < 1e-9)
        if pfmt == "d":
            line = f"{int(val):>10d}" + ("*" if is_bl else " ")
        else:
            line = f"{val:>10{pfmt}}" + ("*" if is_bl else " ")

        for col, _, hb, fmt in available:
            is_best = (idx == bests[col])
            cell = format_val(row[col], fmt, is_best=is_best, is_baseline=is_bl)
            line += f"  {cell:>8s}"
        print(line)

    print(f"\n  * = baseline,  *x* = best,  [x] = baseline value")


def make_latex_table(param_name, df, caption=None):
    """Generate LaTeX table for one parameter sweep."""
    available = [(col, name, hb, fmt) for col, name, hb, fmt in METRIC_COLS if col in df.columns]

    # Find best per metric
    bests = {}
    for col, _, hb, _ in available:
        bests[col] = df[col].idxmax() if hb else df[col].idxmin()

    pfmt = PARAM_FORMAT[param_name]
    n_cols = 1 + len(available)
    col_spec = "r" + "c" * len(available)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = PARAM_LABELS[param_name]
    for _, name, _, _ in available:
        header += f" & {name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for idx, row in df.iterrows():
        val = row["param_value"]
        is_bl = (abs(float(val) - BASELINE[param_name]) < 1e-9)

        val_str = f"{int(val):d}" if pfmt == "d" else f"{val:{pfmt}}"
        if is_bl:
            val_str = f"\\underline{{{val_str}}}"

        cells = [val_str]
        for col, _, hb, fmt in available:
            is_best = (idx == bests[col])
            s = f"{row[col]:{fmt}}"
            if is_best:
                s = f"\\textbf{{{s}}}"
            cells.append(s)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:sensitivity_{param_name}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def print_best_summary(all_dfs):
    """Print a compact summary of which value is best for each (param, metric)."""
    target_metrics = ["sc", "rand_index", "bp_f1", "prec@0.5cm", "rec@0.5cm", "prec@1.0cm", "rec@1.0cm"]
    higher_better = {"sc": True, "rand_index": True, "voi": False,
                     "bp_f1": True, "prec@0.1cm": True, "rec@0.1cm": True,
                     "prec@0.5cm": True, "rec@0.5cm": True,
                     "prec@1.0cm": True, "rec@1.0cm": True}

    print(f"\n{'='*100}")
    print("  Best Configuration Summary")
    print(f"{'='*100}")
    print(f"{'Parameter':<25s} {'Metric':<12s} {'Baseline':>10s} {'Best':>10s} {'Best@':>10s} {'Delta':>10s}")
    print("-" * 80)

    for param_name, df in all_dfs.items():
        for metric in target_metrics:
            if metric not in df.columns:
                continue
            hb = higher_better[metric]
            best_idx = df[metric].idxmax() if hb else df[metric].idxmin()
            best_val = df.loc[best_idx, metric]
            best_param = df.loc[best_idx, "param_value"]

            bl_idx = df.index[np.argmin(np.abs(
                df["param_value"].values.astype(float) - BASELINE[param_name]
            ))]
            bl_val = df.loc[bl_idx, metric]
            delta = best_val - bl_val

            improved = (delta > 0 and hb) or (delta < 0 and not hb)
            arrow = "+" if improved else ""

            print(f"{PARAM_LABELS[param_name]:<25s} {metric:<12s} "
                  f"{bl_val:>10.4f} {best_val:>10.4f} {str(best_param):>10s} "
                  f"{delta:>+10.4f} {arrow}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sensitivity analysis results into tables",
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root directory of sensitivity_analysis_v5rel.py output")
    parser.add_argument("--params", nargs="*", default=None,
                        help="Parameters to include (default: all available)")
    parser.add_argument("--latex", action="store_true",
                        help="Also generate LaTeX tables")

    args = parser.parse_args()

    # Discover available sweep results
    all_dfs = {}
    params_to_show = args.params or PARAM_ORDER

    for param_name in params_to_show:
        summary_path = os.path.join(args.results_dir, f"sweep_{param_name}", "summary.csv")
        if os.path.isfile(summary_path):
            df = pd.read_csv(summary_path)
            all_dfs[param_name] = df
            print(f"[OK] Loaded {param_name}: {len(df)} values")
        else:
            print(f"[SKIP] {summary_path} not found")

    if not all_dfs:
        print("[ERROR] No sweep results found.")
        sys.exit(1)

    # Print tables
    for param_name, df in all_dfs.items():
        print_table(param_name, df)

    # Best summary
    print_best_summary(all_dfs)

    # LaTeX tables
    if args.latex:
        latex_dir = os.path.join(args.results_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)

        for param_name, df in all_dfs.items():
            caption = (f"Sensitivity to {PARAM_LABELS[param_name].lower()} "
                       f"(baseline = {BASELINE[param_name]})")
            latex = make_latex_table(param_name, df, caption=caption)

            latex_path = os.path.join(latex_dir, f"table_{param_name}.tex")
            with open(latex_path, "w") as f:
                f.write(latex)
            print(f"[LaTeX] Saved {latex_path}")

        # Combined LaTeX file
        combined_path = os.path.join(latex_dir, "sensitivity_tables_all.tex")
        with open(combined_path, "w") as f:
            f.write("% Auto-generated sensitivity analysis tables\n")
            f.write("% Baseline: plan=0.3, norm=5°, match=8, depth_rel=0.025\n\n")
            for param_name, df in all_dfs.items():
                caption = (f"Sensitivity to {PARAM_LABELS[param_name].lower()} "
                           f"(baseline = {BASELINE[param_name]})")
                f.write(make_latex_table(param_name, df, caption=caption))
                f.write("\n\n")
        print(f"[LaTeX] Saved combined: {combined_path}")

    print("\n[DONE] Aggregation complete.")


if __name__ == "__main__":
    main()
