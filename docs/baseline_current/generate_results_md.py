"""Generate project_results.md (and optionally PDF/PNG) from evaluation CSVs.

Indoor tables (ScanNet++, Hypersim) use thresholds 0.1cm, 0.5cm, 1.0cm.
Outdoor tables (Synthia, VKITTI2) use thresholds 2.0cm, 5.0cm, 10.0cm.

Usage:
    python generate_results_md.py                           # markdown only
    python generate_results_md.py --pdf --png               # markdown + PDF + PNG
    python generate_results_md.py --output custom_path.md   # custom output path
    python generate_results_md.py --pdf --png --no-md       # PDF + PNG only, skip markdown
    python generate_results_md.py --csv-dir /path/to/csvs   # custom CSV source directory
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

# Default CSV source directory (evaluation results)
DEFAULT_CSV_DIR = "/cluster/scratch/aoezkan/planeseg/eval"

# Dataset definitions
DATASETS = [
    {
        "csv_file": "table_scannetpp.csv",
        "title": "ScanNet++ (indoor real)",
        "thresholds": ["0.1cm", "0.5cm", "1.0cm"],
    },
    {
        "csv_file": "table_hypersim.csv",
        "title": "Hypersim (indoor synthetic)",
        "thresholds": ["0.1cm", "0.5cm", "1.0cm"],
    },
    {
        "csv_file": "table_synthia.csv",
        "title": "Synthia (outdoor synthetic)",
        "thresholds": ["2.0cm", "5.0cm", "10.0cm"],
    },
    {
        "csv_file": "table_vkitti2.csv",
        "title": "VKITTI2 (outdoor synthetic)",
        "thresholds": ["2.0cm", "5.0cm", "10.0cm"],
    },
]

# GT quality comparison table
GT_QUALITY_CSV = "gt_quality_comparison.csv"
GT_QUALITY_TITLE = "Plane Dataset Quality Comparison"
GT_QUALITY_COLS = ["display_name",
                   "P@1mm", "R@1mm", "P@5mm", "R@5mm", "P@10mm", "R@10mm"]
GT_QUALITY_HEADERS = ["Method",
                      "P@1mm", "R@1mm", "F1@1mm", "P@5mm", "R@5mm", "F1@5mm",
                      "P@10mm", "R@10mm", "F1@10mm"]
# Rename display names for clarity
GT_QUALITY_RENAMES = {
    "ScanNet GT": "PlaneRCNN GT (ScanNet)",
}
GT_QUALITY_THRESHOLDS = ["1mm", "5mm", "10mm"]

# Columns always included (before threshold-dependent columns)
BASE_COLS = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
BASE_HEADERS = ["Method", "Scenes", "Frames", "RI", "VOI", "SC"]

# Methods to highlight as "ours" (bold in table figures)
OUR_METHODS = {"MoGe (Ours)", "MoGe (Ours Indoors)"}
OUR_GT_METHODS = {"Our GT (ScanNet++)"}

# Methods excluded from best-value highlighting (kept in table but not considered SOTA)
EXCLUDE_FROM_BEST = {"ZeroPlane (finetuned)", "ZeroPlane (finetuned Indoors)"}

# Methods completely hidden from output
HIDE_METHODS = {"ZeroPlane (finetuned)", "ZeroPlane (finetuned Indoors)"}

# Generalization group: indoor-trained models evaluated on all datasets
GENERALIZATION_KEYWORD = "Indoors"
SEPARATOR_LABEL = "Generalization (indoor-trained)"


def fmt(val, col: str) -> str:
    """Format a cell value based on column name."""
    if col in ("Method", "display_name", "dataset", "split"):
        return str(val)
    if col in ("num_scenes", "num_frames"):
        return str(int(val))
    if pd.isna(val):
        return "-"
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return str(val)


def load_gt_quality(base_dir: Path) -> pd.DataFrame:
    """Load gt_quality_comparison.csv and add F1 columns."""
    csv_path = base_dir / GT_QUALITY_CSV
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Rename display names
    df["display_name"] = df["display_name"].replace(GT_QUALITY_RENAMES)
    # Compute F1 for each threshold
    for t in GT_QUALITY_THRESHOLDS:
        p_col, r_col = f"P@{t}", f"R@{t}"
        if p_col in df.columns and r_col in df.columns:
            p = df[p_col].astype(float)
            r = df[r_col].astype(float)
            denom = p + r
            df[f"F1@{t}"] = (2 * p * r / denom).where(denom > 0, 0.0)

    print(f"[OK] {GT_QUALITY_CSV} → {GT_QUALITY_TITLE} ({len(df)} methods)")
    return df


def build_gt_quality_table(df: pd.DataFrame) -> str:
    """Build markdown table for GT quality comparison."""
    if df.empty:
        return ""

    data_cols = ["display_name"]
    for t in GT_QUALITY_THRESHOLDS:
        data_cols.extend([f"P@{t}", f"R@{t}", f"F1@{t}"])

    missing = [c for c in data_cols if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing}")
        return ""

    lines = []
    lines.append("| " + " | ".join(GT_QUALITY_HEADERS) + " |")
    aligns = ["--------" if h == "Method" else "-------:" for h in GT_QUALITY_HEADERS]
    lines.append("|" + "|".join(aligns) + "|")

    for _, row in df.iterrows():
        cells = [fmt(row[c], c) for c in data_cols]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_gt_quality_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build formatted DataFrame for GT quality figure."""
    if df.empty:
        return pd.DataFrame()

    data_cols = ["display_name"]
    for t in GT_QUALITY_THRESHOLDS:
        data_cols.extend([f"P@{t}", f"R@{t}", f"F1@{t}"])

    missing = [c for c in data_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    display_data = []
    for _, row in df.iterrows():
        display_data.append([fmt(row[c], c) for c in data_cols])

    return pd.DataFrame(display_data, columns=GT_QUALITY_HEADERS)


def find_best_gt_quality(df: pd.DataFrame) -> dict:
    """Find best values in GT quality table (all methods compared, higher is better)."""
    best = {}
    metric_cols = []
    for t in GT_QUALITY_THRESHOLDS:
        metric_cols.extend([f"P@{t}", f"R@{t}", f"F1@{t}"])

    header_map = {}
    data_cols = ["display_name"]
    for t in GT_QUALITY_THRESHOLDS:
        data_cols.extend([f"P@{t}", f"R@{t}", f"F1@{t}"])
    for dc, hdr in zip(data_cols, GT_QUALITY_HEADERS):
        header_map[dc] = hdr

    for col in metric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue
        best_idx = vals.idxmax()
        best[(int(best_idx), header_map[col])] = True

    return best


def is_generalization(method: str) -> bool:
    """Check if a method belongs to the generalization (indoor-trained) group."""
    return GENERALIZATION_KEYWORD in str(method)


def split_main_generalization(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into main methods and generalization (indoor-trained) methods."""
    gen_mask = df["Method"].apply(is_generalization)
    return df[~gen_mask].reset_index(drop=True), df[gen_mask].reset_index(drop=True)


def find_group_best(df_group: pd.DataFrame, all_cols: List[str], all_headers: List[str],
                    row_offset: int = 0) -> dict:
    """Find best (row, header) pairs within a single group, skipping GT and excluded methods."""
    best = {}
    eligible_mask = ~df_group["Method"].str.contains("GT", case=False) & \
                    ~df_group["Method"].isin(EXCLUDE_FROM_BEST)
    eligible_idx = df_group.index[eligible_mask]
    if len(eligible_idx) == 0:
        return best

    for col, header in zip(all_cols, all_headers):
        if col in ("Method", "num_scenes", "num_frames"):
            continue
        if col not in df_group.columns:
            continue
        vals = pd.to_numeric(df_group.loc[eligible_idx, col], errors="coerce")
        if vals.isna().all():
            continue
        if col == "VOI":
            best_local_idx = vals.idxmin()
        else:
            best_local_idx = vals.idxmax()
        best[(int(best_local_idx) + row_offset, header)] = True

    return best


def get_table_cols(thresholds: List[str]) -> Tuple[List[str], List[str]]:
    """Return (data_cols, header_cols) for a given threshold set."""
    threshold_cols = []
    threshold_headers = []
    for t in thresholds:
        for prefix in ["P", "R", "F1"]:
            col = f"{prefix}@{t}"
            threshold_cols.append(col)
            threshold_headers.append(col)
    return BASE_COLS + threshold_cols, BASE_HEADERS + threshold_headers


def build_table(df: pd.DataFrame, thresholds: List[str]) -> str:
    """Build a markdown table with separator between main and generalization groups."""
    all_cols, all_headers = get_table_cols(thresholds)

    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing}")
        return ""

    df_main, df_gen = split_main_generalization(df)

    lines = []
    lines.append("| " + " | ".join(all_headers) + " |")

    aligns = []
    for h in all_headers:
        aligns.append("--------" if h == "Method" else "-------:")
    lines.append("|" + "|".join(aligns) + "|")

    for _, row in df_main.iterrows():
        cells = [fmt(row[c], c) for c in all_cols]
        lines.append("| " + " | ".join(cells) + " |")

    if not df_gen.empty:
        # Separator row
        sep_cells = [f"**{SEPARATOR_LABEL}**"] + [""] * (len(all_headers) - 1)
        lines.append("| " + " | ".join(sep_cells) + " |")
        for _, row in df_gen.iterrows():
            cells = [fmt(row[c], c) for c in all_cols]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# Sentinel value for separator rows in display DataFrames
SEPARATOR_SENTINEL = "__SEPARATOR__"


def build_display_df(df: pd.DataFrame, thresholds: List[str]) -> pd.DataFrame:
    """Build a formatted DataFrame with separator row between groups."""
    all_cols, all_headers = get_table_cols(thresholds)
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df_main, df_gen = split_main_generalization(df)

    display_data = []
    for _, row in df_main.iterrows():
        display_data.append([fmt(row[c], c) for c in all_cols])

    if not df_gen.empty:
        # Separator row
        display_data.append([SEPARATOR_SENTINEL] + [""] * (len(all_headers) - 1))
        for _, row in df_gen.iterrows():
            display_data.append([fmt(row[c], c) for c in all_cols])

    return pd.DataFrame(display_data, columns=all_headers)


def find_best_indices(df: pd.DataFrame, thresholds: List[str]) -> dict:
    """Find best (row, col) pairs separately for main and generalization groups.

    Returns display-row indices accounting for the separator row.
    """
    all_cols, all_headers = get_table_cols(thresholds)
    df_main, df_gen = split_main_generalization(df)

    # Main group: rows 0..len(main)-1 in display
    best = find_group_best(df_main, all_cols, all_headers, row_offset=0)

    if not df_gen.empty:
        # Generalization group: offset by len(main) + 1 (separator row)
        gen_offset = len(df_main) + 1
        best.update(find_group_best(df_gen, all_cols, all_headers, row_offset=gen_offset))

    return best


def render_table_figure(display_df: pd.DataFrame, title: str, best: dict,
                        df_orig: pd.DataFrame, ours_set: set = None):
    """Render a single table as a matplotlib figure. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    if ours_set is None:
        ours_set = OUR_METHODS

    n_rows, n_cols = display_df.shape

    # Column widths: wider for Method, narrow for numeric
    method_width = 2.8
    num_width = 0.72
    col_widths = [method_width] + [num_width] * (n_cols - 1)
    total_width = sum(col_widths)

    row_height = 0.38
    header_height = 0.42
    title_height = 0.55
    total_height = title_height + header_height + n_rows * row_height + 0.15

    fig, ax = plt.subplots(figsize=(total_width, total_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    # Title
    ax.text(total_width / 2, total_height - title_height / 2, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            fontfamily="sans-serif")

    y_top = total_height - title_height

    # Colors
    header_bg = "#2c3e50"
    header_fg = "white"
    row_bg_even = "#f8f9fa"
    row_bg_odd = "white"
    best_bg = "#d4edda"
    ours_fg = "#1a5276"
    grid_color = "#dee2e6"

    # Draw header
    x = 0
    for j, header in enumerate(display_df.columns):
        rect = FancyBboxPatch((x, y_top - header_height), col_widths[j], header_height,
                              boxstyle="square,pad=0", facecolor=header_bg,
                              edgecolor=grid_color, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + col_widths[j] / 2, y_top - header_height / 2, header,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=header_fg, fontfamily="sans-serif")
        x += col_widths[j]

    # Colors for separator
    sep_bg = "#34495e"
    sep_fg = "white"

    # Draw data rows
    for i in range(n_rows):
        y = y_top - header_height - (i + 1) * row_height
        method_name = display_df.iloc[i, 0]

        # Separator row
        if method_name == SEPARATOR_SENTINEL:
            rect = FancyBboxPatch((0, y), total_width, row_height,
                                  boxstyle="square,pad=0", facecolor=sep_bg,
                                  edgecolor=grid_color, linewidth=0.5)
            ax.add_patch(rect)
            ax.text(0.1, y + row_height / 2, SEPARATOR_LABEL,
                    ha="left", va="center", fontsize=7, fontweight="bold",
                    fontstyle="italic", color=sep_fg, fontfamily="sans-serif")
            continue

        is_gt = "GT" in method_name
        is_ours = method_name in ours_set
        base_bg = row_bg_even if i % 2 == 0 else row_bg_odd

        x = 0
        for j, header in enumerate(display_df.columns):
            # Determine cell background
            cell_bg = base_bg
            if (i, header) in best and not is_gt:
                cell_bg = best_bg

            rect = FancyBboxPatch((x, y), col_widths[j], row_height,
                                  boxstyle="square,pad=0", facecolor=cell_bg,
                                  edgecolor=grid_color, linewidth=0.5)
            ax.add_patch(rect)

            cell_text = display_df.iloc[i, j]
            fw = "bold" if (is_ours or is_gt) else "normal"
            fc = ours_fg if is_ours else ("black" if not is_gt else "#555555")
            ha = "left" if j == 0 else "center"
            x_text = x + 0.1 if j == 0 else x + col_widths[j] / 2

            ax.text(x_text, y + row_height / 2, cell_text,
                    ha=ha, va="center", fontsize=7, fontweight=fw,
                    color=fc, fontfamily="sans-serif")
            x += col_widths[j]

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def load_datasets(csv_dir: Path) -> List[Tuple[dict, pd.DataFrame]]:
    """Load all dataset CSVs. Returns list of (dataset_config, dataframe)."""
    results = []
    for ds in DATASETS:
        csv_path = csv_dir / ds["csv_file"]
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df[~df["Method"].isin(HIDE_METHODS)].reset_index(drop=True)
        print(f"[OK] {csv_path.name} → {ds['title']} ({len(df)} methods)")
        results.append((ds, df))
    return results


def generate_md(datasets: List[Tuple[dict, pd.DataFrame]], gt_quality_df: pd.DataFrame,
                output_path: Path):
    """Generate markdown file."""
    sections = []
    sections.append("# Baseline Results\n")

    # GT quality comparison first
    if not gt_quality_df.empty:
        sections.append(f"### {GT_QUALITY_TITLE}\n")
        table = build_gt_quality_table(gt_quality_df)
        if table:
            sections.append(table)
            sections.append("")

    sections.append("Evaluated across 4 datasets spanning indoor/outdoor and real/synthetic domains.\n")
    sections.append("Indoor datasets use tighter thresholds (0.1--1.0cm); "
                    "outdoor datasets use looser thresholds (2.0--10.0cm).\n")

    for ds, df in datasets:
        sections.append(f"### {ds['title']}\n")
        table = build_table(df, ds["thresholds"])
        if table:
            sections.append(table)
            sections.append("")

    md_content = "\n".join(sections) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content)
    print(f"  Written: {output_path}")


def generate_png(datasets: List[Tuple[dict, pd.DataFrame]], gt_quality_df: pd.DataFrame,
                 output_path: Path):
    """Generate a single PNG with all tables stacked vertically."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figs = []

    # GT quality table first
    if not gt_quality_df.empty:
        display_df = build_gt_quality_display_df(gt_quality_df)
        if not display_df.empty:
            best = find_best_gt_quality(gt_quality_df)
            fig = render_table_figure(display_df, GT_QUALITY_TITLE, best,
                                      gt_quality_df, ours_set=OUR_GT_METHODS)
            figs.append(fig)

    for ds, df in datasets:
        display_df = build_display_df(df, ds["thresholds"])
        if display_df.empty:
            continue
        best = find_best_indices(df, ds["thresholds"])
        fig = render_table_figure(display_df, ds["title"], best, df)
        figs.append(fig)

    if not figs:
        print("  No figures to render.")
        return

    # Render each figure to an image array
    images = []
    for fig in figs:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        images.append(img)
        plt.close(fig)

    # Stack vertically with padding
    pad = 20
    total_h = sum(img.shape[0] for img in images) + pad * (len(images) - 1)
    max_w = max(img.shape[1] for img in images)

    combined = np.ones((total_h, max_w, 4), dtype=np.uint8) * 255
    y = 0
    for img in images:
        h, w = img.shape[:2]
        combined[y:y + h, :w] = img
        y += h + pad

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(combined[:, :, :3]).save(str(output_path), dpi=(200, 200))
    print(f"  Written: {output_path}")


def generate_pdf(datasets: List[Tuple[dict, pd.DataFrame]], gt_quality_df: pd.DataFrame,
                 output_path: Path):
    """Generate a single PDF with one table per page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        # GT quality table first
        if not gt_quality_df.empty:
            display_df = build_gt_quality_display_df(gt_quality_df)
            if not display_df.empty:
                best = find_best_gt_quality(gt_quality_df)
                fig = render_table_figure(display_df, GT_QUALITY_TITLE, best,
                                          gt_quality_df, ours_set=OUR_GT_METHODS)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

        for ds, df in datasets:
            display_df = build_display_df(df, ds["thresholds"])
            if display_df.empty:
                continue
            best = find_best_indices(df, ds["thresholds"])
            fig = render_table_figure(display_df, ds["title"], best, df)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

    print(f"  Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate project_results in MD/PDF/PNG from evaluation CSVs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for .md (default: docs/project_results.md)")
    parser.add_argument("--csv-dir", type=str, default=DEFAULT_CSV_DIR,
                        help=f"Directory containing table_*.csv files (default: {DEFAULT_CSV_DIR})")
    parser.add_argument("--pdf", action="store_true", help="Also generate PDF")
    parser.add_argument("--png", action="store_true", help="Also generate PNG")
    parser.add_argument("--no-md", action="store_true", help="Skip markdown generation")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    docs_dir = base_dir.parent
    csv_dir = Path(args.csv_dir)
    output_path = Path(args.output) if args.output else docs_dir / "project_results.md"

    gt_quality_df = load_gt_quality(base_dir)
    datasets = load_datasets(csv_dir)
    if not datasets and gt_quality_df.empty:
        print("No datasets found.")
        return

    if not args.no_md:
        generate_md(datasets, gt_quality_df, output_path)

    if args.pdf:
        pdf_path = output_path.with_suffix(".pdf")
        generate_pdf(datasets, gt_quality_df, pdf_path)

    if args.png:
        png_path = output_path.with_suffix(".png")
        generate_png(datasets, gt_quality_df, png_path)


if __name__ == "__main__":
    main()
