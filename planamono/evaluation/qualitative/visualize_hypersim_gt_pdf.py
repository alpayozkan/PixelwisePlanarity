#!/usr/bin/env python3
"""
Generate PDF visualization of Hypersim GT evaluation: worst/best frames.

Each page: 3 rows x 5 columns (RGB, Depth, GT Planes) x 5 samples.
Prec/Recall at 1mm/5mm/10mm overlaid on each GT plane panel.
Uses visualize_top_components_v2 for plane coloring (top-20, rest gray).

Usage:
    # Worst 20 frames by prec@1.0cm
    python visualize_hypersim_gt_pdf.py --worst-n 20

    # Worst 50 by prec@0.5cm
    python visualize_hypersim_gt_pdf.py --worst-n 50 --sort-metric prec@0.5cm

    # Best 20
    python visualize_hypersim_gt_pdf.py --best-n 20

    # Both worst and best
    python visualize_hypersim_gt_pdf.py --worst-n 20 --best-n 10

    # Custom eval dir and output
    python visualize_hypersim_gt_pdf.py --worst-n 20 --eval-dir /path/to/eval/gt_v2 --output out.pdf
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from planamono.shared.utils.visualization import visualize_top_components_v2
from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.paths import repo_path

# ── Defaults ──────────────────────────────────────────────────────────────────
EVAL_DIR = Path('/cluster/scratch/aoezkan/planeseg/hypersim/eval/gt_v2')
HYPERSIM_ROOT = '/cluster/scratch/ayavuz/dataset/Hypersim_merged'
PLANE_LABEL_ROOT = '/cluster/scratch/ayavuz/dataset/Hypersim_rendered'
PARAMS_ROOT = '/cluster/scratch/ayavuz/dataset/Hypersim_params'
META_CSV = os.path.join(repo_path, 'shared', 'datasets', 'metadata_camera_parameters.csv')
OUTPUT_DIR = Path('/cluster/scratch/aoezkan/planeseg/hypersim/visualizations')

SAMPLES_PER_PAGE = 5
TOP_K_PLANES = 20


def build_dataset(split='val'):
    dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=PLANE_LABEL_ROOT,
        params_root=PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, 'splits', 'hypersim'),
        split=split,
        metadata_csv=META_CSV,
    )
    # Build lookup
    frame_to_idx = {}
    for i in range(len(dataset)):
        s_id, cam, _, fid = dataset.valid_pairs[i][:4]
        frame_to_idx[(s_id, f"{cam}/{fid}")] = i
    return dataset, frame_to_idx


def resolve_frames(df_subset, frame_to_idx):
    """Map result rows to dataset indices, skip missing."""
    frames = []
    for _, row in df_subset.iterrows():
        key = (row['scene_id'], row['frame_idx'])
        if key in frame_to_idx:
            frames.append((key, frame_to_idx[key], row))
    return frames


def render_page(pdf, dataset, frames, page_title):
    """Render one page with up to SAMPLES_PER_PAGE samples (3 rows x N cols)."""
    n = len(frames)
    if n == 0:
        return

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes[:, np.newaxis]

    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)

    for col, (key, didx, row) in enumerate(frames):
        sample = dataset[didx]
        rgb = sample['image'].permute(1, 2, 0).numpy()
        depth = sample['depth'][0].numpy()
        gt_seg = sample['plane'][0].numpy().astype(np.int32)
        H, W = gt_seg.shape
        n_planes = len(np.unique(gt_seg)) - 1

        p01 = row.get('prec@0.1cm', 0)
        p05 = row.get('prec@0.5cm', 0)
        p10 = row.get('prec@1.0cm', 0)
        r01 = row.get('rec@0.1cm', 0)
        r05 = row.get('rec@0.5cm', 0)
        r10 = row.get('rec@1.0cm', 0)

        # Row 0: RGB
        axes[0, col].imshow(rgb)
        axes[0, col].text(
            5, 5, f"{key[0]}\n{key[1]}", fontsize=7, color='white',
            bbox=bbox_props, va='top')
        axes[0, col].axis('off')

        # Row 1: Depth
        valid_d = depth[depth > 0]
        vmin = valid_d.min() if len(valid_d) > 0 else 0
        vmax = np.percentile(valid_d, 99) if len(valid_d) > 0 else 1
        axes[1, col].imshow(depth, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, col].text(
            5, 5, f"{vmin:.1f} - {valid_d.max():.1f} m", fontsize=7,
            color='white', bbox=bbox_props, va='top')
        axes[1, col].axis('off')

        # Row 2: GT seg (top-20) with prec/recall overlaid
        seg_colored = visualize_top_components_v2(
            gt_seg, k=TOP_K_PLANES, ignore_label=0, return_colors=True)
        axes[2, col].imshow(seg_colored)
        metric_text = (
            f"{n_planes} planes\n"
            f"        1mm   5mm  10mm\n"
            f"Prec {p01:.3f} {p05:.3f} {p10:.3f}\n"
            f"Rec  {r01:.3f} {r05:.3f} {r10:.3f}"
        )
        axes[2, col].text(
            5, H - 5, metric_text, fontsize=7, color='white',
            fontfamily='monospace', bbox=bbox_props, va='bottom')
        axes[2, col].axis('off')

    for r, label in enumerate(['RGB', 'Depth', 'GT Planes']):
        axes[r, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                              rotation=0, labelpad=50, va='center')

    fig.suptitle(page_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf(eval_dir, sort_metric, worst_n, best_n, output_path):
    # Load results
    results_csv = eval_dir / 'results.csv'
    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found")
        sys.exit(1)

    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} per-frame results from {results_csv}")

    # Load dataset
    print("Loading dataset...")
    dataset, frame_to_idx = build_dataset()

    # Collect frames to render
    sections = []

    if worst_n > 0:
        worst_rows = df.nsmallest(worst_n * 3, sort_metric)
        frames_worst = resolve_frames(worst_rows, frame_to_idx)[:worst_n]
        sections.append(("WORST", frames_worst))
        print(f"Worst {len(frames_worst)} frames by {sort_metric}")

    if best_n > 0:
        best_rows = df.nlargest(best_n * 3, sort_metric)
        frames_best = resolve_frames(best_rows, frame_to_idx)[:best_n]
        sections.append(("BEST", frames_best))
        print(f"Best {len(frames_best)} frames by {sort_metric}")

    total_pages = sum(
        (len(frames) + SAMPLES_PER_PAGE - 1) // SAMPLES_PER_PAGE
        for _, frames in sections
    )
    print(f"Generating {total_pages} pages -> {output_path}")

    os.makedirs(output_path.parent, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        for section_name, frames in sections:
            for page_start in tqdm(
                range(0, len(frames), SAMPLES_PER_PAGE),
                desc=section_name,
            ):
                page_frames = frames[page_start:page_start + SAMPLES_PER_PAGE]
                page_idx = page_start // SAMPLES_PER_PAGE + 1
                n_pages = (len(frames) + SAMPLES_PER_PAGE - 1) // SAMPLES_PER_PAGE
                page_title = (
                    f"{section_name} by {sort_metric} — "
                    f"page {page_idx}/{n_pages} "
                    f"(samples {page_start+1}-{page_start+len(page_frames)})"
                )
                render_page(pdf, dataset, page_frames, page_title)

    print(f"Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF of Hypersim GT worst/best frames")
    parser.add_argument('--worst-n', type=int, default=0,
                        help='Number of worst frames to include')
    parser.add_argument('--best-n', type=int, default=0,
                        help='Number of best frames to include')
    parser.add_argument('--sort-metric', type=str, default='prec@1.0cm',
                        help='Metric to sort by (default: prec@1.0cm)')
    parser.add_argument('--eval-dir', type=str, default=str(EVAL_DIR),
                        help='Evaluation results directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path (auto-generated if not set)')
    args = parser.parse_args()

    if args.worst_n == 0 and args.best_n == 0:
        parser.error("Provide at least one of --worst-n or --best-n")

    eval_dir = Path(args.eval_dir)
    eval_name = eval_dir.name

    if args.output:
        output_path = Path(args.output)
    else:
        parts = []
        if args.worst_n > 0:
            parts.append(f"worst{args.worst_n}")
        if args.best_n > 0:
            parts.append(f"best{args.best_n}")
        metric_short = args.sort_metric.replace('@', '').replace('.', '')
        fname = f"hypersim_{eval_name}_{'_'.join(parts)}_{metric_short}.pdf"
        output_path = OUTPUT_DIR / fname

    generate_pdf(eval_dir, args.sort_metric, args.worst_n, args.best_n, output_path)


if __name__ == '__main__':
    main()
