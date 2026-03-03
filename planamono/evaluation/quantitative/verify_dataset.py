#!/usr/bin/env python3
"""
Unified Dataset Verification CLI
=================================

Master CLI for verifying ScanNet++ and Hypersim datasets used for
plane segmentation evaluation. Orchestrates check modules and report
generation.

Usage examples:
    # Quick check of ScanNet++ val split (3 scenes, 2 frames each)
    python verify_dataset.py --dataset scannetpp --splits val --max-scenes 3 --sample-frames 2

    # Full check of both datasets
    python verify_dataset.py --dataset all --splits all

    # Fast mode (existence only)
    python verify_dataset.py --dataset scannetpp --skip-frame-content

    # Full verification with all optional checks
    python verify_dataset.py --dataset all --check-predictions --check-plane-ids --visual-report

    # Hypersim with visual report
    python verify_dataset.py --dataset hypersim --splits val --visual-report --visual-output-dir ./vis_report
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent to path so we can import from dataset_verification
sys.path.insert(0, os.path.dirname(__file__))

from dataset_verification.scannetpp_checks import (
    run_all_checks as run_scannetpp_checks,
    DEFAULT_PATHS as SCANNETPP_PATHS,
    DEFAULT_PREDICTION_FOLDERS as SCANNETPP_PRED_FOLDERS,
)
from dataset_verification.hypersim_checks import (
    run_all_checks as run_hypersim_checks,
    DEFAULT_PATHS as HYPERSIM_PATHS,
    DEFAULT_PREDICTION_FOLDERS as HYPERSIM_PRED_FOLDERS,
)
from dataset_verification.report import format_text_report, generate_visual_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified dataset verification for ScanNet++ and Hypersim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset scannetpp --splits val --max-scenes 3 --sample-frames 2
  %(prog)s --dataset all --splits all --check-predictions --check-plane-ids
  %(prog)s --dataset hypersim --splits val test --visual-report
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["scannetpp", "hypersim", "all"],
        default="all",
        help="Which dataset(s) to verify (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        help="Which splits to check (train/val/test/all). Default: all",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of frames to sample per scene/camera for content checks "
             "(default: 5). Set to 0 to disable content checks.",
    )
    parser.add_argument(
        "--skip-frame-content",
        action="store_true",
        help="Skip reading actual frame data (fast mode, existence-only checks)",
    )
    parser.add_argument(
        "--check-predictions",
        action="store_true",
        help="Also verify prediction H5s align with GT",
    )
    parser.add_argument(
        "--check-plane-ids",
        action="store_true",
        help="Cross-check PLY mesh plane IDs vs H5 labels "
             "(ScanNet++ only; checks frame consistency for both)",
    )
    parser.add_argument(
        "--visual-report",
        action="store_true",
        help="Generate matplotlib visual report (histograms, heatmaps, PDF)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Text report output path (default: report_verification.txt in script dir)",
    )
    parser.add_argument(
        "--visual-output-dir",
        type=str,
        default=None,
        help="Visual report output directory (default: verification_visual/ in script dir)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit number of scenes per split (for quick testing)",
    )
    return parser.parse_args()


def resolve_splits(splits_arg):
    """Resolve 'all' to actual split names."""
    if "all" in splits_arg:
        return ["train", "val", "test"]
    return splits_arg


def main():
    args = parse_args()

    if args.sample_frames == 0:
        args.skip_frame_content = True

    splits = resolve_splits(args.splits)
    datasets = (
        ["scannetpp", "hypersim"] if args.dataset == "all"
        else [args.dataset]
    )

    print("=" * 72)
    print(f" Dataset Verification")
    print(f" Datasets: {', '.join(datasets)}")
    print(f" Splits:   {', '.join(splits)}")
    print(f" Options:  sample_frames={args.sample_frames}, "
          f"skip_content={args.skip_frame_content}, "
          f"predictions={args.check_predictions}, "
          f"plane_ids={args.check_plane_ids}, "
          f"visual={args.visual_report}")
    if args.max_scenes:
        print(f" Max scenes per split: {args.max_scenes}")
    print("=" * 72)
    print()

    all_results = []

    # Run ScanNet++ checks
    if "scannetpp" in datasets:
        print("[1/2] Running ScanNet++ checks..." if len(datasets) > 1
              else "Running ScanNet++ checks...")
        config = {
            "splits": splits,
            "sample_frames": args.sample_frames,
            "skip_frame_content": args.skip_frame_content,
            "check_predictions": args.check_predictions,
            "check_plane_ids": args.check_plane_ids,
            "max_scenes": args.max_scenes,
            **SCANNETPP_PATHS,
            "prediction_folders": SCANNETPP_PRED_FOLDERS,
        }
        scannetpp_result = run_scannetpp_checks(config)
        all_results.append(scannetpp_result)
        print()

    # Run Hypersim checks
    if "hypersim" in datasets:
        print("[2/2] Running Hypersim checks..." if len(datasets) > 1
              else "Running Hypersim checks...")
        config = {
            "splits": splits,
            "sample_frames": args.sample_frames,
            "skip_frame_content": args.skip_frame_content,
            "check_predictions": args.check_predictions,
            "check_plane_ids": args.check_plane_ids,
            "max_scenes": args.max_scenes,
            **HYPERSIM_PATHS,
            "prediction_folders": HYPERSIM_PRED_FOLDERS,
        }
        hypersim_result = run_hypersim_checks(config)
        all_results.append(hypersim_result)
        print()

    # Generate text report
    print("Generating text report...")
    report = format_text_report(all_results)
    print(report)

    # Save text report
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "report_verification.txt"
    )
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nText report saved to: {output_path}")

    # Generate visual report
    if args.visual_report:
        visual_dir = args.visual_output_dir or os.path.join(
            os.path.dirname(__file__), "verification_visual"
        )
        print(f"\nGenerating visual report in {visual_dir}...")
        generated = generate_visual_report(all_results, visual_dir)
        for path in generated:
            print(f"  Generated: {path}")
        if generated:
            pdf_files = [p for p in generated if p.endswith(".pdf")]
            if pdf_files:
                print(f"\nVisual report PDF: {pdf_files[0]}")


if __name__ == "__main__":
    main()
