#!/usr/bin/env python3
"""
Text and visual report generation for dataset verification results.

Formats results from scannetpp_checks.run_all_checks() and
hypersim_checks.run_all_checks() into readable text reports and
optional matplotlib visual reports (PNGs + PDF).
"""

import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


# ============================================================
# Text report formatting
# ============================================================

def format_text_report(results: List[Dict]) -> str:
    """Format verification results into a comprehensive text report.

    Args:
        results: list of result dicts from run_all_checks()
                 (one per dataset: scannetpp, hypersim)

    Returns:
        Formatted text report string.
    """
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 72)
    out(" DATASET VERIFICATION REPORT")
    datasets = [r["dataset"] for r in results]
    timestamps = [r["timestamp"] for r in results]
    out(f" Generated: {timestamps[0]}")
    out(f" Datasets:  {', '.join(datasets)}")
    out("=" * 72)
    out()

    for r in results:
        if r["dataset"] == "scannetpp":
            _format_scannetpp(r, out)
        elif r["dataset"] == "hypersim":
            _format_hypersim(r, out)

    # Grand summary
    out("=" * 72)
    out(" GRAND SUMMARY")
    out("=" * 72)
    out()
    for r in results:
        ds = r["dataset"].upper()
        total_scenes = sum(
            sr.get("n_scenes", 0)
            for sr in r["splits"].values()
        )
        total_frames = sum(
            sr.get("total_gt_frames", 0)
            for sr in r["splits"].values()
        )
        total_issues = sum(
            len(sr.get("issues", []))
            for sr in r["splits"].values()
        )
        n_errors = sum(
            len([i for i in sr.get("issues", []) if i[0] == "ISSUE"])
            for sr in r["splits"].values()
        )
        n_warnings = sum(
            len([i for i in sr.get("issues", []) if i[0] == "WARN"])
            for sr in r["splits"].values()
        )

        split_counts = []
        for sname, sr in r["splits"].items():
            if "error" not in sr:
                split_counts.append(f"{sr.get('n_scenes', 0)} {sname}")

        out(f"  {ds}: {' / '.join(split_counts)} scenes")
        out(f"    {total_frames:,} total GT frames")
        out(f"    {n_errors} issues, {n_warnings} warnings")

        # Overlap
        overlap = r.get("split_overlap", {})
        if overlap:
            for pair, ids in overlap.items():
                out(f"    [ISSUE] Split overlap ({pair}): {len(ids)} scenes")
        else:
            out(f"    No split overlap detected")
        out()

    out("=" * 72)
    return "\n".join(lines)


def _format_scannetpp(r: Dict, out):
    """Format ScanNet++ results."""
    out(f"[SCANNETPP] {'=' * 58}")
    out()

    # Paths
    out("  PATHS")
    out("  " + "-" * 40)
    for name, info in r["paths"].items():
        status = "[OK]" if info["exists"] else "[MISSING]"
        out(f"  {status:10s} {name}: {info['path']}")
    out()

    # Split files
    out("  SPLIT FILES")
    out("  " + "-" * 40)
    for name, info in r["split_files"].items():
        if info["exists"]:
            out(f"  {name}.txt: {info['n_scenes']} scenes")
        else:
            out(f"  {name}.txt: [MISSING]")

    overlap = r.get("split_overlap", {})
    if overlap:
        for pair, ids in overlap.items():
            out(f"  [ISSUE] Overlap ({pair}): {len(ids)} scenes: {ids[:5]}{'...' if len(ids) > 5 else ''}")
    else:
        out("  Train/val/test overlap: NONE (OK)")
    out()

    # Per-split results
    for split_name, sr in r["splits"].items():
        if "error" in sr:
            out(f"  SPLIT: {split_name.upper()} — ERROR: {sr['error']}")
            out()
            continue

        _format_scannetpp_split(split_name, sr, out)

    out()


def _format_scannetpp_split(split_name: str, sr: Dict, out):
    """Format a single ScanNet++ split."""
    out(f"  SPLIT: {split_name.upper()} ({sr['n_scenes']} scenes)")
    out("  " + "-" * 40)
    out()

    # Scene summary
    n = sr["n_scenes"]
    out(f"  Scene Summary:")
    out(f"    Scenes total:               {n}")
    out(f"    Scenes with GT plane H5:    {sr['n_scenes_with_gt_h5']:>4d}  ({sr['n_scenes_with_gt_h5']/n*100:.1f}%)" if n else "")
    out(f"    Scenes with RGB dir:        {sr['n_scenes_with_rgb_dir']:>4d}  ({sr['n_scenes_with_rgb_dir']/n*100:.1f}%)" if n else "")
    out(f"    Scenes with depth H5:       {sr['n_scenes_with_depth_h5']:>4d}  ({sr['n_scenes_with_depth_h5']/n*100:.1f}%)" if n else "")
    out(f"    Scenes with sem H5:         {sr['n_scenes_with_sem_h5']:>4d}  ({sr['n_scenes_with_sem_h5']/n*100:.1f}%)" if n else "")
    out(f"    Scenes with pose JSON:      {sr['n_scenes_with_pose_json']:>4d}  ({sr['n_scenes_with_pose_json']/n*100:.1f}%)" if n else "")
    out()

    _report_list(out, "ISSUE", "Scenes MISSING GT plane H5", sr["scenes_missing_gt_h5"])
    _report_list(out, "WARN", "Scenes missing RGB dir", sr["scenes_missing_rgb_dir"])
    _report_list(out, "WARN", "Scenes missing pose JSON", sr["scenes_missing_pose_json"])
    _report_list(out, "WARN", "H5 frame count mismatches", sr["scenes_frame_count_mismatch"])

    if sr["scenes_duplicate_frame_ids"]:
        out(f"    [ISSUE] Scenes with duplicate frame IDs ({len(sr['scenes_duplicate_frame_ids'])}):")
        for scene_id, dups in sr["scenes_duplicate_frame_ids"][:10]:
            out(f"      {scene_id}: {dups[:5]}{'...' if len(dups) > 5 else ''}")
        out()

    # Frame summary
    tf = sr["total_gt_frames"]
    out(f"  Frame Summary:")
    out(f"    Total GT frames:            {tf:>6d}")
    out(f"    Frames with RGB on disk:    {sr['frames_with_rgb']:>6d}  ({sr['frames_with_rgb']/tf*100:.1f}%)" if tf else "")
    out(f"    Frames with pose entry:     {sr['frames_with_pose']:>6d}  ({sr['frames_with_pose']/tf*100:.1f}%)" if tf else "")
    out(f"    Frames missing RGB:         {len(sr['frames_missing_rgb']):>6d}")
    out(f"    Frames missing pose:        {len(sr['frames_missing_pose']):>6d}")
    out()

    # Content quality
    if sr["content_checked"] > 0:
        out(f"  Data Quality (sampled {sr['content_checked']} frames):")
        _report_list(out, "WARN", "RGB unreadable", sr["content_rgb_unreadable"], limit=10)
        _report_list(out, "WARN", "Depth >5% invalid", [
            f"{fid}: {pct:.1f}%" for fid, pct in sr["content_depth_bad"]
        ], limit=10)
        _report_list(out, "WARN", "Plane all-zero", sr["content_plane_all_zero"], limit=10)
        _report_list(out, "WARN", "Plane negative labels", sr["content_plane_negative"], limit=10)

        if sr["depth_mins"]:
            out(f"    Depth range: [{np.min(sr['depth_mins']):.3f}m, {np.max(sr['depth_maxs']):.3f}m]")
        if sr["plane_pct_planars"]:
            pcts = sr["plane_pct_planars"]
            out(f"    Plane coverage: mean {np.mean(pcts):.1f}%, "
                f"min {np.min(pcts):.1f}%, max {np.max(pcts):.1f}%")
        out()

    # Predictions
    if sr["predictions"]:
        n_gt = sr["n_scenes_with_gt_h5"]
        out(f"  Predictions:")
        for method, ps in sr["predictions"].items():
            status = "[OK]" if ps["n_with_h5"] == n_gt else "[ISSUE]"
            out(f"    {status} {method}: {ps['n_with_h5']}/{n_gt} scenes with H5, "
                f"{ps['n_readable']} readable, {ps['n_frame_match']} frame-matched")
            if ps["missing_scenes"]:
                out(f"      Missing: {ps['missing_scenes'][:5]}{'...' if len(ps['missing_scenes']) > 5 else ''}")
        out()

    # Plane ID cross-check
    if sr["plane_id_checks"]:
        pid_checks = sr["plane_id_checks"]
        n_correct = sum(1 for v in pid_checks.values() if v["diagnosis"] == "correct")
        n_remap = sum(1 for v in pid_checks.values() if v["diagnosis"] == "per_frame_remap")
        n_collision = sum(1 for v in pid_checks.values() if v["diagnosis"] == "collision")
        n_no_ply = sum(1 for v in pid_checks.values() if v["diagnosis"] == "no_ply")
        n_total = len(pid_checks)
        out(f"  Plane ID Cross-Check ({n_total} scenes):")
        out(f"    Correct (+1 shift):     {n_correct}")
        out(f"    Per-frame remap:        {n_remap}")
        out(f"    Collision (id=0 lost):  {n_collision}")
        out(f"    No PLY available:       {n_no_ply}")
        if n_remap > 0:
            remapped = [k for k, v in pid_checks.items() if v["diagnosis"] == "per_frame_remap"]
            out(f"    Remapped scenes: {remapped[:10]}{'...' if len(remapped) > 10 else ''}")
        if n_collision > 0:
            collided = [k for k, v in pid_checks.items() if v["diagnosis"] == "collision"]
            out(f"    Collision scenes: {collided[:10]}")
        out()

    out()


def _format_hypersim(r: Dict, out):
    """Format Hypersim results."""
    out(f"[HYPERSIM] {'=' * 59}")
    out()

    # Paths
    out("  PATHS")
    out("  " + "-" * 40)
    for name, info in r["paths"].items():
        status = "[OK]" if info["exists"] else "[MISSING]"
        out(f"  {status:10s} {name}: {info['path']}")
    out()

    # Split files
    out("  SPLIT FILES")
    out("  " + "-" * 40)
    for name, info in r["split_files"].items():
        if info["exists"]:
            out(f"  {name}.txt: {info['n_scenes']} scenes")
        else:
            out(f"  {name}.txt: [MISSING]")

    overlap = r.get("split_overlap", {})
    if overlap:
        for pair, ids in overlap.items():
            out(f"  [ISSUE] Overlap ({pair}): {len(ids)} scenes")
    else:
        out("  Train/val/test overlap: NONE (OK)")
    out()

    # Per-split
    for split_name, sr in r["splits"].items():
        if "error" in sr:
            out(f"  SPLIT: {split_name.upper()} — ERROR: {sr['error']}")
            out()
            continue

        _format_hypersim_split(split_name, sr, out)

    out()


def _format_hypersim_split(split_name: str, sr: Dict, out):
    """Format a single Hypersim split."""
    n = sr["n_scenes"]
    out(f"  SPLIT: {split_name.upper()} ({n} scenes)")
    out("  " + "-" * 40)
    out()

    # Scene summary
    out(f"  Scene Summary:")
    out(f"    Scenes total:               {n}")
    out(f"    Scenes with Hypersim data:  {sr['n_scenes_with_data']:>4d}")
    out(f"    Scenes with GT labels:      {sr['n_scenes_with_gt']:>4d}")
    out(f"    Scenes with params dir:     {sr['n_scenes_with_params']:>4d}")
    out()

    _report_list(out, "ISSUE", "Scenes MISSING data dir", sr["scenes_missing_data"])
    _report_list(out, "ISSUE", "Scenes MISSING GT labels", sr["scenes_missing_gt"])
    _report_list(out, "WARN", "Scenes missing params", sr["scenes_missing_params"])

    # Camera summary
    out(f"  Camera Summary:")
    out(f"    Total cameras (from GT H5): {sr['total_cameras']}")
    out(f"    Intrinsics from CSV:        {len(sr['cameras_csv_intrinsics'])}")
    out(f"    Intrinsics from default:    {len(sr['cameras_default_intrinsics'])}")
    out()

    _report_list(out, "ISSUE", "GT H5 unreadable", sr["cameras_gt_unreadable"])
    _report_list(out, "ISSUE", "Missing RGB dir", sr["cameras_no_rgb_dir"])
    _report_list(out, "ISSUE", "Missing depth dir", sr["cameras_no_depth_dir"])

    if sr["cameras_duplicate_frame_ids"]:
        out(f"    [ISSUE] Cameras with duplicate frame IDs ({len(sr['cameras_duplicate_frame_ids'])}):")
        for cam_str, dups in sr["cameras_duplicate_frame_ids"][:10]:
            out(f"      {cam_str}: {dups[:5]}")
        out()

    # Frame summary
    tf = sr["total_gt_frames"]
    out(f"  Frame Summary:")
    out(f"    Total GT frames:            {tf:>6d}")
    out(f"    Frames with RGB:            {sr['frames_with_rgb']:>6d}  ({sr['frames_with_rgb']/tf*100:.1f}%)" if tf else "")
    out(f"    Frames with depth:          {sr['frames_with_depth']:>6d}  ({sr['frames_with_depth']/tf*100:.1f}%)" if tf else "")
    out(f"    Frames missing RGB:         {len(sr['frames_missing_rgb']):>6d}")
    out(f"    Frames missing depth:       {len(sr['frames_missing_depth']):>6d}")
    out(f"    Frames missing both:        {len(sr['frames_missing_both']):>6d}")
    out()

    # Content quality
    if sr["content_checked"] > 0:
        out(f"  Data Quality (sampled {sr['content_checked']} frames):")
        _report_list(out, "WARN", "RGB >1% bad pixels", [
            f"{fid}: {pct:.1f}% (dtype={dt})"
            for fid, pct, dt in sr["content_rgb_bad"]
        ], limit=10)
        _report_list(out, "WARN", "Depth >5% invalid", [
            f"{fid}: {pct:.1f}%" for fid, pct in sr["content_depth_bad"]
        ], limit=10)
        _report_list(out, "WARN", "Plane all-zero", sr["content_plane_all_zero"], limit=10)
        _report_list(out, "WARN", "Plane negative labels", sr["content_plane_negative"], limit=10)

        if sr["depth_mins"]:
            out(f"    Depth range: [{np.min(sr['depth_mins']):.3f}m, {np.max(sr['depth_maxs']):.3f}m]")
        if sr["plane_pct_planars"]:
            pcts = sr["plane_pct_planars"]
            out(f"    Plane coverage: mean {np.mean(pcts):.1f}%, "
                f"min {np.min(pcts):.1f}%, max {np.max(pcts):.1f}%")
        out()

    # Predictions
    if sr["predictions"]:
        tc = sr["total_cameras"]
        out(f"  Predictions:")
        for method, ps in sr["predictions"].items():
            status = "[OK]" if ps["n_with_h5"] == tc else "[ISSUE]"
            out(f"    {status} {method}: {ps['n_with_h5']}/{tc} cameras with H5, "
                f"{ps['n_readable']} readable, {ps['n_frame_match']} frame-matched")
            if ps["missing_cams"]:
                out(f"      Missing: {ps['missing_cams'][:5]}{'...' if len(ps['missing_cams']) > 5 else ''}")
        out()

    # Plane ID cross-check
    if sr["plane_id_checks"]:
        pid_checks = sr["plane_id_checks"]
        n_consistent = sum(1 for v in pid_checks.values() if v["diagnosis"] == "consistent")
        n_remap = sum(1 for v in pid_checks.values() if v["diagnosis"] == "per_frame_remap")
        n_neg = sum(1 for v in pid_checks.values() if v.get("has_negative_labels"))
        n_total = len(pid_checks)
        out(f"  Plane ID Cross-Check ({n_total} cameras):")
        out(f"    Consistent across frames:   {n_consistent}")
        out(f"    Per-frame remap:            {n_remap}")
        out(f"    Negative labels found:      {n_neg}")
        if n_remap > 0:
            remapped = [k for k, v in pid_checks.items() if v["diagnosis"] == "per_frame_remap"]
            out(f"    Remapped cameras: {remapped[:10]}{'...' if len(remapped) > 10 else ''}")
        out()

    out()


def _report_list(out, severity: str, label: str, items: list,
                 limit: int = 20):
    """Report a list of items with severity tag, truncating if needed."""
    if not items:
        return
    tag = f"[{severity}]"
    out(f"    {tag} {label} ({len(items)}):")
    for item in items[:limit]:
        out(f"      - {item}")
    if len(items) > limit:
        out(f"      ... and {len(items) - limit} more")
    out()


# ============================================================
# Visual report generation
# ============================================================

def generate_visual_report(
    results: List[Dict],
    output_dir: str,
) -> List[str]:
    """Generate visual report with matplotlib.

    Creates:
    1. Plane coverage histogram (per dataset/split)
    2. Num planes histogram (per dataset/split)
    3. Per-scene completeness heatmap
    4. Depth range box plot
    5. Missing data bar chart
    6. Summary PDF combining all figures

    Args:
        results: list of result dicts from run_all_checks()
        output_dir: directory to save PNGs and PDF

    Returns:
        List of generated file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # Collect figures for PDF
    figures = []

    for r in results:
        ds = r["dataset"]

        for split_name, sr in r["splits"].items():
            if "error" in sr or sr.get("content_checked", 0) == 0:
                continue

            prefix = f"{ds}_{split_name}"

            # 1. Plane coverage histogram
            if sr.get("plane_pct_planars"):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(sr["plane_pct_planars"], bins=30, color="steelblue",
                        edgecolor="white", alpha=0.85)
                ax.set_xlabel("Planar pixel coverage (%)")
                ax.set_ylabel("Count")
                ax.set_title(f"{ds.upper()} / {split_name} — Plane Coverage Distribution")
                ax.axvline(np.mean(sr["plane_pct_planars"]), color="red",
                           linestyle="--", label=f"mean={np.mean(sr['plane_pct_planars']):.1f}%")
                ax.legend()
                fig.tight_layout()
                path = os.path.join(output_dir, f"{prefix}_plane_coverage_hist.png")
                fig.savefig(path, dpi=150)
                generated.append(path)
                figures.append(fig)

            # 2. Num planes histogram
            if sr.get("plane_n_labels_list"):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(sr["plane_n_labels_list"], bins=30, color="coral",
                        edgecolor="white", alpha=0.85)
                ax.set_xlabel("Number of unique plane labels")
                ax.set_ylabel("Count")
                ax.set_title(f"{ds.upper()} / {split_name} — Plane Count Distribution")
                ax.axvline(np.mean(sr["plane_n_labels_list"]), color="red",
                           linestyle="--", label=f"mean={np.mean(sr['plane_n_labels_list']):.1f}")
                ax.legend()
                fig.tight_layout()
                path = os.path.join(output_dir, f"{prefix}_plane_count_hist.png")
                fig.savefig(path, dpi=150)
                generated.append(path)
                figures.append(fig)

            # 3. Depth range box plot
            if sr.get("depth_mins") and sr.get("depth_maxs"):
                fig, ax = plt.subplots(figsize=(8, 5))
                data = [sr["depth_mins"], sr["depth_maxs"]]
                labels = ["Depth min (m)", "Depth max (m)"]
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                bp["boxes"][1].set_facecolor("lightsalmon")
                ax.set_ylabel("Depth (meters)")
                ax.set_title(f"{ds.upper()} / {split_name} — Depth Range Distribution")
                fig.tight_layout()
                path = os.path.join(output_dir, f"{prefix}_depth_range_box.png")
                fig.savefig(path, dpi=150)
                generated.append(path)
                figures.append(fig)

    # Per-dataset completeness heatmap
    for r in results:
        ds = r["dataset"]
        for split_name, sr in r["splits"].items():
            if "error" in sr:
                continue
            fig = _make_completeness_heatmap(ds, split_name, sr)
            if fig is not None:
                path = os.path.join(output_dir, f"{ds}_{split_name}_completeness.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                generated.append(path)
                figures.append(fig)

    # Missing data bar chart
    fig = _make_missing_bar_chart(results)
    if fig is not None:
        path = os.path.join(output_dir, "missing_data_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        generated.append(path)
        figures.append(fig)

    # Combine into PDF
    if figures:
        pdf_path = os.path.join(output_dir, "verification_visual_report.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        generated.append(pdf_path)

    # Close all figures
    for fig in figures:
        plt.close(fig)

    return generated


def _make_completeness_heatmap(ds: str, split_name: str, sr: Dict):
    """Create a per-scene completeness heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ds == "scannetpp":
        # Build scene list and check categories
        all_scenes = set()
        categories = ["GT H5", "RGB dir", "Depth H5", "Sem H5", "Pose JSON"]
        missing_sets = {
            "GT H5": set(sr.get("scenes_missing_gt_h5", [])),
            "RGB dir": set(sr.get("scenes_missing_rgb_dir", [])),
            "Depth H5": set(sr.get("scenes_missing_depth_h5", [])),
            "Sem H5": set(sr.get("scenes_missing_sem_h5", [])),
            "Pose JSON": set(sr.get("scenes_missing_pose_json", [])),
        }
        # Collect all scene IDs
        for s in missing_sets.values():
            all_scenes.update(s)
        # Only show scenes that have at least one issue (otherwise heatmap is all green)
        if not all_scenes:
            return None
        scenes = sorted(all_scenes)
    elif ds == "hypersim":
        all_scenes = set()
        categories = ["Data dir", "GT labels", "Params dir"]
        missing_sets = {
            "Data dir": set(sr.get("scenes_missing_data", [])),
            "GT labels": set(sr.get("scenes_missing_gt", [])),
            "Params dir": set(sr.get("scenes_missing_params", [])),
        }
        for s in missing_sets.values():
            all_scenes.update(s)
        if not all_scenes:
            return None
        scenes = sorted(all_scenes)
    else:
        return None

    # Limit to 50 scenes for readability
    if len(scenes) > 50:
        scenes = scenes[:50]

    n_scenes = len(scenes)
    n_cats = len(categories)
    data = np.ones((n_scenes, n_cats))  # 1 = OK (green)

    for ci, cat in enumerate(categories):
        for si, scene_id in enumerate(scenes):
            if scene_id in missing_sets[cat]:
                data[si, ci] = 0  # 0 = missing (red)

    fig, ax = plt.subplots(
        figsize=(max(4, n_cats * 1.2), max(3, n_scenes * 0.3))
    )
    cmap = plt.cm.colors.ListedColormap(["#ff6b6b", "#51cf66"])
    ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_scenes))
    ax.set_yticklabels(scenes, fontsize=6)
    ax.set_title(
        f"{ds.upper()} / {split_name} — Scenes with Issues "
        f"({n_scenes} of {sr['n_scenes']} total)",
        fontsize=10
    )
    fig.tight_layout()
    return fig


def _make_missing_bar_chart(results: List[Dict]):
    """Create a stacked bar chart showing missing components per split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    rgb_missing = []
    depth_missing = []
    gt_missing = []

    for r in results:
        ds = r["dataset"]
        for split_name, sr in r["splits"].items():
            if "error" in sr:
                continue
            labels.append(f"{ds[:4]}/{split_name}")
            tf = sr.get("total_gt_frames", 0)
            if tf == 0:
                rgb_missing.append(0)
                depth_missing.append(0)
                gt_missing.append(0)
                continue

            if ds == "scannetpp":
                rgb_missing.append(len(sr.get("frames_missing_rgb", [])))
                depth_missing.append(0)  # depth is in H5, not separate files
                gt_missing.append(len(sr.get("scenes_missing_gt_h5", [])))
            else:
                rgb_missing.append(len(sr.get("frames_missing_rgb", [])))
                depth_missing.append(len(sr.get("frames_missing_depth", [])))
                gt_missing.append(len(sr.get("scenes_missing_gt", [])))

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar(x - width, rgb_missing, width, label="Frames missing RGB", color="steelblue")
    ax.bar(x, depth_missing, width, label="Frames missing depth", color="coral")
    ax.bar(x + width, gt_missing, width, label="Scenes missing GT", color="gray")
    ax.set_xlabel("Dataset / Split")
    ax.set_ylabel("Count")
    ax.set_title("Missing Data Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig
