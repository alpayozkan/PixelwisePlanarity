#!/usr/bin/env python3
"""
ScanNet++ scene runner for plane extraction.

This script orchestrates the full plane extraction pipeline for a single ScanNet++ scene.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add repository root to path for absolute imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pxwplanar.gt_creation.scannetpp.plane_extraction import run


def cast_config_types(cfg):
    """
    Ensure all parameters have correct types.
    """
    # Integer parameters
    int_keys = [
        "palette_seed", "progress", "jobs",
        "em_max_iters", "min_faces_patch",
        "em_max_iters_small", "min_faces_patch_small",
        "sat_rounds", "irls_max_iters",
        "last_enable", "last_rg_iters",
        "large_split_enable", "large_split_recursive",
        "large_split_verts", "large_split_max_parts", "large_split_min_faces"
    ]

    # Float parameters
    float_keys = [
        "rg_theta_deg", "rg_dist_m", "rg_dihedral_deg", "rg_refit_every",
        "sweep_normal_deg", "sweep_dist_m", "sweep_frac_vertices",
        "em_min_growth", "min_area_patch", "p95_final_max",
        "inlier_frac_min", "dist_thr", "normal_p95_deg_max",
        "thickness_max_mul", "min_width_m", "fill_frac_min",
        "merge_theta_deg", "merge_dist_m",
        "rg_theta_deg_small", "rg_dist_m_small", "rg_dihedral_deg_small",
        "rg_refit_every_small", "sweep_normal_deg_small", "sweep_dist_m_small",
        "sweep_frac_vertices_small", "em_min_growth_small", "min_area_patch_small",
        "normal_p95_deg_max_small", "thickness_max_small_mul", "min_width_m_small",
        "fill_frac_min_small", "sat_normal_deg", "sat_dist_m", "sat_frac_vertices",
        "sat_normal_p95_deg_max", "sat_thickness_max_mul", "sat_min_width_m",
        "sat_fill_frac_min", "irls_eps", "last_dist_m", "last_normal_deg",
        "last_unlabeled_ratio", "last_steal_factor"
    ]

    # String parameters
    str_keys = [
        "backend", "rg_gate_mode", "gate_mode",
        "policy_single_plane_labels", "large_split_mode"
    ]

    # List parameters
    list_keys = ["policy_skip_labels"]

    # Apply casting
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])
    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])
    for k in str_keys:
        if k in cfg:
            cfg[k] = str(cfg[k])
    for k in list_keys:
        if k in cfg and not isinstance(cfg[k], list):
            if isinstance(cfg[k], str):
                cfg[k] = [s.strip() for s in cfg[k].split(",") if s.strip()]
            else:
                cfg[k] = list(cfg[k])

    return cfg


def build_args(scene_id, config_path, input_root, output_root):
    """
    Build arguments for plane extraction.

    Args:
        scene_id: ScanNet++ scene ID
        config_path: Path to YAML config file
        input_root: Root directory for ScanNet++ dataset
        output_root: Root directory for output
    """
    # Load YAML config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = cast_config_types(cfg)

    # Scene-specific paths
    root_dir = os.path.join(input_root, scene_id, "scans")
    out_dir = os.path.join(output_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    mesh_path = os.path.join(root_dir, "mesh_aligned_0.05_semantic.ply")
    segments_json_path = os.path.join(root_dir, "segments.json")
    segments_anno_path = os.path.join(root_dir, "segments_anno.json")

    # Validate inputs
    for f in [mesh_path, segments_json_path, segments_anno_path]:
        if not os.path.isfile(f):
            print(f"[WARN] Missing file: {f}")
            sys.exit(1)

    # Add scene-specific paths to config
    cfg.update({
        "mesh": mesh_path,
        "segments_json": segments_json_path,
        "segments_anno": segments_anno_path,
        "out": out_dir
    })

    return argparse.Namespace(**cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract planes from ScanNet++ scene")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., '0a5c013435')")
    parser.add_argument("--config", type=str,
                       default="../configs/scannetpp_default.yml",
                       help="Path to config YAML file")
    parser.add_argument("--input_root", type=str,
                       default="/path/to/scannetpp/data",
                       help="Root directory of ScanNet++ dataset")
    parser.add_argument("--output_root", type=str,
                       default="/path/to/output",
                       help="Root directory for output")

    args_cli = parser.parse_args()

    print(f"[INFO] Processing scene: {args_cli.scene_id}")
    args = build_args(args_cli.scene_id, args_cli.config,
                     args_cli.input_root, args_cli.output_root)
    run(args)
    print(f"[DONE] Finished scene: {args_cli.scene_id}")
