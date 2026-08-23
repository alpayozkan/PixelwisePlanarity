#!/usr/bin/env python3
"""
Hypersim Scene Runner for Plane Extraction

Orchestrates the Hypersim plane extraction pipeline for a single scene.
Loads configuration from YAML, validates inputs, and runs the extraction.

Usage:
    python scene_runner.py ai_001_001 --config ../configs/hypersim_default.yml
    python scene_runner.py ai_003_007 --config ../configs/hypersim_default.yml --input_root /path/to/hypersim --output_root /path/to/output
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the Hypersim plane extraction pipeline
from pxwplanar.gt_creation.hypersim.plane_extraction import run


def cast_config_types(cfg):
    """
    Ensure all parameters have correct types for compatibility with argparse.Namespace.

    Args:
        cfg: Dictionary loaded from YAML config

    Returns:
        Dictionary with correctly typed values
    """

    # --- Integer parameters ---
    int_keys = [
        "palette_seed", "progress", "jobs",
        "em_max_iters", "min_faces_patch",
        "em_max_iters_small", "min_faces_patch_small",
        "sat_rounds", "irls_max_iters",
        "last_enable", "last_rg_iters",
        "large_split_enable", "large_split_recursive",
        "large_split_verts", "large_split_max_parts", "large_split_min_faces",
        "big_ransac_enable", "big_ransac_max_iters", "big_min_faces_patch",
        "post_ransac_enable", "post_ransac_max_iters", "post_min_faces_patch",
        "post_ransac_max_planes_per_comp",
        "small_claim_enable", "small_claim_require_adjacent", "small_claim_refit",
        "normalize_group_names", "h5_palette_seed", "generate_legend_if_missing",
        "save_area_hist", "area_hist_bins"
    ]

    # --- Float parameters ---
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
        "last_unlabeled_ratio", "last_steal_factor",
        "big_ransac_dist_m", "big_ransac_normal_deg", "big_ransac_min_inlier_frac",
        "big_min_area_patch", "big_p95_final_max", "big_inlier_frac_min",
        "big_normal_p95_deg_max", "big_thickness_max_mul", "big_min_width_m",
        "big_fill_frac_min", "merge_big_theta_deg", "merge_big_dist_m",
        "small_claim_normal_deg", "small_claim_dist_m",
        "post_ransac_dist_m", "post_ransac_normal_deg", "post_ransac_min_inlier_frac",
        "post_min_area_m2"
    ]

    # --- String parameters ---
    str_keys = [
        "backend", "rg_gate_mode", "gate_mode",
        "policy_single_plane_labels", "large_split_mode",
        "out_json_name", "out_ply_name", "area_hist_png_name"
    ]

    # --- List parameters ---
    list_keys = [
        "policy_skip_labels"
    ]

    # --- Apply casting ---
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


def build_args(scene_id, config_path, input_root=None, output_root=None):
    """
    Build argparse.Namespace for a single Hypersim scene.

    Args:
        scene_id: Scene ID (e.g., 'ai_003_007')
        config_path: Path to YAML configuration file
        input_root: Optional override for input data root
        output_root: Optional override for output data root

    Returns:
        argparse.Namespace with all configuration parameters
    """

    # Load YAML configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = cast_config_types(cfg)

    # Base paths from YAML (can be overridden by command line)
    if input_root is None:
        input_root = cfg.pop("input_root")
    else:
        cfg.pop("input_root", None)

    if output_root is None:
        output_root = cfg.pop("output_root")
    else:
        cfg.pop("output_root", None)

    # Scene-specific paths (out_dir is created after input validation below,
    # so a placeholder input_root fails with a clear message)
    scene_dir = os.path.join(input_root, scene_id)
    mesh_dir = os.path.join(scene_dir, "_detail", "mesh")
    out_dir = os.path.join(output_root, scene_id)

    # Expected Hypersim HDF5 files
    h5_verts = os.path.join(mesh_dir, "mesh_vertices.hdf5")
    h5_faces = os.path.join(mesh_dir, "mesh_faces_vi.hdf5")
    h5_face_groups = os.path.join(mesh_dir, "mesh_faces_gi.hdf5")
    groups_csv = os.path.join(mesh_dir, "metadata_groups.csv")
    legend_csv = os.path.join(out_dir, "semantic_legend.csv")

    # Validate required files
    required_files = [h5_verts, h5_faces, h5_face_groups, groups_csv]
    missing_files = [f for f in required_files if not os.path.isfile(f)]

    if missing_files:
        print(f"[ERROR] Missing required input files for scene {scene_id}:")
        for mf in missing_files:
            print(f"   - {mf}")
        print(f"\n[INFO] Expected mesh directory: {mesh_dir}")
        print(f"[INFO] Required files:")
        print(f"   - mesh_vertices.hdf5")
        print(f"   - mesh_faces_vi.hdf5")
        print(f"   - mesh_faces_gi.hdf5")
        print(f"   - metadata_groups.csv")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Set mesh to empty string (Hypersim uses HDF5, not PLY)
    cfg["mesh"] = ""

    # Inject scene-specific I/O into config
    cfg.update({
        "h5_verts": h5_verts,
        "h5_faces": h5_faces,
        "h5_face_groups": h5_face_groups,
        "groups_csv": groups_csv,
        "legend_csv": legend_csv,
        "out": out_dir
    })

    return argparse.Namespace(**cfg)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Hypersim plane extraction pipeline for a single scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single scene with default config
  python scene_runner.py ai_001_001

  # Process with custom config
  python scene_runner.py ai_003_007 --config ../configs/hypersim_custom.yml

  # Override data paths
  python scene_runner.py ai_001_001 \\
      --input_root /data/hypersim \\
      --output_root /output/planes
        """
    )

    parser.add_argument(
        "scene_id",
        type=str,
        help="Scene ID (e.g., ai_001_001, ai_003_007)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/hypersim_default.yml",
        help="Path to YAML configuration file (default: ../configs/hypersim_default.yml)"
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="Override input data root directory (default: from config)"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override output data root directory (default: from config)"
    )

    args_cli = parser.parse_args()

    scene_id = args_cli.scene_id
    config_path = args_cli.config

    # Validate config file exists
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"Hypersim Plane Extraction - Scene: {scene_id}")
    print("=" * 80)
    print(f"Config: {config_path}")

    # Build configuration
    args = build_args(
        scene_id,
        config_path,
        input_root=args_cli.input_root,
        output_root=args_cli.output_root
    )

    print(f"Input:  {args.h5_verts.replace('/mesh_vertices.hdf5', '')}")
    print(f"Output: {args.out}")
    print("-" * 80)

    # Run plane extraction
    try:
        run(args)
        print("-" * 80)
        print(f"[SUCCESS] Finished processing scene: {scene_id}")
        print("=" * 80)
    except Exception as e:
        print("-" * 80)
        print(f"[ERROR] Failed to process scene: {scene_id}")
        print(f"Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
