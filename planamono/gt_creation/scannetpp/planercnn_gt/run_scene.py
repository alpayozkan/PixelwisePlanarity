"""
Convenience wrapper: runs fit_planes.py then render_planes.py for a scene.

Usage:
    python run_scene.py <scene_id>
    python run_scene.py <scene_id> --config planercnn_default.yml
    python run_scene.py <scene_id> --mesh_root /path/to/mesh --output_root /path/to/output
"""

import argparse
import subprocess
import sys
import os
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "planercnn_default.yml")

DEFAULT_MESH_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp_planercnn"
DEFAULT_OUTPUT_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn"


def main():
    parser = argparse.ArgumentParser(description="Run PlaneRCNN GT pipeline (fit + render)")
    parser.add_argument("scene_id", type=str, help="ScanNet++ scene ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: use hardcoded parameters)")
    parser.add_argument("--mesh_root", type=str, default=None,
                        help="Override mesh root (overrides config)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Override output root (overrides config)")
    parser.add_argument("--frame_skip", type=int, default=None,
                        help="Override frame skip (overrides config)")
    args = parser.parse_args()

    # Resolve paths: CLI > config > defaults
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        mesh_root = args.mesh_root or cfg.get("mesh_root", DEFAULT_MESH_ROOT)
        output_root = args.output_root or cfg.get("output_root", DEFAULT_OUTPUT_ROOT)
    else:
        mesh_root = args.mesh_root or DEFAULT_MESH_ROOT
        output_root = args.output_root or DEFAULT_OUTPUT_ROOT

    # Stage 1: fit_planes.py
    print(f"{'='*60}")
    print(f"Stage 1: Fitting planes for {args.scene_id}")
    print(f"{'='*60}")
    fit_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "fit_planes.py"),
        args.scene_id, "--output_root", mesh_root,
    ]
    if args.config is not None:
        fit_cmd.extend(["--config", args.config])
    ret = subprocess.run(fit_cmd, check=False)
    if ret.returncode != 0:
        print(f"[ERR] fit_planes.py failed with code {ret.returncode}")
        sys.exit(ret.returncode)

    # Stage 2: render_planes.py
    print(f"\n{'='*60}")
    print(f"Stage 2: Rendering planes for {args.scene_id}")
    print(f"{'='*60}")
    render_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "render_planes.py"),
        args.scene_id,
        "--mesh_root", mesh_root,
        "--output_root", output_root,
    ]
    if args.config is not None:
        render_cmd.extend(["--config", args.config])
    if args.frame_skip is not None:
        render_cmd.extend(["--frame_skip", str(args.frame_skip)])
    ret = subprocess.run(render_cmd, check=False)
    if ret.returncode != 0:
        print(f"[ERR] render_planes.py failed with code {ret.returncode}")
        sys.exit(ret.returncode)

    print(f"\n{'='*60}")
    print(f"[DONE] PlaneRCNN GT pipeline complete for {args.scene_id}")
    print(f"  Mesh: {mesh_root}/{args.scene_id}/")
    print(f"  H5:   {output_root}/{args.scene_id}/rendered.h5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
