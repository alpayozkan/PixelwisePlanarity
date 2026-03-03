#!/usr/bin/env python3
"""
Orchestrator for running visualization + PDF pipeline locally or via SLURM.

Calls visualize_and_generate_pdf.py for each dataset, either locally
(subprocess.run) or by submitting SLURM jobs.

Usage:
    # Local execution
    python run_visualizations.py --datasets scannetpp hypersim --methods ours zeroplane --mode local

    # SLURM submission (one job per dataset)
    python run_visualizations.py --datasets scannetpp hypersim --methods ours zeroplane --mode slurm

    # Custom SLURM resources
    python run_visualizations.py --datasets scannetpp --mode slurm --slurm-time 4:00:00 --slurm-mem 32G

    # Dry-run (print commands without executing)
    python run_visualizations.py --datasets scannetpp --mode slurm --dry-run
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_SCRIPT = SCRIPT_DIR / "visualize_and_generate_pdf.py"


def build_vis_command(args, dataset: str) -> list:
    """Build the command list for visualize_and_generate_pdf.py."""
    cmd = [sys.executable, str(VIS_SCRIPT), "--dataset", dataset]

    if args.methods:
        cmd += ["--methods"] + args.methods

    cmd += ["--n-samples", str(args.n_samples)]
    cmd += ["--random-seed", str(args.random_seed)]
    cmd += ["--format", args.format]
    cmd += ["--quality", str(args.quality)]
    cmd += ["--max-width", str(args.max_width)]
    cmd += ["--split", args.split]

    if args.specific_frames:
        cmd += ["--specific-frames", args.specific_frames]

    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]

    if args.max_scenes is not None:
        cmd += ["--max-scenes", str(args.max_scenes)]

    return cmd


def run_local(cmd: list, dataset: str, dry_run: bool = False):
    """Run visualization command locally via subprocess."""
    cmd_str = " ".join(cmd)
    print(f"\n[LOCAL] Running for {dataset}:")
    print(f"  {cmd_str}")

    if dry_run:
        print("  (dry-run, not executing)")
        return

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"  [ERROR] Command failed with return code {result.returncode}")
    else:
        print(f"  [DONE] {dataset} completed successfully")


def run_slurm(cmd: list, dataset: str, args, dry_run: bool = False):
    """Submit visualization command as a SLURM job."""
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    cmd_str = " ".join(cmd)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=vis_{dataset}
#SBATCH --output={logs_dir}/vis_{dataset}_%j.out
#SBATCH --error={logs_dir}/vis_{dataset}_%j.err
#SBATCH --time={args.slurm_time}
#SBATCH --mem={args.slurm_mem}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --gpus={args.slurm_gpus}

set -e

# Activate conda environment
source ~/.bashrc
conda activate planeseg

echo "============================================================"
echo "Visualization: {dataset}"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

cd {SCRIPT_DIR}

{cmd_str}

echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
"""

    print(f"\n[SLURM] Submitting job for {dataset}:")
    print(f"  Command: {cmd_str}")
    print(f"  Time: {args.slurm_time}, Mem: {args.slurm_mem}, "
          f"CPUs: {args.slurm_cpus}, GPUs: {args.slurm_gpus}")

    if dry_run:
        print("  (dry-run, not submitting)")
        print(f"  Script content:\n{sbatch_script}")
        return

    # Write sbatch script to a temp file in logs/
    script_path = logs_dir / f"vis_{dataset}.sh"
    with open(script_path, "w") as f:
        f.write(sbatch_script)
    os.chmod(script_path, 0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"  [SUBMITTED] {result.stdout.strip()}")
        print(f"  Script: {script_path}")
    else:
        print(f"  [ERROR] sbatch failed: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate visualization + PDF pipeline (local or SLURM)"
    )

    # Core args
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=["scannetpp", "hypersim"],
                        help="Datasets to visualize")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["local", "slurm"],
                        help="Execution mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    # Passthrough args for visualize_and_generate_pdf.py
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to visualize (GT always included)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random samples")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--specific-frames", type=str, default=None,
                        help="Comma-separated scene:frame pairs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory override")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum scenes to load")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--format", type=str, choices=["pdf", "pptx", "both", "none"],
                        default="both", help="Output document format")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG quality for PDF/PPTX")
    parser.add_argument("--max-width", type=int, default=1920,
                        help="Max image width for PDF/PPTX")

    # SLURM-specific args
    parser.add_argument("--slurm-time", type=str, default="04:00:00",
                        help="SLURM time limit")
    parser.add_argument("--slurm-mem", type=str, default="32G",
                        help="SLURM memory limit")
    parser.add_argument("--slurm-cpus", type=int, default=8,
                        help="SLURM CPUs per task")
    parser.add_argument("--slurm-gpus", type=int, default=0,
                        help="SLURM GPUs")

    args = parser.parse_args()

    print("============================================================")
    print("Visualization Orchestrator")
    print("============================================================")
    print(f"Datasets:  {args.datasets}")
    print(f"Mode:      {args.mode}")
    print(f"Methods:   {args.methods or '(all)'}")
    print(f"N samples: {args.n_samples}")
    print(f"Seed:      {args.random_seed}")
    print(f"Format:    {args.format}")
    if args.dry_run:
        print("DRY RUN - no commands will be executed")
    print("============================================================")

    for dataset in args.datasets:
        cmd = build_vis_command(args, dataset)

        if args.mode == "local":
            run_local(cmd, dataset, dry_run=args.dry_run)
        elif args.mode == "slurm":
            run_slurm(cmd, dataset, args, dry_run=args.dry_run)

    print("\n[DONE] All datasets processed.")


if __name__ == "__main__":
    main()
