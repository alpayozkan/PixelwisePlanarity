"""
Generate dataset statistics CSV for ScanNet++ and Hypersim datasets.

Uses the existing PyTorch dataset classes to count scenes and frames.

Usage:
    python generate_dataset_stats.py
    python generate_dataset_stats.py --output stats.csv
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from planamono.paths import (
    scannetpp_path,
    scannetpp_rend_plane_path,
    hypersim_path,
    hypersim_rend_plane_path,
    repo_path,
)
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.datasets.hypersim import HypersimPlanarityDataset


# Paths
SPLITS_DIR = Path(__file__).parent
SCANNETPP_SPLITS = SPLITS_DIR / "scannetpp"
HYPERSIM_CSV = Path("/cluster/scratch/aoezkan/dataset/Hypersim/splits/metadata_images_split_scene_v1_with_planes.csv")


def get_scannetpp_stats_from_dataset() -> list:
    """Get ScanNet++ statistics by instantiating the dataset classes."""
    rows = []

    for split in ["train", "val"]:
        try:
            # Suppress debug prints by redirecting stdout temporarily
            dataset = ScanNetPPPlaneDataset(
                rgb_root=scannetpp_path,
                plane_label_root=scannetpp_rend_plane_path,
                sem_label_root=scannetpp_rend_plane_path,
                depth_label_root=scannetpp_rend_plane_path,
                split_txt_dir=str(SCANNETPP_SPLITS),
                split=split,
            )
            n_scenes = len(dataset.scene_ids)
            n_frames = len(dataset)
        except Exception as e:
            print(f"[ERROR] Failed to load ScanNet++ {split}: {e}")
            n_scenes = 0
            n_frames = 0

        rows.append({
            "Dataset": "ScanNet++",
            "Split": split,
            "Scenes": n_scenes,
            "Frames": n_frames,
        })

    # Total
    total_scenes = sum(r["Scenes"] for r in rows)
    total_frames = sum(r["Frames"] for r in rows)
    rows.append({
        "Dataset": "ScanNet++",
        "Split": "total",
        "Scenes": total_scenes,
        "Frames": total_frames,
    })

    return rows


def get_hypersim_stats_from_dataset() -> list:
    """Get Hypersim statistics by instantiating the dataset classes."""
    rows = []

    if not HYPERSIM_CSV.exists():
        print(f"[WARN] Hypersim CSV not found: {HYPERSIM_CSV}")
        print("[WARN] Falling back to counting from CSV directly...")
        return get_hypersim_stats_from_csv()

    for split in ["train", "val", "test"]:
        try:
            dataset = HypersimPlanarityDataset(
                root_dir=hypersim_path,
                plane_label_root=hypersim_rend_plane_path,
                filtered_csv_path=str(HYPERSIM_CSV),
                split=split,
            )
            # Count unique scenes
            scene_ids = set(p[0] for p in dataset.valid_pairs)
            n_scenes = len(scene_ids)
            n_frames = len(dataset)
        except Exception as e:
            print(f"[ERROR] Failed to load Hypersim {split}: {e}")
            n_scenes = 0
            n_frames = 0

        rows.append({
            "Dataset": "Hypersim",
            "Split": split,
            "Scenes": n_scenes,
            "Frames": n_frames,
        })

    # Total
    total_scenes = sum(r["Scenes"] for r in rows)
    total_frames = sum(r["Frames"] for r in rows)
    rows.append({
        "Dataset": "Hypersim",
        "Split": "total",
        "Scenes": total_scenes,
        "Frames": total_frames,
    })

    return rows


def get_hypersim_stats_from_csv() -> list:
    """Fallback: get Hypersim stats directly from the CSV file."""
    rows = []

    if not HYPERSIM_CSV.exists():
        print(f"[ERROR] Hypersim CSV not found: {HYPERSIM_CSV}")
        return rows

    df = pd.read_csv(HYPERSIM_CSV)

    for split in ["train", "val", "test"]:
        split_df = df[df["split_partition_name"] == split]
        n_scenes = split_df["scene_name"].nunique()
        n_frames = len(split_df)

        rows.append({
            "Dataset": "Hypersim",
            "Split": split,
            "Scenes": n_scenes,
            "Frames": n_frames,
        })

    # Total
    total_scenes = df["scene_name"].nunique()
    total_frames = len(df)
    rows.append({
        "Dataset": "Hypersim",
        "Split": "total",
        "Scenes": total_scenes,
        "Frames": total_frames,
    })

    return rows


def get_scannetpp_stats_from_splits() -> list:
    """Fallback: get ScanNet++ stats from split files only (no frame counting)."""
    rows = []

    splits = {
        "train": SCANNETPP_SPLITS / "nvs_sem_train_with_planes.txt",
        "val": SCANNETPP_SPLITS / "nvs_sem_val_with_planes.txt",
    }

    for split_name, split_file in splits.items():
        if split_file.exists():
            with open(split_file, "r") as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            n_scenes = len(scene_ids)
        else:
            n_scenes = 0

        rows.append({
            "Dataset": "ScanNet++",
            "Split": split_name,
            "Scenes": n_scenes,
            "Frames": "N/A (use --load-datasets)",
        })

    # Total
    total_scenes = sum(r["Scenes"] for r in rows if isinstance(r["Scenes"], int))
    rows.append({
        "Dataset": "ScanNet++",
        "Split": "total",
        "Scenes": total_scenes,
        "Frames": "N/A",
    })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics CSV")
    parser.add_argument("--output", "-o", type=str, default="dataset_stats.csv",
                        help="Output CSV file path")
    parser.add_argument("--load-datasets", action="store_true",
                        help="Load full datasets to count frames (slower but accurate)")
    args = parser.parse_args()

    all_rows = []

    print("=" * 60)
    print("Collecting ScanNet++ statistics...")
    print("=" * 60)
    if args.load_datasets:
        scannetpp_rows = get_scannetpp_stats_from_dataset()
    else:
        scannetpp_rows = get_scannetpp_stats_from_splits()
    all_rows.extend(scannetpp_rows)

    print("\n" + "=" * 60)
    print("Collecting Hypersim statistics...")
    print("=" * 60)
    if args.load_datasets:
        hypersim_rows = get_hypersim_stats_from_dataset()
    else:
        hypersim_rows = get_hypersim_stats_from_csv()
    all_rows.extend(hypersim_rows)

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Save CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Print table
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
