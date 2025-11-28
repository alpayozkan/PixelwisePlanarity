#!/usr/bin/env python3
"""
Evaluation Runner Script

Runs planarity/segmentation evaluation on ScanNet++ dataset.

Usage:
    python run_evaluation.py --method moge --model_path /path/to/model.pt --dataset_root /path/to/scannetpp
"""
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
import torch
import os
import random

from torch.utils.data import DataLoader

from planamono.shared.datasets import ScanNetPPPlaneDataset
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.evaluation.quantitative.evaluator import evaluate_planarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate planarity/segmentation metrics on ScanNet++"
    )

    # === Method selection ===
    parser.add_argument("--method", type=str, required=True,
                        choices=["moge", "planercnn", "zeroplane", "gt", "monoplane"],
                        help="Which method to evaluate")

    # === Model args (for moge/monoplane) ===
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained MoGe checkpoint (.pt)")
    parser.add_argument("--model_size", type=str, default="large",
                        choices=["small", "middle", "large"],
                        help="MoGe model size")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for MoGe weights (or set MOGE_CACHE_DIR)")
    parser.add_argument("--device", type=str, default="cuda")

    # === Dataset paths ===
    parser.add_argument("--rgb_root", type=str, required=True,
                        help="Root directory of ScanNet++ RGB images")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory containing plane_ours_gt, semantic_gt, etc.")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory containing split files (default: dataset_root/splits)")

    # === Evaluation settings ===
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_scenes", type=int, default=5,
                        help="Maximum number of scenes to evaluate")
    parser.add_argument("--res_h", type=int, default=480)
    parser.add_argument("--res_w", type=int, default=640)

    # === Output ===
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory to save evaluation results")

    # === DataLoader settings ===
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set split_dir default
    if args.split_dir is None:
        args.split_dir = os.path.join(args.dataset_root, "splits")

    print("=" * 60)
    print("Evaluation Runner")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"RGB root: {args.rgb_root}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Split: {args.split}")
    print(f"Max scenes: {args.max_scenes}")
    print(f"Resolution: {args.res_h}x{args.res_w}")
    print("-" * 60)

    # === Setup seeds ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # === Load dataset ===
    print("[INFO] Loading dataset...")
    val_dataset = ScanNetPPPlaneDataset(
        rgb_root=args.rgb_root,
        plane_label_root=os.path.join(args.dataset_root, "plane_ours_gt"),
        sem_label_root=os.path.join(args.dataset_root, "semantic_gt"),
        depth_label_root=os.path.join(args.dataset_root, "depth_gt_rendered"),
        split_txt_dir=args.split_dir,
        split=args.split,
        max_scenes=args.max_scenes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    print(f"[INFO] Loaded {len(val_dataset)} samples")

    # === Load model if needed ===
    inference_model = None

    if args.method in ["moge", "monoplane"]:
        if args.model_path is None:
            print("[ERROR] --model_path required for moge/monoplane methods")
            sys.exit(1)

        print(f"[INFO] Loading MoGe model: {args.model_path}")
        inference_model = MoGePlanarityInference(
            args.model_path,
            model_size=args.model_size,
            device=args.device,
            cache_dir=args.cache_dir
        )

        # Optimizations
        inference_model.model.encoder.use_memory_efficient_attention = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        inference_model.model = inference_model.model.half()
        if hasattr(inference_model.model.encoder, 'enable_pytorch_native_sdpa'):
            inference_model.model.encoder.enable_pytorch_native_sdpa()

        print("[INFO] Model loaded")

    # === Run evaluation ===
    print("-" * 60)
    print(f"[INFO] Running evaluation with method: {args.method}")

    os.makedirs(args.save_dir, exist_ok=True)

    df = evaluate_planarity(
        val_loader,
        inference_model,
        tag=args.method,
        img_res=(args.res_h, args.res_w)
    )

    # === Save results ===
    csv_path = os.path.join(args.save_dir, f"eval_{args.method}_{args.max_scenes}.csv")
    df.to_csv(csv_path, index=False)

    print("=" * 60)
    print(f"[SUCCESS] Evaluation complete")
    print(f"[SAVED] {csv_path}")

    # Print summary
    print("-" * 60)
    print("Summary:")
    print(df.describe())
