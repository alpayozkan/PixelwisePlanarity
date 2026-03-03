#!/usr/bin/env python3
"""
Benchmark vanilla MoGe v2 (without planarity head) inference timing.

Compares:
  1. Vanilla MoGe v2 infer() — original model, depth+normals only
  2. Our MoGe infer() — with planarity head

Usage:
    python planamono/evaluation/benchmark_vanilla_moge.py --checkpoint /cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch1.pt
"""

import os
import sys
import argparse
import time
import random
import json
import numpy as np
import torch
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.moge.moge.model.v2 import MoGeModel


def preprocess_for_moge(rgb_uint8, device):
    resized = cv2.resize(rgb_uint8, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


def load_scannetpp_frames(rgb_root, gt_root, scene_id, n_frames=100, out_h=480, out_w=640):
    gt_h5 = os.path.join(gt_root, scene_id, "rendered.h5")
    pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")
    with open(pose_file) as f:
        pose_data = json.load(f)
    with h5py.File(gt_h5, "r") as hf:
        all_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                    for fid in hf["frame_ids"][:]]
    rgbs = []
    for fid in all_fids:
        if len(rgbs) >= n_frames:
            break
        rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
        if not os.path.exists(rgb_path) or fid not in pose_data:
            continue
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        rgbs.append(cv2.resize(rgb, (out_w, out_h)))
    return rgbs


def load_multi_scene_frames(rgb_root, gt_root, splits_root, n_scenes=10, frames_per_scene=100):
    random.seed(42)
    split_file = os.path.join(splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
    with open(split_file) as f:
        all_scenes = [l.strip() for l in f if l.strip()]

    valid_scenes = []
    for sid in all_scenes:
        gt_h5 = os.path.join(gt_root, sid, "rendered.h5")
        rgb_dir = os.path.join(rgb_root, sid, "iphone", "rgb")
        if os.path.exists(gt_h5) and os.path.isdir(rgb_dir):
            valid_scenes.append(sid)

    n_scenes = min(n_scenes, len(valid_scenes))
    selected = random.sample(valid_scenes, n_scenes)
    selected.sort()

    all_rgbs = []
    for sid in selected:
        rgbs = load_scannetpp_frames(rgb_root, gt_root, sid, frames_per_scene)
        print(f"  {sid}: {len(rgbs)} frames")
        all_rgbs.extend(rgbs)

    return all_rgbs, selected


def benchmark_model(model, rgbs, device, label):
    times = []
    for rgb in tqdm(rgbs, desc=label):
        tensor = preprocess_for_moge(rgb, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model.infer(tensor, num_tokens=1600)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = times[5:]
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / mean_ms

    print(f"\n{'=' * 60}")
    print(f"  {label} ({len(times)} frames)")
    print(f"{'=' * 60}")
    print(f"  Mean: {mean_ms:.1f} ms  (std {std_ms:.1f})")
    print(f"  FPS:  {fps:.1f}")
    return mean_ms, fps


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark vanilla MoGe vs ours")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Our MoGe checkpoint (with planarity head)")
    p.add_argument("--vanilla_model", type=str, default="Ruicheng/moge-2-vitl-normal",
                   help="Vanilla MoGe model name or path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_scenes", type=int, default=10)
    p.add_argument("--frames_per_scene", type=int, default=100)
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--splits_root", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "splits"))
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Scenes: {args.n_scenes}")
    print(f"Frames per scene: {args.frames_per_scene}")

    print("Loading frames...")
    rgbs, selected = load_multi_scene_frames(
        args.scannetpp_rgb_root, args.scannetpp_gt_root,
        args.splits_root, args.n_scenes, args.frames_per_scene)
    print(f"Loaded {len(rgbs)} frames from {len(selected)} scenes")

    # 1. Vanilla MoGe v2
    print("\nLoading vanilla MoGe v2...")
    vanilla_model = MoGeModel.from_pretrained(args.vanilla_model).to(args.device).eval()
    m_vanilla, f_vanilla = benchmark_model(vanilla_model, rgbs, args.device, "Vanilla MoGe v2")
    del vanilla_model
    torch.cuda.empty_cache()

    # 2. Ours (MoGe + planarity head)
    print("\nLoading our MoGe (with planarity)...")
    from planamono.inference.planarity.moge_inference import MoGePlanarityInference
    our_wrapper = MoGePlanarityInference(args.checkpoint, device=args.device)
    our_model = our_wrapper.model
    m_ours, f_ours = benchmark_model(our_model, rgbs, args.device, "Ours (MoGe + planarity)")
    del our_model, our_wrapper
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Vanilla MoGe v2':30s}: {m_vanilla:7.1f} ms  ({f_vanilla:.1f} FPS)")
    print(f"  {'Ours (+ planarity head)':30s}: {m_ours:7.1f} ms  ({f_ours:.1f} FPS)")
    print(f"  {'Overhead':30s}: {m_ours - m_vanilla:+.1f} ms  ({(m_ours/m_vanilla - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
