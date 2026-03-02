#!/usr/bin/env python3
"""
Benchmark per-frame inference timing for all methods.

Runs on a single ScanNet++ scene (100 frames) and reports mean ms and FPS
for each method and its components.

Methods:
  1. Ours (MoGe planarity + plan2seg)
  2. DAv2 (DAv2 depth + depth_to_normal + our planarity + plan2seg)
  3. Metric3D (Metric3D depth+normals + our planarity + plan2seg)
  4. Pseudo-mono (MoGe depth + RANSAC)

Usage:
    python planamono/evaluation/benchmark_timing.py \
        --checkpoint /cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch1.pt \
        --dav2_checkpoint /cluster/scratch/ayavuz/checkpoints/depth_anything_v2_vitl.pth
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.shared.utils.depth_normal import depth_to_normal_remi
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative


def load_scannetpp_frames(rgb_root, gt_root, scene_id, splits_root, n_frames=100, out_h=480, out_w=640):
    """Load n_frames from a ScanNet++ scene."""
    import json
    gt_h5 = os.path.join(gt_root, scene_id, "rendered.h5")
    pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")
    with open(pose_file) as f:
        pose_data = json.load(f)
    with h5py.File(gt_h5, "r") as hf:
        all_fids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                    for fid in hf["frame_ids"][:]]
    rgbs, Ks = [], []
    for fid in all_fids:
        if len(rgbs) >= n_frames:
            break
        rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
        if not os.path.exists(rgb_path) or fid not in pose_data:
            continue
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        native_h, native_w = rgb.shape[:2]
        K = np.array(pose_data[fid]["intrinsic"], dtype=np.float32)
        K[0, :] *= out_w / native_w
        K[1, :] *= out_h / native_h
        rgbs.append(cv2.resize(rgb, (out_w, out_h)))
        Ks.append(K)
    return rgbs, Ks


def preprocess_for_moge(rgb_uint8, device):
    resized = cv2.resize(rgb_uint8, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)


class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        torch.cuda.synchronize()
        self._start = time.perf_counter()
        self._name = name

    def stop(self):
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._start) * 1000  # ms
        if self._name not in self.times:
            self.times[self._name] = []
        self.times[self._name].append(elapsed)

    def report(self, title):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        total_mean = 0
        for name, vals in self.times.items():
            vals = vals[5:]  # skip warmup
            if not vals:
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            total_mean += mean
            print(f"  {name:30s}: {mean:7.1f} ms  (std {std:.1f})")
        fps = 1000.0 / total_mean if total_mean > 0 else 0
        print(f"  {'TOTAL':30s}: {total_mean:7.1f} ms  ({fps:.1f} FPS)")
        return total_mean


def benchmark_ours(moge_model, rgbs, Ks, device, args):
    """Our method: MoGe planarity + depth + normals + plan2seg"""
    timer = Timer()
    import torch.nn.functional as torchF
    for rgb in tqdm(rgbs, desc="Ours (MoGe + plan2seg)"):
        # MoGe forward
        timer.start("moge_forward")
        tensor = preprocess_for_moge(rgb, device)
        with torch.no_grad():
            output = moge_model.model.forward(tensor.unsqueeze(0), num_tokens=1600)
        timer.stop()

        # Extract planarity + normals + depth
        timer.start("moge_postprocess")
        planarity = output['planarity'][0]
        planarity = torchF.interpolate(planarity[None, None], (480, 640), mode='bilinear', align_corners=False)[0, 0]
        planarity_np = planarity.cpu().numpy().astype(np.float32)

        normal = output['normal'][0]
        normal = torchF.interpolate(normal.permute(2, 0, 1)[None], (480, 640), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        norm_mag = torch.norm(normal, dim=2, keepdim=True).clamp(min=1e-8)
        normal = normal / norm_mag
        normals_np = normal.cpu().numpy().astype(np.float32)

        depth = output['depth'][0]
        depth = torchF.interpolate(depth[None, None], (480, 640), mode='bilinear', align_corners=False)[0, 0]
        depth_np = depth.cpu().numpy().astype(np.float32)
        timer.stop()

        # plan2seg
        timer.start("plan2seg")
        mask = (planarity_np > args.planarity_threshold).astype(np.uint8)
        compute_vectorized_planar_segments_v5_relative(
            mask, normals_np, depth_np,
            normal_threshold_rad=args.normal_threshold_rad,
            depth_threshold=args.depth_threshold,
            device=device,
        )
        timer.stop()

    return timer.report("Ours (MoGe + plan2seg)")


def benchmark_dav2(moge_model, dav2_model, rgbs, Ks, device, args):
    """DAv2: DAv2 depth + depth_to_normal + our planarity + plan2seg"""
    timer = Timer()
    import torch.nn.functional as torchF
    for i, rgb in enumerate(tqdm(rgbs, desc="DAv2 + planarity")):
        # MoGe planarity
        timer.start("moge_planarity")
        tensor = preprocess_for_moge(rgb, device)
        with torch.no_grad():
            output = moge_model.model.forward(tensor.unsqueeze(0), num_tokens=1600)
        planarity = output['planarity'][0]
        planarity = torchF.interpolate(planarity[None, None], (480, 640), mode='bilinear', align_corners=False)[0, 0]
        planarity_np = planarity.cpu().numpy().astype(np.float32)
        timer.stop()

        # DAv2 depth
        timer.start("dav2_depth")
        raw_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = dav2_model.infer_image(raw_bgr).astype(np.float32)
        if depth.shape != (480, 640):
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
        timer.stop()

        # Depth to normals
        timer.start("depth_to_normal")
        K = Ks[i]
        normals = depth_to_normal_remi(depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        timer.stop()

        # plan2seg
        timer.start("plan2seg")
        mask = (planarity_np > args.planarity_threshold).astype(np.uint8)
        compute_vectorized_planar_segments_v5_relative(
            mask, normals, depth,
            normal_threshold_rad=args.normal_threshold_rad,
            depth_threshold=args.depth_threshold,
            device=device,
        )
        timer.stop()

    return timer.report("DAv2 + Our Planarity + plan2seg")


def benchmark_metric3d(moge_model, metric3d_model, rgbs, Ks, device, args):
    """Metric3D: depth+normals + our planarity + plan2seg"""
    timer = Timer()
    import torch.nn.functional as torchF

    # Import metric3d_infer from our export script
    from planamono.external.export_metric3d import metric3d_infer

    for i, rgb in enumerate(tqdm(rgbs, desc="Metric3D + planarity")):
        # MoGe planarity
        timer.start("moge_planarity")
        tensor = preprocess_for_moge(rgb, device)
        with torch.no_grad():
            output = moge_model.model.forward(tensor.unsqueeze(0), num_tokens=1600)
        planarity = output['planarity'][0]
        planarity = torchF.interpolate(planarity[None, None], (480, 640), mode='bilinear', align_corners=False)[0, 0]
        planarity_np = planarity.cpu().numpy().astype(np.float32)
        timer.stop()

        # Metric3D depth + normals
        timer.start("metric3d_infer")
        depth, normals = metric3d_infer(metric3d_model, rgb, Ks[i], device)
        timer.stop()

        # plan2seg
        timer.start("plan2seg")
        mask = (planarity_np > args.planarity_threshold).astype(np.uint8)
        compute_vectorized_planar_segments_v5_relative(
            mask, normals, depth,
            normal_threshold_rad=args.normal_threshold_rad,
            depth_threshold=args.depth_threshold,
            device=device,
        )
        timer.stop()

    return timer.report("Metric3D + Our Planarity + plan2seg")


def benchmark_pseudo_mono(moge_model, rgbs, device, args):
    """Pseudo-mono: MoGe depth + sequential RANSAC"""
    timer = Timer()
    from planamono.evaluation.run_pseudo_mono_export import pseudo_mono_infer

    for rgb in tqdm(rgbs, desc="Pseudo-mono (RANSAC)"):
        timer.start("pseudo_mono_total")
        pseudo_mono_infer(moge_model.model, rgb, device=device)
        timer.stop()

    return timer.report("Pseudo-mono (MoGe + RANSAC)")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark per-frame inference timing")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dav2_checkpoint", type=str,
                   default="/cluster/scratch/ayavuz/checkpoints/depth_anything_v2_vitl.pth")
    p.add_argument("--dav2_encoder", type=str, default="vitl")
    p.add_argument("--dav2_repo", type=str, default="~/Depth-Anything-V2")
    p.add_argument("--metric3d_model", type=str, default="metric3d_vit_large")
    p.add_argument("--metric3d_repo", type=str, default="~/Metric3D")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_frames", type=int, default=100)
    p.add_argument("--scene_id", type=str, default=None,
                   help="ScanNet++ scene ID (default: first available test scene)")
    p.add_argument("--planarity_threshold", type=float, default=0.5)
    p.add_argument("--normal_threshold_rad", type=float, default=0.15)
    p.add_argument("--depth_threshold", type=float, default=0.1)
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--splits_root", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "splits"))
    p.add_argument("--skip_metric3d", action="store_true")
    p.add_argument("--skip_dav2", action="store_true")
    p.add_argument("--skip_pseudo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Pick scene
    if args.scene_id is None:
        split_file = os.path.join(args.splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
        with open(split_file) as f:
            args.scene_id = f.readline().strip()
    print(f"Scene: {args.scene_id}")
    print(f"Frames: {args.n_frames}")

    # Load frames
    print("Loading frames...")
    rgbs, Ks = load_scannetpp_frames(
        args.scannetpp_rgb_root, args.scannetpp_gt_root,
        args.scene_id, args.splits_root, args.n_frames)
    print(f"Loaded {len(rgbs)} frames")

    # Load MoGe (shared across methods)
    print("Loading MoGe...")
    moge_wrapper = MoGePlanarityInference(args.checkpoint, device=args.device)

    # 1. Ours
    t_ours = benchmark_ours(moge_wrapper, rgbs, Ks, args.device, args)

    # 2. DAv2
    if not args.skip_dav2:
        print("\nLoading DAv2...")
        dav2_repo = os.path.expanduser(args.dav2_repo)
        sys.path.insert(0, dav2_repo)
        from depth_anything_v2.dpt import DepthAnythingV2
        DAV2_CONFIGS = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        dav2_model = DepthAnythingV2(**DAV2_CONFIGS[args.dav2_encoder])
        dav2_model.load_state_dict(torch.load(args.dav2_checkpoint, map_location='cpu'))
        dav2_model = dav2_model.to(args.device).eval()
        t_dav2 = benchmark_dav2(moge_wrapper, dav2_model, rgbs, Ks, args.device, args)

    # 3. Metric3D
    if not args.skip_metric3d:
        print("\nLoading Metric3D...")
        metric3d_repo = os.path.expanduser(args.metric3d_repo)
        metric3d_model = torch.hub.load(metric3d_repo, args.metric3d_model,
                                         pretrain=True, source='local')
        metric3d_model = metric3d_model.to(args.device).eval()
        t_m3d = benchmark_metric3d(moge_wrapper, metric3d_model, rgbs, Ks, args.device, args)

    # 4. Pseudo-mono
    if not args.skip_pseudo:
        t_pseudo = benchmark_pseudo_mono(moge_wrapper, rgbs, args.device, args)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY (mean ms / FPS)")
    print(f"{'=' * 60}")
    print(f"  {'Ours':30s}: {t_ours:7.1f} ms  ({1000/t_ours:.1f} FPS)")
    if not args.skip_dav2:
        print(f"  {'DAv2 + planarity':30s}: {t_dav2:7.1f} ms  ({1000/t_dav2:.1f} FPS)")
    if not args.skip_metric3d:
        print(f"  {'Metric3D + planarity':30s}: {t_m3d:7.1f} ms  ({1000/t_m3d:.1f} FPS)")
    if not args.skip_pseudo:
        print(f"  {'Pseudo-mono (RANSAC)':30s}: {t_pseudo:7.1f} ms  ({1000/t_pseudo:.1f} FPS)")


if __name__ == "__main__":
    main()
