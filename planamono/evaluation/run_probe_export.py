#!/usr/bin/env python3
"""
Run probe model (conv probe on frozen MoGe normal_head features) on test sets
and save plane segmentation results as H5.

Pipeline: RGB -> frozen MoGe (depth/normals) + probe (planarity) -> plan2seg -> plane labels -> H5

Supports 4 datasets: scannetpp, hypersim, vkitti2, synthia.

Usage:
    python planamono/evaluation/run_probe_export.py --dataset scannetpp

    # All datasets:
    python planamono/evaluation/run_probe_export.py --dataset all

    # Custom checkpoint:
    python planamono/evaluation/run_probe_export.py \
        --probe_checkpoint /path/to/probe.pt --dataset scannetpp
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.moge.moge.model.v2 import MoGeModel, normalized_view_plane_uv
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative

# Re-import scene iterators from pseudo_mono export
from planamono.evaluation.run_pseudo_mono_export import (
    iter_scannetpp_scenes,
    iter_hypersim_scenes,
    iter_vkitti2_scenes,
    iter_synthia_scenes,
    DATASET_ITERS,
)


# ============================================================
# PlanarityProbe (inline copy to avoid circular imports)
# ============================================================

class PlanarityProbe(nn.Module):
    """Conv layers that tap into normal_head's penultimate feature map and predict planarity."""
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Model setup
# ============================================================

def detect_tap_level(model):
    """Detect tap_level and tap_dim from the normal_head architecture."""
    head = model.normal_head
    res_block_dims = []
    for ib in head.input_blocks:
        if isinstance(ib, nn.Conv2d):
            res_block_dims.append(ib.out_channels)
        elif isinstance(ib, nn.Identity):
            idx = len(res_block_dims)
            ob = head.output_blocks[idx]
            if isinstance(ob, nn.Conv2d):
                res_block_dims.append(ob.in_channels)
            else:
                res_block_dims.append(None)
        else:
            res_block_dims.append(None)
    tap_level = len(res_block_dims) - 2
    tap_dim = res_block_dims[tap_level]
    print(f"Normal head res_block dims: {res_block_dims}")
    print(f"Tapping level {tap_level} with {tap_dim} channels")
    return tap_level, tap_dim


def load_probe_model(base_model_path, probe_checkpoint_path, probe_hidden, device):
    """Load frozen MoGe base + trained probe."""
    print(f"Loading base model: {base_model_path}")
    model = MoGeModel.from_pretrained(base_model_path).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    tap_level, tap_dim = detect_tap_level(model)

    probe = PlanarityProbe(in_channels=tap_dim, hidden_dim=probe_hidden).to(device)
    ckpt = torch.load(probe_checkpoint_path, map_location=device)
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe.eval()
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Probe loaded from {probe_checkpoint_path} ({n_params:,} params)")

    return model, probe, tap_level


# ============================================================
# Per-frame inference
# ============================================================

def probe_infer(
    model,
    probe,
    tap_level,
    rgb_uint8: np.ndarray,
    *,
    num_tokens: int = 1600,
    planarity_threshold: float = 0.5,
    normal_threshold_rad: float = 0.087,
    depth_threshold: float = 0.025,
    neighbor_match_count: int = 8,
    out_h: int = 480,
    out_w: int = 640,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run frozen MoGe + probe -> plan2seg.

    Returns:
        labels: (out_h, out_w) int32 with 0 = non-planar, 1..K = planes
    """
    # Preprocess (476x644 as used in training)
    resized = cv2.resize(rgb_uint8, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)

    img_h, img_w = 476, 644
    aspect_ratio = img_w / img_h
    base_h = int((num_tokens / aspect_ratio) ** 0.5)
    base_w = int((num_tokens * aspect_ratio) ** 0.5)

    with torch.no_grad():
        # --- Frozen MoGe forward (encoder + neck + full normal_head tap) ---
        images = tensor.unsqueeze(0)
        batch_size = 1
        dtype = images.dtype

        features, cls_token = model.encoder(images, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level, height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio, dtype=dtype, device=images.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.cat([features[level], uv], dim=1)
        features = model.neck(features)

        # Run normal_head up to tap level to get intermediate features
        head = model.normal_head
        x = None
        for i in range(len(head.res_blocks)):
            feature = head.input_blocks[i](features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = head.res_blocks[i](x)
            if i == tap_level:
                tap_features = x
            if i < len(head.res_blocks) - 1:
                x = head.resamplers[i](x)

        # Probe -> planarity logits
        tap_up = F.interpolate(tap_features, (img_h, img_w), mode='bilinear', align_corners=False)
        planarity_logits = probe(tap_up)  # (1, 1, H, W)
        planarity = torch.sigmoid(planarity_logits[0, 0]).cpu().numpy()

        # Also get depth and normals from full MoGe forward
        output = model.forward(tensor.unsqueeze(0), num_tokens=num_tokens)
        points = output["points"][0].cpu().numpy()
        normals = output["normal"][0].cpu().numpy()
        depth = points[:, :, 2]

    # Resize to output resolution
    depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    normals = cv2.resize(normals, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm_len + 1e-8)
    planarity = cv2.resize(planarity, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Binary planarity mask
    planarity_binary = (planarity > planarity_threshold).astype(np.uint8)

    # plan2seg
    seg_device = "cpu" if "mps" in device else device
    labels, num_planes = compute_vectorized_planar_segments_v5_relative(
        planarity_mask=planarity_binary,
        normal=normals,
        depth=depth,
        normal_threshold_rad=normal_threshold_rad,
        depth_threshold=depth_threshold,
        neighbor_match_count_thresh=neighbor_match_count,
        device=seg_device,
    )
    return labels.astype(np.int32)


# ============================================================
# Export loop (mirrors run_pseudo_mono_export.py)
# ============================================================

def export_dataset(dataset_name, model, probe, tap_level, args):
    if args.output_dir:
        ds_out = os.path.join(args.output_dir, dataset_name)
    else:
        ds_out = f"/cluster/scratch/ayavuz/dataset/probe_planamono_{dataset_name}"
    os.makedirs(ds_out, exist_ok=True)

    total_frames = 0
    total_scenes = 0

    for scene_label, h5_rel, frame_ids, rgbs in DATASET_ITERS[dataset_name](args):
        if args.max_frames is not None and total_frames >= args.max_frames:
            break

        n = len(frame_ids)
        if args.max_frames is not None:
            n = min(n, args.max_frames - total_frames)
            frame_ids = frame_ids[:n]
            rgbs = rgbs[:n]

        planes_all = np.zeros((n, args.height, args.width), dtype=np.uint16)

        for i, rgb in enumerate(tqdm(rgbs, desc=f"  {scene_label}", leave=False)):
            labels = probe_infer(
                model, probe, tap_level, rgb,
                num_tokens=args.num_tokens,
                planarity_threshold=args.planarity_threshold,
                normal_threshold_rad=args.normal_threshold_rad,
                depth_threshold=args.depth_threshold,
                neighbor_match_count=args.neighbor_match_count,
                out_h=args.height,
                out_w=args.width,
                device=args.device,
            )
            planes_all[i] = labels.astype(np.uint16)

        # Save H5
        out_h5 = os.path.join(ds_out, h5_rel)
        os.makedirs(os.path.dirname(out_h5), exist_ok=True)
        with h5py.File(out_h5, "w") as f:
            dt = h5py.string_dtype()
            f.create_dataset("frame_ids", data=frame_ids, dtype=dt)
            f.create_dataset("planes", data=planes_all, dtype=np.uint16)

        total_frames += n
        total_scenes += 1
        tqdm.write(f"  Saved: {out_h5} ({n} frames)")

    print(f"  {dataset_name}: {total_scenes} scene(s), {total_frames} frames -> {ds_out}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Export probe model results on test sets")
    p.add_argument("--probe_checkpoint", type=str,
                   default="/cluster/scratch/ayavuz/moge_HIRES_4datasets_NORMAL_PROBE/probe_epoch2.pt",
                   help="Path to probe checkpoint (.pt)")
    p.add_argument("--base_model", type=str, default="Ruicheng/moge-2-vitl-normal",
                   help="Base MoGe model (HuggingFace path)")
    p.add_argument("--probe_hidden", type=int, default=32)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output root (default: /cluster/scratch/ayavuz/dataset/probe_planamono_{dataset})")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["scannetpp", "hypersim", "vkitti2", "synthia", "all"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)

    # Inference params
    p.add_argument("--num_tokens", type=int, default=1600)
    p.add_argument("--planarity_threshold", type=float, default=0.5)
    p.add_argument("--normal_threshold_rad", type=float, default=0.087,
                   help="Normal angle threshold in radians (~5 degrees)")
    p.add_argument("--depth_threshold", type=float, default=0.025,
                   help="Relative depth threshold (fraction of center depth)")
    p.add_argument("--neighbor_match_count", type=int, default=8,
                   help="Min matching neighbors in 5x5 window")

    # Dataset paths
    p.add_argument("--splits_root", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "splits"))
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASET_ITERS.keys()) if args.dataset == "all" else [args.dataset]

    model, probe, tap_level = load_probe_model(
        args.base_model, args.probe_checkpoint, args.probe_hidden, args.device)

    out_label = args.output_dir or "/cluster/scratch/ayavuz/dataset/probe_planamono_{dataset}"

    print("Probe Export (Frozen MoGe + Conv Probe -> plan2seg)")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Probe:      {args.probe_checkpoint}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Output:     {out_label}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Tokens:     {args.num_tokens}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames} per dataset")
    print("=" * 60)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        export_dataset(ds, model, probe, tap_level, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
