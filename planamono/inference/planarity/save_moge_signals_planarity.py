"""
Run **4-head MoGe-2** inference (with planarity head) and dump the signals
relevant to plane segmentation as one H5 per scene.

Mirrors ``save_moge_signals.py`` but loads a custom .pt checkpoint via
``MoGePlanarityInference`` (which registers the extra ``planarity_head``)
and saves the planarity probability in addition to the standard signals.

Output layout per scene:

    <output_root>/<scene_id>/moge_signals.h5
        frame_ids       (N,)        S<bytes>      e.g. b'frame_000000'
        planarity       (N, H, W)     float16     [0, 1] probability
        normal          (N, H, W, 3)  float16     unit vectors, camera frame
        depth_metric    (N, H, W)     float16     metric depth (scale_head applied)
        mask            (N, H, W)     uint8       0/1 validity (mask > 0.5)
        intrinsics      (N, 3, 3)     float32     pixel-space K (cx=W/2, cy=H/2)
        metric_scale    (N,)          float32     per-image scalar

    attrs:
        resolution           "HxW"
        num_tokens           int
        source_model         "MoGePlanarityInference (4-head)"
        base_model           "Ruicheng/moge-2-vitl-normal"
        model_path           absolute path to the .pt checkpoint
        metric_depth_applied True
        has_planarity        True

The depth pipeline replicates ``MoGeModel.infer()`` manually (forward → recover
focal & shift → apply metric_scale) to bypass the cluster's older
``utils3d 0.1.1`` which is missing ``utils3d.torch.intrinsics_from_focal_center``.

Example
-------
python save_moge_signals_planarity.py \\
    --scenes planamono/splits/scannetpp/test.txt \\
    --model_path /cluster/scratch/ayavuz/trained_checkpoints/checkpoints_last/checkpoints/moge_HIRES_4datasets/model_epoch1.pt \\
    --output_root /cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1 \\
    --frame_step 25 --batch_size 8 --num_tokens 1600
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HOME", "/cluster/scratch/aoezkan/cache/huggingface")

from planamono.moge.moge.utils.geometry_torch import recover_focal_shift  # noqa: E402
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference  # noqa: E402
from planamono.paths import scannetpp_path  # noqa: E402


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def moge_forward_signals_planarity(
    model: torch.nn.Module,
    images: torch.Tensor,
    num_tokens: int,
) -> dict:
    """
    Run a 4-head MoGe-2 forward pass and produce the per-pixel + per-image
    signals we want to save for plane segmentation.

    Replicates ``MoGeModel.infer()`` for everything except the
    ``utils3d.torch.intrinsics_from_focal_center`` call (built manually here),
    and additionally extracts the ``planarity`` head output.

    Args:
        images: (B, 3, H, W) in [0, 1] on the model device.
        num_tokens: ViT tokens (e.g. 1600 = MoGe-2 default).

    Returns dict with numpy arrays (CPU):
        planarity    (B, H, W)    float32   [0, 1] probability
        normal       (B, H, W, 3) float32   unit-norm
        depth_metric (B, H, W)    float32   metric (scale_head applied)
        mask         (B, H, W)    bool      mask > 0.5
        intrinsics   (B, 3, 3)    float32   pixel-space K (cx=W/2, cy=H/2)
        metric_scale (B,)         float32
    """
    B, _, H, W = images.shape
    aspect_ratio = W / H

    out = model(images, num_tokens=num_tokens)
    points = out["points"].float()                              # (B,H,W,3) affine
    normal = out["normal"].float()                              # (B,H,W,3) unit
    mask = out["mask"].float() if "mask" in out else None       # (B,H,W) sigmoid
    metric_scale = out["metric_scale"].float() if "metric_scale" in out else None

    # Planarity head — accept (B, H, W) or (B, 1, H, W)
    if "planarity" not in out:
        raise RuntimeError(
            "Model output is missing 'planarity' — make sure the loaded "
            "checkpoint includes a planarity_head (4-head MoGe)."
        )
    planarity = out["planarity"].float()
    if planarity.ndim == 4 and planarity.shape[1] == 1:
        planarity = planarity.squeeze(1)                        # (B,H,W)

    mask_binary = (mask > 0.5) if mask is not None else None    # (B,H,W) bool

    # Recover focal & shift from the affine point map
    focal, shift = recover_focal_shift(points, mask_binary)     # focal (B,), shift (B,)

    # MoGe normalized intrinsics: cx=cy=0.5; fx, fy depend on aspect
    fx_n = focal / 2.0 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
    fy_n = focal / 2.0 * (1 + aspect_ratio ** 2) ** 0.5
    # Convert to pixel-space K (matches the convention of the legacy inference.h5)
    K = torch.zeros(B, 3, 3, dtype=torch.float32, device=points.device)
    K[:, 0, 0] = fx_n * W
    K[:, 1, 1] = fy_n * H
    K[:, 0, 2] = 0.5 * W
    K[:, 1, 2] = 0.5 * H
    K[:, 2, 2] = 1.0

    # Affine depth = z + shift; metric depth = affine * metric_scale
    affine_depth = points[..., 2] + shift[:, None, None]        # (B,H,W)
    if metric_scale is not None:
        depth_metric = affine_depth * metric_scale[:, None, None]
    else:
        depth_metric = affine_depth

    return {
        "planarity":    planarity.detach().cpu().numpy().astype(np.float32),
        "normal":       normal.detach().cpu().numpy().astype(np.float32),
        "depth_metric": depth_metric.detach().cpu().numpy().astype(np.float32),
        "mask":         (mask_binary.detach().cpu().numpy().astype(np.uint8)
                         if mask_binary is not None
                         else np.ones((B, H, W), dtype=np.uint8)),
        "intrinsics":   K.detach().cpu().numpy().astype(np.float32),
        "metric_scale": (metric_scale.detach().cpu().numpy().astype(np.float32)
                         if metric_scale is not None
                         else np.ones(B, dtype=np.float32)),
    }


# ---------------------------------------------------------------------------
# RGB IO + frame enumeration  (identical to save_moge_signals.py)
# ---------------------------------------------------------------------------

def _load_rgb_batch(rgb_paths: List[str], target_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = target_hw
    imgs = np.empty((len(rgb_paths), H, W, 3), dtype=np.float32)
    for i, p in enumerate(rgb_paths):
        bgr = cv2.imread(p)
        if bgr is None:
            raise FileNotFoundError(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        imgs[i] = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)


def _list_scene_frames(rgb_root: str, scene_id: str, frame_step: int) -> List[str]:
    rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
    if not os.path.isdir(rgb_dir):
        return []
    files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    return files[::frame_step]


def _read_scenes(scenes_arg: str) -> List[str]:
    """Accept either a path to a txt file (one scene per line) or comma-separated list."""
    if os.path.isfile(scenes_arg):
        with open(scenes_arg) as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [s.strip() for s in scenes_arg.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(
    model: torch.nn.Module,
    scene_id: str,
    rgb_root: str,
    output_root: str,
    target_hw: Tuple[int, int],
    num_tokens: int,
    batch_size: int,
    frame_step: int,
    overwrite: bool,
    source_model: str,
    base_model: str,
    model_path: str,
) -> int:
    out_dir = os.path.join(output_root, scene_id)
    out_h5 = os.path.join(out_dir, "moge_signals.h5")
    if os.path.isfile(out_h5) and not overwrite:
        return -1   # signal: skipped

    frames = _list_scene_frames(rgb_root, scene_id, frame_step)
    if not frames:
        return 0

    H, W = target_hw
    N = len(frames)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-allocate datasets in the H5 (chunked + gzip) so memory stays bounded.
    with h5py.File(out_h5, "w") as f:
        f.attrs["resolution"] = f"{H}x{W}"
        f.attrs["num_tokens"] = int(num_tokens)
        f.attrs["source_model"] = source_model
        f.attrs["base_model"] = base_model
        f.attrs["model_path"] = model_path
        f.attrs["metric_depth_applied"] = True
        f.attrs["has_planarity"] = True

        d_planarity = f.create_dataset("planarity",   (N, H, W),
                                       dtype="float16", chunks=(1, H, W),
                                       compression="gzip", compression_opts=4)
        d_normal = f.create_dataset("normal",       (N, H, W, 3),
                                    dtype="float16", chunks=(1, H, W, 3),
                                    compression="gzip", compression_opts=4)
        d_depth = f.create_dataset("depth_metric",  (N, H, W),
                                   dtype="float16", chunks=(1, H, W),
                                   compression="gzip", compression_opts=4)
        d_mask = f.create_dataset("mask",           (N, H, W),
                                  dtype="uint8", chunks=(1, H, W),
                                  compression="gzip", compression_opts=4)
        d_K = f.create_dataset("intrinsics",        (N, 3, 3), dtype="float32")
        d_scale = f.create_dataset("metric_scale",  (N,),       dtype="float32")
        d_fids = f.create_dataset("frame_ids",      (N,),
                                  dtype=h5py.string_dtype(encoding="utf-8"))

        for i in tqdm(range(0, N, batch_size),
                      desc=f"  {scene_id}", leave=False, unit="batch"):
            chunk = frames[i:i + batch_size]
            paths = [os.path.join(rgb_root, scene_id, "iphone", "rgb", fn) for fn in chunk]
            imgs = _load_rgb_batch(paths, target_hw=target_hw).to(next(model.parameters()).device)

            sig = moge_forward_signals_planarity(model, imgs, num_tokens=num_tokens)

            sl = slice(i, i + len(chunk))
            d_planarity[sl] = sig["planarity"].astype(np.float16)
            d_normal[sl] = sig["normal"].astype(np.float16)
            d_depth[sl] = sig["depth_metric"].astype(np.float16)
            d_mask[sl] = sig["mask"]
            d_K[sl] = sig["intrinsics"]
            d_scale[sl] = sig["metric_scale"]
            for j, fn in enumerate(chunk):
                d_fids[i + j] = os.path.splitext(fn)[0]   # 'frame_000000'

    return N


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    "/cluster/scratch/ayavuz/trained_checkpoints/checkpoints_last/checkpoints/"
    "moge_HIRES_4datasets/model_epoch1.pt"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=str, required=True,
                    help="Path to scenes txt (one scene per line) OR comma-separated scene list.")
    ap.add_argument("--rgb_root", type=str,
                    default=os.path.join(scannetpp_path, "data"),
                    help="ScanNet++ data root (frames at <root>/<scene>/iphone/rgb/<frame>.jpg).")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                    help="Path to a 4-head MoGe checkpoint (.pt). Must include planarity_head.")
    ap.add_argument("--base_model", type=str, default="Ruicheng/moge-2-vitl-normal",
                    help="HuggingFace base model id used to seed the architecture before "
                         "loading the .pt checkpoint.")
    ap.add_argument("--resolution", type=str, default="480x640",
                    help="HxW for MoGe input. Default 480x640 to match the legacy inference.h5.")
    ap.add_argument("--num_tokens", type=int, default=1600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--frame_step", type=int, default=25,
                    help="Take every Nth frame from each scene's iPhone RGB folder (default 25).")
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, re-process scenes whose moge_signals.h5 already exists.")
    args = ap.parse_args()

    H, W = (int(x) for x in args.resolution.lower().split("x"))

    scenes = _read_scenes(args.scenes)
    if not scenes:
        ap.error("No scenes parsed from --scenes")

    if not os.path.isfile(args.model_path):
        ap.error(f"--model_path does not exist: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading 4-head checkpoint {args.model_path} ...")
    inference = MoGePlanarityInference(args.model_path, device=device.type)
    # Match save_moge_signals.py optimizations
    inference.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference.model.eval()
    model = inference.model
    print(f"  has scale_head     = {hasattr(model, 'scale_head')}")
    print(f"  has planarity_head = {hasattr(model, 'planarity_head')}")
    print(f"  base_model         = {args.base_model}")

    os.makedirs(args.output_root, exist_ok=True)
    print(f"Processing {len(scenes)} scene(s) → {args.output_root}")
    print(f"  resolution {H}x{W}, num_tokens {args.num_tokens}, "
          f"batch {args.batch_size}, step {args.frame_step}")

    source_model = "MoGePlanarityInference (4-head)"

    total_frames = 0
    skipped = 0
    pbar = tqdm(scenes, desc="Scenes", unit="scene")
    for sid in pbar:
        pbar.set_postfix_str(sid)
        try:
            n = process_scene(
                model=model,
                scene_id=sid,
                rgb_root=args.rgb_root,
                output_root=args.output_root,
                target_hw=(H, W),
                num_tokens=args.num_tokens,
                batch_size=args.batch_size,
                frame_step=args.frame_step,
                overwrite=args.overwrite,
                source_model=source_model,
                base_model=args.base_model,
                model_path=args.model_path,
            )
        except Exception as e:
            tqdm.write(f"[fail] {sid}: {e}")
            continue
        if n == -1:
            skipped += 1
            tqdm.write(f"[skip] {sid}: moge_signals.h5 exists (use --overwrite)")
        elif n == 0:
            tqdm.write(f"[skip] {sid}: no RGB frames found at {args.rgb_root}/{sid}/iphone/rgb/")
        else:
            total_frames += n

    print(f"\nDone. Wrote {total_frames} frames "
          f"({skipped} scene(s) skipped) to {args.output_root}")


if __name__ == "__main__":
    main()
