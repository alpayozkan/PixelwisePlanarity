"""
Verify whether the saved inference.h5 contains metric depth, by comparing:

  (A) saved   = depth field already stored in inference.h5
  (B) infer   = pretrained MoGe-2 model.infer() output (scale_head applied → metric)
  (C) raw_z   = raw forward → points[..., 2] (no scale, affine)
  (D) gt      = gt_depth field from inference.h5 (metric, ground truth)

For each compared pair we report the median ratio depth_other / depth_gt and its
spread. If MoGe-2 is delivering metric depth correctly:
  - (B) infer / (D) gt  should be ≈ 1.0 with low std
  - (A) saved / (D) gt  too, IF the saved dump was produced via model.infer()
  - (C) raw_z / (D) gt  will NOT be ≈ 1.0 (raw point map is affine)

Optional comparison columns:
  --ours_ckpt PATH        re-run a fine-tuned 4-head .pt model and add an "Ours" column
  --signals_h5_root DIR   read depth_metric from <DIR>/<scene>/moge_signals.h5
                          (output of save_moge_signals_planarity.py / save_moge_signals.py).
                          Validates a pre-saved dump matches a live re-run.

Example
-------
python compare_metric_depth.py \\
    --inference_h5 /cluster/scratch/ayavuz/inference/moge_HIRES_4datasets_epoch1/scannetpp/test/09c1414f1b/inference.h5 \\
    --rgb_root /cluster/project/cvg/Shared_datasets/scannet++/data \\
    --scene_id 09c1414f1b --frames 5 --num_tokens 1024
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HOME", "/cluster/scratch/aoezkan/cache/huggingface")

import copy
import torch.nn as nn  # noqa: E402

from planamono.moge.moge.model.v2 import MoGeModel  # noqa: E402
from planamono.moge.moge.utils.geometry_torch import recover_focal_shift  # noqa: E402
from planamono.paths import scannetpp_path  # noqa: E402


def _add_planarity_head(model: MoGeModel) -> None:
    """
    Match `MoGePlanarityInference._add_planarity_head` so that custom
    fine-tuned checkpoints (with a 4-th planarity head trained on top of
    the upstream MoGe-2 normal head) can load their state_dict.
    """
    model.planarity_head = copy.deepcopy(model.normal_head)
    last_conv = None
    for name, module in model.planarity_head.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    name, old_conv = last_conv
    new_conv = nn.Conv2d(
        old_conv.in_channels, 1,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=old_conv.bias is not None,
    )
    parent = model.planarity_head
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_conv)


def _load_ours_model(ckpt_path: str, base_hf: str, device: torch.device) -> MoGeModel:
    """Load a fine-tuned 4-head MoGe checkpoint (.pt with 'model_state_dict')."""
    model = MoGeModel.from_pretrained(base_hf).to(device)
    _add_planarity_head(model)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [ours] missing {len(missing)} keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  [ours] unexpected {len(unexpected)} keys (first 3): {unexpected[:3]}")
    return model.eval().float()


@torch.inference_mode()
def moge_depth_via_forward(model, img: torch.Tensor, num_tokens: int):
    """
    Replicate MoGeModel.infer() depth computation, but skip the
    `utils3d.torch.intrinsics_from_focal_center` call that fails on older utils3d.

    Returns:
        metric_depth   (H, W) float32 — affine depth × metric_scale (if scale_head present)
        affine_depth   (H, W) float32 — recovered affine depth (no scale)
        raw_pz         (H, W) float32 — raw points[..., 2] BEFORE shift
        metric_scale   float | None
    """
    out = model(img.unsqueeze(0), num_tokens=num_tokens)
    points = out["points"].float()                              # (1,H,W,3) affine
    mask = out.get("mask", None)
    metric_scale = out.get("metric_scale", None)
    if metric_scale is not None:
        metric_scale_v = float(metric_scale.float().item())
    else:
        metric_scale_v = None

    mask_binary = (mask.float() > 0.5) if mask is not None else None
    focal, shift = recover_focal_shift(points, mask_binary)     # (1,), (1,)
    raw_pz = points[0, ..., 2].clone()
    affine_depth = (points[0, ..., 2] + shift[0]).clone()
    metric_depth = affine_depth.clone()
    if metric_scale is not None:
        metric_depth = metric_depth * metric_scale.float()[0]
    return (
        metric_depth.float().cpu().numpy(),
        affine_depth.float().cpu().numpy(),
        raw_pz.float().cpu().numpy(),
        metric_scale_v,
    )


def _load_rgb_tensor(path: str, target_hw=None) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_hw is not None:
        H, W = target_hw
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W) in [0,1]
    return t


# ---------------------------------------------------------------------------
# Dump (signals H5) resolver
# ---------------------------------------------------------------------------

class DumpResolver:
    """Caches per-scene frame_id → row index maps for an
    ``<root>/<scene>/moge_signals.h5`` directory layout.

    Looks up ``depth_metric`` for a given (scene_id, frame_id). Returns ``None``
    if the scene H5, the frame_id, or the dataset is missing.
    """

    def __init__(self, root: str):
        self.root = root
        # scene_id → (h5_path, {fid: idx}) or None if scene H5 is missing
        self._cache = {}
        self._warned_missing_scenes = set()
        self._warned_missing_fids = set()

    def _info(self, scene_id: str):
        if scene_id in self._cache:
            return self._cache[scene_id]
        h5p = os.path.join(self.root, scene_id, "moge_signals.h5")
        if not os.path.isfile(h5p):
            self._cache[scene_id] = None
            return None
        with h5py.File(h5p, "r") as f:
            fids_raw = f["frame_ids"][:]
            fid_to_idx = {}
            for i, x in enumerate(fids_raw):
                fid = x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                fid_to_idx[fid] = i
        self._cache[scene_id] = (h5p, fid_to_idx)
        return self._cache[scene_id]

    def load_depth(self, scene_id: str, fid: str) -> np.ndarray:
        info = self._info(scene_id)
        if info is None:
            if scene_id not in self._warned_missing_scenes:
                print(f"  [dump] scene H5 missing under {self.root}/{scene_id}/")
                self._warned_missing_scenes.add(scene_id)
            return None
        h5p, fid_to_idx = info
        if fid not in fid_to_idx:
            key = (scene_id, fid)
            if key not in self._warned_missing_fids:
                print(f"  [dump] frame_id {fid} not in {h5p}")
                self._warned_missing_fids.add(key)
            return None
        with h5py.File(h5p, "r") as f:
            if "depth_metric" not in f:
                return None
            return f["depth_metric"][fid_to_idx[fid]].astype(np.float32)


def _ratio_stats(num: np.ndarray, den: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(num) & np.isfinite(den) & (num > 0.05) & (den > 0.05)
    if m.sum() < 1000:
        return None
    r = (num[m] / den[m]).astype(np.float64)
    return dict(
        n=int(m.sum()),
        median=float(np.median(r)),
        p25=float(np.percentile(r, 25)),
        p75=float(np.percentile(r, 75)),
        mean=float(r.mean()),
        std=float(r.std()),
    )


def _save_comparison_png(
    out_path: str,
    rgb: np.ndarray,
    saved: np.ndarray,
    gt: np.ndarray,
    title: str,
    extras=None,
) -> None:
    """Save a 2-row comparison figure.

    Layout: ``RGB | extras... | Saved | GT`` on row 0, with the bottom row
    showing ``(extra − GT)`` for each extra plus ``Saved − GT``. ``extras`` is
    a list of ``(label, depth_array)`` tuples — one per source to compare. The
    minimum is one extra (upstream MoGe-2); ``--ours_ckpt`` and
    ``--signals_h5_root`` each add another.
    """
    extras = extras or []
    n_extras = len(extras)
    n_cols = 3 + n_extras  # RGB | extras... | Saved | GT

    # ── Color limits ────────────────────────────────────────────────────────
    extra_arrays = [arr for _, arr in extras]
    valid_all = (gt > 0.05) & np.isfinite(saved)
    for arr in extra_arrays:
        valid_all &= np.isfinite(arr)

    depths_for_range = extra_arrays + [saved, gt]
    if valid_all.sum() < 100:
        vmin_d, vmax_d = 0.0, 5.0
    else:
        all_d = np.concatenate([d[valid_all] for d in depths_for_range])
        vmin_d, vmax_d = np.percentile(all_d, [2, 98])

    extra_diffs = [(label, arr - gt) for label, arr in extras]
    diff_saved = saved - gt
    diffs_for_range = [d for _, d in extra_diffs] + [diff_saved]
    if valid_all.sum() < 100:
        vlim_diff = 1.0
    else:
        vlim_diff = float(np.percentile(
            np.abs(np.concatenate([d[valid_all] for d in diffs_for_range])), 98))
        vlim_diff = max(vlim_diff, 1e-3)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4.4 * n_cols, 9))
    gs = fig.add_gridspec(
        nrows=2, ncols=n_cols,
        height_ratios=[1, 1],
        hspace=0.35, wspace=0.05,
        left=0.03, right=0.99, top=0.93, bottom=0.10,
    )
    cmap_d = plt.colormaps["turbo"]
    cmap_diff = plt.colormaps["RdBu_r"]

    def _show(ax, arr, vmin, vmax, cmap, panel_title):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    # Row 0: RGB | extras... | Saved | GT
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB", fontsize=11)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    im_d = None
    for k, (label, arr) in enumerate(extras):
        ax = fig.add_subplot(gs[0, 1 + k])
        im_d = _show(ax, arr, vmin_d, vmax_d, cmap_d, label)

    saved_col = 1 + n_extras
    gt_col = 2 + n_extras
    ax_saved = fig.add_subplot(gs[0, saved_col])
    if im_d is None:
        im_d = _show(ax_saved, saved, vmin_d, vmax_d, cmap_d, "Saved (H5)")
    else:
        _show(ax_saved, saved, vmin_d, vmax_d, cmap_d, "Saved (H5)")
    ax_gt = fig.add_subplot(gs[0, gt_col])
    _show(ax_gt, gt, vmin_d, vmax_d, cmap_d, "GT depth")

    # Row 1: blank | (extra − GT)... | (Saved − GT) | blank
    im_diff = None
    for k, (label, diff) in enumerate(extra_diffs):
        ax = fig.add_subplot(gs[1, 1 + k])
        im_diff = _show(ax, diff, -vlim_diff, vlim_diff, cmap_diff, f"{label} − GT")

    ax_diff_saved = fig.add_subplot(gs[1, saved_col])
    if im_diff is None:
        im_diff = _show(ax_diff_saved, diff_saved, -vlim_diff, vlim_diff, cmap_diff,
                        "Saved − GT")
    else:
        _show(ax_diff_saved, diff_saved, -vlim_diff, vlim_diff, cmap_diff, "Saved − GT")

    # ── Colorbars (span cols 1..n_cols-1, normalized to figure coords) ──────
    left_frac = 1.0 / n_cols + 0.02
    right_frac = 0.99
    width = right_frac - left_frac
    cax_d = fig.add_axes([left_frac, 0.535, width, 0.018])
    cax_diff = fig.add_axes([left_frac, 0.06, width, 0.018])

    cbar_d = fig.colorbar(im_d, cax=cax_d, orientation="horizontal")
    cbar_d.set_label("depth [m]", fontsize=10)
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Δdepth [m]", fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _viz_one_frame(
    model,
    inference_h5: str,
    frame_idx: int,
    scene_id: str,
    rgb_root: str,
    num_tokens: int,
    save_path: str,
    rgb_native: bool = False,
    model_ours=None,
    dump_resolver: "DumpResolver" = None,
):
    """Compute upstream metric / ours metric (optional) / saved dump (optional) /
    saved / gt for a single frame and save the PNG.
    """
    with h5py.File(inference_h5, "r") as f:
        fid_b = f["frame_ids"][frame_idx]
        fid = fid_b.decode() if isinstance(fid_b, (bytes, bytearray)) else str(fid_b)
        saved_depth = f["depth"][frame_idx].astype(np.float32)
        gt_depth = f["gt_depth"][frame_idx].astype(np.float32)
        H, W = saved_depth.shape

    rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
    if not os.path.isfile(rgb_path):
        print(f"[skip] missing RGB: {rgb_path}")
        return False

    target_hw = None if rgb_native else (H, W)
    img_t = _load_rgb_tensor(rgb_path, target_hw=target_hw).to(next(model.parameters()).device)

    metric_depth, affine_depth, raw_z, scale_v = moge_depth_via_forward(
        model, img_t, num_tokens=num_tokens,
    )
    if metric_depth.shape != (H, W):
        metric_depth = cv2.resize(metric_depth, (W, H), interpolation=cv2.INTER_LINEAR)

    ours_metric_depth = None
    ours_scale_v = None
    if model_ours is not None:
        o_metric, _, _, ours_scale_v = moge_depth_via_forward(
            model_ours, img_t, num_tokens=num_tokens,
        )
        if o_metric.shape != (H, W):
            o_metric = cv2.resize(o_metric, (W, H), interpolation=cv2.INTER_LINEAR)
        ours_metric_depth = o_metric

    dump_metric_depth = None
    if dump_resolver is not None:
        d_metric = dump_resolver.load_depth(scene_id, fid)
        if d_metric is not None:
            if d_metric.shape != (H, W):
                d_metric = cv2.resize(d_metric, (W, H), interpolation=cv2.INTER_LINEAR)
            dump_metric_depth = d_metric

    # RGB at H5 resolution, in [0, 1] for matplotlib
    rgb_for_viz = (img_t.cpu().numpy().transpose(1, 2, 0))
    if rgb_for_viz.shape[:2] != (H, W):
        rgb_for_viz = cv2.resize(rgb_for_viz, (W, H), interpolation=cv2.INTER_LINEAR)

    # Mask saved+moge with gt validity for clean visuals (don't alter raw arrays)
    valid = gt_depth > 0.05
    def _vis(arr):
        return np.where(valid, arr, np.nan) if arr is not None else None

    extras = [("Upstream metric (HF)", _vis(metric_depth))]
    if ours_metric_depth is not None:
        extras.append(("Ours metric (.pt)", _vis(ours_metric_depth)))
    if dump_metric_depth is not None:
        extras.append(("Dump (H5)", _vis(dump_metric_depth)))

    saved_vis = _vis(saved_depth)
    gt_vis = _vis(gt_depth)

    # Per-frame title with diagnostic ratios
    def _med(num, den):
        m = valid & (den > 0.05) & np.isfinite(num)
        return float(np.median(num[m] / den[m])) if m.any() else float("nan")
    parts = [
        f"{scene_id} / {fid}",
        f"ups_scale={scale_v:.3f}",
        f"med(saved/gt)={_med(saved_depth, gt_depth):.3f}",
        f"med(ups/gt)={_med(metric_depth, gt_depth):.3f}",
    ]
    if ours_metric_depth is not None:
        parts.append(f"ours_scale={ours_scale_v:.3f}")
        parts.append(f"med(ours/gt)={_med(ours_metric_depth, gt_depth):.3f}")
    if dump_metric_depth is not None:
        parts.append(f"med(dump/gt)={_med(dump_metric_depth, gt_depth):.3f}")
    title = "    ".join(parts)

    _save_comparison_png(
        save_path, rgb_for_viz, saved_vis, gt_vis, title,
        extras=extras,
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inference_h5", type=str, default=None,
                    help="Path to a single inference.h5 (audit / viz mode if --input_root not set)")
    ap.add_argument("--input_root", type=str, default=None,
                    help="Multi-scene root containing <scene>/inference.h5. "
                         "Used for viz mode when --save_dir is set, to draw frames across scenes.")
    ap.add_argument("--rgb_root", type=str,
                    default=os.path.join(scannetpp_path, "data"))
    ap.add_argument("--scene_id", type=str, default=None,
                    help="Override scene_id (otherwise inferred from h5 path)")
    ap.add_argument("--frames", type=int, default=5,
                    help="Audit mode: number of evenly-spaced frames to sample for stats")
    ap.add_argument("--num_tokens", type=int, default=1024)
    ap.add_argument("--hf_model", type=str, default="Ruicheng/moge-2-vitl-normal",
                    help="Upstream HF MoGe-2 model used for the metric reference column.")
    ap.add_argument("--ours_ckpt", type=str, default=None,
                    help="Optional .pt path of our fine-tuned 4-head MoGe checkpoint. "
                         "Adds an 'ours' depth column / panel to the comparison.")
    ap.add_argument("--signals_h5_root", type=str, default=None,
                    help="Optional path to a save_moge_signals(_planarity).py output root "
                         "(layout: <root>/<scene>/moge_signals.h5 with 'depth_metric' + 'frame_ids'). "
                         "Adds a 'Dump (H5)' column matched by scene_id + frame_id.")
    ap.add_argument("--rgb_native", action="store_true",
                    help="Run MoGe on native iPhone resolution (instead of resizing to H5's HxW). "
                         "Predictions are then resampled to HxW for comparison.")
    # Viz mode
    ap.add_argument("--save_dir", type=str, default=None,
                    help="If set, switch to visualization mode: pick N random (scene, frame) pairs "
                         "and save side-by-side comparison PNGs here.")
    ap.add_argument("--n_random", type=int, default=20,
                    help="Number of random frames to draw in viz mode (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.inference_h5 is None and args.input_root is None:
        ap.error("Provide --inference_h5 (single scene) or --input_root (multi-scene)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading upstream {args.hf_model} ...")
    model = MoGeModel.from_pretrained(args.hf_model).to(device).eval()
    print(f"  has scale_head = {hasattr(model, 'scale_head')}")

    model_ours = None
    if args.ours_ckpt is not None:
        print(f"Loading ours {args.ours_ckpt} ...")
        model_ours = _load_ours_model(args.ours_ckpt, args.hf_model, device)
        print(f"  has scale_head = {hasattr(model_ours, 'scale_head')}, "
              f"has planarity_head = {hasattr(model_ours, 'planarity_head')}")

    dump_resolver = None
    if args.signals_h5_root is not None:
        if not os.path.isdir(args.signals_h5_root):
            ap.error(f"--signals_h5_root not a directory: {args.signals_h5_root}")
        dump_resolver = DumpResolver(args.signals_h5_root)
        print(f"Reading dump signals from {args.signals_h5_root}")

    # ----- VIZ MODE -----
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        rng = np.random.default_rng(args.seed)

        # Build a pool of (scene_id, h5_path, frame_idx)
        pool = []
        if args.input_root is not None:
            for sid in sorted(os.listdir(args.input_root)):
                h5p = os.path.join(args.input_root, sid, "inference.h5")
                if not os.path.isfile(h5p):
                    continue
                with h5py.File(h5p, "r") as f:
                    n = f["depth"].shape[0]
                pool.extend([(sid, h5p, j) for j in range(n)])
        else:
            sid = args.scene_id or Path(args.inference_h5).parent.name
            with h5py.File(args.inference_h5, "r") as f:
                n = f["depth"].shape[0]
            pool = [(sid, args.inference_h5, j) for j in range(n)]

        if len(pool) == 0:
            ap.error("Pool is empty — no inference.h5 found")
        n_pick = min(args.n_random, len(pool))
        picks = rng.choice(len(pool), size=n_pick, replace=False)
        picks_sorted = sorted(picks.tolist())

        print(f"Viz mode: drawing {n_pick} random frames from a pool of {len(pool)}")
        for k, idx in enumerate(picks_sorted):
            sid, h5p, fi = pool[idx]
            out_path = os.path.join(args.save_dir, f"{k:02d}_{sid}_frame{fi:05d}.png")
            ok = _viz_one_frame(
                model=model,
                inference_h5=h5p,
                frame_idx=fi,
                scene_id=sid,
                rgb_root=args.rgb_root,
                num_tokens=args.num_tokens,
                save_path=out_path,
                rgb_native=args.rgb_native,
                model_ours=model_ours,
                dump_resolver=dump_resolver,
            )
            print(f"  [{k+1}/{n_pick}] {sid} frame {fi} → {out_path}{' (skipped)' if not ok else ''}")
        print(f"Done. PNGs in {args.save_dir}")
        return

    # ----- AUDIT MODE (original) -----
    if args.inference_h5 is None:
        ap.error("Audit mode requires --inference_h5 (or set --save_dir for viz mode)")
    scene_id = args.scene_id or Path(args.inference_h5).parent.name

    with h5py.File(args.inference_h5, "r") as f:
        N, H, W = f["depth"].shape
        idxs = np.linspace(0, N - 1, args.frames).astype(int)

        rows = []
        for i in idxs:
            fid_b = f["frame_ids"][i]
            fid = fid_b.decode() if isinstance(fid_b, (bytes, bytearray)) else str(fid_b)

            saved_depth = f["depth"][i].astype(np.float32)
            gt_depth = f["gt_depth"][i].astype(np.float32)

            rgb_path = os.path.join(args.rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
            target_hw = None if args.rgb_native else (H, W)
            img = _load_rgb_tensor(rgb_path, target_hw=target_hw).to(device)

            metric_depth, affine_depth, raw_z, metric_scale_v = moge_depth_via_forward(
                model, img, num_tokens=args.num_tokens,
            )
            infer_depth = metric_depth   # upstream metric output

            ours_metric = ours_affine = None
            ours_scale_v = None
            if model_ours is not None:
                ours_metric, ours_affine, _, ours_scale_v = moge_depth_via_forward(
                    model_ours, img, num_tokens=args.num_tokens,
                )

            dump_metric = None
            if dump_resolver is not None:
                dump_metric = dump_resolver.load_depth(scene_id, fid)

            def _resize(a):
                return a if (a is None or a.shape == (H, W)) else cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
            infer_depth = _resize(infer_depth)
            affine_depth = _resize(affine_depth)
            raw_z = _resize(raw_z)
            ours_metric = _resize(ours_metric)
            ours_affine = _resize(ours_affine)
            dump_metric = _resize(dump_metric)

            valid = (gt_depth > 0.05)

            r_saved = _ratio_stats(saved_depth, gt_depth, valid)
            r_infer = _ratio_stats(infer_depth, gt_depth, valid)
            r_affine = _ratio_stats(affine_depth, gt_depth, valid)
            r_raw = _ratio_stats(raw_z, gt_depth, valid)
            r_saved_vs_infer = _ratio_stats(saved_depth, infer_depth, valid & (infer_depth > 0.05))
            r_saved_vs_affine = _ratio_stats(saved_depth, affine_depth, valid & (affine_depth > 0.05))
            r_ours = (_ratio_stats(ours_metric, gt_depth, valid)
                      if ours_metric is not None else None)
            r_ours_aff = (_ratio_stats(ours_affine, gt_depth, valid)
                          if ours_affine is not None else None)
            r_saved_vs_ours = (_ratio_stats(saved_depth, ours_metric,
                                            valid & (ours_metric > 0.05))
                               if ours_metric is not None else None)
            r_saved_vs_ours_aff = (_ratio_stats(saved_depth, ours_affine,
                                                valid & (ours_affine > 0.05))
                                   if ours_affine is not None else None)
            r_dump = (_ratio_stats(dump_metric, gt_depth, valid)
                      if dump_metric is not None else None)
            r_saved_vs_dump = (_ratio_stats(saved_depth, dump_metric,
                                            valid & (dump_metric > 0.05))
                               if dump_metric is not None else None)
            r_ours_vs_dump = (_ratio_stats(ours_metric, dump_metric,
                                           valid & (dump_metric > 0.05))
                              if (ours_metric is not None and dump_metric is not None)
                              else None)

            rows.append((i, fid, metric_scale_v, ours_scale_v,
                         r_saved, r_infer, r_affine, r_raw,
                         r_saved_vs_infer, r_saved_vs_affine,
                         r_ours, r_ours_aff,
                         r_saved_vs_ours, r_saved_vs_ours_aff,
                         r_dump, r_saved_vs_dump, r_ours_vs_dump))

    # --- Print ---
    print(f"\nScene {scene_id} — {len(rows)} frames @ {H}x{W}\n")

    def fmt(r):
        return f"{'n/a':>14}" if r is None else f"med={r['median']:.3f} σ={r['std']:.3f}".rjust(14)

    have_ours = model_ours is not None
    if have_ours:
        print(f"{'frame':>6} {'ups_sc':>7} {'ours_sc':>8} | "
              f"{'saved/gt':>14} {'ups/gt':>14} {'ours/gt':>14} {'ours_aff/gt':>14} | "
              f"{'saved/ups':>14} {'saved/ours':>14} {'saved/ours_aff':>14}")
    else:
        print(f"{'frame':>6} {'ups_sc':>7} | "
              f"{'saved/gt':>14} {'ups/gt':>14} {'affine/gt':>14} {'raw_z/gt':>14} | "
              f"{'saved/ups':>14} {'saved/affine':>14}")
    print("-" * (175 if have_ours else 145))

    have_dump = dump_resolver is not None
    for row in rows:
        (i, fid, msc, omsc, ra, rb, raff, rc, rab, raab,
         rours, roursaff, rsv_ours, rsv_oursaff,
         rdump, rsv_dump, rours_dump) = row
        msc_s = "n/a" if msc is None else f"{msc:.3f}"
        suffix_dump = ""
        if have_dump:
            suffix_dump = f" | {fmt(rdump)} {fmt(rsv_dump)}"
            if have_ours:
                suffix_dump += f" {fmt(rours_dump)}"
        if have_ours:
            omsc_s = "n/a" if omsc is None else f"{omsc:.3f}"
            print(f"{i:>6} {msc_s:>7} {omsc_s:>8} | "
                  f"{fmt(ra)} {fmt(rb)} {fmt(rours)} {fmt(roursaff)} | "
                  f"{fmt(rab)} {fmt(rsv_ours)} {fmt(rsv_oursaff)}"
                  + suffix_dump)
        else:
            print(f"{i:>6} {msc_s:>7} | {fmt(ra)} {fmt(rb)} {fmt(raff)} {fmt(rc)} | "
                  f"{fmt(rab)} {fmt(raab)}"
                  + suffix_dump)

    def agg(col_idx, label, suffix=""):
        vals = [r[col_idx]['median'] for r in rows if r[col_idx] is not None]
        if not vals: return f"  {label}: n/a"
        v = np.array(vals)
        return (f"  {label:<28}: mean={v.mean():.3f}, std={v.std():.3f}, "
                f"range=[{v.min():.3f}, {v.max():.3f}]{suffix}")

    print(f"\nAcross-frame aggregate of per-frame median ratios:")
    print(agg(4, "(A) saved / gt"))
    print(agg(5, "(B) upstream / gt", "    <-- ≈1.0 ⇒ scale_head works in upstream"))
    print(agg(6, "(B') upstream affine / gt"))
    print(agg(7, "(C) raw_z / gt"))
    print(agg(8, "(A/B) saved / upstream", "  <-- ≈1.0 ⇒ saved is metric"))
    print(agg(9, "(A/B') saved / upstream affine", "  <-- ≈1.0 ⇒ saved is affine"))
    if have_ours:
        print()
        print(agg(10, "(D) ours metric / gt", "    <-- ≈1.0 ⇒ ours produces metric depth"))
        print(agg(11, "(D') ours affine / gt"))
        print(agg(12, "(A/D) saved / ours metric", "  <-- ≈1.0 ⇒ saved == ours metric"))
        print(agg(13, "(A/D') saved / ours affine", "  <-- ≈1.0 ⇒ saved == ours affine (no scale)"))
    if have_dump:
        print()
        print(agg(14, "(E) dump / gt", "    <-- ≈1.0 ⇒ dump is metric"))
        print(agg(15, "(A/E) saved / dump", "  <-- ≈1.0 ⇒ saved == dump"))
        if have_ours:
            print(agg(16, "(D/E) ours / dump",
                      "  <-- ≈1.0 ⇒ dump matches live --ours_ckpt re-run"))


if __name__ == "__main__":
    main()
