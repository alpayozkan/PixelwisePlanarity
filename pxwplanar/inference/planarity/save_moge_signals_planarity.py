"""
Run **4-head MoGe-2** inference (with planarity head) and dump the signals
relevant to plane segmentation as one H5 per scene.

Supports four datasets via ``--dataset``:

    scannetpp     ScanNet++ iPhone JPGs (per-scene folders).
                  RGB lives at  <data_root>/<scene>/iphone/rgb/<frame>.jpg
    nyuv2         NYU-v2 ZeroPlane "_d2" NPZ files.
                  Single virtual scene "nyuv2".
    sevenscenes   7-Scenes ZeroPlane "_d2" NPZ files.
                  One H5 per logical scene (chess, fire, ...).
    hypersim      Hypersim HDR HDF5 frames (tone-mapped like the training
                  loader). One H5 per (scene, camera): "<scene>/<cam>".
                  Frames enumerated from the rendered plane-GT H5s so signals
                  align with evaluation.

Output layout per scene (identical schema across datasets):

    <output_root>/<scene_id>/moge_signals.h5
        frame_ids       (N,)        S<bytes>      e.g. b'frame_000000'
        planarity       (N, H, W)     float16     [0, 1] probability
        normal          (N, H, W, 3)  float16     unit vectors, camera frame
        depth_metric    (N, H, W)     float16     metric depth
                                                  (scale_head applied)
        mask            (N, H, W)     uint8       0/1 validity (mask > 0.5)
        intrinsics      (N, 3, 3)     float32     pixel-space K (cx=W/2, cy=H/2)
        metric_scale    (N,)          float32     per-image scalar

    attrs:
        dataset              "scannetpp" / "nyuv2" / "sevenscenes" / "hypersim"
        split                only for NPZ datasets (e.g. "test" / "val")
        resolution           "HxW"
        num_tokens           int
        source_model         "MoGePlanarityInference (4-head)"
        base_model           "Ruicheng/moge-2-vitl-normal"
        model_path           absolute path to the .pt checkpoint
        metric_depth_applied True
        has_planarity        True

For NPZ-based datasets, frame_ids are zero-padded NPZ-sample indices
("frame_000123") so they sort lexicographically and match the order in the
corresponding ``<dataset>PlaneDataset.valid_pairs``.

The depth pipeline replicates ``MoGeModel.infer()`` manually (forward → recover
focal & shift → apply metric_scale), building the intrinsics matrix explicitly
to keep the pixel-space convention independent of the utils3d version.

Examples
--------
ScanNet++ test split:
    python save_moge_signals_planarity.py \\
        --dataset scannetpp \\
        --scenes splits/scannetpp/test.txt \\
        --model_path <checkpoint.pt> \\
        --output_root <signals_root>/scannetpp \\
        --frame_step 25 --batch_size 8 --num_tokens 1600

NYU-v2 (auto-discovers samples):
    python save_moge_signals_planarity.py \\
        --dataset nyuv2 \\
        --model_path <checkpoint.pt> \\
        --output_root <signals_root>/nyuv2 \\
        --batch_size 16 --num_tokens 1600

7-Scenes (one H5 per scene; --scenes optional comma filter):
    python save_moge_signals_planarity.py \\
        --dataset sevenscenes \\
        --model_path <checkpoint.pt> \\
        --output_root <signals_root>/sevenscenes \\
        --batch_size 16 --num_tokens 1600

Hypersim test split (one H5 per scene/cam; --scenes optional scene filter):
    python save_moge_signals_planarity.py \\
        --dataset hypersim \\
        --model_path <checkpoint.pt> \\
        --output_root <signals_root>/hypersim \\
        --batch_size 8 --num_tokens 1600
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MoGe.moge.utils.geometry_torch import recover_focal_shift  # noqa: E402
from pxwplanar.inference.planarity.moge_inference import (
    MoGePlanarityInference,  # noqa: E402
)
from pxwplanar.paths import (
    hypersim_path,
    nyuv2_path,
    scannetpp_path,
    sevenscenes_path,
)  # noqa: E402

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
    points = out["points"].float()  # (B,H,W,3) affine
    normal = out["normal"].float()  # (B,H,W,3) unit
    mask = out["mask"].float() if "mask" in out else None  # (B,H,W) sigmoid
    metric_scale = (
        out["metric_scale"].float() if "metric_scale" in out else None
    )

    # Planarity head — accept (B, H, W) or (B, 1, H, W)
    if "planarity" not in out:
        raise RuntimeError(
            "Model output is missing 'planarity' — make sure the loaded "
            "checkpoint includes a planarity_head (4-head MoGe)."
        )
    planarity = out["planarity"].float()
    if planarity.ndim == 4 and planarity.shape[1] == 1:
        planarity = planarity.squeeze(1)  # (B,H,W)

    mask_binary = (mask > 0.5) if mask is not None else None  # (B,H,W) bool

    # Recover focal & shift from the affine point map
    focal, shift = recover_focal_shift(
        points, mask_binary
    )  # focal (B,), shift (B,)

    # MoGe normalized intrinsics: cx=cy=0.5; fx, fy depend on aspect
    fx_n = focal / 2.0 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
    fy_n = focal / 2.0 * (1 + aspect_ratio**2) ** 0.5
    # Convert to pixel-space K (matches the convention of the legacy
    # inference.h5)
    K = torch.zeros(B, 3, 3, dtype=torch.float32, device=points.device)
    K[:, 0, 0] = fx_n * W
    K[:, 1, 1] = fy_n * H
    K[:, 0, 2] = 0.5 * W
    K[:, 1, 2] = 0.5 * H
    K[:, 2, 2] = 1.0

    # Affine depth = z + shift; metric depth = affine * metric_scale
    affine_depth = points[..., 2] + shift[:, None, None]  # (B,H,W)
    if metric_scale is not None:
        depth_metric = affine_depth * metric_scale[:, None, None]
    else:
        depth_metric = affine_depth

    return {
        "planarity": planarity.detach().cpu().numpy().astype(np.float32),
        "normal": normal.detach().cpu().numpy().astype(np.float32),
        "depth_metric": depth_metric.detach().cpu().numpy().astype(np.float32),
        "mask": (
            mask_binary.detach().cpu().numpy().astype(np.uint8)
            if mask_binary is not None
            else np.ones((B, H, W), dtype=np.uint8)
        ),
        "intrinsics": K.detach().cpu().numpy().astype(np.float32),
        "metric_scale": (
            metric_scale.detach().cpu().numpy().astype(np.float32)
            if metric_scale is not None
            else np.ones(B, dtype=np.float32)
        ),
    }


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


@dataclass
class FrameSpec:
    """Adapter-specific info needed to load one frame's RGB."""

    frame_id: str  # written to H5 frame_ids
    jpg_path: str | None = None  # for scannetpp
    npz_path: str | None = None  # for nyuv2 / sevenscenes
    hdf5_path: str | None = None  # for hypersim


def _resize_rgb(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize RGB to target HW with INTER_LINEAR if needed.

    Returns H×W×3 float32 in [0,1].
    """
    H, W = target_hw
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


class _Adapter:
    """Base interface — subclasses provide list_scenes / list_frames /
    load_rgb_batch."""

    name: str = ""
    split: str | None = None  # written to H5 attrs for NPZ datasets

    def list_scenes(self) -> list[str]:
        raise NotImplementedError

    def list_frames(self, scene_id: str, frame_step: int) -> list[FrameSpec]:
        raise NotImplementedError

    def load_rgb_batch(
        self, frames: list[FrameSpec], target_hw: tuple[int, int]
    ) -> torch.Tensor:
        raise NotImplementedError


class _ScanNetppAdapter(_Adapter):
    """Per-scene JPG folder loader matching the legacy
    save_moge_signals.py layout."""

    name = "scannetpp"

    def __init__(self, rgb_root: str, scene_list: list[str]):
        self._rgb_root = rgb_root
        self._scene_list = scene_list

    def list_scenes(self) -> list[str]:
        return list(self._scene_list)

    def list_frames(self, scene_id: str, frame_step: int) -> list[FrameSpec]:
        rgb_dir = os.path.join(self._rgb_root, scene_id, "iphone", "rgb")
        if not os.path.isdir(rgb_dir):
            return []
        files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".jpg"))
        files = files[:: max(1, frame_step)]
        return [
            FrameSpec(
                frame_id=os.path.splitext(fn)[0],  # 'frame_000000'
                jpg_path=os.path.join(rgb_dir, fn),
            )
            for fn in files
        ]

    def load_rgb_batch(
        self, frames: list[FrameSpec], target_hw: tuple[int, int]
    ) -> torch.Tensor:
        H, W = target_hw
        imgs = np.empty((len(frames), H, W, 3), dtype=np.float32)
        for i, fs in enumerate(frames):
            bgr = cv2.imread(fs.jpg_path)
            if bgr is None:
                raise FileNotFoundError(fs.jpg_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs[i] = _resize_rgb(rgb, target_hw)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


class _NPZAdapter(_Adapter):
    """Common RGB loader for ZeroPlane '_d2.npz' datasets
    (nyuv2, sevenscenes)."""

    def load_rgb_batch(
        self, frames: list[FrameSpec], target_hw: tuple[int, int]
    ) -> torch.Tensor:
        H, W = target_hw
        imgs = np.empty((len(frames), H, W, 3), dtype=np.float32)
        for i, fs in enumerate(frames):
            d = np.load(fs.npz_path, allow_pickle=True)
            # raw_image is the high-res 480x640 BGR array.
            bgr = d["raw_image"]
            rgb = bgr[..., ::-1]  # BGR -> RGB
            imgs[i] = _resize_rgb(rgb, target_hw)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


class _NYUv2Adapter(_NPZAdapter):
    name = "nyuv2"

    def __init__(self, data_root: str, split: str):
        from pxwplanar.shared.datasets.nyuv2_plane_dataset import (
            NYUv2PlaneDataset,
        )

        self.split = split
        ds = NYUv2PlaneDataset(data_root, split=split)
        # NYUv2 is a single virtual scene.
        # valid_pairs = [(npz_path, sample_idx), ...].
        self._frames: list[FrameSpec] = [
            FrameSpec(
                frame_id=f"frame_{int(sample_idx):06d}",
                npz_path=npz_path,
            )
            for npz_path, sample_idx in ds.valid_pairs
        ]

    def list_scenes(self) -> list[str]:
        return ["nyuv2"]

    def list_frames(self, scene_id: str, frame_step: int) -> list[FrameSpec]:
        if scene_id != "nyuv2":
            return []
        return self._frames[:: max(1, frame_step)]


class _SevenScenesAdapter(_NPZAdapter):
    name = "sevenscenes"

    def __init__(
        self,
        data_root: str,
        split: str,
        scene_filter: list[str] | None = None,
    ):
        from pxwplanar.shared.datasets.sevenscenes_plane_dataset import (
            SevenScenesPlaneDataset,
        )

        self.split = split
        ds = SevenScenesPlaneDataset(data_root, split=split)
        # valid_pairs = [(npz_path, sample_idx, scene_id, origin), ...]
        by_scene: dict[str, list[FrameSpec]] = defaultdict(list)
        for npz_path, sample_idx, scene_id, _origin in ds.valid_pairs:
            by_scene[scene_id].append(
                FrameSpec(
                    frame_id=f"frame_{int(sample_idx):06d}",
                    npz_path=npz_path,
                )
            )
        self._by_scene = dict(by_scene)
        self._scene_ids = sorted(by_scene.keys())
        if scene_filter:
            wanted = {s.strip() for s in scene_filter if s.strip()}
            self._scene_ids = [s for s in self._scene_ids if s in wanted]

    def list_scenes(self) -> list[str]:
        return list(self._scene_ids)

    def list_frames(self, scene_id: str, frame_step: int) -> list[FrameSpec]:
        return self._by_scene.get(scene_id, [])[:: max(1, frame_step)]


class _HypersimAdapter(_Adapter):
    """Hypersim HDR HDF5 loader.

    Enumerates frames via ``HypersimPlaneDataset`` so the exported signals
    cover exactly the frames present in the rendered plane-GT H5s (one per
    scene/camera). Each (scene, camera) pair becomes one logical scene
    "<scene_id>/<cam_name>", giving the per-camera H5 layout evaluation expects.
    HDR frames are tone-mapped with the same routine as the training loader.
    """

    name = "hypersim"

    def __init__(
        self,
        data_root: str,
        split: str,
        scene_filter: list[str] | None = None,
    ):
        from pxwplanar.paths import hypersim_params_path, hypersim_rendered_path
        from pxwplanar.shared.datasets.hypersim_plane_dataset import (
            HypersimPlaneDataset,
        )

        self.split = split
        ds = HypersimPlaneDataset(
            hypersim_root=data_root,
            plane_label_root=hypersim_rendered_path,
            params_root=hypersim_params_path,
            split_txt_dir=str(_REPO_ROOT / "splits" / "hypersim"),
            split=split,
        )
        self._tonemap = ds._tonemap_rgb_robust
        # valid_pairs = [(scene_id, cam_name, idx, fid, rgb_path, ...), ...]
        by_scene: dict[str, list[FrameSpec]] = defaultdict(list)
        for scene_id, cam_name, _idx, fid, rgb_path, *_rest in ds.valid_pairs:
            by_scene[f"{scene_id}/{cam_name}"].append(
                FrameSpec(
                    frame_id=str(fid),
                    hdf5_path=rgb_path,
                )
            )
        self._by_scene = dict(by_scene)
        self._scene_ids = sorted(by_scene.keys())
        if scene_filter:
            wanted = {s.strip() for s in scene_filter if s.strip()}
            # Filter accepts scene ids ('ai_001_001') or scene/cam
            # ('ai_001_001/cam_00')
            self._scene_ids = [
                s
                for s in self._scene_ids
                if s in wanted or s.split("/")[0] in wanted
            ]

    def list_scenes(self) -> list[str]:
        return list(self._scene_ids)

    def list_frames(self, scene_id: str, frame_step: int) -> list[FrameSpec]:
        return self._by_scene.get(scene_id, [])[:: max(1, frame_step)]

    def load_rgb_batch(
        self, frames: list[FrameSpec], target_hw: tuple[int, int]
    ) -> torch.Tensor:
        H, W = target_hw
        imgs = np.empty((len(frames), H, W, 3), dtype=np.float32)
        for i, fs in enumerate(frames):
            with h5py.File(fs.hdf5_path, "r") as f:
                rgb = f[list(f.keys())[0]][:]  # (H, W, 3)
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            elif rgb.dtype == np.uint16:
                rgb = rgb.astype(np.float32) / 65535.0
            else:
                rgb = self._tonemap(rgb.astype(np.float32))  # HDR → [0,1]
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            imgs[i] = rgb.astype(np.float32)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


def _build_adapter(args, target_hw: tuple[int, int]) -> _Adapter:
    if args.dataset == "scannetpp":
        if not args.scenes:
            raise SystemExit(
                "--scenes is required for --dataset scannetpp "
                "(path to a txt with one scene per line)."
            )
        scene_list = _read_scenes(args.scenes)
        if not scene_list:
            raise SystemExit(f"No scenes parsed from --scenes={args.scenes}")
        rgb_root = args.data_root or os.path.join(scannetpp_path, "data")
        return _ScanNetppAdapter(rgb_root=rgb_root, scene_list=scene_list)

    if args.dataset == "nyuv2":
        return _NYUv2Adapter(
            data_root=args.data_root or nyuv2_path,
            split=args.split or "test",
        )

    if args.dataset == "sevenscenes":
        scene_filter = _read_scenes(args.scenes) if args.scenes else None
        return _SevenScenesAdapter(
            data_root=args.data_root or sevenscenes_path,
            split=args.split or "val",
            scene_filter=scene_filter,
        )

    if args.dataset == "hypersim":
        scene_filter = _read_scenes(args.scenes) if args.scenes else None
        return _HypersimAdapter(
            data_root=args.data_root or hypersim_path,
            split=args.split or "test",
            scene_filter=scene_filter,
        )

    raise SystemExit(f"Unknown --dataset: {args.dataset}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_scenes(scenes_arg: str) -> list[str]:
    """Accept either a path to a txt file (one scene per line) or
    comma-separated list."""
    if scenes_arg and os.path.isfile(scenes_arg):
        with open(scenes_arg) as f:
            return [
                ln.strip() for ln in f if ln.strip() and not ln.startswith("#")
            ]
    return [s.strip() for s in scenes_arg.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------


def process_scene(
    model: torch.nn.Module,
    adapter: _Adapter,
    scene_id: str,
    frames: list[FrameSpec],
    output_root: str,
    target_hw: tuple[int, int],
    num_tokens: int,
    batch_size: int,
    overwrite: bool,
    source_model: str,
    base_model: str,
    model_path: str,
) -> int:
    out_dir = os.path.join(output_root, scene_id)
    out_h5 = os.path.join(out_dir, "moge_signals.h5")
    if os.path.isfile(out_h5) and not overwrite:
        return -1  # signal: skipped

    if not frames:
        return 0

    H, W = target_hw
    N = len(frames)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-allocate datasets in the H5 (chunked + gzip) so memory stays bounded.
    # Write to a temp name and rename on success so a failed run cannot leave a
    # truncated moge_signals.h5 behind (which a rerun would skip as complete).
    tmp_h5 = out_h5 + ".tmp"
    with h5py.File(tmp_h5, "w") as f:
        f.attrs["dataset"] = adapter.name
        if adapter.split is not None:
            f.attrs["split"] = adapter.split
        f.attrs["resolution"] = f"{H}x{W}"
        f.attrs["num_tokens"] = int(num_tokens)
        f.attrs["source_model"] = source_model
        f.attrs["base_model"] = base_model
        f.attrs["model_path"] = model_path
        f.attrs["metric_depth_applied"] = True
        f.attrs["has_planarity"] = True

        d_planarity = f.create_dataset(
            "planarity",
            (N, H, W),
            dtype="float16",
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
        )
        d_normal = f.create_dataset(
            "normal",
            (N, H, W, 3),
            dtype="float16",
            chunks=(1, H, W, 3),
            compression="gzip",
            compression_opts=4,
        )
        d_depth = f.create_dataset(
            "depth_metric",
            (N, H, W),
            dtype="float16",
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
        )
        d_mask = f.create_dataset(
            "mask",
            (N, H, W),
            dtype="uint8",
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
        )
        d_K = f.create_dataset("intrinsics", (N, 3, 3), dtype="float32")
        d_scale = f.create_dataset("metric_scale", (N,), dtype="float32")
        d_fids = f.create_dataset(
            "frame_ids", (N,), dtype=h5py.string_dtype(encoding="utf-8")
        )

        device = next(model.parameters()).device
        for i in tqdm(
            range(0, N, batch_size),
            desc=f"  {scene_id}",
            leave=False,
            unit="batch",
        ):
            chunk = frames[i : i + batch_size]
            imgs = adapter.load_rgb_batch(chunk, target_hw=target_hw).to(device)

            sig = moge_forward_signals_planarity(
                model, imgs, num_tokens=num_tokens
            )

            sl = slice(i, i + len(chunk))
            d_planarity[sl] = sig["planarity"].astype(np.float16)
            d_normal[sl] = sig["normal"].astype(np.float16)
            d_depth[sl] = sig["depth_metric"].astype(np.float16)
            d_mask[sl] = sig["mask"]
            d_K[sl] = sig["intrinsics"]
            d_scale[sl] = sig["metric_scale"]
            for j, fs in enumerate(chunk):
                d_fids[i + j] = fs.frame_id

    os.replace(tmp_h5, out_h5)
    return N


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import contextlib

from pxwplanar.paths import planarity_hf_repo  # noqa: E402
from pxwplanar.paths import (
    planarity_model_path as DEFAULT_MODEL_PATH,  # noqa: E402
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["scannetpp", "nyuv2", "sevenscenes", "hypersim"],
        help="Which dataset adapter to use.",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root. "
        "scannetpp default: <scannetpp_path>/data; "
        "nyuv2 default: paths.nyuv2_path; "
        "sevenscenes default: paths.sevenscenes_path; "
        "hypersim default: paths.hypersim_path.",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="scannetpp: REQUIRED — txt path or comma-separated scene list. "
        "sevenscenes: OPTIONAL — comma-separated scene-name filter "
        "(default = all 7 scenes). hypersim: OPTIONAL — filter by "
        "scene id or scene/cam. nyuv2: ignored.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (nyuv2 default 'test', sevenscenes default 'val', "
        "hypersim default 'test' — resolved via splits/hypersim/<split>.txt).",
    )
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to a 4-head MoGe checkpoint (.pt) or HF repo id; must "
        "include planarity_head. Default falls back to the HF release "
        "when the local file is absent.",
    )
    ap.add_argument(
        "--base_model",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="HuggingFace base model id used to seed the architecture before "
        "loading the .pt checkpoint.",
    )
    ap.add_argument(
        "--resolution",
        type=str,
        default="480x640",
        help=(
            "HxW for MoGe input. "
            "Default 480x640 to match the legacy inference.h5."
        ),
    )
    ap.add_argument("--num_tokens", type=int, default=1600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Take every Nth frame from each scene (default 1 = every frame). "
        "Set 25 for the legacy scannetpp subsampling.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, re-process scenes whose moge_signals.h5 already exists.",
    )
    args = ap.parse_args()

    H, W = (int(x) for x in args.resolution.lower().split("x"))

    if not os.path.isfile(args.model_path):
        if args.model_path == DEFAULT_MODEL_PATH:
            # Default path not present locally: fall back to the HF release.
            print(
                f"[INFO] {args.model_path} not found — "
                f"using HF checkpoint {planarity_hf_repo}"
            )
            args.model_path = planarity_hf_repo
        elif args.model_path.endswith(".pt") or args.model_path.count("/") != 1:
            # Explicit checkpoint file (or not shaped like a HF repo id)
            ap.error(f"--model_path does not exist: {args.model_path}")
        # else: exactly one slash, no .pt — a HF repo id for from_pretrained

    # Build adapter (errors out on missing args).
    adapter = _build_adapter(args, target_hw=(H, W))
    scenes = adapter.list_scenes()
    if not scenes:
        ap.error("Adapter returned no scenes — nothing to do.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading 4-head checkpoint {args.model_path} ...")
    inference = MoGePlanarityInference.from_pretrained(
        args.model_path, device=device.type
    )
    inference.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference.model.eval()
    model = inference.model
    print(f"  has scale_head     = {hasattr(model, 'scale_head')}")
    print(f"  has planarity_head = {hasattr(model, 'planarity_head')}")
    print(f"  base_model         = {args.base_model}")

    os.makedirs(args.output_root, exist_ok=True)
    print(
        f"Dataset: {adapter.name}"
        + (f" (split={adapter.split})" if adapter.split else "")
    )
    print(f"Processing {len(scenes)} scene(s) → {args.output_root}")
    print(
        f"  resolution {H}x{W}, num_tokens {args.num_tokens}, "
        f"batch {args.batch_size}, step {args.frame_step}"
    )

    source_model = "MoGePlanarityInference (4-head)"

    total_frames = 0
    skipped = 0
    pbar = tqdm(scenes, desc="Scenes", unit="scene")
    for sid in pbar:
        pbar.set_postfix_str(sid)
        try:
            frames = adapter.list_frames(sid, frame_step=args.frame_step)
            n = process_scene(
                model=model,
                adapter=adapter,
                scene_id=sid,
                frames=frames,
                output_root=args.output_root,
                target_hw=(H, W),
                num_tokens=args.num_tokens,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                source_model=source_model,
                base_model=args.base_model,
                model_path=args.model_path,
            )
        except Exception as e:
            tqdm.write(f"[fail] {sid}: {e}")
            # clean up the partial temp file so it does not accumulate
            tmp = os.path.join(args.output_root, sid, "moge_signals.h5.tmp")
            if os.path.isfile(tmp):
                with contextlib.suppress(OSError):
                    os.remove(tmp)
            continue
        if n == -1:
            skipped += 1
            tqdm.write(
                f"[skip] {sid}: moge_signals.h5 exists (use --overwrite)"
            )
        elif n == 0:
            tqdm.write(f"[skip] {sid}: no frames found")
        else:
            total_frames += n

    print(
        f"\nDone. Wrote {total_frames} frames "
        f"({skipped} scene(s) skipped) to {args.output_root}"
    )
    # A run that wrote nothing and skipped nothing found no data at all —
    # fail loudly instead of exiting 0 (matters in SLURM arrays)
    if total_frames == 0 and skipped == 0:
        sys.exit(
            "[ERROR] 0 frames written and 0 scenes skipped — "
            "check the dataset root and --scenes list"
        )


if __name__ == "__main__":
    main()
