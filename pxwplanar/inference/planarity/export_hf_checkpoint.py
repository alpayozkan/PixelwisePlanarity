#!/usr/bin/env python3
"""Repackage the trained 4-head planarity checkpoint for Hugging Face.

The training checkpoints (``model_epoch1.pt``) store the weights under
``model_state_dict`` next to optimizer-side metadata, and rely on the loader
to rebuild the module tree by downloading the MoGe-2 base weights and
grafting a planarity head onto them (``MoGePlanarityInference``'s legacy
path). This script converts such a checkpoint into MoGe's *native*
``from_pretrained`` format::

    {'model_config': <base config + planarity_head entry>,
     'model':        <the 559 trained tensors>}

so ``MoGeModel.from_pretrained(<hf_repo_or_path>)`` builds the 4-head model
directly — one download, no base fetch, no head surgery.

The base config is taken from the MoGe-2 release checkpoint
(``Ruicheng/moge-2-vitl-normal``, cached by huggingface_hub); the
``planarity_head`` entry is a copy of the ``normal_head`` config, which is
exactly how the head was constructed at training time (MoGeModel.__init__
rewrites its final ``dim_out`` to 1 channel).

After writing, the export is verified by loading it back through
``MoGeModel.from_pretrained`` and comparing every tensor bitwise against the
source checkpoint.

Usage:
    python pxwplanar/inference/planarity/export_hf_checkpoint.py [--src <model_epoch1.pt>] --out <dir>
"""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402
from MoGe.moge.model.v2 import MoGeModel  # noqa: E402
from pxwplanar.paths import planarity_model_path  # noqa: E402

BASE_REPO = "Ruicheng/moge-2-vitl-normal"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=str, default=planarity_model_path,
                    help="Training checkpoint (default: paths.planarity_model_path)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory; writes <out>/model.pt")
    args = ap.parse_args()

    print(f"[LOAD] training checkpoint: {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    print(f"       {len(state_dict)} tensors; epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss'):.4f} val_accuracy={ckpt.get('val_accuracy'):.4f}")

    print(f"[LOAD] base model_config from {BASE_REPO}")
    base_path = hf_hub_download(BASE_REPO, filename="model.pt")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    config = dict(base["model_config"])
    # The planarity head is a normal-head ConvStack whose final dim_out
    # MoGeModel.__init__ rewrites to 1 channel.
    config["planarity_head"] = dict(config["normal_head"])

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "model.pt")
    print(f"[SAVE] {out_path}")
    torch.save({"model_config": config, "model": state_dict}, out_path)

    # ---- verification: rebuild through the loader everyone will use ----
    print("[VERIFY] MoGeModel.from_pretrained(export) vs source tensors")
    model = MoGeModel.from_pretrained(out_path)
    rebuilt = model.state_dict()
    src_keys, new_keys = set(state_dict), set(rebuilt)
    if src_keys != new_keys:
        sys.exit(f"[FAIL] key sets differ: missing={sorted(src_keys - new_keys)[:5]} "
                 f"extra={sorted(new_keys - src_keys)[:5]}")
    bad = [k for k in src_keys if not torch.equal(rebuilt[k], state_dict[k])]
    if bad:
        sys.exit(f"[FAIL] {len(bad)} tensors differ, e.g. {bad[:5]}")
    if not hasattr(model, "planarity_head"):
        sys.exit("[FAIL] rebuilt model has no planarity_head")
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"[OK] {len(src_keys)} tensors bitwise-identical; planarity_head present; "
          f"{size_gb:.2f} GB")


if __name__ == "__main__":
    main()
