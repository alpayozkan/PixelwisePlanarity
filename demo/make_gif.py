#!/usr/bin/env python3
"""
Assemble the per-frame combined montages from demo/outputs/ into an animated
GIF (demo/assets/demo.gif by default). Run demo/run_demo.py first.
"""
import os
import sys
import glob
import argparse
from pathlib import Path

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", type=str, default=str(DEMO_DIR / "outputs"))
    parser.add_argument("--out", type=str, default=str(DEMO_DIR / "assets" / "demo.gif"))
    parser.add_argument("--width", type=int, default=1440,
                        help="GIF width in pixels (montages are downscaled to this)")
    parser.add_argument("--duration", type=int, default=1500,
                        help="Per-frame display time in ms")
    args = parser.parse_args()

    montages = sorted(glob.glob(os.path.join(args.outputs_dir, "*", "combined.png")))
    if not montages:
        sys.exit(f"No combined.png found under {args.outputs_dir} — run demo/run_demo.py first")

    frames = []
    for path in montages:
        im = Image.open(path).convert("RGB")
        h = round(im.height * args.width / im.width)
        im = im.resize((args.width, h), Image.LANCZOS)
        frames.append(im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=args.duration, loop=0, optimize=True)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"{args.out}: {len(frames)} frames, {frames[0].width}x{frames[0].height}, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
