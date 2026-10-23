"""Episode visualizer: frames.npz -> animated GIF.

Usage:
    python scripts/make_video.py episodes/<stamp>/ [--out demo.gif]
                                                   [--fps 8] [--scale 0.75]

Reads frames.npz and meta.json from a logged episode and writes an
animated GIF with a caption bar (instruction, target, caged or not).
Pillow-only — no extra dependencies.

This is the v1 visualizer: offline rendering of logged episodes.
A live interactive viewer (mujoco.viewer) needs a display and doesn't
fit the headless container story, so it's out of scope for now.
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def load_episode(epi_dir):
    epi_dir = pathlib.Path(epi_dir)
    frames = np.load(epi_dir / "frames.npz")["frames"]
    meta = json.loads((epi_dir / "meta.json").read_text())
    return frames, meta


def caption_text(meta):
    status = "CAGED" if meta.get("caged") else "no cage"
    return (f"{meta.get('instruction', '')}  |  "
            f"target: {meta.get('target_color', '?')}  |  {status}")


def make_gif(frames, meta, out_path, fps=8, scale=0.75):
    draw_caption = True
    gif_frames = []
    for fr in frames:
        img = Image.fromarray(fr)
        if scale != 1.0:
            img = img.resize((int(img.width * scale),
                              int(img.height * scale)))
        if draw_caption:
            bar_h = 28
            canvas = Image.new("RGB", (img.width, img.height + bar_h),
                               (20, 20, 20))
            canvas.paste(img, (0, 0))
            d = ImageDraw.Draw(canvas)
            d.text((8, 7), caption_text(meta), fill=(240, 240, 240))
            img = canvas
        gif_frames.append(img)
    gif_frames[0].save(
        out_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / fps), loop=0)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="episodes/<stamp>/ directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--scale", type=float, default=0.75)
    args = ap.parse_args()

    frames, meta = load_episode(args.episode)
    out = pathlib.Path(args.out or
                       f"{pathlib.Path(args.episode).name}.gif")
    make_gif(frames, meta, out, fps=args.fps, scale=args.scale)
    print(f"wrote {out} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
