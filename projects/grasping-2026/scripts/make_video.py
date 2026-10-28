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
from PIL import Image, ImageDraw, ImageFont


def _load_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def status_text(status):
    """One-line live grasp status for a frame overlay.

    status shape (from planning.primitives.frame_status): links =
    distinct arm links touching the target; enclosed/caged/fc booleans.
    """
    yn = lambda b: "Y" if b else "N"
    return (f"links:{status['links']} enc:{yn(status['enclosed'])} "
            f"cage:{yn(status['caged'])} FC:{yn(status['fc'])}")


def overlay_status(img, status):
    """Draw a live-status bar across the top of a frame image.

    Drawn ON the frame (not added), so sizes never change and existing
    layout math still holds.
    """
    bar_h = 20
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, bar_h], fill=(20, 20, 20))
    d.text((6, 4), status_text(status), font=_load_font(14),
           fill=(240, 240, 240))
    return img


def make_side_by_side_gif(left_frames, right_frames, left_label,
                          right_label, left_caption, right_caption,
                          out_path, fps=15, scale=0.75,
                          left_info=None, right_info=None):
    """Stitch two frame sequences side by side into one GIF.

    left_frames/right_frames: numpy RGB arrays (H, W, 3) from the same
    renderer, so same size (right is resized to left's size if not).
    The shorter sequence is padded with its last frame — the finished
    half holds its final pose while the other half keeps moving.
    A header strip labels each half; a footer strip shows each half's
    outcome; a divider separates the halves.
    left_info/right_info: optional per-frame status lists (same shape
    as planning.primitives.frame_status); each half's frames get their
    own live-status bar, padded with the last status like frames.
    """
    if not left_frames or not right_frames:
        raise ValueError("both frame sequences must be non-empty")
    left = [Image.fromarray(fr) for fr in left_frames]
    right = [Image.fromarray(fr) for fr in right_frames]
    if scale != 1.0:
        left = [im.resize((int(im.width * scale), int(im.height * scale)))
                for im in left]
        right = [im.resize((int(im.width * scale), int(im.height * scale)))
                 for im in right]
    # Match sizes (both come from the same renderer, but be safe).
    w, h = left[0].size
    right = [im.resize((w, h)) if im.size != (w, h) else im
             for im in right]
    # Pad the shorter sequence with its final frame (held pose),
    # and its status list with its final status.
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    if left_info:
        left_info = list(left_info) + [left_info[-1]] * (n - len(left_info))
    if right_info:
        right_info = list(right_info) + [right_info[-1]] * (
            n - len(right_info))

    bar_h, divider = 28, 4
    total_w = w * 2 + divider
    total_h = bar_h * 2 + h
    font = _load_font(16)
    gif_frames = []
    for i, (l, r) in enumerate(zip(left, right)):
        if left_info and i < len(left_info):
            l = overlay_status(l.copy(), left_info[i])
        if right_info and i < len(right_info):
            r = overlay_status(r.copy(), right_info[i])
        canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        # Header: label centered in each half.
        for label, x0 in ((left_label, 0), (right_label, w + divider)):
            tw = d.textlength(label, font=font)
            d.text((x0 + (w - tw) / 2, 6), label, font=font,
                   fill=(240, 240, 240))
        canvas.paste(l, (0, bar_h))
        canvas.paste(r, (w + divider, bar_h))
        # Footer: outcome caption per half, left-aligned.
        d = ImageDraw.Draw(canvas)
        d.text((8, bar_h + h + 6), left_caption, font=font,
               fill=(240, 240, 240))
        d.text((w + divider + 8, bar_h + h + 6), right_caption,
               font=font, fill=(240, 240, 240))
        gif_frames.append(canvas)
    gif_frames[0].save(
        out_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / fps), loop=0)
    return out_path


def load_episode(epi_dir):
    epi_dir = pathlib.Path(epi_dir)
    frames = np.load(epi_dir / "frames.npz")["frames"]
    meta = json.loads((epi_dir / "meta.json").read_text())
    return frames, meta


def caption_text(meta):
    # Prefer force_closure (the true grasp metric) when present;
    # fall back to caged for old episodes that predate it.
    passed = meta.get("force_closure", meta.get("caged"))
    status = "FORCE CLOSURE" if passed else "no grasp"
    return (f"{meta.get('instruction', '')}  |  "
            f"target: {meta.get('target_color', '?')}  |  {status}")


def make_gif(frames, meta, out_path, fps=8, scale=0.75, frame_info=None):
    draw_caption = True
    gif_frames = []
    for i, fr in enumerate(frames):
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
            # Live status goes in the bottom strip, NOT over the frame:
            # the caption text already occupies the top of the frame.
            if frame_info is not None and i < len(frame_info):
                d.text((8, img.height + 7), status_text(frame_info[i]),
                       font=_load_font(14), fill=(240, 240, 240))
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
