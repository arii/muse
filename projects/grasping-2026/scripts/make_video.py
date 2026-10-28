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


def _save_gif(gif_frames, out_path, fps):
    """Save GIF frames with ONE shared palette.

    Pillow's default GIF save quantizes each frame independently, so
    the same box color can map to different palette entries (and
    shift hue) from frame to frame, and the first frame's palette
    can drop a color that appears later. Quantizing every frame to
    the first frame's adaptive palette, with no dithering (flat
    MuJoCo colors, no gradients to dither), keeps colors stable
    across the video. The first frame is the initial scene, which
    contains every box color, the blue arm, and the grey table.
    """
    palette_img = gif_frames[0].quantize(
        colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [fr.quantize(palette=palette_img,
                             dither=Image.Dither.NONE)
                 for fr in gif_frames]
    quantized[0].save(
        out_path, save_all=True, append_images=quantized[1:],
        duration=int(1000 / fps), loop=0)
    return out_path


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
    bg = ((20, 95, 45) if status.get("fc")
          else (95, 75, 15) if status.get("caged")
          else (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, bar_h], fill=bg)
    d.text((6, 4), status_text(status), font=_load_font(14),
           fill=(240, 240, 240))
    return img


def make_multi_panel_gif(panels, out_path, fps=15, scale=0.75):
    """Stitch N frame sequences horizontally into one synchronized GIF.

    panels: list of dicts with keys:
        frames: non-empty list/array of RGB frames (H, W, 3)
        label: header title for the column
        caption: footer outcome string for the column
        frame_info: optional per-frame status list
    Shorter sequences hold their final frame and status while longer
    sequences keep moving.
    """
    if not panels or any(len(p["frames"]) == 0 for p in panels):
        raise ValueError("all panel frame sequences must be non-empty")
    scaled_frames = []
    infos = []
    for p in panels:
        imgs = [Image.fromarray(fr) for fr in p["frames"]]
        if scale != 1.0:
            imgs = [im.resize((int(im.width * scale), int(im.height * scale)))
                    for im in imgs]
        scaled_frames.append(imgs)
        infos.append(list(p["frame_info"]) if p.get("frame_info") else None)
    w, h = scaled_frames[0][0].size
    scaled_frames = [
        [im.resize((w, h)) if im.size != (w, h) else im for im in seq]
        for seq in scaled_frames
    ]
    n = max(len(seq) for seq in scaled_frames)
    for idx in range(len(scaled_frames)):
        scaled_frames[idx] += [scaled_frames[idx][-1]] * (
            n - len(scaled_frames[idx]))
        if infos[idx]:
            infos[idx] += [infos[idx][-1]] * (n - len(infos[idx]))

    bar_h, divider = 28, 4
    n_cols = len(panels)
    total_w = w * n_cols + divider * (n_cols - 1)
    total_h = bar_h * 2 + h
    font = _load_font(15 if n_cols <= 2 else 13)
    gif_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for c, p in enumerate(panels):
            x0 = c * (w + divider)
            frame_img = scaled_frames[c][i]
            if infos[c] and i < len(infos[c]):
                frame_img = overlay_status(frame_img.copy(), infos[c][i])
            tw = d.textlength(p["label"], font=font)
            d.text((x0 + (w - tw) / 2, 6), p["label"], font=font,
                   fill=(240, 240, 240))
            canvas.paste(frame_img, (x0, bar_h))
            d = ImageDraw.Draw(canvas)
            d.text((x0 + 6, bar_h + h + 6), p["caption"], font=font,
                   fill=(240, 240, 240))
        gif_frames.append(canvas)
    return _save_gif(gif_frames, out_path, fps)


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
    return _save_gif(gif_frames, out_path, fps)


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
    return _save_gif(gif_frames, out_path, fps)


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
