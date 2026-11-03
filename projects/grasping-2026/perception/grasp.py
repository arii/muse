"""Grasp synthesis: object mask (+ depth) -> ranked grasp candidates.

Approach (classical, no learning required for v1):
  1. Extract the object contour from the SAM mask.
  2. Sample antipodal point pairs on the contour (normals ~opposed).
  3. Score each pair by: gripper width fit, approach clearance, and
     distance from the mask centroid (prefer central grasps).
  4. Return grasps ranked by score.

Grasp representation: dict(x, y, angle, width, score) in image frame;
callers project to the robot frame using the camera model.

Implementation is numpy/PIL only (no cv2): boundary pixels come from
4-neighborhood erosion, normals from the blurred-mask gradient, and
pairs are kept when normals oppose (dot < -0.9) with the segment
midpoint inside the mask.
"""
import math

import numpy as np
from PIL import Image, ImageFilter


def _boundary_normals(mask):
    """Boundary pixel coords + outward unit normals. All numpy/PIL."""
    m = np.asarray(mask, dtype=bool)
    inside = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
              & m[1:-1, :-2] & m[1:-1, 2:])
    boundary = np.zeros_like(m)
    boundary[1:-1, 1:-1] = m[1:-1, 1:-1] & ~inside
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    smooth = np.asarray(Image.fromarray(
        (m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=2)), dtype=float) / 255.0
    gy, gx = np.gradient(smooth)
    n = np.stack([-gx[ys, xs], -gy[ys, xs]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.stack([xs, ys], axis=1).astype(float), n


def synthesize_grasps(mask, depth=None, n_candidates=50, seed=0,
                      n_samples=2000):
    """Rank grasp candidates on a boolean object mask.

    Samples boundary point pairs, keeps antipodal ones (normal dot
    < -0.9, chord midpoint inside the mask), scores by centrality
    plus sane width, and returns the top `n_candidates` dicts
    (x, y, angle, width, score) sorted by descending score. Empty
    list if no valid candidate is found. Deterministic for `seed`.
    """
    pts, normals = _boundary_normals(mask)
    if len(pts) < 2:
        return []
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    c = mask_centroid(mask)
    radius = math.sqrt(m.sum() / math.pi) + 1e-9
    rng = np.random.default_rng(seed)
    grasps = []
    seen = set()
    for _ in range(n_samples):
        i, j = rng.integers(0, len(pts), size=2)
        if i == j:
            continue
        if float(normals[i] @ normals[j]) >= -0.9:
            continue
        p1, p2 = pts[i], pts[j]
        mid = (p1 + p2) / 2.0
        mx, my = int(round(mid[0])), int(round(mid[1]))
        if not (0 <= mx < w and 0 <= my < h and m[my, mx]):
            continue
        width = float(np.linalg.norm(p1 - p2))
        if width < 1e-6:
            continue
        key = (round(mid[0], 1), round(mid[1], 1), round(width, 1))
        if key in seen:
            continue
        seen.add(key)
        centrality = -math.hypot(mid[0] - c[0], mid[1] - c[1]) / radius
        # Prefer widths near the object diameter, punish extremes.
        width_fit = -abs(width / (2.0 * radius) - 1.0)
        score = centrality + 0.5 * width_fit
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        grasps.append({"x": float(mid[0]), "y": float(mid[1]),
                       "angle": float(angle), "width": width,
                       "score": float(score)})
        if len(seen) > 4 * n_candidates:
            break
    grasps.sort(key=lambda g: g["score"], reverse=True)
    return grasps[:n_candidates]


def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
