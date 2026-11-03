"""Tests for M3 grasp synthesis (numpy/PIL only, no fixtures)."""
import math

import numpy as np

from perception.grasp import mask_centroid, synthesize_grasps


def _rect_mask(h=60, w=80, box=(20, 15, 60, 45)):
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


def test_synthesize_finds_ranked_antipodal_grasps():
    grasps = synthesize_grasps(_rect_mask(), n_candidates=10)
    assert len(grasps) > 0
    scores = [g["score"] for g in grasps]
    assert scores == sorted(scores, reverse=True)
    for g in grasps:
        assert set(g) == {"x", "y", "angle", "width", "score"}
        assert g["width"] > 0
    # Top grasp spans the object (width near the 40px box width).
    assert grasps[0]["width"] > 20


def test_synthesize_midpoints_inside_mask():
    m = _rect_mask()
    for g in synthesize_grasps(m, n_candidates=20):
        assert m[int(round(g["y"])), int(round(g["x"]))]


def test_synthesize_empty_mask_gives_none():
    assert synthesize_grasps(np.zeros((10, 10), dtype=bool)) == []


def test_synthesize_deterministic():
    m = _rect_mask()
    a = synthesize_grasps(m, n_candidates=10, seed=1)
    b = synthesize_grasps(m, n_candidates=10, seed=1)
    assert [(g["x"], g["y"], g["score"]) for g in a] == \
        [(g["x"], g["y"], g["score"]) for g in b]


def test_mask_centroid():
    assert mask_centroid(_rect_mask()) == (39.5, 29.5)
    assert mask_centroid(np.zeros((4, 4), dtype=bool)) is None
