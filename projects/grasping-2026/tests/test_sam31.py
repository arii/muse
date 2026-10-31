"""Tests for the SAM 3.1 hosted-API path (mocked HTTP — no key needed)."""
import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

from perception.segment import (
    MetaAPISAMSegmenter, SAMSegmenter, text_from_instruction,
)


def _png_mask(h=8, w=10):
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[2:5, 3:7] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)[:500]

    def json(self):
        return self._payload


class _Req:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, json))
        return _Resp(self.payload)


@pytest.fixture
def seg():
    s = MetaAPISAMSegmenter(model="sam-3.1", api_key="test-key")
    s._requests = _Req({"data": [
        {"mask": np.zeros((4, 5)).tolist(), "score": 0.2},
        {"mask": _png_mask(4, 5), "score": 0.9},
    ]})
    return s


def test_text_from_instruction():
    assert text_from_instruction("grasp the red box in isolation") == "red box"
    assert text_from_instruction("pick up the blue cylinder") == "blue cylinder"
    assert text_from_instruction("grab a green cube without moving") == "green cube"


def test_segment_returns_top_scoring_bool_mask(seg):
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = seg.segment(rgb, {"text": "red box"})
    assert mask.dtype == bool
    assert mask.shape == (4, 5)
    assert mask[2:4, 3:5].all() and mask.sum() == 4


def test_segment_accepts_raw_string(seg):
    mask = seg.segment(np.zeros((4, 5, 3), dtype=np.uint8), "red box")
    assert mask.shape == (4, 5)


def test_segment_sends_text_and_model(seg):
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    seg.segment(rgb, {"text": "red box"})
    url, body = seg._requests.calls[0]
    assert url == "https://api.meta.ai/v1/segmentations"
    assert body["model"] == "sam-3.1" and body["text"] == "red box"
    assert body["image"].startswith("data:image/png;base64,")


def test_segment_boxes_prompt_rejected(seg):
    with pytest.raises(ValueError, match="text-prompt only"):
        seg.segment(np.zeros((4, 5, 3), dtype=np.uint8),
                    {"boxes": [[0, 0, 1, 1]]})


def test_missing_key_raises():
    os.environ.pop("META_API_KEY", None)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaAPISAMSegmenter(model="x")


def test_unrecognized_response_shape_raises(seg):
    seg._requests = _Req({"surprise": []})
    with pytest.raises(ValueError, match="Unrecognized"):
        seg.segment(np.zeros((4, 5, 3), dtype=np.uint8), {"text": "red box"})


def test_local_stub_untouched():
    with pytest.raises(NotImplementedError):
        SAMSegmenter().segment(np.zeros((2, 2, 3), dtype=np.uint8),
                               {"boxes": [[0, 0, 1, 1]]})
