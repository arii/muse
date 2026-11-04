"""Tests for the SAM 3.1 hosted-API path (mocked HTTP — no key needed).

The streaming fixture is a recorded real response: tests/fixtures/
sam31_red_square.sse captured 2026-11-03 from POST /v1/responses with
concept "red square" on a 160x120 image (red square at rows 40:80,
cols 60:100). Parsing is deterministic; no network, no key.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from perception.segment import (
    MetaAPISAMSegmenter, SAMSegmenter, text_from_instruction,
)

_FIXTURE = (Path(__file__).parent / "fixtures" / "sam31_red_square.sse"
            ).read_bytes().splitlines(keepends=True)


class _FakeStreamResp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.text = "" if status == 200 else f"error {status}"

    def iter_lines(self):
        return iter(self._lines)


class _FakeRequests:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None, stream=None):
        self.calls.append({"url": url, "headers": headers, "json": json,
                           "timeout": timeout, "stream": stream})
        return self._resp


def _seg(lines=_FIXTURE):
    s = MetaAPISAMSegmenter(model="sam-3.1", api_key="test-key")
    s._requests = _FakeRequests(_FakeStreamResp(lines))
    return s


def test_text_from_instruction():
    assert text_from_instruction("grasp the red box in isolation") == "red box"
    assert text_from_instruction("pick up the blue cylinder") == "blue cylinder"
    assert text_from_instruction("grab a green cube without moving") == "green cube"


def test_segment_returns_full_frame_bool_mask():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = _seg().segment(rgb, {"text": "red square"})
    assert mask.dtype == bool
    assert mask.shape == (120, 160)
    # Recorded truth: the 40x40 red square at rows 40:80, cols 60:100.
    assert mask.sum() == 1600
    assert mask[40:80, 60:100].all()


def test_segment_accepts_raw_string():
    mask = _seg().segment(np.zeros((120, 160, 3), dtype=np.uint8),
                          "red square")
    assert mask.shape == (120, 160)


def test_segment_posts_to_responses_api():
    seg = _seg()
    seg.segment(np.zeros((120, 160, 3), dtype=np.uint8), {"text": "red square"})
    call = seg._requests.calls[0]
    assert call["url"] == "https://api.meta.ai/v1/responses"
    assert call["stream"] is True
    assert call["headers"]["Authorization"] == "Bearer test-key"
    body = call["json"]
    assert body["model"] == "sam-3.1" and body["stream"] is True
    (msg,) = body["input"]
    assert msg["type"] == "message" and msg["role"] == "user"
    text_part, img_part = msg["content"]
    assert text_part == {"type": "input_text", "text": "red square"}
    assert img_part["type"] == "input_image"
    assert img_part["image_url"].startswith("data:image/png;base64,")


def test_segment_boxes_prompt_rejected():
    with pytest.raises(ValueError, match="text-prompt only"):
        _seg().segment(np.zeros((4, 5, 3), dtype=np.uint8),
                       {"boxes": [[0, 0, 1, 1]]})


def test_missing_key_raises():
    os.environ.pop("META_API_KEY", None)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaAPISAMSegmenter(model="sam-3.1")


def test_missing_model_raises():
    os.environ.pop("META_MODEL_ID", None)
    with pytest.raises(RuntimeError, match="META_MODEL_ID"):
        MetaAPISAMSegmenter(api_key="test-key")


_ZERO_SSE = (b'data: {"type": "response.output_text.delta", "item_id": "i",'
             b' "output_index": 0, "content_index": 0, "delta": ""}\n'
             b'\n'
             b'data: {"type": "response.content_part.done", "item_id": "i",'
             b' "output_index": 0, "content_index": 0,'
             b' "part": {"type": "output_text", "text": ""}}\n'
             b'\n'
             b'data: {"type": "response.completed",'
             b' "response": {"id": "r", "status": "completed"}}\n'
             ).splitlines(keepends=True)


def test_zero_matches_raises_loudly():
    # Zero matches is a valid API outcome; segment() refuses to guess.
    with pytest.raises(ValueError, match="zero matches"):
        _seg(_ZERO_SSE).segment(np.zeros((120, 160, 3), dtype=np.uint8),
                                {"text": "purple elephant"})


def test_http_error_raises_with_status():
    seg = MetaAPISAMSegmenter(model="sam-3.1", api_key="test-key")
    seg._requests = _FakeRequests(_FakeStreamResp([], status=500))
    with pytest.raises(RuntimeError, match="500"):
        seg.segment(np.zeros((4, 5, 3), dtype=np.uint8), {"text": "red square"})


def test_local_bad_prompt_rejected():
    seg = SAMSegmenter(checkpoint="nonexistent.pth")
    with pytest.raises(ValueError, match="boxes"):
        seg.segment(np.zeros((2, 2, 3), dtype=np.uint8), {"text": "red"})


def test_local_missing_checkpoint_raises(monkeypatch):
    monkeypatch.delenv("SAM_CHECKPOINT", raising=False)
    seg = SAMSegmenter(checkpoint=None)
    with pytest.raises(RuntimeError, match="SAM_CHECKPOINT"):
        seg.segment(np.zeros((4, 5, 3), dtype=np.uint8),
                    {"boxes": [[0, 0, 1, 1]]})


@pytest.mark.skipif(not os.environ.get("SAM_CHECKPOINT"),
                    reason="needs SAM_CHECKPOINT (358MB binary, not in CI)")
def test_local_segments_canary_box():
    """Live SAM: box-prompted mask covers the red box on seed 2."""
    from env.whole_arm_grasp import WholeArmGraspEnv
    env = WholeArmGraspEnv(seed=2)
    env.reset(seed=2)
    rgb = env.render()
    h, w = rgb.shape[:2]
    # Red pixels bound the target: prompt with their bbox (+margin).
    red = ((rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int) > 60)
           & (rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int) > 60))
    ys, xs = np.nonzero(red)
    assert len(xs) > 50, "canary frame has no red box?"
    box = [max(0, xs.min() - 5), max(0, ys.min() - 5),
           min(w, xs.max() + 5), min(h, ys.max() + 5)]
    mask = SAMSegmenter().segment(rgb, {"boxes": [box]})
    assert mask.dtype == bool and mask.shape == (h, w)
    # Mask must substantially cover the red pixels (IoU-style check).
    inter = (mask & red).sum()
    union = (mask | red).sum()
    assert inter / union > 0.5, f"mask/red IoU {inter / union:.2f}"
