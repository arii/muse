"""Segmentation stage: grounded detection -> pixel mask via SAM.

Interface:
    segmenter = SAMSegmenter(checkpoint="sam_vit_h_4b8939.pth")
    mask = segmenter.segment(rgb, prompt={"boxes": [[x1, y1, x2, y2]]})
    # mask: np.ndarray[bool], shape (H, W)

The `prompt` comes from the grounding stage (perception/grounding.py):
a bounding box or point derived from the VLM's answer.

Setup:
    pip install segment-anything  # in requirements.lock
    # checkpoint (ViT-B, CPU-friendly) lives in models/, never committed:
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    export SAM_CHECKPOINT=models/sam_vit_b_01ec64.pth
"""
import asyncio
import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image


class SAMSegmenter:
    def __init__(self, checkpoint=None, device="cpu", model_type="vit_b"):
        self.checkpoint = (checkpoint or os.environ.get("SAM_CHECKPOINT"))
        self.device = device
        self.model_type = model_type
        self._predictor = None

    def _load(self):
        if not self.checkpoint:
            raise RuntimeError(
                "SAM checkpoint not configured — pass checkpoint= or set "
                "SAM_CHECKPOINT (see module docstring).")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as e:
            raise RuntimeError(
                "segment-anything is not installed (pip install "
                "segment-anything).") from e
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        self._predictor = SamPredictor(sam)

    def segment(self, rgb, prompt):
        """Return a boolean mask for the prompted object.

        prompt: {"boxes": [[x1, y1, x2, y2], ...]} (pixel coords) and/or
        {"points": [[x, y], ...], "point_labels": [1/0, ...]}.
        Returns the top-scoring mask as bool (H, W).
        """
        boxes = prompt.get("boxes") if isinstance(prompt, dict) else None
        points = prompt.get("points") if isinstance(prompt, dict) else None
        if boxes is None and points is None:
            raise ValueError(
                "SAMSegmenter needs prompt={'boxes': ...} and/or "
                "prompt={'points': ..., 'point_labels': ...}.")
        if self._predictor is None:
            self._load()
        rgb = np.asarray(rgb, dtype=np.uint8)
        self._predictor.set_image(rgb)
        kwargs = {}
        if boxes is not None:
            kwargs["box"] = np.asarray(boxes[0], dtype=float)
        if points is not None:
            kwargs["point_coords"] = np.asarray(points, dtype=float)
            kwargs["point_labels"] = np.asarray(
                prompt.get("point_labels", [1] * len(points)),
                dtype=int)
        masks, scores, _ = self._predictor.predict(**kwargs)
        return np.asarray(masks[int(np.argmax(scores))]).astype(bool)


_TEXT_PROMPT_VERBS = re.compile(
    r"^(?:grasp|pick(?:\s+up)?|grab|take|hold|lift)\s+(?:the|a|an)\s+",
    re.IGNORECASE)
_TEXT_PROMPT_QUALIFIERS = re.compile(
    r"\s+(?:in\s+isolation|without\s+moving.*|carefully|gently)\s*$",
    re.IGNORECASE)


def text_from_instruction(instruction):
    """Extract the object phrase for SAM 3.1 from a VLM-style instruction.

    "grasp the red box in isolation" -> "red box"
    """
    phrase = _TEXT_PROMPT_VERBS.sub("", instruction or "").strip()
    phrase = _TEXT_PROMPT_QUALIFIERS.sub("", phrase).strip()
    return phrase


def _sse_data_dicts(lines):
    """Yield parsed JSON payloads from raw SSE stream lines."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


async def _adapt_events(lines):
    """Yield parser-shaped event dicts from SSE lines.

    The API streams `response.content_part.done` but not
    `response.output_text.done`, which the parser uses to close a lane —
    synthesize it (same fix as Meta's official cookbook).
    """
    completed = set()
    for data in _sse_data_dicts(lines):
        lane = (data.get("item_id"), data.get("output_index"),
                data.get("content_index"))
        t = data.get("type")
        if t == "response.output_text.done":
            if lane not in completed:
                completed.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (t == "response.content_part.done" and lane not in completed
                and isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            completed.add(lane)
            yield {"type": "response.output_text.done",
                   "item_id": data.get("item_id"),
                   "output_index": data.get("output_index"),
                   "content_index": data.get("content_index"),
                   "text": part["text"]}


async def _parse_image_stream(lines):
    """Run the official parser over SSE lines -> ImageSegmentationResult."""
    from meta_sam_parser import (image_segmentation_format,
                                 parse_responses_stream)
    parsed = parse_responses_stream(_adapt_events(lines),
                                    image_segmentation_format())
    return await parsed.final_result()


class MetaAPISAMSegmenter:
    """SAM 3.1 segmentation through Meta's hosted Model API.

    Verified 2026-11-03 against the live API (dev.meta.ai docs + smoke test):
    SAM 3.1 is served over the Responses API — POST {base}/v1/responses with
    {"model": "sam-3.1", "stream": True, "input": [one user message carrying
    an input_text concept phrase and an input_image data URL]}. Segmentation
    arrives as special-token text in the output_text lane (one line per frame:
    <Nf>id<|box;...|><|mask;...|>,...), decoded with the official
    meta_sam_parser package. Mask rasters are box-relative: each mask is
    composited onto its box to build the full-frame mask.

    Config:
        META_API_KEY  - Bearer token from the Meta Developer Console
        META_MODEL_ID - exact model id string, e.g. "sam-3.1"
                        (GET {base}/models with your key lists what's live)
        META_BASE_URL - default https://api.meta.ai/v1
    """

    def __init__(self, model=None, base_url=None, api_key=None):
        import requests  # deferred so offline use never needs it
        self._requests = requests
        self.api_key = api_key or os.environ.get("META_API_KEY")
        self.model = model or os.environ.get("META_MODEL_ID")
        self.base_url = (base_url or os.environ.get("META_BASE_URL")
                         or "https://api.meta.ai/v1").rstrip("/")
        if not self.api_key:
            raise RuntimeError("Set META_API_KEY (Meta Developer Console).")
        if not self.model:
            raise RuntimeError(
                "Set META_MODEL_ID — e.g. 'sam-3.1' (list live models via "
                f"GET {self.base_url}/models with your key).")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    @staticmethod
    def _encode_image(rgb):
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "PNG")
        return ("data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii"))

    def _request_body(self, rgb, text):
        return {
            "model": self.model,
            "stream": True,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image",
                     "image_url": self._encode_image(rgb)},
                ],
            }],
        }

    def segment(self, rgb, prompt):
        """Return the largest-match bool mask for the prompted concept.

        prompt: {"text": "red box"}, or a raw string. The hosted API is
        text-prompt only, so a boxes prompt raises a clear error.
        Zero matches raises ValueError — a valid API outcome, but loud:
        the caller decides what "object absent" means.
        """
        if isinstance(prompt, str):
            prompt = {"text": prompt}
        if not isinstance(prompt, dict) or "text" not in prompt:
            raise ValueError(
                "MetaAPISAMSegmenter needs prompt={'text': ...} — the "
                "hosted SAM 3.1 API is text-prompt only.")
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        r = self._requests.post(
            self.base_url + "/responses", headers=self._headers(),
            json=self._request_body(rgb, prompt["text"]),
            timeout=120, stream=True)
        if r.status_code != 200:
            raise RuntimeError(
                f"SAM 3.1 API returned {r.status_code}: {r.text[:500]}")
        full_masks = self._parse_stream(r.iter_lines(), h, w)
        if not full_masks:
            raise ValueError(
                f"SAM 3.1 returned zero matches for {prompt['text']!r} — "
                "valid API outcome; the concept is not visible.")
        # Largest foreground wins (no confidence requested by default).
        return max(full_masks, key=lambda m: m.sum())

    def _parse_stream(self, lines, h, w):
        """Parse an SSE line stream into full-frame bool masks, one per match.

        Mask rasters are box-relative: each is resized to its box and
        composited onto a full-frame canvas.
        """
        from meta_sam_parser import decode_mask_to_raster
        result = asyncio.run(_parse_image_stream(lines))
        if result.outcome.status != "completed":
            raise RuntimeError(
                f"SAM 3.1 stream did not complete: {result.outcome.status}")
        errs = [d for d in result.diagnostics if d.severity == "error"]
        if errs:
            raise RuntimeError("SAM 3.1 output failed to parse: "
                               + "; ".join(d.message for d in errs))
        boxes = {rec.object_id: rec for rec in result.records
                 if rec.kind == "box"}
        full_masks = []
        for rec in result.records:
            if rec.kind != "mask":
                continue
            box = boxes.get(rec.object_id)
            if box is None:
                raise RuntimeError(
                    f"SAM 3.1 mask id={rec.object_id} has no paired box.")
            raster = np.frombuffer(
                decode_mask_to_raster(rec.mask),
                dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
            x1, y1 = max(0, box.left), max(0, box.top)
            x2, y2 = min(w, box.right), min(h, box.bottom)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            m = np.asarray(
                Image.fromarray((raster * 255).astype(np.uint8)).resize(
                    (bw, bh), Image.NEAREST)) > 0
            full = np.zeros((h, w), dtype=bool)
            full[y1:y2, x1:x2] = m
            full_masks.append(full)
        return full_masks
