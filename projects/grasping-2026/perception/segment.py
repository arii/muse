"""Segmentation stage: grounded detection -> pixel mask via SAM.

Interface:
    segmenter = SAMSegmenter(checkpoint="sam_vit_h_4b8939.pth")
    mask = segmenter.segment(rgb, prompt={"boxes": [[x1, y1, x2, y2]]})
    # mask: np.ndarray[bool], shape (H, W)

The `prompt` comes from the grounding stage (perception/grounding.py):
a bounding box or point derived from the VLM's answer.

Setup:
    pip install segment-anything
    # download a checkpoint from the SAM repo (ViT-H recommended)

Status: STUB — interface is fixed; local SAM wiring is TODO.
"""
import base64
import io
import os
import re

import numpy as np
from PIL import Image


class SAMSegmenter:
    def __init__(self, checkpoint=None, device="cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def _load(self):
        # TODO: lazy-load SamPredictor here once a checkpoint is available.
        # from segment_anything import SamPredictor, sam_model_registry
        # sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint)
        # sam.to(self.device); self._predictor = SamPredictor(sam)
        raise NotImplementedError(
            "SAM checkpoint not configured — set one up per the docstring.")

    def segment(self, rgb, prompt):
        """Return a boolean mask for the prompted object."""
        if self._predictor is None:
            self._load()
        raise NotImplementedError("SAM inference not yet wired.")


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


class MetaAPISAMSegmenter:
    """SAM 3.1 segmentation through Meta's hosted Model API.

    The API takes an image plus a short text phrase ("red bicycle") and
    returns bounding boxes with pixel-level masks — every instance of the
    concept at once. Prompt it with the object description the grounding
    stage already produces (text_from_instruction pulls "red box" out of
    "grasp the red box in isolation").

    Config:
        META_API_KEY  - Bearer key from the Meta Developer Console
        META_MODEL_ID - exact model id string; check your acceptance email,
                        or GET {META_BASE_URL}/models to list what's live
        META_BASE_URL - default https://api.meta.ai/v1

    Schema note: the request path and response field names are drafted
    from launch coverage of the API, not Meta's official reference.
    Run scripts/smoke_sam31.py to verify against the live API:
    _parse_response raises a descriptive error showing the raw keys when
    the shape doesn't match, so the first call tells you exactly what to
    fix.
    """

    _REQUEST_PATH = "/segmentations"  # VERIFY against Developer Console docs

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
                "Set META_MODEL_ID — the exact model id string from your "
                "acceptance email, or list models via GET "
                f"{self.base_url}/models with your key.")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    @staticmethod
    def _encode_image(rgb):
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(
            buf.getvalue()).decode("ascii")

    def segment(self, rgb, prompt):
        """Return the top-scoring bool mask for the prompted concept.

        prompt: {"text": "red box"}, or a raw string. The local-checkpoint
        box/point prompt shape does not apply here — the hosted API is
        text-prompt only, so a boxes prompt raises a clear error.
        """
        if isinstance(prompt, str):
            prompt = {"text": prompt}
        if "text" not in prompt:
            raise ValueError(
                "MetaAPISAMSegmenter needs prompt={'text': ...} — the "
                "hosted SAM 3.1 API is text-prompt only.")
        rgb = np.asarray(rgb, dtype=np.uint8)
        body = {
            "model": self.model,
            "image": self._encode_image(rgb),
            "text": prompt["text"],
        }
        r = self._requests.post(self.base_url + self._REQUEST_PATH,
                                headers=self._headers(), json=body,
                                timeout=60)
        if r.status_code != 200:
            raise RuntimeError(
                f"SAM 3.1 API returned {r.status_code}: {r.text[:500]}")
        return self._parse_response(r.json(), rgb.shape[:2])

    @staticmethod
    def _parse_response(data, shape):
        """Pick the top-scoring mask; return bool (H, W).

        VERIFY: accepted shapes are {"data": [{mask, score, bbox}]} and
        {"masks": [...]}; masks arrive as nested 0/1 lists or base64 PNG.
        """
        h, w = shape
        items = data.get("data", data.get("masks"))
        if not items:
            raise ValueError(
                "Unrecognized SAM 3.1 response shape — top-level keys: "
                + ", ".join(sorted(data.keys())))

        def _decode(mask):
            if isinstance(mask, str):  # base64 PNG
                raw = base64.b64decode(mask.split(",", 1)[-1])
                return np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 0
            return np.asarray(mask) > 0

        scored = [(_decode(it["mask"]), float(it.get("score", -1)))
                  for it in items if "mask" in it]
        if not scored:
            raise ValueError("No masks found in SAM 3.1 response items.")
        masks, scores = zip(*scored)
        if all(s < 0 for s in scores):  # no scores: keep the largest
            masks, scores = zip(*sorted(
                scored, key=lambda ms: ms[0].sum(), reverse=True))
        mask = np.asarray(masks[int(np.argmax(scores))])
        if mask.shape != (h, w):  # API returned a different resolution
            mask = np.asarray(
                Image.fromarray(mask).resize((w, h), Image.NEAREST))
        return mask.astype(bool)
