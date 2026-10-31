"""Smoke-test SAM 3.1 through the Meta Model API.

Lists live models first (OpenAI-compatible /models), then segments a
synthetic red square on white and saves the mask.

    META_API_KEY=... META_MODEL_ID=... python scripts/smoke_sam31.py

The first call tells you exactly what's wrong: a missing env var fails
here, and an unrecognized response shape raises from
MetaAPISAMSegmenter._parse_response showing the raw keys.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.segment import MetaAPISAMSegmenter, text_from_instruction  # noqa: E402


def list_models(seg):
    r = seg._requests.get(seg.base_url + "/models",
                           headers=seg._headers(), timeout=30)
    r.raise_for_status()
    ids = [m.get("id") for m in r.json().get("data", [])]
    print(f"[smoke] {len(ids)} models live on {seg.base_url}:")
    for i in ids:
        print(f"  - {i}")
    return ids


def main():
    seg = MetaAPISAMSegmenter()  # raises clearly if env is missing
    list_models(seg)

    rgb = np.full((120, 160, 3), 255, dtype=np.uint8)
    rgb[40:80, 60:100] = (220, 40, 40)  # red square
    instruction = "grasp the red square in isolation"
    mask = seg.segment(rgb, {"text": text_from_instruction(instruction)})
    print(f"[smoke] mask shape {mask.shape}, dtype {mask.dtype}, "
          f"coverage {mask.mean():.3f}")
    out = "/tmp/sam31_smoke_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(out)
    print(f"[smoke] mask saved to {out}")


if __name__ == "__main__":
    main()
