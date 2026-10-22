"""VLM grounding: natural-language instruction + top-down image -> target object.

NebiusVLMLabeller talks to an OpenAI-compatible chat-completions endpoint
(Nebius Token Factory by default). Set:
    NEBIUS_API_KEY   - your Token Factory key
    NEBIUS_BASE_URL  - default https://api.tokenfactory.nebius.com/v1  (verify!)
    NEBIUS_VLM_MODEL - e.g. a Nemotron vision model id (check the Nebius
                       model catalog — TODO: pin the exact id)

HeuristicLabeller matches color words so the demo loop runs fully offline.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

PROMPT = (
    "You see a top-down view of a robot arm and several colored boxes on a table. "
    "The boxes are: {objects}. The task instruction is: \"{instruction}\". "
    "Which box should the robot grasp? Reply with ONLY a JSON object like "
    '{"target_index": 2, "rationale": "short reason"}.'
)


def _rgb_to_data_url(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class NebiusVLMLabeller:
    def __init__(self, model=None, api_key=None, base_url=None):
        import requests  # deferred so offline use never needs it
        self._requests = requests
        self.model = model or os.environ.get("NEBIUS_VLM_MODEL", "")
        if not self.model:
            raise ValueError(
                "Set NEBIUS_VLM_MODEL to a Nemotron vision model id "
                "(see the Nebius model catalog).")
        self.api_key = api_key or os.environ["NEBIUS_API_KEY"]
        self.base_url = (base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        ).rstrip("/")

    def label(self, rgb, objects, instruction):
        desc = ", ".join(
            f"index {o['index']}: {o['color']} box" for o in objects)
        content = [
            {"type": "text",
             "text": PROMPT.format(objects=desc, instruction=instruction)},
            {"type": "image_url",
             "image_url": {"url": _rgb_to_data_url(np.asarray(rgb))}},
        ]
        r = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [
                {"role": "user", "content": content}],
                "max_tokens": 200, "temperature": 0},
            timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        # be liberal: find the first {...} in the reply
        start, end = text.index("{"), text.rindex("}") + 1
        out = json.loads(text[start:end])
        return int(out["target_index"]), out.get("rationale", "")


class HeuristicLabeller:
    """Offline fallback: match a color word in the instruction."""

    def label(self, rgb, objects, instruction):
        instr = instruction.lower()
        for o in objects:
            if o["color"] in instr:
                return o["index"], f"heuristic: matched '{o['color']}'"
        return objects[0]["index"], "heuristic: defaulted to first object"


def make_labeller():
    if os.environ.get("NEBIUS_API_KEY"):
        try:
            return NebiusVLMLabeller()
        except ValueError as e:
            print(f"[vlm] {e} -- falling back to heuristic")
    return HeuristicLabeller()
