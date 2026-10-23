"""VLM grounding: natural-language instruction + top-down image -> action plan.

The task is framed as "grasp the target IN ISOLATION". The VLM reasons about
preconditions: which box to grasp, and which boxes must be pushed away first
for the grasp to succeed. E.g.:
    "grasp the red box in isolation"
-> {"grasp_target": 0, "push_away": [1], "rationale": "..."}.

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
import re

import numpy as np
from PIL import Image

PROMPT = (
    "You see a top-down view of a robot arm and several colored boxes on a table. "
    "The boxes are: {objects}. "
    "The goal is: \"{instruction}\". "
    "The grasp must happen IN ISOLATION — no other box within reach of the "
    "grasp target when the arm closes around it. "
    "What preconditions must be satisfied before grasping can succeed? "
    "Which box is the grasp target, and are there boxes that must be pushed "
    "away first to isolate it? Reply with ONLY a JSON object like "
    '\'{{"grasp_target": 2, "push_away": [0], "rationale": "short reason"}}\'. '
    "Use an empty push_away list if no preconditions are needed."
    # NOTE: doubled braces — they are literal in the formatted prompt.
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
        self.last_raw = None  # raw model reply from the last label() call
        self.model = model or os.environ.get("NEBIUS_VLM_MODEL", "")
        if not self.model:
            raise ValueError(
                "Set NEBIUS_VLM_MODEL to a Nemotron vision model id "
                "(see the Nebius model catalog).")
        self.api_key = api_key or os.environ["NEBIUS_API_KEY"]
        self.base_url = (base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        ).rstrip("/")

    def plan(self, rgb, objects, instruction):
        """Instruction -> {"grasp_target": idx, "push_away": [idxs], "rationale": str}."""
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
        self.last_raw = text
        # be liberal: find the first {...} in the reply
        start, end = text.index("{"), text.rindex("}") + 1
        out = json.loads(text[start:end])
        valid = {o["index"] for o in objects}
        grasp = int(out["grasp_target"])
        push = [int(i) for i in out.get("push_away", [])]
        if grasp not in valid or any(i not in valid for i in push):
            raise ValueError(f"VLM returned out-of-range indices: {out}")
        return {"grasp_target": grasp, "push_away": push,
                "rationale": out.get("rationale", "")}

    def label(self, rgb, objects, instruction):
        """Back-compat: (grasp_target_index, rationale)."""
        plan = self.plan(rgb, objects, instruction)
        return plan["grasp_target"], plan["rationale"]


class HeuristicLabeller:
    """Offline fallback: match color words in the instruction.

    Understands "grasp/pick up/grab the <color>" and
    "push the <color> ... away".
    """

    def plan(self, rgb, objects, instruction):
        instr = instruction.lower()
        colors = {o["color"]: o["index"] for o in objects}
        grasp_idx = None
        m = re.search(r"(grasp|pick up|grab)\s+(the\s+)?(\w+)", instr)
        if m and m.group(3) in colors:
            grasp_idx = colors[m.group(3)]
        push_idxs = []
        for m in re.finditer(r"push\s+(the\s+)?(\w+)", instr):
            if m.group(2) in colors and colors[m.group(2)] not in push_idxs:
                push_idxs.append(colors[m.group(2)])
        if grasp_idx is None:
            for color, idx in colors.items():
                if color in instr:
                    grasp_idx = idx
                    break
            else:
                grasp_idx = objects[0]["index"]
        # don't push the grasp target itself
        push_idxs = [i for i in push_idxs if i != grasp_idx]
        rationale = f"heuristic: grasp '{objects[grasp_idx]['color']}'"
        if push_idxs:
            pushed = ", ".join(f"'{objects[i]['color']}'" for i in push_idxs)
            rationale += f", push away {pushed}"
        return {"grasp_target": grasp_idx, "push_away": push_idxs,
                "rationale": rationale}

    def label(self, rgb, objects, instruction):
        """Back-compat: (grasp_target_index, rationale)."""
        plan = self.plan(rgb, objects, instruction)
        return plan["grasp_target"], plan["rationale"]


def make_labeller():
    if os.environ.get("NEBIUS_API_KEY"):
        try:
            return NebiusVLMLabeller()
        except ValueError as e:
            print(f"[vlm] {e} -- falling back to heuristic")
    return HeuristicLabeller()
