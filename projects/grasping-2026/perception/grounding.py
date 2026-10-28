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
    "away first to isolate it? "
    "Your ENTIRE reply must be ONLY a JSON object, with no explanation, "
    "no reasoning steps, no markdown, and no text before or after it. "
    "Reply exactly like this (with your own values): "
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
        self.last_raw = None  # raw model reply text from the last plan() call
        self.last_response = None  # full API response JSON (for debugging)
        self.last_status = None  # HTTP status of the last API call
        self.model = model or os.environ.get("NEBIUS_VLM_MODEL", "")
        if not self.model:
            raise ValueError(
                "Set NEBIUS_VLM_MODEL to a Nemotron vision model id "
                "(see the Nebius model catalog).")
        self.api_key = api_key or os.environ["NEBIUS_API_KEY"]
        self.base_url = (base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        ).rstrip("/")

    def _diagnose_null_content(self, msg, data):
        """Build a diagnostic block explaining why content is None.

        Categorizes the failure so the user can fix it without guessing:
        reasoning model, tool calls, non-vision model, empty response, etc.
        """
        lines = [
            f"VLM returned no text content.",
            f"  model: {self.model}",
            f"  endpoint: {self.base_url}/chat/completions",
            f"  HTTP status: {self.last_status}",
            f"  message keys: {sorted(msg.keys())}",
        ]
        if msg.get("reasoning_content"):
            lines.append(
                "  -> reasoning model: output is in reasoning_content, "
                "not content. (This labeller now reads reasoning_content "
                "as a fallback — if you see this, that fallback also "
                "returned empty.)")
        if msg.get("tool_calls"):
            lines.append(
                f"  -> model returned tool_calls instead of text: "
                f"{json.dumps(msg['tool_calls'])[:300]}. The model wants "
                "to call a tool; this prompt expects a plain JSON reply.")
        if msg.get("refusal"):
            lines.append(f"  -> model refused: {msg['refusal'][:300]}")
        if not msg.get("content") and not msg.get("reasoning_content"):
            lines.append(
                "  -> content AND reasoning_content are both null. Likely "
                "causes: (a) NEBIUS_VLM_MODEL is not a vision model, or "
                "doesn't support image_url in chat completions; (b) the "
                "model id is wrong/retired; (c) the provider returned an "
                "empty completion. Verify the model id with "
                "GET /v1/models and check it lists vision support.")
        lines.append(f"  full message: {json.dumps(msg)[:1000]}")
        # usage hints (rate limits, token counts) if present
        usage = data.get("usage")
        if usage:
            lines.append(f"  usage: {json.dumps(usage)}")
        return "\n".join(lines)

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
                "max_tokens": 800, "temperature": 0},
            timeout=60)
        self.last_status = r.status_code
        r.raise_for_status()
        data = r.json()
        self.last_response = data
        msg = data["choices"][0]["message"]
        # Some models return content=None (reasoning models put output in
        # reasoning_content; non-vision models may null it entirely).
        text = msg.get("content") or msg.get("reasoning_content")
        self.last_raw = text
        if text is None:
            raise ValueError(self._diagnose_null_content(msg, data))
        # be liberal: find the first {...} in the reply
        try:
            start, end = text.index("{"), text.rindex("}") + 1
        except ValueError:
            # No JSON — try to salvage the answer from prose. The model
            # often writes "Grasp target: Red box (index 0)" and names
            # the crowding box. Extract indices from those sentences.
            prose_plan = self._parse_prose(text, objects)
            if prose_plan is not None:
                return prose_plan
            raise ValueError(
                f"VLM reply contained no JSON object (model={self.model}). "
                f"Raw reply:\n{text[:1000]}\n"
                "The model answered in prose instead of JSON. Likely "
                "causes: (a) the model is not instruction-tuned for JSON "
                "output; (b) max_tokens cut it off before the JSON; "
                "(c) the prompt's JSON example wasn't followed. Try a "
                "different model id, or raise max_tokens.")
        out = json.loads(text[start:end])
        valid = {o["index"] for o in objects}
        grasp = int(out["grasp_target"])
        push = [int(i) for i in out.get("push_away", [])]
        if grasp not in valid or any(i not in valid for i in push):
            raise ValueError(f"VLM returned out-of-range indices: {out}")
        return {"grasp_target": grasp, "push_away": push,
                "rationale": out.get("rationale", "")}

    def _parse_prose(self, text, objects):
        """Salvage a plan from a prose reply (no JSON).

        Looks for "grasp target" + an index or color, and "push" +
        an index or color. Returns None if no grasp target found.
        """
        import re
        colors = {o["color"]: o["index"] for o in objects}
        valid = set(colors.values())
        low = text.lower()

        def _idx_near(pattern, window=120):
            """Index/color mentioned within `window` chars after `pattern`.

            Tries every match of the pattern (the first mention is often
            in an intro sentence, far from the actual index) and returns
            the first one that yields an index or color.
            """
            for m in re.finditer(pattern, low):
                snippet = low[m.end():m.end() + window]
                mi = re.search(r"index\s+(\d+)", snippet)
                if mi and int(mi.group(1)) in valid:
                    return int(mi.group(1))
                for color, idx in colors.items():
                    if color in snippet:
                        return idx
            return None

        grasp = _idx_near(r"grasp target")
        if grasp is None:
            return None
        push = []
        for m in re.finditer(r"push", low):
            # color/index may be before ("blue box must be pushed") or
            # after ("push the blue box") the word "push" — take the
            # NEAREST mention, not the first color in dict order (the
            # window often contains the grasp target's color too).
            lo = max(0, m.start() - 60)
            hi = m.start() + 80
            window = low[lo:hi]
            push_pos = m.start() - lo
            best_idx, best_dist = None, None
            mi = re.search(r"index\s+(\d+)", window)
            if mi and int(mi.group(1)) in valid:
                # index mentions are precise; prefer the nearest one
                for mm in re.finditer(r"index\s+(\d+)", window):
                    if int(mm.group(1)) in valid:
                        d = abs(mm.start() - push_pos)
                        if best_dist is None or d < best_dist:
                            best_idx, best_dist = int(mm.group(1)), d
            if best_idx is None:
                for c, i in colors.items():
                    for mm in re.finditer(re.escape(c), window):
                        d = abs(mm.start() - push_pos)
                        if best_dist is None or d < best_dist:
                            best_idx, best_dist = i, d
            if best_idx is not None and best_idx != grasp \
                    and best_idx not in push:
                push.append(best_idx)
        return {"grasp_target": grasp, "push_away": push,
                "rationale": "parsed from prose (model did not return JSON)"}

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
