"""VLM grounding: natural-language instruction + top-down image -> action plan.

The task is framed as "grasp the target IN ISOLATION". The VLM reasons about
preconditions AND about how to act: which box to grasp, which boxes to push
away first (and how hard), and which way the arm should point as it reaches
in. E.g.:
    "grasp the red box in isolation"
-> {"grasp_target": 0,
    "push_away": [{"index": 1, "effort": "shove"}],
    "approach_side": "right",
    "rationale": "..."}.

The plan is semantic by design: the VLM never estimates coordinates. The
executor (scripts/demo_vlm_grasp.py) resolves each choice to geometry via
planning.primitives — effort to a push distance, approach_side to a cage
yaw — so a plan choice visibly changes the trajectory. Earlier the plan
only named objects and the executor ignored even that unless the word
"push" appeared in the instruction, which made VLM and heuristic runs
frame-for-frame identical.

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

# JSON schema for the VLM plan, enforced via response_format
# (Nebius Token Factory structured output). With strict mode the API
# guarantees the reply matches this shape — no prose, no parsing hacks.
# Docs: https://docs.tokenfactory.nebius.com/ai-models-inference/json
# Caveat: not every model supports JSON mode (check the model card for
# the "JSON mode" tag). If the API rejects response_format, plan()
# raises — pick a supporting model instead of degrading silently.
# Strict mode requires every property in `required`, including the
# nested push_away items.
EFFORTS = ("nudge", "clear", "shove")
# approach_side stays relative (near/far/left/right), NOT absolute
# +x/-x/+y/-y, on purpose: the cage yaw is rooted at the arm base, so
# the entry yaw must stay near the base-to-target direction to reach
# the target at all (see planning.primitives.approach_face). An
# absolute cardinal yaw would point the arm east/west/north/south
# regardless of where the target is and miss it whenever the target
# is not in that cardinal direction. The ambiguity the absolute
# proposal was fixing is instead removed in the prompt: near/far are
# anchored to the visible base (grey circle at the centre), and
# left/right are defined as image CCW/CW, which in this top-down
# frame (right=+x, up=+y) is absolute. An absolute approach POINT
# (tip offset from the target in +x/-x/+y/-y, via IK) would be a
# different executor, not a label change — not done here.
APPROACH_SIDES = ("near", "far", "left", "right")
PLAN_SCHEMA = {
    "name": "grasp_plan",
    "schema": {
        "type": "object",
        "properties": {
            "grasp_target": {
                "type": "integer",
                "description": "Index of the box to grasp",
            },
            "push_away": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Index of a box to push away",
                        },
                        "effort": {
                            "type": "string",
                            "enum": list(EFFORTS),
                            "description": "How hard to push it: nudge "
                                           "(barely crowding), clear "
                                           "(close to the target), "
                                           "shove (touching it)",
                        },
                    },
                    "required": ["index", "effort"],
                    "additionalProperties": False,
                },
                "description": "Boxes to push away first, in order, "
                               "to isolate the grasp target",
            },
            "approach_side": {
                "type": "string",
                "enum": list(APPROACH_SIDES),
                "description": "Which side the arm reaches in from, "
                               "relative to the target as seen in the "
                               "image: near = the side facing the "
                               "base (grey circle at the centre), "
                               "far = the opposite side, left = 90 "
                               "degrees counterclockwise from the "
                               "base-to-target direction as seen in "
                               "the image (image counterclockwise is "
                               "+x toward +y), right = 90 degrees "
                               "clockwise from that direction. Pick "
                               "the side whose sweep keeps the arm "
                               "clear of other boxes.",
            },
            "rationale": {
                "type": "string",
                "description": "Short reason for the choice",
            },
        },
        "required": ["grasp_target", "push_away", "approach_side",
                     "rationale"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT = (
    "You see a top-down view of a robot arm and several colored boxes on a table. "
    "The boxes are:\n{objects}\n\n"
    "Coordinate frame: top-down, right is +x, up is +y. The robot arm's "
    "base is the grey circle at the centre of the image. "
    "Goal: \"{instruction}\". "
    "Constraint: the grasp must occur IN ISOLATION — no other box within "
    "reach of the grasp target when the arm closes around it. "
    "Decide:\n"
    "1. grasp_target: index of the target box.\n"
    "2. push_away: ordered list of boxes to push away first, each with "
    "effort (nudge if barely crowding, clear if close to the target, "
    "shove if touching or nearly touching it). Empty if already isolated.\n"
    "3. approach_side: which side the arm reaches in from, relative to "
    "the target as seen in the image: near = the side facing the base, "
    "far = the opposite side, left = 90 degrees counterclockwise from "
    "the base-to-target direction as seen in the image (image "
    "counterclockwise is +x toward +y), right = 90 degrees clockwise "
    "from that direction. Pick the side whose sweep keeps the arm clear "
    "of other boxes.\n"
    "4. rationale: one short sentence explaining the choice.\n"
    "Output: do not output any prose, thoughts, or markdown outside the "
    "JSON object. Put any explanation strictly inside the 'rationale' "
    "field. Keep internal reasoning brief — this is a simple scene "
    "judgment, not a long derivation. No concrete example is given on "
    "purpose: the response_format schema defines the exact shape, and "
    "a concrete example would bias the indices toward its values."
)


def normalize_plan(out, valid):
    """Validate a raw plan dict into the canonical plan shape.

    Accepts the current shape and the legacy one (push_away as bare
    indices, no approach_side) so old ledger entries, cached plans,
    and hand-written plans keep working: a bare index means effort
    "clear", a missing approach_side means "near". Anything else
    invalid raises — no silent degradation.
    """
    grasp = int(out["grasp_target"])
    if grasp not in valid:
        raise ValueError(f"grasp_target out of range: {out}")
    pushes = []
    for item in out.get("push_away", []):
        if isinstance(item, dict):
            idx, effort = int(item["index"]), item.get("effort", "clear")
        else:
            idx, effort = int(item), "clear"
        if idx not in valid:
            raise ValueError(f"push_away index out of range: {out}")
        if effort not in EFFORTS:
            raise ValueError(f"unknown push effort {effort!r}: {out}")
        pushes.append({"index": idx, "effort": effort})
    side = out.get("approach_side", "near")
    if side not in APPROACH_SIDES:
        raise ValueError(f"unknown approach_side {side!r}: {out}")
    return {"grasp_target": grasp, "push_away": pushes,
            "approach_side": side,
            "rationale": out.get("rationale", "")}


def _rgb_to_data_url(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class NebiusVLMLabeller:
    def __init__(self, model=None, api_key=None, base_url=None,
                 ledger=None):
        import requests  # deferred so offline use never needs it
        self._requests = requests
        self.last_raw = None  # raw model reply text from the last plan() call
        self.last_response = None  # full API response JSON (for debugging)
        self.last_status = None  # HTTP status of the last API call
        self.last_cache_hit = None  # True if last plan() reused ledger cache
        self.model = model or os.environ.get("NEBIUS_VLM_MODEL", "")
        if not self.model:
            raise ValueError(
                "Set NEBIUS_VLM_MODEL to a Nemotron vision model id "
                "(see the Nebius model catalog).")
        self.api_key = api_key or os.environ["NEBIUS_API_KEY"]
        self.base_url = (base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        ).rstrip("/")
        # Audit ledger + cache (episodes/vlm_ledger.jsonl by default).
        # Pass ledger=False to disable, or a VLMCallLedger instance.
        if ledger is False:
            self.ledger = None
        elif ledger is not None:
            self.ledger = ledger
        else:
            from perception.ledger import VLMCallLedger
            self.ledger = VLMCallLedger()

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
        choices = data.get("choices") or []
        finish = choices[0].get("finish_reason") if choices else None
        if finish:
            lines.append(f"  finish_reason: {finish}")
        usage = data.get("usage") or {}
        completion = usage.get("completion_tokens")
        reasoning = usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens") if isinstance(
                usage.get("completion_tokens_details"), dict) else None
        if reasoning is None:
            reasoning = usage.get("reasoning_tokens")
        if finish == "length" or (completion is not None and
                                  reasoning == completion):
            lines.append(
                "  -> REASONING TOKEN EXHAUSTION: the model spent the "
                "entire completion budget on its thinking trace and "
                "never emitted the JSON answer (finish_reason=length, "
                f"reasoning_tokens={reasoning}, completion_tokens="
                f"{completion}). Fix: lower NEBIUS_REASONING_EFFORT "
                "(low/none) or raise NEBIUS_MAX_TOKENS — do not parse "
                "reasoning_content as a plan, it is the thinking trace, "
                "not the schema-constrained answer.")
        elif msg.get("reasoning_content"):
            lines.append(
                "  -> reasoning model: output is in reasoning_content, "
                "not content. With structured output this should not "
                "happen — the JSON belongs in content.")
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
        """Instruction -> canonical plan dict (see normalize_plan).

        Uses Nebius structured output (response_format json_schema, strict)
        so the API guarantees the reply matches PLAN_SCHEMA — no prose,
        no regex parsing. If the model/API rejects response_format (not
        every model supports JSON mode), plan() raises — pick a model
        with the "JSON mode" tag instead of degrading silently.

        Ledger: before calling the API, checks the audit ledger for an
        identical (model, instruction, scene) call and reuses the cached
        plan if found — no API call, no cost. Every invocation (API call
        or cache hit) is appended to the ledger.
        """
        from perception.ledger import image_hash, make_key, \
            is_canonical_plan
        import datetime
        desc = "\n".join(
            f"- index {o['index']}: {o['color']} box" for o in objects)
        prompt = PROMPT.format(objects=desc, instruction=instruction)
        img_hash = image_hash(rgb)
        key = make_key(self.model, instruction, objects, img_hash,
                       prompt=prompt)
        base_entry = {
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "model": self.model,
            "instruction": instruction,
            "objects": [{"index": o["index"], "color": o["color"]}
                        for o in objects],
            "image_sha256": img_hash,
            "prompt": prompt,
            "cache_key": key,
        }
        # Cache hit: same scene + instruction + prompt + schema seen
        # before — reuse, no API. A stored plan in legacy shape
        # (bare push_away ints, no approach_side) is NOT a hit: it
        # would default to clear/near and is exactly how the
        # 2026-10-28 rerun got 0 API calls and identical VLM/heuristic
        # trajectories. Fall through to a fresh API call instead.
        if self.ledger is not None and self.ledger.enabled:
            cached = self.ledger.lookup(key)
            if cached is not None and is_canonical_plan(cached):
                print(f"[vlm] ledger cache hit ({key}) — no API call")
                self.last_cache_hit = True
                self.ledger.record({**base_entry, "response": None,
                                    "plan": cached, "usage": None,
                                    "cache_hit": True})
                return cached
            if cached is not None:
                print(f"[vlm] ledger entry ({key}) is legacy-shaped — "
                      "ignoring cache, calling API")
        self.last_cache_hit = False
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": _rgb_to_data_url(np.asarray(rgb))}},
        ]
        # Reasoning models (e.g. Nemotron-3.5-Super-VL) spend tokens
        # on a thinking trace BEFORE the answer. max_tokens must cover
        # reasoning + JSON: 800 was consumed entirely by reasoning
        # (reasoning_tokens == max_tokens), leaving content null, and
        # 4096 still exhausted on the 2026-10-28 S5 rerun (07:44 UTC,
        # content null at the 4096 cap). Two levers, not one:
        # (1) reasoning_effort="low" — Nebius Token Factory supports
        # none/minimal/low/medium/high/xhigh, and this task is a
        # simple scene judgment, not a long derivation; (2) max_tokens
        # 8192, matching the Nebius example for this model. Both are
        # env-overridable (NEBIUS_REASONING_EFFORT, NEBIUS_MAX_TOKENS)
        # so the S5 can try "none" or a higher cap without a code
        # change. If the provider rejects reasoning_effort, plan()
        # raises — do NOT silently drop it, or the exhaustion returns
        # unnoticed. reasoning_content is never parsed as a plan:
        # it is the thinking trace, not the schema-constrained answer,
        # and parsing it would break the strict-output contract.
        max_tokens = int(os.environ.get("NEBIUS_MAX_TOKENS", "8192"))
        reasoning_effort = os.environ.get("NEBIUS_REASONING_EFFORT",
                                          "low")
        body = {"model": self.model, "messages": [
            {"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": reasoning_effort,
            "response_format": {"type": "json_schema",
                                "json_schema": PLAN_SCHEMA}}
        try:
            return self._plan_via_api(rgb, objects, base_entry, content,
                                      body)
        except Exception as e:
            # Failures are ledger entries too — they cost tokens and
            # the audit trail is incomplete without them. Never cached
            # (plan=None is not indexed), just recorded, then re-raised.
            if self.ledger is not None and self.ledger.enabled:
                usage = {}
                if self.last_response:
                    usage = self.last_response.get("usage") or {}
                self.ledger.record({**base_entry,
                                    "response": self.last_raw,
                                    "plan": None,
                                    "usage": {
                                        "prompt_tokens": usage.get(
                                            "prompt_tokens"),
                                        "completion_tokens": usage.get(
                                            "completion_tokens"),
                                        "total_tokens": usage.get(
                                            "total_tokens")},
                                    "cache_hit": False,
                                    "error": str(e)[:500]})
            raise

    def _plan_via_api(self, rgb, objects, base_entry, content, body):
        """One API call + parse. Raises on any failure (caller logs it)."""
        r = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=60)
        self.last_status = r.status_code
        # No retry without response_format: if the model/API rejects
        # structured output, that's a model-selection finding, not
        # something to silently degrade around. Pick a model with the
        # "JSON mode" tag. raise_for_status surfaces the 400.
        r.raise_for_status()
        data = r.json()
        self.last_response = data
        msg = data["choices"][0]["message"]
        # With structured output, content holds the JSON. No
        # reasoning_content fallback: if content is None, that's a
        # real failure, not something to paper over.
        text = msg.get("content")
        self.last_raw = text
        if text is None:
            raise ValueError(self._diagnose_null_content(msg, data))
        # be liberal: find the first {...} in the reply (with structured
        # output this is the whole reply; without it, it may be embedded)
        try:
            start, end = text.index("{"), text.rindex("}") + 1
        except ValueError:
            raise ValueError(
                f"VLM reply contained no JSON object (model={self.model}). "
                f"Raw reply:\n{text[:1000]}\n"
                "The model answered in prose instead of JSON. If you are "
                "seeing this with response_format enabled, the model does "
                "not support structured output — pick a model with the "
                "'JSON mode' tag, or accept the heuristic fallback.")
        out = json.loads(text[start:end])
        valid = {o["index"] for o in objects}
        plan = normalize_plan(out, valid)
        if self.ledger is not None and self.ledger.enabled:
            usage = data.get("usage") or {}
            self.ledger.record({**base_entry, "response": text,
                                "plan": plan,
                                "usage": {
                                    "prompt_tokens": usage.get(
                                        "prompt_tokens"),
                                    "completion_tokens": usage.get(
                                        "completion_tokens"),
                                    "total_tokens": usage.get(
                                        "total_tokens")},
                                "cache_hit": False})
        return plan

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
        return {"grasp_target": grasp_idx,
                "push_away": [{"index": i, "effort": "clear"}
                              for i in push_idxs],
                "approach_side": "near",
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
