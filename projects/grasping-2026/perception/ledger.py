"""Audit ledger for VLM API calls: every plan() invocation is recorded,
and identical inputs reuse a cached plan instead of calling the API again.

Why: VLM calls cost money and time. The same scene (same seed, same
instruction) should not cost a second API call. The ledger is also the
audit trail — it shows exactly what was sent and what came back.

Format: JSONL, one object per line, append-only. Each entry:
    timestamp      ISO-8601 UTC time of the call (or cache reuse)
    model          VLM model id
    instruction    user instruction (e.g. "grasp the red box in isolation")
    objects        scene objects [{index, color}] as the VLM saw them
    image_sha256   SHA-256 of the rendered RGB frame (scene fingerprint)
    prompt         full prompt text sent to the API
    response       raw reply text from the API (None on a cache hit)
    plan           parsed plan {grasp_target, push_away, rationale}
    usage          token usage {prompt_tokens, completion_tokens,
                    total_tokens} from the API (None on a cache hit)
    cache_hit      True if this entry reused a cached plan (no API call)
    cache_key      deterministic key for the (model, instruction, scene)

Cache key: version + model + instruction + objects + image hash +
prompt hash. Same key means the VLM saw the same scene, instruction,
prompt and plan schema, so the cached plan applies. A cache hit still
appends an entry (cache_hit=True) so the ledger shows every plan()
invocation, API call or not. Legacy-shaped stored plans are never
served (see is_canonical_plan), even if their key matches.

Location: episodes/vlm_ledger.jsonl by default (inside the episodes
volume, so it persists across container runs). Override with
NEBIUS_LEDGER_PATH. Set NEBIUS_LEDGER_PATH="" to disable the ledger
(no logging, no caching).
"""
import hashlib
import json
import os
import pathlib

import numpy as np


def image_hash(rgb):
    """SHA-256 of a rendered frame — the scene fingerprint."""
    return hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()


# Cache version: bump this whenever the plan schema, the prompt, or
# the meaning of a cached plan changes. Old keys must NOT be reused —
# a pre-v2 ledger entry (bare-integer push_away, no approach_side)
# would otherwise be served as if it were a v2 plan, defaulting to
# effort="clear" / approach_side="near" and making VLM and heuristic
# runs identical (2026-10-28 compare rerun: 0 API calls, all cache
# hits). The prompt hash covers prompt drift within a version.
CACHE_VERSION = "grasp-plan-v2"


def make_key(model, instruction, objects, img_hash, prompt=None):
    """Deterministic cache key for one VLM call.

    Includes the cache version and, when given, the full prompt text:
    the same scene/instruction under a different prompt or schema is
    a different call, not a cache hit.
    """
    obj_str = ",".join(f"{o['index']}:{o['color']}" for o in objects)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16] \
        if prompt else "no-prompt"
    raw = (f"{CACHE_VERSION}|{model}|{instruction}|{obj_str}|"
           f"{img_hash}|{prompt_hash}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def is_canonical_plan(plan):
    """True if a stored plan already has the v2 shape.

    Legacy entries (push_away as bare ints, or no approach_side) are
    NOT canonical: serving them would silently default effort/side.
    They are rejected by the labeller even if their key matches.
    """
    if not isinstance(plan, dict):
        return False
    if "grasp_target" not in plan or "approach_side" not in plan:
        return False
    pushes = plan.get("push_away", [])
    if not isinstance(pushes, list):
        return False
    for item in pushes:
        if not isinstance(item, dict):
            return False
        if "index" not in item or "effort" not in item:
            return False
    return True


class VLMCallLedger:
    """Append-only JSONL ledger with a cache lookup."""

    def __init__(self, path=None):
        path = path if path is not None else os.environ.get(
            "NEBIUS_LEDGER_PATH", "episodes/vlm_ledger.jsonl")
        self.path = pathlib.Path(path) if path else None
        self._index = None  # cache_key -> plan, built lazily

    @property
    def enabled(self):
        return self.path is not None

    def _load_index(self):
        """Build cache_key -> plan from the ledger file (once)."""
        if self._index is not None:
            return self._index
        self._index = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not entry.get("cache_hit") and entry.get("plan"):
                    self._index[entry["cache_key"]] = entry["plan"]
        return self._index

    def lookup(self, key):
        """Cached plan for this key, or None."""
        if not self.enabled:
            return None
        return self._load_index().get(key)

    def record(self, entry):
        """Append one entry to the ledger (creates dirs as needed)."""
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        # keep the in-memory index in sync so a later lookup in this
        # process sees plans recorded earlier in this process
        if not entry.get("cache_hit") and entry.get("plan"):
            self._load_index()[entry["cache_key"]] = entry["plan"]
