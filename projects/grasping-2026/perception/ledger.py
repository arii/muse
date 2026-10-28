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

Cache key: model + instruction + objects + image hash. Same key means
the VLM saw the same scene and instruction, so the cached plan applies.
A cache hit still appends an entry (cache_hit=True) so the ledger shows
every plan() invocation, API call or not.

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


def make_key(model, instruction, objects, img_hash):
    """Deterministic cache key for one VLM call."""
    obj_str = ",".join(f"{o['index']}:{o['color']}" for o in objects)
    raw = f"{model}|{instruction}|{obj_str}|{img_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
