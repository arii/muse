# VLM grounding, worked end to end

This is the tutorial for the first stage of the pipeline: turning a
natural-language instruction and a camera image into *which object to grasp*.
Everything here runs against the real code in `perception/grounding.py`.

## The VLM's one job

The pipeline is a hybrid: the VLM handles **semantics** (which thing the
words refer to) and classical code handles **geometry** (where it is, how to
move). Grounding is the seam between them. Its contract is tiny:

```
(top-down RGB image, object list, instruction) -> target_index
```

It never outputs poses, grasps, or joint angles. One integer. That narrow
contract is what makes it swappable and testable.

## Honest status

`make_labeller()` picks `NebiusVLMLabeller` when `NEBIUS_API_KEY` is set and
falls back to `HeuristicLabeller` (color-word matching) otherwise. The demo,
the tests, and CI all run on the heuristic today — the VLM path is wired but
has never been exercised against a live model. It needs two things:

1. `NEBIUS_API_KEY` — a Token Factory key (the free Nebius AI Builder
   Program includes Token Factory credit).
2. `NEBIUS_VLM_MODEL` — a pinned vision model id, e.g.
   `nvidia/Nemotron-3.5-Super-VL` (verify against `GET /v1/models`;
   the catalog changes).

## The problem setup: goal, preconditions, gate

The task isn't "grasp the red box" — it's "grasp the red box **in isolation**".
That framing matters because it turns an open-ended instruction into a
well-posed problem with three parts:

1. **Goal.** What the user wants: `grasp the red box in isolation`. The
   "in isolation" is a success criterion, not decoration — it tells the
   system what "done" looks like.
2. **Preconditions.** Before grasping can succeed, the VLM reasons about
   what must be true first: which boxes crowd the target and must be pushed
   away. This is the `push_away` list in the plan — not a user command, but
   the VLM's own scene analysis. The prompt explicitly asks: "What
   preconditions must be satisfied before grasping can succeed?"
3. **Gate.** After the pushes, the demo checks the `isolation` metric
   against a threshold (0.15 m). Only if the preconditions are met does the
   grasp start. The result (`preconditions_met`) is logged to every episode.

This is deliberately not full loop closure (no re-planning mid-execution).
It's the minimal structure that makes the problem well-posed: the VLM
decides *what must happen*, the metric decides *whether it happened*, and
the controllers execute. If the preconditions fail, the episode log says so
instead of silently attempting a doomed grasp.

## The data flow, step by step

`NebiusVLMLabeller.plan(rgb, objects, instruction)` returns
`{"grasp_target": idx, "push_away": [idxs], "rationale": str}`:

1. **Render.** `rgb` is the 480×640 top-down MuJoCo frame. Top-down matters:
   the scene is planar, so a single orthographic-ish view shows every
   object with minimal occlusion.
2. **Describe the candidates as text.** The object list becomes
   `"index 0: red box, index 1: blue box, ..."`. The VLM does not discover
   objects — it *selects among known indices*. This removes a whole class of
   hallucination (invented objects) at the cost of trusting the simulator's
   object list. In the real world this list would come from a detector.
3. **Build the prompt.** The template in `PROMPT`:
   - states the scene ("top-down view of a robot arm and several colored
     boxes"),
   - lists the candidates,
   - quotes the instruction verbatim,
   - demands **only** a JSON object
     `{"grasp_target": N, "push_away": [M], "rationale": "..."}` —
     which box to grasp, and which to push away first to isolate it.
4. **Send image + text.** The frame is PNG-encoded, base64'd, and sent as an
   `image_url` data URL alongside the text in one OpenAI-compatible
   `chat/completions` call (`temperature=0`, `max_tokens=200`).
5. **Parse liberally, validate strictly.** The reply's first `{...}`
   substring is JSON-decoded (models add prose despite being told not to),
   but the indices are validated against the known objects — an
   out-of-range index raises `ValueError` instead of guessing.

`label()` is kept for back-compat and returns `(grasp_target, rationale)`.
The heuristic fallback (`HeuristicLabeller.plan`) parses
"grasp/pick up/grab the \<color\>" and "push the \<color\> ... away" with
regexes, so the full instruction-driven loop runs offline.

## Worked example

Instruction: `"grasp the red box, first push the blue box away"`, seed 0
(objects: red, blue, green, yellow).

The exact text sent (besides the image):

```
You see a top-down view of a robot arm and several colored boxes on a table.
The boxes are: index 0: red box, index 1: blue box, index 2: green box,
index 3: yellow box. The task instruction is: "grasp the red box, first
push the blue box away". Which box should the robot grasp? Are there boxes
it should push away first to isolate the grasp target? Reply with ONLY a
JSON object like '{"grasp_target": 2, "push_away": [0],
"rationale": "short reason"}'. Use an empty push_away list if no pushing
is needed.
```

A good reply:

```
{"grasp_target": 0, "push_away": [1], "rationale": "push blue aside, grasp red"}
```

`plan()` returns
`{"grasp_target": 0, "push_away": [1], "rationale": "push blue aside, grasp red"}`
and stashes the raw reply on `labeller.last_raw` for inspection. The demo
then singulates exactly the listed objects; "without moving the red box" in
the instruction switches the push-revert to strict mode (target must stay
within 10 cm).

## Design decisions worth knowing

- **temperature=0.** Grounding is a classification task; sampling adds
  nothing but flakiness.
- **Candidates as text, not just pixels.** The model could probably pick the
  red box from pixels alone, but naming the indices in text pins the output
  vocabulary to valid choices.
- **Rationale field.** Cheap and useful: it makes wrong answers debuggable
  ("chose index 2 because...") instead of silent.
- **No retries in the labeller.** One call, one parse. Retry policy belongs
  to the caller, which knows whether a re-ask is safe.

## Failure modes

| Symptom | Cause | What the code does |
|---|---|---|
| `ValueError` from `text.index("{")` | reply had no JSON (refusal, API error text) | raises — caller must handle |
| Wrong `target_index` | model misread color / occlusion | returned as-is; rationale helps diagnose |
| Index out of range | hallucinated index | **not checked** — downstream `env.objects[idx]` would `IndexError` |
| Timeout (60 s) | slow model / network | `requests` raises; no retry |
| 401/404 from Token Factory | bad key / wrong model id | `raise_for_status()` raises |

The out-of-range gap is real and small: validate
`0 <= idx < len(objects)` before trusting the answer.

## Making sure it works

Two layers, matching the two things that can break:

1. **The code path** (no key needed, runs in CI):
   `tests/test_pipeline.py` mocks the HTTP layer and asserts the request
   shape (URL, auth header, model id, text+image content) and that a noisy
   reply (`Sure! {...} done.`) still parses — plus that a non-JSON reply
   raises instead of guessing.
2. **The live model** (needs the key + model id):
   `scripts/smoke_vlm.py` renders scenes with known colors, asks the VLM,
   and checks it picks correctly across 5 seed/instruction cases. It prints
   the exact prompt and raw reply for the first case, so you see precisely
   what the model saw and said.

```bash
NEBIUS_API_KEY=... NEBIUS_VLM_MODEL=nvidia/Nemotron-3.5-Super-VL \
  python scripts/smoke_vlm.py
```

## Swapping models

Only `NEBIUS_VLM_MODEL` changes. Candidates on Token Factory:

- `nvidia/Nemotron-3.5-Super-VL` — flagship, 120B MoE; $0.30/1M input,
  $0.90/1M output tokens.
- `nvidia/nemotron-nano-12b-v2-vl` — smaller, cheaper, faster.

A grounding call is one 480×640 PNG + ~100 tokens of prompt, so per-call
cost is fractions of a cent on either. The interesting experiment is
accuracy-per-dollar across paraphrased instructions ("grab the crimson
cube", "the red one on the left"), not cost.

## Where the VLM could do more

Right now the VLM fires once per episode. Making it load-bearing:

- **Re-grasp decisions.** After a failed cage, show the VLM the current
  frame and ask whether to retry, push clutter first, or pick a different
  approach — language as the recovery policy.
- **Singulation targeting.** "Which boxes are crowding the red box?" turns
  the push-away phase into a VLM decision instead of a distance heuristic.
- **Failure narration.** The rationale field, logged per episode, becomes
  the voiceover for the demo video.

## Exercises

1. Run the smoke test with the heuristic (`make_labeller` with no key) and
   with the live model; compare the rationales.
2. Add the index-range validation noted above, plus a test for it.
3. Paraphrase the instruction five ways and measure grounding accuracy per
   model — that's the VLM swap study in miniature.
