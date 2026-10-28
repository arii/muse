"""Live smoke test for the Nebius VLM labeller.

Requires:
    NEBIUS_API_KEY   - Token Factory API key
    NEBIUS_VLM_MODEL - vision model id, e.g. nvidia/Nemotron-3.5-Super-VL

Renders scenes with known box colors, asks the VLM which box to grasp,
and checks it picks the right one. Prints the exact prompt sent and the
raw model reply, so you can see what the model saw and said.

Usage:
    NEBIUS_API_KEY=... NEBIUS_VLM_MODEL=... python scripts/smoke_vlm.py
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.whole_arm_grasp import WholeArmGraspEnv
from perception.grounding import NebiusVLMLabeller, PROMPT

CASES = [
    # (seed, instruction, expected color)
    (0, "grasp the red box", "red"),
    (0, "grasp the blue box", "blue"),
    (1, "grasp the green box", "green"),
    (1, "pick up the yellow block", "yellow"),
    (2, "grasp the red box", "red"),
]


def main():
    missing = [v for v in ("NEBIUS_API_KEY", "NEBIUS_VLM_MODEL")
               if not os.environ.get(v)]
    if missing:
        sys.exit(f"set {', '.join(missing)} first — see docs/vlm-grounding-tutorial.md")

    labeller = NebiusVLMLabeller()
    print(f"[smoke] model: {labeller.model}")
    print(f"[smoke] endpoint: {labeller.base_url}/chat/completions\n")

    passed, failed = 0, 0
    for seed, instruction, want in CASES:
        env = WholeArmGraspEnv(seed=seed)
        env.reset(seed=seed)
        rgb = env.render()
        try:
            idx, rationale = labeller.label(rgb, env.objects, instruction)
        except Exception as e:
            failed += 1
            print(f"[FAIL] seed={seed} instruction={instruction!r} — "
                  f"VLM error:\n{e}")
            print(f"  last_response: "
                  f"{__import__('json').dumps(labeller.last_response)[:1500]}")
            continue
        got = env.objects[idx]["color"]
        ok = got == want
        passed, failed = passed + ok, failed + (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] seed={seed} "
              f"instruction={instruction!r} -> chose {got} "
              f"(wanted {want}); rationale: {rationale}")
        if seed == CASES[0][0] and instruction == CASES[0][1]:
            desc = ", ".join(f"index {o['index']}: {o['color']} box"
                             for o in env.objects)
            print("\n--- exact prompt sent (first case) ---")
            print(PROMPT.format(objects=desc, instruction=instruction))
            print("--- raw model reply ---")
            print(labeller.last_raw)
            print("---\n")

    print(f"\n[smoke] {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
