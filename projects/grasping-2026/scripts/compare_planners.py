"""Compare planners on identical scenes: VLM vs scripted (heuristic).

Runs the same episode pipeline (plan -> singulate -> gate -> cage)
with each planner on each seed, then reports per-seed results and
aggregate performance. The goal criterion is force closure; caged,
reward, and isolation are diagnostics. VLM cost (API calls, tokens,
ledger cache hits) is reported alongside — the heuristic costs nothing.

Usage:
    # via compose (VLM needs NEBIUS_API_KEY from .env):
    docker compose run --build --rm --entrypoint python demo \
        scripts/compare_planners.py --seeds 0-9
    # single planner, explicit seeds:
    python scripts/compare_planners.py --seeds 0,5,7 --planners heuristic

VLM mode here is PURE: no fallback to heuristic on failure. A VLM
failure on a seed counts as a VLM failure (planner_error set), so the
comparison is VLM vs heuristic, not VLM-or-heuristic vs heuristic.
The demo's own --planner vlm mode still falls back for robustness;
this script does not, deliberately.

Output: per-seed table, then aggregates per planner. --out writes the
full per-episode rows as JSON for later analysis.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.whole_arm_grasp import WholeArmGraspEnv  # noqa: E402
from scripts.demo_vlm_grasp import run_episode  # noqa: E402


def parse_seeds(spec):
    """'0-9' -> range(0, 10); '0,5,7' -> [0, 5, 7]."""
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def run_one(seed, planner, instruction):
    """One episode with one planner. Returns a metrics row (never raises)."""
    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    frames = [env.render()]
    row = {"seed": seed, "planner": planner, "instruction": instruction,
           "planner_error": None, "api_call": False, "tokens": 0}
    try:
        if planner == "heuristic":
            from perception.grounding import HeuristicLabeller
            labeller = HeuristicLabeller()
        else:
            from perception.grounding import make_labeller
            labeller = make_labeller()
        plan = labeller.plan(frames[0], env.objects, instruction)
        if planner == "vlm":
            # last_cache_hit False => API call (tokens from last_response)
            row["api_call"] = labeller.last_cache_hit is False
            usage = (labeller.last_response or {}).get("usage") or {}
            row["tokens"] = usage.get("total_tokens") or 0
    except Exception as e:
        row["planner_error"] = str(e)[:300]
        return row
    metrics, _, _ = run_episode(env, plan, instruction, frames,
                                verbose=False)
    row.update(metrics)
    return row


def aggregate(rows, planner):
    """Aggregate metrics for one planner's rows."""
    rs = [r for r in rows if r["planner"] == planner]
    n = len(rs)
    ok = [r for r in rs if not r["planner_error"]]
    def rate(key):
        return (sum(1 for r in ok if r.get(key)) / len(ok) if ok else 0.0)
    def mean(key):
        vals = [r[key] for r in ok if key in r]
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "planner": planner,
        "episodes": n,
        "planner_errors": n - len(ok),
        "force_closure_rate": rate("force_closure"),
        "caged_rate": rate("caged"),
        "preconditions_rate": rate("preconditions_met"),
        "mean_reward": mean("total_reward"),
        "mean_isolation_delta": mean("isolation_after") - mean(
            "isolation_before") if ok else 0.0,
        "api_calls": sum(1 for r in rs if r["api_call"]),
        "total_tokens": sum(r["tokens"] for r in rs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction",
                    default="grasp the red box in isolation")
    ap.add_argument("--seeds", default="0-9",
                    help="'0-9' or '0,5,7'")
    ap.add_argument("--planners", default="vlm,heuristic",
                    help="comma-separated, subset of vlm,heuristic")
    ap.add_argument("--out", default=None,
                    help="write per-episode rows as JSON")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    rows = []
    for seed in seeds:
        for planner in planners:
            row = run_one(seed, planner, args.instruction)
            rows.append(row)
            status = (f"FC={row.get('force_closure')} "
                      f"caged={row.get('caged')} "
                      f"rew={row.get('total_reward', 0):.1f}"
                      if not row["planner_error"]
                      else f"ERROR: {row['planner_error'][:80]}")
            print(f"seed {seed:>3} {planner:<9} {status}")

    print()
    for planner in planners:
        agg = aggregate(rows, planner)
        print(f"== {planner} ({agg['episodes']} episodes, "
              f"{agg['planner_errors']} planner errors) ==")
        print(f"  force_closure rate: {agg['force_closure_rate']:.2f}")
        print(f"  caged rate:         {agg['caged_rate']:.2f}")
        print(f"  preconditions rate: {agg['preconditions_rate']:.2f}")
        print(f"  mean reward:        {agg['mean_reward']:.1f}")
        print(f"  mean isolation Δ:   {agg['mean_isolation_delta']:+.3f} m")
        if planner == "vlm":
            print(f"  API calls:          {agg['api_calls']} "
                  f"(rest ledger cache hits)")
            print(f"  total tokens:       {agg['total_tokens']}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
