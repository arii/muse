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
    """One episode with one planner. Returns (metrics row, frames).

    Never raises: a planner failure is recorded as planner_error.
    """
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
        return row, frames
    metrics, frames, _ = run_episode(env, plan, instruction, frames,
                                     verbose=False)
    row.update(metrics)
    return row, frames


def outcome_caption(row):
    """One-line outcome for a GIF footer."""
    if row["planner_error"]:
        return "planner FAILED"
    fc = "FORCE CLOSURE" if row.get("force_closure") else "no grasp"
    return (f"{fc} | caged={row.get('caged')} | "
            f"reward={row.get('total_reward', 0):.1f}")


def write_html_report(rows, gif_map, out_path, instruction, planners):
    """One-page comparison report: aggregates + per-seed GIFs + plans.

    rows: per-episode metric rows (as in --out JSON). gif_map:
    seed -> {planner: gif path, "side_by_side": gif path}. Paths are
    made relative to the report's directory so the page works from
    wherever --html points, as long as the GIFs stay beside it.
    """
    import html as _html

    from pathlib import Path
    out_path = Path(out_path)
    report_dir = out_path.parent

    def rel(p):
        try:
            return Path(p).relative_to(report_dir).as_posix()
        except ValueError:
            return Path(p).as_posix()

    def esc(v):
        return _html.escape("" if v is None else str(v))

    def fmt_pushes(push_away):
        # Plan push_away: [{index, effort}] (or legacy bare indices).
        if not push_away:
            return "none"
        parts = []
        for p in push_away:
            if isinstance(p, dict):
                parts.append(f"{p['index']}({p.get('effort', 'clear')})")
            else:
                parts.append(str(p))
        return ", ".join(parts)

    parts = ["<html><head><meta charset='utf-8'>",
             "<style>body{font-family:sans-serif;max-width:1100px;"
             "margin:2em auto;padding:0 1em;line-height:1.45}"
             "table{border-collapse:collapse;width:100%}"
             "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left}"
             "img{max-width:100%}h2{margin-top:2em;border-top:2px solid #222;"
             "padding-top:.5em}.pass{color:green;font-weight:bold}"
             ".fail{color:#888}.error{background:#fee;border:1px solid #c00;"
             "padding:6px}.card{border:1px solid #ccc;padding:12px;"
             "margin:12px 0}.card h3{margin:0 0 6px}"
             "</style></head><body>",
             f"<h1>VLM vs scripted grasping comparison</h1>",
             f"<p>instruction: {esc(instruction)}</p>",
             "<p>Read this first: force_closure is the goal criterion. "
             "Caged is a strict diagnostic (loop must close, see "
             "env.whole_arm_grasp._loop_closed). A GIF that shows the "
             "box sliding away is a failed grasp even if reward > 0 — "
             "reward sums contact over steps and inflates on long "
             "non-passing episodes.</p>",
             "<h2>Aggregate</h2><table><tr><th>planner</th>"
             "<th>episodes</th><th>force_closure</th><th>caged</th>"
             "<th>preconditions</th><th>mean reward</th>"
             "<th>mean isolation Δ</th><th>API calls</th>"
             "<th>tokens</th></tr>"]
    for planner in planners:
        agg = aggregate(rows, planner)
        parts.append(
            f"<tr><td>{esc(planner)}</td><td>{agg['episodes']}</td>"
            f"<td>{agg['force_closure_rate']:.2f}</td>"
            f"<td>{agg['caged_rate']:.2f}</td>"
            f"<td>{agg['preconditions_rate']:.2f}</td>"
            f"<td>{agg['mean_reward']:.1f}</td>"
            f"<td>{agg['mean_isolation_delta']:+.3f} m</td>"
            f"<td>{agg['api_calls']}</td><td>{agg['total_tokens']}</td>"
            "</tr>")
    parts.append("</table>")
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["planner"]] = r
    for seed in sorted(by_seed):
        parts.append(f"<h2>seed {seed}</h2>")
        parts.append("<table><tr><th>planner</th><th>FC</th>"
                     "<th>caged</th><th>preconditions</th>"
                     "<th>reward</th><th>isolation</th>"
                     "<th>plan</th><th>cost</th></tr>")
        for planner in planners:
            r = by_seed[seed].get(planner)
            if r is None:
                continue
            if r["planner_error"]:
                plan_cell = f"ERROR: {esc(r['planner_error'][:200])}"
                cost = "-"
            else:
                plan_cell = (f"target {r.get('target_index')} "
                             f"({esc(r.get('target_color'))}), push "
                             f"{esc(fmt_pushes(r.get('push_away')))}, "
                             f"approach "
                             f"{esc(r.get('approach_side', 'near'))}: "
                             f"{esc(r.get('rationale'))}")
                cost = (f"{r['tokens']} tokens" if planner == "vlm"
                        else "no API")
            fc_cls = "pass" if r.get("force_closure") else "fail"
            parts.append(
                f"<tr><td>{esc(planner)}</td>"
                f"<td class='{fc_cls}'>{r.get('force_closure')}</td>"
                f"<td>{r.get('caged')}</td>"
                f"<td>{r.get('preconditions_met')}</td>"
                f"<td>{r.get('total_reward', 0):.1f}</td>"
                f"<td>{r.get('isolation_before', 0):.3f} → "
                f"{r.get('isolation_after', 0):.3f}</td>"
                f"<td>{plan_cell}</td><td>{cost}</td></tr>")
        parts.append("</table>")
        gifs = gif_map.get(seed, {})
        if gifs.get("side_by_side"):
            parts.append(f"<img src='{esc(rel(gifs['side_by_side']))}' "
                         f"alt='seed {seed} side by side'>")
        for planner in planners:
            if gifs.get(planner):
                parts.append(f"<p><a href='{esc(rel(gifs[planner]))}'>"
                             f"{esc(planner)} GIF</a></p>")
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    return out_path


def write_markdown_summary(rows, out_path, instruction, planners):
    """Plain-text companion to the HTML report, for chat / PDF.

    The HTML report's GIFs do not survive a print-to-PDF (linked
    files, not embedded), so this file carries the whole result in
    text: aggregates + one line per seed per planner, with errors
    shown in full, not hidden behind a GIF link.
    """
    lines = [f"# Compare summary", f"",
             f"instruction: {instruction}", f""]
    for planner in planners:
        agg = aggregate(rows, planner)
        lines.append(
            f"## {planner}: {agg['episodes']} episodes, "
            f"{agg['planner_errors']} planner errors, "
            f"FC {agg['force_closure_rate']:.2f}, "
            f"caged {agg['caged_rate']:.2f}, "
            f"preconditions {agg['preconditions_rate']:.2f}, "
            f"mean reward {agg['mean_reward']:.1f}, "
            f"mean isolation Δ {agg['mean_isolation_delta']:+.3f} m, "
            f"API calls {agg['api_calls']}, tokens {agg['total_tokens']}")
    lines.append("")
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["planner"]] = r
    for seed in sorted(by_seed):
        for planner in planners:
            r = by_seed[seed].get(planner)
            if r is None:
                continue
            if r["planner_error"]:
                lines.append(f"- seed {seed} {planner}: ERROR "
                             f"{r['planner_error'][:300]}")
            else:
                lines.append(
                    f"- seed {seed} {planner}: FC={r.get('force_closure')} "
                    f"caged={r.get('caged')} "
                    f"pre={r.get('preconditions_met')} "
                    f"reward={r.get('total_reward', 0):.1f} "
                    f"iso {r.get('isolation_before', 0):.3f}->"
                    f"{r.get('isolation_after', 0):.3f} "
                    f"plan target={r.get('target_index')} "
                    f"push={r.get('push_away')} "
                    f"approach={r.get('approach_side')} "
                    f"tokens={r.get('tokens', 0)}")
    pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
    return out_path


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
    ap.add_argument("--gifs", action="store_true",
                    help="save per-seed GIFs under episodes/compare-<seed>/: "
                    "one per planner plus a side-by-side composite")
    ap.add_argument("--gif-scale", type=float, default=0.75,
                    help="GIF scale factor (default 0.75)")
    ap.add_argument("--gif-fps", type=float, default=15,
                    help="GIF frame rate (default 15)")
    ap.add_argument("--html", nargs="?", const="episodes/compare-report.html",
                    default=None,
                    help="write a one-page HTML report (implies --gifs; "
                    "default path episodes/compare-report.html)")
    args = ap.parse_args()
    if args.html:
        args.gifs = True

    seeds = parse_seeds(args.seeds)
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    rows = []
    frames_by_seed = {}  # seed -> {planner: (row, frames)}
    for seed in seeds:
        for planner in planners:
            row, frames = run_one(seed, planner, args.instruction)
            rows.append(row)
            frames_by_seed.setdefault(seed, {})[planner] = (row, frames)
            status = (f"FC={row.get('force_closure')} "
                      f"caged={row.get('caged')} "
                      f"rew={row.get('total_reward', 0):.1f}"
                      if not row["planner_error"]
                      else f"ERROR: {row['planner_error'][:80]}")
            print(f"seed {seed:>3} {planner:<9} {status}")

    if args.gifs:
        from scripts.make_video import make_gif, make_side_by_side_gif
        gif_map = {}
        for seed, per_planner in frames_by_seed.items():
            out_dir = pathlib.Path("episodes") / f"compare-{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            gif_map[seed] = {}
            for planner, (row, frames) in per_planner.items():
                meta = {"instruction": args.instruction,
                        "target_color": row.get("target_color", "?"),
                        "force_closure": row.get("force_closure", False)}
                gif_path = out_dir / f"{planner}.gif"
                make_gif(frames, meta, gif_path,
                         fps=args.gif_fps, scale=args.gif_scale,
                         frame_info=row.get("frame_info"))
                gif_map[seed][planner] = str(gif_path)
            if len(per_planner) >= 2:
                # Side-by-side: first two planners, in --planners order.
                (p1, (r1, f1)), (p2, (r2, f2)) = [
                    (p, per_planner[p]) for p in planners
                    if p in per_planner][:2]
                sbs_path = out_dir / "side_by_side.gif"
                make_side_by_side_gif(
                    f1, f2, p1, p2, outcome_caption(r1),
                    outcome_caption(r2), sbs_path,
                    fps=args.gif_fps, scale=args.gif_scale,
                    left_info=r1.get("frame_info"),
                    right_info=r2.get("frame_info"))
                gif_map[seed]["side_by_side"] = str(sbs_path)
                print(f"seed {seed}: side-by-side GIF -> {sbs_path}")
            else:
                print(f"seed {seed}: GIFs -> {out_dir}/ "
                      f"(need 2+ planners for side-by-side)")
        if args.html:
            report = write_html_report(rows, gif_map, args.html,
                                       args.instruction, planners)
            print(f"HTML report -> {report}")
            summary = write_markdown_summary(
                rows, pathlib.Path(args.html).with_suffix(".md")
                if str(args.html).endswith(".html")
                else pathlib.Path("episodes/compare-summary.md"),
                args.instruction, planners)
            print(f"Markdown summary -> {summary}")

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
