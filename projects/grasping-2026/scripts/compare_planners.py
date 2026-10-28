"""Compare planners on identical scenes: VLM, heuristic, PPO, and diffusion.

Runs the same seed across each requested planner/policy, then reports
per-seed results and aggregate performance. The goal criterion is force
closure; caged, reward, target displacement, and isolation are
diagnostics. VLM cost (API calls, tokens, ledger cache hits) is
reported alongside — heuristic, PPO, and diffusion cost nothing.

Usage:
    # via compose (VLM needs NEBIUS_API_KEY from .env):
    docker compose run --build --rm --entrypoint python demo \
        scripts/compare_planners.py --seeds 0-9 --gifs --html
    # include learned policies (ppo, diffusion) alongside vlm & heuristic:
    docker compose run --build --rm --entrypoint python demo \
        scripts/compare_planners.py --seeds 0-9 \
        --planners vlm,heuristic,ppo,diffusion --gifs --html

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

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.whole_arm_grasp import WholeArmGraspEnv  # noqa: E402
from planning.primitives import frame_status  # noqa: E402
from scripts.demo_vlm_grasp import run_episode  # noqa: E402

_PPO_CACHE = {}
_DIFF_CACHE = {}


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


def _resolve_default_model(candidates):
    for c in candidates:
        p = pathlib.Path(c)
        if p.exists():
            return str(p)
    return candidates[0]


def _run_learned(seed, planner, instruction, model_path=None, max_steps=200):
    """Roll out a trained PPO or Diffusion policy on `seed` and return
    (row, frames) matching the compare_planners row schema."""
    from env.gym_wrapper import GraspGymWrapper

    w = GraspGymWrapper(seed=seed, max_steps=max_steps, n_objects=4)
    obs, _ = w.reset(seed=seed)
    env = w.env
    target_idx = w.target_index
    pos0 = env._obs()["objects"][target_idx][:2].copy()
    iso_before = env.target_isolation()
    _, init_info = env._reward()
    frames = [env.render()]
    f_info = [frame_status(init_info)]

    total_rew = 0.0
    steps = 0
    info = init_info

    if planner == "ppo":
        from stable_baselines3 import PPO
        mpath = model_path or _resolve_default_model(
            ["models/grasp_ppo_500k.zip", "models/grasp_ppo.zip"])
        if mpath not in _PPO_CACHE:
            _PPO_CACHE[mpath] = PPO.load(mpath)
        model = _PPO_CACHE[mpath]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = w.step(action)
            total_rew += float(r)
            steps += 1
            if steps % 2 == 0 or terminated or truncated:
                frames.append(env.render())
                f_info.append(frame_status(info))
            if terminated or truncated:
                break
        rationale = f"PPO policy ({pathlib.Path(mpath).stem}, {steps} steps)"
    elif planner == "diffusion":
        import torch
        from diffusers import DDIMScheduler
        from scripts.diffusion_policy import DiffusionMLP, sample_chunk
        mpath = model_path or _resolve_default_model(
            ["models/diffusion_v2.pt", "models/diffusion.pt"])
        if mpath not in _DIFF_CACHE:
            ckpt = torch.load(mpath, map_location="cpu", weights_only=False)
            dm = DiffusionMLP(ckpt["obs_dim"], chunk=ckpt["chunk"],
                              hidden=ckpt["hidden"])
            dm.load_state_dict(ckpt["state"])
            _DIFF_CACHE[mpath] = (
                dm,
                DDIMScheduler(num_train_timesteps=100, clip_sample=True),
                torch.from_numpy(ckpt["obs_mean"]).float(),
                torch.from_numpy(ckpt["obs_std"]).float(),
                ckpt["chunk"],
            )
        dm, sched, obs_mean, obs_std, chunk_len = _DIFF_CACHE[mpath]
        torch.manual_seed(seed)
        while True:
            obs_n = (torch.from_numpy(obs).float() - obs_mean) / obs_std
            chunk = sample_chunk(dm, sched, obs_n, chunk_len, num_steps=20)
            for k in range(min(8, len(chunk))):
                obs, r, terminated, truncated, info = w.step(chunk[k].numpy())
                total_rew += float(r)
                steps += 1
                if steps % 2 == 0 or terminated or truncated:
                    frames.append(env.render())
                    f_info.append(frame_status(info))
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        rationale = f"Diffusion BC ({pathlib.Path(mpath).stem}, {steps} steps)"
    else:
        raise ValueError(f"unknown learned planner {planner!r}")

    pos1 = env._obs()["objects"][target_idx][:2].copy()
    iso_after = env.target_isolation()
    row = {
        "seed": seed,
        "planner": planner,
        "instruction": instruction,
        "planner_error": None,
        "api_call": False,
        "tokens": 0,
        "target_index": target_idx,
        "target_color": env.objects[target_idx]["color"],
        "rationale": rationale,
        "push_away": [],
        "approach_side": "policy",
        "pushes": [],
        "isolation_before": iso_before,
        "isolation_after": iso_after,
        "preconditions_met": iso_after >= 0.15,
        "links_touching": info["links_touching"],
        "enclosed": info["enclosed"],
        "caged": info["caged"],
        "force_closure": info["force_closure"],
        "total_reward": float(total_rew),
        "target_start_r": float(np.linalg.norm(pos0)),
        "target_end_r": float(np.linalg.norm(pos1)),
        "target_disp": float(np.linalg.norm(pos1 - pos0)),
        "steps": steps,
        "frame_info": f_info,
    }
    return row, frames


def run_one(seed, planner, instruction, ppo_model=None, diffusion_model=None):
    """One episode with one planner. Returns (metrics row, frames).

    Never raises: a planner failure is recorded as planner_error.
    """
    if planner in ("ppo", "diffusion"):
        try:
            mpath = ppo_model if planner == "ppo" else diffusion_model
            return _run_learned(seed, planner, instruction, model_path=mpath)
        except Exception as e:
            env = WholeArmGraspEnv(seed=seed)
            env.reset(seed=seed)
            return {
                "seed": seed, "planner": planner, "instruction": instruction,
                "planner_error": str(e)[:300], "api_call": False, "tokens": 0
            }, [env.render()]

    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    pos0 = env._obs()["objects"][0][:2].copy()
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
            c_usage = getattr(labeller, "last_cached_usage", None) or {}
            row["cached_tokens"] = c_usage.get("total_tokens") or 0
    except Exception as e:
        row["planner_error"] = str(e)[:300]
        return row, frames
    t_idx = plan.get("grasp_target", 0)
    pos0 = env._obs()["objects"][t_idx][:2].copy()
    metrics, frames, _ = run_episode(env, plan, instruction, frames,
                                     verbose=False)
    pos1 = env._obs()["objects"][t_idx][:2].copy()
    metrics["target_start_r"] = float(np.linalg.norm(pos0))
    metrics["target_end_r"] = float(np.linalg.norm(pos1))
    metrics["target_disp"] = float(np.linalg.norm(pos1 - pos0))
    metrics["steps"] = len(frames) - 1
    row.update(metrics)
    return row, frames


def outcome_caption(row):
    """One-line outcome for a GIF footer."""
    if row["planner_error"]:
        return "planner FAILED"
    fc = "FORCE CLOSURE" if row.get("force_closure") else "no grasp"
    disp = row.get("target_disp")
    disp_str = f" | disp={disp:.2f}m" if disp is not None else ""
    return (f"{fc} | caged={row.get('caged')} | "
            f"reward={row.get('total_reward', 0):.1f}{disp_str}")


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

    def fmt_primitives(r):
        """Format concrete executed geometric primitives (push / scripted_cage)."""
        if r.get("planner") in ("ppo", "diffusion"):
            return (f"<code>env.step(q_target)</code> × {r.get('steps', 0)} steps "
                    f"(closed-loop joint targets in [-2.6, 2.6] rad)")
        lines = []
        pushes = r.get("pushes") or []
        if pushes:
            for p in pushes:
                if "start_xy" in p:
                    st = p.get("status", "kept")
                    badge = ("<span style='color:#0a7d2c'>[kept]</span>"
                             if st == "kept"
                             else "<span style='color:#b45309'>[reverted]</span>")
                    lines.append(
                        f"1. <code>push(obj{p['index']}:{esc(p.get('color',''))}, "
                        f"start_xy={p['start_xy']}, "
                        f"theta={p['theta_rad']:+.2f}rad/{p['theta_deg']:+.0f}°, "
                        f"dist={p['distance_m']:.2f}m)</code> {badge}"
                    )
                else:
                    lines.append(
                        f"1. <code>push(obj{p.get('index')}, "
                        f"effort={esc(p.get('effort'))}, "
                        f"dist={p.get('distance_m', 0):.2f}m)</code>"
                    )
        else:
            lines.append("1. <code>singulate: none (no push executed)</code>")
        ct = r.get("cage_trace") or {}
        if ct:
            lines.append(
                f"2. <code>scripted_cage(target_xy={ct.get('target_xy')}, "
                f"face={ct.get('face_rad', 0):+.2f}rad, "
                f"lead={ct.get('lead_rad', 0):.2f}rad, "
                f"curl={esc(ct.get('curl_dir'))}, "
                f"open_q={ct.get('open_q_rad', 0):+.2f}rad → "
                f"final_q={ct.get('final_curl_q')}, "
                f"steps={ct.get('steps_executed')})</code>"
            )
        return "<br>".join(lines)

    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["planner"]] = r

    # Build quick seed navigation pills
    nav_pills = []
    for seed in sorted(by_seed):
        any_fc = any(
            by_seed[seed][p].get("force_closure")
            for p in by_seed[seed]
            if not by_seed[seed][p].get("planner_error")
        )
        pill_cls = "pill-pass" if any_fc else "pill-fail"
        nav_pills.append(
            f"<a class='pill {pill_cls}' href='#seed-{seed}'>"
            f"seed {seed} {'✓' if any_fc else ''}</a>"
        )

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Grasping Policy & Planner Comparison</title>",
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:1280px;"
        "margin:1.5em auto;padding:0 1.2em;line-height:1.45;color:#1a1a1a;"
        "background:#fafafa}"
        "h1{margin-bottom:0.2em}"
        "h2{margin-top:1.8em;border-top:2px solid #222;padding-top:.5em}"
        "table{border-collapse:collapse;width:100%;background:#fff;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:1em}"
        "th,td{border:1px solid #d8d8d8;padding:6px 10px;text-align:left;"
        "font-size:0.93em}"
        "th{background:#f0f2f5;font-weight:600}"
        "img{max-width:100%;border:1px solid #ccc;border-radius:4px}"
        ".pass{color:#0a7d2c;font-weight:bold;background:#e8f8ed}"
        ".fail{color:#777}"
        ".warn{color:#b45309;font-weight:600;background:#fef3c7}"
        ".error{background:#fee;border:1px solid #c00;padding:6px}"
        ".banner{background:#eef4ff;border-left:4px solid #2563eb;"
        "padding:10px 14px;margin:12px 0;font-size:0.94em}"
        ".nav-bar{position:sticky;top:0;background:#fafafa;padding:8px 0;"
        "border-bottom:1px solid #ddd;z-index:10;display:flex;gap:6px;"
        "flex-wrap:wrap;align-items:center}"
        ".pill{text-decoration:none;padding:3px 9px;border-radius:12px;"
        "font-size:0.85em;border:1px solid #ccc}"
        ".pill-pass{background:#d1fae5;color:#065f46;border-color:#34d399;"
        "font-weight:bold}"
        ".pill-fail{background:#f3f4f6;color:#4b5563}"
        ".gif-grid{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(240px,1fr));gap:10px;margin-top:8px}"
        ".gif-card{background:#fff;border:1px solid #ddd;padding:6px;"
        "border-radius:4px;text-align:center;font-size:0.85em}"
        "details{margin:6px 0 16px}"
        "summary{cursor:pointer;color:#2563eb;font-size:0.9em;font-weight:500}"
        "</style></head><body>",
        "<h1>Grasping Comparison: VLM vs Heuristic vs Learned Policies</h1>",
        f"<p><strong>Instruction:</strong> <code>{esc(instruction)}</code></p>",
        "<div class='banner'><strong>Read this first:</strong> "
        "<code>force_closure</code> is the goal criterion (contact wrenches "
        "positively span R³). <code>caged</code> is a strict diagnostic "
        "(target inside arm polygon AND tip-to-link1 loop closed &lt; 0.15 m). "
        "<code>target Δxy</code> shows how far the target box was displaced: "
        "values &gt; 0.30 m indicate the arm knocked the target box away. "
        "Reward sums contact over steps and inflates on long non-passing "
        "episodes.</div>",
        "<div class='nav-bar'><strong>Jump to:</strong> "
        + "".join(nav_pills)
        + "</div>",
        "<h2>Aggregate Summary</h2><table><tr><th>planner</th>"
        "<th>episodes</th><th>errors</th><th>force_closure</th><th>caged</th>"
        "<th>preconditions</th><th>mean reward</th>"
        "<th>mean isolation Δ</th><th>mean target Δxy</th>"
        "<th>API calls</th><th>tokens</th></tr>",
    ]
    for planner in planners:
        agg = aggregate(rows, planner)
        fc_cls = "pass" if agg["force_closure_rate"] > 0 else ""
        tok_cell = (f"{agg['total_tokens']} ({agg['cached_tokens']} cached)"
                    if agg.get("cached_tokens", 0) > 0
                    else str(agg["total_tokens"]))
        parts.append(
            f"<tr><td><strong>{esc(planner)}</strong></td>"
            f"<td>{agg['episodes']}</td>"
            f"<td>{agg['planner_errors']}</td>"
            f"<td class='{fc_cls}'>{agg['force_closure_rate']:.2f}</td>"
            f"<td>{agg['caged_rate']:.2f}</td>"
            f"<td>{agg['preconditions_rate']:.2f}</td>"
            f"<td>{agg['mean_reward']:.1f}</td>"
            f"<td>{agg['mean_isolation_delta']:+.3f} m</td>"
            f"<td>{agg['mean_target_disp']:.3f} m</td>"
            f"<td>{agg['api_calls']}</td><td>{tok_cell}</td>"
            "</tr>")
    parts.append("</table>")

    for seed in sorted(by_seed):
        parts.append(f"<h2 id='seed-{seed}'>seed {seed}</h2>")
        parts.append("<table><tr><th>planner</th><th>FC</th>"
                     "<th>caged</th><th>preconditions</th>"
                     "<th>reward</th><th>isolation</th>"
                     "<th>target Δxy (r0→r1)</th>"
                     "<th>semantic plan &amp; executed primitives</th>"
                     "<th>cost</th></tr>")
        for planner in planners:
            r = by_seed[seed].get(planner)
            if r is None:
                continue
            if r["planner_error"]:
                plan_cell = f"ERROR: {esc(r['planner_error'][:200])}"
                cost = "-"
                disp_cell = "-"
                disp_cls = ""
            else:
                prim_html = fmt_primitives(r)
                plan_cell = (
                    f"<div><strong>Plan:</strong> target {r.get('target_index')} "
                    f"({esc(r.get('target_color'))}), push "
                    f"{esc(fmt_pushes(r.get('push_away')))}, "
                    f"approach <code>{esc(r.get('approach_side', 'near'))}</code>"
                    f"</div>"
                    f"<div style='margin-top:4px;font-size:0.9em;color:#1e293b;"
                    f"background:#f8fafc;padding:4px 6px;border-left:3px solid #64748b'>"
                    f"<strong>Executed Primitives:</strong><br>{prim_html}</div>"
                    f"<div style='margin-top:3px;font-size:0.85em;color:#555'>"
                    f"<em>Rationale:</em> {esc(r.get('rationale'))}</div>"
                )
                if planner == "vlm":
                    if r.get("api_call"):
                        cost = f"{r['tokens']} tokens (live API)"
                    elif r.get("cached_tokens"):
                        cost = (f"0 tokens (cached; orig "
                                f"{r['cached_tokens']} tok)")
                    else:
                        cost = f"{r['tokens']} tokens (cached)"
                else:
                    cost = "no API"
                disp = r.get("target_disp")
                r0 = r.get("target_start_r")
                r1 = r.get("target_end_r")
                if disp is not None and r0 is not None and r1 is not None:
                    disp_cell = f"{disp:.3f} m ({r0:.2f}→{r1:.2f} m)"
                    disp_cls = "warn" if disp > 0.35 and not r.get("force_closure") else ""
                else:
                    disp_cell = "-"
                    disp_cls = ""
            fc_cls = "pass" if r.get("force_closure") else "fail"
            cage_cls = "pass" if r.get("caged") else ""
            pre_cls = "" if r.get("preconditions_met") else "warn"
            parts.append(
                f"<tr><td><strong>{esc(planner)}</strong></td>"
                f"<td class='{fc_cls}'>{r.get('force_closure')}</td>"
                f"<td class='{cage_cls}'>{r.get('caged')}</td>"
                f"<td class='{pre_cls}'>{r.get('preconditions_met')}</td>"
                f"<td>{r.get('total_reward', 0):.1f}</td>"
                f"<td>{r.get('isolation_before', 0):.3f} → "
                f"{r.get('isolation_after', 0):.3f}</td>"
                f"<td class='{disp_cls}'>{disp_cell}</td>"
                f"<td>{plan_cell}</td><td>{cost}</td></tr>")
        parts.append("</table>")
        gifs = gif_map.get(seed, {})
        if gifs.get("side_by_side"):
            parts.append(f"<img src='{esc(rel(gifs['side_by_side']))}' "
                         f"alt='seed {seed} side by side'>")
        indiv = [
            f"<div class='gif-card'><a href='{esc(rel(gifs[p]))}'>"
            f"<strong>{esc(p)} GIF</strong></a><br>"
            f"<img src='{esc(rel(gifs[p]))}' alt='{esc(p)} seed {seed}'></div>"
            for p in planners if gifs.get(p)
        ]
        if indiv:
            parts.append(
                "<details><summary>Individual planner GIFs ("
                + ", ".join(esc(p) for p in planners if gifs.get(p))
                + ")</summary><div class='gif-grid'>"
                + "".join(indiv)
                + "</div></details>"
            )
            for planner in planners:
                if gifs.get(planner):
                    parts.append(
                        f"<p style='display:none'>"
                        f"<a href='{esc(rel(gifs[planner]))}'>"
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
    lines = ["# Compare summary", "",
             f"instruction: {instruction}", ""]
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
            f"mean target disp {agg['mean_target_disp']:.3f} m, "
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
                disp = r.get("target_disp", 0.0)
                lines.append(
                    f"- seed {seed} {planner}: FC={r.get('force_closure')} "
                    f"caged={r.get('caged')} "
                    f"pre={r.get('preconditions_met')} "
                    f"reward={r.get('total_reward', 0):.1f} "
                    f"iso {r.get('isolation_before', 0):.3f}->"
                    f"{r.get('isolation_after', 0):.3f} "
                    f"disp={disp:.3f}m "
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
        vals = [r[key] for r in ok if key in r and r[key] is not None]
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
        "mean_target_disp": mean("target_disp"),
        "api_calls": sum(1 for r in rs if r["api_call"]),
        "total_tokens": sum(r["tokens"] for r in rs),
        "cached_tokens": sum(r.get("cached_tokens", 0) for r in rs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction",
                    default="grasp the red box in isolation")
    ap.add_argument("--seeds", default="2,6",
                    help="seed specification, e.g. '2,6' (default canary seeds) or '0-9'")
    ap.add_argument("--planners", default="vlm,heuristic,ppo,diffusion",
                    help="comma-separated subset of vlm,heuristic,ppo,diffusion")
    ap.add_argument("--ppo-model", default=None,
                    help="path to PPO .zip checkpoint (default auto-detect)")
    ap.add_argument("--diffusion-model", default=None,
                    help="path to Diffusion .pt checkpoint (default auto-detect)")
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
    req_planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    # Filter out ppo/diffusion only if default was used and no model exists on disk
    planners = []
    for p in req_planners:
        if p == "ppo" and args.ppo_model is None:
            if not (pathlib.Path("models/grasp_ppo_500k.zip").exists()
                    or pathlib.Path("models/grasp_ppo.zip").exists()):
                continue
        if p == "diffusion" and args.diffusion_model is None:
            if not (pathlib.Path("models/diffusion_v2.pt").exists()
                    or pathlib.Path("models/diffusion.pt").exists()):
                continue
        planners.append(p)

    rows = []
    frames_by_seed = {}  # seed -> {planner: (row, frames)}
    for seed in seeds:
        for planner in planners:
            row, frames = run_one(seed, planner, args.instruction,
                                  ppo_model=args.ppo_model,
                                  diffusion_model=args.diffusion_model)
            rows.append(row)
            frames_by_seed.setdefault(seed, {})[planner] = (row, frames)
            status = (f"FC={row.get('force_closure')} "
                      f"caged={row.get('caged')} "
                      f"rew={row.get('total_reward', 0):.1f} "
                      f"disp={row.get('target_disp', 0):.2f}m"
                      if not row["planner_error"]
                      else f"ERROR: {row['planner_error'][:80]}")
            print(f"seed {seed:>3} {planner:<9} {status}")

    if args.gifs:
        from scripts.make_video import (make_gif, make_multi_panel_gif,
                                        make_side_by_side_gif)
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
            ordered = [(p, per_planner[p]) for p in planners
                       if p in per_planner]
            if len(ordered) == 2:
                (p1, (r1, f1)), (p2, (r2, f2)) = ordered
                sbs_path = out_dir / "side_by_side.gif"
                make_side_by_side_gif(
                    f1, f2, p1, p2, outcome_caption(r1),
                    outcome_caption(r2), sbs_path,
                    fps=args.gif_fps, scale=args.gif_scale,
                    left_info=r1.get("frame_info"),
                    right_info=r2.get("frame_info"))
                gif_map[seed]["side_by_side"] = str(sbs_path)
                print(f"seed {seed}: side-by-side GIF -> {sbs_path}")
            elif len(ordered) >= 3:
                sbs_path = out_dir / "side_by_side.gif"
                panel_scale = min(args.gif_scale, 0.50 if len(ordered) >= 4 else 0.60)
                panels = [
                    {
                        "frames": f,
                        "label": p,
                        "caption": outcome_caption(r),
                        "frame_info": r.get("frame_info"),
                    }
                    for p, (r, f) in ordered
                ]
                make_multi_panel_gif(
                    panels, sbs_path, fps=args.gif_fps, scale=panel_scale)
                gif_map[seed]["side_by_side"] = str(sbs_path)
                print(f"seed {seed}: {len(ordered)}-panel GIF -> {sbs_path}")
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
        print(f"  mean target disp:   {agg['mean_target_disp']:.3f} m")
        if planner == "vlm":
            print(f"  API calls:          {agg['api_calls']} "
                  f"(rest ledger cache hits)")
            print(f"  total tokens:       {agg['total_tokens']}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
