# Approaches, Action Space, and How the Pieces Fit

A learning-oriented guide to the grasping-2026 experiment: what the
robot controls, what it sees, how each of the six approaches turns
that into motion, and how to read the comparison report. Written for
someone coming from the thesis setup (2D top-down sim, 3-DOF arm,
SARSA on continuous states).

## 1. The task

A 3-link planar arm in MuJoCo, viewed top-down, must grasp a target
box (red, index 0) among distractor boxes on a table. All boxes are
planar joints: they slide and yaw, never tip or lift. The arm has
three revolute joints and no gripper — it grasps *whole-arm*, by
wrapping its links around the box (caging / force closure), exactly
like the thesis cross-section arm.

Differences from the thesis setup:

- 3 joints on a **single** arm (no mirrored dual-arm mode yet).
- **Clutter**: 4 boxes by default. Some scenes need a distractor
  pushed aside (singulation) before the grasp.
- The policy zoo: scripted, VLM, RL, and imitation approaches on the
  same scenes, compared head to head.

## 2. Action space

Every approach ultimately emits the same thing: **target joint
angles** for the three arm joints, one command per environment step.

- Raw simulator action: `[q1, q2, q3]` in radians, clipped to
  ±2.6 rad (`JOINT_LIMIT`). The joints are soft position
  actuators: each `env.step(action)` sets the position targets and
  runs 20 MuJoCo substeps, so the arm chases the target smoothly
  rather than teleporting.
- Learned policies (PPO/BC/diffusion) output **normalized** actions
  in `[-1, 1]^3`, scaled by ×2.6 in the wrapper. This is the SB3
  convention: symmetric network outputs, one scale factor.
- `max_delta_q` (currently 1.0 for PPO): caps how far the commanded
  target may move from the current joints in one step. Without it,
  random exploration commands full-range 5.2 rad swings that spike
  the physics solver. Think of it as a slew-rate limit; demo-like
  motions (< 0.1 rad/step) never touch it.

Link lengths are 0.30 / 0.25 / 0.20 m. Workspace radius ≈ 0.75 m;
boxes spawn at polar r ∈ [0.28, 0.40] m, θ ∈ ±2.4 rad around
the base (0.2 rad of margin under the j1 limit); box yaw is
not randomized, it only changes through contact.

Thesis parallel: your 3-DOF continuous action vector is the same
idea; the only additions are the [-1, 1] normalization and the
slew-rate cap.

## 3. Observation space

Learned policies see a flat float32 vector (no images anywhere in
the loop — the VLM is the only component that looks at pixels, and
only once per episode, to *plan*).

For the default 4 objects **with** contact features (18 dims):

| Slice | Dims | Contents |
|---|---|---|
| `qpos` | 3 | joint angles (rad) |
| `qvel` | 3 | joint velocities |
| `target_xy` | 2 | target box center |
| `target_half` | 2 | target box half-extents |
| distractors | 3 × 2 | each distractor's (dx, dy) **relative to the target**, sorted by angle around it |
| contact | 2 | links_touching / 3, enclosed flag |

Without `--contact-obs` the last two dims are dropped (16 dims).
Two design points that matter:

- **Relative + angle-sorted distractors.** Index order made identical
  scenes look different under permutation, and diffusion hedged to
  the mean because of it. Sorting by angle around the target gives
  permutation invariance. All checkpoints predating this change
  are stale.
- **Contact features are privileged.** They expose the reward signal
  (touching links, enclosure) directly instead of forcing the net
  to infer contact from geometry. PPO needs them; BC/diffusion
  train on demos that include them.

## 4. Reward and success criteria

Base reward per step: `0.5 × links_touching + 2.0 × force_closure`.
Episode ends (terminated) on force closure, or truncates at 200
steps. Three nested notions of "grasped", weakest first:

1. **links_touching** — how many links contact the target (0–3).
   The shaping signal; also the camping exploit (lean on the box
   for 200 steps ≈ +200 reward without grasping).
2. **force_closure (FC)** — the goal criterion. Contact wrenches
   on the target span opposing directions (friction-cone test), so
   the grasp resists disturbance. A pinch can pass FC without
   surrounding the box.
3. **caged** — strictest diagnostic: ≥2 links touching, the
   target center inside the arm's joint polygon, *and* the arm
   loop closed around it. FC without caged ≈ a pinch; caged
   without FC ≈ surrounding but not squeezing.

Scene metrics: **isolation** (min center-distance from target to
any distractor; ≥ 0.15 m counts as "preconditions met") and
**target displacement** (how far the target moved — high values
mean shoving, not grasping).

PPO-only shaping (train time; eval uses the base reward so reports
stay comparable): `-2.0` per step (camping then nets −200, so any
grasp wins) and `+20` on the force-closure step (without it PPO
discovered grasps and then *unlearned* them, because ending the
episode forfeited the camp reward).

## 5. The seven approaches

They fall into two families. **Plan-based** approaches
(heuristic/VLM) run open-loop scripts: plan once, push, then execute
a fixed cage trajectory. **Closed-loop** approaches
(reactive/BC/diffusion/PPO) observe every step and react.

### Heuristic — the scripted reference

`HeuristicLabeller` parses the instruction with regexes ("grasp the
*red* box" → target 0; "push the blue box away" → push list).
For the standard instruction it emits: target red, no pushes,
approach "near". The executor then runs `scripted_cage`: park link
1 beside the target, curl links 2–3 around it over ~120 interpolated
steps, stopping early on FC + caged. Deterministic — the same seed
always gives the same trajectory. Everything learned is measured
against this bar (currently 7/10).

### VLM — same executor, smarter plan

`NebiusVLMLabeller` sends the top-down render + object list +
instruction to a vision-language model (Nemotron-3.5-Super-VL via
Nebius) with a strict JSON schema. The model returns a *semantic*
plan — no coordinates, just decisions:

```json
{"grasp_target": 0,
 "push_away": [{"index": 1, "effort": "shove"}],
 "approach_side": "near",
 "rationale": "..."}
```

The executor resolves each choice to geometry: effort → push
distance (nudge/clear/shove), approach_side → cage entry yaw.
Every call is logged to `episodes/vlm_ledger.jsonl`, which doubles
as a cache: identical scene + instruction + prompt replays cost $0
(the report shows "0 tokens (cached)"). Key lesson from this
project: the VLM only earns its keep through **singulation** —
where no push is needed it reduces to the heuristic.

Selection-gap study (sandwich/adjacent × seeds 0–9, 2 fresh VLM
calls): blind `autopush` (clear every crowder within 0.35 m)
ties heuristic 6/10 on sandwich (fixes 3, 9; breaks 5, 8) and
beats it 8/10 on adjacent (fixes 8, 9; breaks nothing) — so on
adjacent, push *action* alone is the whole game. On the two
decisive sandwich seeds VLM matches neither: s3 autopush-✓
VLM-✗, s8 heuristic-✓ autopush-✗ VLM-✓. And VLM planned
identical targets on all three probes (shove 3, shove 1) — the
same every-crowder set autopush clears. Follow-up
(`--autopush-effort shove`): autopush-shove reproduces the VLM
pattern exactly on all three observed sandwich seeds (s3✗ s8✓
s9✓) and scores 7/10 overall (fixes 5, 8; breaks 3) — so on
sandwich, VLM ≡ blind-push-every-crowder-with-shove, and
*effort* fully explains the VLM-vs-autopush gap. But effort is
scene-dependent, not universally-better: on adjacent, shove
*loses* seed 3 (7/10) while clear keeps 8/10 — and the VLM
shoved there too (cached s7), so it does not adapt effort to
the scene either — and it never withholds pushes where blind
pushing harms (s8: VLM pushed and won, autopush-clear pushed
and lost). Conclusion: on these clutter presets no VLM
selection edge is observed at all; VLM ≡ autopush-shove on
every seed both ran. The `decoy` preset (sandwich blockers +
loose distractor toward the base) was built to provide the
remaining case — asymmetric clutter where target choice
matters — and the 10-seed scan still finds no selective-push
regime: seeds 0–5 fail under every arm (decoy blocks the
corridor unpushed), 6 solves under every arm (degraded, no
decoy), 7–8 are effort-gated (shove wins, clear loses), and 9
is a *withholding* regime (ladder wins push-free, any shove
breaks it). The live VLM on decoy 9 shows no restraint either
(pushed blue-shove + green-clear, skipped yellow, lost —
1 call, 6326 tokens). Net: across sandwich, adjacent, and
decoy, VLM target/effort/restraint selection never beats the
matching blind rule.

### Reactive — the deliberate floor

Proportional tip servo toward the target (capped 5 cm/step, DLS IK),
then open-loop curl of joints 2–3 on contact. No plan, no memory
beyond the current error. It scores 0/10 by design: any approach
that can't beat "walk at it and curl" is broken. Useful when tuning
— if a learned policy matches reactive, its perception is noise.

### BC — single-step cloning

A 3-layer MLP, obs → action, trained with plain MSE on the demo
pairs (`train_bc.py`, 150 epochs). One forward pass per env step,
no memory, no chunking. This is the "simplest thing that could
work" for imitation, and the baseline diffusion must beat. It
currently matches the heuristic at 7/10.

### Diffusion — chunked cloning with a denoiser

State-based Diffusion Policy (Chi et al.) without images: instead
of one action, the net predicts a **chunk of 16 future actions**,
conditioned on the current obs. Inference runs DDIM sampling
(20 denoising steps) and executes the first 8 actions before
replanning (**receding horizon**: plan 16, do 8, re-observe).
Why chunks? Temporal consistency — consecutive actions come from
one coherent sample instead of 16 independent guesses.

Two denoiser nets live in `diffusion_policy.py`:

- **MLP (legacy)**: concat [noisy chunk, obs, timestep] → MLP →
  predicted noise. Simple, but it underfit chunks (49% relative
  error, hedged to the mean) and scored 0/10.
- **FiLM UNet-1D**: convolutions along the time axis with
  down/upsampling, conditioned by FiLM scale/shift from obs +
  timestep embeddings. The temporal + conditioning bias fixes the
  hedging (5/10, gentlest of the grasping policies at 0.16 m
  mean displacement — reactive is gentler but never grasps).

Training details that matter: demo obs are normalized (mean/std
with floors so dead dims don't explode); FC-anchored windows
oversample the pinch transition ×20; demos hold the final pose
for +16 steps so the policy learns to *stay* grasped.

### PPO — reinforcement learning from scratch

Stable-Baselines3 PPO on the gym wrapper, 500k steps, zero demos.
It learns purely from the shaped reward (contact payments − time
penalty + FC bonus). v3 reaches 7/10 with the contact obs +
anti-camp recipe; without the +20 bonus it discovered grasps and
unlearned them. Price of RL here: high displacement (0.65 m) —
it shoves boxes around and still gets paid, since the reward
doesn't punish displacement.

Follow-up: the same recipe + `--fail-penalty 100` (thesis −100
on timeout/out-of-reach, `models/grasp_ppo_fail100.zip`, local
only) also scores 7/10 — the cliff that killed linear SARSA
does not hurt deep PPO (dense contact payments + GAE bridge
it). Different misses though: {0, 4, 8} vs v3's {0, 1, 4}.
Wins are near-instant (2–7 steps); seed 8 is a 200-step camp
(contact payments, no FC), so the −100 did not cure camping
either — it just moved the failure.

### SARSA(λ) — the thesis learner, ported

`scripts/train_sarsa.py` + `scripts/tile_coding.py`: linear
SARSA(λ) with hashed tile coding, the closest port of the Anders
(2014) learner. 27 discrete joint-increment actions ({−0.1, 0,
+0.1}³ per step), tile-coded linear Q over 6 continuous features
(q₀, q₁, q₂, red-x, red-y, red-yaw), the exact thesis reward
schedule (goal 0, failures −100, contacts −0.25/−0.5, else −1
per step via `thesis_reward=True`), ε-greedy, accumulating
traces (λ = 0.9). (Tiling, coords) pairs hash into a fixed
memory à la Sutton & Barto §12.7, so resolution (7³×9²×5) stays
affordable at 4096 buckets. Differs from the thesis only in the
plant (MuJoCo top-down scene instead of Box2D) and episode
length (60 steps).

What porting it taught us (all measured, seeds 0–29 probes):

- **The contact desert.** From home starts, a random joint walk
  touches the red block in ~1/10 episodes — the −0.25/−0.5
  shaping is nearly absent, so there is almost no gradient to
  climb. The thesis never faced this (single object, dense
  contact). Fix, straight from the thesis: noise on the initial
  position (`--random-init 2.0` samples starting joints), which
  raises target-touch episodes to ~1/3.
- **Optimism learns to hide.** With γ = 1 and −100 timeouts,
  true Q ≈ −160 while weights start at 0; timeout (the 97%
  outcome) smears −100 over every trail via traces, so explored
  regions go deeply negative and the greedy policy surfs the
  optimism frontier into unvisited corners (3000 eps, α = 1.0:
  weights ∈ [−883, +70], greedy touches 2/10 vs random 4/10).
  Near-starts (`--near-start 0.25`: tip within 25 cm of red)
  restore dense contact and 4× the training FC rate (12–14%).
- **FC is reachable by chance.** ~3% of random-init training
  episodes stumble into force closure, so the 0-terminal signal
  exists in the training distribution — the agent's job is to
  carve those accidents into a greedy corridor.
- **Raw pose features are sub-tile.** XY tiles span ~17 cm but
  contact happens within ~2 cm: touching vs 10 cm away often
  share all tiles, so the shaping signal is unlearnable noise.
  `--contact-state` appends exact touch-count + enclosed dims
  (same fix PPO needed as `contact_obs`), letting Q represent
  "on contact, tighten".

Result: **negative — 0/10 eval FC in all six configurations**
(3000 episodes × 60 steps each: home/random-init/near starts ×
6/8 features × −100/0 timeout × absolute/relative xy). Greedy never
beats chance at home-start eval; the sixth config (tip-relative
xy + contact, `--relative`) shows the first sign of life
*within* the near-start training distribution: 13/60 FC vs
random 7/60 (one-sided p ≈ 0.07, suggestive not significant),
driven by better contact→FC conversion (fewer touch steps,
identical time-to-first-touch) — i.e. the exact contact dims
support a weak reactive close, while coarse tiles still cannot
servo the approach. The 2014 result depended on dense contact
in a small workspace; on the harder cluttered problem, linear
tiles don't scale — the MLP in PPO (7/10) earns its keep by
learning adaptive nonlinear features. (Absolute-xy configs:
0–2/20 FC vs random 1–2/20 in-distribution, and two learn to
*avoid* contact via optimism surfing.) The port is faithful
and the machinery verified (tile/traces/unit tests all pass) —
the failure is representational: FC value
depends on sub-tile geometry (2 cm alignments under ~17 cm
tiles) that uniform hashed tiles alias away, and 85–97%
timeouts let the −100 cliff (or converged pessimism) drown the
shaping gradient. See §8.

## 6. How they relate (data flow)

```text
scripted_cage ──demonstrates──▶ demos_cage_fc (130 FC episodes)
        │                                │
        │ mined by                       ├──▶ BC (MSE, 1 step)
        ▼                                └──▶ Diffusion (chunks)
 heuristic/VLM                         PPO ◀── reward only, no demos
 executor                              reactive ◀── no learning at all
```

The honesty constraint on cloning: **BC/diffusion cannot exceed
their demonstrator.** They train on scripted-cage trajectories, so
their ceiling is "the cage, generalized to new scenes" — worth
remembering when BC ties heuristic at 7/10: it learned the script,
it didn't invent grasping. PPO is the only approach that
discovered anything the scripts can't do; SARSA(λ) tried from
reward alone too and learned nothing at eval (0/10 × 6 configs) — the
cluttered problem defeats linear tiles, which is exactly why
the learned-features half of this project exists.

## 7. Reading the comparison report

`episodes/compare-report.html` (companion `.md` for print/PDF) is
generated by `compare_planners.py --gifs --html`: identical seeds
across planners, one row per seed per planner, aggregate table up
top, per-seed side-by-side GIFs below. Metrics glossary:

- **force_closure rate** — the headline. Fraction of seeds ending
  in force closure.
- **caged rate** — strict enclosure fraction. Below FC rate means
  pinches, not wraps.
- **preconditions rate** — fraction of scenes starting isolated
  (≥ 0.15 m). A property of the *seeds*, not the planner.
- **singulation rate** — fraction of crowded starts
  (isolation < 0.15 m) ending isolated. 0.00 across the board
  right now (see §8).
- **mean steps / reward** — efficiency and raw return. Reward is
  *not* comparable across planners (learned ones accumulate contact
  payments differently); steps roughly is.
- **isolation Δ / target disp** — did the scene get more isolated,
  and did the target get shoved? Positive Δ + low disp = clean.
- **API calls / tokens** — VLM cost; cached reruns show 0.

## 8. Current results and known failure modes (seeds 0–9)

| Planner | FC | Notes |
|---|---|---|
| Heuristic | 10/10 | reference script + 4-rung cage ladder |
| BC v2 | 7/10 | cloned the script, lower disp |
| PPO v3 | 7/10 | learned it, but shoves (0.65 m) |
| PPO + fail 100 | 7/10 | ties v3; misses 0,4,8 (v3: 0,1,4); wins in 2–7 steps |
| Diffusion UNet | 5/10 | gentlest grasps (0.16 m) |
| Reactive | 0/10 | floor, as designed |
| VLM (canary 2,4) | 2/2 | inherits cage ladder; $0 via cache |
| VLM (seeds 0-9) | 9/10 | lone miss seed 8 (wrong side); ladder fixes |
| VLM (adjacent 7) | 1/1, 1 attempt | push saves 3 ladder rungs vs push-free |
| VLM (sandwich 9) | 1/1 vs heur 0/1 | strict win restored: kept shear, first-try cage |
| autopush (sandwich 0-9) | 6/10 | ties heuristic; fixes 3,9 breaks 5,8 |
| autopush (adjacent 0-9) | 8/10 | +2 over heuristic (fixes 8,9), breaks nothing |
| autopush-shove (sandwich) | 7/10 | matches VLM 3/3 (s3✗ s8✓ s9✓); fixes 5,8 breaks 3 |
| autopush-shove (adjacent) | 7/10 | shove loses s3 vs clear's 8/10: effort is scene-dependent |
| decoy scan (0-9) | no selective regime | 0-5 all-fail, 6 all-pass, 7-8 effort-gated, 9 withholding |
| VLM (decoy 9) | 0/1 | pushed blue/green, lost where ladder wins push-free: no restraint |
| VLM (sandwich 3,8) | 1/2 | s8 kept (autopush broke it), s3 lost (autopush kept it) |
| SARSA(λ) thesis port | 0/10 ×6 configs | negative; relative-xy shows weak in-dist sign of life (13/60 vs 7/60, p≈0.07) |

Seed 0 falls only to the scripted pair (heuristic + VLM); seeds
2/5/7 fall to all capable ones. Four live failure modes, all
diagnosed with probes:

1. **VLM singulation is region-dependent.** Canary pushes skip:
   seed 2's radial start has 0/13 oriented-reachable waypoints
   (now skipped cleanly instead of DLS-fallback harm), seed 4's
   shear is unreachable both sides — so VLM == heuristic wherever
   the cage alone decides. But pushes DO land in kinder geometry:
   on the `adjacent` preset (10 mm single neighbor, seed 7) the
   VLM planned shove-blue, kept it (iso 0.098 → 0.294), and
   grasped first try while the heuristic needed all 4 ladder
   rungs. The full 0-9 VLM run (9/10, 8 fresh calls) found one
   VLM-worse case — seed 8, where side=left misses twice and near
   wins first try — fixed by the side ladder (mirror side after
   both entries fail), which restores structural VLM >= heuristic.
   The strict-win frontier is now the `sandwich` preset (two-sided
   15 mm, seed 9): VLM shoves both neighbors (blue kept) and grasps
   first try while the push-free ladder fails all 4 rungs.
   At 8 mm multi-sided gaps (surround/row/gate) one side always
   reverts, so min-isolation never crosses the gate — VLM == 0/3
   there too, with correct plans and insufficient primitive.
2. **Seed-4 cage bulldozes from every side.** The scripted cage
   knocks the target 0.59 m with zero final contacts, identically
   for near/left/right/far entries — while BC grasps the same
   scene. Diagnosed: the opening yaw sweeps extended links through
   blue, which dominoes into red before the arm ever touches it.
   Always-tucking before the yaw was tried and reverted (7/10 →
   6/10: fixed 0/4/9, broke canary 2 plus 5/7/8 — arrival-path
   edits reshuffle wrap outcomes). Fixed instead with a
   snapshot-and-retry second entry: a missed untucked cage
   retries tucked from a pristine restore (the entries'
   union is 10/10). Heuristic and VLM now 10/10 and 2/2.
3. **Singulation rate 0.00 everywhere.** Follows from (1): no
   push has produced a kept isolation gain on the benchmark
   seeds. Adversarial presets (surround/row/gate) exist for
   stress-testing this once pushes land.
4. **Push execution bulldozes the target in dense scenes.**
   Surround s0: a 0.06 m nudge moves red 0.40+ m (6× the
   command — arm motion through the 8 mm cluster, not the push
   itself), the keep-rule correctly reverts, isolation stays
   flat, ladder fails all 4. Order- and effort-independent
   (4 push orders revert the same pushes; nudge/clear/shove all
   revert them too); decoy s0 obj2 is the same mechanism
   (0.22 m target motion).
   Push-log entries now carry `tmove_m`/`dmove_m` telemetry.
   Fix: collision-aware push approach — the characterization
   test `test_surround_pushes_revert_on_target_bulldoze` is the
   red-to-green target for that work.

## 9. Code map

| Question | File |
|---|---|
| Task physics, reward, FC test | `env/whole_arm_grasp.py` |
| Obs/action spaces, shaping | `env/gym_wrapper.py` |
| IK, push, cage, reactive | `planning/primitives.py`, `scripts/demo_vlm_grasp.py` |
| VLM + heuristic planners | `perception/grounding.py` |
| Compare loop, metrics, report | `scripts/compare_planners.py` |
| PPO / BC / diffusion training | `scripts/train_grasp.py`, `train_bc.py`, `train_diffusion.py` |
| Denoisers + DDIM sampling | `scripts/diffusion_policy.py` |
| Demo collection / HF export | `scripts/collect_demos.py`, `scripts/export_hf_dataset.py` |
| This guide's numbers | `episodes/compare-0-9.json` |

## 10. Suggested learning path

1. Watch one seed's side-by-side GIF, then trace it in code:
   `compare_planners.py` → planner → `env.step`.
2. Change one action-space knob (`max_delta_q`, slew limit) and
   re-run a canary seed — feel what it constrains.
3. Train BC on 10 demos vs 130 (`--demos`, `--epochs`) and watch
   the FC rate move with data, not architecture.
4. Read `FiLMUNet1D.forward` against the MLP's: *where* does the
   obs enter each net? That's the whole hedging story.
5. Run the VLM on one fresh seed with `--planners vlm` and read
   its ledger entry: image hash in, JSON plan out, cost attached.
