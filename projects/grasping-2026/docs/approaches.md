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

Multi-pass singulation (`--max-passes`, default 1 = off)
re-sweeps the push list until the 0.15 m gate or a fruitless
pass. It compounds partial progress where single sweeps stall:
decoy s0 flips fail→win (pass 2 re-nudges blue). But outcomes
are order-chaotic — identical kept-push sets in different
orders diverge by pass 3 (solver-history effects the snapshot
doesn't capture), and extra passes can over-push a winnable
scene (nearest-first order loses the same seed multi-pass
wins by hand order). Matrix: sandwich-shove unchanged 7/10,
decoy-shove unchanged 3/10, surround-shove 1/10. Cached VLM
replays (sandwich 9 win, decoy 9 loss) are unchanged too —
extra passes correctly no-op via the fruitless-pass stop.
Verdict: experimental, non-default; the robust fix for dense
scenes remains approach-aware sequencing, not more sweeps.

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
sits 3 below the ladder heuristic's 10/10 on random scenes. On presets (zero-shot,
never in demos): sandwich 4/10, adjacent 4/10, surround 2/10
({2,7} — beating every script's 0/10 there, barely), decoy
1/10 ({6}). Cloning degrades everywhere out of distribution,
worst where the demonstrator is weakest.

Coverage, not capacity, was the binding constraint:
`collect_ppo_demos.py --scene` (new) gathered 112 pc5
episodes on seeds 10–39 across all five scenes (27/25/
19/22/19 — disjoint from eval seeds 0–9), and BC
retrained on them (`models/bc_pc5mix.pt`) jumps to
**37/50**: random 9, surround 8, decoy 6, sandwich 8,
adjacent 6 — doubling v2's 18/50 and beating four of
six PPO checkpoints (only pc5's 40 and mixw's 39 lead).
Cloning distills PPO nearly whole; the 3-point gap is
the price of single-step MSE. Rarely cages, though —
pinches like its teacher's modal win.

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

Preset generalization (zero-shot): sandwich 5/10 (its only win
over BC's 4/10), adjacent 3/10, surround 0/10, decoy 0/10 —
total collapse on the two hardest presets, where even BC
scratches 2 and 1. Chunking doesn't transfer what the demos
never show. Surround ordering, complete: fail100 9/10 > v3
4/10 > BC 2/10 > diffusion = scripts 0/10 — reward-learning >
cloning > scripting on hard out-of-distribution clutter.

Diffusion does NOT follow BC up the coverage ladder:
retrained on the same 112 pc5 episodes, it falls to
8/50 (3/0/1/3/1 — worse than script-demo diffusion's
13/50 on 4 of 5 scenes). Mechanism, measured: pc5
episodes average 25 steps (min 2) vs scripted demos'
137, so chunk-16 windows are mostly hold-last padding
and total data is 6.4× smaller (2.8k vs 17.8k steps).
Chunking semantics need long demos; single-step MSE
doesn't care. Diffusion's data appetite, not its
denoiser, is the binding constraint here.

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
Wins are near-instant (2–7 steps); seed 8 is a 200-step chase
(flickering contact, red never still, one grazed enclosed
step at t=50) — the −100 did not cure non-closing either,
it just moved the failure.

Preset generalization (both checkpoints zero-shot, never
trained on presets): fail100 crushes the hardest scripted
preset, surround, **9/10** (only seed 7 fails) where every
planner scores 0 — bulldoze-with-restraint: shove the crowd
while the −100 keeps red in reach. v3 manages 4/10 there
({0,2,6,8}; unpunished shoving loses red out of reach).
Elsewhere: decoy 7/10 v3 vs 6/10 fail100, sandwich 6/10 both,
adjacent 4/10 v3 vs 5/10 fail100. So the −100 teaches PPO
exactly the skill dense clutter demands — while the same −100
killed linear SARSA stone dead. Representation is the whole
game.

Per-step traces retired the word "camping" for three distinct
fail100 failure modes, all with `enclosed=0` throughout: (1)
**chase** (adjacent 1): 200 steps of 1↔2-link flicker, red
batted 0.1–0.3 m, never enclosed; (2) **rest** (adjacent 2):
static 2-link contact, red frozen, the only true camp;
(3) **shove-out + idle** (surround 7): red ejected past 1 m
by t=40, then 160 contactless steps — not a camp at all.
Wins are all pinches (caged=False) in 12–16 steps. Common
thread: contact pays 0.5/link but enclosure pays nothing, so
there is no gradient from touching to wrapping. Fix tested:
`--enclosure-bonus 1.0` (train-only, every enclosed step
incl. FC) — at step_penalty 2.0 a 2-link enclosed hold nets
exactly 0 (no farm), above a camp's −1/step and far below
FC's +20 premium (`models/grasp_ppo_enc1`, local only).

enc1 (same 500k recipe + enclosure bonus) is the best
overall policy at 36/50 FC (fail100 33, v3 28): random
**9/10** (only seed 0 fails — ties VLM), decoy 8/10 (best
of all), sandwich 7/10 (best of all), adjacent 5/10 (ties
fail100), but surround **7/10** — a −2 regression vs
fail100's 9. Caging jumps everywhere (random 6/10, sandwich
4/10, surround/adjacent 3/10; decoy stays all-pinch):
the bonus teaches wrapping, and wrapping wins — except in
the densest clutter, where fail100's bulldoze-and-pinch
beats enc1's enclose-attempts (surround seeds 0–1 stall
back into 150+ contact-reward rest). Learning curve on
random: 5/10 at 200k → 8/10 at 450k → 9/10 at 500k.

Fourth failure mode from the traces: **whiff** — first
touch ejects red (2.4 m by t=20 on random seed 0) before
any contact payment, then 200 idle steps. Seed 0 random is
missed by all three PPO checkpoints (v3/fail100/enc1) while
scripts win it: PPO's first approach is too violent for
that geometry, and training's −100 terminates whiffs, so
no recovery is ever learned.

Action-level traces sharpened the surround story: enc1's
surround-0 "rest" is **push-into-pin** — saturated actions
(|a|≈1.7) with 0.2 rad/step joint motion against a frozen
scene (red pinned at 0.52 m displacement for 180 steps),
while fail100 on the identical scene keeps red moving and
pinches at t=16. enc1 learned hold-contact as the route to
enclosure on loose random scenes; on surround the wrap
never comes and it shoves into the pin forever. The reward
already punishes this (−1/step + timeout −100) — the
policy simply never meets pins in training. Fix under
test: mixed-scene training (`--train-scenes`, round-robin
layouts across resets, duplicates weight the mix;
infeasible presets re-draw 3× then fall back to random,
counted) — `models/grasp_ppo_mix5`, equal 5-way mix, 500k
steps, 0 training fallbacks.

mix5 lands at 35/50 — a wash with enc1's 36, with
competence reallocated, not added: surround jumps back to
**9/10** (matches fail100; the only miss is low-contact
seed 7, no push-into-pin anywhere) and random holds 9/10
while winning seed 0 caged — the first PPO to take the
nemesis seed. But decoy falls 8→6 and sandwich 7→6, both
below enc1. Traces show why: **strategy interference**.
mix5 opens with surround-style bulldozing everywhere, and
on restraint scenes the first touch whiffs red 1.3–1.8 m
out of reach (decoy 4, sandwich 0), followed by 180 steps
of contactless flailing. Training on a scene can hurt it:
decoy/sandwich transferred better from random-only enc1
(500k focused steps) than from 100k direct steps
shouting over a bulldoze prior. Random mix5 misses: {1}.
Learning curve on random: 0/10 at 100k → 6/10 at 250k →
9/10 at 500k — the 5-way mix learns slower, then arrives.

Follow-up on interference: fail100's decoy misses are
{3,4,5,8} (all low-contact non-engagement) — notably it
wins the withholding seed 9 caged in 12 steps,
while mix5 whiffs it. Both random-only policies win 9;
only the surround-exposed one loses it, which sharpens
the story: surround calibrates a hotter opening shove
than random scenes do, and decoy punishes exactly that.
Test confirmed: `models/grasp_ppo_mixw`, weighted mix
(random×2 + all four presets: random 1/3, surround 1/6),
500k steps, 0 fallbacks — **39/50**, the new best
overall. Surround holds 9/10 (halving its share cost
nothing), random holds 9/10 (seed 0 wins again, 6 caged),
sandwich jumps to **8/10** (best of all, +1 over enc1),
adjacent to **6/10** (best of all learned; scripts still
lead via autopush 8), decoy recovers to 7/10 (ties v3;
enc1's 8 still leads — withholding seed 9 still missed).
The interferer was the surround share, and mix weight was
the fix: less bulldoze-calibration, more random grounding.
Curve on random: 7/10 at 250k → 9/10 at 500k.

All five remaining mixw failures trace to one mode:
**whiff**. Adjacent {0,4,8,9} and decoy 9 all eject red
1.0–1.7 m out of reach on first touch (t=2–6), then flail
contactless for ~195 steps. The timeout −100 already
punishes whiffs, but ~197 steps after the fatal shove —
too delayed for credit assignment. Fix tested: `--precontact-disp-penalty 5.0`
(train-only; taxes red per-step motion while
links_touching==0, densely and immediately; post-contact
bulldozing stays exempt) — `models/grasp_ppo_pc5`, mixw
recipe otherwise, 500k, 0 fallbacks — **40/50**, best
overall. The whiff is dead where it lived: decoy jumps to
8/10 (ties enc1; withholding seed 9 won caged in 12
steps), sandwich to **9/10** (best of all; seed 2 even
converts a 178-step chase). Random and surround hold 9/10.
One regression: adjacent falls 6→5. Its traces split:
seed 1 is a 1-link graze-freeze (gentler approach lands
in a new local optimum), seed 0 a *slow* crowder-mediated
ejection (0.69 m over ~20 steps — only −0.17/step at
coef 5, too weak to reshape). So the coef bites violent
whiffs but not quasi-static shoves; a hotter coef is the
obvious next run. Chase resurfaces on 2 seeds (surround 3,
sandwich 1, ~100 rew fails) — the residual mode when red
neither ejects nor grasps.

Hotter coef tried and killed: `models/grasp_ppo_pc20`
(same recipe, `--precontact-disp-penalty 20.0`) sat at
0/10 on random at 250k with ep_rew_mean flat at ≈−130
since 150k (pc5's curve at the same points: −107, −85,
climbing). Killed at 300k. Diagnosis — the **whiff
subsidy**: out-of-reach terminates the episode, so a
whiff costs −2×~15 steps −100 ≈ −130 while a slow
honest failure costs −2×200 −100 = −500. Whiffing is
the least-bad failure, and no coef below ~370 flips
that ordering (coef×1 m ejection vs 370 time saved;
above 370 the policy freezes instead — worse). The coef
knob is useless here; the fix must be structural:
charge whiffs the full bleed (non-terminating
out-of-reach, or equivalent economics), so slow-gentle
failure outranks ejection and FC still outranks both.

Structural fix tried and killed: `--no-terminate-out-
of-reach` (out-of-reach latches −100 once, no early
exit) — `models/grasp_ppo_pc5nt`, pc5 recipe otherwise,
killed at 300k. It converged to the exact mirror trap:
ep_rew_mean pinned at −501 (the slow-failure number),
0/10 with 0.0 reward on every seed — the arm waves in
free space and never approaches red. Inverting the
basin (freeze −500 now beats whiff −620) just moved
the least-bad failure; neither ordering gives a
*gradient toward* FC, because approaching red pays
nothing while any jostle along the way is punished.
Next: potential-based approach shaping (pay tip→red
progress pre-contact, Ng-style, bounded so hovering
can't exploit it) — guidance toward contact, not just
punishment for disturbing it.

Under training: `--approach-potential 2.0`
(Phi = −k·dist(tip, red) while untouched, else 0;
F = Phi'−Phi at gamma=1, so excursions telescope to
~0 and hovering pays exactly 0) — `models/grasp_ppo_ap2`,
pc5 recipe with the precontact tax *swapped out* for
the potential (clean attribution: must beat mixw's 39
to show guidance adds anything, pc5's 40 to retire
the tax). Pays approach AND re-approach after losing
red; whiff flight reads negative automatically.

Verdict: the swap fails both bars — `models/grasp_ppo_ap2`
lands at **38/50** (mixw 39, pc5 40). Random holds 9/10
and adjacent recovers to 6/10 (mixw's level), but
surround slips 9→8, decoy 8→7 (withholding 9 lost),
sandwich 9→8. Traces: whiffs return without the tax
(surround 0 ejects red 1.3 m on first touch; decoy 9
bleeds out through flickering chase). Guidance at k=2
(±0.1–0.3/step against −2 time and sparse +20) is too
weak to matter, while the removed −5 ejection bite was
load-bearing; tip-approach payments may also distort
toward tip-first poking over enveloping (all ap2
surround wins are pinches, caged 0/10). The tax stands.
Untested: potential + tax combined — but the potential
showed no positive signal anywhere, so this thread
closes here; pc5's 40/50 stands as best.

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
| PPO fail100 (surround 0-9) | 9/10 | scripts score 0 here; seed 7 shoves red out then idles |
| PPO v3 (surround 0-9) | 4/10 | passes 0,2,6,8; shoves red out elsewhere |
| PPO presets (decoy/sand/adj) | 6-7/6/4-5 | v3/fail100 close; fail100 failures are chase/rest/shove-out, all enc=0 |
| BC presets (surr/decoy/sand/adj) | 2/1/4/4 of 10 | degrades OOD; beats scripts' 0 only on surround |
| Diffusion presets (surr/decoy/sand/adj) | 0/0/5/3 of 10 | total collapse on 2 hardest; only beats BC on sandwich |
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
| multipass-3 (sandwich/decoy/surround) | 7/3/1 of 10 | helps one seed (decoy s0 hand-order), neutral else; order-chaotic |
| defer-blocked (+multipass-3) | 7/3/1 of 10 | identical miss sets: reorders correctly, arrival graze binds |
| tactile force probe | gate viable, shove not | contact at 0.01-0.04 N / 1.3 cm red motion; glide + chatter after |
| VLM (sandwich 3,8) | 1/2 | s8 kept (autopush broke it), s3 lost (autopush kept it) |
| SARSA(λ) thesis port | 0/10 ×6 configs | negative; relative-xy shows weak in-dist sign of life (13/60 vs 7/60, p≈0.07) |
| autopush (random 0-9) | 10/10 | blind pushing ties the ladder on easy scenes; costs nothing |
| PPO enc1 (random 0-9) | 9/10, 6 caged | ties VLM; only seed 0 fails (first-touch whiff) |
| PPO enc1 (presets) | 7/8/7/5 surr/decoy/sand/adj | best overall 36/50; surround −2 vs fail100 (enclose-attempts stall) |
| PPO mix5 (random 0-9) | 9/10, 4 caged | holds enc1; wins nemesis seed 0 caged; only seed 1 fails |
| PPO mix5 (presets) | 9/6/6/5 surr/decoy/sand/adj | 35/50: surround fixed, decoy/sandwich regress (bulldoze interference) |
| PPO mixw (random 0-9) | 9/10, 6 caged | holds enc1/mix5; only seed 4 fails |
| PPO mixw (presets) | 9/7/8/6 surr/decoy/sand/adj | 39/50 best overall; sandwich/adjacent best of all |
| PPO pc5 (random 0-9) | 9/10, 4 caged | holds; only seed 2 fails (low contact) |
| PPO pc5 (presets) | 9/8/9/5 surr/decoy/sand/adj | 40/50 best overall; whiffs dead, adjacent −1 |
| PPO ap2 (random 0-9) | 9/10, 3 caged | holds; only seed 0 fails |
| PPO ap2 (presets) | 8/7/8/6 surr/decoy/sand/adj | 38/50: swap fails both bars, tax stands |
| BC on pc5 demos (112 eps) | 9/8/6/8/6 rand/surr/decoy/sand/adj | 37/50: coverage, not capacity, bound cloning |
| Diffusion on pc5 demos | 3/0/1/3/1 | 8/50: short PPO eps (25 steps) starve chunk-16 |

Scene × planner matrix (FC /10, seeds 0–9, default configs;
VLM preset grids unmeasured — API cost; probes only):

| Scene | Heur | Reac | Auto | BC | Diff | v3 | f100 | enc1 | mix5 | mixw | pc5 | ap2 | BCmx | DFmx | VLM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 10 | 0 | 10 | 7 | 5 | 7 | 7 | 9 | 9 | 9 | 9 | 9 | 9 | 3 | 9 |
| surround | 0 | 0 | 0 | 2 | 0 | 4 | 9 | 7 | 9 | 9 | 9 | 8 | 8 | 0 | — |
| decoy | 2 | 0 | 2 | 1 | 0 | 7 | 6 | 8 | 6 | 7 | 8 | 7 | 6 | 1 | — |
| sandwich | 6 | 0 | 6 | 4 | 5 | 6 | 6 | 7 | 6 | 8 | 9 | 8 | 8 | 3 | — |
| adjacent | 6 | 0 | 8 | 4 | 3 | 4 | 5 | 5 | 5 | 6 | 5 | 6 | 6 | 1 | — |

Per-seed wins (FC): surround v3 {0,2,6,8}, BC {2,7};
decoy heur {6,9}, auto {3,6}, BC {6}; sandwich heur
{2,4,5,6,7,8}, auto {2,3,4,6,7,9}, BC {4,5,6,7}, diff
{2,4,5,7,8}; adjacent heur {2–7}, auto {2–9}, BC
{2,4,5,7}, diff {2,4,5}; enc1 random {1–9}, surround
{2,4–9}, decoy {1–7,9}, sandwich {1,2,4–8}, adjacent
{1–4,6}; mix5 random {0,2–9}, surround {0–6,8,9},
decoy {0,2,3,5,7,8}, sandwich {2–5,7,9}, adjacent
{2–5,7}; mixw random {0–3,5–9}, surround {0–6,8,9},
decoy {0–4,6,8}, sandwich {0–6,8}, adjacent
{1–3,5–7}; pc5 random {0,1,3–9}, surround {0–2,4–9},
decoy {0,1,3,4,5,6,7,9}, sandwich {0,2–9}, adjacent
{3–7}; ap2 random {1–9}, surround {1–8}, decoy
{0,1,3–7}, sandwich {0,2–7,9}, adjacent {2,3,5–8};
BCmix random {1–9}, surround {0–3,5,6,8,9}, decoy
{1,2,4–6,9}, sandwich {1–7,9}, adjacent {1–3,6–8};
DFmix random {3,5,7}, surround {}, decoy {6},
sandwich {5–7}, adjacent {5}.
The fresh scripted grid
reproduced the earlier gap runs seed-for-seed
(determinism check passes). Three readings: (1) scripts
collapse only on surround/decoy, where restraint +
bulldozing must mix — exactly where PPO wins; (2) cloning
never beats its demonstrator anywhere (BC ≤ heur on all 5
scenes); (3) autopush ≥ heuristic on all 5 scenes (ties 4,
wins adjacent +2) — blind pushing never hurts, sometimes
helps.

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
4. **Push approach bulldozes the target in dense scenes.**
   Surround s0: red moves 0.12 m during the 33-step approach,
   before the push begins (execution adds ~0.01) — a 0.06 m
   nudge totals 0.40+ m target motion, the keep-rule correctly
   reverts, isolation stays flat, ladder fails all 4. Ruled
   out, all measured: push order, push effort, Cartesian-line
   approach (reshuffles, one regression), staged via detour
   (worse: 0.73 m — travel length dominates), quasi-static
   micro-stepping (worse: sustained drag). Geometry, not
   dynamics: link segments provably cross box discs (link1
   7 cm through the east neighbor; push start pose grazes red),
   so no single-push path shape can fix it. Corridor-first
   sequencing was tried next (`--defer-blocked`: link-sweep
   clearance check defers blocked pushes until clear ones go
   first): mechanically correct (surround s0 reorders 2-first),
   outcome-neutral on all 30 seeds (surround 1, decoy 3,
   sandwich 7 of 10, identical miss sets) — because arrival
   graze, not transit, binds once corridors clear. Two more
   inversions failed: deliberate red-transport (corner-drag red
   south into the open: tip wedges, red flung 0.27 m, iso
   drops) and bulldoze-through (`--tmove-cap 0.50`: shoves keep
   with red displaced 0.43 m, but displaced red is ungrasped —
   the revert rule's 0.12 m cap turns out to protect graspability,
   not just tidiness). Eight approaches ruled out total; dense
   3-sided 8 mm clutter wants closed-loop control, not open-loop
   primitives — proven by PPO fail100's 9/10 on surround (shove
   the crowd, keep red in reach), which also reframes the gap as
   scripts-vs-learning rather than task impossibility. A force-
   observability probe (2 mm tip steps into red, cleared scene)
   says contact-gating is viable but force-shoving is not:
   first contact registers at 0.008–0.04 N with red moved
   ~1.3 cm (usable stop threshold), but contact is intermittent
   (shove → glide → catch-up: 0.000, 0.038, 0.000, …,
   occasional 1 N spikes) because table friction is too low to
   hold red — it escapes at 5.5 cm over a 3 cm commanded press.
   So a future servo can stop *before* bulldozing, but cannot
   push precisely. Push-log entries carry `approach` +
   `tmove_m`/`dmove_m` telemetry;
   `test_surround_pushes_revert_on_target_bulldoze` stays red.

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
