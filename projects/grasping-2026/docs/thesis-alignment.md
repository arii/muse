# Thesis alignment: Anders (2014) vs grasping-2026

Source: `892740209-MIT.pdf` at workspace root ("Learning a Strategy for
Whole-Arm Grasping"), read 2026-11-03 (§3.2 MDP, §3.3 sim, §4.1 eval).
No 1-1 replication — similar-type setup only (thesis source unavailable).

## What matches

| Thesis | Ours | Notes |
|---|---|---|
| Continuous state (joints + object x, y, θ) | q, qvel, target xy+half, distractor offsets | Ours adds qvel + clutter |
| Incremental joint actions | Absolute joint targets (±2.6) | Ours is absolute, not ±0.1 steps |
| PD controller to commanded angles | MuJoCo position actuators, 20 substeps | Same quasi-static role |
| Reward: goal ≥ 0, failure −100, contact shaping, −1/step | 0.5/link + 2.0 FC, terminate on FC (+ train-only penalty/bonus) | Same shape; ours lacks failure states |
| Force closure = origin in wrench convex hull | Same test (pairwise separating directions) | Verified against pinch/poke cases |
| Eval: 100 test points, boolean success + return | Canary seeds + seeds 0–9, FC + reward | Ours is thinner; region coverage TBD |

## Explicitly out of scope

- **Bimanual / mirrored dual arms + torso contacts.** Thesis §4.2's headline
  result. Ours is single 3-link arm; dual-arm doubles the env + demo work.
- **Sim-to-real (PR2).** No physical robot; sim-only by project goal.
- **Failure-state shaping.** Thesis terminates on timeout (35 steps),
  arm self-collision, and object-out-of-reach with −100. Ours truncates
  silently at 200 — part of why PPO camps. Cheap to add later.
- **Uncertainty robustness** (thesis §4.1.5 noise studies). We train/test
  on exact state; domain randomization is future work.
- **SARSA-family learner.** Replaced by PPO/SAC + BC + diffusion (the
  modern-methods learning goal). A linear-gradient SARSA port would be
  the faithful baseline; not scheduled.

## Regime notes (Gualtieri precedent: train-in-clutter matters)

- Isolation-vs-clutter: heuristic scores 0.50 FC on isolated canary but
  0.00 on adversarial presets (surround/row/gate) — clutter gap documented.
- Train/test-regime: BC-1obj narrow-vs-wide ablation below; cross-regime
  (1-obj model on 4-obj env) is architecturally blocked by obs-dim
  mismatch — needs a shared obs (e.g. fixed slots, distant slots zeroed).

## Narrow-vs-wide BC ablation (2026-11-03)

- BC-1obj trained on 1-object demos seeds 0–99 (87 FC): eval seeds 0–2
  (result recorded in compare report).
- Narrow variant: trained on seeds 0–9 only (~8 FC demos), same eval.
- Expectation (thesis §4.1): narrow training generalizes worse across
  the region; gap size measures scene-coverage sensitivity.
