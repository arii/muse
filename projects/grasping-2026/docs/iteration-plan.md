# Grasping-2026: System Analysis, Canary Seeds & Iteration Plan

**Date:** 2026-10-28  
**Goal:** Reduce evaluation to **2 working canary seeds (`Seeds 2 & 6`)** with clear initial conditions, make VLM/Heuristic/Diffusion policies work reliably on them without wasting VLM API tokens on 10-seed runs, and report explicit executed geometric primitives (`push(...)`, `scripted_cage(...)`) instead of opaque text paragraphs.

---

## 1. Quick-Run CLI Helper (`scripts/run_eval.sh`)

To avoid repetitive `docker compose run ...` CLI commands and prevent accidental 10-seed VLM API calls, use [`scripts/run_eval.sh`](../scripts/run_eval.sh) (which defaults to `--seeds 2,6`):

```bash
# Default: run 2 canary seeds (2,6) across all policies (uses ledger cache for VLM -> 0 API calls)
./scripts/run_eval.sh

# Offline-only mode (heuristic, ppo, diffusion) — zero VLM API risk even on new seeds
./scripts/run_eval.sh --no-vlm --seeds 2,6

# Single-seed fast check
./scripts/run_eval.sh --seeds 2

# Re-collect planar cage demos and retrain models/diffusion_v2.pt before evaluating
./scripts/run_eval.sh --retrain-diffusion
```

---

## 2. How the VLM Policy Works & How It Reports Actions

Previously, the HTML report only displayed a free-text paragraph (`target 0 (red), push none, approach near: <rationale>`), making it unclear what the VLM policy actually outputs vs. what the low-level controller executes.

The pipeline is a **two-stage hierarchical policy**:

1. **Stage 1 — VLM Semantic Grounding (`perception/grounding.py`):**
   * **Input:** Top-down RGB render (`480×640`) + object list (`index, color`) + natural-language instruction (`"grasp the red box in isolation"`).
   * **Output (Strict JSON Schema):**
     ```json
     {
       "grasp_target": 0,
       "push_away": [{"index": 1, "effort": "shove"}],
       "approach_side": "near",
       "rationale": "..."
     }
     ```
2. **Stage 2 — Geometric Action Primitive Resolver (`scripts/demo_vlm_grasp.py`, `planning/primitives.py`):**
   * **Precondition Push Primitive (`planning.primitives.push`):**
     For each entry in `push_away`, computes the distractor's target-facing edge `start_xy`, push angle `theta` (pointing radially away from the target box), and push distance `dist` (`nudge=0.06m`, `clear=0.15m`, `shove=0.25m`), then drives the oriented 3rd link (`ik_tip_oriented`) along `(start_xy, theta, dist)`. Reverts via physics snapshot if isolation does not improve.
   * **Whole-Arm Enveloping Cage Primitive (`scripts.demo_vlm_grasp.scripted_cage`):**
     Computes a radius-adaptive lead angle `lead` around `target_xy` (`r, theta_t`), parks `link1` at `face = theta_t - s * lead` while opening `link2, link3` (`open_q = -0.35 * s`) away from the box so base rotation never sweeps across the target, and smoothly interpolates joint targets through `4` waypoints to `final_q = [face, 1.85*s, 2.25*s]` until `force_closure=True` and `caged=True`.

Both the **Semantic Plan** and the **Executed Geometric Primitives** (`push(obj, start_xy, theta, dist)` and `scripted_cage(target_xy, face, lead, curl, open_q -> final_q, steps)`) are now explicitly logged in `metrics["pushes"]` and `metrics["cage_trace"]` and rendered in [`episodes/compare-report.html`](../episodes/compare-report.html).

---

## 3. Root-Cause Fixes Applied

### 3.1 Simulator (`env/whole_arm_grasp.py`)
* **Bug Fixed in `_obs()`:** `qvel = self.data.qvel[:3].copy()` read the first 3 DOFs of `data.qvel`, which after adding planar object joints (`obj0_x, obj0_y, obj0_yaw`) could alias object velocities if joint ordering changed. Updated to `self.data.qvel[self._arm_qpos].copy()`.

### 3.2 Enveloping Grasp Controller (`scripts/demo_vlm_grasp.py:scripted_cage`)
* **Why Old `scripted_cage` Failed (1/10 FC):**
  1. Tucked `q2=+1.8*s, q3=+2.0*s` *toward* the target side and then unfolded to `+0.15*s`, whipping `link2` and `link3` across `theta_target` like a baseball bat and knocking the target box `0.55–2.67 m` away.
  2. Sent step-function joint targets without linear interpolation.
* **Fix:**
  * Opens `q2, q3` slightly to the *outside* (`-0.35 * s`) while yawing `j1` to the nearer side of the target (`face = theta_t - s * lead`), then smoothly interpolates across `4` waypoints (`-0.35*s -> +0.25*s -> +0.85/1.05*s -> +1.85/2.25*s`), stopping when both `force_closure=True` and `caged=True`.
  * Across `Seeds 0–99`, this single fix jumped `scripted_cage` from **`10%` FC (`1/10`) to `51%` FC (`51/100`) and `44%` closed-loop `caged` (`44/100`)**.

### 3.3 Why Diffusion Was Outputting Noise (`scripts/train_diffusion.py`)
* **Root Cause Diagnosis:**
  1. **Observation Normalization Blow-Up (`max_obs_diff = 346.6`):** In `train_diffusion.py`, `obs_std = flat.std(axis=0) + 1e-6` had tiny standard deviations (`~1e-3`) on un-contacted distractor offsets and quasi-static `qvel`. During closed-loop rollouts, a `0.02` velocity or position perturbation at step 24–32 produced a `17x–346x` standard-deviation spike in `obs_n`, saturating the MLP denoiser output to `[-1, -1, +1]`.
  2. **100-Step Noise Schedule Mismatch:** `DDPMScheduler(num_train_timesteps=100)` with default linear `beta_end=0.02` left $\bar{\alpha}_{99} = 0.364$ (`60%` clean signal at $t=99$), whereas inference starts from pure Gaussian noise $\mathcal{N}(0, I)$.
* **Fix:**
  * Floored `obs_std` at `0.15` (and masked `qvel` variance so transient velocity spikes cannot corrupt state conditioning).
  * Switched to `beta_schedule="squaredcos_cap_v2"` with `clip_sample=True` and low-$t$ weighted MSE loss so $\bar{\alpha}_{99} \to 0$ at $t=99$ and final denoising steps are smooth.

---

## 4. Selected 2 Easy Canary Seeds (`Seeds 2 & 6`)

Instead of running all 10 seeds (`0–9`), we use **Seeds 2 and 6** (`--seeds 2,6`) as our fast short-term benchmark:
* **Seed 2 (`r = 0.311 m, θ = -55.4°, iso = 0.474 m`):** Isolated red target box in the lower-right quadrant. Both VLM and Heuristic achieve **`FC=True, caged=True`** with `3` links touching and `0.10 m` target displacement.
* **Seed 6 (`r = 0.345 m, θ = -43.1°, iso = 0.357 m`):** Isolated red target box at mid-radius. Both VLM and Heuristic achieve **`FC=True, caged=True`** with `0.11 m` target displacement.
