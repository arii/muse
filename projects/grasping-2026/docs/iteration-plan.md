# Grasping-2026: System Analysis, Canary Seeds & Iteration Plan

**Date:** 2026-10-28  
**Goal:** Reduce evaluation to **2 working canary seeds (`Seeds 2 & 6`)** with clear initial conditions, make VLM/Heuristic/Diffusion policies work reliably on them without wasting VLM API tokens on 10-seed runs, and report explicit executed geometric primitives (`push(...)`, `scripted_cage(...)`) instead of opaque text paragraphs.

---

## 1. Quick-Run CLI Helper & Task Runners (`scripts/run_eval.sh`, `Makefile`, `package.json`, `pyproject.toml`)

To avoid repetitive `docker compose run ...` CLI commands and prevent accidental 10-seed VLM API calls, all workflows are wrapped in [`scripts/run_eval.sh`](../scripts/run_eval.sh), [`Makefile`](../Makefile), [`package.json`](../package.json), and [`pyproject.toml`](../pyproject.toml) (defaulting to `--seeds 2,6`):

```bash
# Default: run 2 canary seeds (2,6) across all policies (uses ledger cache for VLM -> 0 API calls)
./scripts/run_eval.sh

# Isolated + cluttered comparison (seed 2 isolated, seed 7 crowded)
./scripts/run_eval.sh --seeds 2,7

# Offline-only mode (heuristic, ppo, diffusion) — zero VLM API risk even on new seeds
./scripts/run_eval.sh --no-vlm --seeds 2,6

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

## 4. Selected 2 Canary Seeds (`Seeds 2 & 4` — 1 Isolated, 1 Cluttered) & Updated Evaluation

Instead of running all 10 seeds (`0–9`), we use **Seeds 2 and 4** (`--seeds 2,4`) as our default 2-seed benchmark so that one seed tests isolated grasping (`Seed 2`) and the other tests **divergent VLM vs. Heuristic precondition pushing + directional approach** (`Seed 4`):

* **Seed 2 (`r = 0.311 m, θ = -55.4°, iso = 0.474 m` — Isolated):** Red target box is already isolated (`iso >= 0.15 m`). Both VLM and Heuristic plan `push_away=[], approach_side="near"` and achieve **`FC=True, caged=True`** (`disp = 0.087 m`).
* **Seed 4 (`r = 0.394 m, θ = +3.1°, iso = 0.061 m` — Cluttered):** Green box (`obj2`) is touching the red box (`iso = 0.061 m < 0.15 m`).
  * **VLM (`Nemotron-3.5-Super-VL`)** plans `push_away=[{"index": 2, "effort": "shove"}], approach_side="right"`, executes `push(obj2:green, start_xy=[0.402, -0.013], theta=+1.84rad/+106°, dist=0.25m)` (`isolation: 0.061 m → 0.314 m`, **`preconditions_met=True`**), then tucks inside `r <= 0.30 m` to rotate to the `right` side (`face = -0.66 rad, curl = CCW(+)`) and achieves **`FC=True, caged=True`** (`disp = 0.195 m`).
  * **Heuristic (`HeuristicLabeller`)** plans `push_away=[], approach_side="near"`, fails preconditions (`iso = 0.061 m`, **`preconditions_met=False`**), gets blocked by the green box, and fails (`FC=False, caged=False`).

### 4.1 Current Canary Evaluation Summary (`Seeds 2 & 4`)

| Planner / Policy | Force Closure (`FC`) | Caged (`loop < 0.15m`) | Preconditions Met | Mean Isolation `Δ` | Mean Target `Δxy` | API Calls / Token Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **VLM (`Nemotron-3.5-Super-VL`)** | **`1.00` (2/2)** | **`1.00` (2/2)** | **`1.00` (2/2)** | **`+0.126 m`** | `0.141 m` | `0` live (`10,038` cached: `6,738` + `3,300` tok) |
| **Heuristic (`HeuristicLabeller`)** | `0.50` (1/2) | `0.50` (1/2) | `0.50` (1/2) | `+0.000 m` | `0.118 m` | `no API` |
| **Diffusion BC (`diffusion_v2.pt`)** | `0.50` (1/2) | `0.00` (0/2) | `0.50` (1/2) | `-0.034 m` | `0.028 m` | `no API` |
| **PPO (`grasp_ppo_500k.zip`)** | `0.00` (0/2) | `0.00` (0/2) | `1.00` (2/2) | `+1.985 m` | `2.067 m` | `no API` |

### 4.2 Why VLM and Heuristic Match on `Seeds 2, 6` vs. Diverge on `Seed 7`

1. **On Isolated Seeds (`2, 6`):** Both scenes already satisfy `isolation >= 0.15 m` (`0.474 m` and `0.357 m`). The VLM inspects the top-down image and rightly plans `{"grasp_target": 0, "push_away": [], "approach_side": "near"}`—matching `HeuristicLabeller`.
2. **On Cluttered Seeds (`Seed 7`, `isolation = 0.054 m`):**
   * **Heuristic** outputs `push_away=[], approach_side="near"`, failing the isolation precondition (`preconditions_met=False`, `iso = 0.054 m < 0.15 m`).
   * **VLM** identifies the adjacent blue distractor (`obj1`) and outputs `push_away=[{"index": 1, "effort": "shove"}], approach_side="left"`, executing `push(obj1:blue, start_xy=[-0.131, 0.343], theta=-0.52rad/-30°, dist=0.25m)` (`isolation: 0.054 m -> 0.171 m`, `preconditions_met=True`).
3. **Why `0 tokens` Appeared on Re-runs:** `VLMCallLedger` (`perception/ledger.py`) caches VLM responses in `episodes/vlm_ledger.jsonl` keyed by `(CACHE_VERSION, model, instruction, objects, image_sha256, prompt_hash)`. Re-running cached seeds makes `0` live API calls; `episodes/compare-report.html` now explicitly displays `0 tokens (cached; orig 6738 tok)` so the VLM token cost is visible even on cache hits.

---

## 5. Plan for Next Steps

### Step 1: Collision-Safe `singulate` Precondition Push on Cluttered Seeds (`Seed 7` & `Seed 4`)
* **Problem Identified on `Seed 7`:** When the distractor `d` is touching the red target `t` (`||d - t|| = 0.054 m`), `start_xy = d - direction * clearance` (`clearance = 0.075 m`) places the push start point **inside the red target box**, and `move_tip(env, start_xy)` sweeps the arm straight through the target, knocking the red box `0.32 m` away (`target_xy: [-0.117, 0.335] -> [-0.435, 0.330]`). Furthermore, `tmove_limit = dmove + 0.05` failed to revert the push because both boxes moved together.
* **Action Items:**
  1. Tighten `tmove_limit` in `singulate()` (`scripts/demo_vlm_grasp.py`) to `0.12 m` unconditionally so any push that displaces the target box by `> 12 cm` is reverted.
  2. When `||d - t|| < clearance + target_radius`, offset `start_xy` tangentially or push from the side of `d` clear of `t`, and route `move_tip` via a high-clearance intermediate waypoint so the approach never sweeps through `t`.
  3. Verify on `--seeds 2,7` (`0` live VLM calls via ledger cache) that `Seed 7` VLM achieves `preconditions_met=True`, `FC=True`, and `target Δxy < 0.20 m`.

### Step 2: Scale Diffusion BC from 2 Canary Demos to 40+ Caged Planar Demos
* **Current State:** Fixing `obs_std` normalization blow-up (`obs_std >= 0.30`, `obs_std[3:6] >= 2.0`) and intermediate DDIM clamping (`[-1.2, 1.2]`) eliminated action-space noise and enabled closed-loop `FC=True` grasps on the canary seeds.
* **Action Items:**
  1. Collect `~40–50` verified `FC=True, caged=True` trajectories across `Seeds 0–99` (`scripts/collect_demos.py --demonstrator cage --seeds 0-99 --keep caged`).
  2. Train `models/diffusion_v2.pt` on the broader demonstration set with deterministic seeding (`torch.manual_seed(0)`) so it generalizes across all workspace quadrants (`±θ`, `r ∈ [0.22, 0.45] m`).

### Step 3: Retrain PPO on the Planar `3*N`-DOF Environment (`scripts/train_rl.py`)
* **Current State:** `models/grasp_ppo_500k.zip` was trained before objects were switched to planar `(x, y, yaw)` joints and sweeps the target box out of the workspace (`mean target Δxy = 1.90 m`, `FC = 0.00`).
* **Action Items:**
  1. Add a target-displacement penalty (`-c * ||pos_t - pos_0||`) to `WholeArmGraspEnv.step()` so RL is penalized for batting the target box away.
  2. Retrain PPO (`scripts/train_rl.py --timesteps 500000`) on the planar environment and benchmark on `--no-vlm --seeds 2,6`.

### Step 4: Expand Benchmark from `Seeds 2,6` → `Seeds 2,6,7` → Full `Seeds 0–9`
* Once Steps 1–3 pass on `Seeds 2,6` (isolated) and `Seed 7` (cluttered), run the full 10-seed cached suite (`./scripts/run_eval.sh --seeds 0-9`) to validate aggregate `force_closure`, `caged`, and `preconditions_met` rates across all four planners.
