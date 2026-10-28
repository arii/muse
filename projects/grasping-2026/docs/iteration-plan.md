# Grasping-2026: System Analysis, Canary Seeds & Iteration Plan

**Date:** 2026-10-28 (Post-commit `10da1bf`)  
**Goal:** Achieve reliable Force Closure (`FC=True`) and closed-loop Caging (`caged=True`) across both the VLM/Heuristic classical pipeline and the learned PPO/Diffusion policies without wasting iteration time running all 10 seeds on every change.

---

## 1. Current Baseline Summary (`Seeds 0–9` on Commit `10da1bf`)

| Approach | Force Closure (`FC`) | Closed-Loop `caged` | Preconditions Met (`iso >= 0.15m`) | Mean Isolation $\Delta$ | Mean Target Displacement ($\Delta xy$) | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vlm`** (`Nemotron-3.5-Super-VL`) | `0.10` (1/10, Seed 4) | `0.10` (1/10, Seed 4) | **`1.00`** (10/10) | **`+0.037 m`** | `~0.81 m` | `scripted_cage` elbow/unfold sweeps knock box away; `approach_side` (`left`/`far`) points `link1` 90°–180° away from box |
| **`heuristic`** (color regex) | `0.10` (1/10, Seed 4) | `0.10` (1/10, Seed 4) | `0.80` (8/10) | `+0.000 m` | `~0.81 m` | No singulation on crowded Seeds 4 & 7; `scripted_cage` knocks target away (`0.44–2.67 m`) on 9/10 seeds |
| **`ppo`** (`grasp_ppo_500k`, stale) | `0.10` (1/10, Seed 9) | `0.00` (0/10) | — | — | `~2.10 m` | Trained on old `freejoint` + old spawn without `--max-delta-q` or `--step-penalty`; flings boxes across plane |
| **`diffusion_v2`** (retrained on 13 planar FC demos) | `0.10` (1/10, Seed 8) | `0.00` (0/10) | — | — | `~1.15 m` | Loss converged (`0.994 → 0.309`), recalls `4/13` (`31%`) demo seeds, but capped by small (13-episode) demonstrator dataset |

---

## 2. Component-by-Component Root Cause Analysis

### 2.1 Simulator & Physics (`env/whole_arm_grasp.py`, `env/arm.xml`)
* **What Fixed (Working):**
  * Replacing 6-DOF `freejoint` boxes with planar `slide_x` + `slide_y` + `hinge_yaw` joints (`z = 0.06 m` fixed) completely eliminated 3D box tipping, z-wedging, and MuJoCo `QACC` solver explosions (`0` warnings across 100 rollouts).
  * Tightening the closed-loop caging metric (`_loop_closed`: tip within `0.15 m` of `link1` segment) eliminated false-positive open-arm cages (e.g., Seed 5).
  * Narrowing spawn range to $r \in [0.28, 0.40]\text{ m}$, $\theta \in [-2.4, +2.4]\text{ rad}$ eliminated unreachable spawns beyond the `j1` joint limit (`±2.6 rad`).
* **Remaining Simulator Issues:**
  1. **Boxes Spawn Inside the Arm's Elbow Radius (`r = 0.28–0.37 m`):**
     `link1` is `0.30 m` long (`0.335 m` with capsule radius). Even when `j2` and `j3` are tightly folded (`q2=1.8, q3=2.0`), the wrist `j2` sits at radius:
     $$r_{j2} = \sqrt{0.30^2 + 0.25^2 + 2(0.30)(0.25)\cos(1.8)} + 0.030 = \mathbf{0.374\text{ m}}$$
     Because boxes spawn at $r \in [0.28, 0.40]\text{ m}$, **rotating base joint `j1` while folded sweeps the `0.374 m` elbow/wrist straight through any box spawned between `0.28 m` and `0.37 m`** (e.g., Seed 1 at `r = 0.341 m` is knocked `2.67 m` away; Seed 8 at `r = 0.319 m` is knocked `1.22 m` away).
  2. **Low In-Plane Sliding Damping:**
     The planar `slide_x` and `slide_y` joints have no joint damping (`damping=0`), so a single high-velocity contact from an un-interpolated joint step sends boxes skating `1–6 m` across the frictionless joint coordinates.

### 2.2 Heuristic & `scripted_cage` Controller (`scripts/demo_vlm_grasp.py`)
`scripted_cage` currently executes 5 open-loop phases:
```python
phases = [
    ([q_cur[0], 1.8 * s, 2.0 * s], 60),   # Phase 1: tuck in-place
    ([face, 1.8 * s, 2.0 * s], 60),       # Phase 2: yaw while tucked
    ([face, 0.15 * s, 0.15 * s], 120),    # Phase 3: unfold/extend
    ([face, 0.55 * s, 0.55 * s], 120),    # Phase 4: reach
    ([face, 1.8 * s, 2.2 * s], 200),      # Phase 5: curl
]
```
This controller fails on 9/10 seeds due to **three geometric bugs**:
1. **Phase 2 → Phase 3 Unfolds *Across* the Target Box (Baseball-Bat Sweep):**
   When `face = theta_t - 0.35` (`20°` CW of the target) and `s = +1`, Phase 2 holds `q2 = +1.8, q3 = +2.0`—which folds `link2` and `link3` **counter-clockwise across the front of the target box** (`theta_t + 1.45 rad`). When Phase 3 commands `q2 = +0.15, q3 = +0.15`, `link2` and `link3` **unfold clockwise directly through `theta_t` at radius `0.30–0.75 m`**, swatting the target box `0.55–0.65 m` away before Phase 4/5 even begins!
2. **No Joint Trajectory Interpolation Between Phases:**
   Each phase runs `for _ in range(ticks // 5): env.step(np.array(angles))`. Unlike `move_tip()` (which linearly interpolates `(1 - a) * q_start + a * q_end`), `scripted_cage` commands the final phase target on tick 0, causing the position actuators (`kp = 40/30/20`) to whip at peak angular velocity on the first step of every phase.
3. **Insufficient Lead Angle (`0.35 rad`) for `link1` Clearance:**
   At `r = 0.32 m`, a box with half-width `0.045 m` plus link capsule radius `0.035 m` subtends `~0.28 rad` from the origin, and the `j1` elbow sits at `r = 0.30 m`. A `0.35 rad` lead angle leaves `j1` pressing into the near corner of the box so `link2` pushes the box outward instead of wrapping around its far face (`~0.55–0.65 rad` lead angle is needed so `j1` clears the side of the box and `link2` reaches beyond `r_target`).

### 2.3 VLM Grounding & Action Space (`perception/grounding.py`, `planning/primitives.py`)
* **What Works:**
  * Structured output (`PLAN_SCHEMA`) + `reasoning_effort="low"` + `max_tokens=8192` achieved **`10/10` valid plans (`0` errors)**.
  * Precondition reasoning (`push_away` + `effort`) achieved **`100%` preconditions met (`1.00`)**, accurately shoving crowding boxes on Seed 4 (`iso: 0.061 → 0.314 m`) and Seed 7 (`iso: 0.054 → 0.171 m`).
* **What Is Misconfigured (`approach_side` Action Space):**
  * In `planning/primitives.py:approach_face()`, `approach_side` (`"near" | "far" | "left" | "right"`) adds `0, +π, +π/2, -π/2` directly to the base yaw `j1` (`face = theta_t + offset`).
  * Because the arm is **rooted at `(0, 0)`** (`link1 = 0.30 m`), setting `j1 = theta_t + π` (`far`, Seed 3) or `j1 = theta_t + π/2` (`left`, Seed 7) aims `link1` **90°–180° away from the target box**, causing the arm to curl in empty space (`rew = 0.0`).
  * **Fix:** For a base-rooted planar arm, `approach_side` must control **wrap chirality and lead offset around `theta_t`**, not a `90°/180°` base rotation:
    * `"right"` / `"near"` (CCW wrap): park `link1` on the clockwise/right side of the target (`theta_t - lead_offset`) and curl CCW (`sign = +1`).
    * `"left"` / `"far"` (CW wrap): park `link1` on the counter-clockwise/left side of the target (`theta_t + lead_offset`) and curl CW (`sign = -1`).

### 2.4 Diffusion Policy & PPO (`scripts/train_grasp.py`, `scripts/train_diffusion.py`)
* **`train_diffusion.py` Fix Verified:** The `min_steps_per_epoch=50` floor increased training from `100` to `2,500` gradient steps, dropping DDPM MSE loss from `0.994 → 0.309` and achieving `FC=0.10` on unseen Seed 8 (`len=2`) and `31%` (`4/13`) recall on the demo seeds.
* **Bottleneck:** Behavior cloning is upper-bounded by its demonstrator (`models/grasp_ppo_500k.zip`), which was trained on the old 6-DOF `freejoint` env and only achieves `13/100` mostly 1-step reflex pinches on the new planar env. Retraining PPO on the planar env with `--contact-obs --step-penalty 2.0 --max-delta-q 1.0` is required before collecting a larger (~100+ episode) FC demo dataset.

---

## 3. Short-Term Iteration Strategy: 2 Canary Seeds (`Seed 2` & `Seed 7`)

Running all 10 seeds across 4 planners takes ~3 minutes and 40 episode renders. Instead, we will iterate exclusively on **2 canary seeds** (`--seeds 2,7`) that cover every failure mode in **~12 seconds** (0 API tokens via ledger cache):

### Canary 1: **Seed 2** — Isolated Enveloping Grasp (`r = 0.311 m, θ = -55°, iso = 0.474 m`)
* **Why Seed 2:** The red target box is completely isolated (`iso = 0.474 m`, no push needed) right in the middle of the enveloping workspace (`r = 0.311 m`), at a negative angle (`-55°`).
* **Current Failure:** `FC=False, caged=False, disp=0.654 m` (Phase 2→3 unfold whips `link2` across the box, knocking it from `r = 0.311 m` to `0.728 m`).
* **Target Pass Criterion:** `FC=True`, `caged=True`, `target_disp < 0.15 m` for both `vlm` and `heuristic`.

### Canary 2: **Seed 7** — Crowded Singulation + Left-Approach Grasp (`r = 0.355 m, θ = +109°, iso = 0.054 m`)
* **Why Seed 7:**
  1. Starts heavily crowded by the blue box (`obj1` at `iso = 0.054 m < 0.15 m`), testing `scripted_singulate` (`push 1(shove)` → `iso = 0.171 m >= 0.15 m`).
  2. VLM selects `approach_side = "left"`, directly testing the `approach_face` / `_cage_lead_and_sign` chirality mapping (`cw` wrap from the `+θ` / left side of the target) vs. Heuristic's `push=[], approach=near`.
* **Current Failure:** VLM singulates (`iso = 0.171 m, pre=True`), then `approach=left` points `link1` `+90°` away from the box (`rew=0.0, disp=0.585 m`).
* **Target Pass Criterion:** VLM achieves `preconditions_met=True`, `FC=True`, `caged=True`, `target_disp < 0.20 m` (diverging from and outperforming Heuristic).

### Fast Canary Command (~12s)
```bash
docker compose run --rm \
  -v $(pwd)/env:/workspace/env \
  -v $(pwd)/planning:/workspace/planning \
  -v $(pwd)/perception:/workspace/perception \
  -v $(pwd)/scripts:/workspace/scripts \
  --entrypoint python demo \
  scripts/compare_planners.py --seeds 2,7 --planners vlm,heuristic --gifs --html
```

---

## 4. Next Steps Execution Plan

1. **Step 1 — Fix `approach_face` and `_cage_lead_and_sign` Action Space Mapping (`planning/primitives.py`, `scripts/demo_vlm_grasp.py`):**
   * Map `approach_side` (`near`/`right` → CCW wrap from `theta_t - lead`; `left`/`far` → CW wrap from `theta_t + lead`) so `link1` always parks adjacent to the target (`|delta| ≈ 0.55–0.65 rad`) rather than `90°–180°` away.
2. **Step 2 — Fix `scripted_cage` Trajectory so It Never Sweeps Through the Target (`scripts/demo_vlm_grasp.py`):**
   * **Interpolate joint targets** smoothly across steps in each phase (like `move_tip`) instead of step-jumping on tick 0.
   * **Fold to the outside (`-s`) or extend along `face` before curling (`+s`):** Never fold `link2`/`link3` onto the target side (`+s`) and unfold across `theta_t`. Instead:
     1. Retract/arch `link2, link3` to the **outside** (`-0.6 * s, -0.6 * s`) while yawing `q1` to `face = theta_t - s * lead_offset` so `link2`/`link3` trail behind `link1` away from the target.
     2. Straighten along `face` (`q2 = 0.15 * s, q3 = 0.15 * s`) completely clear of the target box.
     3. Sweep `q2, q3` inward (`+s`) around the far side of the target box while gently closing `q1` toward `theta_t` to pinch the box between `link1` and `link3`.
3. **Step 3 — Verify on Canary Seeds (`Seeds 2 & 7`), Then Full `Seeds 0–9`:**
   * Iterate on `--seeds 2,7` until both `Seed 2` and `Seed 7` achieve `FC=True, caged=True` with low target displacement.
   * Re-run `--seeds 0-9 --gifs --html` and `pytest -v` to confirm high FC rate across all 10 seeds and 100% test pass rate.
4. **Step 4 — Retrain PPO & Diffusion on the Planar Environment:**
   * Run 100k–300k PPO training with `--contact-obs --step-penalty 2.0 --max-delta-q 1.0`.
   * Collect planar FC demos (`collect_ppo_demos.py`) and retrain `models/diffusion.pt` (`train_diffusion.py`).
