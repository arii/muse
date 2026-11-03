"""Gymnasium wrapper for WholeArmGraspEnv — local RL training, no API.

The base env is close to gym already (reset -> obs dict, step ->
obs/reward/done/info) but has no spaces and returns the old 4-tuple.
This wrapper adds gymnasium spaces and the 5-tuple, so
stable-baselines3 can train on it directly.

Observation (flat float32, 10 + 2*(n_objects-1) dims; 16 for the
default 4 objects):
    [q1, q2, q3, qvel1..3, target_x, target_y,
     target_half_x, target_half_y,
     distractor_dx, distractor_dy (x (n_objects-1), relative to target)]
Distractor offsets are relative to the target (translation-invariant)
and ordered by object index (excluding the target).

Action: Box(-1, 1, (3,)) — SB3's recommended symmetric range; the
wrapper scales by 2.6 to the joint limits the base env clips to.

Reward: the env's own _reward (0.5 * links touching + 2.0 *
force_closure). No shaping added here: the contact term already
shapes toward the grasp, and the force-closure bonus is the goal.

Terminated: force_closure achieved (success).
Truncated: max_steps reached without force closure.
Target: index 0 (red) by default — matches "grasp the red box".

Optional levers (all default OFF, so a running job and its
checkpoints stay loadable under the defaults):
- contact_obs: append [links_touching/3, enclosed] to the obs. The
  base obs forces the policy to infer contact geometrically from q +
  target pos; explicit contact features make the reward signal
  directly observable.
- step_penalty: subtracted from the reward every step. The base
  reward pays 0.5/link/step with no time cost, so resting against
  the box for 200 steps scores ~200 without grasping. A penalty
  only fixes the ranking if it exceeds the max per-step contact
  reward (1.5 for 3 links): at 0.5, a 2-link camp still nets
  (1.0-0.5)*200 = 100 vs ~2 for a 2-step force closure. Use
  1.5-2.0 (2.0 gives margin); at 2.0 a camp nets -200 while a
  2-step FC nets about -1, so FC wins and stalling is punished.
- max_delta_q: cap on the per-step joint-target change (radians,
  None = off). Random actions command full-range swings (up to
  5.2 rad in one step), which is what spikes the MuJoCo solver
  (QACC warnings). The cap limits the commanded target, not the
  contact physics, so it preserves grasp mechanics. Default off:
  a tight cap would also slow the 1-2 step pinches, so opt in
  per run (1.0 is a reasonable value) and keep it identical
  across train/eval/collect for a given policy.
- fail_penalty: thesis failure states (Anders 2014, S3.2/S4.2.2),
  train-only like success_bonus: object beyond arm's length
  (0.75 m) terminates at -penalty, and a non-goal timeout takes
  -penalty. The thesis used -100 with 35-step episodes
  (pass --max-steps 35 to match). Default 0 = fully off: no
  penalty, no extra termination, no new info keys.
- thesis_reward: replace the whole schedule with the exact thesis
  one (S3.2) for the SARSA port: goal 0; failure (out-of-reach,
  timeout) -100; >=2 contacts -0.25; 1 contact -0.5; else -1.
  Mutually exclusive with step_penalty/success_bonus/fail_penalty
  (raises). Default off.
- enclosure_bonus: added on enclosed steps, FC or not (train-only
  like success_bonus). Default 0 = off.
- scenes: layout names cycled round-robin across resets for
  mixed-scene training ('random' keeps the reset scene, the
  rest are env.scenes presets). Duplicates weight the mix.
  Default ("random",) = old behavior.
- precontact_disp_penalty: coef * red per-step motion taxed
  while links_touching==0 (train-only like success_bonus).
  Default 0 = off.
"""
import math

import gymnasium as gym
import numpy as np

from env.scenes import PRESETS, apply_scene
from env.whole_arm_grasp import WholeArmGraspEnv


class GraspGymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=None, target_index=0, max_steps=200,
                 n_objects=4, contact_obs=False, step_penalty=0.0,
                 max_delta_q=None, success_bonus=0.0,
                 fail_penalty=0.0, thesis_reward=False,
                 thesis_timeout=-100.0, enclosure_bonus=0.0,
                 scenes=("random",), precontact_disp_penalty=0.0):
        super().__init__()
        self.env = WholeArmGraspEnv(n_objects=n_objects, seed=seed)
        self.target_index = target_index
        self.max_steps = max_steps
        self.contact_obs = contact_obs
        self.step_penalty = step_penalty
        self.max_delta_q = max_delta_q
        self.fail_penalty = fail_penalty
        self.thesis_reward = thesis_reward
        # Timeout price under thesis_reward (default: exact thesis
        # -100). Lowered only for ablations: with sparse contact
        # the modal outcome is timeout, and -100 smears over every
        # trail and drowns the shaping gradient (see train_sarsa).
        self.thesis_timeout = float(thesis_timeout)
        if thesis_reward and (step_penalty or success_bonus
                              or fail_penalty or enclosure_bonus
                              or precontact_disp_penalty):
            raise ValueError("thesis_reward replaces the reward "
                             "schedule; it cannot combine with "
                             "step_penalty/success_bonus/fail_penalty/"
                             "enclosure_bonus/precontact_disp_penalty")
        # success_bonus: added on force-closure steps (train-only
        # shaping; eval/compare use 0 so reports stay comparable).
        # The base +2.0 FC bonus never compensates a forfeited 200-step
        # camp (~200 raw), so PPO unlearns discovered grasps; a +20
        # bonus makes FC dominate and preserves grasping behavior.
        self.success_bonus = success_bonus
        # enclosure_bonus: added on enclosed steps, FC or not
        # (train-only like success_bonus). Contact pays 0.5/link
        # but enclosure pays nothing, so PPO rests in contact
        # (enc=0) instead of wrapping; +1.0 at step_penalty 2.0
        # nets a 2-link enclosed hold exactly 0 (no farm),
        # strictly above a camp's -1/step.
        self.enclosure_bonus = enclosure_bonus
        # scenes: round-robin layout mix across resets (mixed-scene
        # training); duplicates weight the mix. Default ("random",)
        # preserves the old behavior exactly.
        self.scenes = tuple(scenes)
        for s in self.scenes:
            if s != "random" and s not in PRESETS:
                raise ValueError(f"unknown scene {s!r} (expected 'random' "
                                 f"or one of {PRESETS})")
        if any(s != "random" for s in self.scenes) and n_objects != 4:
            raise ValueError("scene presets need n_objects=4")
        self._resets = 0
        self.scene = self.scenes[0]
        self.scene_fallbacks = 0
        # precontact_disp_penalty: coef * red motion taxed on steps
        # with links_touching==0 (train-only like success_bonus).
        # Whiffs already end in timeout -100, but ~197 steps after
        # the fatal shove — too delayed for credit assignment.
        # This taxes the ejection itself, densely; post-contact
        # red motion (bulldoze-pinch wins) stays exempt.
        self.precontact_disp_penalty = precontact_disp_penalty
        self._prev_red = None
        self._steps = 0
        # Obs: q(3) + qvel(3) + target xy(2) + target half(2)
        #      + (n_objects-1) distractors x (dx, dy)
        #      + [links_touching/3, enclosed] if contact_obs.
        obs_dim = 10 + 2 * (n_objects - 1) + (2 if contact_obs else 0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _flat_obs(self, info=None):
        obs = self.env._obs()
        q = obs["qpos"].astype(np.float32)
        qvel = obs["qvel"].astype(np.float32)
        t = self.target_index
        tpos = obs["objects"][t][:2].astype(np.float32)
        thalf = np.array(self.env.objects[t]["half"][:2],
                         dtype=np.float32)
        # OBS CHANGE (2026-11-03): distractors sorted by angle around
        # the target (was: index order). Index order makes identical
        # scenes look different under permutation, which BC/diffusion
        # cannot learn from sparse scenes (diffusion hedged to the
        # mean). ALL checkpoints trained before this date are stale.
        rel = []
        for o in self.env.objects:
            if o["index"] == t:
                continue
            pos = obs["objects"][o["index"]][:2].astype(np.float32)
            d = pos - tpos
            rel.append((math.atan2(float(d[1]), float(d[0])), d))
        distractors = [d for _, d in sorted(rel)]
        parts = [q, qvel, tpos, thalf] + distractors
        if self.contact_obs:
            if info is None:
                parts.append(np.zeros(2, dtype=np.float32))
            else:
                parts.append(np.array(
                    [info["links_touching"] / 3.0,
                     1.0 if info["enclosed"] else 0.0], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        # Base reset() clears target_index; set it after.
        self.env.target_index = self.target_index
        self.scene = self.scenes[self._resets % len(self.scenes)]
        self._resets += 1
        if self.scene != "random":
            # Presets are feasibility-checked (adjacent has no
            # neighbor spot on some resets): re-draw up to 3
            # times, else keep a random scene and count it.
            # _layout raises before placing anything, so a
            # failed attempt leaves the scene untouched.
            for _ in range(3):
                try:
                    apply_scene(self.env, self.scene)
                    break
                except ValueError:
                    self.env.reset()
                    self.env.target_index = self.target_index
            else:
                self.scene = "random"
                self.scene_fallbacks += 1
        self._steps = 0
        if self.precontact_disp_penalty:
            self._prev_red = np.array(self.env._target_xy(), dtype=float)
        return self._flat_obs(), {"scene": self.scene}

    def step(self, action):
        # Policy outputs [-1, 1]; scale to the joint limits.
        target = np.asarray(action, dtype=np.float64) * 2.6
        if self.max_delta_q is not None:
            prev_q = self.env._obs()["qpos"].astype(np.float64)
            target = np.clip(target, prev_q - self.max_delta_q,
                             prev_q + self.max_delta_q)
        _, reward, _, info = self.env.step(target)
        self._steps += 1
        terminated = bool(info["force_closure"])
        truncated = self._steps >= self.max_steps
        shaped = float(reward) - self.step_penalty
        if terminated:
            shaped += self.success_bonus
        if info["enclosed"]:
            shaped += self.enclosure_bonus
        if self.precontact_disp_penalty:
            red = np.array(self.env._target_xy(), dtype=float)
            if not info["links_touching"]:
                shaped -= (self.precontact_disp_penalty
                           * float(np.linalg.norm(red - self._prev_red)))
            self._prev_red = red
        if self.fail_penalty:
            # Thesis failure states (Anders 2014, S3.2/S4.2.2): timeout
            # and object-beyond-arm's-length end the episode at -100.
            # (The third thesis failure, arm self-collision, is
            # dual-arm-only, so it has no single-arm counterpart.)
            tpos = self.env._obs()["objects"][self.target_index][:2]
            if float(np.linalg.norm(tpos)) > 0.75:  # 0.30+0.25+0.20
                terminated = True
                shaped -= self.fail_penalty
                info["failed_out_of_reach"] = True
            elif truncated and not terminated:
                shaped -= self.fail_penalty
                info["failed_timeout"] = True
        if self.thesis_reward:
            # Exact thesis schedule (S3.2): goal 0; failure -100;
            # contacts -0.25/-0.5; else -1. Contact locations are the
            # touching links; out-of-reach reuses the 0.75 m rule.
            tpos = self.env._obs()["objects"][self.target_index][:2]
            oor = float(np.linalg.norm(tpos)) > 0.75
            if terminated:
                shaped = 0.0
            elif oor or truncated:
                terminated = oor
                truncated = (not oor) and truncated
                shaped = (-100.0 if oor else self.thesis_timeout)
                info["failed_out_of_reach" if oor
                     else "failed_timeout"] = True
            else:
                n = info["links_touching"]
                shaped = -0.25 if n >= 2 else (-0.5 if n == 1 else -1.0)
        return (self._flat_obs(info), shaped, terminated, truncated, info)

    def render(self):
        return self.env.render()
