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
"""
import gymnasium as gym
import numpy as np

from env.whole_arm_grasp import WholeArmGraspEnv


class GraspGymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=None, target_index=0, max_steps=200,
                 n_objects=4):
        super().__init__()
        self.env = WholeArmGraspEnv(n_objects=n_objects, seed=seed)
        self.target_index = target_index
        self.max_steps = max_steps
        self._steps = 0
        # Obs: q(3) + qvel(3) + target xy(2) + target half(2)
        #      + (n_objects-1) distractors x (dx, dy).
        obs_dim = 10 + 2 * (n_objects - 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _flat_obs(self):
        obs = self.env._obs()
        q = obs["qpos"].astype(np.float32)
        qvel = obs["qvel"].astype(np.float32)
        t = self.target_index
        tpos = obs["objects"][t][:2].astype(np.float32)
        thalf = np.array(self.env.objects[t]["half"][:2],
                         dtype=np.float32)
        distractors = []
        for o in self.env.objects:
            if o["index"] == t:
                continue
            pos = obs["objects"][o["index"]][:2].astype(np.float32)
            distractors.append(pos - tpos)
        return np.concatenate(
            [q, qvel, tpos, thalf] + distractors).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        # Base reset() clears target_index; set it after.
        self.env.target_index = self.target_index
        self._steps = 0
        return self._flat_obs(), {}

    def step(self, action):
        # Policy outputs [-1, 1]; scale to the joint limits.
        _, reward, _, info = self.env.step(
            np.asarray(action, dtype=np.float64) * 2.6)
        self._steps += 1
        terminated = bool(info["force_closure"])
        truncated = self._steps >= self.max_steps
        return self._flat_obs(), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()
