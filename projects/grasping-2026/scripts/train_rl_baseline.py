"""2014-style RL baseline — notes and stub.

The thesis trained RL from scratch (weeks, in 2014). The 2026 version of that
experiment:

Option A — Brax (fastest, recommended):
    * Reimplement WholeArmGraspEnv in Brax (same reward: link contacts + cage
      bonus). Brax trains PPO/SAC on GPU in minutes.
    * Free GPU hours: Kaggle notebooks give ~30 GPU hrs/week — plenty.

Option B — Stable-Baselines3 + Gymnasium:
    * Wrap env/whole_arm_grasp.py in a gymnasium.Env (obs = qpos/qvel/object
      poses; action = 3 joint targets). SB3 SAC trains on CPU overnight or on
      a Kaggle GPU in ~1-2 hrs for this small env.

The comparison that matters for the video/paper:
    2014 RL-from-scratch (weeks, 2D pybox2d)  vs  2026 VLM zero-shot (seconds)
    vs  2026 RL-from-scratch (minutes, Brax).

TODO: implement once the VLM loop (scripts/demo_vlm_grasp.py) is solid.
"""
print(__doc__)
