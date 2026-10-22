"""Motion planning: start configuration -> goal configuration -> trajectory.

Stage 1 (now): joint-space linear interpolation with timed waypoints.
Stage 2 (TODO): collision-aware planning (RRT) against the sim scene,
    then time-parameterization. The interface stays the same.

A trajectory is an (T, n_joints) array of joint targets the controller
can step through with the env's position actuators.
"""
import numpy as np


def plan(q_start, q_goal, n_steps=60):
    """Interpolate a joint-space trajectory from q_start to q_goal."""
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    alpha = np.linspace(0, 1, n_steps)[:, None]
    return (1 - alpha) * q_start + alpha * q_goal


def plan_to_pose(*args, **kwargs):
    """TODO: Cartesian goal -> joint trajectory (needs IK + collision check)."""
    raise NotImplementedError("Cartesian planning not yet implemented.")
