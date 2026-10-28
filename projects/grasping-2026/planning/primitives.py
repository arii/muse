"""Geometric action primitives for the 3-link planar arm.

Traditional manipulation framework: the action space is Cartesian, not
object-index-based. The VLM (or a planner) decides *what* to do in semantic
terms; a resolver turns that into these primitives, which do the geometry.

Primitives:
    move_tip(env, target_xy, frames)  - IK move of the tip to a point
    push(env, start_xy, theta, dist, frames) - Cartesian push along theta
    envelop(env, target_xy, frames)   - whole-arm grasp skill (trajectory,
                                        not a pose — see below)

The enveloping grasp is deliberately a *named skill*, not a (x, y, theta)
tuple: wrapping 3 links around an object is a trajectory (approach, wrap,
close), not a single end-effector pose.

Control: env.step(q) drives position actuators toward q with 20 physics
substeps per call, so motions are quasi-static and contacts resolve.
"""
import numpy as np

LINK_LENGTHS = (0.30, 0.25, 0.20)
JOINT_LIMIT = 2.6


# ---------- kinematics ----------

def fk(q):
    """Tip (x, y) for joint angles q = [q0, q1, q2]."""
    l0, l1, l2 = LINK_LENGTHS
    a0 = q[0]
    a1 = q[0] + q[1]
    a2 = q[0] + q[1] + q[2]
    return np.array([
        l0 * np.cos(a0) + l1 * np.cos(a1) + l2 * np.cos(a2),
        l0 * np.sin(a0) + l1 * np.sin(a1) + l2 * np.sin(a2),
    ])


def jacobian(q):
    """2x3 geometric Jacobian of the tip position."""
    l0, l1, l2 = LINK_LENGTHS
    a0 = q[0]
    a1 = q[0] + q[1]
    a2 = q[0] + q[1] + q[2]
    return np.array([
        [-l0 * np.sin(a0) - l1 * np.sin(a1) - l2 * np.sin(a2),
         -l1 * np.sin(a1) - l2 * np.sin(a2),
         -l2 * np.sin(a2)],
        [l0 * np.cos(a0) + l1 * np.cos(a1) + l2 * np.cos(a2),
         l1 * np.cos(a1) + l2 * np.cos(a2),
         l2 * np.cos(a2)],
    ])


def ik_dls(target_xy, q_seed, lam=0.1, max_iter=100, tol=1e-4,
           max_step=0.2):
    """Damped least-squares IK. Seeded from q_seed for continuity —
    successive calls with nearby targets stay on the same branch."""
    q = np.array(q_seed, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    for _ in range(max_iter):
        err = target - fk(q)
        if np.linalg.norm(err) < tol:
            break
        J = jacobian(q)
        A = J @ J.T + (lam ** 2) * np.eye(2)
        dq = J.T @ np.linalg.solve(A, err)
        n = np.linalg.norm(dq)
        if n > max_step:
            dq = dq * (max_step / n)
        q = np.clip(q + dq, -JOINT_LIMIT, JOINT_LIMIT)
    return q


# ---------- primitives ----------

def _qpos(env):
    return env._obs()["qpos"]


def frame_status(info):
    """Compact per-frame grasp status for GIF overlays and traces.

    Shape contract (consumed by scripts/make_video.py): links =
    distinct arm links touching the target, enclosed/caged/fc booleans.
    """
    return {"links": int(info["links_touching"]),
            "enclosed": bool(info["enclosed"]),
            "caged": bool(info["caged"]),
            "fc": bool(info["force_closure"])}


def move_tip(env, target_xy, frames, n_interp=25, settle=8,
             frame_info=None):
    """Move the tip to target_xy via IK + joint interpolation, then
    settle (extra steps at the final q so the actuators catch up)."""
    q_cur = _qpos(env)
    q_tgt = ik_dls(target_xy, q_cur)
    for a in np.linspace(0.0, 1.0, n_interp):
        _, _, _, info = env.step((1 - a) * q_cur + a * q_tgt)
        frames.append(env.render())
        if frame_info is not None:
            frame_info.append(frame_status(info))
    for _ in range(settle):
        _, _, _, info = env.step(q_tgt)
        frames.append(env.render())
        if frame_info is not None:
            frame_info.append(frame_status(info))
    return q_tgt


def push(env, start_xy, theta, dist, frames, n_waypoints=12, settle=3,
         frame_info=None):
    """Cartesian push: approach start_xy, then drive the tip along
    direction theta for distance dist. Waypoints are close together and
    each is IK-solved from the previous q, so the tip tracks a straight
    line and contact forces push the object quasi-statically. A few
    settle steps per waypoint keep the (soft) position actuators from
    lagging behind the commanded line.

    start_xy should be at the object's near edge (not its center) so the
    tip makes contact instead of interpenetrating.
    """
    move_tip(env, start_xy, frames, frame_info=frame_info)
    direction = np.array([np.cos(theta), np.sin(theta)])
    start = np.asarray(start_xy, dtype=float)
    q = _qpos(env)
    for i in range(1, n_waypoints + 1):
        wp = start + direction * (dist * i / n_waypoints)
        q = ik_dls(wp, q)
        for _ in range(settle):
            _, _, _, info = env.step(q)
            frames.append(env.render())
            if frame_info is not None:
                frame_info.append(frame_status(info))
    return q


def envelop(env, target_xy, frames):
    """Whole-arm enveloping grasp skill. Parameterized by target position;
    executes the yaw -> reach -> curl trajectory. Returns the step info
    dict (links_touching, caged, isolation)."""
    q_cur = _qpos(env)
    # Phase 1: yaw the base toward the target (arm curled = small swept disc)
    theta_t = float(np.arctan2(target_xy[1], target_xy[0]))
    q_yaw = np.array([theta_t, q_cur[1], q_cur[2]])
    for a in np.linspace(0.0, 1.0, 20):
        env.step((1 - a) * q_cur + a * q_yaw)
        frames.append(env.render())
    # Phase 2: reach — extend the tip toward the target
    q_reach = ik_dls(target_xy, q_yaw)
    for a in np.linspace(0.0, 1.0, 25):
        env.step((1 - a) * q_yaw + a * q_reach)
        frames.append(env.render())
    # Phase 3: curl — wrap the links around the target
    q_curl = q_reach + np.array([0.0, 0.5, 0.7])
    q_curl = np.clip(q_curl, -JOINT_LIMIT, JOINT_LIMIT)
    info = {}
    for a in np.linspace(0.0, 1.0, 30):
        _, _, _, info = env.step((1 - a) * q_reach + a * q_curl)
        frames.append(env.render())
    return info
