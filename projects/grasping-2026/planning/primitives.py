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


# Push effort (a planner's semantic choice) -> push distance in meters.
# nudge: barely inside crowding range; clear: the normal push; shove:
# jammed against the target, needs one long push instead of several.
EFFORT_DISTANCES = {"nudge": 0.06, "clear": 0.15, "shove": 0.25}


def approach_face(target_xy, side="near"):
    """Cage yaw for an approach side (planner's semantic choice).

    The cage yaws toward the target, then reaches and curls. `side`
    biases that entry direction: near = toward the target (the
    historical behavior); far = opposite; left/right = 90° around
    the target from the arm's side, counterclockwise/clockwise as
    seen in the top-down image (image up = +y, right = +x, so image
    CCW is world CCW). This is an entry-direction bias on an
    open-loop cage, not a tip-placement guarantee — the reach/curl
    phases are fixed and the arm is rooted at the origin.
    """
    base = float(np.arctan2(target_xy[1], target_xy[0]))
    offsets = {"near": 0.0, "far": np.pi,
               "left": np.pi / 2, "right": -np.pi / 2}
    if side not in offsets:
        raise ValueError(f"unknown approach_side {side!r} "
                         f"(expected one of {sorted(offsets)})")
    face = base + offsets[side]
    # Wrap to [-pi, pi]: the env clips joint targets to +-2.6, and an
    # unwrapped face+pi (up to 2pi) would clip to a wrong yaw.
    return float((face + np.pi) % (2 * np.pi) - np.pi)


def ik_tip_oriented(target_xy, link_angle):
    """Tip IK with the last link at a fixed absolute angle.

    Why this exists (2026-10-28): plain tip IK (ik_dls, seeded from
    the current pose) picks whatever branch is nearest, and for a
    push that branch usually puts link3 *across* the object: the
    wrist sits east/north of the tip, so the link3 segment overlaps
    the box, and the solver spins the box south/west instead of
    pushing it along theta (seed 7 push: box moved -0.10 m in y,
    not +x). A push needs link3 *behind* the tip, pointing along
    theta: wrist = tip - 0.20 * (cos theta, sin theta), then a
    2-link IK for the wrist. Returns None when that wrist is
    unreachable (distance outside [0.05, 0.55] or a joint limit) —
    a tip push from that start is geometrically impossible, and
    the caller must fall back (or the singulate revert must catch
    it), not silently push with a crossing link.
    """
    l0, l1, l2 = LINK_LENGTHS
    target = np.asarray(target_xy, dtype=float)
    wrist = target - l2 * np.array([np.cos(link_angle),
                                    np.sin(link_angle)])
    d = float(np.linalg.norm(wrist))
    if d > l0 + l1 or d < abs(l0 - l1):
        return None
    cos_q1 = (d * d - l0 * l0 - l1 * l1) / (2 * l0 * l1)
    if abs(cos_q1) > 1.0:
        return None
    for sign in (+1, -1):
        q1 = sign * float(np.arccos(np.clip(cos_q1, -1, 1)))
        q0 = float(np.arctan2(wrist[1], wrist[0]) - np.arctan2(
            l1 * np.sin(q1), l0 + l1 * np.cos(q1)))
        q2 = float(link_angle - (q0 + q1))
        # Wrap q2 to [-pi, pi] for the limit check.
        q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
        q = np.array([q0, q1, q2])
        if np.all(np.abs(q) <= JOINT_LIMIT):
            return q
    return None


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

    The tip path is solved with link3 held at angle theta (see
    ik_tip_oriented): link3 behind the tip, pointing along the push,
    so contact is tip-on-face, not link3-across-the-box. When that
    wrist is unreachable the code falls back to plain ik_dls — the
    old crossing behaviour — and singulate's revert logic is the
    guardrail for those pushes.
    """
    direction = np.array([np.cos(theta), np.sin(theta)])
    start = np.asarray(start_xy, dtype=float)
    q_oriented = ik_tip_oriented(start, theta)
    if q_oriented is not None:
        # Approach to the oriented start pose (joint interpolation,
        # like move_tip, but to the oriented q, not the DLS q).
        q_cur = _qpos(env)
        for a in np.linspace(0.0, 1.0, 25):
            _, _, _, info = env.step((1 - a) * q_cur + a * q_oriented)
            frames.append(env.render())
            if frame_info is not None:
                frame_info.append(frame_status(info))
        for _ in range(8):
            _, _, _, info = env.step(q_oriented)
            frames.append(env.render())
            if frame_info is not None:
                frame_info.append(frame_status(info))
        q = q_oriented
    else:
        move_tip(env, start_xy, frames, frame_info=frame_info)
        q = _qpos(env)
    for i in range(1, n_waypoints + 1):
        wp = start + direction * (dist * i / n_waypoints)
        q_wp = ik_tip_oriented(wp, theta)
        q = q_wp if q_wp is not None else ik_dls(wp, q)
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
