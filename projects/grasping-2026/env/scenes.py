"""Hand-designed adversarial clutter scenes (DexSinGrasp SNC style).

Random-seed scenes exercise the average case; these presets pack
distractors at fingertip-scale gaps (< 25 mm) around the target so
singulation is NECESSARY. The 8 mm presets (surround/row/gate) beat
the current push primitive: execution bulldozes the target (0.06 m
nudge moves red 0.40+ m — arm motion through the cluster, not the
push), the keep-rule correctly reverts, isolation stays flat, and
the ladder fails all 4 — order/effort-independent. The fix is
collision-aware push approach; they measure that future work.
The 10 mm single-neighbor `adjacent` preset was the VLM-win regime
until the side ladder grasped it push-free; the `sandwich` preset
(two-sided 15 mm clutter) is the current frontier: the push-free
ladder fails all 4 rungs on seed 9, push-then-ladder wins. The
`decoy` preset adds a loose distractor toward the arm base: blind
pushing shoves it into the approach corridor (harm), so only
*selective* pushing wins — the VLM-selection regime neither
sandwich (VLM identical to autopush-shove) nor adjacent provides.

Poses derive from the reset box sizes, so each (seed, preset) scene
is deterministic. Yaws are 0 (axis-aligned). All spots clear the
reset-pose arm (straight along +x).
"""
import math

PRESETS = ("surround", "row", "gate", "adjacent", "sandwich", "decoy")

_GAP = 0.008  # 8 mm corner gaps: singulation-necessary clutter


def _half(env, i):
    sx, sy, _ = env.objects[i]["half"]
    return sx, sy


def _layout(env, preset):
    """Return {index: (x, y)} placements for a preset."""
    if env.n_objects != 4:
        raise ValueError(f"scene presets need 4 objects, "
                         f"env has {env.n_objects}")
    g = _GAP
    if preset == "surround":
        # Target with N/E/W neighbors at gap (south left open).
        tx, ty = 0.32, 0.14
        sx0, sy0 = _half(env, 0)
        _, sy1 = _half(env, 1)
        sx2, _ = _half(env, 2)
        sx3, _ = _half(env, 3)
        return {0: (tx, ty),
                1: (tx, ty + sy0 + sy1 + g),
                2: (tx + sx0 + sx2 + g, ty),
                3: (tx - sx0 - sx3 - g, ty)}
    if preset == "row":
        # West-to-east row, target second (boxed in along the row).
        y = 0.16
        x3 = 0.20
        sx3, _ = _half(env, 3)
        sx0, _ = _half(env, 0)
        sx1, _ = _half(env, 1)
        sx2, _ = _half(env, 2)
        x0 = x3 + sx3 + sx0 + g
        x1 = x0 + sx0 + sx1 + g
        x2 = x1 + sx1 + sx2 + g
        return {3: (x3, y), 0: (x0, y), 1: (x1, y), 2: (x2, y)}
    if preset == "adjacent":
        # Medium clutter: blue 10 mm east of red's reset spot (others
        # untouched), unless another box is in the way (seed 2: green
        # sits east) — then 10 mm west. Raises if neither side fits.
        t = env._obj_xy(0)
        sx0, _ = _half(env, 0)
        sx1, _ = _half(env, 1)
        rad1 = math.hypot(*_half(env, 1))
        cands = [(t[0] + sx0 + sx1 + 0.010, t[1]),
                 (t[0] - sx0 - sx1 - 0.010, t[1])]
        for bx, by in cands:
            clear = True
            for j in (2, 3):
                q = env._obj_xy(j)
                if math.hypot(bx - q[0], by - q[1]) < \
                        rad1 + math.hypot(*_half(env, j)) + 0.005:
                    clear = False
            if clear:
                return {1: (bx, by)}
        raise ValueError("adjacent preset: no overlap-free neighbor spot")
    if preset == "sandwich":
        # Two-sided medium clutter: blue 15 mm east + yellow 15 mm
        # west of red's reset spot (green untouched). Each neighbor
        # falls back to the other side when blocked, else stays at
        # reset (seed 2 degrades: green sits east). Seed 9 forms the
        # full sandwich and is the VLM-win regime.
        t = env._obj_xy(0)
        sx0, _ = _half(env, 0)
        # Red excluded: the 15 mm face gap is exact by construction
        # (circumradii legitimately overlap at that gap).
        occupied = [(env._obj_xy(2), _half(env, 2))]

        def fits(xy, i):
            ri = math.hypot(*_half(env, i))
            return all(math.hypot(xy[0] - q[0], xy[1] - q[1])
                       >= ri + math.hypot(*h) + 0.005
                       for q, h in occupied)

        out = {}
        for i, dx in ((1, 1.0), (3, -1.0)):
            sxi, _ = _half(env, i)
            for sign in (dx, -dx):
                xy = (t[0] + sign * (sx0 + sxi + 0.015), t[1])
                if fits(xy, i):
                    out[i] = xy
                    occupied.append((xy, _half(env, i)))
                    break
        return out
    if preset == "decoy":
        # Asymmetric clutter: sandwich E/W blockers (15 mm, both must
        # move) plus green as a LOOSE decoy 0.22 m toward the arm base
        # (face gap ~120 mm, so the radial shove moves only the decoy
        # — no target contact, no revert — and lands it in the
        # approach corridor; at 0.16 m the push jostled red and the
        # revert rule erased the harm). Blind
        # pushing clears the blockers nearest-first, then ruins the
        # scene with the decoy push; selective pushing (blockers
        # only) should win. Falls back to the mirror spot (away from
        # base) when blocked, else green stays at reset (degraded).
        t = env._obj_xy(0)
        sx0, _ = _half(env, 0)
        green_reset = (env._obj_xy(2), _half(env, 2))
        occupied = []

        def fits(xy, i, others):
            ri = math.hypot(*_half(env, i))
            return all(math.hypot(xy[0] - q[0], xy[1] - q[1])
                       >= ri + math.hypot(*h) + 0.005
                       for q, h in others)

        out = {}
        for i, dx in ((1, 1.0), (3, -1.0)):
            sxi, _ = _half(env, i)
            for sign in (dx, -dx):
                xy = (t[0] + sign * (sx0 + sxi + 0.015), t[1])
                if fits(xy, i, [green_reset] + occupied):
                    out[i] = xy
                    occupied.append((xy, _half(env, i)))
                    break
        tnorm = math.hypot(t[0], t[1]) or 1.0
        toward = (-t[0] / tnorm, -t[1] / tnorm)
        r2 = math.hypot(*_half(env, 2))
        for sign in (1.0, -1.0):
            xy = (t[0] + sign * toward[0] * 0.22,
                  t[1] + sign * toward[1] * 0.22)
            # Arm-clear (same formula as the preset test): the
            # toward-base spot can land on the reset arm (+x) or the
            # base disc when red sits near the axis or at min radius.
            cx = min(max(xy[0], 0.0), 0.75)
            arm_d = math.hypot(xy[0] - cx, xy[1]) - r2 - 0.035
            if arm_d >= 0.0 and fits(xy, 2, occupied):
                out[2] = xy
                break
        return out
    if preset == "gate":
        # W neighbor plus a two-deep clutter chain east of the target.
        tx, ty = 0.30, 0.14
        sx0, _ = _half(env, 0)
        sx1, _ = _half(env, 1)
        sx2, _ = _half(env, 2)
        sx3, _ = _half(env, 3)
        x1 = tx + sx0 + sx1 + g
        x3 = x1 + sx1 + sx3 + g
        return {0: (tx, ty),
                1: (x1, ty),
                2: (tx - sx0 - sx2 - g, ty),
                3: (x3, ty)}
    raise ValueError(f"unknown scene preset {preset!r} "
                     f"(expected one of {PRESETS})")


def apply_scene(env, preset):
    """Place boxes per a preset; 'random' keeps the reset scene."""
    if preset == "random":
        return
    for i, (x, y) in _layout(env, preset).items():
        env.set_object_pose(i, (x, y), yaw=0.0)
