"""Planar whole-arm grasping environment (MuJoCo).

A 3-link arm sweeps a horizontal plane above a table. Each episode spawns
N randomized box objects within reach. The task: grasp a target object
using the whole arm.

GOAL CRITERION: force closure. The grasp passes iff the contact
wrenches positively span R^3 (the grasp resists arbitrary wrenches) —
the true grasping metric. Geometric caging (target inside the arm
polygon) is reported separately: it traps the object but does not by
itself guarantee the grasp holds. Enveloping grasps — the arm wrapping
fully around the object to get opposing contacts — are what meet the
force-closure criterion.

Reward:
    r = 0.5 * (# distinct arm links touching target)
        + 2.0 * force_closure

info also carries "isolation": min xy center-distance from the target
object to any other object — the singulation objective (push clutter
away before grasping).
"""
import pathlib

import mujoco
import numpy as np

XML_PATH = pathlib.Path(__file__).parent / "arm.xml"

LINK_GEOMS = ("link1_geom", "link2_geom", "link3_geom")
JOINT_NAMES = ("j1", "j2", "j3")

PALETTE = {
    "red": (0.85, 0.2, 0.2, 1),
    "blue": (0.2, 0.4, 0.85, 1),
    "green": (0.2, 0.7, 0.3, 1),
    "yellow": (0.9, 0.8, 0.2, 1),
    "purple": (0.6, 0.3, 0.8, 1),
}


class WholeArmGraspEnv:
    def __init__(self, n_objects=4, seed=None):
        self.n_objects = n_objects
        self.rng = np.random.default_rng(seed)
        self.target_index = None
        self.model = None
        self.data = None
        self.renderer = None

    # ---------- setup ----------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spec = mujoco.MjSpec.from_file(str(XML_PATH))
        self.objects = []
        colors = list(PALETTE)
        for i in range(self.n_objects):
            r = self.rng.uniform(0.28, 0.55)
            th = self.rng.uniform(-np.pi, np.pi)
            x, y = r * np.cos(th), r * np.sin(th)
            sx = self.rng.uniform(0.03, 0.06)
            sy = self.rng.uniform(0.03, 0.06)
            sz = 0.06
            cname = colors[i % len(colors)]
            body = spec.worldbody.add_body(name=f"obj{i}", pos=[x, y, sz])
            # PLANAR constraint (2026-10-28 fix): boxes used to be
            # freejoint 6-DOF bodies in a supposedly planar task.
            # The arm contacts them at z=0.09, above their center
            # (0.06), so a push tips them; a hard PPO swing launches
            # them — seed 12's red box ended at (-20.0, 1.6), seed
            # 33's at z=0.104 wedged on top of a link with a tipped
            # quaternion, and force closure (a planar wrench test)
            # then passed on a non-planar jam. Two slide joints
            # (x, y) + one hinge (yaw about z) keep every box flat
            # on the table, at fixed z, for the whole episode: it
            # can translate and spin in-plane, never tip, lift, or
            # roll off the table in z. qpos per object is now
            # [slide_x, slide_y, yaw] displacements from the spawn
            # pose below — world xy comes from body_xpos, not qpos.
            body.add_joint(name=f"obj{i}_x",
                           type=mujoco.mjtJoint.mjJNT_SLIDE,
                           axis=[1, 0, 0])
            body.add_joint(name=f"obj{i}_y",
                           type=mujoco.mjtJoint.mjJNT_SLIDE,
                           axis=[0, 1, 0])
            body.add_joint(name=f"obj{i}_yaw",
                           type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[0, 0, 1])
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, sz],
                rgba=PALETTE[cname],
                friction=[0.8, 0.005, 0.0001],
                name=f"obj{i}_geom",
            )
            self.objects.append({"index": i, "name": f"obj{i}",
                                 "color": cname, "half": (sx, sy, sz),
                                 "spawn_xy": (x, y)})
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self._arm_qpos = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            for jn in JOINT_NAMES
        ]
        self._obj_qposadr = {
            o["index"]: self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o['name']}_x")]
            for o in self.objects
        }
        self._obj_body_id = {
            o["index"]: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, o["name"])
            for o in self.objects
        }
        mujoco.mj_forward(self.model, self.data)
        self.target_index = None
        return self._obs()

    # ---------- interaction ----------
    def step(self, action, substeps=20):
        """action: target joint angles (radians) for j1..j3."""
        self.data.ctrl[:] = np.clip(action, -2.6, 2.6)
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        reward, info = self._reward()
        return self._obs(), reward, False, info

    def render(self):
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render()

    def snapshot(self):
        """Opaque physics snapshot, for restore()."""
        return (self.data.qpos.copy(), self.data.qvel.copy())

    def restore(self, snap):
        """Restore a snapshot() — reverts physics to the saved state."""
        self.data.qpos[:], self.data.qvel[:] = snap
        mujoco.mj_forward(self.model, self.data)

    # ---------- helpers ----------
    def _obj_xy(self, index):
        """World xy center of an object (body_xpos, not qpos: with
        slide/yaw joints qpos holds displacements from spawn)."""
        return self.data.xpos[self._obj_body_id[index]][:2].copy()

    def set_object_pose(self, index, xy, yaw=0.0):
        """Place an object (tests / fixtures only): slide displacements
        are xy minus the spawn pose, yaw is the hinge angle."""
        adr = self._obj_qposadr[index]
        sx, sy = self.objects[index]["spawn_xy"]
        self.data.qpos[adr] = float(xy[0]) - sx
        self.data.qpos[adr + 1] = float(xy[1]) - sy
        self.data.qpos[adr + 2] = float(yaw)
        mujoco.mj_forward(self.model, self.data)

    def _obs(self):
        qpos = self.data.qpos[self._arm_qpos].copy()
        qvel = self.data.qvel[:3].copy()
        obj_poses = {}
        for o in self.objects:
            # Synthesize the old 7-vector pose [x, y, z, quat] so
            # callers that only read [:2] (all of them) keep working,
            # and the quaternion is always a pure yaw rotation.
            adr = self._obj_qposadr[o["index"]]
            yaw = float(self.data.qpos[adr + 2])
            xy = self._obj_xy(o["index"])
            obj_poses[o["index"]] = np.array([
                xy[0], xy[1], o["half"][2],
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        return {"qpos": qpos, "qvel": qvel, "objects": obj_poses,
                "target": self.target_index}

    def target_isolation(self):
        """Min xy center-distance from the target object to any other
        object (None when no target is set). Grows as the target gets
        isolated from clutter — the singulation objective."""
        if self.target_index is None:
            return None
        xy = {i: self._obj_xy(i) for i in self._obj_body_id}
        t = xy[self.target_index]
        others = [float(np.linalg.norm(xy[i] - t))
                  for i in xy if i != self.target_index]
        # No distractors: vacuously isolated.
        return min(others) if others else float("inf")

    def _geom_name(self, gid):
        n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        return n.decode() if isinstance(n, bytes) else n

    def _arm_joint_xy(self):
        """Arm polyline vertices (base, joint1, joint2, tip) in world xy.

        Same FK convention as planning/primitives.fk: cumulative angles,
        link lengths 0.30 / 0.25 / 0.20.
        """
        q = self.data.qpos[self._arm_qpos]
        a0 = q[0]
        a1 = q[0] + q[1]
        a2 = q[0] + q[1] + q[2]
        base = np.array([0.0, 0.0])
        j1 = base + 0.30 * np.array([np.cos(a0), np.sin(a0)])
        j2 = j1 + 0.25 * np.array([np.cos(a1), np.sin(a1)])
        tip = j2 + 0.20 * np.array([np.cos(a2), np.sin(a2)])
        return [base, j1, j2, tip]

    def _target_xy(self):
        return self._obj_xy(self.target_index)

    @staticmethod
    def _point_in_polygon(point, poly):
        """Ray casting: True if point is inside polygon poly."""
        x, y = point
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x < xinters:
                    inside = not inside
        return inside

    def _contact_wrenches(self):
        """Planar contact wrenches on the target, one per friction-cone edge.

        Each contact contributes 2 wrenches (the friction cone's edges).
        Forces are oriented as forces ON the target; moments are about
        the target center. Friction mu = min(target, link) tangential
        friction (conservative). Returns a list of (fx, fy, mz) arrays.
        """
        if self.target_index is None:
            return []
        target_geom = f"obj{self.target_index}_geom"
        center = self._obj_xy(self.target_index)
        wrenches = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = self._geom_name(c.geom1), self._geom_name(c.geom2)
            if target_geom not in (g1, g2):
                continue
            link = g2 if g1 == target_geom else g1
            if link not in LINK_GEOMS:
                continue
            # contact.frame is flat [normal, tangent, tangent2]; normal
            # points geom1 -> geom2. Orient it as force ON the target.
            normal = np.asarray(c.frame)[0:3][:2].astype(float)
            if g1 == target_geom:
                normal = -normal
            nn = np.linalg.norm(normal)
            if nn < 1e-9:
                continue
            normal = normal / nn
            tangent = np.array([-normal[1], normal[0]])
            mu = min(
                float(self.model.geom_friction[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      target_geom)][0]),
                float(self.model.geom_friction[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      link)][0]))
            alpha = np.arctan(mu)
            for s in (+1, -1):
                f = normal * np.cos(alpha) + tangent * (s * np.sin(alpha))
                r = np.asarray(c.pos)[:2] - center
                mz = r[0] * f[1] - r[1] * f[0]
                wrenches.append(np.array([f[0], f[1], mz]))
        return wrenches

    @staticmethod
    def _force_closure(wrenches):
        """True if contact wrenches positively span R^3 (force closure).

        Equivalent to the origin lying strictly inside the convex hull
        of the wrenches. No-scipy test: a nonzero separating direction
        exists iff some pair's cross product (or its negative) has all
        wrenches on one side — extreme rays of the separating cone are
        pair intersections. If no such direction exists, the origin is
        strictly inside and the grasp is force closure. Needs >=4
        wrenches (2 contacts x 2 cone edges) to span R^3.
        """
        W = np.array(wrenches, dtype=float)
        n = len(W)
        if n < 4:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                u = np.cross(W[i], W[j])
                norm = np.linalg.norm(u)
                if norm < 1e-9:
                    continue
                u = u / norm
                for s in (u, -u):
                    if np.min(W @ s) >= -1e-9:
                        return False
        return True

    def _reward(self):
        touching = set()
        target_geom = (f"obj{self.target_index}_geom"
                       if self.target_index is not None else None)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = self._geom_name(c.geom1), self._geom_name(c.geom2)
            if target_geom and target_geom in (g1, g2):
                other = g2 if g1 == target_geom else g1
                if other in LINK_GEOMS:
                    touching.add(other)
        n = len(touching)
        # Enclosure: target center inside the arm polygon. Contact alone
        # (e.g. two links on the same face) is a poke, not a cage.
        enclosed = (self.target_index is not None and
                    self._point_in_polygon(self._target_xy(),
                                           self._arm_joint_xy()))
        caged = n >= 2 and enclosed
        # Force closure: the GOAL criterion. Contact wrenches
        # positively spanning R^3 means the grasp resists arbitrary
        # wrenches. Distinct from caging: a geometric cage traps the
        # object without necessarily being force closure, and a
        # force-closure pinch holds without caging. The reward bonus
        # follows force closure — it is what passing means.
        force_closure = self._force_closure(self._contact_wrenches())
        return 0.5 * n + (2.0 if force_closure else 0.0), {
            "links_touching": n, "caged": caged, "enclosed": enclosed,
            "force_closure": force_closure,
            "isolation": self.target_isolation()}
