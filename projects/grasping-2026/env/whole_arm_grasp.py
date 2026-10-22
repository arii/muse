"""Planar whole-arm grasping environment (MuJoCo).

A 3-link arm sweeps a horizontal plane above a table. Each episode spawns
N randomized box objects within reach. The task: envelop ("cage") a target
object using the whole arm — contacts on >=2 distinct links counts as a cage.

Reward (starting point — tune freely):
    r = 0.5 * (# distinct arm links touching target) + 2.0 * caged
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
            body.add_freejoint(name=f"obj{i}_free")
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, sz],
                rgba=PALETTE[cname],
                friction=[0.8, 0.005, 0.0001],
                name=f"obj{i}_geom",
            )
            self.objects.append({"index": i, "name": f"obj{i}",
                                 "color": cname, "half": (sx, sy, sz)})
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self._arm_qpos = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            for jn in JOINT_NAMES
        ]
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

    # ---------- helpers ----------
    def _obs(self):
        qpos = self.data.qpos[self._arm_qpos].copy()
        qvel = self.data.qvel[:3].copy()
        obj_poses = {}
        for o in self.objects:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{o['name']}_free")
            adr = self.model.jnt_qposadr[jid]
            obj_poses[o["index"]] = self.data.qpos[adr:adr + 7].copy()
        return {"qpos": qpos, "qvel": qvel, "objects": obj_poses,
                "target": self.target_index}

    def _geom_name(self, gid):
        n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        return n.decode() if isinstance(n, bytes) else n

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
        caged = n >= 2
        return 0.5 * n + (2.0 if caged else 0.0), {
            "links_touching": n, "caged": caged}
