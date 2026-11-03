"""SARSA(λ) with hashed tile coding (thesis-faithful learner).

Port of Anders (2014): 27 discrete joint-increment actions
({-0.1, 0, +0.1}^3 per step), tile-coded linear Q over
(q0, q1, q2, target_x, target_y, target_yaw), the exact thesis
reward schedule (0/-100/-0.25/-0.5/-1), 60-step episodes,
epsilon-greedy, randomized initial joints (thesis: noise on the
initial position), accumulating eligibility traces (λ=0.9).

Differs from the thesis only in the plant: MuJoCo top-down planar
scene (possibly with distractors) instead of Box2D, single target
slot. Distractors are present but invisible to the tiles (thesis
state has no clutter terms); use --n-objects 1 for the closest
match, 4 (default) for benchmark parity.

Usage:
    python scripts/train_sarsa.py --episodes 5000 --out models/sarsa
    python scripts/train_sarsa.py --episodes 200  # smoke test
"""
import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper  # noqa: E402
from scripts.tile_coding import TileCoder  # noqa: E402


def target_yaw(env, target_index):
    """Planar yaw of a box from its (w, x, y, z) quaternion."""
    q = env._obs()["objects"][target_index][3:7]
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def tile_state(obs, yaw, touch=0, enclosed=False,
               include_contact=False):
    """State vector: joints + target (x, y, yaw) [+ contact]."""
    s = [obs[0], obs[1], obs[2], obs[6], obs[7], yaw]
    if include_contact:
        s += [min(touch, 3) + 0.5, (1.5 if enclosed else 0.5)]
    return np.array(s, dtype=float)


def make_coder(tilings, tiles, memory, include_contact=False):
    """TileCoder over q(±2.6) x target-xy(±0.75) x yaw(wrapped)
    [+ touch-count(4) x enclosed(2) when include_contact]."""
    lows = [-2.6] * 3 + [-0.75] * 2 + [-np.pi]
    highs = [2.6] * 3 + [0.75] * 2 + [np.pi]
    if include_contact:
        lows += [0.0, 0.0]
        highs += [4.0, 2.0]
        tiles = list(tiles) + [4, 2]
    return TileCoder(lows=lows, highs=highs, tiles=tiles,
                     tilings=tilings, wrap=(5,), memory=memory)


class LinearSarsa:
    """SARSA(λ) with per-action tile-coded linear Q and
    accumulating eligibility traces (λ=0 recovers one-step)."""

    def __init__(self, coder, n_actions=27, alpha=0.1, gamma=1.0,
                 lmbda=0.9, seed=0):
        self.coder = coder
        self.n_actions = n_actions
        self.alpha = alpha / coder.tilings
        self.gamma = gamma
        self.lmbda = lmbda
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((n_actions, coder.n_tiles), dtype=float)
        self.z = np.zeros((n_actions, coder.n_tiles), dtype=float)

    def reset_traces(self):
        """Clear eligibility traces (call at episode start)."""
        self.z.fill(0.0)

    def q_values(self, state):
        """Q(s, .) for all actions."""
        t = self.coder.tiles_for(state)
        return np.array([self.w[a][t].sum()
                         for a in range(self.n_actions)])

    def act(self, state, epsilon):
        """Epsilon-greedy action (random tie-break)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(state)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, state, action, reward, next_state, next_action,
               done):
        """SARSA(λ) update (no-op target when done)."""
        t = self.coder.tiles_for(state)
        target = reward
        if not done:
            nt = self.coder.tiles_for(next_state)
            target += self.gamma * self.w[next_action][nt].sum()
        td_err = target - self.w[action][t].sum()
        self.z[action][t] += 1.0
        self.w += self.alpha * td_err * self.z
        if done:
            self.z.fill(0.0)
        else:
            self.z *= self.gamma * self.lmbda
        return float(td_err)


def run_episode(agent, env, epsilon, actions, max_delta=2.6,
                random_init=0.0, near_start=0.0, rng=None,
                include_contact=False):
    """One SARSA episode; returns (return, fc, steps).

    random_init > 0 samples the starting joints uniform in
    [-random_init, +random_init] (thesis: noise on the initial
    position). near_start > 0 additionally rejection-samples
    starts with the tip within near_start of the red block, so
    contact is dense from step 0 and the -0.25/-0.5 shaping —
    not the -100 timeout — dominates the TD signal. (Without
    this, timeout is the 97% outcome, traces smear -100 over
    every trail, and the greedy policy learns to hide in
    unvisited corners: 2/10 eval touches vs 4/10 random.)
    Eval always uses home starts.
    """
    obs, _ = env.reset()
    agent.reset_traces()
    if random_init > 0 or near_start > 0:
        red = env.env._obs()["objects"][env.target_index][:2]
        best_q, best_d = None, float("inf")
        for _ in range(50 if near_start > 0 else 1):
            q0 = rng.uniform(-random_init, random_init, 3)
            env.env.data.qpos[env.env._arm_qpos] = q0
            mujoco.mj_forward(env.env.model, env.env.data)
            tip = env.env._arm_joint_xy()[-1]
            dist = float(np.linalg.norm(tip - red))
            if dist < best_d:
                best_q, best_d = q0.copy(), dist
            if near_start <= 0 or dist <= near_start:
                break
        else:
            env.env.data.qpos[env.env._arm_qpos] = best_q
            mujoco.mj_forward(env.env.model, env.env.data)
        obs = env._flat_obs()
    yaw = target_yaw(env.env, env.target_index)
    state = tile_state(obs, yaw, include_contact=include_contact)
    action = agent.act(state, epsilon)
    q_cmd = obs[:3].astype(float).copy()
    total, steps = 0.0, 0
    fc = False
    while True:
        q_cmd = np.clip(q_cmd + actions[action], -max_delta, max_delta)
        obs, reward, terminated, truncated, info = env.step(
            q_cmd / max_delta)
        total += reward
        steps += 1
        done = terminated or truncated
        fc = fc or bool(info["force_closure"])
        yaw = target_yaw(env.env, env.target_index)
        next_state = tile_state(
            obs, yaw, touch=info.get("links_touching", 0),
            enclosed=info.get("enclosed", False),
            include_contact=include_contact)
        next_action = agent.act(next_state, epsilon) if not done else 0
        agent.update(state, action, reward, next_state, next_action,
                     done)
        state, action = next_state, next_action
        if done:
            break
    return total, fc, steps


def evaluate(agent, args, actions):
    """Greedy FC rate + mean return over the eval seeds."""
    fcs, rets = [], []
    for seed in args.eval_seeds:
        env = GraspGymWrapper(
            seed=seed, target_index=0, max_steps=args.max_steps,
            n_objects=args.n_objects, thesis_reward=True,
            thesis_timeout=args.timeout_penalty)
        total, fc, _ = run_episode(
            agent, env, 0.0, actions,
            include_contact=args.contact_state)
        fcs.append(fc)
        rets.append(total)
    return sum(fcs) / len(fcs), sum(rets) / len(rets)


def main(argv=None):
    """Train linear SARSA; eval checkpoints; save weights."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--random-init", type=float, default=2.0)
    ap.add_argument("--near-start", type=float, default=0.25)
    ap.add_argument("--contact-state", action="store_true")
    ap.add_argument("--timeout-penalty", type=float, default=-100.0)
    ap.add_argument("--n-objects", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eps0", type=float, default=0.2)
    ap.add_argument("--eps1", type=float, default=0.01)
    ap.add_argument("--tilings", type=int, default=8)
    ap.add_argument("--tiles", type=int, nargs=6,
                    default=[7, 7, 7, 9, 9, 5])
    ap.add_argument("--memory", type=int, default=4096)
    ap.add_argument("--lmbda", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/sarsa")
    args = ap.parse_args(argv)

    coder = make_coder(args.tilings, args.tiles, args.memory,
                       args.contact_state)
    agent = LinearSarsa(coder, alpha=args.alpha, gamma=args.gamma,
                        lmbda=args.lmbda, seed=args.seed)
    actions = coder.all_actions()
    train_rng = np.random.default_rng(args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sarsa] {coder.n_tiles} tiles/action x 27 actions, "
          f"episodes={args.episodes} max_steps={args.max_steps} "
          f"objects={args.n_objects}", flush=True)

    best_fc, t0 = -1.0, time.time()
    for ep in range(1, args.episodes + 1):
        frac = ep / max(args.episodes, 1)
        eps = args.eps0 + frac * (args.eps1 - args.eps0)
        env = GraspGymWrapper(
            seed=int(train_rng.integers(1_000_000)),
            target_index=0, max_steps=args.max_steps,
            n_objects=args.n_objects, thesis_reward=True,
            thesis_timeout=args.timeout_penalty)
        total, fc, steps = run_episode(
            agent, env, eps, actions,
            random_init=args.random_init,
            near_start=args.near_start, rng=train_rng,
            include_contact=args.contact_state)
        if ep % max(args.eval_every // 5, 1) == 0:
            print(f"[sarsa] ep={ep} eps={eps:.3f} ret={total:.1f} "
                  f"fc={int(fc)} steps={steps}", flush=True)
        if ep % args.eval_every == 0 or ep == args.episodes:
            fc_rate, mean_ret = evaluate(agent, args, actions)
            el = (time.time() - t0) / 60
            print(f"[sarsa] EVAL ep={ep}: fc={fc_rate:.2f} "
                  f"ret={mean_ret:.1f} ({el:.0f} min)", flush=True)
            if fc_rate > best_fc:
                best_fc = fc_rate
                meta = {"tilings": args.tilings, "tiles": args.tiles,
                        "memory": args.memory, "alpha": args.alpha,
                        "gamma": args.gamma, "lmbda": args.lmbda,
                        "max_steps": args.max_steps,
                        "random_init": args.random_init,
                        "near_start": args.near_start,
                        "contact_state": args.contact_state,
                        "timeout_penalty": args.timeout_penalty,
                        "n_objects": args.n_objects,
                        "episodes": ep, "eval_fc": fc_rate}
                save_path = out.with_suffix(".npz")
                np.savez_compressed(save_path, weights=agent.w,
                                    meta=np.array([repr(meta)]))
                print(f"[sarsa] saved {save_path} (fc={fc_rate:.2f})",
                      flush=True)
    print(f"[sarsa] done: best eval FC={best_fc:.2f}")


if __name__ == "__main__":
    main()
