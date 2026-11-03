"""demonstrator.py: wrap the Repo B motion source into probe-pose trajectories.

Produces both trajectory kinds, mirroring the arm simulator's own
train split (see scripts/generate_action_dataset.py):
  clean : parametric ground-truth maneuvers (movement_poses; no IK/physics/policy)
  noisy : the trained PPO policy tracing the same maneuver (real, imperfect tracking)

Both are resampled to the canonical 10 Hz and carry the recognizer's exact features: body
twist [vx,vy,vz,wx,wy,wz] (SI units, via twist6 which uses the vendored se3_log) and the
start-relative pose rel6. We do NOT add the AprilTag measurement noise here: that is a
kinematics-only augmentation, and adding it would desync the stored twist from the pose the
B-mode is rendered at. The maneuver label and instruction string come straight from Repo B.

Poses are returned as (N,4,4) SE(3) in Repo B's tool frame. The dataset generator replays each
maneuver's RELATIVE motion on a base pose over the target organ, which leaves the body twist
(hence the labels) unchanged while making the render intersect anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    poses: np.ndarray      # (N,4,4) SE(3), Repo B tool frame
    twist: np.ndarray      # (N,6) body twist [vx,vy,vz,wx,wy,wz], SI, raw (NaN at frame 0)
    rel6: np.ndarray       # (N,6) start-relative pose [x,y,z,rx,ry,rz]
    valid: np.ndarray      # (N,) bool
    label: int             # index into MOVEMENTS (== CLASSES_5)
    action: str
    instruction: str
    source: str            # "clean" | "noisy"

    def __len__(self):
        return len(self.poses)


def _to_T(pr) -> np.ndarray:
    pos, R = pr
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _stack_T(poses_pr) -> np.ndarray:
    return np.stack([_to_T(pr) for pr in poses_pr], axis=0)


class Demonstrator:
    """Repo B motion source (env + trained policy), exposing clean and noisy trajectories."""

    DEFAULT_CONFIG = "configs/train/ppo_reach_moves_bc.yaml"
    DEFAULT_CKPT = None   # v1 only: pass ckpt=, the arm simulator's policy

    def __init__(self, config: str | None = None, ckpt: str | None = None,
                 out_rate_hz: float = 10.0, swap_rock_fan: bool = True):
        from us_rl_sim.config import build_env_config, load_train_config, make_reach_env
        from us_rl_sim.demos import action_data as A
        from us_rl_sim.demos import parametric as P

        self._A = A
        self._P = P
        self.MOVEMENTS = list(P.MOVEMENTS)              # == CLASSES_5
        self.out_rate_hz = out_rate_hz
        self.dt_c = 1.0 / out_rate_hz
        self.swap = bool(swap_rock_fan)

        cfg = build_env_config(load_train_config(config or self.DEFAULT_CONFIG))
        self.env = make_reach_env(cfg)
        self._policy = None
        self._ckpt = ckpt or self.DEFAULT_CKPT

    @property
    def policy(self):
        if self._policy is None:
            from stable_baselines3 import PPO
            self._policy = PPO.load(self._ckpt)
        return self._policy

    # features shared by both sources
    def _features(self, poses_native_pr):
        A = self._A
        poses_10 = A.poses_at_rate(poses_native_pr, self.env.dt, self.out_rate_hz)
        rel = A.rel6(poses_10)
        tw = A.twist6(poses_10, self.dt_c, out_rate_hz=self.out_rate_hz)
        return _stack_T(poses_10), np.asarray(tw, np.float64), np.asarray(rel, np.float32)

    def clean(self, action: str, rng) -> Trajectory:
        A, P = self._A, self._P
        params = P.sample_params(action, rng)
        poses_native, _ = A.movement_poses(self.env, action, params, swap_rock_fan=self.swap)
        T, tw, rel = self._features(poses_native)
        valid = np.isfinite(tw).all(axis=1)
        return Trajectory(poses=T, twist=tw, rel6=rel, valid=valid,
                          label=self.MOVEMENTS.index(action), action=action,
                          instruction=P.describe(action, params), source="clean")

    def noisy(self, action: str, rng, seed: int) -> Trajectory:
        A, P = self._A, self._P
        params = P.sample_params(action, rng)
        plan = A.movement_plan(self.env, action, params, swap_rock_fan=self.swap)
        qpos, _labels = A.policy_session(self.env, self.policy, [(action, plan)], seed=seed)
        if len(qpos) < 2:
            # policy produced a degenerate track; fall back to a clean maneuver
            return self.clean(action, rng)
        T, tw, rel = self._features(A.fk_poses(self.env, qpos))
        valid = np.isfinite(tw).all(axis=1)
        return Trajectory(poses=T, twist=tw, rel6=rel, valid=valid,
                          label=self.MOVEMENTS.index(action), action=action,
                          instruction=P.describe(action, params), source="noisy")

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
