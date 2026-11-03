"""USBGym/ViewMatch-v0 with a random policy and with the image-servo base alone, on the CPU.

Run from the repository root:

    PYTHONPATH=src python examples/view_match_gym.py --episodes 4

It needs the anatomical CT caches (data/cache_anat/) and the screened-windows file
(data/cache/windows/usbg-windows-v1.json); README.md, "Data preparation", builds both from the public
TotalSegmentator data (or set USBG_WINDOWS to the windows file shipped in src/usbg/data/). No GPU is needed. Episode k is episode k of the
paper's crossed evaluation design on the six held-out patients. The pose error printed at the end of
each episode is privileged simulator state, shown as a diagnostic only; neither controller reads it.
"""
import argparse

import gymnasium as gym
import numpy as np

import usbg.gym_env  # noqa: F401  (registers USBGym/ViewMatch-v0)
from usbg.gym_env import calibrate_servo_base


def run(env, policy, episodes, label):
    for k in range(episodes):
        obs, info = env.reset(options={"eval_episode": k})
        start, ret, done = info["privileged"]["pose_error_mm"], 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            ret += reward
            done = terminated or truncated
        print(f"{label:<12} {info['patient']} {info['target_organ']:<6} window {info['window']}: "
              f"return {ret:+.3f}, match {info['match']:.3f}, "
              f"pose error {start:.1f} -> {info['privileged']['pose_error_mm']:.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps")
    ap.add_argument("--calib-pairs", type=int, default=500, help="frame pairs for the servo calibration")
    a = ap.parse_args()

    # 1. Random policy. Without a base the action is the full normalized twist.
    env = gym.make("USBGym/ViewMatch-v0", patients="test", texture="physics", mask="ct", device=a.device)
    env.action_space.seed(0)
    run(env, lambda obs: env.action_space.sample(), a.episodes, "random")
    env.close()

    # 2. Image-servo base alone: calibrated on the TRAIN patients, then a zero residual.
    base = calibrate_servo_base(n=a.calib_pairs, seed=0, device=a.device)
    env = gym.make("USBGym/ViewMatch-v0", patients="test", texture="physics", mask="ct",
                   base="servo", servo_base=base, device=a.device)
    run(env, lambda obs: np.zeros(6, np.float32), a.episodes, "servo base")
    env.close()


if __name__ == "__main__":
    main()
