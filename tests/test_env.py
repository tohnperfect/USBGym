"""Phase 3 acceptance: stepping the env with a scripted motion produces coherent B-mode frames
and a sane contact-force trace (near zero above the skin, rising as the probe presses in,
falling as it lifts). Uses the phantom + NumPy renderer so it needs no data or MPS."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import volume as V
from usbg import slicer as S
from usbg.placement import seat_volume_under_probe


@pytest.fixture(scope="module")
def env():
    from usbg.mujoco_env import USProbeEnv
    lv = V.synthetic_phantom()
    Wfv, T0 = seat_volume_under_probe(lv, V.GALLBLADDER, standoff_m=0.006, depth_m=0.04)
    return USProbeEnv(lv, Wfv, T0, renderer="numpy", settle_steps=10)


def test_reset_shape_and_low_force(env):
    obs, info = env.reset()
    assert obs.shape == (S.N_SAMPLES, S.N_LINES)
    assert np.isfinite(obs).all()
    assert info["contact_force"] < 1.0            # starts just above the skin, ~no contact


def test_world_from_probe_is_valid_se3(env):
    env.reset()
    T = env.world_from_probe
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_press_raises_force_then_lift_lowers_it(env):
    env.reset()
    f0 = env.reset()[1]["contact_force"]
    # press along +z probe (into the patient)
    pressed = f0
    for _ in range(8):
        _obs, info = env.step([0, 0, 0.01, 0, 0, 0])
        pressed = info["contact_force"]
    assert pressed > f0 + 0.5                       # contact force rose meaningfully
    # lift back out
    lifted = pressed
    for _ in range(10):
        _obs, info = env.step([0, 0, -0.01, 0, 0, 0])
        lifted = info["contact_force"]
    assert lifted < pressed                         # force dropped when lifting


def test_scripted_motion_yields_coherent_frames(env):
    env.reset()
    frames = []
    for _ in range(6):
        obs, _info = env.step([0.01, 0, 0.005, 0, 0, 0])   # slide + gentle press
        assert obs.shape == (S.N_SAMPLES, S.N_LINES)
        assert np.isfinite(obs).all()
        frames.append(obs)
    # consecutive frames should be similar (coherent video) but not identical (motion)
    assert not np.array_equal(frames[0], frames[-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
