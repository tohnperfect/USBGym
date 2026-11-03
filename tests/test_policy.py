"""Phase 6 unit tests: the policy builds and forwards to the right shapes, TwistNorm round-trips,
and the frame dataset stacks history correctly. Full training and rollout belonged to the retired v1 behavior-cloning task."""
from __future__ import annotations

import numpy as np
import pytest

from usbg.policy import TwistNorm, BModeFrameDataset, BModePolicy
from usbg.contracts_bridge import TARGET_CHANNELS, N_CLASSES, CLASSES_5


def _clip(n, label, size=32, tid=0):
    rng = np.random.default_rng(tid)
    tw = rng.normal(size=(n, 6)).astype(np.float64)
    tw[0] = np.nan
    return {"images": rng.integers(0, 256, (n, size, size), dtype=np.uint8),
            "twist": tw, "valid": np.isfinite(tw).all(1), "label": label,
            "source": "clean", "case": "t", "traj_id": tid}


def test_twist_norm_roundtrip():
    clips = [_clip(10, 0, tid=0), _clip(8, 3, tid=1)]
    tn = TwistNorm.fit(clips)
    z = tn.normalize(clips[0]["twist"][1:])
    back = tn.denormalize(z)
    assert np.allclose(back, clips[0]["twist"][1:], atol=1e-9)
    assert tn.mean.shape == (6,) and tn.std.shape == (6,)


def test_frame_dataset_history_stack():
    clips = [_clip(10, 2, size=32, tid=0)]
    tn = TwistNorm.fit(clips)
    ds = BModeFrameDataset(clips, tn, image_size=32, history=4)
    assert len(ds) == int(clips[0]["valid"].sum())    # only valid frames
    x, man, tw = ds[0]
    assert x.shape == (4, 32, 32) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert man == 2 and tw.shape == (6,)


def test_policy_forward_shapes():
    import torch
    net = BModePolicy.build(history=4)
    x = torch.randn(5, 4, 128, 128)
    man = torch.randint(0, len(CLASSES_5), (5,))
    out = net(x, man)
    assert out["twist"].shape == (5, TARGET_CHANNELS)
    assert out["next_logits"].shape == (5, N_CLASSES)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
