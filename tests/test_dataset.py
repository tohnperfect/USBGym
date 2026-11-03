"""Phase 4 acceptance: the dataset round-trips (npz + LeRobot-style), shapes/labels align with
par.contracts, and the demonstrator yields recognizer-compatible twist. Core schema tests use
synthetic clips (no CT / PPO needed); one light test exercises the real clean demonstrator."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from usbg import dataset as D
from usbg.contracts_bridge import CLASSES_5, CHANNELS_TWIST, IN_CHANNELS


def _synthetic_clip(n, label, source, case, tid, size=32):
    rng = np.random.default_rng(tid)
    imgs = rng.integers(0, 256, size=(n, size, size), dtype=np.uint8)
    poses = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    tw = rng.normal(size=(n, 6)).astype(np.float32)
    tw[0] = np.nan                                   # frame-0 twist undefined (recognizer rule)
    rel = rng.normal(size=(n, 6)).astype(np.float32)
    valid = np.isfinite(tw).all(axis=1)
    return D.ClipSample(images=imgs, poses=poses, twist=tw, rel6=rel, valid=valid,
                        label=label, instruction=f"do {CLASSES_5[label]}", source=source,
                        case=case, traj_id=tid)


@pytest.fixture(scope="module")
def clips():
    return [_synthetic_clip(12, 0, "clean", "s1086", 0),
            _synthetic_clip(8, 3, "noisy", "s0777", 1)]


def test_npz_round_trip(tmp_path, clips):
    path = str(tmp_path / "ds.npz")
    _p, meta = D.write_npz(path, clips, image_size=32)
    loaded, lmeta = D.load_npz(path)
    assert len(loaded) == 2
    assert lmeta["classes"] == list(CLASSES_5)
    assert lmeta["n_clean"] == 1 and lmeta["n_noisy"] == 1
    c0 = loaded[0]
    assert c0["images"].shape == (12, 32, 32) and c0["images"].dtype == np.uint8
    assert c0["twist"].shape == (12, 6) and c0["rel6"].shape == (12, 6)
    assert c0["images"].shape[0] == c0["twist"].shape[0] == c0["poses"].shape[0]
    assert 0 <= c0["label"] <= 4


def test_frame0_invalid_and_anticipation(clips):
    c = clips[0]
    assert not bool(c.valid[0])                      # frame-0 twist is NaN
    assert bool(c.valid[1])
    assert (c.next_id() == -1).all()                 # single-action clip: no next maneuver
    assert c.ttnb()[0] == len(c.images) - 1 and c.ttnb()[-1] == 0


def test_channels_match_contract(clips):
    assert len(CHANNELS_TWIST) == 6 and IN_CHANNELS == 12
    assert clips[0].twist.shape[1] == 6


def test_lerobot_export(tmp_path, clips):
    outd = str(tmp_path / "lerobot")
    _o, info = D.write_lerobot(outd, clips, image_size=32)
    assert os.path.exists(os.path.join(outd, "meta", "info.json"))
    assert os.path.exists(os.path.join(outd, "data", "chunk-000", "episode_000000.jsonl"))
    assert os.path.isdir(os.path.join(outd, "images", "observation.images.bmode", "episode_000000"))
    assert info["fps"] == 10.0 and info["total_episodes"] == 2
    # a frame record carries a 6-DOF action and a task string
    with open(os.path.join(outd, "data", "chunk-000", "episode_000000.jsonl")) as fh:
        rec = json.loads(fh.readline())
    assert len(rec["action"]) == 6 and "task" in rec


def test_contact_sheet(tmp_path, clips):
    out = str(tmp_path / "sheet.png")
    D.contact_sheet(clips, out)
    assert os.path.exists(out)


@pytest.mark.slow
def test_clean_demonstrator_twist_shape():
    # light integration: the clean parametric demonstrator yields recognizer-shaped twist (v1 only: it needs the
    # arm simulator's package us_rl_sim, which is not part of this repository)
    pytest.importorskip("us_rl_sim")
    from usbg.demonstrator import Demonstrator
    demo = Demonstrator()
    try:
        tr = demo.clean("slide", np.random.default_rng(0))
        assert tr.twist.shape[1] == 6 and tr.rel6.shape[1] == 6
        assert tr.poses.shape[1:] == (4, 4)
        assert tr.label == CLASSES_5.index("slide")
        assert not np.isfinite(tr.twist[0]).all()    # frame-0 NaN
    finally:
        demo.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
