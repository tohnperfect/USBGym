"""Unit tests for the v2.1 texture plug-ins (usbg.plug_texture). CPU only, random tensors, no patient data."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("variant", ["fastcut", "dclgan", "unsb"])
def test_trainer_step_save_load_roundtrip(tmp_path, variant):
    from usbg import plug_texture as PT  # noqa: F401  (registers the plug-ins)
    from usbg import texture_gan as TG
    torch.manual_seed(0)
    g = TG.build_trainer(variant, "cpu", n_blocks=6)
    sim = torch.rand(2, 1, 32, 32) * 2 - 1
    real = torch.rand(2, 1, 32, 32) * 2 - 1
    for _ in range(2):
        losses = g.step(sim, real)
    assert all(np.isfinite(v) for v in losses.values())
    p = str(tmp_path / f"{variant}.pt")
    g.save(p)
    tr = TG.load_translator(p, "cpu")
    x = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    a, b = tr(x), tr(x)
    assert a.shape == x.shape and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0
    np.testing.assert_array_equal(a, b)                      # every texture stage is deterministic
    if variant == "unsb":
        assert type(tr).__name__ == "UNSBTranslator" and not getattr(tr, "wants_label", False)
    else:
        assert type(tr).__name__ == "TextureTranslator"      # unchanged inference path (G_s2r)


def test_unsb_schedule_matches_official_form():
    from usbg.plug_texture import unsb_times
    t = unsb_times(5)
    assert t.shape == (6,) and t[0] == 0 and abs(t[-1] - 1) < 1e-6 and abs(t[1] - 0.5) < 1e-6
    assert np.all(np.diff(t) > 0)


def test_histmatch_fit_is_monotone_and_maps_to_real(tmp_path):
    from usbg import plug_texture as PT
    from usbg import texture_gan as TG
    rng = np.random.default_rng(0)
    sim = (rng.random((10, 128, 128)) ** 2).astype(np.float32)
    real = rng.random((10, 128, 128)).astype(np.float32)
    lut, st = PT.fit_histmatch(sim, real)
    assert lut.shape == (256,) and np.all(np.diff(lut) >= 0)
    assert abs(st["mapped_mean"] - st["real_mean"]) < 0.02
    p = str(tmp_path / "hm.pt")
    PT.save_histmatch(p, lut, st)
    tr = TG.load_translator(p, "cpu")
    out = tr(sim[0])
    m = PT.sector_mask(128, 128)
    assert np.all(out[~m] == 0)                              # outside the sector stays background


def test_widen_choices_is_process_local():
    code = ("import argparse; from usbg import plug_texture as PT; PT.widen_choices();"
            "ap = argparse.ArgumentParser(); ap.add_argument('--translator', choices=['physics', 'cut']);"
            "print(ap.parse_args(['--translator', 'unsb']).translator)")
    env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.strip() == "unsb", out.stderr
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", choices=["physics", "cut"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--translator", "unsb"])              # this process was never widened
