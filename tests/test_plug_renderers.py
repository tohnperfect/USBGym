"""Fast checks of usbg.plug.renderers (v2.1 pluggability, renderer slot; exploratory). Synthetic label slices only,
no patient data. The SIMUS check uses a very sparse phantom so it runs in seconds on one CPU process."""
import numpy as np
import pytest
import torch

from usbg import render_lotus as R
from usbg._reuse import N_LINES, N_SAMPLES
from usbg.plug import renderers as PR


def _has_lotus():
    try:
        PR.lotus_dir()
        return True
    except FileNotFoundError:
        return False


# the original LOTUS code is the git submodule third_party/lotus (README, Install)
needs_lotus = pytest.mark.skipif(not _has_lotus(), reason="third_party/lotus not checked out: git submodule update --init")


def _slice(seed=0):
    """A canonical (N_LINES, N_SAMPLES) label slice: gel gap, soft tissue, a liver block, a vessel and a bone."""
    lab = np.full((N_LINES, N_SAMPLES), 12, np.uint8)
    lab[:, :12] = 0                                   # leading gap (gel under the anatomical seat)
    lab[30:130, 150:420] = 1                          # liver
    lab[70:90, 250:280] = 3                           # vessel
    lab[120:140, 300:320] = 10                        # bone
    return lab


@needs_lotus
def test_registered_and_contract():
    PR.register_all()
    assert {"lotus_reimpl", "lotus_orig", "simus"} <= set(R.list_renderers())
    lab = _slice()
    for name in ("lotus_reimpl", "lotus_orig"):
        img = R.render(lab, name, device="cpu", cfg=R.RenderConfig(seed=3))
        assert img.shape == (N_SAMPLES, N_LINES) and img.dtype == np.float32
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


@needs_lotus
def test_deterministic_and_seeded():
    lab = _slice()
    for fn in (PR.render_lotus_reimpl, PR.render_lotus_original):
        a = fn(lab, device="cpu", seed=5)
        assert np.array_equal(a, fn(lab, device="cpu", seed=5))
        assert not np.array_equal(a, fn(lab, device="cpu", seed=6))
        assert np.array_equal(a, fn(lab, device="cpu", cfg=R.RenderConfig(seed=5)))   # cfg.seed is the seed


def test_reimpl_equals_builtin_lotus():
    lab = _slice()
    assert np.array_equal(PR.render_lotus_reimpl(lab, device="cpu", seed=2),
                          R.render_bmode_lotus(lab, device="cpu", seed=2))


@needs_lotus
def test_global_rng_untouched():
    lab = _slice()
    torch.manual_seed(11); ref = torch.rand(4)
    torch.manual_seed(11)
    PR.render_lotus_reimpl(lab, device="cpu", seed=1)
    PR.render_lotus_original(lab, device="cpu", seed=1)
    assert torch.equal(ref, torch.rand(4))


@needs_lotus
def test_cuda_redirect_is_scoped():
    to0 = torch.Tensor.to
    PR.render_lotus_original(_slice(), device="cpu", seed=0)
    assert torch.Tensor.to is to0
    with pytest.raises((AssertionError, RuntimeError)):
        torch.zeros(1).to("cuda")                     # no CUDA here: the patch must not leak


def test_lotus_label_map_covers_ontology():
    ids = PR.lotus_orig_labels(np.arange(13, dtype=np.int64)[None, :].repeat(2, 0), gel=False)
    assert set(np.unique(ids)) <= set(PR.LOTUS_IDS)
    gap = PR.lotus_orig_labels(_slice(), gel=True)
    assert (gap[:, :12] == PR.LOTUS_GEL_ID).all()


@needs_lotus
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_lotus_orig_cpu_mps_close():
    lab = _slice()
    a = PR.render_lotus_original(lab, device="cpu", seed=4)
    b = PR.render_lotus_original(lab, device="mps", seed=4)
    assert float(np.abs(a - b).max()) < 1e-4


@needs_lotus
def test_install_routes_physics_and_restores():
    lab = _slice()
    cfg = R.RenderConfig(seed=7)
    ref = R.render(lab, "physics", device="cpu", cfg=cfg)
    try:
        PR.install("lotus_orig")
        got = R.render(lab, "physics", device="cpu", cfg=cfg)
        assert np.array_equal(got, PR.render_lotus_original(lab, device="cpu", seed=7))
        with PR.plugged("physics"):
            assert np.array_equal(R.render(lab, "physics", device="cpu", cfg=cfg), ref)
        assert PR.active() == "lotus_orig"
        assert PR.stats()["lotus_orig"]["n"] >= 1
    finally:
        PR.uninstall()
    assert np.array_equal(R.render(lab, "physics", device="cpu", cfg=cfg), ref)
    assert not getattr(R.render, "__usbg_plug__", False)


def test_simus_tiny_phantom():
    pytest.importorskip("pymust")
    lab = _slice()
    kw = dict(density_per_mm2=0.05, workers=1, grid_samples=64, n_elements=64)
    a = PR.render_simus(lab, seed=1, simus=kw)
    assert a.shape == (N_SAMPLES, N_LINES) and a.dtype == np.float32
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert np.array_equal(a, PR.render_simus(lab, seed=1, simus=kw))
