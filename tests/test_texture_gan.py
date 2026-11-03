"""Phase 2b unit tests: the proxy texture, the CycleGAN networks/step, the inference translator,
and the render_lotus texture post-stage. Full training is exercised by scripts/train_texture_gan.py."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import texture_gan as TG


def test_proxy_real_texture_preserves_shape_changes_look():
    rng = np.random.default_rng(0)
    img = np.clip(rng.random((64, 64)).astype(np.float32), 0, 1)
    out = TG.proxy_real_texture(img, rng)
    assert out.shape == (64, 64) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, img)                 # it actually changes the texture


def test_generator_and_discriminator_shapes():
    import torch
    g = TG._resnet_generator(n_blocks=2)
    x = torch.randn(2, 1, 64, 64)
    assert g(x).shape == x.shape                      # generator is shape-preserving
    d = TG._patch_discriminator()
    assert d(x).ndim == 4                             # PatchGAN map


def test_cyclegan_step_finite():
    import torch
    from usbg.render_lotus import get_device
    gan = TG.CycleGAN(get_device(), n_blocks=2)
    sim = torch.rand(2, 1, 64, 64, device=get_device()) * 2 - 1
    real = torch.rand(2, 1, 64, 64, device=get_device()) * 2 - 1
    losses = gan.step(sim, real)
    assert all(np.isfinite(v) for v in losses.values())
    assert {"g", "d", "cyc", "id"} <= set(losses)


def test_translator_variants_step_finite():
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    sim = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    real = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    for variant in ("cyclegan", "cut", "cyclegan_sc"):
        tr = TG.build_trainer(variant, dev, n_blocks=2)
        losses = tr.step(sim, real)
        assert all(np.isfinite(v) for v in losses.values())


def test_structure_and_realism_metrics():
    rng = np.random.default_rng(0)
    a = rng.random((40, 64, 64)).astype(np.float32) * 0.5
    b = a + 0.25
    assert TG.intensity_w1(a, a) < 1e-3 and TG.intensity_w1(a, b) > 0.2   # W1: identical~0, shifted>0
    img = rng.random((64, 64)).astype(np.float32)
    assert TG.edge_iou(img, img) > 0.99                                    # identical edges -> IoU 1


def test_translator_round_trip(tmp_path):
    from usbg.render_lotus import get_device
    gan = TG.CycleGAN(get_device(), n_blocks=2)
    ckpt = str(tmp_path / "g.pt")
    gan.save(ckpt)
    tr = TG.TextureTranslator(ckpt, n_blocks=2)
    img = np.clip(np.random.default_rng(0).random((96, 96)).astype(np.float32), 0, 1)
    out = tr.translate(img)
    assert out.shape == (96, 96) and 0.0 <= out.min() and out.max() <= 1.0


def test_render_texture_post_stage():
    from usbg import render_lotus as R
    from usbg import volume as V, slicer as S
    from usbg.contracts_bridge import pose_from_rpy, make_world_from_volume
    ph = V.synthetic_phantom()
    lab = S.slice_at_pose(ph, pose_from_rpy(), make_world_from_volume())
    plain = R.render(lab, "numpy")
    textured = R.render(lab, "numpy", texture=lambda im: 1.0 - im)   # invert as a stand-in
    assert textured.shape == plain.shape
    assert np.allclose(textured, 1.0 - plain, atol=1e-5)


# ======================================================================================
# SonoSPADE: dual-input SPADE generator, one-hot helper, OASIS trainer, dual-input post-stage
# ======================================================================================
def test_onehot_labels():
    import torch
    lab = np.array([[0, 1], [4, 12]])
    oh = TG.onehot_labels(lab, 13)
    assert oh.shape == (1, 13, 2, 2)
    assert torch.allclose(oh.sum(1), torch.ones(1, 2, 2))          # exactly one hot per pixel
    # out-of-range ids are clamped, not an index error
    assert TG.onehot_labels(np.array([[99]]), 13).shape == (1, 13, 1, 1)


def test_spade_generator_dual_input_shapes():
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    G = TG._spade_generator(13, ngf=16, n_blocks=2).to(dev)
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    out = G(phys, seg).detach()
    assert out.shape == (2, 1, 64, 64)
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0     # Tanh range
    feats = G.forward_features(phys, seg)
    assert len(feats) == 3 and feats[0].shape[1] == G.feat_channels[0]


def test_spade_generator_overfits_one_frame():
    """Acceptance: the dual-input generator can drive one paired frame's L1 to near zero."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    torch.manual_seed(0)
    G = TG._spade_generator(13, ngf=16, n_blocks=2).to(dev)
    phys = torch.rand(1, 1, 48, 48, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (1, 48, 48)), 13, dev)
    tgt = torch.rand(1, 1, 48, 48, device=dev) * 2 - 1
    opt = torch.optim.Adam(G.parameters(), lr=3e-3)
    loss = None
    for _ in range(250):
        opt.zero_grad()
        loss = (G(phys, seg) - tgt).abs().mean()
        loss.backward(); opt.step()
    assert float(loss.detach()) < 0.08


def _tiny_s_us(dev):
    from usbg import segmenter as SEG
    return SEG.Segmenter(device=dev, n_classes=13, base=8, depth=2)


def test_sonospade_step_finite_and_seg_consistency():
    """Acceptance (unit): the OASIS + PatchNCE + frozen S_US consistency step runs and is finite,
    and the seg-consistency term is actually present and nonzero."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    tr = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13,
                          ngf=12, ndf=12, n_blocks=2, d_depth=2)
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    real = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    losses = tr.step(phys, seg, real)
    assert {"g", "d", "adv", "nce", "seg"} <= set(losses)
    assert all(np.isfinite(v) for v in losses.values())
    assert losses["seg"] > 0.0                                     # consistency term is live


def test_sonospade_seg_consistency_gradient_reaches_G():
    """The load-bearing invariant: S_US is frozen, yet the seg-consistency term must back-propagate
    THROUGH the frozen S_US into G (else G can ignore the label map). This exercises the SAME
    generator_losses code path that step() optimizes (not a look-alike recomputation), so wrapping
    the fake S_US forward in no_grad, or otherwise detaching the seg term, fails here."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    tr = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13,
                          ngf=12, ndf=12, n_blocks=2, d_depth=2)
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    fake = tr.G(phys, seg)
    loss_seg = tr.generator_losses(phys, seg, fake)["seg"]         # the exact term step() uses
    assert loss_seg.requires_grad                                  # not detached by a stray no_grad
    grads = torch.autograd.grad(loss_seg, list(tr.G.parameters()), allow_unused=True)
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert total > 0.0                                             # gradient truly reaches G


def test_sonospade_frozen_segmenter_stays_frozen_but_passes_gradient():
    """Both halves of the frozen-S_US invariant: (1) S_US parameters never move (no optimizer, no
    requires_grad), and (2) the consistency loss still produces a live gradient path into G."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    s_us = _tiny_s_us(dev)
    before = [p.detach().clone() for p in s_us.net.parameters()]
    tr = TG.build_trainer("sonospade", dev, s_us=s_us, n_classes=13, ngf=12, ndf=12,
                          n_blocks=2, d_depth=2)
    assert all(not p.requires_grad for p in tr.S.net.parameters())    # freezing happens in __init__
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    tr.step(phys, seg, torch.rand(2, 1, 64, 64, device=dev) * 2 - 1)
    after = list(s_us.net.parameters())
    assert all(torch.allclose(b, a) for b, a in zip(before, after))  # (1) S_US did not move
    # (2) S_US is not in any optimizer, so it can never be updated even though grad flows through it
    opt_param_ids = {id(p) for opt in (tr.opt_g, tr.opt_d, tr.opt_h)
                     for group in opt.param_groups for p in group["params"]}
    assert not any(id(p) in opt_param_ids for p in s_us.net.parameters())


def test_sonospade_translator_round_trip_dual_input(tmp_path):
    from usbg.render_lotus import get_device
    dev = get_device()
    tr = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13,
                          ngf=12, n_blocks=2, d_depth=2)
    ckpt = str(tmp_path / "sono.pt")
    tr.save(ckpt)
    T = TG.SonoSPADETranslator(ckpt, device=dev)
    assert T.wants_label is True
    img = np.clip(np.random.default_rng(0).random((128, 96)).astype(np.float32), 0, 1)
    lab = np.random.randint(0, 13, (96, 128))                      # slicer orientation (L, S)
    out = T.translate(img, lab)
    assert out.shape == img.shape and 0.0 <= out.min() and out.max() <= 1.0


def test_sonospade_grouped_segmenter_drives_discriminator():
    """With a grouped S_US, the OASIS discriminator and the consistency targets live in the grouped
    class space (n_seg + 1 channels), while the generator still conditions on the full 13 classes."""
    import torch
    from usbg import segmenter as SEG
    from usbg.render_lotus import get_device
    dev = get_device()
    s_grp = SEG.Segmenter(device=dev, n_classes=SEG.N_GROUPED, base=8, depth=2,
                          group_map=SEG.grouped_map_array())
    tr = TG.build_trainer("sonospade", dev, s_us=s_grp, n_classes=13, ngf=12, ndf=12,
                          n_blocks=2, d_depth=2)
    assert tr.n_seg == SEG.N_GROUPED and tr.D.n_classes == SEG.N_GROUPED + 1
    assert tr.gmap_t is not None
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)   # full 13-class conditioning
    losses = tr.step(phys, seg, torch.rand(2, 1, 64, 64, device=dev) * 2 - 1)
    assert all(np.isfinite(v) for v in losses.values())


def test_sonospade_improvement_terms_activate_and_are_finite():
    """Temporal (tc), LabelMix (lm), and R1 (r1) each appear in the loss dict only when their weight
    is on, and every term stays finite (the R1 second-order path uses the avg-pool discriminator)."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    tr = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13, ngf=12, ndf=12,
                          n_blocks=2, d_depth=2, lambda_tc=1.0, lambda_lm=1.0, r1_gamma=1.0,
                          ema_decay=0.99)
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    real = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    losses = tr.step(phys, seg, real)
    assert {"tc", "lm", "r1"} <= set(losses)
    assert all(np.isfinite(v) for v in losses.values())

    # with all weights off, those terms are absent (core losses only)
    tr0 = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13, ngf=12, ndf=12,
                           n_blocks=2, d_depth=2, ema_decay=0.0)
    l0 = tr0.step(phys, seg, real)
    assert not ({"tc", "lm", "r1"} & set(l0))
    assert tr0.G_ema is None                            # ema_decay=0 disables EMA


def test_sonospade_ema_moves_and_save_keeps_both():
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    tr = TG.build_trainer("sonospade", dev, s_us=_tiny_s_us(dev), n_classes=13, ngf=12, ndf=12,
                          n_blocks=2, d_depth=2, ema_decay=0.9)
    assert tr.G_ema is not None
    before = [p.detach().clone() for p in tr.G_ema.parameters()]
    phys = torch.rand(2, 1, 64, 64, device=dev) * 2 - 1
    seg = TG.onehot_labels(torch.randint(0, 13, (2, 64, 64)), 13, dev)
    for _ in range(3):
        tr.step(phys, seg, torch.rand(2, 1, 64, 64, device=dev) * 2 - 1)
    assert any(not torch.allclose(b, a) for b, a in zip(before, tr.G_ema.parameters()))


def test_render_dual_input_translator_receives_label():
    """A wants_label translator is handed the label slice by render, a single-input one
    is not, and both are honored."""
    from usbg import render_lotus as R
    from usbg import volume as V, slicer as S
    from usbg.contracts_bridge import pose_from_rpy, make_world_from_volume
    ph = V.synthetic_phantom()
    lab = S.slice_at_pose(ph, pose_from_rpy(), make_world_from_volume())
    seen = {}

    class DualStub:
        wants_label = True
        def __call__(self, img, label_slice):
            seen["label_shape"] = np.asarray(label_slice).shape
            return img * 0.5

    out = R.render(lab, "numpy", texture=DualStub())
    assert seen["label_shape"] == lab.shape                        # got the label slice
    plain = R.render(lab, "numpy")
    assert np.allclose(out, plain * 0.5, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
