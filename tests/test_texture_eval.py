"""Per-tissue texture metric tests: speckle SNR, GLCM, per-class W1 monotonicity, temporal flicker,
and a TSTR smoke (train a segmenter on synthetic, test on held-out synthetic)."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import texture_eval as TE


def test_speckle_snr_basic():
    # a (near) constant region carries no speckle -> NaN (NOT a huge finite number that would
    # silently corrupt np.nanmean aggregates); a genuinely noisy region -> finite small SNR
    flat = np.full((32, 32), 0.5, np.float32)
    assert np.isnan(TE.speckle_snr(flat))
    near_flat = flat + np.float32(1e-5)                  # std well below min_std
    assert np.isnan(TE.speckle_snr(near_flat))
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(0.5, 0.2, (32, 32)).astype(np.float32), 0, 1)
    assert 0.5 < TE.speckle_snr(noisy) < 20


def test_speckle_snr_constant_region_not_folded_into_aggregate():
    """A flat class region must not corrupt the per-class SNR aggregate (the known bug: the
    sd+eps floor used to return ~5e5 for a flat 0.5 region, which np.nanmean did not filter)."""
    img = np.zeros((16, 32), np.float32)
    lab = np.zeros((16, 32), np.int64)
    lab[:, :16] = 1; img[:, :16] = 0.5                   # class 1: perfectly flat (no speckle)
    rng = np.random.default_rng(0)
    lab[:, 16:] = 4; img[:, 16:] = np.clip(rng.normal(0.5, 0.15, (16, 16)), 0, 1)  # class 4: speckle
    snr = TE.speckle_snr_per_class(img, lab, min_px=16)
    assert 1 not in snr                                  # flat class dropped, not a huge number
    assert 4 in snr and snr[4] < 100


def test_speckle_snr_per_class():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), np.float32)
    lab = np.zeros((32, 32), np.int64)
    lab[:16] = 1; lab[16:] = 4
    img[:16] = np.clip(rng.normal(0.6, 0.12, (16, 32)), 0, 1)   # both regions carry speckle
    img[16:] = np.clip(rng.normal(0.3, 0.08, (16, 32)), 0, 1)
    snr = TE.speckle_snr_per_class(img, lab, min_px=16)
    assert 1 in snr and 4 in snr                     # both textured classes measured


def test_glcm_per_class_runs():
    rng = np.random.default_rng(0)
    img = rng.random((48, 48)).astype(np.float32)
    lab = np.zeros((48, 48), np.int64); lab[:24] = 1; lab[24:] = 7
    g = TE.glcm_per_class(img, lab, min_px=32)
    assert 1 in g and "contrast" in g[1] and "homogeneity" in g[1]
    assert np.isfinite(g[1]["contrast"]) and np.isfinite(g[1]["homogeneity"])


def test_intensity_w1_per_class_monotonic():
    lab = np.zeros((32, 32), np.int64); lab[:16] = 1; lab[16:] = 4
    a = [np.full((32, 32), 0.4, np.float32) for _ in range(4)]
    b_close = [np.full((32, 32), 0.42, np.float32) for _ in range(4)]
    b_far = [np.full((32, 32), 0.8, np.float32) for _ in range(4)]
    labs = [lab] * 4
    w_close = TE.intensity_w1_per_class(a, labs, b_close, labs, min_px=16)
    w_far = TE.intensity_w1_per_class(a, labs, b_far, labs, min_px=16)
    assert w_close[1] < w_far[1]                     # closer distributions -> smaller W1


def test_label_swap_sensitivity_distinguishes_per_tissue_from_global():
    """The metric must score a label-sensitive translator high (its output changes locally when a
    tissue's label changes) and a label-ignoring global recolor near zero (the one trap)."""
    rng = np.random.default_rng(0)
    img = rng.random((48, 48)).astype(np.float32)
    lab = np.zeros((48, 48), np.int64)
    lab[:24] = 1; lab[24:, :24] = 4; lab[24:, 24:] = 7

    # per-tissue translator: output in a pixel depends on that pixel's label (local, label-driven)
    def per_tissue(im, l):
        return np.clip(im * 0.3 + np.asarray(l, np.float32) / 12.0, 0, 1).astype(np.float32)
    # global recolor: ignores the label entirely
    def global_recolor(im, l):
        return (1.0 - np.asarray(im, np.float32)).astype(np.float32)

    s_pt = TE.label_swap_sensitivity(per_tissue, img, lab, min_px=64)
    s_gl = TE.label_swap_sensitivity(global_recolor, img, lab, min_px=64)
    assert s_pt and s_gl                                   # both measured some classes
    # per-tissue: in-region change is large and localized (out-region ~ 0 -> big locality)
    for c, v in s_pt.items():
        assert v["in_region"] > 0.05 and v["out_region"] < 1e-4
    # global recolor: label swap changes nothing -> in-region ~ 0
    for c, v in s_gl.items():
        assert v["in_region"] < 1e-6
    mean_loc_pt = np.mean([v["locality"] for v in s_pt.values()])
    mean_loc_gl = np.mean([v["locality"] for v in s_gl.values()])
    assert mean_loc_pt > mean_loc_gl                       # per-tissue is far more label-local


def test_temporal_flicker():
    base = np.random.default_rng(0).random((4, 16, 16)).astype(np.float32)
    steady = np.stack([base[0], base[0], base[0]])
    jittery = np.stack([base[0], base[1], base[2]])
    assert TE.temporal_flicker(steady) < TE.temporal_flicker(jittery)
    assert np.isnan(TE.temporal_flicker(base[0:1]))  # need >= 2 frames


def test_summarize_per_class():
    dicts = [{1: 1.0, 4: 2.0}, {1: 3.0}]
    out = TE.summarize_per_class(dicts)
    assert abs(out[1] - 2.0) < 1e-6 and abs(out[4] - 2.0) < 1e-6


def test_tstr_dice_smoke():
    """Train a segmenter on a learnable synthetic set, test on a held-out copy: Dice is finite and
    the harness returns (mean, per_class)."""
    from usbg.render_lotus import get_device
    rng = np.random.default_rng(0)
    H = 40
    def make(n):
        imgs, labs = [], []
        for _ in range(n):
            lab = np.zeros((H, H), np.int64); lab[:H // 2] = 1; lab[H // 2:] = 7
            img = lab.astype(np.float32) / 12 + rng.normal(0, 0.03, (H, H)).astype(np.float32)
            imgs.append(img); labs.append(lab)
        return np.stack(imgs), np.stack(labs)
    Xtr, Ytr = make(8); Xte, Yte = make(3)
    mean_dice, per_class = TE.tstr_dice(Xtr, Ytr, Xte, Yte, device=get_device(),
                                        base=12, depth=2, iters=300)
    assert np.isfinite(mean_dice) and (1 in per_class or 7 in per_class)
    assert mean_dice > 0.5                            # a learnable set transfers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
