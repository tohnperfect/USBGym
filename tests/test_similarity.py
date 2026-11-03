"""Tests for the pluggable B-mode similarity metrics."""
from __future__ import annotations

import numpy as np
import pytest

from usbg.similarity import NCCSimilarity, SSIMSimilarity, EmbeddingSimilarity, load_similarity


@pytest.mark.parametrize("sim", [NCCSimilarity(), SSIMSimilarity()])
def test_self_similarity_is_max_and_range(sim):
    rng = np.random.default_rng(0)
    a = rng.random((160, 120)).astype(np.float32)
    b = rng.random((160, 120)).astype(np.float32)
    s_self, s_diff = sim.score(a, a), sim.score(a, b)
    assert 0.0 <= s_diff <= s_self <= 1.0 + 1e-6
    assert s_self > 0.98                              # identical -> near 1
    assert s_diff < s_self                            # different -> lower


def test_monotone_with_deviation():
    # a shifted/altered image should score lower as it deviates more
    rng = np.random.default_rng(1)
    base = rng.random((128, 128)).astype(np.float32)
    sim = SSIMSimilarity()
    near = np.clip(base + 0.05 * rng.random((128, 128)), 0, 1).astype(np.float32)
    far = np.clip(base + 0.5 * rng.random((128, 128)), 0, 1).astype(np.float32)
    assert sim.score(base, near) > sim.score(base, far)


def test_embedding_similarity_identical():
    from usbg.similarity import build_bmode_encoder
    enc = build_bmode_encoder(embed_dim=32).eval()
    sim = EmbeddingSimilarity(enc, device="cpu", size=64)
    a = np.random.default_rng(2).random((100, 80)).astype(np.float32)
    assert sim.score(a, a) > 0.98                     # cosine of identical embeddings -> ~1


def test_load_similarity_dispatch():
    assert isinstance(load_similarity("ncc"), NCCSimilarity)
    assert isinstance(load_similarity("ssim"), SSIMSimilarity)
    with pytest.raises(ValueError):
        load_similarity("nope")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
