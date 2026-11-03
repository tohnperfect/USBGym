"""Body-surface + registration helpers: pure-numpy checks (no CT, no MuJoCo).

Covers the geometry the body-contact-surface demo relies on: a watertight procedural avatar,
convex-slab decomposition along the long axis, exact recovery of a known similarity by Umeyama,
and the anatomically-anchored (non-flipping) CT->avatar registration on a synthetic torso.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trimesh")

from usbg import body_surface as B
from usbg import register_body as RB


def test_procedural_torso_is_watertight_and_adult_sized():
    v, f = B.procedural_torso_mesh(height_m=0.62)
    m = B.mesh_metrics(v, f)
    assert m["watertight"]
    ext = np.array(m["extent_m"])
    assert 0.55 < ext[2] < 0.7                     # superior-inferior height
    assert 0.25 < ext[0] < 0.45                    # left-right width
    assert 0.18 < ext[1] < 0.30                    # anterior-posterior thickness


def test_convex_slabs_span_axis_and_are_capped():
    v, f = B.procedural_torso_mesh()
    slabs = B.convex_slabs(v, axis=2, k=10, overlap=0.5, max_verts=64)
    assert 8 <= len(slabs) <= 10
    assert all(len(s) <= 64 for s in slabs)        # inline-MJCF vertex cap honoured
    zlo = min(s[:, 2].min() for s in slabs)
    zhi = max(s[:, 2].max() for s in slabs)
    assert zlo <= v[:, 2].min() + 1e-6 and zhi >= v[:, 2].max() - 1e-6


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(150, 3))
    s, th, t = 1.7, 0.6, np.array([0.5, -0.3, 0.2])
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    dst = (s * R @ src.T).T + t
    T = RB.umeyama(src, dst)
    assert np.allclose(RB._apply(T, src), dst, atol=1e-9)
    assert abs(np.cbrt(np.linalg.det(T[:3, :3])) - s) < 1e-6


def test_axis_aligned_registration_does_not_flip_anterior():
    # a small "torso": anterior (+y) tagged points must stay anterior after registration
    rng = np.random.default_rng(1)
    src = rng.normal(size=(400, 3)) * np.array([0.12, 0.09, 0.3])
    dst = src * 1.15 + np.array([0.4, -0.2, 0.1])   # same axes, scaled + shifted (no rotation)
    T, info = RB.register_ct_to_avatar(src, dst, init="identity", icp_iters=4, sample=400)
    R = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    assert ang < 5.0                                # essentially no rotation -> no A/P flip
    assert info["rms_m"] < 0.02
