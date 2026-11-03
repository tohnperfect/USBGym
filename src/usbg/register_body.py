"""register_body.py: align a CT skin point cloud onto an external avatar surface (Variant B).

Variant A (CT-derived body) needs no registration: the contact mesh IS the CT skin, so the
anatomy and the surface share one frame and ``avatar_from_volume`` is the identity.

Variant B uses a generic avatar (procedural torso, or a CC0 / SMPL mesh) whose surface does NOT
match the specific patient. To still image the CT when the probe touches the avatar, we need a
transform ``avatar_from_volume`` that maps the CT (volume) frame onto the avatar frame. We solve a
SIMILARITY transform (rotation + uniform scale + translation) from the two skin point clouds:
first a correspondence-free PCA initialisation, then a few ICP refinements with a closed-form
Umeyama fit. Uniform scale (not full affine) keeps the anatomy undistorted -- the residual
surface-vs-anatomy mismatch is real and is exactly the honest signal the A-vs-B comparison shows.

numpy + scipy only.
"""
from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True) -> np.ndarray:
    """Closed-form least-squares similarity mapping src -> dst for CORRESPONDING points.

    src, dst : (N,3) matched point sets. Returns a 4x4 homogeneous transform T with
    dst ~= (s R) src + t. Umeyama (1991): SVD of the cross-covariance with a determinant sign fix
    so R is a proper rotation, and scale from the ratio of variances.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd.T @ xs) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:            # reflection fix -> proper rotation
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        s = float((D * np.diag(S)).sum() / var_s)
    else:
        s = 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def _pca_axes(pts: np.ndarray):
    """Return (centroid, principal axes as columns sorted by descending extent, sqrt-eigvals)."""
    mu = pts.mean(0)
    c = np.cov((pts - mu).T)
    w, Vv = np.linalg.eigh(c)                               # ascending
    order = np.argsort(w)[::-1]
    return mu, Vv[:, order], np.sqrt(np.clip(w[order], 0, None))


def pca_similarity_init(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Correspondence-free similarity init aligning src's principal axes onto dst's.

    Axis sign is fixed by requiring each aligned src axis to point the same way as its dst axis
    (a PCA eigenvector's sign is arbitrary). Scale is the ratio of principal-extent RMS.
    """
    mu_s, Vs, ws = _pca_axes(src)
    mu_d, Vd, wd = _pca_axes(dst)
    # flip each src axis so, after mapping, it agrees in sign with the matching dst axis
    for i in range(3):
        if np.dot(Vd[:, i], Vs[:, i]) < 0:
            Vs[:, i] = -Vs[:, i]
    R = Vd @ Vs.T
    if np.linalg.det(R) < 0:                                # keep it a proper rotation
        Vs[:, 2] = -Vs[:, 2]
        R = Vd @ Vs.T
    s = float(wd.sum() / max(ws.sum(), 1e-9))
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def axis_aligned_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Uniform-scale + translation, IDENTITY rotation, aligning the two centroids and RMS radii.

    Use this when src and dst already share anatomical axes (both RAS: +x right, +y anterior,
    +z superior), as the CT volume and the RAS-oriented avatar do here. A free PCA rotation would
    happily flip a near-symmetric torso's anterior<->posterior; anchoring the rotation to identity
    makes anterior map to anterior, so pressing the avatar's belly images the CT's belly.
    """
    cs, cd = src.mean(0), dst.mean(0)
    rs = np.sqrt(((src - cs) ** 2).sum() / len(src))
    rd = np.sqrt(((dst - cd) ** 2).sum() / len(dst))
    s = float(rd / max(rs, 1e-9))
    T = np.eye(4)
    T[:3, :3] = s * np.eye(3)
    T[:3, 3] = cd - s * cs
    return T


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def register_ct_to_avatar(src: np.ndarray, dst: np.ndarray, *, icp_iters: int = 6,
                          init: str = "identity", sample: int = 4000,
                          seed: int = 0) -> tuple[np.ndarray, dict]:
    """Similarity transform ``avatar_from_volume`` mapping CT skin (src) onto avatar surface (dst).

    src, dst  : (N,3) / (M,3) skin point clouds in metres (CT and avatar, respectively).
    init      : "identity" (axis-aligned; both clouds are RAS -- the default, and the only one
                that keeps anterior->anterior on a near-symmetric torso) or "pca".
    icp_iters : nearest-neighbour + Umeyama refinements after the init (0 = init only).
    Returns (T 4x4, info) with info = {rms_m, scale, iters}. rms_m is the point-to-nearest RMS
    residual after the last iteration -- the honest measure of surface-vs-anatomy mismatch.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if sample and len(src) > sample:
        src_s = src[rng.choice(len(src), sample, replace=False)]
    else:
        src_s = src

    T = pca_similarity_init(src_s, dst) if init == "pca" else axis_aligned_similarity(src_s, dst)
    tree = cKDTree(dst)
    rms = float(np.sqrt((tree.query(_apply(T, src_s), k=1)[0] ** 2).mean()))
    for _ in range(int(icp_iters)):
        moved = _apply(T, src_s)
        d, idx = tree.query(moved, k=1)
        rms = float(np.sqrt((d ** 2).mean()))
        T = umeyama(src_s, dst[idx], with_scale=True)       # global refit src -> matched dst
    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    return T, {"rms_m": rms, "scale": scale, "iters": int(icp_iters)}
