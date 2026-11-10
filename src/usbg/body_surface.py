"""body_surface.py: turn a labeled CT into a body-skin surface mesh (and a generic torso).

The gym's contact surface used to be a flat box. This module extracts the REAL patient skin
from a CT: the outer boundary of everything that is not air (``labels != BACKGROUND``) is the
body surface. Because those vertices come from the same voxel grid and affine as the anatomy the
slicer samples, a mesh built here and the CT it came from live in ONE frame -- so when the mesh
is used as the MuJoCo contact geometry, "register the CT to the body" is the identity (Variant A).

It also builds a deterministic generic torso (``procedural_torso_mesh``) as the offline default
for the external-avatar variant, and surface-sampling helpers for CT<->avatar registration
(register_body.py, Variant B).

Everything is in METRES, using ``LabeledVolume.affine_m`` (voxel -> world). Meshes are plain
(verts (N,3) float, faces (M,3) int) so callers stay independent of trimesh; save/load/sample go
through trimesh where a mesh library genuinely helps.
"""
from __future__ import annotations

import os

import numpy as np

from .volume import BACKGROUND


# ======================================================================================
# CT skin extraction
# ======================================================================================
def body_mask(lv, *, largest_cc: bool = True, fill_holes: bool = True,
              close_iters: int = 1) -> np.ndarray:
    """Boolean (I,J,K) mask of the body: not air, largest connected component, cavities filled.

    Filling holes matters: bowel gas / lumen voxels are BACKGROUND *inside* the body, and without
    filling they would spawn internal surfaces when meshed. Lungs are labelled LUNG (not
    background), so they are already solid. Keeping only the largest component drops the CT table
    and stray high-HU speckle that the HU fill can pick up as fat/soft tissue.
    """
    from scipy import ndimage as ndi

    mask = np.asarray(lv.labels) != BACKGROUND
    if close_iters:
        mask = ndi.binary_closing(mask, iterations=int(close_iters))
    if largest_cc:
        lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
        if n > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0                                   # background label
            mask = lab == int(counts.argmax())
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    return mask


def extract_skin_mesh(lv, *, step_size: int = 2, smooth_iters: int = 12,
                      decimate_to: int | None = None, **mask_kw):
    """Marching-cubes skin surface of a LabeledVolume, in world metres.

    lv          : volume.LabeledVolume (labels + affine_m).
    step_size   : marching-cubes stride; >1 coarsens the mesh (fewer triangles) with no extra deps.
    smooth_iters: Taubin smoothing passes (shrink-free) to take the voxel staircase off the skin.
    decimate_to : optional target face count if a quadric-decimation backend is installed; skipped
                  (with the step_size mesh returned as-is) when none is available.
    mask_kw     : forwarded to body_mask (largest_cc, fill_holes, close_iters).

    Returns (verts_m (N,3) float64, faces (M,3) int64).
    """
    from skimage import measure

    mask = body_mask(lv, **mask_kw)
    # level 0.5 -> the isosurface between body (1) and air (0); verts are (i,j,k) voxel coords.
    verts_vox, faces, _normals, _vals = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, step_size=int(step_size))
    A = np.asarray(lv.affine_m, dtype=np.float64)
    verts_m = verts_vox @ A[:3, :3].T + A[:3, 3]            # voxel -> world metres

    verts_m, faces = _postprocess(verts_m, faces, smooth_iters=smooth_iters,
                                  decimate_to=decimate_to)
    return verts_m, faces


def _postprocess(verts, faces, *, smooth_iters: int = 0, decimate_to: int | None = None):
    """Taubin-smooth and (best-effort) decimate a mesh via trimesh; returns (verts, faces)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces), process=True)
    if smooth_iters:
        # Taubin (lambda/mu) smoothing does not shrink the volume the way plain Laplacian does.
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iters))
    if decimate_to and len(mesh.faces) > decimate_to:
        try:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=int(decimate_to))   # trimesh >= 4.5
            except TypeError:
                mesh = mesh.simplify_quadric_decimation(int(decimate_to))              # older trimesh
        except Exception:
            pass                                            # no backend (open3d/fast-simplification)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


# ======================================================================================
# Mesh IO + surface sampling (used by the scene generator and by registration)
# ======================================================================================
def save_mesh(verts, faces, path: str) -> str:
    """Write a mesh to disk (format from the extension: .stl, .obj, .ply)."""
    import trimesh

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                    faces=np.asarray(faces), process=False).export(path)
    return path


def load_mesh(path: str):
    """Load a mesh file into (verts_m (N,3), faces (M,3)). Keeps the largest component."""
    import trimesh

    m = trimesh.load(path, process=True, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def sample_surface(verts, faces, n: int = 20000, seed: int = 0) -> np.ndarray:
    """Uniformly sample n points (metres) on a mesh surface (for point-cloud registration)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces), process=False)
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, int(n), seed=int(rng.integers(1 << 31)))
    return np.asarray(pts, dtype=np.float64)


def skin_point_cloud(lv, n: int = 20000, seed: int = 0, **kw) -> np.ndarray:
    """Convenience: sample n points on the CT skin surface (metres). Variant-B source cloud."""
    v, f = extract_skin_mesh(lv, **kw)
    return sample_surface(v, f, n=n, seed=seed)


def _farthest_point_subsample(pts: np.ndarray, m: int) -> np.ndarray:
    """Greedy farthest-point subsample to m points (keeps the extreme/spread points)."""
    n = len(pts)
    if n <= m:
        return pts
    idx = [0]
    d = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(m - 1):
        j = int(d.argmax())
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return pts[idx]


def convex_slabs(verts, *, axis: int = 2, k: int = 12, overlap: float = 0.5,
                 min_pts: int = 8, max_verts: int = 96) -> list[np.ndarray]:
    """Decompose a body mesh into K convex "slabs" along its long (superior-inferior) axis.

    Why: in MuJoCo a single ``type="mesh"`` collision geom only collides against the mesh's
    CONVEX HULL, and one hull of a whole torso bridges the waist taper -- so the flanks (which we
    DO scan) would have the wrong contact geometry. But a torso is convex in every axial
    cross-section and non-convex only along its length, so a stack of per-band convex hulls
    reconstructs the tapered barrel faithfully on the anterior surface AND both flanks (the
    posterior spinal groove is the only over-filled region, and we never scan it). Uses only
    numpy + scipy, is deterministic, and gives MuJoCo its best-conditioned convex-vs-convex path.

    verts   : (N,3) mesh vertices in the mesh frame (metres).
    axis    : long axis to band along (2 == z, the S-I axis in the RAS-oriented meshes here).
    k       : number of bands.
    overlap : fraction of a band height each band is grown on each side, so a probe straddling a
              seam still lands on a slab (seam double-contact is bounded and summed downstream).
    min_pts : bands with fewer points than this are skipped (degenerate hull guard).

    Returns a list of (Mi,3) arrays: the hull-vertex positions of each slab (MuJoCo rebuilds the
    hull from these points, so faces are not needed).
    """
    from scipy.spatial import ConvexHull, QhullError

    v = np.asarray(verts, dtype=np.float64)
    lo, hi = float(v[:, axis].min()), float(v[:, axis].max())
    edges = np.linspace(lo, hi, k + 1)
    band_h = (hi - lo) / k
    pad = overlap * band_h
    slabs: list[np.ndarray] = []
    for i in range(k):
        a, b = edges[i] - pad, edges[i + 1] + pad
        sel = (v[:, axis] >= a) & (v[:, axis] <= b)
        pts = v[sel]
        if len(pts) < min_pts:
            continue
        try:
            hull = ConvexHull(pts)
        except QhullError:
            continue                                        # coplanar / degenerate band
        hv = pts[hull.vertices]
        if max_verts and len(hv) > max_verts:
            hv = _farthest_point_subsample(hv, max_verts)   # cap inline MJCF size
        slabs.append(hv)
    return slabs


def mesh_metrics(verts, faces) -> dict:
    """Small provenance/QC dict: vertex/face counts, bbox (m), watertightness."""
    import trimesh

    v = np.asarray(verts, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=v, faces=np.asarray(faces), process=False)
    lo, hi = v.min(0), v.max(0)
    return {"n_verts": int(len(v)), "n_faces": int(len(faces)),
            "bbox_min_m": [float(x) for x in lo], "bbox_max_m": [float(x) for x in hi],
            "extent_m": [float(x) for x in (hi - lo)], "watertight": bool(mesh.is_watertight)}


# ======================================================================================
# Procedural generic torso (offline default avatar; superseded by a real CC0 / SMPL mesh)
# ======================================================================================
# Anterior faces +y, superior +z, right +x -- the same RAS convention as the reoriented CT, so
# the scene generator seats CT-derived and avatar meshes with one shared world_from_body rule.
#
# Profile: semi-axes (left-right a, front-back b) as a function of height z from hips (0) to neck.
# A handful of control levels lofted with a smooth (pchip) interpolation gives a waist, a chest,
# and shoulder taper without importing anybody's scan.
_TORSO_PROFILE = [
    # z_frac,  a (R-L semi-axis),  b (A-P semi-axis)
    (0.00, 0.150, 0.115),   # hips
    (0.18, 0.135, 0.100),   # lower abdomen
    (0.34, 0.120, 0.092),   # waist (narrowest)
    (0.52, 0.140, 0.105),   # lower ribs
    (0.70, 0.165, 0.115),   # chest
    (0.86, 0.180, 0.108),   # shoulders (widest L-R)
    (1.00, 0.090, 0.075),   # base of neck (taper in)
]


def procedural_torso_mesh(height_m: float = 0.62, n_rings: int = 48, n_theta: int = 64,
                          cap_ends: bool = True):
    """A smooth generic upper-body torso as (verts_m, faces), lofted from elliptical rings.

    Deterministic (no RNG, no download). Dimensions are adult-torso plausible and roughly match
    the height of an abdominal CT so the same camera/seating framing works for both avatars.
    """
    from scipy.interpolate import PchipInterpolator

    prof = np.asarray(_TORSO_PROFILE, dtype=np.float64)
    fa = PchipInterpolator(prof[:, 0], prof[:, 1])
    fb = PchipInterpolator(prof[:, 0], prof[:, 2])

    zf = np.linspace(0.0, 1.0, n_rings)
    z = zf * height_m
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    verts = []
    for zi, zval in zip(zf, z):
        a, b = float(fa(zi)), float(fb(zi))
        ring = np.stack([a * ct, b * st, np.full_like(ct, zval)], axis=1)   # x=R-L, y=A-P, z=S-I
        verts.append(ring)
    verts = np.concatenate(verts, axis=0)

    faces = []
    for r in range(n_rings - 1):
        base = r * n_theta
        nxt = base + n_theta
        for t in range(n_theta):
            t2 = (t + 1) % n_theta
            faces.append([base + t, nxt + t, nxt + t2])
            faces.append([base + t, nxt + t2, base + t2])
    faces = np.asarray(faces, dtype=np.int64)

    if cap_ends:
        verts, faces = _cap_tube_ends(verts, faces, n_rings, n_theta)
    return verts.astype(np.float64), faces


def _cap_tube_ends(verts, faces, n_rings, n_theta):
    """Close the bottom and top rings with a centre fan so the torso is watertight."""
    faces = list(map(list, faces))
    bottom_center = len(verts)
    top_center = len(verts) + 1
    b0 = verts[:n_theta].mean(0)
    t0 = verts[(n_rings - 1) * n_theta:].mean(0)
    verts = np.vstack([verts, b0, t0])
    base_top = (n_rings - 1) * n_theta
    for t in range(n_theta):
        t2 = (t + 1) % n_theta
        faces.append([bottom_center, t2, t])                       # bottom fan (inward normal ok)
        faces.append([top_center, base_top + t, base_top + t2])    # top fan
    return verts, np.asarray(faces, dtype=np.int64)


# ======================================================================================
# v2.1.2 (opt-in): the skin-tangent contact plane of a start pose
# ======================================================================================
# Fixed by the pre-specified protocol before any arm-B output exists:
#   point p   the first point at which the probe's beam axis enters the CT body (body_mask), for the start pose
#             as sampled (no lift yet); the search runs along the axis and begins 0.10 m behind the probe face;
#             when the axis never meets the body, the skin-mesh vertex closest to the face centre serves as p;
#   normal n  mean of the outward normals of the skin-mesh faces less than 10 mm from p, weighted by face
#             area (extract_skin_mesh defaults);
#   frame     mask and mesh are in volume metres (affine_m); world_from_volume maps them to the world frame
#             of the episode; one cache per patient.
# Readings fixed in code: the body is entered where the trilinear interpolation of the body
# mask first crosses 0.5 upward (the iso-level marching cubes uses), marched in 0.1 mm steps and refined by
# bisection to 1e-3 mm; "within 10 mm" is the exact point-triangle distance (a KD-tree on face centroids only
# pre-selects candidates; rtree is not installed, so trimesh's proximity queries are not used); the mesh winding is
# not trusted for the side (marching cubes gives inward normals on these CTs), so n is made outward by a mask test
# (mask at p + 3 mm n below the mask at p - 3 mm n), with the beam rule (n against the beam) only when the test
# cannot tell.
PLANE_SEARCH_BACK_M = 0.10
PLANE_SEARCH_FWD_M = 0.30
PLANE_STEP_M = 1e-4
PLANE_BISECT_TOL_M = 1e-6
PLANE_RADIUS_M = 0.010
PLANE_ISO = 0.5
PLANE_ORIENT_M = 0.003

_SKIN_MODELS: dict = {}


def trilinear_mask(mask, ijk):
    """Trilinear interpolation of a boolean (I,J,K) mask at float voxel coordinates ijk (N,3); 0 outside the grid."""
    ijk = np.asarray(ijk, np.float64).reshape(-1, 3)
    shp = np.asarray(mask.shape)
    i0 = np.floor(ijk).astype(np.int64)
    f = ijk - i0
    out = np.zeros(len(ijk), np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + np.array([dx, dy, dz])
                w = ((f[:, 0] if dx else 1.0 - f[:, 0]) * (f[:, 1] if dy else 1.0 - f[:, 1])
                     * (f[:, 2] if dz else 1.0 - f[:, 2]))
                ok = np.all((idx >= 0) & (idx < shp), axis=1)
                v = np.zeros(len(ijk), np.float64)
                v[ok] = mask[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
                out += w * v
    return out


def closest_points_on_triangles(p, tri):
    """Closest point of each triangle tri (M,3,3) to the point p (3,): the region test of Ericson, Real-Time
    Collision Detection, 5.1.5, vectorized. Returns (M,3)."""
    p = np.asarray(p, np.float64).reshape(3)
    tri = np.asarray(tri, np.float64).reshape(-1, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    res = np.empty_like(a)
    done = np.zeros(len(a), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (d1 <= 0) & (d2 <= 0)
        res[m] = a[m]; done |= m
        m = ~done & (d3 >= 0) & (d4 <= d3)
        res[m] = b[m]; done |= m
        m = ~done & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        res[m] = a[m] + (d1[m] / (d1[m] - d3[m]))[:, None] * ab[m]; done |= m
        m = ~done & (d6 >= 0) & (d5 <= d6)
        res[m] = c[m]; done |= m
        m = ~done & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        res[m] = a[m] + (d2[m] / (d2[m] - d6[m]))[:, None] * ac[m]; done |= m
        m = ~done & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        w = (d4[m] - d3[m]) / ((d4[m] - d3[m]) + (d5[m] - d6[m]))
        res[m] = b[m] + w[:, None] * (c[m] - b[m]); done |= m
        m = ~done
        den = 1.0 / (va[m] + vb[m] + vc[m])
        res[m] = a[m] + ab[m] * (vb[m] * den)[:, None] + ac[m] * (vc[m] * den)[:, None]
    bad = ~np.all(np.isfinite(res), axis=1)                 # degenerate (zero-area) faces: nearest vertex
    if bad.any():
        k = np.argmin(np.linalg.norm(tri[bad] - p[None, None, :], axis=2), axis=1)
        res[bad] = tri[bad][np.arange(int(bad.sum())), k]
    return res


def _case_key(lv):
    meta = getattr(lv, "meta", None) or {}
    return (meta.get("case") or f"id{id(lv)}", tuple(np.asarray(lv.labels).shape),
            np.asarray(lv.affine_m, np.float64).tobytes())


class SkinModel:
    """One patient's body mask, skin mesh (extract_skin_mesh defaults, volume metres) and face index, built once and
    cached per patient (skin_model)."""

    def __init__(self, lv):
        from scipy.spatial import cKDTree
        self.case = (getattr(lv, "meta", None) or {}).get("case")
        self.mask = body_mask(lv)
        self.A = np.asarray(lv.affine_m, np.float64)
        self.Ainv = np.linalg.inv(self.A)
        self.verts, self.faces = extract_skin_mesh(lv)
        self.tri = self.verts[self.faces]
        self.area_normal = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        self.centroids = self.tri.mean(axis=1)
        self.max_circ = float(np.linalg.norm(self.tri - self.centroids[:, None, :], axis=2).max())
        self.face_tree = cKDTree(self.centroids)
        self.vert_tree = cKDTree(self.verts)

    def mask_at(self, pts_v):
        """Trilinear body-mask value at volume-metre points (N,3)."""
        pts_v = np.asarray(pts_v, np.float64).reshape(-1, 3)
        M = self.Ainv[:3, :3]
        ijk = (pts_v[:, None, :] * M[None, :, :]).sum(axis=2) + self.Ainv[:3, 3]   # elementwise (no BLAS)
        return trilinear_mask(self.mask, ijk)

    def beam_entry(self, o_v, b_v):
        """(p_v, t_m) where the axis o_v + t b_v first enters the body for t from -0.10 m on, or None."""
        ts = np.arange(-PLANE_SEARCH_BACK_M, PLANE_SEARCH_FWD_M + 0.5 * PLANE_STEP_M, PLANE_STEP_M)
        f = self.mask_at(o_v[None, :] + ts[:, None] * b_v[None, :])
        up = np.flatnonzero((f[:-1] < PLANE_ISO) & (f[1:] >= PLANE_ISO))
        if len(up) == 0:
            return None
        lo, hi = float(ts[up[0]]), float(ts[up[0] + 1])
        while hi - lo > PLANE_BISECT_TOL_M:
            mid = 0.5 * (lo + hi)
            if self.mask_at(o_v + mid * b_v)[0] >= PLANE_ISO:
                hi = mid
            else:
                lo = mid
        return o_v + hi * b_v, hi

    def faces_within(self, p_v, radius=PLANE_RADIUS_M):
        """Indices of the faces whose exact point-triangle distance to p_v is <= radius."""
        cand = np.asarray(self.face_tree.query_ball_point(p_v, radius + self.max_circ), dtype=np.int64)
        if len(cand) == 0:
            return cand
        d = np.linalg.norm(closest_points_on_triangles(p_v, self.tri[cand]) - p_v[None, :], axis=1)
        return np.sort(cand[d <= radius])

    def distance_to_mesh(self, p_v, radius=0.02):
        """Exact distance (m) from p_v to the nearest face within radius + the largest face size."""
        cand = np.asarray(self.face_tree.query_ball_point(p_v, radius + self.max_circ), dtype=np.int64)
        if len(cand) == 0:
            return float("inf")
        return float(np.linalg.norm(closest_points_on_triangles(p_v, self.tri[cand]) - p_v[None, :], axis=1).min())

    def outward_normal(self, p_v, b_v, radius=PLANE_RADIUS_M):
        """(n_v unit outward, number of faces used, orientation source)."""
        idx = self.faces_within(p_v, radius)
        if len(idx) == 0:
            raise ValueError(f"no skin-mesh face within {radius * 1000:.0f} mm of the contact point")
        s = self.area_normal[idx].sum(axis=0)
        n = s / np.linalg.norm(s)
        out_v, in_v = self.mask_at(np.stack([p_v + PLANE_ORIENT_M * n, p_v - PLANE_ORIENT_M * n]))
        if out_v < in_v:
            return n, int(len(idx)), "mask"
        if out_v > in_v:
            return -n, int(len(idx)), "mask"
        return (n if float(n @ b_v) < 0 else -n), int(len(idx)), "beam"


def skin_model(lv, *, cache=True):
    """The SkinModel of lv's patient, built once per patient (keyed by case id, grid shape and affine)."""
    key = _case_key(lv)
    if cache and key in _SKIN_MODELS:
        return _SKIN_MODELS[key]
    m = SkinModel(lv)
    if cache:
        _SKIN_MODELS[key] = m
    return m


def clear_skin_cache():
    _SKIN_MODELS.clear()


def skin_plane(lv, world_from_volume, T_start_prelift, *, model=None):
    """The skin-tangent contact plane of a start pose (the pre-specified protocol): (p_world (3,), n_world (3,) unit
    outward, info). info = {point_source "beam_entry" | "nearest_vertex", entry_t_mm (signed distance of p along
    the beam from the face; None for the vertex fallback), n_faces, orientation "mask" | "beam", case}."""
    sm = model if model is not None else skin_model(lv)
    Wfv = np.asarray(world_from_volume, np.float64)
    Vfw = np.linalg.inv(Wfv)
    T = np.asarray(T_start_prelift, np.float64)
    o_v = Vfw[:3, :3] @ T[:3, 3] + Vfw[:3, 3]
    b_v = Vfw[:3, :3] @ T[:3, 2]
    b_v = b_v / np.linalg.norm(b_v)
    hit = sm.beam_entry(o_v, b_v)
    if hit is not None:
        p_v, t = hit
        src, t_mm = "beam_entry", float(t * 1000.0)
    else:
        p_v = sm.verts[int(sm.vert_tree.query(o_v)[1])].copy()
        src, t_mm = "nearest_vertex", None
    n_v, nf, osrc = sm.outward_normal(p_v, b_v)
    p_w = Wfv[:3, :3] @ p_v + Wfv[:3, 3]
    n_w = Wfv[:3, :3] @ n_v
    n_w = n_w / np.linalg.norm(n_w)
    return p_w, n_w, {"point_source": src, "entry_t_mm": t_mm, "n_faces": nf, "orientation": osrc,
                      "case": sm.case, "p_volume_m": [float(x) for x in p_v]}
