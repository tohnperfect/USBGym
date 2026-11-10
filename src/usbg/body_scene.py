"""body_scene.py: a human body as the probe's contact surface (opt-in), and a viewable MuJoCo scene of it.

By default the probe presses a box whose top face is the skin's tangent plane at the target window (mujoco_env).
This module replaces that box by a body surface:

  "ct_skin" : the patient's own skin, the outer surface of the CT (body_surface.skin_model). It comes from the voxel
              grid the slicer images, so it fits the anatomy exactly.
  "avatar"  : a generic adult torso shipped with the package (data/avatar_torso.obj, CC0 1.0, see NOTICE), resized
              and moved onto the patient's CT skin by a similarity fit (register_body.register_ct_to_avatar from the
              identity-rotation start). The fit leaves a residual (BodyGeometry.info["reg_rms_mm"], tens of mm), so
              under the probe the avatar skin can lie above or below the CT skin: contact then happens on a surface
              that is not the imaged one. BodySurface.gap_mm measures it at a pose.

The CT keeps its rigid seat (placement.seat_volume_under_probe): a body never moves or scales the anatomy.

MuJoCo collides a mesh geom through its convex hull only, and one hull of a torso would bridge the waist, so a body
collides through K convex slabs stacked along its superior-inferior axis (body_surface.convex_slabs). The press cap
and the start lift use the same slabs (BodySurface.surface_plane).

The CT skin and the avatar need the `body` extra (`pip install -e ".[body]"`: scikit-image, trimesh,
fast-simplification). Those packages are imported inside the functions, so importing this module needs none of them.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

SURFACES = ("box", "ct_skin", "avatar")   # "box" is the default flat contact surface of mujoco_env
BODY_KINDS = ("ct_skin", "avatar")
SLAB_K = 12                 # convex collision slabs along the superior-inferior axis
SLAB_OVERLAP = 0.5          # each slab grown by half a band on both sides (body_surface.convex_slabs)
SLAB_MAX_VERTS = 96
VISUAL_FACES = 20000        # face budget of the drawn mesh in a scene file
AVATAR_FILE = "avatar_torso.obj"
REG_SRC_POINTS = 4000       # CT skin samples of the avatar fit
REG_DST_POINTS = 6000       # avatar surface samples of the avatar fit
_GEOMS: dict = {}


def avatar_path() -> str:
    """Path of the packaged avatar torso: Wavefront OBJ in metres, +x right, +y anterior, +z superior."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", AVATAR_FILE)


def load_avatar(path=None):
    """(verts (N, 3) metres, faces (M, 3)) of an avatar torso mesh; default the packaged one."""
    from . import body_surface as B
    return B.load_mesh(path or avatar_path())


def _apply(T, pts) -> np.ndarray:
    T = np.asarray(T, np.float64)
    return np.asarray(pts, np.float64) @ T[:3, :3].T + T[:3, 3]


@dataclass
class BodyGeometry:
    """One patient's body in its own frame.

    kind             : "ct_skin" or "avatar".
    verts, faces     : the body mesh in metres (drawn in scene files; collision uses the slabs).
    slabs            : list of (Mi, 3) point sets, one convex collision slab each.
    volume_from_body : 4x4 body frame -> CT volume frame: the identity for the CT skin, the inverse of the
                       similarity fit for the avatar (so it can scale).
    info             : case, n_slabs, n_faces; for the avatar also mesh, reg_rms_mm (residual of the fit, in the
                       avatar's frame) and reg_scale (scale of the CT-to-avatar fit; the avatar is drawn at 1/scale).
    """
    kind: str
    verts: np.ndarray
    faces: np.ndarray
    slabs: list
    volume_from_body: np.ndarray
    info: dict = field(default_factory=dict)


def _slabs(verts, k):
    from . import body_surface as B
    return B.convex_slabs(verts, axis=2, k=int(k), overlap=SLAB_OVERLAP, max_verts=SLAB_MAX_VERTS)


def ct_body(lv, *, k=SLAB_K, cache=True) -> BodyGeometry:
    """The patient's CT skin as a BodyGeometry (the mesh of body_surface.skin_model, shared with skin_plane)."""
    from . import body_surface as B
    key = ("ct_skin", B._case_key(lv), int(k))
    if cache and key in _GEOMS:
        return _GEOMS[key]
    sm = B.skin_model(lv, cache=cache)
    slabs = _slabs(sm.verts, k)
    g = BodyGeometry("ct_skin", sm.verts, sm.faces, slabs, np.eye(4),
                     {"case": sm.case, "n_slabs": len(slabs), "n_faces": int(len(sm.faces))})
    if cache:
        _GEOMS[key] = g
    return g


def avatar_body(lv, *, mesh=None, k=SLAB_K, icp_iters=6, seed=0, cache=True) -> BodyGeometry:
    """The avatar torso (default: the packaged one; mesh = any OBJ/STL/PLY torso in metres with +y anterior and +z
    superior) fitted to the patient's CT skin. The fit maps CT skin samples onto avatar surface samples
    (register_body.register_ct_to_avatar, init "identity": a free rotation can turn a near-symmetric torso
    back to front); its inverse places the avatar in the CT's frame."""
    from . import body_surface as B
    from . import register_body as RB
    path = os.path.abspath(mesh or avatar_path())
    key = ("avatar", B._case_key(lv), path, int(k), int(icp_iters), int(seed))
    if cache and key in _GEOMS:
        return _GEOMS[key]
    av_v, av_f = load_avatar(path)
    sm = B.skin_model(lv, cache=cache)
    src = B.sample_surface(sm.verts, sm.faces, n=REG_SRC_POINTS, seed=seed)
    dst = B.sample_surface(av_v, av_f, n=REG_DST_POINTS, seed=seed)
    afv, reg = RB.register_ct_to_avatar(src, dst, init="identity", icp_iters=int(icp_iters), seed=int(seed))
    slabs = _slabs(av_v, k)
    info = {"case": sm.case, "n_slabs": len(slabs), "n_faces": int(len(av_f)), "mesh": os.path.basename(path),
            "reg_rms_mm": float(reg["rms_m"] * 1000.0), "reg_scale": float(reg["scale"])}
    g = BodyGeometry("avatar", av_v, av_f, slabs, np.linalg.inv(afv), info)
    if cache:
        _GEOMS[key] = g
    return g


def body_geometry(lv, kind, *, mesh=None, k=SLAB_K) -> BodyGeometry:
    """ct_body or avatar_body by name ("ct_skin" | "avatar")."""
    if kind == "ct_skin":
        return ct_body(lv, k=k)
    if kind == "avatar":
        return avatar_body(lv, mesh=mesh, k=k)
    raise ValueError(f"body kind {kind!r}: expected one of {BODY_KINDS}")


def clear_cache():
    _GEOMS.clear()


def _fmt(a, spec="{:.5f}") -> str:
    return " ".join(map(spec.format, np.asarray(a).ravel()))


class BodySurface:
    """A BodyGeometry placed in the MuJoCo world: the slabs (and the mesh) in world coordinates, and the facet planes
    of the slabs for the press cap and the start lift. Build it with place()."""

    def __init__(self, geom: BodyGeometry, world_from_body):
        from scipy.spatial import ConvexHull
        self.geom = geom
        self.kind = geom.kind
        self.world_from_body = np.asarray(world_from_body, np.float64)
        self.slabs, self._eqs = [], []
        for s in geom.slabs:
            p = _apply(self.world_from_body, s)
            h = ConvexHull(p)
            self.slabs.append(p[h.vertices])
            self._eqs.append(np.asarray(h.equations, np.float64))     # rows (n, b): n . x + b <= 0 inside, |n| = 1
        if not self.slabs:
            raise ValueError("the body has no collision slab")

    def surface_plane(self, T):
        """(p, n) of the body surface at the probe face centre c of pose T: n is the outward normal of the slab
        facet that bounds c most tightly, p the projection of c on that facet's plane. Per slab the signed distance
        is max_j (n_j . c + b_j) (exact inside the slab, a lower bound outside it); the slab with the smallest one
        is the surface the probe touches."""
        c = np.asarray(T, np.float64)[:3, 3]
        best_d, best_n = np.inf, None
        for eq in self._eqs:
            d = eq[:, :3] @ c + eq[:, 3]
            j = int(np.argmax(d))
            if d[j] < best_d:
                best_d, best_n = float(d[j]), eq[j, :3].copy()
        return c - best_d * best_n, best_n

    def clamp(self, T, min_clear_m=0.0):
        """placement.clamp_start_outside_plane on surface_plane(T): (T_clamped, lift_m). min_clear_m = 0 lifts a
        start out of the body; -max_press_m is the press cap."""
        from .placement import clamp_start_outside_plane
        p, n = self.surface_plane(T)
        return clamp_start_outside_plane(T, p, n, min_clear_m=min_clear_m)

    def gap_mm(self, T) -> float:
        """Signed height (mm) of the probe face centre of T above the body surface: + outside, - pressed in."""
        p, n = self.surface_plane(T)
        return float(n @ (np.asarray(T, np.float64)[:3, 3] - p)) * 1000.0

    def visual_mesh(self, max_faces=VISUAL_FACES):
        """(verts, faces) of the drawn mesh in world coordinates, decimated to about max_faces (needs
        fast-simplification; without it the full mesh is returned with a warning)."""
        from . import body_surface as B
        v, f = self.geom.verts, self.geom.faces
        if len(f) > max_faces:
            v, f = B._postprocess(v, f, decimate_to=int(max_faces))
            if len(f) > 2 * max_faces:
                warnings.warn(f"mesh not decimated ({len(f)} faces): install fast-simplification for a smaller "
                              "scene file", stacklevel=2)
        v, f = np.asarray(v, np.float64), np.asarray(f, np.int64)
        tri = v[f]
        if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
            f = f[:, ::-1]        # marching cubes winds the skin inward; the viewer shades the outward side
        return _apply(self.world_from_body, v), f

    def mjcf(self, *, visual=False, visual_faces=VISUAL_FACES, rgba="0.85 0.62 0.55 1"):
        """(assets_xml, patient_xml) for mujoco_env._mjcf: one mesh asset and one static collision geom per slab,
        named "patient", "patient_1", ... with the box's contact parameters. With visual=True the body mesh is added
        as a drawn geom that does not collide, and the slabs move to geom group 3 (hidden in the MuJoCo viewer until
        that group is switched on)."""
        from .mujoco_env import PATIENT_SOLREF, PATIENT_SOLIMP
        assets = []
        geoms = [f"    <!-- patient: {self.kind} body surface, {len(self.slabs)} convex collision slabs -->"]
        group, slab_rgba = ("3", "0.3 0.6 0.9 0.35") if visual else ("0", rgba)
        for k, s in enumerate(self.slabs):
            name = "patient" if k == 0 else f"patient_{k}"
            assets.append(f'    <mesh name="{name}_hull" vertex="{_fmt(s)}"/>')
            geoms.append(f'    <geom name="{name}" type="mesh" mesh="{name}_hull" group="{group}" rgba="{slab_rgba}" '
                         f'solref="{PATIENT_SOLREF}" solimp="{PATIENT_SOLIMP}"/>')
        if visual:
            v, f = self.visual_mesh(visual_faces)
            assets.append(f'    <mesh name="body_visual" inertia="shell" vertex="{_fmt(v)}" '
                          f'face="{_fmt(f, "{:d}")}"/>')
            geoms.append(f'    <geom name="body_visual" type="mesh" mesh="body_visual" contype="0" conaffinity="0" '
                         f'rgba="{rgba}"/>')
        return "\n  <asset>\n" + "\n".join(assets) + "\n  </asset>", "\n".join(geoms)


def place(geom: BodyGeometry, world_from_volume) -> BodySurface:
    """Put a body into the world of a seat: world_from_body = world_from_volume @ geom.volume_from_body."""
    return BodySurface(geom, np.asarray(world_from_volume, np.float64) @ geom.volume_from_body)


def _quat(R) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF xyaxes of a camera at eye looking at target (the camera looks along its -z)."""
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _fmt(np.concatenate([x, y]), "{:.4f}")


def scene_xml(lv, world_from_volume, T_target, kind="ct_skin", *, visual=True, mesh=None, k=SLAB_K):
    """A self-contained MJCF scene (inline meshes) of the probe resting on a body, and its record.

    T_target is a probe pose on the skin; placement.seat_volume_under_probe gives world_from_volume and the pose of a
    screened window. The probe (and the mocap target it is welded to) starts at T_target lifted out of the body
    along the nearest slab facet. A floor under the body, a light and a camera "overview" aimed at the probe are
    added for viewing. Returns (xml, record) with record = {surface, world_from_volume, world_from_body,
    probe_start, start_lift_mm, window_gap_mm (+ = the window face lies outside the body surface), n_slabs, info}."""
    from .mujoco_env import _mjcf
    geom = body_geometry(lv, kind, mesh=mesh, k=k)
    surf = place(geom, world_from_volume)
    T_target = np.asarray(T_target, np.float64)
    T0, lift = surf.clamp(T_target)
    assets_xml, patient_xml = surf.mjcf(visual=visual)
    zmin = min(float(s[:, 2].min()) for s in surf.slabs)
    c = T0[:3, 3]
    eye = c + np.array([0.55, -0.65, 0.45])
    extra = [
        f'    <geom name="floor" type="plane" pos="{c[0]:.4f} {c[1]:.4f} {zmin - 0.002:.4f}" size="1 1 0.05" '
        'rgba="0.25 0.28 0.32 1" contype="0" conaffinity="0"/>',
        f'    <light pos="{c[0]:.4f} {c[1]:.4f} {c[2] + 1.2:.4f}" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>',
        f'    <camera name="overview" pos="{_fmt(eye, "{:.4f}")}" xyaxes="{_look_at(eye, c)}"/>',
    ]
    view = '\n  <visual>\n    <global offwidth="1280" offheight="960"/>\n  </visual>'   # offscreen image size
    xml = _mjcf(assets_xml=view + assets_xml, patient_xml=patient_xml + "\n" + "\n".join(extra))
    pose = f'pos="{_fmt(c, "{:.6f}")}" quat="{_fmt(_quat(T0[:3, :3]), "{:.6f}")}"'
    for old in ('name="target" mocap="true" pos="0 0 0.02"', 'name="probe" pos="0 0 0.02"'):
        if xml.count(old) != 1:
            raise RuntimeError(f"scene template changed: {old!r} not found once")
        xml = xml.replace(old, old.rsplit(" pos=", 1)[0] + " " + pose)
    record = {"surface": kind, "world_from_volume": np.asarray(world_from_volume, np.float64).tolist(),
              "world_from_body": surf.world_from_body.tolist(), "probe_start": T0.tolist(),
              "start_lift_mm": float(lift * 1000.0), "window_gap_mm": surf.gap_mm(T_target),
              "n_slabs": len(surf.slabs), "info": dict(geom.info)}
    return xml, record
