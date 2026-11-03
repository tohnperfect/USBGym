"""placement.py: put the probe where it will actually see the target anatomy.

Phase 1/2 use these to aim the imaging plane at an organ so slices and renders are meaningful.
Phase 3 (mujoco_env) reuses target_centroid_m to seat the CT under the MuJoCo probe surface.
"""
from __future__ import annotations

import numpy as np

from .contracts_bridge import pose_from_rpy
from .seatmode import seat_mode, SEAT_MODES  # noqa: F401  (re-exported: the one seat switch)
from .windows import NoAcceptedWindow  # noqa: F401  (re-exported for callers and the segmenter repo)

# Supine seat rotation of the arm repo's body scenes (the arm simulator's body-scene script),
# vendored by value: body anterior (+y RAS) -> world +z. For a pure-anterior window whose probe +x is
# the patient's right, seat_from_window yields exactly this rotation (tests/test_placement_anatomical).
R_SEAT = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]], dtype=np.float64)
PROBE_HALF = (0.008, 0.008, 0.012)        # mujoco_env probe housing half-extents (face corners for the clamp)


def target_centroid_m(lv, label: int) -> np.ndarray:
    """World (volume-frame) centroid in metres of a canonical label, via the volume affine."""
    ijk = np.argwhere(lv.labels == label)
    if len(ijk) == 0:
        from .volume import USBG_LABEL_NAMES
        raise ValueError(f"label {label} ({USBG_LABEL_NAMES.get(label, label)}) absent in volume")
    c_vox = ijk.mean(axis=0)
    return (lv.affine_m[:3, :3] @ c_vox) + lv.affine_m[:3, 3]


def base_pose_over(lv, label: int, depth_m: float = 0.03) -> np.ndarray:
    """CANONICAL-ONLY legacy helper (bypasses the seat switch; beam points superior from inside the
    trunk). Use seat_volume_under_probe for anything reported. A probe pose whose central scanline
    passes through the target at ~depth_m.

    With world_from_volume identity (volume frame == world), the probe origin sits at the
    target centroid shifted back along the axial (+z) axis by depth_m, so scanlines run +z
    into the organ and it appears at roughly depth_m.
    """
    c = target_centroid_m(lv, label)
    return pose_from_rpy(tx=float(c[0]), ty=float(c[1]), tz=float(c[2] - depth_m))


def _seat_canonical(lv, label: int, standoff_m: float, depth_m: float):
    """The v1 seat, unchanged: the CT unrotated and the probe turned by 180 deg about x; in an RAS
    volume the beam then points INFERIOR across a coronal plane, with the probe face within the trunk.
    Kept only for USBG_SEAT=canonical."""
    # probe looking down into the tissue: rotate 180 deg about x so +z_probe -> -z_world
    T0 = pose_from_rpy(tx=0.0, ty=0.0, tz=standoff_m, rx=180.0)
    # central scanline point at depth_m in the world:  T0 @ (0,0,depth_m)
    p_world = (T0[:3, :3] @ np.array([0.0, 0.0, depth_m])) + T0[:3, 3]
    centroid = target_centroid_m(lv, label)
    world_from_volume = np.eye(4, dtype=np.float64)          # identity rotation (axis-aligned)
    world_from_volume[:3, 3] = p_world - centroid            # translation only
    return world_from_volume, T0


def seat_from_window(T_vp, standoff_m: float = 0.0):
    """Seat for a screened window T_vp (probe -> volume): returns (world_from_volume, T0) with
    world_from_volume @ T_vp == pose_from_rpy(rx=180), so the window face lands at the world origin
    with the beam on world -z and the MuJoCo box top (z=0) is the skin's tangent plane there.
    T0 = pose_from_rpy(tz=standoff_m, rx=180): + hovers above the skin, - presses into it."""
    Wfv = pose_from_rpy(rx=180.0) @ np.linalg.inv(np.asarray(T_vp, np.float64))
    return Wfv, pose_from_rpy(tz=standoff_m, rx=180.0)


def seat_volume_under_probe(lv, label: int, standoff_m: float | None = None,
                            depth_m: float | None = None, *, window: int = 0):
    """Place the CT under the MuJoCo skin surface (box top at z=0) so the probe images the target.

    Returns (world_from_volume, T0_world_from_probe); T0 always images along world -z with its face
    at (0, 0, standoff_m). The mode comes from $USBG_SEAT (usbg.seatmode, default anatomical):

    anatomical: the probe rests on the skin of the patient at screened acoustic window `window`
        (index into usbg.windows.windows_for(lv, label), taken modulo the count), beam into the body.
        standoff_m defaults to 0. depth_m is canonical-only and raises here (depth is anatomical).
        Raises NoAcceptedWindow when the (patient, organ) has no screened window.
    canonical: the v1 seat (see _seat_canonical); defaults standoff 0.005, depth 0.05.
    """
    if seat_mode() == "canonical":
        return _seat_canonical(lv, label, 0.005 if standoff_m is None else standoff_m,
                               0.05 if depth_m is None else depth_m)
    if depth_m is not None:
        raise ValueError("depth_m is canonical-only; under USBG_SEAT=anatomical the target depth is "
                         "set by the screened acoustic window (drop the argument at the call site)")
    from . import windows as W
    wins = W.windows_for(lv, label)
    if not wins:
        raise NoAcceptedWindow(f"no screened window for label {label} in case {W.case_id(lv)}")
    return seat_from_window(wins[int(window) % len(wins)], 0.0 if standoff_m is None else standoff_m)


def n_windows(lv, label: int) -> int:
    """Number of screened windows for (case, label); 1 under the canonical seat."""
    if seat_mode() == "canonical":
        return 1
    from . import windows as W
    return len(W.windows_for(lv, label))


def clamp_start_outside_skin(T_start, min_clear_m: float = 0.0, probe_half=PROBE_HALF):
    """Lift a start pose along world +z (the skin's outward normal at the window) until the probe
    face centre and the four housing face corners are >= min_clear_m above the box top (z=0), so no
    episode starts with the probe inside the patient. Returns (T_clamped, lifted_m)."""
    bx, by, _ = probe_half
    C = np.array([[0, 0, 0, 1.0]] + [[sx * bx, sy * by, 0.0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
    T = np.asarray(T_start, np.float64).copy()
    zmin = float((T @ C.T)[2].min())
    lift = max(0.0, min_clear_m - zmin)
    T[2, 3] += lift
    return T, lift


# ---- v2.1.2 (opt-in): a contact plane through a point p with unit normal n. No default path
# calls these; clamp_start_outside_skin (the plane z = 0 with normal +z) stays the default everywhere.
def _face_points(T, probe_half=PROBE_HALF):
    """World positions (5, 3) of the probe face centre and its four housing face corners."""
    bx, by, _ = probe_half
    C = np.array([[0, 0, 0, 1.0]] + [[sx * bx, sy * by, 0.0, 1.0] for sx in (-1, 1) for sy in (-1, 1)])
    return (np.asarray(T, np.float64) @ C.T)[:3].T


def plane_gap_m(T, p, n, probe_half=PROBE_HALF):
    """(face-centre height, lowest of the centre and the four face corners) above the plane through p with unit
    normal n, in metres: > 0 = outside (hovering), < 0 = below the plane (pressed in)."""
    p = np.asarray(p, np.float64).reshape(3)
    n = np.asarray(n, np.float64).reshape(3)
    d = (_face_points(T, probe_half) - p[None, :]) @ n
    return float(d[0]), float(d.min())


def clamp_start_outside_plane(T_start, p, n, min_clear_m: float = 0.0, probe_half=PROBE_HALF):
    """The plane-general clamp: translate a pose along the unit normal n until the probe face centre and the four
    housing face corners are >= min_clear_m above the plane through p (signed distance n . (c - p)). The start
    lift uses min_clear_m 0, the press cap min_clear_m = -max_press_m. With p = 0 and n = +z it returns the pose and
    the lift of clamp_start_outside_skin. Returns (T_clamped, lifted_m)."""
    n = np.asarray(n, np.float64).reshape(3)
    T = np.asarray(T_start, np.float64).copy()
    lift = max(0.0, min_clear_m - plane_gap_m(T, p, n, probe_half)[1])
    if lift > 0.0:
        T[:3, 3] += lift * n
    return T, lift
