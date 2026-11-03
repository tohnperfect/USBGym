"""windows.py: screened acoustic windows for the anatomical seat (D4).

A WINDOW is a probe pose in the CT volume frame, T_vp (4x4, probe -> volume, metres), whose face sits
on the patient's own skin (the 0.5 iso-surface of the smoothed body mask), whose beam points into
the body within TILT_MAX_DEG of the inward skin normal, and whose in-plane angle phi is one of PHIS.
placement.seat_from_window turns it into the seat (world_from_volume, T0) so that the MuJoCo box top
(z=0) is the skin's tangent plane at the window and nothing downstream changes.

Screen (all must hold): fat or soft tissue at the face; the central scanline reaches the target with
no obstruction; >= 80% of target scanlines have no bone, lung, GI, gas or out-of-CT sample before the
target; target fraction >= FAMILIES[organ].min_frac; a shadow proxy (cumulative Fresnel transmission x
attenuation x default TGC) >= -20 dB on >= 50% of target pixels; <= 5% of samples outside the CT; and
the CT skin lies between -5 and +10 mm from the tangent plane over a disk of radius FLAT_RADIUS_M. Distinct
windows = deterministic non-maximum suppression with faces >= 20 mm apart, ranked by frac*clean.

Display convention (clinical): image left = probe -x. phi=0 is a long-axis view with probe +x caudal
(cranial on image left); phi=-90 is transverse with probe +x = patient LEFT (patient right on image
left). PHIS sweeps 0 to -150 deg so every window follows the clinical marker convention.

Windows are expensive (40-100 s per case), so scripts/build_windows.py precomputes them into a
versioned JSON (WINDOWS_PATH); windows_for() only reads it, and refuses a stale or missing file.
Volume frame: RAS+ (+x patient right, +y anterior, +z superior), diagonal affines (data/cache*).
Ported from the read-only prototype outputs/v2/_recipes/seat/anat_windows.py.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import volume as V

WINDOWS_VERSION = "usbg-windows-v1"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOWS_PATH = os.environ.get("USBG_WINDOWS",
                              os.path.join(_REPO, "data", "cache", "windows", f"{WINDOWS_VERSION}.json"))

OBSTRUCT = (V.BONE, V.LUNG, V.GI)
FACE_OK = (V.FAT, V.SOFT_TISSUE)
FOV_DEG, APEX_M, N_LINES, N_SAMPLES = 60.0, 0.04, 160, 512      # us_render convex grid
PHIS = (0.0, -30.0, -60.0, -90.0, -120.0, -150.0)
TILT_MAX_DEG = 10.0
NMS_M = 0.02
FLAT_RADIUS_M = 0.02            # = the view-match lateral start perturbation in the anatomical seat
FLAT_BELOW_MM, FLAT_ABOVE_MM = 10.0, 5.0
SCREEN = dict(clean_min=0.8, lit_min=0.5, fov_max=0.05)
FAMILIES = {
    # organ: azimuth range (deg; 0 = anterior, +90 = patient right) and minimum target fraction
    "liver":        dict(th=(-30.0, 75.0), min_frac=0.15),       # anterior / subcostal / anterolateral
    "gallbladder":  dict(th=(-15.0, 80.0), min_frac=0.01),
    "kidney_right": dict(th=(60.0, 135.0), min_frac=0.05),       # right flank, antero- to postero-lateral
    "kidney_left":  dict(th=(-135.0, -60.0), min_frac=0.05),     # left flank
    "spleen":       dict(th=(-135.0, -60.0), min_frac=0.05),     # screened but exposed only with USBG_ALLOW_SPLEEN=1
}
ORGANS_BUILT = ("liver", "gallbladder", "kidney_right", "kidney_left", "spleen")
# Landmarks: the reward's organs (view_match.MATCH_ORGANS). A liver view must also show at least one
# other landmark (vessel / gallbladder / kidney, e.g. portal vein, IVC, Morison's pouch): the liver
# interior is a single label (hepatic veins are not segmented), so a liver-only view has no layout
# structure to match. Windows of every organ are ranked by landmark count, then target coverage.
LANDMARK_LABELS = (V.LIVER, V.GALLBLADDER, V.VESSEL, V.KIDNEY, V.SPLEEN)
LANDMARK_MIN_FRAC = 0.01
MIN_LANDMARKS = {"liver": 1}
_ORGAN_LABEL = {"liver": V.LIVER, "gallbladder": V.GALLBLADDER, "kidney_right": V.KIDNEY,
                "kidney_left": V.KIDNEY, "spleen": V.SPLEEN}

# physics-renderer constants (render_lotus) for the shadow proxy, by canonical label id
_Z = {0: 0.0004, 1: 1.65, 2: 1.53, 3: 1.61, 4: 1.62, 5: 1.60, 6: 1.62, 7: 1.62, 8: 1.62, 9: 1.38,
      10: 7.80, 11: 0.0004, 12: 1.63}
_MU = {0: 1.64, 1: 0.70, 2: 0.10, 3: 0.18, 4: 1.00, 5: 0.70, 6: 0.70, 7: 0.80, 8: 1.09, 9: 0.63,
       10: 5.00, 11: 1.64, 12: 0.54}
_ZT = np.array([_Z[i] for i in range(13)], np.float32)
_MUT = np.array([_MU[i] for i in range(13)], np.float32)
_ATTEN_COEF, _TGC_REF_MU = 0.9, 0.55


class NoAcceptedWindow(LookupError):
    """No screened window exists for this (patient, organ) in the windows file."""


# ------------------------------------------------------------------------------------ geometry
def scan_grid(depth_m=None, n_lines=N_LINES, n_samples=N_SAMPLES):
    """us_render's convex grid with a parametric depth. (L,S,3) probe frame, metres."""
    depth_m = float(V_IMG_DEPTH()) if depth_m is None else depth_m
    depth = np.linspace(0.0, depth_m, n_samples)
    angs = np.deg2rad(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, n_lines))
    r = APEX_M + depth
    X = np.sin(angs)[:, None] * r[None, :]
    Zc = np.cos(angs)[:, None] * r[None, :] - APEX_M
    return np.stack([X, np.zeros_like(X), Zc], -1), depth


def V_IMG_DEPTH():
    from ._reuse import us_render
    return us_render.IMG_DEPTH_M


def rot_about(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def probe_rotation(beam, phi_deg):
    """Columns = probe x, y, z in the volume frame. phi=0: probe +x = caudal (projected on the image
    plane); negative phi rotates +x toward patient LEFT for an anterior window (see module doc)."""
    z = np.asarray(beam, float); z = z / np.linalg.norm(z)
    inf = np.array([0.0, 0.0, -1.0])
    x0 = inf - (inf @ z) * z
    if np.linalg.norm(x0) < 1e-6:                      # beam along the body axis (never on the skin)
        x0 = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x0 = x0 / np.linalg.norm(x0)
    x = rot_about(z, np.deg2rad(phi_deg)) @ x0
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


# ------------------------------------------------------------------------------------ the case
class _Case:
    def __init__(self, lv, raw_root=None, smooth_vox=4.0):
        from scipy import ndimage as ndi
        self.ndi = ndi
        self.cid = case_id(lv)
        self.labels = lv.labels
        self.A = np.asarray(lv.affine_m, np.float64)
        self.invA = np.linalg.inv(self.A)
        self.sp = np.abs(np.diag(self.A)[:3])
        if not (np.allclose(self.A[:3, :3], np.diag(np.diag(self.A)[:3])) and (np.diag(self.A)[:3] > 0).all()):
            raise ValueError(f"{self.cid}: windows need a diagonal RAS+ affine")
        self.shape = np.array(self.labels.shape)
        self.body = self._body_mask()
        self.fsm = ndi.gaussian_filter(self.body.astype(np.float32), smooth_vox)
        self.side = self._split_kidney(raw_root)

    def _body_mask(self):
        ndi = self.ndi
        m = self.labels != V.BACKGROUND
        m = ndi.binary_closing(m, iterations=1)
        lab, n = ndi.label(m)
        if n > 1:
            cnt = np.bincount(lab.ravel()); cnt[0] = 0
            m = lab == int(cnt.argmax())
        for k in range(m.shape[2]):                     # 2-D fill per axial slice (gas -> body)
            m[:, :, k] = ndi.binary_fill_holes(m[:, :, k])
        return m

    def _split_kidney(self, raw_root):
        """Left/right kidney masks. Raw TotalSegmentator masks when present (exact), else the sign of x
        relative to the body midline (the anatomical cache has no bladder in KIDNEY)."""
        K = self.labels == V.KIDNEY
        if not K.any():
            return {"right": K, "left": K}
        seg = os.path.join(raw_root or os.path.join(_REPO, "data", "raw", "totalseg_small"), self.cid, "segmentations")
        if os.path.isdir(seg):
            import nibabel as nib
            out = {}
            for side in ("right", "left"):
                p = os.path.join(seg, f"kidney_{side}.nii.gz")
                if not os.path.exists(p):
                    out[side] = np.zeros_like(K)
                    continue
                a = np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj) > 0.5
                if a.shape != K.shape:
                    raise ValueError(f"{self.cid}: kidney_{side} mask shape {a.shape} != {K.shape}")
                out[side] = K & a
            return out
        ii = np.argwhere(self.body[:, :, self.body.shape[2] // 2])
        mid_i = ii[:, 0].mean()
        xi = np.arange(K.shape[0])[:, None, None]
        return {"right": K & (xi > mid_i), "left": K & (xi <= mid_i)}

    def target_mask(self, organ):
        if organ == "liver":
            return self.labels == V.LIVER
        if organ == "gallbladder":
            return self.labels == V.GALLBLADDER
        if organ == "spleen":
            return self.labels == V.SPLEEN
        if organ in ("kidney_right", "kidney_left"):
            return self.side[organ.split("_")[1]]
        raise KeyError(organ)

    def vox(self, p):
        return self.invA[:3, :3] @ np.asarray(p, float) + self.invA[:3, 3]

    def world(self, ijk):
        return self.A[:3, :3] @ np.asarray(ijk, float) + self.A[:3, 3]

    def f_at(self, pts):
        v = (self.invA[:3, :3] @ np.atleast_2d(pts).T) + self.invA[:3, 3][:, None]
        return self.ndi.map_coordinates(self.fsm, v, order=1, mode="constant", cval=0.0)

    def normal_at(self, p, h=0.003):
        g = np.array([(self.f_at(p + h * e) - self.f_at(p - h * e))[0] for e in np.eye(3)])
        n = -g
        return n / (np.linalg.norm(n) + 1e-12)

    def skin_points(self, z_lo, z_hi, th_lo, th_hi, dz=0.010, dth=7.5, zmargin=0.012):
        """Radial march per axial slice from the slice's body centroid. Azimuth th: 0 = anterior."""
        out = []
        kz0 = self.vox(np.array([0, 0, z_lo]))[2]; kz1 = self.vox(np.array([0, 0, z_hi]))[2]
        zstep = dz / self.sp[2]
        kmin = max(int(np.ceil(zmargin / self.sp[2])), int(np.floor(min(kz0, kz1))))
        kmax = min(self.shape[2] - 1 - int(np.ceil(zmargin / self.sp[2])), int(np.ceil(max(kz0, kz1))))
        for k in np.arange(kmin, kmax + 1, zstep):
            k = int(round(k))
            sl = self.body[:, :, k]
            if sl.sum() < 500:
                continue
            ij = np.argwhere(sl)
            ci, cj = ij.mean(0)
            for th in np.arange(th_lo, th_hi + 1e-6, dth):
                t = np.deg2rad(th)
                di, dj = np.sin(t), np.cos(t)            # +i = +x (right), +j = +y (anterior)
                s, last, trunc = 0.0, None, False
                while True:
                    i, j = ci + s * di, cj + s * dj
                    ii, jj = int(round(i)), int(round(j))
                    if not (0 <= ii < sl.shape[0] and 0 <= jj < sl.shape[1]):
                        trunc = last is not None
                        break
                    if sl[ii, jj]:
                        last = (i, j)
                    elif last is not None and s * self.sp[0] > 0.02:
                        def _out(q):
                            a, b = int(round(ci + (s + q) * di)), int(round(cj + (s + q) * dj))
                            return (not sl[a, b]) if (0 <= a < sl.shape[0] and 0 <= b < sl.shape[1]) else True
                        if all(_out(q) for q in (1, 2, 3, 4)):
                            break
                    s += 0.5
                if last is None or trunc:
                    continue
                ii, jj = last
                if min(ii, jj) < 3 or ii > sl.shape[0] - 4 or jj > sl.shape[1] - 4:
                    continue                             # field-of-view truncated body edge
                out.append(dict(th=float(th), k=k, p=self.world([ii, jj, k])))
        return out

    def face_on_skin(self, p, n, span=0.02, step=0.00025):
        ts = np.arange(-span, span + 1e-9, step)
        f = self.f_at(p[None, :] + ts[:, None] * n[None, :])
        inside = f >= 0.5
        if not inside.any() or inside[-1]:
            return None
        return p + ts[np.where(inside)[0].max()] * n

    def slice(self, T_vp, grid):
        pts = grid.reshape(-1, 3)
        T = np.asarray(T_vp, float)
        pv = np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]
        vox = np.einsum("ij,nj->ni", self.invA[:3, :3], pv) + self.invA[:3, 3]
        idx = np.rint(vox).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < self.shape), axis=1)
        lab = np.zeros(len(pts), np.uint8)
        ii = idx[inb]
        lab[inb] = self.labels[ii[:, 0], ii[:, 1], ii[:, 2]]
        L, S = grid.shape[:2]
        return lab.reshape(L, S), inb.reshape(L, S), idx.reshape(L, S, 3)

    def curvature_gap(self, T_vp, radius=FLAT_RADIUS_M, n_dir=8):
        """(max, min) distance in mm between the tangent plane and the CT skin within the disk:
        + = skin BELOW the plane (gel gap), - = skin ABOVE it (the box would cut into the body)."""
        R = np.asarray(T_vp)[:3, :3]; t = np.asarray(T_vp)[:3, 3]
        b = R[:, 2]
        gaps = []
        for r in (radius / 2, radius):
            for a in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
                q = t + r * (np.cos(a) * R[:, 0] + np.sin(a) * R[:, 1])
                ts = np.arange(-0.03, 0.03, 0.0005)
                f = self.f_at(q[None, :] + ts[:, None] * b[None, :])
                inside = np.where(f >= 0.5)[0]
                if len(inside):
                    gaps.append(ts[inside.min()] * 1000)
        return (float(np.max(gaps)), float(np.min(gaps))) if gaps else (float("nan"), float("nan"))


def _screen(case, T_vp, tmask, grid, depth_m):
    lab, inb, idx = case.slice(T_vp, grid)
    L, S = lab.shape
    dz = depth_m / (S - 1)
    tm = np.zeros((L, S), bool)
    tm[inb] = tmask[idx[inb][:, 0], idx[inb][:, 1], idx[inb][:, 2]]
    body = lab != V.BACKGROUND
    has_body = body.any(1)
    entry = np.where(has_body, body.argmax(1), S)
    after = np.arange(S)[None, :] >= entry[:, None]
    obstr = (np.isin(lab, OBSTRUCT) | ((lab == V.BACKGROUND) & after) | (~inb)) & after
    has_t = tm.any(1)
    first_t = np.where(has_t, tm.argmax(1), S)
    ob_before = np.array([obstr[l, entry[l]:first_t[l]].any() if has_t[l] else True for l in range(L)])
    tlines = np.where(has_t)[0]
    clean = (~ob_before) & has_t
    c = L // 2
    labg = lab.copy()
    labg[~after] = V.SOFT_TISSUE                         # leading gap is gel-coupled in the renderer
    Zm = _ZT[labg]; MUm = _MUT[labg]
    Rr = np.zeros_like(Zm)
    Rr[:, 1:] = ((Zm[:, 1:] - Zm[:, :-1]) / (Zm[:, 1:] + Zm[:, :-1] + 1e-6)) ** 2
    Rr = np.clip(Rr, 0, 0.999)
    E = np.cumprod(1 - Rr, 1) * np.exp(-2 * _ATTEN_COEF * np.cumsum((MUm - _TGC_REF_MU) * dz * 100, 1))
    nt = tm.sum()
    cnt = np.bincount(lab.ravel(), minlength=13) / float(L * S)
    return dict(
        organ_frac={int(k): round(float(cnt[k]), 4) for k in LANDMARK_LABELS},
        frac=float(nt / (L * S)),
        clean_frac=float(clean[tlines].mean()) if len(tlines) else 0.0,
        center_ok=bool(has_t[c] and clean[c]),
        face_lab=int(lab[c, entry[c]]) if has_body[c] else -1,
        t_depth_mm=float(first_t[c] * dz * 1000) if has_t[c] else float("nan"),
        lit20=float((E[tm] >= 1e-2).mean()) if nt else 0.0,
        fov_out=float((~inb).mean()),
        gap_center_mm=float(entry[c] * dz * 1000) if has_body[c] else float("nan"),
    )


def _passes(m, min_frac):
    return (m["frac"] >= min_frac and m["clean_frac"] >= SCREEN["clean_min"] and m["center_ok"]
            and m["face_lab"] in FACE_OK and m["lit20"] >= SCREEN["lit_min"] and m["fov_out"] <= SCREEN["fov_max"])


def find_windows(case, organ, depth_m=None, dz=0.010, dth=7.5, zpad=0.04):
    """Screen every candidate window for one organ; return the distinct accepted windows (ordered)."""
    depth_m = float(V_IMG_DEPTH()) if depth_m is None else depth_m
    tmask = case.target_mask(organ)
    if not tmask.any():
        return []
    ijk = np.argwhere(tmask)
    tc = case.world(ijk.mean(0))
    zlo = case.world([0, 0, ijk[:, 2].min()])[2] - zpad
    zhi = case.world([0, 0, ijk[:, 2].max()])[2] + zpad
    fam = FAMILIES[organ]
    grid, _ = scan_grid(depth_m)
    cline = grid[N_LINES // 2:N_LINES // 2 + 1]
    cands = []
    for s in case.skin_points(zlo, zhi, fam["th"][0], fam["th"][1], dz=dz, dth=dth):
        n = case.normal_at(s["p"])
        face = case.face_on_skin(s["p"], n)
        if face is None:
            continue
        b0 = -n
        beams = [("normal", b0)]
        v = tc - face; v = v / np.linalg.norm(v)
        ang = np.degrees(np.arccos(np.clip(b0 @ v, -1, 1)))
        if ang > 0.5:
            beams.append(("to_target", rot_about(np.cross(b0, v), np.deg2rad(min(TILT_MAX_DEG, ang))) @ b0))
        for bname, b in beams:
            T = np.eye(4); T[:3, :3] = probe_rotation(b, 0.0); T[:3, 3] = face
            lab_c, inb_c, idx_c = case.slice(T, cline)
            tm_c = np.zeros(lab_c.shape, bool)
            tm_c[inb_c] = tmask[idx_c[inb_c][:, 0], idx_c[inb_c][:, 1], idx_c[inb_c][:, 2]]
            if not tm_c.any():
                continue
            for phi in PHIS:
                T = np.eye(4); T[:3, :3] = probe_rotation(b, phi); T[:3, 3] = face
                m = _screen(case, T, tmask, grid, depth_m)
                if not _passes(m, fam["min_frac"]):
                    continue
                m["n_landmarks"] = sum(1 for k, f in m["organ_frac"].items()
                                       if k != _ORGAN_LABEL[organ] and f >= LANDMARK_MIN_FRAC)
                if m["n_landmarks"] < MIN_LANDMARKS.get(organ, 0):
                    continue
                gmax, gmin = case.curvature_gap(T)
                if not (np.isfinite(gmax) and gmax <= FLAT_BELOW_MM and gmin >= -FLAT_ABOVE_MM):
                    continue
                m.update(organ=organ, th=s["th"], k=int(s["k"]), beam=bname,
                         tilt_deg=float(np.degrees(np.arccos(np.clip(b @ b0, -1, 1)))), phi=float(phi),
                         normal=[float(x) for x in n], face=[float(x) for x in face],
                         flat_gap_mm=[gmax, gmin], T_vp=np.round(T, 10).tolist())
                cands.append(m)
    order = sorted(cands, key=lambda m: (-min(m["n_landmarks"], 2), -round(m["frac"] * m["clean_frac"], 6),
                                         m["k"], m["th"], m["phi"], m["beam"]))
    acc = []
    for m in order:
        f = np.asarray(m["face"])
        if all(np.linalg.norm(f - np.asarray(a["face"])) >= NMS_M for a in acc):
            acc.append(m)
    for i, m in enumerate(acc):
        m["window_id"] = i
    return acc


# ------------------------------------------------------------------------------------ file API
def case_id(lv) -> str:
    meta = getattr(lv, "meta", {}) or {}
    if meta.get("case"):
        return str(meta["case"])
    p = meta.get("ct_path")
    if p:
        return os.path.basename(os.path.dirname(p))
    raise ValueError("LabeledVolume has no case id (meta['case'] or meta['ct_path']); "
                     "load it with volume.load_cache or set lv.meta['case']")


def labels_md5(lv) -> str:
    return hashlib.md5(np.ascontiguousarray(lv.labels).tobytes()).hexdigest()


def params() -> dict:
    return dict(version=WINDOWS_VERSION, img_depth_m=float(V_IMG_DEPTH()), phis=list(PHIS),
                tilt_max_deg=TILT_MAX_DEG, nms_m=NMS_M, flat_radius_m=FLAT_RADIUS_M,
                flat_below_mm=FLAT_BELOW_MM, flat_above_mm=FLAT_ABOVE_MM, screen=SCREEN,
                families=FAMILIES, fov_deg=FOV_DEG, apex_m=APEX_M, landmark_labels=list(LANDMARK_LABELS),
                landmark_min_frac=LANDMARK_MIN_FRAC, min_landmarks=MIN_LANDMARKS)


def build_case(lv, organs=ORGANS_BUILT, raw_root=None) -> dict:
    case = _Case(lv, raw_root=raw_root)
    return {"case": case.cid, "labels_md5": labels_md5(lv),
            "windows": {o: find_windows(case, o) for o in organs}}


_FILE = {"path": None, "mtime": None, "data": None}
_CHECKED: dict = {}


def load_file(path=None) -> dict:
    path = path or WINDOWS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"windows file {path} missing; build it with "
                                f"`USBG_SEAT=anatomical PYTHONPATH=src .venv/bin/python scripts/build_windows.py`")
    mt = os.path.getmtime(path)
    if _FILE["path"] != path or _FILE["mtime"] != mt:
        d = json.load(open(path))
        if d.get("params") != json.loads(json.dumps(params())):
            raise RuntimeError(f"{path} was built with different window parameters (version "
                               f"{d.get('params', {}).get('version')}); rebuild it")
        _FILE.update(path=path, mtime=mt, data=d)
    return _FILE["data"]


def file_md5(path=None) -> str:
    return hashlib.md5(open(path or WINDOWS_PATH, "rb").read()).hexdigest()


def _organ_lists(label):
    if label == V.LIVER:
        return ["liver"]
    if label == V.GALLBLADDER:
        return ["gallbladder"]
    if label == V.KIDNEY:
        return ["kidney_right", "kidney_left"]
    if label == V.SPLEEN:
        return ["spleen"] if os.environ.get("USBG_ALLOW_SPLEEN", "0") == "1" else []
    return []


def windows_meta(lv, label) -> list:
    """Accepted windows for (case, label) as metadata dicts, in window-index order. KIDNEY interleaves
    right and left flank windows (R0, L0, R1, L1, ...) so a modulo index stays side-balanced."""
    d = load_file()
    cid = case_id(lv)
    if cid not in d["cases"]:
        raise NoAcceptedWindow(f"case {cid} is not in the windows file {WINDOWS_PATH}")
    rec = d["cases"][cid]
    key = (cid, id(lv))
    if key not in _CHECKED:
        if labels_md5(lv) != rec["labels_md5"]:
            raise RuntimeError(f"{cid}: labels differ from those the windows were screened on "
                               f"(bladder-free cache expected; rebuild windows or check USBG_SEAT)")
        _CHECKED[key] = True
    lists = [rec["windows"].get(o, []) for o in _organ_lists(label)]
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x):
                out.append(x[i])
    return out


def windows_for(lv, label) -> list:
    """List of 4x4 T_vp for (case, label), in window-index order."""
    return [np.asarray(m["T_vp"], np.float64) for m in windows_meta(lv, label)]
