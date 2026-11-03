"""label_controls.py: testbed-side label-information controls for label-conditioned texture stages.

the v2.1 study. EXPLORATORY: every output these controls feed carries
``"analysis_status": "exploratory, v2.1"`` in its provenance block; the archived v2 numbers stay primary.
No texture stage is retrained or modified here: the published SonoSPADE weights and method are used
as frozen inference-only components, and every control lives on the testbed side.

(a) Inference-only label controls. A stage that takes a label (``wants_label``) is given a CONTROL label
    slice; its physics input, the CT label it returns and the mask source keep the true pose:
      none       the true slice (identical to the default render path);
      elev<d>    the slice taken at a d mm offset along the probe elevational axis (probe-frame +Y; the
                 imaging plane is the probe X-Z plane, X lateral, Z axial), e.g. elev3, elev6, elev-3;
      permute    the true slice with the five match organs deranged (PERMUTATION below).
    ``LabelControl`` is the callable ``view_match.curvi_render(..., cond_label=)`` takes; ``render_parts``
    returns the same render plus the true and conditioning labels (for Dice against both).

(b) A non-learned label-painted physics stage (``LabelPaintedTranslator``), written as an ordinary
    ``wants_label`` TextureTranslator plugin and loadable from a small checkpoint through
    ``texture_gan.load_translator`` (registered arch ``usbg_label_painted_v1``; import this module first).
    It takes the same two inputs as SonoSPADE (the physics sector frame and the label) and paints, in
    the display domain:
      * per-label echogenicity: renal cortex relative to liver from the published hepatorenal index
        (display gray-level ratio liver / renal cortex). The physics renderer's own HRI is measured once
        on TRAIN-patient physics renders (never on real frames) and divided out, so painted frames
        realize the published ratio on average. Labels with no quantitative published normal ratio stay
        at the physics level (gain 1), and fluid (gallbladder, vessel lumen) is already anechoic.
      * boundary reflectivity: organ capsules and vessel / gallbladder walls drawn as 1-pixel echogenic
        lines, ``capsule_db`` above the local parenchyma level on the log-compressed display.
    The literature table and its sources are in LITERATURE; VARIANTS holds the central row ('lit') and
    the one gain-sensitivity row ('x2': both log-contrasts doubled). Nothing is fitted to Vitale, AUL or
    KidneyUS frames.

Present results as a property of label-conditioned stages, never as an audit of a particular method.
"""
from __future__ import annotations

import datetime as _dt
import os

import numpy as np

from . import render_lotus as R
from . import slicer as S
from . import texture_gan as TG
from .segmenter import align_label_to_image
from .volume import (BACKGROUND, BONE, GALLBLADDER, KIDNEY, LIVER, LUNG, PANCREAS, SPLEEN, SOFT_TISSUE,
                     VESSEL)

ANALYSIS_STATUS = "exploratory, v2.1"
PLAN_STEP = "v2.1 study, testbed-side label-information controls"

# ======================================================================================================
# (a) inference-only label controls
# ======================================================================================================
# A derangement of the five match organs: every organ is relabeled as a different one. Solid organs
# cycle liver -> kidney -> spleen -> liver; the two fluid classes swap. Other labels are unchanged.
PERMUTATION = {LIVER: KIDNEY, KIDNEY: SPLEEN, SPLEEN: LIVER, GALLBLADDER: VESSEL, VESSEL: GALLBLADDER}
_PERM_LUT = np.arange(256, dtype=np.uint8)
for _a, _b in PERMUTATION.items():
    _PERM_LUT[_a] = _b
ELEV_AXIS = (0.0, 1.0, 0.0)          # probe-frame elevational unit vector (normal to the imaging plane)


def _probe_offset(d_m: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = np.asarray(ELEV_AXIS) * float(d_m)
    return M


class LabelControl:
    """A conditioning-label control: callable (lv, T, world_from_volume, lab_lin) -> label slice."""

    def __init__(self, name: str, kind: str, offset_mm: float = 0.0):
        if kind not in ("none", "elev", "permute"):
            raise ValueError(f"unknown label control kind {kind!r}")
        self.name, self.kind, self.offset_mm = name, kind, float(offset_mm)

    def __call__(self, lv, T, world_from_volume, lab_lin):
        if self.kind == "none":
            return lab_lin
        if self.kind == "permute":
            return _PERM_LUT[np.asarray(lab_lin, np.uint8)]
        T2 = np.asarray(T, np.float64) @ _probe_offset(self.offset_mm / 1000.0)
        return S.slice_at_pose(lv, T2, world_from_volume)

    @property
    def is_identity(self) -> bool:
        return self.kind == "none"

    def describe(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        if self.kind == "elev":
            d.update(offset_mm=self.offset_mm, axis_probe_frame=list(ELEV_AXIS),
                     note="label slice at T @ translate(offset * axis); physics stays at T")
        if self.kind == "permute":
            d.update(permutation={int(k): int(v) for k, v in PERMUTATION.items()})
        return d


def parse_control(spec: str) -> LabelControl:
    """'none' | 'permute' | 'elev<d>' with d in mm (e.g. elev3, elev6, elev-3, elev2.5)."""
    s = str(spec).strip().lower()
    if s == "none":
        return LabelControl("none", "none")
    if s == "permute":
        return LabelControl("permute", "permute")
    if s.startswith("elev"):
        return LabelControl(s, "elev", float(s[4:]))
    raise ValueError(f"label control {spec!r}: expected none, permute or elev<mm>")


def render_parts(lv, T, world_from_volume, translator, cfg=None, size=128, control=None) -> dict:
    """The exact view_match.curvi_render pipeline, returning its parts: img_sec (physics sector frame),
    lab_sec (true CT label, canonical ids), cond_sec (the label the translator was conditioned on, or
    None for a stage that takes no label) and tex (the textured frame). With control None or 'none'
    the texture is bit-identical to curvi_render's (checked by the v2.1 smoke test)."""
    from . import view_match as VM
    lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), world_from_volume)
    img_lin = R.render(lab_lin, "physics", cfg=cfg)
    img_sq = VM._resize01(img_lin, size)
    lab_sq = align_label_to_image(lab_lin, (size, size))
    img_sec, lab_sec = VM.warp_to_sector(img_sq, lab_sq)
    cond_sec = None
    if getattr(translator, "wants_label", False):
        cond_sec = lab_sec
        if control is not None and not control.is_identity:
            cond_lin = control(lv, T, world_from_volume, lab_lin)
            cond_sec = VM.warp_to_sector(img_sq, align_label_to_image(cond_lin, (size, size)))[1]
        tex = translator.translate_aligned(img_sec, cond_sec)
    elif translator is not None:
        tex = translator(img_sec)
    else:
        tex = img_sec
    return {"img_sec": img_sec, "lab_sec": np.asarray(lab_sec, np.int64),
            "cond_sec": None if cond_sec is None else np.asarray(cond_sec, np.int64),
            "tex": np.asarray(tex, np.float32)}


def control_label_lin(control, lv, T, world_from_volume):
    """The conditioning label a control WOULD give at pose T (also for a stage that takes no label, so
    a label-blind reference stage can be scored against the same control label)."""
    lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), world_from_volume)
    return lab_lin if control is None else control(lv, T, world_from_volume, lab_lin)


def sector_label(lab_lin, size=128) -> np.ndarray:
    """A slicer-orientation label warped into the curvilinear sector (as curvi_render does)."""
    from . import view_match as VM
    lab_sq = align_label_to_image(lab_lin, (size, size))
    return VM.warp_to_sector(np.zeros((size, size), np.float32), lab_sq)[1]


# ------------------------------------------------------------------------------------------------ Dice
SIX_NAMES = ("background", "liver", "gallbladder", "vessel", "kidney", "spleen")
_TO_SIX = np.zeros(256, np.int64)
for _i, _c in enumerate((LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN), start=1):
    _TO_SIX[_c] = _i


def to_six(lab) -> np.ndarray:
    """Canonical ids (0..12) or the repo mask ontology {0,1,2,3,4,7} -> the segmenter's 6-class scheme
    (the same map as us_seg.ontology.canonical_to_six; non-target labels -> background)."""
    return _TO_SIX[np.asarray(lab, np.int64)]


def frame_dice(pred6, ref6, eps=1e-6) -> dict:
    """{class name: Dice} over the foreground classes present in the reference frame."""
    out = {}
    for c in range(1, 6):
        t = ref6 == c
        if not t.any():
            continue
        p = pred6 == c
        out[SIX_NAMES[c]] = float((2.0 * np.logical_and(p, t).sum() + eps) / (p.sum() + t.sum() + eps))
    return out


def frame_mean_dice(pred6, ref6):
    d = frame_dice(pred6, ref6)
    return float(np.mean(list(d.values()))) if d else float("nan")


class DiceAccumulator:
    """Per-class Dice averaged over frames where the class is present in the reference, and the mean
    over classes: the definition of us_seg.training.per_class_dice / mean_fg_dice (held-out sim Dice)."""

    def __init__(self):
        self.d = {n: [] for n in SIX_NAMES[1:]}

    def add(self, pred6, ref6):
        for k, v in frame_dice(pred6, ref6).items():
            self.d[k].append(v)

    def result(self) -> dict:
        pc = {k: (float(np.mean(v)), len(v)) for k, v in self.d.items() if v}
        return {"per_class": {k: round(v[0], 4) for k, v in pc.items()},
                "support": {k: v[1] for k, v in pc.items()},
                "mean_fg": round(float(np.mean([v[0] for v in pc.values()])), 4) if pc else None}


# ======================================================================================================
# (b) the non-learned label-painted physics stage
# ======================================================================================================
PAINT_ARCH = "usbg_label_painted_v1"
CAPSULE_LABELS = (LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN, PANCREAS)   # organs drawn with a wall line
NO_LINE_NEIGHBOURS = (BACKGROUND, BONE, LUNG)       # no line against gel/outside, or bone/air (already bright)
FLOOR = 0.02                                        # display level treated as "no echo" (shadow / gel)
DISPLAY_DR_DB = 50.0                                # nominal log-compression window of the physics renderer
DISPLAY_GAMMA = float(R.GAMMA)                      # physics display gamma (img01 = u ** gamma)

LITERATURE = {
    "hri_normal": {
        "value": 1.14, "sd": 0.2,
        "quantity": "hepatorenal index: display gray-level ratio liver / right renal cortex",
        "source": "K. Shiralkar, S. Johnson, E.I. Bluth, R.H. Marshall, A. Dornelles, P.M. Gulotta, Improved "
                  "method for calculating hepatic steatosis using the hepatorenal index, J. Ultrasound Med. 34 "
                  "(2015) 1051-1059, https://doi.org/10.7863/ultra.34.6.1051 (PMID 26014325): mean HRI 1.14 "
                  "+/- 0.2 without steatosis",
    },
    "hri_upper": {
        "value": 1.28,
        "source": "R.H. Marshall et al., AJR Am. J. Roentgenol. 199 (2012) 997-1002 (the HRI in steatosis "
                  "screening): an HRI of 1.28 is the cutoff with 100% sensitivity for more than 5% fat (upper "
                  "end of normal)",
    },
    "isoechoic_organs": "spleen, pancreas and the generic tissues: no quantitative normal echogenicity "
                        "ratio to liver was found (textbook appearance only: spleen iso- to slightly "
                        "hyperechoic, pancreas iso- to hyperechoic), so they stay at the physics level",
    "fluid": "gallbladder and vessel lumen are anechoic; the physics table already renders them far below "
             "liver (mean backscatter 0.02 and 0.001 against 0.40), so they are left unchanged",
    "capsule": "organ capsules and vessel and gallbladder walls appear as thin echogenic lines in clinical "
               "B-mode (textbook appearance). No quantitative capsule-to-parenchyma level was found, so "
               "the line level is a fixed design choice (12 dB, a quarter of the 50 dB display window) "
               "that the sensitivity row doubles",
    "status": "fixed by the pre-specified protocol before any run; exploratory",
}
VARIANTS = {
    "lit": {"hri_target": 1.14, "capsule_db": 12.0,
            "note": "central row: published normal HRI, 12 dB capsule line"},
    "x2": {"hri_target": round(1.14 ** 2, 4), "capsule_db": 24.0,
           "note": "sensitivity row for the gain: the two log-contrasts doubled (HRI 1.30, about the 1.28 "
                   "upper normal; 24 dB capsule line)"},
}
# TRAIN-render calibration geometry: the segmenter repo's frame sampler defaults (us_seg build_dataset)
CALIB_SETUP_KW = dict(lateral_m=0.02, axial_m=0.008, fan_deg=15.0, roll_deg=4.0)


def _box3(u):
    p = np.pad(u, 1, mode="edge")
    return sum(p[1 + dy:1 + dy + u.shape[0], 1 + dx:1 + dx + u.shape[1]]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0


def capsule_edges(lab) -> np.ndarray:
    """Bool mask of 1-pixel boundary pixels: a pixel of a CAPSULE_LABELS organ with a 4-neighbour of a
    different label that is not background, bone or lung (so no line along the sector edge or gel)."""
    lab = np.asarray(lab, np.int64)
    own = np.isin(lab, CAPSULE_LABELS)
    p = np.pad(lab, 1, mode="constant", constant_values=BACKGROUND)
    H, W = lab.shape
    edge = np.zeros_like(own)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        edge |= (nb != lab) & ~np.isin(nb, NO_LINE_NEIGHBOURS)
    return own & edge


class LabelPaintedTranslator:
    """Non-learned per-label painting of a physics frame (see the module doc). Dual input like SonoSPADE:
    translate_aligned(img01, aligned_label) for a label already in the image frame, translate(img01,
    label_slice) for a slicer-orientation label. Deterministic, numpy only."""

    wants_label = True

    def __init__(self, params: dict, ckpt_path: str | None = None):
        self.params = dict(params)
        self.ckpt_path = ckpt_path
        lut = np.ones(256, np.float32)
        for k, v in (self.params.get("label_gains") or {}).items():
            lut[int(k)] = float(v)
        self.gain_lut = lut
        self.capsule_db = float(self.params.get("capsule_db", 0.0))
        self.dr_db = float(self.params.get("display_dr_db", DISPLAY_DR_DB))
        self.gamma = float(self.params.get("display_gamma", DISPLAY_GAMMA))
        self.floor = float(self.params.get("floor", FLOOR))

    def translate_aligned(self, img01: np.ndarray, aligned_label: np.ndarray) -> np.ndarray:
        img = np.asarray(img01, np.float32)
        lab = np.asarray(aligned_label)
        if lab.shape != img.shape:
            from PIL import Image
            lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize(img.shape[::-1], Image.NEAREST))
        lab = lab.astype(np.int64)
        out = np.clip(img * self.gain_lut[lab], 0.0, 1.0)                 # per-label echogenicity
        if self.capsule_db > 0:                                           # capsule / wall lines
            u = out ** (1.0 / self.gamma)                                 # log-display level (linear in dB)
            ubar = _box3(u)
            line = np.minimum(1.0, ubar + self.capsule_db / self.dr_db)
            paint = capsule_edges(lab) & (ubar > self.floor ** (1.0 / self.gamma))
            u = np.where(paint, np.maximum(u, line), u)
            out = u ** self.gamma
        return np.asarray(out, np.float32)

    def translate(self, img01: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
        img = np.asarray(img01, np.float32)
        return self.translate_aligned(img, align_label_to_image(label_slice, img.shape))

    __call__ = translate


@TG.register_translator(PAINT_ARCH)
def load_label_painted(ckpt_path, device=None):
    """texture_gan.load_translator hook for a label-painted checkpoint (registered on import)."""
    import torch
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if st.get("arch") != PAINT_ARCH:
        raise ValueError(f"{ckpt_path}: arch {st.get('arch')!r} is not {PAINT_ARCH!r}")
    return LabelPaintedTranslator(st["params"], ckpt_path=os.path.abspath(ckpt_path))


def _region_mean(img, lab, label, erode=2, floor=FLOOR):
    from scipy.ndimage import binary_erosion
    m = lab == label
    if erode:
        m = binary_erosion(m, iterations=erode)
    m &= img > floor
    return (float(img[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)


def measure_physics_hri(cases, n_poses=20, seed=0, size=128, min_px=100, setup_kw=None, min_frames=30,
                        log=print) -> dict:
    """The physics renderer's own hepatorenal index on TRAIN-patient renders (DR off, a speckle seed per
    frame), from liver and kidney windows, target and perturbed-start views: per frame mean liver /
    mean kidney display level over insonified pixels of 2-pixel-eroded masks, median over frames with
    both organs >= min_px. No real frame is read."""
    from . import goal_rl as G
    from .placement import NoAcceptedWindow
    from .volume import load_cache
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    setup_kw = dict(setup_kw or CALIB_SETUP_KW)
    rng = np.random.default_rng(seed)
    ratios, per_case, extra = [], {}, {"spleen_over_liver": [], "soft_over_liver": []}
    for case in cases:
        lv = load_cache(os.path.join(repo, "data", "cache", f"{case}.npz"))
        used = 0
        for organ in (KIDNEY, LIVER):
            for _k in range(n_poses):
                try:
                    Wfv, T_tgt, T_start, _m = G.sample_setup(lv, organ, rng, return_meta=True, **setup_kw)
                except NoAcceptedWindow:
                    break
                for T in (T_tgt, T_start):
                    cfg = R.RenderConfig(seed=int(rng.integers(2 ** 31 - 1)))
                    p = render_parts(lv, T, Wfv, None, cfg, size)
                    img, lab = p["img_sec"], p["lab_sec"]
                    ml, nl = _region_mean(img, lab, LIVER)
                    mk, nk = _region_mean(img, lab, KIDNEY)
                    if nl >= min_px and nk >= min_px and mk > 0:
                        ratios.append(ml / mk); used += 1
                    if nl >= min_px:
                        ms, ns = _region_mean(img, lab, SPLEEN)
                        if ns >= min_px:
                            extra["spleen_over_liver"].append(ms / ml)
                        mt, nt = _region_mean(img, lab, SOFT_TISSUE)
                        if nt >= min_px:
                            extra["soft_over_liver"].append(mt / ml)
        per_case[case] = used
    if len(ratios) < min_frames:
        raise RuntimeError(f"only {len(ratios)} TRAIN frames show both liver and kidney (< {min_frames}); "
                           "raise n_poses")
    r = np.asarray(ratios)
    out = {"hri_phys_median": float(np.median(r)), "hri_phys_mean": float(r.mean()),
           "hri_phys_iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
           "n_frames": int(len(r)), "frames_per_case": per_case, "cases": list(cases), "seed": int(seed),
           "n_poses_per_case_organ": int(n_poses), "setup_kw": setup_kw, "dr": "off (speckle seed per frame)",
           "min_px": int(min_px), "min_frames": int(min_frames), "mask_erosion_px": 2, "floor": FLOOR,
           "diagnostic_ratios_median": {k: (float(np.median(v)) if v else None) for k, v in extra.items()}}
    log(f"[calib] physics HRI median {out['hri_phys_median']:.3f} (IQR {out['hri_phys_iqr'][0]:.3f}-"
        f"{out['hri_phys_iqr'][1]:.3f}) over {out['n_frames']} TRAIN frames")
    return out


def painted_params(variant: str, calib: dict) -> dict:
    v = VARIANTS[variant]
    g_k = calib["hri_phys_median"] / v["hri_target"]      # painted HRI = HRI_phys / g_k = target
    return {"variant": variant, "label_gains": {int(KIDNEY): round(float(g_k), 6)},
            "hri_target": v["hri_target"], "capsule_db": v["capsule_db"], "display_dr_db": DISPLAY_DR_DB,
            "display_gamma": DISPLAY_GAMMA, "floor": FLOOR, "capsule_labels": [int(x) for x in CAPSULE_LABELS],
            "no_line_neighbours": [int(x) for x in NO_LINE_NEIGHBOURS], "note": v["note"]}


def write_painted_checkpoint(path: str, variant: str, calib: dict) -> dict:
    """Save a label-painted 'checkpoint' (arch + parameters, no weights) at a NEW path and its
    provenance sidecar. Refuses to overwrite an existing file. Returns the saved state."""
    import torch
    from . import provenance as PROV
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; v2.1 artifacts are new files only")
    state = {"arch": PAINT_ARCH, "params": painted_params(variant, calib), "calibration": calib,
             "literature": LITERATURE, "analysis_status": ANALYSIS_STATUS, "plan_step": PLAN_STEP,
             "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)
    used = [c for c, n in calib["frames_per_case"].items() if n > 0]
    PROV.write_provenance(path, "translator", train_cases=used, iters=0, seed=calib["seed"], dr_mode="off",
                          dr_notes="non-learned stage: only the physics HRI was measured on these TRAIN "
                                   "renders (DR off); no real frame and no gradient step",
                          extra={"arch": PAINT_ARCH, "variant": variant, "params": state["params"],
                                 "analysis_status": ANALYSIS_STATUS, "non_learned": True})
    return state


__all__ = ["ANALYSIS_STATUS", "PERMUTATION", "LabelControl", "parse_control", "render_parts",
           "control_label_lin", "sector_label", "to_six", "frame_dice", "frame_mean_dice", "DiceAccumulator",
           "PAINT_ARCH", "LITERATURE", "VARIANTS", "LabelPaintedTranslator", "load_label_painted",
           "capsule_edges", "measure_physics_hri", "painted_params", "write_painted_checkpoint"]
