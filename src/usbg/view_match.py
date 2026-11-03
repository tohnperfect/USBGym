"""view_match.py: the curvilinear ultrasound view-reproduction RL task core.

Goal (redefined closed-loop task): an agent moves a virtual probe over a user-provided CT + organ
segmentation (a LabeledVolume) to render a synthetic B-mode -- in the clinical CURVILINEAR (sector)
geometry, textured per-tissue by SonoSPADE -- that reproduces a GIVEN target ultrasound view. The
match is anchored on organ segmentation (per-tissue layout) AND appearance, so the render must put the
right texture in the right organ to match the target: exactly what per-tissue synthesis provides and a
global recolor does not.

Two target regimes (see scripts/train_view_match_rl.py):
  * self-supervised: the target is this same volume rendered (through the same translator) at a hidden
    probe pose; the agent reproduces it (ground-truth pose known -> clean eval + imitation warm-start).
  * real registration: the target is a real US image + its organ segmentation (e.g. Kaggle Vitale);
    the agent finds the closest CT-derived view (the target is not exactly achievable).

This module is the reusable core: the sector warp (matching the measured Kaggle geometry), the
per-pose curvilinear render, and the combined segmentation+appearance match score.
"""
from __future__ import annotations

import os as _os

import numpy as np

from . import slicer as S
from . import render_lotus as R
from .segmenter import align_label_to_image
from .volume import LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN

# Mask source for the label that curvi_render RETURNS (and thus everything scores/consumes: IBVS
# calibration, rollout, imitation, DAgger, evaluation, the real-registration current mask, and the
# seg_organ_iou reward). Default: the trained cross-domain segmenter (usbg.us_segmenter), so sim
# training and real-probe deployment share ONE mask source. Set
# USBG_LEARNED_SEG=0 to fall back to the CT-derived oracle label (lab_sec); used only to produce the
# oracle reference row. The texture is unchanged either way (lab_sec still conditions SonoSPADE).
_USE_LEARNED_SEG = _os.environ.get("USBG_LEARNED_SEG", "1").strip().lower() not in ("0", "false", "no", "")
_US = None
if _USE_LEARNED_SEG:
    from . import us_segmenter as _US


def set_mask_source(seg) -> None:
    """Set the cross-domain mask source the view-match reward's segmentation anchor uses. ``seg`` must
    expose ``segment(img01) -> int64 mask`` in the repo ontology {0,1,2,3,4,7}; pass ``None`` to use the
    CT-oracle label (equivalent to ``USBG_LEARNED_SEG=0``). Call before the first rollout. This is the
    public alternative to binding the module-private ``usbg.view_match._US`` directly."""
    global _US
    _US = seg


# organs used for the segmentation-overlap match (exactly the classes the real Kaggle masks provide)
MATCH_ORGANS = (LIVER, GALLBLADDER, VESSEL, KIDNEY, SPLEEN)

# measured Kaggle Vitale sector geometry at 128x128 (matches scripts/curvi_warp.py, i.e. the geometry
# SonoSPADE-curvi and its baselines were trained on).
SECTOR = dict(apex_x=68.0, apex_y=-28.0, half_angle_deg=23.0, r_min=36.0, r_max=128.0)


def _sector_coords(H, W, p, rot_deg=0.0, tx=0.0, ty=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ax, ay = p["apex_x"] + tx * W, p["apex_y"] + ty * H
    dx, dy = xx - ax, yy - ay
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.degrees(np.arctan2(dx, dy)) - rot_deg
    half = p["half_angle_deg"]
    mask = (np.abs(theta) <= half) & (r >= p["r_min"]) & (r <= p["r_max"])
    src_col = (theta + half) / (2.0 * half) * (W - 1)
    src_row = (r - p["r_min"]) / (p["r_max"] - p["r_min"]) * (H - 1)
    return src_row, src_col, mask


def warp_to_sector(img: np.ndarray, label: np.ndarray, p=SECTOR):
    """Warp a linear (image, label) into the measured convex sector; outside the sector -> background."""
    from scipy.ndimage import map_coordinates
    H, W = img.shape
    sr, sc, mask = _sector_coords(H, W, p)
    coords = np.stack([sr.ravel(), sc.ravel()])
    wi = map_coordinates(img.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    wi = (wi.reshape(H, W) * mask).astype(np.float32)
    wl = map_coordinates(label.astype(np.float32), coords, order=0, mode="constant", cval=0.0)
    wl = (wl.reshape(H, W) * mask).astype(np.int64)
    return wi, wl


def _resize01(img01, size):
    from PIL import Image
    a = np.clip(np.asarray(img01, np.float32) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32) / 255.0


def curvi_render(lv, T, world_from_volume, translator, cfg=None, size=128, cond_label=None):
    """Render the probe view at world pose T as a CURVILINEAR, translator-textured B-mode, and return
    the aligned curvilinear canonical label slice used to condition/score it.

    translator: a usbg.texture_gan translator (SonoSPADE dual-input, or CUT/CycleGAN single-input), or
    None for the raw physics sector render. Returns (img01 (size,size) float32, label (size,size) int64).

    cond_label (opt-in, v2.1 label-information controls; the default None leaves every step unchanged):
    a callable (lv, T, world_from_volume, lab_lin) -> label slice (N_LINES, N_SAMPLES) that CONDITIONS a
    label-conditioned (wants_label) translator in place of the true slice. The physics input, the CT
    label returned and the learned mask source keep the true pose. Ignored by single-input
    translators and by the raw physics render. See usbg.label_controls."""
    lab_lin = S.slice_at_pose(lv, np.asarray(T, np.float64), world_from_volume)   # (N_LINES,N_SAMPLES)
    img_lin = R.render(lab_lin, "physics", cfg=cfg)                                # (N_SAMPLES,N_LINES) [0,1]
    img_sq = _resize01(img_lin, size)
    lab_sq = align_label_to_image(lab_lin, (size, size))
    img_sec, lab_sec = warp_to_sector(img_sq, lab_sq)
    if getattr(translator, "wants_label", False):
        cond_sec = lab_sec
        if cond_label is not None:                       # opt-in label control (never on the default path)
            cond_lin = cond_label(lv, T, world_from_volume, lab_lin)
            cond_sec = warp_to_sector(img_sq, align_label_to_image(cond_lin, (size, size)))[1]
        tex = translator.translate_aligned(img_sec, cond_sec)
    elif translator is not None:
        tex = translator(img_sec)
    else:
        tex = img_sec
    tex = np.asarray(tex, np.float32)
    if _US is not None:
        lab_out = _US.segment(tex)                       # learned cross-domain mask, repo ontology {0,1,2,3,4,7}
    else:
        lab_out = np.asarray(lab_sec, np.int64)          # CT-derived oracle label (USBG_LEARNED_SEG=0)
    return tex, lab_out


def seg_organ_iou(lab_a: np.ndarray, lab_b: np.ndarray, organs=MATCH_ORGANS, min_pixels=20) -> float:
    """Mean per-organ IoU between two curvilinear label slices, over the organs present in either
    (with >= min_pixels). Texture-independent organ-layout match. Returns 1.0 if neither has organs."""
    a = np.asarray(lab_a); b = np.asarray(lab_b)
    ious = []
    for o in organs:
        pa, pb = (a == o), (b == o)
        if pa.sum() < min_pixels and pb.sum() < min_pixels:
            continue
        inter = int((pa & pb).sum()); union = int((pa | pb).sum())
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else 1.0


def sector_appearance(cur_img, tgt_img, thresh=0.04) -> float:
    """Content-focused appearance match in [0,1]: normalized cross-correlation of the SECTOR-INTERIOR
    (lit) pixels only. Whole-frame SSIM/NCC on curvilinear frames is dominated by the shared black
    border (~63% of the frame) and is nearly flat across poses; masking to the lit sector exposes the
    internal per-organ texture -- which tracks the probe pose under per-tissue synthesis and is washed
    out by a global recolor."""
    a = np.asarray(cur_img, np.float32); b = np.asarray(tgt_img, np.float32)
    m = (a > thresh) | (b > thresh)
    if int(m.sum()) < 50:
        return 0.0
    av = a[m] - a[m].mean(); bv = b[m] - b[m].mean()
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    ncc = float((av * bv).sum() / denom) if denom > 1e-8 else 0.0
    return 0.5 * (ncc + 1.0)


def combined_score(cur_img, cur_lab, tgt_img, tgt_lab, sim=None, w_seg=0.5, w_app=0.5) -> float:
    """Match score: w_seg * organ-layout IoU (texture-independent anchor) + w_app * sector-interior
    appearance NCC (where per-tissue-correct texture makes the render actually resemble the target and
    a global recolor does not). `sim` is accepted for API compatibility but appearance uses
    sector_appearance (whole-frame similarity is dominated by the shared curvilinear border)."""
    seg = seg_organ_iou(cur_lab, tgt_lab)
    app = sector_appearance(cur_img, tgt_img)
    return w_seg * seg + w_app * app


def pose_error(T_cur, T_tgt):
    """(translation mm, rotation deg) between two world poses -- the translator-independent ground-truth
    metric for the self-supervised regime."""
    from .contracts_bridge import invert_T, se3_log
    rel = se3_log(invert_T(np.asarray(T_tgt)) @ np.asarray(T_cur))
    return float(np.linalg.norm(rel[:3]) * 1000.0), float(np.degrees(np.linalg.norm(rel[3:])))
