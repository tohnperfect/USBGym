"""Generate simulated (B-mode image, 6-class label) training pairs by driving `usbg`.

Each frame: seat the probe on an organ (`sample_setup`), draw a fresh per-frame domain-randomized
render config (`domain_randomize`), render a SonoSPADE-textured curvilinear frame + its CT label
slice (`curvi_render`), and remap the canonical label to the contiguous 6-class scheme.

Domain randomization is drawn **per frame** (not once per episode) for maximum appearance variance:
the RL "same-pipeline" rule does not apply here; the label is pixel-exact and untouched by DR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gym_bridge
from . import ontology as ONT


@dataclass
class SimFrame:
    image: np.ndarray            # (H,W) float32 in [0,1]
    label6: np.ndarray           # (H,W) uint8 in {0..5}
    case: str                    # source case stem, e.g. "s0011"
    seat_organ: str              # "liver" | "kidney" | "spleen"
    pose: str                    # "target" | "start"
    organs_present: dict = field(default_factory=dict)  # 6-class name -> pixel count (fg only)
    seat: dict = field(default_factory=dict)            # sample_setup meta (+ window side/id), anatomical


# Seat organs. Anatomical seat (v2): liver and kidney (spleen is DROPPED as a target: the gym exposes
# no spleen window unless USBG_ALLOW_SPLEEN=1); spleen stays a segmenter CLASS, seen incidentally in
# left-flank kidney views. Gallbladder/vessel are picked up incidentally inside these views.
# Canonical seat (v1 reproduction): liver, kidney, spleen, as the shipped Dataset011-034 builds.
SEAT_ORGANS_V2 = ("liver", "kidney")
SEAT_ORGANS_CANONICAL = ("liver", "kidney", "spleen")
SEAT_ORGANS = SEAT_ORGANS_CANONICAL


def default_seat_organs(seat: str) -> tuple:
    return SEAT_ORGANS_V2 if seat == "anatomical" else SEAT_ORGANS_CANONICAL


def _seat_map():
    from usbg import volume as V
    return {"liver": V.LIVER, "kidney": V.KIDNEY, "spleen": V.SPLEEN}


def organ_presence(label6: np.ndarray, min_px: int = 20) -> dict:
    """Foreground 6-class name -> pixel count, for classes with >= min_px pixels."""
    out = {}
    u, c = np.unique(label6, return_counts=True)
    for k, v in zip(u.tolist(), c.tolist()):
        if k == 0 or v < min_px:
            continue
        out[ONT.CLASS_NAMES[k]] = int(v)
    return out


def load_translator(name: str):
    """Load a texture translator by alias/path, or None. Kept pluggable."""
    gym_bridge.ensure_usbg_importable()
    from usbg import texture_gan as TG
    from usbg import render_lotus as R

    ckpt = gym_bridge.translator_ckpt(name)
    if ckpt is None:
        return None
    return TG.load_translator(ckpt, device=R.get_device())


def iter_sim_frames(
    case_paths: list[str],
    translator,
    *,
    n_poses: int = 8,
    size: int = 128,
    seed: int = 0,
    include_start: bool = True,
    setup_kw: dict | None = None,
    seat_organs=SEAT_ORGANS,
    strict_seat: bool = False,
    skipped: list | None = None,
    skip_pairs=(),
    case_meta: dict | None = None,
):
    """Yield SimFrame pairs over cases x seat-organs x poses.

    - case_paths: list of <gym>/data/cache/*.npz paths (gym_bridge.cache_cases()). Each is loaded with
      gym_bridge.load_case -> usbg.volume.load_cache, so under USBG_SEAT=anatomical the bladder-free
      data/cache_anat twin is read (and verified), under canonical the v1 cache.
    - translator: a usbg translator (or None) from load_translator().
    - n_poses: pose draws per (case, organ). Each draw yields the target view, plus the perturbed
      start view when include_start (2x frames).
    - setup_kw: overrides for sample_setup (lateral_m, axial_m, fan_deg, roll_deg); wider ->
      more view diversity (and more incidental gallbladder/vessel coverage).
    - seat_organs: which organs to seat on (v2 anatomical seat: ("liver", "kidney")).
      Seating goes through the gym's goal_rl.sample_setup -> placement.seat_volume_under_probe, so
      the USBG_SEAT switch (anatomical|canonical) applies here exactly as at every gym call site.
      Anatomical: sample_setup draws a screened window per pose from rng and clamps the start outside
      the skin; the window index/side and clamp are recorded per frame (SimFrame.seat).
    - strict_seat: if the case lacks the organ, or the gym raises placement.NoAcceptedWindow, SKIP
      that (case, organ) and append it to `skipped`, instead of the legacy silent fallback to liver
      (which yielded liver views tagged seat_organ="kidney").
    - skip_pairs: explicit (case, organ) pairs to skip (logged in `skipped`, reason "excluded"). A skip
      happens before any rng draw, exactly like a gym NoAcceptedWindow, so the rng stream of the other
      pairs does not depend on why a pair was skipped.
    - case_meta: if a dict, filled with {case: {cache_variant, cache_path, n_windows{organ}}}.
    """
    gym_bridge.ensure_usbg_importable()
    from usbg import volume as V
    from usbg import view_match as VM
    from usbg import goal_rl as G
    from usbg import render_lotus as R
    from usbg import placement as P

    # Labels must be the CT oracle (USBG_LEARNED_SEG=0): with a learned segmenter bound, curvi_render
    # returns ITS mask as the label (circular). view_match reads the env at import, so check the module.
    if getattr(VM, "_US", None) is not None:
        raise RuntimeError("usbg.view_match has a learned segmenter bound (USBG_LEARNED_SEG!=0 at import); "
                           "sim training labels must be the CT oracle. Export USBG_LEARNED_SEG=0.")

    seat_map = _seat_map()
    setup_kw = setup_kw or {}
    seat = P.seat_mode()
    skip_pairs = {(str(c), str(o)) for c, o in (skip_pairs or ())}
    # gym contract: placement.NoAcceptedWindow (a LookupError) when a (patient, organ) has no screened
    # window. Any other exception (a bug, a bad kwarg) must propagate, never be read as a skip.
    no_window = P.NoAcceptedWindow
    rng = np.random.default_rng(seed)

    for cpath in case_paths:
        case = _case_stem(cpath)
        lv = gym_bridge.load_case(cpath)
        if case_meta is not None:
            case_meta[case] = {"cache_variant": lv.meta.get("cache_variant"),
                               "cache_path": lv.meta.get("cache_path"), "n_windows": {}}
        for organ in seat_organs:
            oid = seat_map[organ]
            if (case, organ) in skip_pairs:
                if skipped is not None:
                    skipped.append({"case": case, "organ": organ, "reason": "excluded (--skip-pairs)"})
                continue
            if not lv.has(oid):
                if strict_seat:
                    if skipped is not None:
                        skipped.append({"case": case, "organ": organ, "reason": "label absent"})
                    continue
                oid = V.LIVER  # legacy guard, mirrors every gym caller (canonical builds only)
            wmeta = None
            if seat == "anatomical":
                try:
                    from usbg import windows as W
                    wmeta = W.windows_meta(lv, oid)
                except no_window:
                    wmeta = []
                if case_meta is not None:
                    case_meta[case]["n_windows"][organ] = len(wmeta)
            for k in range(n_poses):
                # fresh per-frame randomness: pose perturbation + appearance
                try:
                    Wfv, T_tgt, T_start, meta = G.sample_setup(lv, oid, rng, return_meta=True, **setup_kw)
                except no_window as e:   # ONLY the gym's explicit "no accepted window" signal
                    if not strict_seat or k > 0:
                        raise
                    if skipped is not None:   # no accepted window for this (case, organ)
                        skipped.append({"case": case, "organ": organ, "reason": str(e)[:200]})
                    break
                smeta = dict(meta)
                if wmeta and meta.get("window") is not None:
                    w = wmeta[int(meta["window"])]
                    smeta.update(window_organ=w.get("organ"), window_id=w.get("window_id"),
                                 phi=w.get("phi"), th=w.get("th"))
                poses = [("target", T_tgt)]
                if include_start:
                    poses.append(("start", T_start))
                for pose_name, T in poses:
                    fmeta = dict(smeta)
                    if pose_name == "target":   # the skin clamp only ever moves the START pose
                        fmeta.update(start_clamped=False, clamp_mm=0.0)
                    cfg = R.domain_randomize(R.RenderConfig(), rng)
                    img, lab_canon = VM.curvi_render(lv, T, Wfv, translator, cfg=cfg, size=size)
                    lab6 = ONT.canonical_to_six(lab_canon).astype(np.uint8)
                    yield SimFrame(
                        image=np.asarray(img, np.float32),
                        label6=lab6,
                        case=case,
                        seat_organ=organ,
                        pose=pose_name,
                        organs_present=organ_presence(lab6),
                        seat=fmeta,
                    )


def _case_stem(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]
