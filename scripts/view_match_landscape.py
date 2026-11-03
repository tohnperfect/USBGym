#!/usr/bin/env python3
"""view_match_landscape.py: is the view-match reward LEARNABLE, and does per-tissue texture help?

The redefined closed-loop task (scripts/train_view_match_rl.py) rewards the agent for reproducing a
target curvilinear US view, combining organ-layout match (segmentation IoU) and appearance similarity
(render vs target). Whether an agent can solve it hinges on whether that reward is DISCRIMINATIVE: does
the match score rise monotonically as the probe pose approaches the target?

This script measures exactly that, per texture stage, without training any policy. For many target views
we sample perturbed poses across pose-distance bins and score the match under physics / CUT / CycleGAN /
SonoSPADE. Per stage it reports mean terms by pose-distance bin and, per target view, the Spearman
correlation between each term and -pose-distance (higher = more learnable).

Two seats (usbg.seatmode, env USBG_SEAT):

* canonical: the v1 landscape table protocol, kept byte-for-byte for reproduction (random organ per target
  from rng(0), seat over its centroid, DR off with RenderConfig(seed=0), each stage rendered through
  view_match.curvi_render, so the segmentation term uses whatever mask source view_match was imported
  with). The legacy JSON keys are unchanged; per-view records, a view-id-paired Wilcoxon and a
  provenance block are added. Without --translators/--tex-manifest the legacy (withdrawn, D3)
  checkpoints are used, ONLY so the shipped table stays reproducible.
* anatomical (default): target views are the screened acoustic windows (usbg.windows) of each --organs
  per case, the first --targets-per-case per organ in window order (KIDNEY interleaves right/left);
  a (case, organ) without a window is skipped and logged. Samples slide in the skin's tangent plane and
  tilt about the contact point, are clamped outside the skin (placement.clamp_start_outside_skin) and
  then pressed by |tz| (press-only, never lift-off); the clamp fraction is logged. Each view has its own
  rng and speckle seed (crc32 of its view id), so a view's samples do not depend on which other views
  run. The physics sector frame is rendered ONCE per (pose, cfg) and every stage is applied to it. Terms
  per sample and stage: appearance (sector NCC), the CT-oracle seg IoU (stage independent), the learned
  seg IoU under a per-stage segmenter (--seg-ckpt / --seg-manifest; nothing is loaded implicitly), and
  the combined reward w_seg*seg + w_app*app for each --weights pair and each seg source.
  --dr arms: off (target and samples RenderConfig(seed=s_view)), shared (one domain_randomize draw A
  for target and samples), cross (target A, samples an independent draw B held across the view's
  samples), crossseed (DR off but samples use seed s_view+1: separates speckle-seed decorrelation from
  knob DR). Tests: Wilcoxon paired BY VIEW ID (NaNs kept, pairs need both finite) across stages within an
  arm (--ref vs each other stage) and across arms within a stage (off vs each other arm).

The segmentation term is NOT texture independent under a learned mask (D11): with --seg-ckpt the
learned seg term differs per stage, and it is reported separately from the CT-oracle term.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np

LEGACY_TR = {"physics": None, "cut": "data/dataset/exp_real_cut_curvi.pt",
             "cyclegan": "data/dataset/exp_real_cyclegan_curvi.pt",
             "sonospade": "data/dataset/exp_real_sonospade_curvi.pt"}
STAGE_ORDER = ["physics", "cut", "cyclegan", "sonospade"]
BINS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 60)]                # pose-distance (mm) bins
# physics-cue ablation: turn a cue off in the RenderConfig to attribute the reward's learnability to it.
# shadow_gain=0 removes reflection shadows; speckle_strength=0 removes correlated speckle; a huge
# focal_scale stops the beam widening off-focus (depth-invariant PSF).
CUE_ABLATIONS = {
    "none": {},
    "shadow": {"shadow_gain": 0.0},
    "speckle": {"speckle_strength": 0.0},
    "psf": {"focal_scale": 1.0e6},
    "all": {"shadow_gain": 0.0, "speckle_strength": 0.0, "focal_scale": 1.0e6},
}
ARMS = ("off", "shared", "cross", "crossseed")
# anatomical sampler: slide (m), press (m), tilt (deg), roll (deg), each scaled by s ~ U(0.1, 1)
SAMPLER = dict(slide=0.04, press=0.02, tilt=30.0, roll=8.0)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)                  # all-finite input (every v1 call) is unchanged
    if not ok.all():
        x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _kv(items, flag):
    out = {}
    for kv in items or []:
        if "=" not in kv:
            raise SystemExit(f"{flag} expects stage=path, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = None if v.strip() in ("", "none", "None", "null") else v.strip()
    return out


def _finite(x):
    return x is not None and x == x


def paired_test(views, a_get, b_get):
    """Wilcoxon signed-rank of a vs b paired by view (views where both are finite)."""
    pairs = [(a_get(v), b_get(v)) for v in views]
    pairs = [(a, b) for a, b in pairs if _finite(a) and _finite(b)]
    out = {"n_pairs": len(pairs)}
    if not pairs:
        out["note"] = "no view with both values finite (degenerate)"
        return out
    a = np.asarray([p[0] for p in pairs], float); b = np.asarray([p[1] for p in pairs], float)
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float((a - b).mean()),
               winrate_a=float(np.mean(a > b)))
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a, b):
            out["note"] = "all differences zero (term identical across the pair)"
        else:
            st, p = wilcoxon(a, b)
            out.update(statistic=float(st), p_value=float(p))
    except Exception as e:                                   # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------------------------------------------------- stages
def resolve_stages(args, seat):
    """{stage: ckpt or None}. Anatomical: --tex-manifest and/or --translators are REQUIRED (the legacy
    defaults are the withdrawn D3 checkpoints). Canonical: legacy defaults, overridable."""
    stages = {}
    if args.tex_manifest:
        with open(args.tex_manifest) as fh:
            man = json.load(fh)
        for k, v in man.items():
            if k.startswith("_"):
                continue
            stages[k] = v
    stages.update(_kv(args.translators, "--translators"))
    if not stages:
        if seat == "anatomical":
            raise SystemExit("USBG_SEAT=anatomical needs --translators stage=ckpt ... or --tex-manifest "
                             "(the hard-coded defaults are the withdrawn leaky *_curvi.pt checkpoints)")
        stages = dict(LEGACY_TR)
    elif seat == "canonical":
        merged = dict(LEGACY_TR)
        merged.update(stages)
        stages = merged
    stages.setdefault("physics", None)
    order = [k for k in STAGE_ORDER if k in stages] + [k for k in stages if k not in STAGE_ORDER]
    return {k: stages[k] for k in order}


def resolve_segs(args):
    segs = _kv(args.seg_ckpt, "--seg-ckpt")
    if args.seg_manifest:
        with open(os.path.expanduser(args.seg_manifest)) as fh:
            man = json.load(fh)
        for k in list(args._stage_names):
            key = f"{k}_s{args.seg_seed}"
            if key in man and k not in segs:
                segs[k] = man[key]
    return {k: os.path.expanduser(v) for k, v in segs.items() if v}


def lineage_cases(path, depth=0):
    """Transitive training CT cases of an artifact through its provenance sidecars (None = no sidecar)."""
    from usbg import provenance as P
    if not path:
        return set()
    sc = P.sidecar_path(path)
    if not os.path.exists(sc):
        return None
    rec = P.load_provenance(sc)
    cases = set(rec.get("train_cases") or []) | set(rec.get("val_cases") or [])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for par in rec.get("parents") or []:
        pp = par.get("path", "")
        pp = pp if os.path.isabs(pp) else os.path.join(repo, pp)
        if depth < 8:
            sub = lineage_cases(pp, depth + 1)
            if sub:
                cases |= sub
    return cases


def check_lineage(paths: dict, eval_cases, allow_unchecked: bool, what: str):
    """Refuse a stage / segmenter whose training lineage touches an evaluated case; warn (or refuse,
    under the anatomical seat) when it has no provenance sidecar."""
    report = {}
    for k, p in paths.items():
        if not p:
            continue
        cs = lineage_cases(p)
        if cs is None:
            report[k] = "no provenance sidecar (lineage unchecked)"
            if not allow_unchecked:
                raise SystemExit(f"{what} {k}={p} has no provenance sidecar; pass --allow-unchecked-lineage "
                                 f"only for a descriptive legacy cell")
            print(f"WARNING: {what} {k}={p}: no provenance sidecar, lineage unchecked")
            continue
        hit = sorted(cs & set(eval_cases))
        if hit:
            raise SystemExit(f"{what} {k}={p} was trained (transitively) on evaluated cases {hit}")
        report[k] = f"ok: {len(cs)} training cases, none evaluated"
    return report


# ------------------------------------------------------------------------------------- canonical
def run_canonical(args, TR):
    """The v1 landscape table protocol, byte-for-byte (legacy loop and legacy keys)."""
    from usbg import volume as V, render_lotus as R, view_match as VM
    from usbg.placement import seat_volume_under_probe
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.texture_gan import load_translator
    from usbg.similarity import load_similarity

    translators = {k: (None if v is None else load_translator(v, args.device)) for k, v in TR.items()}
    sim = load_similarity(args.similarity, device=args.device)  # noqa: F841  (legacy: loaded, unused)
    cfg = R.RenderConfig(seed=0, **CUE_ABLATIONS[args.ablate_cue])
    if args.ablate_cue != "none":
        print(f"[ablation] physics cue disabled: {args.ablate_cue}  -> {CUE_ABLATIONS[args.ablate_cue]}")
    organs = [V.LIVER, V.KIDNEY, V.SPLEEN]
    bins = BINS
    rng = np.random.default_rng(0)

    per = {k: {"match": {b: [] for b in bins}, "app": {b: [] for b in bins}} for k in TR}
    corr_pts = {k: {"app": [], "dist": []} for k in TR}
    per_view_corr = {k: [] for k in TR}
    per_view = []

    for case in args.cases:
        lv = V.load_cache(os.path.join("data/cache", f"{case}.npz"))
        for _ti in range(args.targets_per_case):
            view_pts = {k: {"app": [], "dist": []} for k in TR}
            oid = organs[rng.integers(len(organs))]
            oid = oid if lv.has(oid) else V.LIVER
            Wfv, T_tgt = seat_volume_under_probe(lv, oid, depth_m=0.05)
            # render the target once per translator
            tgt = {k: VM.curvi_render(lv, T_tgt, Wfv, t, cfg) for k, t in translators.items()}
            for si in range(args.samples_per_target):
                s = float(rng.uniform(0.1, 1.0))          # vary magnitude so all distance bins populate
                p = pose_from_rpy(tx=rng.uniform(-.04, .04) * s, ty=rng.uniform(-.04, .04) * s,
                                  tz=rng.uniform(-.02, .02) * s, rx=rng.uniform(-30, 30) * s,
                                  ry=rng.uniform(-30, 30) * s, rz=rng.uniform(-8, 8) * s)
                T = T_tgt @ p
                dmm = VM.pose_error(T, T_tgt)[0]
                b = next((bb for bb in bins if bb[0] <= dmm < bb[1]), None)
                for k, t in translators.items():
                    img, lab = VM.curvi_render(lv, T, Wfv, t, cfg)
                    ti, tl = tgt[k]
                    app = VM.sector_appearance(img, ti)       # content-focused (sector interior)
                    m = VM.combined_score(img, lab, ti, tl)
                    if b is not None:
                        per[k]["match"][b].append(m); per[k]["app"][b].append(app)
                    corr_pts[k]["app"].append(app); corr_pts[k]["dist"].append(dmm)
                    view_pts[k]["app"].append(app); view_pts[k]["dist"].append(dmm)
            rec = {"case": case, "view_id": f"{case}/t{_ti}", "organ": V.USBG_LABEL_NAMES.get(oid, str(oid)),
                   "window": None, "corr": {}}
            for k in TR:                                   # one correlation per target view
                c = spearman(view_pts[k]["app"], [-d for d in view_pts[k]["dist"]])
                rec["corr"][k] = c
                if c == c:
                    per_view_corr[k].append(c)
            per_view.append(rec)

    n_views = len(per_view_corr["sonospade"]) if "sonospade" in per_view_corr else len(per_view)
    summary = {"ablate_cue": args.ablate_cue, "bins_mm": [list(b) for b in bins],
               "n_target_views": n_views,
               "cases": list(args.cases), "targets_per_case": args.targets_per_case,
               "samples_per_target": args.samples_per_target, "similarity": args.similarity,
               "translators": {}}
    print(f"{'translator':10s} | " + " ".join(f"{f'{b[0]}-{b[1]}mm':>9s}" for b in bins)
          + " | app~closeness (per-view mean+-std)")
    for k in TR:
        appbin = [float(np.mean(per[k]["app"][b])) if per[k]["app"][b] else float("nan") for b in bins]
        matchbin = [float(np.mean(per[k]["match"][b])) if per[k]["match"][b] else float("nan") for b in bins]
        pv = np.asarray(per_view_corr[k], float)
        summary["translators"][k] = {"app_by_bin": appbin, "match_by_bin": matchbin,
                                     "app_closeness_spearman_pooled":
                                         spearman(corr_pts[k]["app"], [-d for d in corr_pts[k]["dist"]]),
                                     "app_closeness_per_view_mean": float(pv.mean()),
                                     "app_closeness_per_view_std": float(pv.std()),
                                     "app_closeness_per_view": [float(x) for x in pv]}
        print(f"{k:10s} | " + " ".join(f"{a:9.3f}" for a in appbin) + f" |  {pv.mean():+.3f} +- {pv.std():.3f}")
    if "sonospade" in per_view_corr:
        # legacy head-to-head (pairs by list index after NaN-dropping; kept so the shipped p-values
        # reproduce) plus the corrected pairing by view id
        son = np.asarray(per_view_corr["sonospade"], float)
        print("\nSonoSPADE more-learnable-reward (higher app~closeness) per target view:")
        for k in [x for x in ["physics", "cut", "cyclegan"] if x in TR]:
            o = np.asarray(per_view_corr[k], float); n = min(len(son), len(o))
            win = float(np.mean(son[:n] > o[:n]))
            summary["translators"][k]["sonospade_winrate_vs_this"] = win
            try:
                from scipy.stats import wilcoxon
                stat, pval = wilcoxon(son[:n], o[:n])
                summary["translators"][k]["wilcoxon_vs_sonospade"] = {
                    "statistic": float(stat), "p_value": float(pval), "n_pairs": int(n)}
                sig = "p < 0.05" if pval < 0.05 else "p >= 0.05"
                print(f"  vs {k:9s}: SonoSPADE higher on {win*100:.0f}% of {n} views  "
                      f"(d={son[:n].mean()-o[:n].mean():+.3f}, Wilcoxon p={pval:.4g}, {sig})")
            except Exception as e:
                summary["translators"][k]["wilcoxon_vs_sonospade"] = {"error": str(e)}
                print(f"  vs {k:9s}: SonoSPADE higher on {win*100:.0f}% of {n} views  (d={son[:n].mean()-o[:n].mean():+.3f})")
            summary["translators"][k]["wilcoxon_vs_sonospade_by_view_id"] = paired_test(
                per_view, lambda v: v["corr"]["sonospade"], lambda v, k=k: v["corr"][k])
    summary["per_view"] = per_view
    summary["seat_mode"] = "canonical"
    summary["mask_source"] = "learned (view_match import-time default)" if VM._US is not None else "ct_oracle"
    summary["note"] = ("v1 protocol: 'views' are random organ draws per case; the seat is deterministic per "
                       "(case, organ), so repeated draws are repeated poses (pseudo-replication).")
    return summary


# ------------------------------------------------------------------------------------- anatomical
def contact_sample(T_tgt, rng):
    """One landscape sample around a window pose: (T, dist-free draws, lift_m, press_m)."""
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.placement import clamp_start_outside_skin
    c = SAMPLER
    s = float(rng.uniform(0.1, 1.0))
    tx, ty = rng.uniform(-c["slide"], c["slide"]) * s, rng.uniform(-c["slide"], c["slide"]) * s
    tz = rng.uniform(-c["press"], c["press"]) * s
    rx, ry = rng.uniform(-c["tilt"], c["tilt"]) * s, rng.uniform(-c["tilt"], c["tilt"]) * s
    rz = rng.uniform(-c["roll"], c["roll"]) * s
    T, lift = clamp_start_outside_skin(T_tgt @ pose_from_rpy(tx=tx, ty=ty, tz=0.0, rx=rx, ry=ry, rz=rz))
    T = T.copy()
    T[2, 3] -= abs(tz)                                    # press-only, never lift-off
    return T, float(lift), float(abs(tz))


def view_seed(view_id, seed):
    return int(zlib.crc32(f"{seed}:{view_id}".encode()) & 0x7FFFFFFF)


def run_anatomical(args, TR):
    from dataclasses import replace
    from usbg import volume as V, render_lotus as R, view_match as VM, slicer as S, windows as W
    from usbg.placement import seat_volume_under_probe
    from usbg.segmenter import align_label_to_image
    from usbg.texture_gan import load_translator
    from usbg.us_segmenter import load_segmenter

    VM.set_mask_source(None)                # this path never reads view_match's import-time segmenter
    cue = CUE_ABLATIONS[args.ablate_cue]
    translators = {k: (None if v is None else load_translator(v, args.device)) for k, v in TR.items()}
    seg_paths = resolve_segs(args)
    segs = {k: load_segmenter(p, device=args.seg_device or args.device) for k, p in seg_paths.items()
            if k in TR}
    if seg_paths and set(seg_paths) - set(TR):
        print(f"note: segmenters for stages not evaluated ignored: {sorted(set(seg_paths) - set(TR))}")
    weights = [tuple(float(x) for x in w.split("/")) for w in args.weights]
    organ_label = {"liver": V.LIVER, "kidney": V.KIDNEY, "gallbladder": V.GALLBLADDER}

    def phys_sector(lab_lin, cfg):          # exactly view_match.curvi_render's physics + warp steps
        img_lin = R.render(lab_lin, "physics", device=args.physics_device, cfg=cfg)
        return VM.warp_to_sector(VM._resize01(img_lin, 128), align_label_to_image(lab_lin, (128, 128)))

    def tex(k, wi, wl):
        t = translators[k]
        if getattr(t, "wants_label", False):
            return np.asarray(t.translate_aligned(wi, wl), np.float32)
        return np.asarray(t(wi) if t is not None else wi, np.float32)

    def cfgs(arm, vseed):
        A = replace(R.domain_randomize(R.RenderConfig(), np.random.default_rng(vseed)), **cue)
        B = replace(R.domain_randomize(R.RenderConfig(), np.random.default_rng(vseed + 7)), **cue)
        off = R.RenderConfig(seed=vseed, **cue)
        return {"off": (off, off), "shared": (A, A), "cross": (A, B),
                "crossseed": (off, R.RenderConfig(seed=vseed + 1, **cue))}[arm]

    terms = ["app", "seg_ct"] + (["seg_learned"] if segs else [])
    comb = [(f"combined_ct_{ws:g}/{wa:g}", "seg_ct", ws, wa) for ws, wa in weights]
    if segs:
        comb += [(f"combined_learned_{ws:g}/{wa:g}", "seg_learned", ws, wa) for ws, wa in weights]
    all_terms = terms + [c[0] for c in comb]
    stages = list(TR)
    binacc = {a: {k: {t: [[] for _ in BINS] for t in all_terms} for k in stages} for a in args.dr}
    pooled = {a: {k: {"app": [], "dist": []} for k in stages} for a in args.dr}
    per_view, skipped, wcount = [], [], {}
    lifts, presses = [], []
    t0 = time.time()
    for case in args.cases:
        lv = V.load_cache(os.path.join("data/cache", f"{case}.npz"))
        wcount[case] = {}
        for organ in args.organs:
            label = organ_label[organ]
            try:
                metas = W.windows_meta(lv, label)
                why = "no accepted window for this organ"
            except W.NoAcceptedWindow as e:
                metas, why = [], str(e)
            wcount[case][organ] = len(metas)
            if not metas:
                skipped.append({"case": case, "organ": organ, "reason": why})
                print(f"[skip] {case} {organ}: {why}")
                continue
            take = metas[:args.targets_per_case]
            if len(take) < args.targets_per_case:
                print(f"[short] {case} {organ}: {len(take)} of {args.targets_per_case} requested windows")
            for wi, m in enumerate(take):
                tv = time.time()
                Wfv, T_tgt = seat_volume_under_probe(lv, label, window=wi)
                vid = f"{case}/{m['organ']}/w{m['window_id']}"
                vseed = view_seed(vid, args.seed)
                rng = np.random.default_rng(vseed)
                samples, vl = [], []
                for _ in range(args.samples_per_target):
                    T, lift, press = contact_sample(T_tgt, rng)
                    samples.append(T); vl.append(lift); lifts.append(lift); presses.append(press)
                dist = np.array([VM.pose_error(T, T_tgt)[0] for T in samples])
                bidx = [next((i for i, bb in enumerate(BINS) if bb[0] <= d < bb[1]), None) for d in dist]
                lab_t = S.slice_at_pose(lv, T_tgt, Wfv)
                labs = [S.slice_at_pose(lv, T, Wfv) for T in samples]
                rec = {"view_id": vid, "case": case, "organ": m["organ"], "target": organ,
                       "window_index": wi, "window_id": int(m["window_id"]), "n_windows": len(metas),
                       "vseed": vseed, "clamp": {"lifted_frac": float(np.mean(np.asarray(vl) > 1e-4)),
                                                 "lift_mm_mean": float(np.mean(vl) * 1e3)},
                       "terms": {}}
                for arm in args.dr:
                    cfg_t, cfg_o = cfgs(arm, vseed)
                    wi_t, wl_t = phys_sector(lab_t, cfg_t)
                    tgt = {k: tex(k, wi_t, wl_t) for k in stages}
                    tgt_seg = {k: segs[k].segment(tgt[k]) for k in segs}
                    vals = {k: {t: [] for t in terms} for k in stages}
                    for si, lab in enumerate(labs):
                        wi_s, wl_s = phys_sector(lab, cfg_o)
                        seg_ct = VM.seg_organ_iou(wl_s, wl_t)
                        for k in stages:
                            img = tex(k, wi_s, wl_s)
                            vals[k]["app"].append(VM.sector_appearance(img, tgt[k]))
                            vals[k]["seg_ct"].append(seg_ct)
                            if segs:
                                vals[k]["seg_learned"].append(
                                    VM.seg_organ_iou(segs[k].segment(img), tgt_seg[k]) if k in segs else float("nan"))
                    rec["terms"][arm] = {}
                    for k in stages:
                        v = {t: np.asarray(vals[k][t], float) for t in terms}
                        for name, src, ws, wa in comb:
                            v[name] = ws * v[src] + wa * v["app"]
                        for t in all_terms:
                            for i, b in enumerate(bidx):
                                if b is not None and v[t][i] == v[t][i]:
                                    binacc[arm][k][t][b].append(float(v[t][i]))
                        pooled[arm][k]["app"] += v["app"].tolist(); pooled[arm][k]["dist"] += dist.tolist()
                        for t in all_terms:
                            rec["terms"][arm].setdefault(t, {})[k] = spearman(v[t], -dist)
                rec["corr"] = dict(rec["terms"][args.dr[0]]["app"])       # primary arm, appearance
                per_view.append(rec)
                print(f"  {vid:28s} {time.time()-tv:6.1f}s  app~closeness[{args.dr[0]}] " +
                      " ".join(f"{k}={rec['corr'][k]:+.3f}" for k in stages), flush=True)

    li = np.asarray(lifts, float)
    out = {"seat_mode": "anatomical", "cases": list(args.cases), "organs": list(args.organs),
           "targets_per_case": args.targets_per_case, "samples_per_target": args.samples_per_target,
           "bins_mm": [list(b) for b in BINS], "ablate_cue": args.ablate_cue, "dr_arms": list(args.dr),
           "weights": args.weights, "seed": args.seed, "stages": TR, "seg_ckpts": seg_paths,
           "terms": all_terms, "n_target_views": len(per_view), "window_counts": wcount, "skipped": skipped,
           "sampler": {**SAMPLER, "scale": "s ~ U(0.1, 1) per sample",
                       "rule": "slide+tilt about the contact point, clamp outside the skin, press |tz|"},
           "clamp": {"n_samples": int(li.size), "clamped_frac": float((li > 1e-4).mean()) if li.size else 0.0,
                     "lift_mm_mean": float(li.mean() * 1e3) if li.size else 0.0,
                     "lift_mm_max": float(li.max() * 1e3) if li.size else 0.0,
                     "press_mm_mean": float(np.mean(presses) * 1e3) if presses else 0.0},
           "mask_source": {"seg_ct": "CT-oracle label of the sector slice (stage independent)",
                           "seg_learned": ({k: seg_paths[k] for k in segs} if segs else None)},
           "render": {"physics_device": args.physics_device or "default (render_lotus.get_device)",
                      "translator_device": args.device},
           "results": {}, "per_view": per_view, "tests": {"across_stages": {}, "across_arms": {}},
           "timing_s": round(time.time() - t0, 1)}
    for arm in args.dr:
        out["results"][arm] = {}
        for k in stages:
            r = {f"{t}_by_bin": [float(np.mean(x)) if x else float("nan") for x in binacc[arm][k][t]]
                 for t in all_terms}
            for t in all_terms:
                pv = np.asarray([v["terms"][arm][t][k] for v in per_view], float)
                f = pv[np.isfinite(pv)]
                r[t] = {"per_view_mean": float(f.mean()) if f.size else float("nan"),
                        "per_view_std": float(f.std()) if f.size else float("nan"),
                        "n_views": int(f.size), "per_view": [float(x) for x in pv]}
            r["app_closeness_spearman_pooled"] = spearman(pooled[arm][k]["app"], [-d for d in pooled[arm][k]["dist"]])
            out["results"][arm][k] = r
    ref = args.ref if args.ref in stages else None
    for arm in args.dr:
        out["tests"]["across_stages"][arm] = {}
        for t in all_terms:
            if ref is None:
                break
            out["tests"]["across_stages"][arm][t] = {
                k: paired_test(per_view, lambda v, t=t: v["terms"][arm][t][ref],
                               lambda v, t=t, k=k: v["terms"][arm][t][k])
                for k in stages if k != ref}
    if len(args.dr) > 1:
        base = args.dr[0]
        for k in stages:
            out["tests"]["across_arms"][k] = {
                t: {a: paired_test(per_view, lambda v, t=t: v["terms"][base][t][k],
                                   lambda v, t=t, a=a: v["terms"][a][t][k]) for a in args.dr[1:]}
                for t in all_terms}
    # legacy-shaped summary of the primary arm (appearance term), for older readers and the figure
    arm = args.dr[0]
    out["translators"] = {}
    for k in stages:
        r = out["results"][arm][k]
        pv = [x for x in r["app"]["per_view"] if x == x]
        d = {"app_by_bin": r["app_by_bin"], "match_by_bin": r.get(f"combined_ct_0.5/0.5_by_bin"),
             "match_by_bin_note": "CT-oracle seg IoU, 0.5/0.5 weights (legacy combined_score)",
             "app_closeness_spearman_pooled": r["app_closeness_spearman_pooled"],
             "app_closeness_per_view_mean": r["app"]["per_view_mean"],
             "app_closeness_per_view_std": r["app"]["per_view_std"], "app_closeness_per_view": pv}
        if ref and k != ref:
            tt = out["tests"]["across_stages"][arm]["app"][k]
            d[f"{ref}_winrate_vs_this"] = tt.get("winrate_a")
            d[f"wilcoxon_vs_{ref}"] = tt
        out["translators"][k] = d
    print(f"\n[{arm}] {'stage':10s} | " + " ".join(f"{f'{b[0]}-{b[1]}mm':>9s}" for b in BINS)
          + " | app~closeness per view (mean+-std)")
    for k in stages:
        d = out["translators"][k]
        print(f"[{arm}] {k:10s} | " + " ".join(f"{a:9.3f}" for a in d["app_by_bin"])
              + f" |  {d['app_closeness_per_view_mean']:+.3f} +- {d['app_closeness_per_view_std']:.3f}")
        if ref and k != ref:
            tt = d[f"wilcoxon_vs_{ref}"]
            print(f"        {ref} vs {k}: win {tt.get('winrate_a', float('nan'))*100:.0f}% of {tt['n_pairs']} "
                  f"views, p={tt.get('p_value', float('nan')):.4g} (paired by view id; descriptive)")
    print(f"clamp outside skin: {out['clamp']['clamped_frac']*100:.1f}% of samples lifted "
          f"(mean {out['clamp']['lift_mm_mean']:.2f} mm); press mean {out['clamp']['press_mm_mean']:.1f} mm; "
          f"skipped {[(s['case'], s['organ']) for s in skipped]}")
    return out


# ------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=["s1086", "s0720", "s0011", "s0344", "s1373"])
    ap.add_argument("--targets-per-case", type=int, default=4,
                    help="canonical: random target draws per case; anatomical: windows PER ORGAN per case")
    ap.add_argument("--samples-per-target", type=int, default=40)
    ap.add_argument("--similarity", default="ssim", help="canonical only (loaded, unused, as in v1)")
    ap.add_argument("--out", default="outputs/view_match/landscape.json")
    ap.add_argument("--device", default="cpu", help="translators (and segmenters unless --seg-device)")
    ap.add_argument("--physics-device", default=None,
                    help="anatomical: physics render device (default: render_lotus.get_device(), as curvi_render)")
    ap.add_argument("--ablate-cue", default="none", choices=list(CUE_ABLATIONS),
                    help="disable a physics cue to attribute reward learnability to it: "
                         "shadow (no acoustic shadowing), speckle (no correlated speckle), "
                         "psf (depth-invariant point spread), or all")
    ap.add_argument("--translators", nargs="+", default=None, metavar="STAGE=CKPT",
                    help="texture stages, e.g. physics= cut=... cyclegan=... sonospade=... (required under "
                         "the anatomical seat unless --tex-manifest)")
    ap.add_argument("--translator", action="append", default=[], metavar="STAGE=CKPT",
                    help="repeatable single-stage form of --translators")
    ap.add_argument("--tex-manifest", default=None,
                    help="JSON {stage: ckpt|null} (outputs/v2/textures/manifest.json)")
    ap.add_argument("--organs", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "gallbladder"],
                    help="anatomical: organs whose accepted windows are the target views")
    ap.add_argument("--dr", nargs="+", default=["off"], choices=ARMS, help="anatomical: DR arms (first = primary)")
    ap.add_argument("--seg-ckpt", nargs="*", default=[], metavar="STAGE=CKPT",
                    help="anatomical: per-stage learned segmenter for the learned seg term")
    ap.add_argument("--seg-manifest", default=None,
                    help="segmenter repo outputs/v2/seg_manifest.json {<stage>_s<seed>: path}")
    ap.add_argument("--seg-seed", type=int, default=0, help="which segmenter seed --seg-manifest supplies")
    ap.add_argument("--seg-device", default=None)
    ap.add_argument("--weights", nargs="+", default=["0.6/0.4", "0.5/0.5"], metavar="WSEG/WAPP",
                    help="anatomical: combined-reward weight pairs")
    ap.add_argument("--ref", default="sonospade", help="reference stage of the across-stage tests")
    ap.add_argument("--seed", type=int, default=0, help="anatomical: per-view rng/speckle seed salt")
    ap.add_argument("--allow-unchecked-lineage", action="store_true",
                    help="anatomical: accept a stage/segmenter without a provenance sidecar (descriptive cells)")
    args = ap.parse_args(argv)
    args.translators = (args.translators or []) + args.translator

    from usbg.placement import seat_mode
    from usbg import provenance as P
    seat = seat_mode()
    TR = resolve_stages(args, seat)
    args._stage_names = list(TR)
    if seat == "canonical":
        if args.dr != ["off"] or args.seg_ckpt or args.seg_manifest:
            raise SystemExit("--dr/--seg-ckpt/--seg-manifest are anatomical-seat options; the canonical "
                             "branch is the v1 protocol (DR off, view_match's mask source)")
        summary = run_canonical(args, TR)
        lineage = {k: ("no sidecar" if p and lineage_cases(p) is None else "ok") for k, p in TR.items() if p}
    else:
        if "0.5/0.5" not in args.weights:
            args.weights = list(args.weights) + ["0.5/0.5"]       # legacy match_by_bin needs it
        lineage = check_lineage(TR, args.cases, args.allow_unchecked_lineage, "texture stage")
        lineage.update({f"seg:{k}": v for k, v in
                        check_lineage(resolve_segs(args), args.cases, args.allow_unchecked_lineage,
                                      "segmenter").items()})
        summary = run_anatomical(args, TR)
    segp = summary.get("seg_ckpts") or {}
    summary["provenance"] = P.collect(
        stages={k: P.texture_info(k, p) for k, p in TR.items()},
        stage_sidecars={k: (P.sidecar_path(p) if p and os.path.exists(P.sidecar_path(p)) else None)
                        for k, p in TR.items()},
        seg_ckpts={k: {"path": os.path.abspath(p), "md5": P.md5(p)} for k, p in segp.items()},
        lineage=lineage, cohort={"eval": list(args.cases)}, dr_arms=list(args.dr),
        stage_blocks={k: P.result_block(texture=p, segmenter=segp.get(k),
                                        mask_source=("learned" if segp.get(k) else "ct") if seat == "anatomical"
                                        else summary.get("mask_source"),
                                        dr_mode=",".join(args.dr), eval_cases=args.cases,
                                        target_set=list(args.organs) if seat == "anatomical" else
                                        ["liver", "kidney", "spleen"])
                      for k, p in TR.items()})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    _save_figure(summary, args.out.replace(".json", ".png"))
    print(f"wrote {args.out} and {args.out.replace('.json', '.png')}")
    return 0


def _save_figure(d, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    order = [k for k in STAGE_ORDER if k in d["translators"]] + \
        [k for k in d["translators"] if k not in STAGE_ORDER]
    col = {"physics": "#888", "cut": "#e08a2b", "cyclegan": "#c0392b", "sonospade": "#2e86c1"}
    binlab = [f"{a}-{b}" for a, b in d["bins_mm"]]; x = np.arange(len(binlab))
    # The paper renders this at \columnwidth (~3.5in). Authoring it 11in wide meant a ~3.1x
    # downscale, so 8pt labels reached the page at ~2.6pt and were unreadable. Author it near its
    # final size and the type survives the scale.
    FS = 9
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for k in order:
        ax[0].plot(x, d["translators"][k]["app_by_bin"], marker="o", ms=3.5, color=col.get(k, "#555"),
                   lw=2 if k == "sonospade" else 1.2, label=k)
    ax[0].set_xticks(x); ax[0].set_xticklabels(binlab, fontsize=FS - 1)
    ax[0].set_xlabel("pose distance (mm)", fontsize=FS)
    ax[0].set_ylabel("appearance match", fontsize=FS)
    ax[0].tick_params(labelsize=FS - 1)
    ax[0].legend(fontsize=FS - 1, frameon=False)
    ax[0].set_title("steeper = more pose-sensitive", fontsize=FS)
    means = [d["translators"][k]["app_closeness_per_view_mean"] for k in order]
    stds = [d["translators"][k]["app_closeness_per_view_std"] for k in order]
    ax[1].bar(range(len(order)), means, yerr=stds, color=[col.get(k, "#555") for k in order], capsize=3,
              edgecolor="k", lw=.5)
    ax[1].set_xticks(range(len(order)))
    ax[1].set_xticklabels(order, fontsize=FS - 1, rotation=12)
    ax[1].set_ylabel("app~closeness corr", fontsize=FS)
    ax[1].tick_params(labelsize=FS - 1)
    ax[1].set_title("higher = more learnable", fontsize=FS)
    # no suptitle: the LaTeX caption already says what the figure is, and it cost vertical space
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
