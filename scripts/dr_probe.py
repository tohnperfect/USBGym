#!/usr/bin/env python3
"""dr_probe.py: every knob of the domain randomization (DR), with the MEASURED effect of each.

For every knob render_lotus.domain_randomize draws (the table is read from render_lotus.DR_RANGES, and
checked at start against what domain_randomize actually draws, so it cannot drift from the code): the
default, range, distribution and sampling unit per code path, and the mean |pixel change| in the
eroded sector of the frame the policy sees (sector-warped physics render, optionally after each texture
stage) when the knob ALONE moves to each end of its range, speckle seed held fixed.

Poses: --poses-per-case per patient at its screened acoustic windows (usbg.windows; organs alternate,
windows round-robin), jittered by a slide of +/-2 cm in the skin's tangent plane, tilt +/-10 deg and
roll +/-5 deg about the contact point, clamped outside the skin, then pressed 0 to 5 mm.

Per-label knobs (z_scale, mu_scale, bs_scale) are measured three ways:
  per_label : one label at a time to lo/hi (labels with >= 1% of in-sector pixels); mean and max label
  uniform   : every label together (DEGENERATE: the Fresnel coefficient is scale invariant, so uniform
              z_scale is inert except at the gel/skin interface, whose gel impedance is not scaled;
              uniform bs_scale largely cancels in the percentile white level)
  drawn     : the knob alone drawn per label exactly as domain_randomize draws it (5 draws)
Reference rows: a speckle-seed-only change, a 5 mm in-plane slide at the default config, and a full DR
draw (5 draws, speckle seed fixed).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from dataclasses import asdict, fields, replace

import numpy as np

UNITS = {
    "mujoco_env / goal_rl.GoalRender.episode_cfg (randomize=True)": "per episode (target and observations share it)",
    "train_view_match_rl.py --randomize episode / --eval-randomize episode": "per episode",
    "make_dataset.py clips (--randomize)": "per clip (held across its frames)",
    "make_dataset.py --pairs-dr full (texture-stage pairs)": "per frame",
    "eval_realism.py --sector --dr full|both": "per view",
    "view_match_landscape.py --dr shared|cross": "per target view (cross: a second draw for the samples)",
    "Tables III/IV, view_match_landscape --dr off": "off (speckle seed only)",
}
# scripts/make_dataset.py::_mild_randomize (the pairs_mild profile; NOT used by any v2 artifact)
MILD = {"gain": (0.85, 1.15), "tgc_slope": (0.7, 1.3), "psf_sigma_lat0": (0.9, 1.4),
        "psf_focus_depth": (0.03, 0.08), "speckle_strength": (0.7, 1.1), "noise_sigma": (0.0, 0.03),
        "dyn_range_db": (48, 56)}


class _RecRng:
    """Stand-in rng that records every uniform(lo, hi) call and returns a unique marker value."""

    def __init__(self):
        self.calls = []

    def uniform(self, lo, hi):
        self.calls.append((float(lo), float(hi)))
        return 1000.0 + len(self.calls)

    def integers(self, *a, **k):
        return 0


def introspect_dr(R):
    """(table, problems): what domain_randomize draws, recovered by marker values, vs R.DR_RANGES."""
    rec = _RecRng()
    cfg = R.domain_randomize(R.RenderConfig(), rec)
    marker = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, dict):
            for L, x in v.items():
                if isinstance(x, float) and x >= 1000.0:
                    marker[int(round(x - 1000.0))] = (f.name, L)
        elif isinstance(v, float) and v >= 1000.0:
            marker[int(round(v - 1000.0))] = (f.name, None)
    drawn = {}
    for i, (lo, hi) in enumerate(rec.calls, start=1):
        name, L = marker[i]
        drawn.setdefault(name, set()).add((lo, hi))
    problems = []
    table = getattr(R, "DR_RANGES", None)
    if table is None:
        problems.append("render_lotus.DR_RANGES missing")
    else:
        if set(table) != set(drawn):
            problems.append(f"DR_RANGES keys {sorted(table)} != drawn fields {sorted(drawn)}")
        for k, rngs in drawn.items():
            if len(rngs) != 1 or (k in table and tuple(map(float, table[k])) not in rngs):
                problems.append(f"{k}: drawn ranges {sorted(rngs)} vs DR_RANGES {table.get(k)}")
    order = [marker[i][0] for i in range(1, len(rec.calls) + 1)]
    seen = []
    for k in order:
        if k not in seen:
            seen.append(k)
    if table is not None and seen != list(table):
        problems.append(f"draw order {seen} != DR_RANGES order {list(table)}")
    a = R.domain_randomize(R.RenderConfig(), np.random.default_rng(1))
    b = R.domain_randomize(R.RenderConfig(), np.random.default_rng(2))
    changed = sorted(f.name for f in fields(a) if asdict(a)[f.name] != asdict(b)[f.name] and f.name != "seed")
    if table is not None and set(changed) != set(table):
        problems.append(f"fields changed between two draws {changed} != DR_RANGES keys {sorted(table)}")
    return {k: sorted(v)[0] for k, v in drawn.items()}, seen, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=None, help="default: benchmark.TEST_PATIENTS")
    ap.add_argument("--organs", nargs="+", default=["liver", "kidney"], choices=["liver", "kidney", "gallbladder"])
    ap.add_argument("--poses-per-case", type=int, default=10)
    ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--through", nargs="*", default=[], metavar="STAGE=CKPT", help="texture stages")
    ap.add_argument("--tex-manifest", default=None, help="JSON {stage: ckpt|null}")
    ap.add_argument("--draws", type=int, default=5, help="drawn / full-DR reference draws per pose")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-canonical", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex", default=None, help="also write a LaTeX table here")
    args = ap.parse_args(argv)

    from scipy.ndimage import binary_erosion
    from usbg import render_lotus as R, view_match as VM, slicer as S
    from usbg.segmenter import align_label_to_image
    from usbg.contracts_bridge import pose_from_rpy
    from usbg.texture_gan import load_translator
    from usbg.placement import seat_mode
    from usbg import provenance as P
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_realism import window_views, resolve_case

    if seat_mode() == "canonical" and not args.allow_canonical:
        raise SystemExit("dr_probe measures at the anatomical windows; USBG_SEAT=canonical needs --allow-canonical")
    if args.cases is None:
        from usbg.benchmark import TEST_PATIENTS
        args.cases = list(TEST_PATIENTS)
    drawn, order, problems = introspect_dr(R)
    if problems:
        raise SystemExit("DR table drifted from render_lotus.domain_randomize:\n  " + "\n  ".join(problems))
    per_label = [k for k in order if k in getattr(R, "DR_PER_LABEL", ("z_scale", "mu_scale", "bs_scale"))]
    scalar = [k for k in order if k not in per_label]

    t0 = time.time()
    _, _, M = VM._sector_coords(128, 128, VM.SECTOR)
    Me = binary_erosion(M, iterations=args.erode) if args.erode > 0 else M
    paths = {}
    if args.tex_manifest:
        with open(args.tex_manifest) as fh:
            paths.update({k: v for k, v in json.load(fh).items() if v and not k.startswith("_")})
    for kv in args.through:
        k, p = kv.split("=", 1)
        if p:
            paths[k] = p
    stages = {"physics": None, **{k: load_translator(p, args.device) for k, p in paths.items()}}

    def frames(lab, cfg):
        """{stage: sector frame} from ONE physics render."""
        img = R.render(lab, "physics", device=args.device, cfg=cfg)
        wi, wl = VM.warp_to_sector(VM._resize01(img, 128), align_label_to_image(lab, (128, 128)))
        out = {}
        for s, tr in stages.items():
            x = wi if tr is None else (tr.translate_aligned(wi, wl) if getattr(tr, "wants_label", False) else tr(wi))
            out[s] = np.asarray(x, np.float32)
        return out, wl

    acc = {s: {} for s in stages}

    def add(key, a, b):
        for s in stages:
            acc[s].setdefault(key, []).append(float(np.abs(a[s] - b[s])[Me].mean()))

    n_render, skipped, wcount, lifts = 0, [], {}, []
    ranges = dict(slide=0.02, tilt=10.0, roll=5.0)
    for case in args.cases:
        lv = resolve_case(case)
        views, sk, cnt = window_views(lv, case, args.organs, args.poses_per_case, args.seed + 10, ranges, 0.005)
        skipped += sk; wcount[case] = cnt
        for vi, v in enumerate(views):
            lab = v["lab"]; lifts.append(v["lift"])
            seed = int(zlib.crc32(f"dr:{args.seed}:{case}:{vi}".encode()) & 0x7FFFFFFF)
            base = R.RenderConfig(seed=seed)
            ref, wl = frames(lab, base); n_render += 1
            u, c = np.unique(wl[Me], return_counts=True)
            present = [int(L) for L, k in zip(u, c) if L in R._LABELS and k >= 0.01 * Me.sum()]
            for k in scalar:
                lo, hi = R.DR_RANGES[k]
                for end, val in (("lo", lo), ("hi", hi)):
                    add(f"{k}|{end}", frames(lab, replace(base, **{k: float(val)}))[0], ref); n_render += 1
            for k in per_label:
                lo, hi = R.DR_RANGES[k]
                for end, val in (("lo", lo), ("hi", hi)):
                    per = {s: [] for s in stages}
                    for L in present:
                        f, _ = frames(lab, replace(base, **{k: {L: float(val)}})); n_render += 1
                        for s in stages:
                            per[s].append(float(np.abs(f[s] - ref[s])[Me].mean()))
                    for s in stages:
                        acc[s].setdefault(f"{k}|per_label_mean|{end}", []).append(float(np.mean(per[s])) if per[s] else 0.0)
                        acc[s].setdefault(f"{k}|per_label_max|{end}", []).append(float(np.max(per[s])) if per[s] else 0.0)
                    add(f"{k}|uniform|{end}", frames(lab, replace(base, **{k: {L: float(val) for L in R._LABELS}}))[0], ref)
                    n_render += 1
                dr_rng = np.random.default_rng(seed)
                for _ in range(args.draws):
                    draw = {L: float(dr_rng.uniform(lo, hi)) for L in R._LABELS}
                    add(f"{k}|drawn", frames(lab, replace(base, **{k: draw}))[0], ref); n_render += 1
            add("ref|speckle_seed_only", frames(lab, base.with_seed(seed + 1))[0], ref)
            slide = S.slice_at_pose(lv, v["T"] @ pose_from_rpy(tx=0.005), v["Wfv"])     # 5 mm in-plane slide
            add("ref|pose_5mm_slide", frames(slide, base)[0], ref)
            for j in range(args.draws):
                full = replace(R.domain_randomize(R.RenderConfig(), np.random.default_rng(seed * 10 + j)), seed=seed)
                add("ref|full_DR_draw", frames(lab, full)[0], ref)
            n_render += 2 + args.draws
        print(f"  {case}: {len(views)} poses ({cnt}), {n_render} renders so far, {time.time()-t0:.0f}s", flush=True)

    def summ(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}
    dflt = R.RenderConfig()
    out = {"seat_mode": seat_mode(), "cases": list(args.cases), "organs": list(args.organs),
           "poses_per_case": args.poses_per_case, "erode_px": args.erode, "window_counts": wcount,
           "skipped": skipped, "jitter": {**ranges, "press_max_mm": 5.0},
           "clamp": {"clamped_frac": float((np.asarray(lifts) > 1e-4).mean()) if lifts else 0.0},
           "metric": "mean |pixel change| in the eroded sector, [0,1] intensity, speckle seed fixed",
           "dr_draw_order": order, "knobs": {}, "not_randomized": {}, "mild_profile_pairs_mild": MILD,
           "sampling_units": UNITS, "stages": paths,
           "effects": {s: {k: summ(v) for k, v in d.items()} for s, d in acc.items()},
           "n_renders": n_render, "wall_s": round(time.time() - t0, 1)}
    for k in order:
        lo, hi = R.DR_RANGES[k]
        out["knobs"][k] = {"default": 1.0 if k in per_label else getattr(dflt, k), "range": [lo, hi],
                           "distribution": "uniform, independent per label" if k in per_label else "uniform",
                           "per_label": k in per_label}
    out["knobs"]["seed"] = {"default": 0, "range": [0, 2 ** 31 - 2], "distribution": "uniform integer",
                            "per_label": False, "note": "speckle + electronic-noise field"}
    for f in fields(dflt):
        if f.name not in R.DR_RANGES and f.name != "seed":
            out["not_randomized"][f.name] = getattr(dflt, f.name) if not isinstance(getattr(dflt, f.name), dict) else "1.0 per label"
    out["provenance"] = P.collect(stages={k: P.texture_info(k, p) for k, p in paths.items()},
                                  cohort={"eval": list(args.cases)})
    e = out["effects"]["physics"]
    print(f"{'knob':18s} {'lo':>8s} {'hi':>8s}   (mean |dpix| x100, physics, in-sector)")
    for k in scalar:
        print(f"{k:18s} {100*e[k+'|lo']['mean']:8.3f} {100*e[k+'|hi']['mean']:8.3f}")
    for k in per_label:
        print(f"{k:18s} per-label {100*e[k+'|per_label_mean|lo']['mean']:.3f}/{100*e[k+'|per_label_mean|hi']['mean']:.3f}"
              f"  max-label {100*e[k+'|per_label_max|lo']['mean']:.3f}/{100*e[k+'|per_label_max|hi']['mean']:.3f}"
              f"  uniform {100*e[k+'|uniform|lo']['mean']:.3f}/{100*e[k+'|uniform|hi']['mean']:.3f}"
              f"  drawn {100*e[k+'|drawn']['mean']:.3f}")
    for k in ("ref|speckle_seed_only", "ref|pose_5mm_slide", "ref|full_DR_draw"):
        if k in e:
            print(f"{k:24s} {100*e[k]['mean']:.3f}")
    print("renders", n_render, "wall_s", out["wall_s"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    if args.tex:
        write_tex(out, args.tex, scalar, per_label)
    print(f"wrote {args.out}" + (f" and {args.tex}" if args.tex else ""))
    return 0


def write_tex(out, path, scalar, per_label):
    """knob | default | range | effect at lo / hi (x100, physics and each stage)."""
    st = list(out["effects"])
    def cell(s, key):
        e = out["effects"][s].get(key)
        return f"{100 * e['mean']:.2f}" if e else "--"
    lines = ["% generated by scripts/dr_probe.py; mean |pixel change| x100 in the eroded sector",
             "\\begin{tabular}{l l l " + " ".join("r r" for _ in st) + "}", "\\hline",
             "knob & default & range & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{s} lo / hi}}" for s in st)
             + " \\\\", "\\hline"]
    for k in scalar:
        kn = out["knobs"][k]
        lines.append(f"{k.replace('_', chr(92) + '_')} & {kn['default']:g} & [{kn['range'][0]:g}, {kn['range'][1]:g}] & "
                     + " & ".join(f"{cell(s, k + '|lo')} & {cell(s, k + '|hi')}" for s in st) + " \\\\")
    for k in per_label:
        kn = out["knobs"][k]
        lines.append(f"{k.replace('_', chr(92) + '_')} (per label) & 1 & [{kn['range'][0]:g}, {kn['range'][1]:g}] & "
                     + " & ".join(f"{cell(s, k + '|per_label_mean|lo')} & {cell(s, k + '|per_label_mean|hi')}"
                                  for s in st) + " \\\\")
    for key, lab in (("ref|speckle_seed_only", "speckle seed only"), ("ref|pose_5mm_slide", "5 mm slide"),
                     ("ref|full_DR_draw", "full DR draw")):
        lines.append(f"\\textit{{{lab}}} & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cell(s, key)}}}" for s in st)
                     + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
