#!/usr/bin/env python3
"""plug_render_agg.py: one summary of the renderer-slot pluggability runs (v2.1 pluggability, exploratory).

Reads what scripts/plug_render_{bench,realism,run}.py wrote under --root (default
outputs/v2_1/plug/renderer) plus the paper's physics reference rows, and writes one JSON:

  cost        per renderer and device: median render and frame ms (bench), first-call ms, determinism,
              CPU-vs-MPS difference, the in-repo-vs-original LOTUS parity block
  realism     per renderer: W1 (as rendered and robust renorm), KID, FID, C2ST, per-case W1; the realism table parity
  landscape   per renderer and design: appearance-term learnability (per-view Spearman of S_app and of
              S (weights 0.6 on S_seg and 0.4 on S_app) against -pose distance, mean and sd over views; pooled Spearman; S_app by
              distance bin) and a Wilcoxon signed-rank test paired by view id against the physics run of the
              same design (view_match_landscape.paired_test)
  closed_loop per renderer x rung x policy source x weights x contact: final pose error, angle, S, S_seg,
              keep-best pose error, start S, over the crossed held-out episodes, mean and sd over seeds; the
              reference is the physics rows of the paper (outputs/v2/table4/ladder/ibvs_base_physics_ct_s*.json,
              outputs/v2/table4/contact/ctmask_physics_s*.json)

Missing inputs are listed under "missing" (the summary is written with whatever exists). Never overwrites.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATUS = "exploratory, v2.1 pluggability"
METRICS = ("pos", "ang", "match", "seg", "keepbest_pos", "keepbest_ang", "static_match", "start_pos", "peak_force")


def _load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _renderer_of(d, default="physics"):
    p = (d or {}).get("plug") or ((d or {}).get("provenance") or {}).get("plug") or {}
    return p.get("renderer", default) if isinstance(p, dict) else default


def _ms(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    return {"mean": float(a.mean()) if a.size else None, "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "n": int(a.size), "values": [float(x) for x in a]}


def cost(root, missing):
    d = _load(os.path.join(root, "bench", "bench.json"))
    if d is None:
        missing.append("bench/bench.json")
        return None
    out = {"case": d.get("case"), "lotus_parity": d.get("lotus_parity"), "renderers": {}}
    for n, r in d.get("renderers", {}).items():
        out["renderers"][n] = {
            "first_call_ms": r.get("first_call_ms"), "cpu_vs_mps_max_abs": r.get("cpu_vs_mps_max_abs"),
            "load": [r.get("load_before"), r.get("load_after")],
            "devices": {dev: {"render_median_ms": s["render"]["median_ms"], "frame_median_ms": s["frame"]["median_ms"],
                              "frame_iqr_ms": [s["frame"]["p25_ms"], s["frame"]["p75_ms"]],
                              "deterministic": s.get("deterministic_same_seed")}
                        for dev, s in r.get("devices", {}).items()},
            "simus": (r.get("simus") or {}).get("config")}
    return out


def realism(root, missing):
    files = sorted(glob.glob(os.path.join(root, "realism", "*.json")))
    if not files:
        missing.append("realism/*.json")
        return None
    out = {"files": files, "renderers": {}, "parity_with_table2": {}}
    for f in files:
        d = _load(f) or {}
        for n, v in d.get("variants", {}).items():
            if n in out["renderers"]:
                continue
            out["renderers"][n] = {k: v.get(k) for k in ("w1_in_sector", "w1_in_sector_robust_renorm", "kid", "kid_std",
                                                         "fid_secondary", "c2st_auc_mean", "c2st_auc_std",
                                                         "mean_in_sector", "w1_per_case", "n")}
            out["renderers"][n]["source"] = os.path.basename(f)
            out["renderers"][n]["render_s_per_frame"] = (d.get("render_s_per_frame") or {}).get(n)
        if d.get("parity_with_table2"):
            out["parity_with_table2"][os.path.basename(f)] = d["parity_with_table2"]
        if d.get("c2st_permutation_null_physics"):
            out.setdefault("c2st_null_physics", {})[os.path.basename(f)] = d["c2st_permutation_null_physics"]
    return out


def landscape(root, missing, paper_ref):
    sys.path.insert(0, HERE)
    from view_match_landscape import paired_test
    files = sorted(glob.glob(os.path.join(root, "landscape", "*.json")))
    if not files:
        missing.append("landscape/*.json")
        return None
    runs = []
    for f in files:
        d = _load(f)
        if not d or "results" not in d:
            continue
        runs.append((f, _renderer_of(d), (d.get("targets_per_case"), d.get("samples_per_target")), d))
    ref = _load(paper_ref) if paper_ref else None
    out = {"files": [r[0] for r in runs], "runs": {}, "paper_physics_parity": None}
    phys = {}
    for f, rend, design, d in runs:
        arm = d["dr_arms"][0]
        r = d["results"][arm]["physics"]
        key = f"{rend}|tpc{design[0]}|spt{design[1]}"
        out["runs"][key] = {"renderer": rend, "design": {"targets_per_case": design[0], "samples_per_target": design[1],
                                                         "cases": d.get("cases"), "dr": arm},
                            "file": f, "n_views": r["app"]["n_views"],
                            "app_spearman_per_view": {"mean": r["app"]["per_view_mean"], "sd": r["app"]["per_view_std"]},
                            "S_spearman_per_view": {"mean": r["combined_ct_0.6/0.4"]["per_view_mean"],
                                                    "sd": r["combined_ct_0.6/0.4"]["per_view_std"]},
                            "app_spearman_pooled": r.get("app_closeness_spearman_pooled"),
                            "app_by_bin": r.get("app_by_bin"), "bins_mm": d.get("bins_mm")}
        if rend == "physics":
            phys[design] = d
    for f, rend, design, d in runs:
        if rend == "physics" or design not in phys:
            continue
        a, b = phys[design], d
        va = {v["view_id"]: v for v in a["per_view"]}
        vb = {v["view_id"]: v for v in b["per_view"]}
        ids = [i for i in va if i in vb]
        pv = [{"a": va[i], "b": vb[i]} for i in ids]
        arm = a["dr_arms"][0]
        key = f"{rend}|tpc{design[0]}|spt{design[1]}"
        out["runs"][key]["vs_physics_same_design"] = {
            t: paired_test(pv, lambda v, t=t: v["a"]["terms"][arm][t]["physics"],
                           lambda v, t=t: v["b"]["terms"][arm][t]["physics"])
            for t in ("app", "combined_ct_0.6/0.4")}
    if ref is not None:
        for f, rend, design, d in runs:
            if rend != "physics" or design != (ref.get("targets_per_case"), ref.get("samples_per_target")):
                continue
            a = ref["results"]["off"]["physics"]["app"]["per_view"]
            b = d["results"][d["dr_arms"][0]]["physics"]["app"]["per_view"]
            out["paper_physics_parity"] = {"ref": paper_ref, "n": len(a), "max_abs_diff":
                                           float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))
                                           if len(a) == len(b) else None}
    return out


def _closed_rows(d, renderer, source):
    rows = []
    for r in d.get("results", []):
        rows.append({"renderer": renderer, "rung": d.get("rung"), "source": source, "weights": r.get("weights"),
                     "contact": r.get("contact"), "eval_randomize": r.get("eval_randomize"),
                     "seed": d.get("seed", d.get("calib_seed")), "n_episodes": r.get("n_episodes"),
                     **{m: r.get(m) for m in METRICS}})
    return rows


def closed_loop(root, missing):
    rows = []
    files = sorted(glob.glob(os.path.join(root, "ladder", "*.json")))
    if not files:
        missing.append("ladder/*.json")
    for f in files:
        d = _load(f)
        if not d or "results" not in d:
            continue
        rend = _renderer_of(d)
        pj = ((d.get("provenance") or {}).get("policy_json") or "")
        src = "v2 physics-trained" if "/outputs/v2/table4/" in pj else (
            "trained under " + _renderer_of(_load(pj) or {}, "?") if pj else "none (base or controller)")
        rows += [dict(r, file=os.path.basename(f), reference="plug") for r in _closed_rows(d, rend, src)]
    for s in (0, 1, 2):                                   # the paper's physics reference rows
        for p, src in ((os.path.join(REPO, "outputs", "v2", "table4", "ladder", f"ibvs_base_physics_ct_s{s}.json"),
                        "none (base or controller)"),
                       (os.path.join(REPO, "outputs", "v2", "table4", "contact", f"ctmask_physics_s{s}.json"),
                        "v2 physics-trained")):
            d = _load(p)
            if d:
                rows += [dict(r, file=os.path.relpath(p, REPO), reference="paper v2") for r in _closed_rows(d, "physics", src)]
    groups = {}
    for r in rows:
        k = (r["renderer"], r["rung"], r["source"], r["weights"], r["contact"], r.get("eval_randomize"),
             r.get("reference"), r.get("n_episodes"))
        groups.setdefault(k, []).append(r)
    summ = []
    for k, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        summ.append({"renderer": k[0], "rung": k[1], "source": k[2], "weights": k[3], "contact": k[4],
                     "eval_randomize": k[5], "reference": k[6], "n_episodes": k[7], "seeds": sorted({r["seed"] for r in rs if r["seed"] is not None}),
                     "files": sorted({r["file"] for r in rs}),
                     **{m: _ms([r.get(m) for r in rs]) for m in METRICS}})
    return {"rows": rows, "summary": summ}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "outputs", "v2_1", "plug", "renderer"))
    ap.add_argument("--paper-landscape", default=os.path.join(REPO, "outputs", "v2", "view_match",
                                                              "landscape_heldout.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    from usbg import provenance as P
    from usbg.plug import renderers as PR
    missing = []
    out = {"analysis_status": STATUS, "root": os.path.abspath(args.root),
           "methods": {n: PR.CITATIONS[n] for n in ("physics", *PR.PLUG_NAMES)},
           "not_run": PR.INFEASIBLE,
           "cost": cost(args.root, missing), "realism": realism(args.root, missing),
           "landscape": landscape(args.root, missing, args.paper_landscape),
           "closed_loop": closed_loop(args.root, missing), "missing": missing}
    prov = P.collect(root=os.path.abspath(args.root))
    prov["analysis_status"] = STATUS
    out["provenance"] = prov
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for s in out["closed_loop"]["summary"]:
        print(f"{s['reference']:8s} {s['renderer']:13s} {s['rung']:10s} {s['source'][:24]:24s} {str(s['weights']):14s} {s['contact']:3s} "
              f"pos {s['pos']['mean'] if s['pos']['mean'] is not None else float('nan'):6.2f} "
              f"S {s['match']['mean'] if s['match']['mean'] is not None else float('nan'):.3f} "
              f"kb {s['keepbest_pos']['mean'] if s['keepbest_pos']['mean'] is not None else float('nan'):6.2f} "
              f"n_seeds {len(s['seeds'])}")
    print(f"missing: {missing}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
