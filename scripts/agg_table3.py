#!/usr/bin/env python3
"""agg_table3.py: landscape-table statistics from a view_match_landscape.py JSON.

Unit of analysis. Target views are nested in patients, so the per-view Wilcoxon (what the v1 table
shipped) is DESCRIPTIVE only. The inferential tests are at the patient level:

  * per-view paired Wilcoxon (descriptive), pairing BY VIEW ID; a view enters a contrast only when both
    stages are finite on it (the v1 script paired by list index after dropping NaNs, which mis-pairs
    whenever one stage has a NaN view, e.g. --ablate-cue all);
  * patient-level: per patient, the mean Fisher-z (arctanh) of the reference stage minus the mean
    Fisher-z of the other stage over that patient's views; a two-sided Wilcoxon signed-rank over
    patients (with n = 6 the smallest attainable p is 0.03125), sign consistency k/n, and Holm across
    the contrasts;
  * a patient-cluster bootstrap 95% CI of the mean per-view Fisher-z difference (patients resampled
    with replacement, all of a patient's views kept together);
  * a linear mixed model (z ~ is_ref, patient random intercept, view variance component) ONLY if
    statsmodels is importable in this interpreter; it is never installed by this script.

Organs: views of different target organs are reported separately (--organs, default every organ
present, each on its own) and pooled only as an extra block; the landscape table quotes the liver block. A
canonical (v1) JSON is one pooled block, exactly as the shipped table.

Input: the per_view list of the landscape JSON. The anatomical (v2) records carry
terms[arm][term][stage]; --arm/--term pick the quantity (default: the primary arm, appearance). The
canonical (v1) records carry corr[stage] (appearance, DR off). For a pre-v2 JSON without per_view the
view order is rebuilt from cases x targets_per_case, which is valid only when no view was NaN.

Usage:
  .venv/bin/python scripts/agg_table3.py outputs/v2/view_match/landscape_heldout.json \
      --out outputs/v2/view_match/table3_stats.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    run = 0.0
    for r, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out.tolist()


def fisher_z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999))


def load_views(d, arm=None, term="app"):
    """[{view_id, case, organ_group, val: {stage: rho}}] for the chosen arm/term."""
    if "per_view" in d:
        out = []
        for v in d["per_view"]:
            if "terms" in v:
                a = arm or d.get("dr_arms", ["off"])[0]
                val = v["terms"][a][term]
            else:
                if term != "app" or (arm not in (None, "off")):
                    raise SystemExit("canonical (v1) per_view records hold the appearance term, DR off only")
                val = v["corr"]
            # anatomical: one block per target organ; canonical (v1): one pooled block, as shipped
            grp = v.get("target") if "terms" in v else "all"
            out.append({"view_id": v["view_id"], "case": v["case"], "organ_group": grp,
                        "val": {k: (np.nan if x is None else float(x)) for k, x in val.items()}})
        return out
    cases, k = d["cases"], d["targets_per_case"]
    lists = {s: t["app_closeness_per_view"] for s, t in d["translators"].items()}
    n = len(cases) * k
    if any(len(v) != n for v in lists.values()):
        raise SystemExit("pre-v2 JSON has NaN-dropped views; per-view pairing cannot be reconstructed")
    return [{"view_id": f"{cases[i // k]}/t{i % k}", "case": cases[i // k], "organ_group": "all",
             "val": {s: lists[s][i] for s in lists}} for i in range(n)]


def contrast(views, ref, other, boot, rng):
    from scipy.stats import wilcoxon
    pairs = [(v["case"], v["val"].get(ref), v["val"].get(other)) for v in views]
    pairs = [p for p in pairs if p[1] is not None and p[2] is not None and np.isfinite(p[1]) and np.isfinite(p[2])]
    c = {"n_views": len(pairs)}
    if not pairs:
        c["note"] = "no view with both stages finite"
        return c
    cs = np.array([p[0] for p in pairs]); a = np.array([p[1] for p in pairs]); b = np.array([p[2] for p in pairs])
    c.update(mean_ref=float(a.mean()), std_ref_pop=float(a.std()), mean_other=float(b.mean()),
             std_other_pop=float(b.std()), winrate_views=float(np.mean(a > b)))
    if np.allclose(a, b):
        c["view_wilcoxon"] = {"note": "identical on every view"}
    else:
        st, p = wilcoxon(a, b)
        c["view_wilcoxon"] = {"statistic": float(st), "p": float(p), "n": len(pairs),
                              "note": "descriptive; views are nested in patients"}
    za, zb = fisher_z(a), fisher_z(b)
    pats = sorted(set(cs.tolist()))
    dz = np.array([za[cs == q].mean() - zb[cs == q].mean() for q in pats])
    pl = {"n_patients": len(pats), "per_patient_dz": {q: float(x) for q, x in zip(pats, dz)},
          "mean_dz": float(dz.mean()), "sign_consistency": f"{int((dz > 0).sum())}/{len(pats)}",
          "n_views_per_patient": {q: int((cs == q).sum()) for q in pats}}
    if len(pats) >= 2 and not np.allclose(dz, 0):
        st2, p2 = wilcoxon(dz)
        pl.update(wilcoxon_statistic=float(st2), wilcoxon_p=float(p2),
                  min_attainable_p=float(2.0 / 2 ** len(pats)))
    c["patient_level"] = pl
    idx_of = {q: np.where(cs == q)[0] for q in pats}
    diff = za - zb
    bs = np.empty(boot)
    for i in range(boot):
        pick = rng.choice(len(pats), size=len(pats), replace=True)
        bs[i] = diff[np.concatenate([idx_of[pats[j]] for j in pick])].mean()
    c["cluster_boot_dz"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                  float(np.percentile(bs, 97.5))],
                            "n_boot": int(boot), "unit": "patient"}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        c["lmm"] = {"skipped": "statsmodels is not importable in this interpreter (not installed by design)"}
        return c
    rows = [{"case": q, "view": f"{q}#{i}", "is_ref": s, "z": float(z)}
            for i, (q, z1, z2) in enumerate(zip(cs, za, zb)) for s, z in ((1.0, z1), (0.0, z2))]
    df = pd.DataFrame(rows)
    try:
        m = smf.mixedlm("z ~ is_ref", df, groups=df["case"], re_formula="1",
                        vc_formula={"view": "0 + C(view)"}).fit(reml=True)
        c["lmm"] = {"beta_ref_minus_other_z": float(m.params["is_ref"]), "p": float(m.pvalues["is_ref"]),
                    "converged": bool(getattr(m, "converged", True))}
    except Exception as e:                                              # pragma: no cover
        c["lmm"] = {"error": f"{type(e).__name__}: {e}"}
    return c


def block(views, ref, others, boot, seed):
    rng = np.random.default_rng(seed)
    res = {"n_views": len(views), "n_patients": len({v["case"] for v in views}), "contrasts": {}}
    for s in others:
        res["contrasts"][s] = contrast(views, ref, s, boot, rng)
    for key, get in (("patient_wilcoxon_p_holm", lambda c: c.get("patient_level", {}).get("wilcoxon_p")),
                     ("lmm_p_holm", lambda c: (c.get("lmm") or {}).get("p"))):
        ps = [get(res["contrasts"][s]) for s in others]
        if any(p is not None for p in ps):
            for s, pa in zip(others, holm([np.nan if p is None else p for p in ps])):
                res["contrasts"][s].setdefault("holm", {})[key] = pa
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default=None, help="DR arm (anatomical JSON); default the primary arm")
    ap.add_argument("--term", default="app", help="app | seg_ct | seg_learned | combined_* (anatomical JSON)")
    ap.add_argument("--ref", default="sonospade")
    ap.add_argument("--organs", nargs="*", default=None, help="organ groups to report (default: all present)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.json) as fh:
        d = json.load(fh)
    views = load_views(d, args.arm, args.term)
    stages = list(views[0]["val"]) if views else []
    if args.ref not in stages:
        raise SystemExit(f"reference stage {args.ref!r} not in {stages}")
    others = [s for s in stages if s != args.ref]
    groups = args.organs or sorted({v["organ_group"] for v in views})
    res = {"source": args.json, "seat_mode": d.get("seat_mode", (d.get("provenance") or {}).get("seat_mode")),
           "arm": args.arm or (d.get("dr_arms") or ["off"])[0], "term": args.term, "ref": args.ref,
           "n_views": len(views), "by_organ": {}}
    for g in groups:
        sub = [v for v in views if v["organ_group"] == g]
        if sub:
            res["by_organ"][g] = block(sub, args.ref, others, args.boot, args.seed)
    if len(groups) > 1:
        res["pooled_all_organs"] = block(views, args.ref, others, args.boot, args.seed)
        res["pooled_note"] = "descriptive: organs are reported separately; do not quote the pooled block as the landscape table"
    for g, b in list(res["by_organ"].items()) + ([("pooled", res["pooled_all_organs"])] if len(groups) > 1 else []):
        print(f"[{g}] {b['n_views']} views, {b['n_patients']} patients ({res['arm']}, {args.term})")
        for s, c in b["contrasts"].items():
            if "patient_level" not in c:
                print(f"   {args.ref} vs {s}: {c.get('note')}")
                continue
            pl = c["patient_level"]
            print(f"   {args.ref} vs {s:9s} views: {c['mean_ref']:+.3f} vs {c['mean_other']:+.3f} "
                  f"p_view={c['view_wilcoxon'].get('p', float('nan')):.4g} | patients: dz={pl['mean_dz']:+.3f} "
                  f"sign {pl['sign_consistency']} p={pl.get('wilcoxon_p', float('nan')):.4g} "
                  f"(Holm {c.get('holm', {}).get('patient_wilcoxon_p_holm', float('nan')):.4g}) "
                  f"CI95 dz [{c['cluster_boot_dz']['ci95'][0]:+.3f}, {c['cluster_boot_dz']['ci95'][1]:+.3f}] "
                  f"LMM {c['lmm'].get('p', c['lmm'].get('skipped', c['lmm'].get('error')))}")
    try:
        from usbg import provenance as P
        res["provenance"] = {"input_md5": P.md5(args.json), "git": P.git_info(),
                             "input_provenance_seat": (d.get("provenance") or {}).get("seat_mode")}
    except Exception as e:                                              # pragma: no cover
        res["provenance"] = {"error": repr(e)}
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
