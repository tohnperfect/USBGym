#!/usr/bin/env python3
"""POST HOC harvest and statistics layer on the existing USB Gym v2 outputs (the post hoc analyses of the v2.1 plan).

Every analysis in this file is POST HOC, outside the v2 plan (whose commit and sha256 the output's provenance
records). No episode is run, no policy, texture stage or segmenter is touched, and no v2 result file is modified. The
script reads the episode and registration tables of outputs/v2/table4, the result JSONs of that folder (for the
trajectories per step), the 27 trainer logs (REINFORCE reward) and the eval_<stage>_s<seed>.json summaries of the
segmenter repository (Dice on held-out simulated frames and on real frames).

Outputs (all new files):
  outputs/v2/posthoc/posthoc_v2.json        every number, with a provenance block (sha256 of every input)
  outputs/v2/posthoc/step_curves.csv        patient-level mean error and match S against step, oracle-ladder rows
  outputs/v2/posthoc/reinforce_reward.csv   REINFORCE training reward against iteration, 27 trainer runs

Parts:
  (a) rotation error for every rung, arm, stage and phase: final_ang is already the geodesic angle on SO(3) of the
      relative rotation (par.se3), and errors per axis are not logged.
  (b) success rates, final and keep-best, at the thresholds 5 mm / 5 deg and 10 mm / 10 deg, per patient.
  (c) results per organ (liver, kidney).
  (d) after imitation minus the servo base, for each arm and stage (translation and rotation).
  (e) error and match S against step from the logged pos_traj and match_traj (rotation and organ IoU per
      step are computed during evaluation but not saved).
  (f) statistics sensitivity for every planned Q2 contrast: exact two-sided sign-flip p (n = 6), t-based
      95% CI and p, Holm within each family, a patient-by-seed crossed bootstrap, and a simulation of the
      CI-and-sign rule's type-I error and power at n = 6 (the rule exactly as collect_results_v2.rule runs it).
  (g) equivalence tests (TOST) of the RL change on the 8 policy arms, margin 1 mm.
  (h) crossed bootstraps (targets x CTs x seeds, and targets x seeds) for the registration on real frames.
  No mixed model: statsmodels is not installed in .venv and nothing is installed by this script.

Usage: PYTHONPATH=src .venv/bin/python scripts/posthoc_v2.py [--nsim 10000] [--out outputs/v2/posthoc]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys

import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
import script_dirs as SD  # noqa: E402

ROOT = SD.results_root()              # the result files: the code tree, or the result root in public mode
V2 = os.path.join(ROOT, "outputs", "v2")
T4 = os.path.join(V2, "table4")
SEG = SD.segmenter_root(required=False)      # the segmenter repository (ROOT/segmenter in public mode)
OUT_DIR = os.path.join(V2, "posthoc")
OUT_JSON = os.path.join(OUT_DIR, "posthoc_v2.json")
EPS_CSV = os.path.join(T4, "episodes_long.csv")
REG_CSV = os.path.join(T4, "registration_long.csv")
STAGES = ("physics", "cut", "cyclegan", "sonospade")
ARMS = (("ctmask", "ct"), ("matched", "learned"))
NBOOT = 10000            # resamples, as in the v2 plan (seed 0)
BOOT_SEED = 0
SIM_SEED = 20261014      # simulation seed, fixed before the simulation was run
TOST_MARGIN_MM = 1.0     # the RL threshold of the v2 plan, used post hoc as an equivalence margin
SUCCESS = {"5mm_5deg": (5.0, 5.0), "10mm_10deg": (10.0, 10.0)}
EMDASH, TRIPLE = chr(0x2014), "-" * 3   # banned in every written file
def _private():
    import plan_inputs as PI
    return PI.private()


def _rule_text(key):
    """A text of the post hoc record and of RESULTS.md (plan_inputs.text: the development build's own wording)."""
    import plan_inputs as PI
    return PI.text("ph " + key)


def rel(p):
    if SD.public():
        if SEG and (p == SEG or p.startswith(SEG + os.sep)):
            return "segmenter/" + os.path.relpath(p, SEG)
        return SD.display(p)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT) else _private().tilde(p)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== inputs
def trajectory_files():
    """The per-episode result JSONs, selected exactly as scripts/agg_table4_v2.py selects them."""
    import agg_table4_v2 as A
    return [f for f in A.result_files(T4) if not rel(f).startswith("outputs/v2/table4/registration/")]


def trainer_logs():
    fs = []
    for grp in ("ctmask", "matched", "dr_on"):
        fs += sorted(glob.glob(os.path.join(T4, grp, "*_deploy_s[0-9].log")))
    return fs


def seg_eval_files():
    return sorted(glob.glob(os.path.join(SEG, "outputs", "v2", "eval_*_s[0-9].json")))


def digest(paths):
    """sha256 over the sorted (relative path, file sha256) list: one fingerprint for a file set."""
    items = sorted((rel(p), sha256(p)) for p in paths)
    h = hashlib.sha256("\n".join(f"{a} {b}" for a, b in items).encode()).hexdigest()
    return {"n_files": len(items), "sha256": h}


def input_fingerprint():
    """Fingerprint of every input; the collector compares it with posthoc_v2.json to detect a stale file."""
    return {"episodes_long.csv": {"path": rel(EPS_CSV), "sha256": sha256(EPS_CSV)},
            "registration_long.csv": {"path": rel(REG_CSV), "sha256": sha256(REG_CSV)},
            "trajectory_jsons": dict(digest(trajectory_files()), glob="outputs/v2/table4/<group>/*.json"),
            "trainer_logs": dict(digest(trainer_logs()), glob="outputs/v2/table4/{ctmask,matched,dr_on}/*_deploy_s?.log"),
            "seg_eval_jsons": dict(digest(seg_eval_files()), glob="segmenter/outputs/v2/eval_*_s?.json" if SD.public() else
                                   _private().SEG_EVAL_GLOB)}


def load_eps():
    import pandas as pd
    d = pd.read_csv(EPS_CSV, float_precision="round_trip", dtype={"seed": str})
    d["patient"] = d["patient"].astype(str)
    return d


def load_reg():
    import pandas as pd
    return pd.read_csv(REG_CSV, float_precision="round_trip", dtype={"seed": str, "frame": str})


# ===================================================================================== statistics
def boot_ci(x, nboot=NBOOT, seed=BOOT_SEED, q=(2.5, 97.5)):
    """Percentile bootstrap of the mean, the collector's exact procedure (seed 0, 10,000 resamples)."""
    d = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(d):
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))]


def signflip_p(d, alternative="two-sided", mc=200000, seed=BOOT_SEED):
    """Paired sign-flip (randomization) test of the mean. Exact over all 2^n sign vectors for n <= 16,
    Monte Carlo (mc flips, seed 0, observed pattern counted) above that."""
    d = np.asarray(d, float)
    n = len(d)
    obs = d.mean()
    if n <= 16:
        S = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        S = np.vstack([np.ones(n), rng.choice((1.0, -1.0), size=(mc, n))])
        exact = False
    m = (S * d).mean(1)
    tol = 1e-12 * max(1.0, abs(obs))
    if alternative == "two-sided":
        p = float(np.mean(np.abs(m) >= abs(obs) - tol))
    elif alternative == "greater":
        p = float(np.mean(m >= obs - tol))
    else:
        p = float(np.mean(m <= obs + tol))
    return p, exact


def t_stats(d, level=0.95):
    from scipy import stats as st
    d = np.asarray(d, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tq = float(st.t.ppf(0.5 + level / 2, n - 1))
    t = m / se if se > 0 else float("inf") * np.sign(m)
    p = float(2 * st.t.sf(abs(t), n - 1)) if se > 0 else 0.0
    return {"sd": sd, "se": se, "t": float(t), "df": n - 1, "p_t": p, "ci_t": [m - tq * se, m + tq * se],
            "dz": m / sd if sd > 0 else float("nan")}


def sign_test_p(d):
    from scipy import stats as st
    d = np.asarray(d, float)
    k = int((d > 0).sum())
    n = int((d != 0).sum())
    return float(st.binomtest(k, n, 0.5).pvalue) if n else float("nan")


def crossed_boot(D, nboot=NBOOT, seed=BOOT_SEED, q=(2.5, 97.5)):
    """Crossed (multiway, pigeonhole) bootstrap of the grand mean of an array D whose axes are independent
    random factors (patients x seeds, or targets x CTs x seeds). Every axis is resampled with replacement on
    its own; the statistic is the weighted mean over the observed cells, sum(w_i u_j v_k Y_ijk) / sum(w_i u_j
    v_k), with the resampling counts as weights, so cells that were not run (NaN) carry no weight. With a
    complete grid this is the plain mean over the resampled grid. Resamples whose weights select no observed
    cell are dropped and counted."""
    D = np.asarray(D, float)
    M = np.isfinite(D).astype(float)
    Y = np.where(M > 0, D, 0.0)
    rng = np.random.default_rng(seed)
    W = []
    for n in D.shape:
        idx = rng.integers(0, n, size=(nboot, n))
        flat = (idx + n * np.arange(nboot)[:, None]).ravel()
        W.append(np.bincount(flat, minlength=nboot * n).reshape(nboot, n).astype(float))
    if D.ndim == 2:
        num = np.einsum("bi,ij,bj->b", W[0], Y, W[1])
        den = np.einsum("bi,ij,bj->b", W[0], M, W[1])
    elif D.ndim == 3:
        out = []
        for A in (Y, M):
            T = np.einsum("bf,fk->bk", W[0], A.reshape(A.shape[0], -1)).reshape(nboot, A.shape[1], A.shape[2])
            out.append(np.einsum("bcs,bc,bs->b", T, W[1], W[2]))
        num, den = out
    else:
        raise ValueError("crossed_boot supports 2 or 3 axes")
    ok = den > 0
    stat = num[ok] / den[ok]
    return [float(np.percentile(stat, q[0])), float(np.percentile(stat, q[1]))], int((~ok).sum())


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return [float(x) for x in adj]


def contrast(a_pp, b_pp, a_ps=None, b_ps=None):
    """Paired patient-level contrast a - b. a_pp / b_pp: {patient: mean}. a_ps / b_ps: optional
    {patient: {seed: mean}} for the patient-by-seed crossed bootstrap (a side without seeds is broadcast)."""
    pats = sorted(set(a_pp) & set(b_pp))
    d = np.array([a_pp[p] - b_pp[p] for p in pats])
    m, ci = boot_ci(d)
    sgn = np.sign(m)
    agree = int((np.sign(d) == sgn).sum())
    holds = bool((ci[0] > 0 or ci[1] < 0) and agree >= 5)
    p_sf, _ = signflip_p(d)
    res = {"n": len(d), "mean": m, "ci_boot": ci, "sign": agree, "holds_rule": holds,
           "p_signflip_exact": p_sf, "p_sign_test": sign_test_p(d), **t_stats(d),
           "per_patient": {p: float(x) for p, x in zip(pats, d)},
           "dissenting_patients": [p for p, x in zip(pats, d) if np.sign(x) != sgn]}
    if a_ps is not None or b_ps is not None:
        seeds = sorted(set().union(*[set(v) for v in (a_ps or b_ps).values()]))
        if a_ps is not None and b_ps is not None:
            sb = sorted(set().union(*[set(v) for v in b_ps.values()]))
            if sb != seeds:
                seeds = None
        if seeds:
            def val(ps, pp, p, s):
                return ps[p][s] if ps is not None else pp[p]
            D = np.array([[val(a_ps, a_pp, p, s) - val(b_ps, b_pp, p, s) for s in seeds] for p in pats])
            res["ci_crossed_patient_seed"] = crossed_boot(D)[0]
            res["seeds"] = seeds
            res["per_seed_mean"] = {s: float(D[:, j].mean()) for j, s in enumerate(seeds)}
    return res


def tost(d, margin, D_ps=None):
    """Two one-sided tests of |mean| < margin on paired differences d (t-based, primary) with the exact
    sign-flip version and the 90% crossed bootstrap CI as sensitivity checks."""
    from scipy import stats as st
    d = np.asarray(d, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    p_lo = float(st.t.sf((m + margin) / se, n - 1))      # H0: mean <= -margin
    p_hi = float(st.t.cdf((m - margin) / se, n - 1))     # H0: mean >= +margin
    tq = float(st.t.ppf(0.95, n - 1))
    sf_lo, _ = signflip_p(d + margin, "greater")
    sf_hi, _ = signflip_p(d - margin, "less")
    res = {"n": n, "mean": m, "margin": margin, "ci90_t": [m - tq * se, m + tq * se],
           "p_lower": p_lo, "p_upper": p_hi, "p_tost_t": max(p_lo, p_hi),
           "equivalent_t": bool(max(p_lo, p_hi) < 0.05),
           "p_tost_signflip": max(sf_lo, sf_hi), "equivalent_signflip": bool(max(sf_lo, sf_hi) < 0.05)}
    if D_ps is not None:
        ci = crossed_boot(D_ps, q=(5.0, 95.0))[0]
        res["ci90_crossed_patient_seed"] = ci
        res["equivalent_crossed_ci90"] = bool(-margin < ci[0] and ci[1] < margin)
    return res


# ===================================================================================== oracle-ladder rows
def table4_rows():
    """(row id, label, filter) for every oracle-ladder row, with the collector's filters (contact on, DR off)."""
    rows = [("static", "Static start", dict(group="ladder", rung="static")),
            ("scripted", "Scripted sweep (open loop)", dict(group="ladder", rung="scripted")),
            ("pose_oracle", "Analytic pose oracle (privileged)", dict(group="ladder", rung="pose_oracle"))]
    for arm, mask in ARMS:
        for st in STAGES:
            rows.append((f"{arm}/{st}/base", f"{arm} {st} servo base",
                         dict(group="ladder", rung="ibvs_base", stage=st, mask=mask)))
            for ph in ("after_imitation", "final"):
                rows.append((f"{arm}/{st}/{ph}", f"{arm} {st} {ph}", dict(group=arm, arm=arm, stage=st, phase=ph)))
    return rows


def select(d, flt, contact="on", dr="off"):
    m = (d["contact"] == contact) & (d["eval_randomize"] == dr)
    for k, v in flt.items():
        m &= d[k].astype(str) == str(v)
    return d[m]


def add_success(d):
    d = d.copy()
    for k, (mm, deg) in SUCCESS.items():
        d[f"succ_{k}"] = ((d["final_pos"] <= mm) & (d["final_ang"] <= deg)).astype(float)
        d[f"kb_succ_{k}"] = ((d["keepbest_pos"] <= mm) & (d["keepbest_ang"] <= deg)).astype(float)
    d["succ_5mm_only"] = (d["final_pos"] <= 5.0).astype(float)
    d["succ_5deg_only"] = (d["final_ang"] <= 5.0).astype(float)
    return d


METRICS = {"final_pos": "final translation error (mm)", "final_ang": "final rotation error, geodesic (deg)",
           "keepbest_pos": "keep-best translation error (mm)", "keepbest_ang": "keep-best rotation error (deg)",
           "start_pos": "start translation error (mm)", "start_ang": "start rotation error (deg)",
           "final_match": "final match score S", "final_seg": "final organ-layout IoU (mask used by the arm)",
           "succ_5mm_5deg": "success, final <= 5 mm and <= 5 deg", "succ_10mm_10deg": "success, final <= 10 mm and <= 10 deg",
           "kb_succ_5mm_5deg": "success at keep-best, <= 5 mm and <= 5 deg",
           "kb_succ_10mm_10deg": "success at keep-best, <= 10 mm and <= 10 deg",
           "succ_5mm_only": "final <= 5 mm (translation only)", "succ_5deg_only": "final <= 5 deg (rotation only)",
           "contact_frac": "fraction of steps in contact", "peak_force": "peak contact force (N)"}


def pp_mean(sel, metric):
    return {p: float(v) for p, v in sel.groupby("patient")[metric].mean().items()}


def ps_mean(sel, metric):
    out = collections.defaultdict(dict)
    for (p, s), v in sel.groupby(["patient", "seed"])[metric].mean().items():
        out[str(p)][str(s)] = float(v)
    return dict(out)


def summarize_rows(d):
    rows = {}
    for rid, lab, flt in table4_rows():
        sel = select(d, flt)
        if sel.empty:
            raise SystemExit(f"row {rid}: no episodes")
        r = {"label": lab, "filter": dict(flt, contact="on", eval_randomize="off"), "n_episodes": int(len(sel)),
             "seeds": sorted(sel["seed"].astype(str).unique()), "metrics": {}, "per_organ": {}, "per_seed": {}}
        for mt in METRICS:
            pp = pp_mean(sel, mt)
            v = np.array(list(pp.values()))
            r["metrics"][mt] = {"mean": float(v.mean()), "sd_over_patients": float(v.std(ddof=1)),
                                "min_patient": float(v.min()), "max_patient": float(v.max()), "per_patient": pp}
        for mt in ("succ_5mm_5deg", "succ_10mm_10deg"):
            r["metrics"][mt]["ci_boot_patients"] = boot_ci(list(r["metrics"][mt]["per_patient"].values()))[1]
        for organ, so in sel.groupby("target"):
            r["per_organ"][organ] = {mt: {"mean": float(np.mean(list(pp_mean(so, mt).values()))),
                                          "per_patient": pp_mean(so, mt)}
                                     for mt in ("final_pos", "final_ang", "final_match", "succ_5mm_5deg",
                                                "succ_10mm_10deg", "start_pos", "start_ang")}
        for s, ss in sel.groupby("seed"):
            r["per_seed"][str(s)] = {mt: float(np.mean(list(pp_mean(ss, mt).values())))
                                     for mt in ("final_pos", "final_ang", "final_match", "succ_5mm_5deg",
                                                "succ_10mm_10deg")}
        rows[rid] = r
    return rows


def check_against_collector(rows, d):
    """Every final_pos row mean and every Q2 contrast must reproduce the collector (RESULTS.json table4)."""
    import collect_results_v2 as C
    fresh = {}
    C.table4(fresh)
    t4 = fresh["table4"]
    bad = []
    for k, v in t4["rungs_mm"].items():
        if abs(rows[k]["metrics"]["final_pos"]["mean"] - v) > 1e-9 * max(1, abs(v)):
            bad.append(k)
    for k, v in t4["cell_means_mm"].items():
        if abs(rows[k]["metrics"]["final_pos"]["mean"] - v) > 1e-9 * max(1, abs(v)):
            bad.append(k)
    if bad:
        raise SystemExit(f"post hoc row means do not reproduce the collector: {bad}")
    return t4


# ===================================================================================== contrasts
def q2_families(rows, metric="final_pos"):
    """The 25 Q2 contrasts of the v2 plan, in the collector's order and sign convention."""
    import plan_inputs as PI
    fam = collections.OrderedDict()
    fam["Q2a_observability_gap"] = {"label": PI.text("q2a label"),
                                    "items": {st: (f"ctmask/{st}/final", "pose_oracle") for st in STAGES}}
    fam["Q2b_perception_gap"] = {"label": PI.text("q2b label"),
                                 "items": {st: (f"matched/{st}/final", f"ctmask/{st}/final") for st in STAGES}}
    fam["Q2c_rl_contribution"] = {"label": PI.text("q2c label"),
                                  "items": {f"{a}/{st}": (f"{a}/{st}/final", f"{a}/{st}/after_imitation")
                                            for a, _ in ARMS for st in STAGES}}
    pairs = {}
    for i, a in enumerate(STAGES):
        for b in STAGES[i + 1:]:
            pairs[f"{a}-{b}"] = (f"ctmask/{a}/final", f"ctmask/{b}/final")
    fam["Q2d_ctmask_pairs"] = {"label": PI.text("q2d label"), "items": pairs}
    fam["Q2e_learned_sonospade_vs"] = {"label": PI.text("q2e label"),
                                       "items": {o: ("matched/sonospade/final", f"matched/{o}/final")
                                                 for o in ("physics", "cut", "cyclegan")}}
    return fam


def posthoc_families():
    fam = collections.OrderedDict()
    fam["PH_imitation_minus_base"] = {"label": "Imitation minus servo base (after imitation minus base)",
                                      "items": {f"{a}/{st}": (f"{a}/{st}/after_imitation", f"{a}/{st}/base")
                                                for a, _ in ARMS for st in STAGES}}
    fam["PH_final_minus_base"] = {"label": "Final minus servo base",
                                  "items": {f"{a}/{st}": (f"{a}/{st}/final", f"{a}/{st}/base")
                                            for a, _ in ARMS for st in STAGES}}
    fam["PH_base_minus_static"] = {"label": "Servo base minus static start",
                                   "items": {f"{a}/{st}": (f"{a}/{st}/base", "static")
                                             for a, _ in ARMS for st in STAGES}}
    fam["PH_final_minus_static"] = {"label": "Final minus static start",
                                    "items": {f"{a}/{st}": (f"{a}/{st}/final", "static")
                                              for a, _ in ARMS for st in STAGES}}
    return fam


def run_family(rows, fam, metric, planned):
    out = collections.OrderedDict()
    for fid, f in fam.items():
        items = collections.OrderedDict()
        for name, (a, b) in f["items"].items():
            ra, rb = rows[a], rows[b]
            items[name] = dict(contrast(ra["metrics"][metric]["per_patient"], rb["metrics"][metric]["per_patient"],
                                        ra.get("_ps", {}).get(metric), rb.get("_ps", {}).get(metric)),
                               a=a, b=b)
        names = list(items)
        for key, src in (("holm_signflip", "p_signflip_exact"), ("holm_t", "p_t")):
            adj = holm([items[n][src] for n in names])
            for n, x in zip(names, adj):
                items[n][key] = x
        m = len(names)
        out[fid] = {"label": f["label"], "metric": metric, _rule_text("planned key"): planned, "post_hoc_stats": True,
                    "m": m, "holm_first_threshold": 0.05 / m, "items": items,
                    "n_rule_holds": sum(items[n]["holds_rule"] for n in names),
                    "n_holm_signflip_survive": sum(items[n]["holm_signflip"] <= 0.05 for n in names),
                    "n_holm_t_survive": sum(items[n]["holm_t"] <= 0.05 for n in names)}
    return out


def holm_all(families):
    """Holm over the union of the given families (a stricter sensitivity: all Q2 contrasts as one family)."""
    keys = [(f, n) for f, F in families.items() for n in F["items"]]
    res = {}
    for key, src in (("holm_signflip_all", "p_signflip_exact"), ("holm_t_all", "p_t")):
        adj = holm([families[f]["items"][n][src] for f, n in keys])
        for (f, n), x in zip(keys, adj):
            res.setdefault(f"{f}/{n}", {})[key] = x
    return {"m": len(keys), "items": res,
            "n_survive_signflip": sum(v["holm_signflip_all"] <= 0.05 for v in res.values()),
            "n_survive_t": sum(v["holm_t_all"] <= 0.05 for v in res.values())}


def per_organ_contrasts(d):
    """Per-organ versions of the perception gap and SonoSPADE-vs-stage contrasts (translation and rotation)."""
    out = {}
    for organ in ("liver", "kidney"):
        do = d[d["target"] == organ]
        cache = {}
        for rid, _, flt in table4_rows():
            sel = select(do, flt)
            cache[rid] = {mt: pp_mean(sel, mt) for mt in ("final_pos", "final_ang")}
        o = {}
        for mt in ("final_pos", "final_ang"):
            o[mt] = {
                "perception_gap": {st: contrast(cache[f"matched/{st}/final"][mt], cache[f"ctmask/{st}/final"][mt])
                                   for st in STAGES},
                "learned_sonospade_vs": {s: contrast(cache["matched/sonospade/final"][mt], cache[f"matched/{s}/final"][mt])
                                         for s in ("physics", "cut", "cyclegan")},
                "observability_gap": {st: contrast(cache[f"ctmask/{st}/final"][mt], cache["pose_oracle"][mt])
                                      for st in STAGES},
                "rl_contribution": {f"{a}/{st}": contrast(cache[f"{a}/{st}/final"][mt],
                                                          cache[f"{a}/{st}/after_imitation"][mt])
                                    for a, _ in ARMS for st in STAGES}}
            for blk in o[mt].values():
                for v in blk.values():
                    v.pop("per_patient", None)
        out[organ] = o
    return out


# ===================================================================================== rule simulation
def rule_vectorized(Dsim, idx):
    """collect_results_v2.rule applied to every row of Dsim (nsim x n) with the collector's fixed
    resampling matrix idx (seed 0, 10,000 x n). Returns (holds, ci_excludes_0, sign_ok, mean, lo, hi)."""
    m = Dsim.mean(1)
    lo = np.empty(len(Dsim))
    hi = np.empty(len(Dsim))
    for i in range(0, len(Dsim), 100):
        bs = Dsim[i:i + 100][:, idx].mean(2)                     # (b, nboot)
        q = np.percentile(bs, [2.5, 97.5], axis=1)
        lo[i:i + 100], hi[i:i + 100] = q[0], q[1]
    agree = (np.sign(Dsim) == np.sign(m)[:, None]).sum(1)
    ci_ex = (lo > 0) | (hi < 0)
    sign_ok = agree >= 5
    return ci_ex & sign_ok, ci_ex, sign_ok, m, lo, hi


def simulate_rule(nsim, n=6, seed=SIM_SEED):
    """Simulated type-I error and power at n = 6: the CI-and-sign rule (as collect_results_v2.rule applies it), the
    exact sign-flip test and the paired t-test, two-sided at alpha 0.05. Null scenarios: normal, Student t (3 degrees
    of freedom), centered exponential. Power: normal differences shifted by delta standard deviations."""
    from scipy import stats as st
    import collect_results_v2 as C
    rng0 = np.random.default_rng(BOOT_SEED)
    idx = rng0.integers(0, n, size=(NBOOT, n))                   # the collector's resampling matrix
    rng = np.random.default_rng(seed)
    S = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    tcrit = st.t.ppf(0.975, n - 1)

    def run(D):
        holds, ci_ex, sign_ok, m, lo, hi = rule_vectorized(D, idx)
        fm = np.einsum("ij,kj->ik", D, S) / n                    # (nsim, 64) sign-flipped means
        p_sf = (np.abs(fm) >= np.abs(m)[:, None] - 1e-12).mean(1)
        sd = D.std(1, ddof=1)
        t = m / (sd / np.sqrt(n))
        return {"rule": float(holds.mean()), "ci_excludes_0": float(ci_ex.mean()),
                "sign_at_least_5_of_6": float(sign_ok.mean()), "signflip_exact_p_le_0.05": float((p_sf <= 0.05).mean()),
                "t_test_p_le_0.05": float((np.abs(t) > tcrit).mean())}, (holds, m, lo, hi)

    scen = collections.OrderedDict()
    nulls = {"normal": lambda: rng.standard_normal((nsim, n)),
             "t3": lambda: rng.standard_t(3, (nsim, n)),
             "centered_exponential": lambda: rng.exponential(1.0, (nsim, n)) - 1.0}
    check = None
    for name, gen in nulls.items():
        D = gen()
        r, extra = run(D)
        scen[f"null_{name}"] = dict(r, delta_sd=0.0, distribution=name)
        if check is None:                                        # verify against the collector's own rule()
            k = min(300, nsim)
            mism = 0
            for i in range(k):
                h, mm, ci, _, _ = C.rule(list(D[i]))
                if (h != bool(extra[0][i]) or abs(mm - extra[1][i]) > 1e-12 or abs(ci[0] - extra[2][i]) > 1e-12
                        or abs(ci[1] - extra[3][i]) > 1e-12):
                    mism += 1
            check = {"n_checked": k, "n_mismatch": mism}
            if mism:
                raise SystemExit(f"vectorized rule disagrees with collect_results_v2.rule on {mism}/{k} datasets")
    for delta in (0.5, 1.0, 1.5, 2.0, 3.0):
        D = rng.standard_normal((nsim, n)) + delta
        r, _ = run(D)
        scen[f"normal_delta_{delta:g}sd"] = dict(r, delta_sd=delta, distribution="normal")
    return {"nsim": nsim, "n": n, "seed": seed, "nboot": NBOOT, "boot_seed": BOOT_SEED,
            "rule_definition": _rule_text("rule definition"),
            "verified_against_collector_rule": check, "scenarios": scen,
            "smallest_two_sided_exact_signflip_p": 2.0 / 2 ** n,
            "exact_p_of_a_5_of_6_pattern_at_least": 4.0 / 2 ** n,
            "holm_first_threshold_by_family_size": {m: 0.05 / m for m in (2, 3, 4, 6, 8, 25)},
            "note": ("At n = 6 the smallest two-sided exact sign-flip p is 2/64 = 1/32 = 0.03125. Holm's first "
                     "threshold is 0.05/m, which is below 0.03125 for every family of m >= 2 contrasts, so no exact "
                     "sign-flip test on six patients can survive Holm in any Q2 family (m = 3 to 8). A 5/6 sign "
                     "pattern has an exact p of at least 4/64 = 0.0625.")}


# ===================================================================================== registration (h)
def registration_crossed(reg):
    import pandas as pd
    import collect_results_v2 as C
    fresh = {}
    C.registration(fresh)
    ref = fresh["registration"]
    seg = reg[reg["mask"] == "seg"].copy()
    out = collections.OrderedDict()
    for (st, src, organ), g in sorted(seg.groupby(["stage", "src", "organ"])):
        frames = list(g["frame"].unique())        # first-appearance order, as the collector's per-target dict
        cts = sorted(g["ct"].unique())
        seeds = sorted(g["seed"].unique())
        cell = {}
        for metric, col in (("keepbest_gain", None), ("final_gain", None), ("static_iou", "static_iou"),
                            ("final_iou", "final_iou"), ("keepbest_iou", "keepbest_iou")):
            if metric == "keepbest_gain":
                v = g["keepbest_iou"] - g["static_iou"]
            elif metric == "final_gain":
                v = g["final_iou"] - g["static_iou"]
            else:
                v = g[col]
            gg = g.assign(_v=v.values)
            Y = gg.pivot_table(index="frame", columns=["ct", "seed"], values="_v", aggfunc="mean")
            Y = Y.reindex(index=frames, columns=pd.MultiIndex.from_product([cts, seeds], names=["ct", "seed"]))
            arr = Y.values.reshape(len(frames), len(cts), len(seeds))
            ts = gg.pivot_table(index="frame", columns="seed", values="_v", aggfunc="mean").reindex(
                index=frames, columns=seeds).values                  # target x seed, CTs averaged within
            if np.isnan(ts).any():
                raise SystemExit(f"registration {st}/{src}/{organ}: incomplete target x seed grid")
            per_t = ts.mean(1)
            mean, ci_t = boot_ci(per_t)
            e = {"mean": mean, "ci_targets": ci_t}
            if metric in ("keepbest_gain", "final_gain"):
                e["ci_crossed_targets_cts_seeds"], e["crossed_dropped_resamples"] = crossed_boot(arr)
                e["ci_crossed_targets_seeds"] = crossed_boot(ts)[0]
                e["p_signflip_targets"], e["signflip_exact"] = signflip_p(per_t)
                e["p_t_targets"] = t_stats(per_t)["p_t"]
                e["targets_positive"] = int((per_t > 0).sum())
            cell[metric] = e
        key = f"{st}/{src}/{organ}"
        r = ref[key]
        if (abs(r["mean"] - cell["keepbest_gain"]["mean"]) > 1e-9
                or max(abs(a - b) for a, b in zip(r["ci"], cell["keepbest_gain"]["ci_targets"])) > 1e-9):
            raise SystemExit(f"registration {key}: target bootstrap does not reproduce the collector")
        cell.update(n_targets=len(frames), n_cts=len(cts), n_seeds=len(seeds),
                    cts_per_target=int(g.groupby("frame")["ct"].nunique().max()),
                    observed_cells_frac=float(np.isfinite(arr).mean()), q5_holds_target_ci=bool(r["holds"]),
                    q5_holds_targets_seeds_ci=bool(cell["keepbest_gain"]["ci_crossed_targets_seeds"][0] > 0),
                    q5_holds_crossed_ci=bool(cell["keepbest_gain"]["ci_crossed_targets_cts_seeds"][0] > 0))
        out[key] = cell
    keys = list(out)
    for metric in ("keepbest_gain", "final_gain"):
        for src_p, dst in (("p_signflip_targets", "holm_signflip"), ("p_t_targets", "holm_t")):
            adj = holm([out[k][metric][src_p] for k in keys])
            for k, x in zip(keys, adj):
                out[k][metric][dst] = x
    ceiling = None
    f = os.path.join(V2, "fig1", "matched_layout.json")
    if os.path.exists(f):
        b = json.load(open(f)).get("best_iou_summary") or {}
        ceiling = {"median_best_iou": b.get("median"), "mean_best_iou": b.get("mean"), "source": rel(f),
                   "definition": "matched-layout pose search against the 60 annotated Vitale frames (Fig. 1); "
                                 "a reference for Vitale liver/kidney layouts, not a bound for AUL or KidneyUS"}
    summ = {"n_cells": len(keys),
            "q5_ci_above_0": {"target_bootstrap_plan": sum(out[k]["q5_holds_target_ci"] for k in keys),
                              "targets_x_seeds": sum(out[k]["q5_holds_targets_seeds_ci"] for k in keys),
                              "targets_x_cts_x_seeds": sum(out[k]["q5_holds_crossed_ci"] for k in keys),
                              "holm_t_targets_le_0.05": sum(out[k]["keepbest_gain"]["holm_t"] <= 0.05 for k in keys),
                              "holm_signflip_targets_le_0.05": sum(out[k]["keepbest_gain"]["holm_signflip"] <= 0.05
                                                                   for k in keys)},
            "cells_holding": {"target_bootstrap_plan": [k for k in keys if out[k]["q5_holds_target_ci"]],
                              "targets_x_seeds": [k for k in keys if out[k]["q5_holds_targets_seeds_ci"]],
                              "targets_x_cts_x_seeds": [k for k in keys if out[k]["q5_holds_crossed_ci"]],
                              "holm_t_targets": [k for k in keys if out[k]["keepbest_gain"]["holm_t"] <= 0.05]},
            "final_gain_negative_all_stages_kidneyus": all(out[f"{st}/kidneyus/kidney"]["final_gain"]["mean"] < 0
                                                           for st in STAGES)}
    return {"post_hoc": True, "summary": summ, "cells": out, "layout_reference": ceiling,
            "design": ("every real target is run with all 3 seeds and with 3 of the 6 held-out CTs (rotating with the "
                       "target index, the same 3 for every seed; eval_real_registration.py --cts-per-target 3), one "
                       "run each: targets x seeds is a complete grid, CTs are incompletely crossed with targets"),
            "bootstrap": (f"{NBOOT} resamples, seed {BOOT_SEED}. targets x seeds: targets and seeds resampled "
                          "independently over the complete grid (CTs averaged within each target and seed). "
                          "targets x CTs x seeds: all three resampled independently, statistic = weighted mean over "
                          "the observed cells (pigeonhole bootstrap for an incomplete crossed design). With 3 seeds "
                          "and 6 CTs the factor-level bootstrap has few levels and tends to understate their variance."),
            "subject_clustering": ("not applied: subject IDs (Vitale sessions, KidneyUS stems) are not in "
                                   "registration_long.csv or the registration JSONs; targets are resampled by frame. "
                                   "AUL has no subject ID in any logged output.")}


# ===================================================================================== step curves (e)
def load_trajectories():
    """Per-episode pos_traj and match_traj from the result JSONs, keyed like episodes_long.csv."""
    recs = []
    for f in trajectory_files():
        try:
            dj = json.load(open(f))
        except Exception:
            continue
        if not isinstance(dj, dict):
            continue
        r0 = os.path.relpath(f, T4)
        group = r0.split(os.sep)[0] if os.sep in r0 else "."
        prov = dj.get("provenance") or {}
        m = prov.get("mask") or {}
        if "after_imitation" in dj:
            for ph in ("after_imitation", "final"):
                res = dj[ph]
                for e in res.get("episodes", []):
                    recs.append(dict(source=r0, group=group, arm=group, rung="policy", stage=dj["translator"],
                                     mask="learned" if m.get("learned") else "ct", seed=str(dj["seed"]), phase=ph,
                                     contact=res.get("contact"), eval_randomize=res.get("eval_randomize"), **_ep(e)))
        elif "rung" in dj:
            for r in dj["results"]:
                phase = {"final": "final", "imit": "after_imitation"}.get(r.get("weights"), r.get("weights"))
                for e in r.get("episodes", []):
                    recs.append(dict(source=r0, group=group, arm=dj.get("arm") or group, rung=dj["rung"],
                                     stage=dj["translator"], mask=dj.get("mask_source"),
                                     seed=str(dj.get("seed", dj.get("calib_seed"))), phase=phase,
                                     contact=r.get("contact"), eval_randomize=r.get("eval_randomize"), **_ep(e)))
    return recs


def _ep(e):
    return {"ep": e.get("ep"), "patient": e.get("patient"), "target": e.get("target"), "window": e.get("window"),
            "final_pos": e.get("final_pos"), "start_pos": e.get("start_pos"), "keepbest_step": e.get("keepbest_step"),
            "pos_traj": e.get("pos_traj"), "match_traj": e.get("match_traj")}


def step_curves(d):
    import pandas as pd
    recs = load_trajectories()
    tr = pd.DataFrame(recs)
    tr["ep"] = tr["ep"].astype(int)
    # every CSV episode must have exactly one trajectory, whose endpoints equal the CSV values
    key = ["source", "phase", "contact", "eval_randomize", "ep"]
    dd = d.copy()
    dd["ep"] = dd["ep"].astype(int)
    mrg = dd[key + ["final_pos", "start_pos"]].merge(tr[key + ["pos_traj"]], on=key, how="left", validate="one_to_one")
    if mrg["pos_traj"].isna().any():
        raise SystemExit(f"{int(mrg['pos_traj'].isna().sum())} CSV episodes have no logged trajectory")
    end_err = np.max(np.abs(np.array([t[-1] for t in mrg["pos_traj"]]) - mrg["final_pos"].values))
    start_err = np.max(np.abs(np.array([t[0] for t in mrg["pos_traj"]]) - mrg["start_pos"].values))
    if end_err > 6e-4 or start_err > 6e-4:
        raise SystemExit(f"trajectory endpoints disagree with episodes_long.csv ({end_err}, {start_err})")
    lens = sorted({len(t) for t in tr["pos_traj"]})
    out, csv_rows = collections.OrderedDict(), []
    for rid, lab, flt in table4_rows():
        sel = select(tr, flt)
        P = {}
        for metric, col in (("pos", "pos_traj"), ("S", "match_traj")):
            per_p = {p: np.mean(np.stack(g[col].values), 0) for p, g in sel.groupby("patient")}
            A = np.stack([per_p[p] for p in sorted(per_p)])
            P[metric] = A
            mean, sd = A.mean(0), A.std(0, ddof=1)
            for k in range(A.shape[1]):
                csv_rows.append({"row": rid, "metric": metric, "step": k, "mean": float(mean[k]),
                                 "sd_over_patients": float(sd[k]), "n_patients": A.shape[0],
                                 "n_episodes": int(len(sel))})
        mp, ms = P["pos"].mean(0), P["S"].mean(0)
        traj = np.stack(sel["pos_traj"].values)
        k_min = int(np.argmin(mp))
        out[rid] = {"label": lab, "n_episodes": int(len(sel)), "n_steps": int(traj.shape[1] - 1),
                    "pos_mean_by_step": [float(x) for x in mp], "S_mean_by_step": [float(x) for x in ms],
                    "pos_start": float(mp[0]), "pos_final": float(mp[-1]), "pos_min": float(mp[k_min]),
                    "pos_min_step": k_min, "pos_last4_deltas": [float(x) for x in np.diff(mp)[-4:]],
                    "frac_episodes_improving_at_last_step": float(np.mean(traj[:, -1] < traj[:, -2])),
                    "frac_episodes_min_at_last_step": float(np.mean(np.argmin(traj, 1) == traj.shape[1] - 1)),
                    "S_start": float(ms[0]), "S_final": float(ms[-1]), "S_max": float(ms.max()),
                    "S_max_step": int(np.argmax(ms)),
                    "keepbest_step_mean": float(np.mean(sel["keepbest_step"].astype(float)))}
    return out, csv_rows, {"trajectory_lengths": lens, "max_abs_endpoint_diff_mm": float(max(end_err, start_err)),
                           "n_trajectories": int(len(tr)), "n_csv_episodes_matched": int(len(mrg))}


# ===================================================================================== REINFORCE reward curves
def reward_curves():
    pat = re.compile(r"^\s*iter\s+(\d+)/(\d+)\s+reward\s+([+-]?\d+\.\d+)\s+bc\s+([+-]?\d+\.\d+)\s+logstd_mean\s+([+-]?\d+\.\d+)")
    from scipy import stats as st
    runs, rows = {}, []
    for f in trainer_logs():
        grp = os.path.basename(os.path.dirname(f))
        mm = re.match(r"(\w+?)_deploy_s(\d)\.log$", os.path.basename(f))
        stage, seed = mm.group(1), mm.group(2)
        pts = []
        for line in open(f):
            m = pat.match(line)
            if m:
                pts.append((int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
        if not pts:
            continue
        it = np.array([p[0] for p in pts], float)
        rw = np.array([p[1] for p in pts])
        k = max(1, len(rw) // 4)
        sp = st.spearmanr(it, rw)
        lr = st.linregress(it, rw)
        runs[f"{grp}/{stage}/s{seed}"] = {"n_points": len(pts), "first_quarter_mean": float(rw[:k].mean()),
                                          "last_quarter_mean": float(rw[-k:].mean()),
                                          "last_minus_first_quarter": float(rw[-k:].mean() - rw[:k].mean()),
                                          "spearman_iter_reward": float(sp.statistic), "spearman_p": float(sp.pvalue),
                                          "slope_per_100_iter": float(lr.slope * 100), "logstd_mean": pts[-1][3]}
        for p in pts:
            rows.append({"run": f"{grp}/{stage}/s{seed}", "arm": grp, "stage": stage, "seed": seed, "iter": p[0],
                         "reward": p[1], "bc": p[2], "logstd_mean": p[3]})
    by_arm = {}
    for k, v in runs.items():
        a, s, _ = k.split("/")
        by_arm.setdefault(f"{a}/{s}", []).append(v)
    summ = {k: {"n_runs": len(v), "last_minus_first_quarter_mean": float(np.mean([x["last_minus_first_quarter"] for x in v])),
                "spearman_mean": float(np.mean([x["spearman_iter_reward"] for x in v])),
                "n_runs_spearman_p_lt_0.05": int(sum(x["spearman_p"] < 0.05 for x in v))}
            for k, v in sorted(by_arm.items())}
    return {"post_hoc": True, "runs": runs, "by_arm_stage": summ,
            "note": "training reward printed every 15 iterations (20 points per run); no held-out error curve "
                    "exists (only the _imit.pt and final .pt weights are saved)"}, rows


# ===================================================================================== segmenter Dice harvest
def seg_dice():
    out = {}
    for f in seg_eval_files():
        m = re.match(r"eval_(\w+?)_s(\d)\.json$", os.path.basename(f))
        if not m:
            continue
        j = json.load(open(f))
        sv = j.get("sim_val") or {}
        rec = {"sim_heldout_mean": sv.get("mean"), "sim_heldout_frames": sv.get("frames"),
               "sim_heldout_liver": (sv.get("per_class") or {}).get("liver"),
               "sim_heldout_kidney": (sv.get("per_class") or {}).get("kidney"),
               "sim_cases": (j.get("sim_build") or {}).get("cases")}
        for key, lab in (("vitale_holdout_v2.npz", "vitale_holdout"), ("aul_liver_test.npz", "aul"),
                         ("kidneyus_holdout_test.npz", "kidneyus")):
            b = j.get(key) or {}
            for organ, v in (b.get("per_class") or {}).items():
                if organ in ("liver", "kidney"):
                    rec[f"real_{lab}_{organ}"] = v
            rec[f"real_{lab}_frames"] = b.get("frames")
        out.setdefault(m.group(1), {})[f"s{m.group(2)}"] = rec
    return {"post_hoc": True, "harvest": True, "per_stage_seed": out,
            "source": _private().SEG_HARVEST_SRC}


# ===================================================================================== pose metric note
def pose_metric_note(d):
    """How final_pos relates to the probe-face displacement p (target frame): final_pos = |V^-1 p| (the translation of
    the SE(3) logarithm) lies between |p| and f(theta)|p|, f(theta) = theta / (2 sin(theta / 2)); final_ang is the
    rotation angle theta."""
    sel = d[(d["contact"] == "on") & (d["eval_randomize"] == "off")]
    th = np.radians(sel["final_ang"].values)
    f = np.where(th > 1e-9, th / (2 * np.sin(th / 2)), 1.0)
    return {"final_ang": "SO(3) geodesic angle of R_tgt^T R_cur in degrees: arccos((trace - 1) / 2), "
                         "computed as |so3_log| in par.se3 via usbg.view_match.pose_error",
            "final_pos": "norm (mm) of the translational part v = V^-1 p of se3_log(T_tgt^-1 T_cur); |p| <= |v| <= "
                         "f(theta)|p| with f(theta) = theta / (2 sin(theta / 2))",
            "max_final_ang_deg": float(sel["final_ang"].max()),
            "max_overstatement_bound_pct": float((f.max() - 1) * 100),
            "mean_overstatement_bound_pct": float((f.mean() - 1) * 100),
            "per_axis": "not logged: pose_error returns only the two norms"}


def start_lift(d):
    sel = d[(d["contact"] == "on") & (d["eval_randomize"] == "off") & (d["group"] == "ladder") & (d["rung"] == "static")
            & (d["stage"] == "physics")]
    cl = sel["start_clamped"].astype(str).str.lower() == "true"
    return {"episodes": int(len(sel)), "frac_start_clamped": float(cl.mean()),
            "clamp_mm_mean_over_clamped": float(sel.loc[cl, "clamp_mm"].mean()),
            "clamp_mm_mean_over_all": float(sel["clamp_mm"].mean()),
            "note": "the same crossed episodes (and start poses) are used by every arm; values from the static rung"}


# ===================================================================================== cases
def case_notes(rows, fams):
    pats = sorted(rows["static"]["metrics"]["final_pos"]["per_patient"])
    tab = {p: {k: rows[k]["metrics"]["final_pos"]["per_patient"][p]
               for k in ("static", "pose_oracle", "ctmask/physics/base", "ctmask/physics/final",
                         "matched/sonospade/final", "matched/physics/final")} for p in pats}
    dis = {}
    for fid, F in fams.items():
        for n, it in F["items"].items():
            if it["sign"] < it["n"]:
                dis[f"{fid}/{n}"] = {"sign": f"{it['sign']}/{it['n']}", "dissenting": it["dissenting_patients"]}
    return {"per_patient_final_pos": tab, "dissenting_patients_in_q2_contrasts": dis}


# ===================================================================================== headline
def headline(J):
    R = J["rows"]
    g = lambda k, m: R[k]["metrics"][m]["mean"]
    H = []
    ang_ct_final = [g(f"ctmask/{s}/final", "final_ang") for s in STAGES]
    ang_ln_final = [g(f"matched/{s}/final", "final_ang") for s in STAGES]
    H.append(_rule_text("rot head").format(
        st=g('static', 'final_ang'), po=g('pose_oracle', 'final_ang'), sc=g('scripted', 'final_ang'),
        cb=g('ctmask/physics/base', 'final_ang'), c0=min(ang_ct_final), c1=max(ang_ct_final), l0=min(ang_ln_final),
        l1=max(ang_ln_final)))
    rl_ang = J["contrasts_posthoc_rotation"]["Q2c_rl_contribution"]["items"]
    up = [k for k, v in rl_ang.items() if v["mean"] > 0]
    H.append(f"REINFORCE (final minus after imitation) raises mean rotation error in {len(up)} of {len(rl_ang)} arms "
             f"(range {min(v['mean'] for v in rl_ang.values()):+.2f} to {max(v['mean'] for v in rl_ang.values()):+.2f} deg).")
    for k in ("succ_5mm_5deg", "succ_10mm_10deg"):
        H.append(_rule_text("succ head").format(
            lab=k.replace('succ_', '').replace('_', ' and '), po=100 * g('pose_oracle', k),
            cb=100 * g('ctmask/physics/base', k), c0=100 * min(g(f'ctmask/{s}/final', k) for s in STAGES),
            c1=100 * max(g(f'ctmask/{s}/final', k) for s in STAGES),
            lb0=100 * min(g(f'matched/{s}/base', k) for s in STAGES), lb1=100 * max(g(f'matched/{s}/base', k) for s in STAGES),
            l0=100 * min(g(f'matched/{s}/final', k) for s in STAGES), l1=100 * max(g(f'matched/{s}/final', k) for s in STAGES)))
    tost = J["tost_rl"]["items"]
    eq = [k for k, v in tost.items() if v["equivalent_t"]]
    neq = [k for k, v in tost.items() if not v["equivalent_t"]]
    H.append(f"TOST at 1 mm (t-based, policy rows): equivalence shown for {len(eq)} of {len(tost)} arms"
             + (f"; not shown for {', '.join(neq)}" if neq else "") + ".")
    q2 = J["contrasts_q2"]
    H.append("Holm on exact sign-flip p: " + ", ".join(f"{k} {v['n_holm_signflip_survive']}/{v['m']}" for k, v in q2.items())
             + " survive; Holm on paired t p: " + ", ".join(f"{k} {v['n_holm_t_survive']}/{v['m']}" for k, v in q2.items()) + ".")
    sim = J["rule_simulation"]["scenarios"]
    H.append(_rule_text("sim head") + f"{sim['null_normal']['rule']:.3f}, "
             f"t3 {sim['null_t3']['rule']:.3f}, skewed {sim['null_centered_exponential']['rule']:.3f} "
             f"(exact sign-flip test {sim['null_normal']['signflip_exact_p_le_0.05']:.3f}, t-test "
             f"{sim['null_normal']['t_test_p_le_0.05']:.3f}).")
    cx = {k: v["ci90_crossed_patient_seed"] for k, v in tost.items()}
    out_cx = [f"{k} [{v[0]:+.2f}, {v[1]:+.2f}]" for k, v in cx.items() if not (-1 < v[0] and v[1] < 1)]
    H.append(f"With seed variance included (patient x seed crossed bootstrap), the 90% CI of the RL contribution lies "
             f"inside +/-1 mm for {len(cx) - len(out_cx)} of {len(cx)} arms" + (f"; outside: {', '.join(out_cx)}" if out_cx else "") + ".")
    q2e = J["contrasts_q2"]["Q2e_learned_sonospade_vs"]["items"]
    H.append("SonoSPADE minus stage (learned-mask final): " + "; ".join(
        f"{k} {v['mean']:+.2f} mm, {v['sign']}/{v['n']}, exact p {v['p_signflip_exact']:.4f}, t p {v['p_t']:.4f}, "
        f"Holm t {v['holm_t']:.4f}, dissenting {','.join(v['dissenting_patients']) or 'none'}" for k, v in q2e.items()) + ".")
    bs = J["contrasts_posthoc"]["final_pos"]["PH_base_minus_static"]["items"]
    H.append("Learned-mask servo base minus static start: " + "; ".join(
        f"{k.split('/')[1]} {v['mean']:+.2f} mm ({v['sign']}/{v['n']}, t-CI [{v['ci_t'][0]:+.2f}, {v['ci_t'][1]:+.2f}])"
        for k, v in bs.items() if k.startswith("matched/")) + ".")
    im = J["contrasts_posthoc"]["final_pos"]["PH_imitation_minus_base"]["items"]
    ima = J["contrasts_posthoc"]["final_ang"]["PH_imitation_minus_base"]["items"]
    H.append("Imitation minus servo base, CT mask: translation " + ", ".join(
        f"{k.split('/')[1]} {v['mean']:+.2f} mm ({v['sign']}/{v['n']}{', rule holds' if v['holds_rule'] else ''})"
        for k, v in im.items() if k.startswith("ctmask/")) + "; rotation " + ", ".join(
        f"{v['mean']:+.2f}" for k, v in ima.items() if k.startswith("ctmask/")) + " deg.")
    rs = J["registration_crossed"]["summary"]["q5_ci_above_0"]
    H.append(f"Registration Q5 cells (of 16) whose keep-best-gain CI lies above 0: target bootstrap (plan) "
             f"{rs['target_bootstrap_plan']}, targets x seeds {rs['targets_x_seeds']}, targets x CTs x seeds "
             f"{rs['targets_x_cts_x_seeds']}; target-level t-test surviving Holm over 16 cells: {rs['holm_t_targets_le_0.05']}.")
    sc = J["step_curves"]
    b = sc["ctmask/physics/base"]
    H.append(f"CT-mask servo base error against step: {b['pos_start']:.2f} mm at step 0, {b['pos_final']:.2f} mm at step "
             f"{b['n_steps']}, last four deltas " + ", ".join(f"{x:+.2f}" for x in b["pos_last4_deltas"])
             + f" mm; {100 * b['frac_episodes_improving_at_last_step']:.0f}% of episodes still improving at the last step.")
    return H


# ===================================================================================== main
def git_head():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts/posthoc_v2.py"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return {"rev": rev, "posthoc_script_uncommitted": bool(dirty)}
    except Exception as e:
        return {"error": repr(e)}


def environment():
    import pandas
    import scipy
    try:
        import statsmodels  # noqa: F401
        sm = True
    except Exception:
        sm = False
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "statsmodels_available": sm,
            "mixed_model": _rule_text("mixed model") if not sm
            else "statsmodels present but not used by this script"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--nsim", type=int, default=10000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    PLAN, PLAN_SHA = os.path.join(ROOT, _private().PLAN_V2_REL), os.path.join(ROOT, _private().PLAN_V2_SHA_REL)
    plan_sha = sha256(PLAN)
    plan_rec = open(PLAN_SHA).read().split()[0] if os.path.exists(PLAN_SHA) else None
    if plan_rec and plan_rec != plan_sha:
        raise SystemExit(f"{os.path.basename(PLAN)} no longer matches its recorded sha256")

    d = add_success(load_eps())
    rows = summarize_rows(d)
    t4 = check_against_collector(rows, d)
    for rid, _, flt in table4_rows():                       # per (patient, seed) means for the crossed bootstrap
        sel = select(d, flt)
        if sel["seed"].nunique() > 1:
            rows[rid]["_ps"] = {mt: ps_mean(sel, mt) for mt in ("final_pos", "final_ang")}

    J = collections.OrderedDict()
    J["label"] = _private().POSTHOC_LABEL
    J["provenance"] = {"script": rel(os.path.abspath(__file__)), "script_sha256": sha256(os.path.abspath(__file__)),
                       "git": git_head(), "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "analysis_plan": {"path": rel(PLAN), "sha256": plan_sha, "recorded_sha256": plan_rec,
                                         "commit": _private().PLAN_V2_COMMIT},
                       "inputs": input_fingerprint(), "environment": environment(),
                       "parameters": {"nboot": NBOOT, "boot_seed": BOOT_SEED, "nsim": a.nsim, "sim_seed": SIM_SEED,
                                      "tost_margin_mm": TOST_MARGIN_MM, "success_thresholds": SUCCESS,
                                      "table4_filter": "contact on, evaluation DR off (the collector's Table IV filter)",
                                      "unit": "patient (n = 6); per-patient means over organ, window and seed"},
                       "checks": {"row_means_reproduce_collector": True}}
    J["pose_metric"] = pose_metric_note(d)
    J["start_lift"] = start_lift(d)
    q2 = q2_families(rows)
    J["contrasts_q2"] = run_family(rows, q2, "final_pos", planned=True)
    for fid, key in (("Q2a_observability_gap", "observability_gap"), ("Q2b_perception_gap", "perception_gap"),
                     ("Q2c_rl_contribution", "rl_contribution"), ("Q2d_ctmask_pairs", "ctmask_pairs"),
                     ("Q2e_learned_sonospade_vs", "learned_vs")):
        for n, it in J["contrasts_q2"][fid]["items"].items():
            r = t4[key][n]
            if (abs(r["mean"] - it["mean"]) > 1e-9 or r["sign"] != it["sign"] or r["holds"] != it["holds_rule"]
                    or max(abs(x - y) for x, y in zip(r["ci"], it["ci_boot"])) > 1e-9):
                raise SystemExit(f"{fid}/{n}: does not reproduce RESULTS.json table4.{key}")
    J["provenance"]["checks"]["q2_contrasts_reproduce_collector"] = True
    J["contrasts_q2_holm_all"] = holm_all(J["contrasts_q2"])
    J["contrasts_posthoc_rotation"] = run_family(rows, q2, "final_ang", planned=False)
    J["contrasts_posthoc"] = {"final_pos": run_family(rows, posthoc_families(), "final_pos", planned=False),
                              "final_ang": run_family(rows, posthoc_families(), "final_ang", planned=False)}
    J["per_organ_contrasts"] = per_organ_contrasts(d)

    tost_items = collections.OrderedDict()
    for arm, _ in ARMS:
        for st in STAGES:
            a_, b_ = rows[f"{arm}/{st}/final"], rows[f"{arm}/{st}/after_imitation"]
            pats = sorted(a_["metrics"]["final_pos"]["per_patient"])
            dd = [a_["metrics"]["final_pos"]["per_patient"][p] - b_["metrics"]["final_pos"]["per_patient"][p] for p in pats]
            seeds = sorted(a_["_ps"]["final_pos"][pats[0]])
            D = np.array([[a_["_ps"]["final_pos"][p][s] - b_["_ps"]["final_pos"][p][s] for s in seeds] for p in pats])
            tost_items[f"{arm}/{st}"] = tost(dd, TOST_MARGIN_MM, D)
    J["tost_rl"] = {"post_hoc": True, "margin_mm": TOST_MARGIN_MM, "rows": "policy rows only (the 8 trained arms)",
                    "margin_source": _rule_text("margin source"),
                    "decision": "equivalence at 1 mm if the t-based TOST p < 0.05 (equivalently, the 90% t-CI lies "
                                "inside (-1, +1) mm); a claim for every arm is an intersection-union test, so no "
                                "multiplicity adjustment is needed", "items": tost_items,
                    "n_equivalent_t": sum(v["equivalent_t"] for v in tost_items.values()),
                    "n_equivalent_signflip": sum(v["equivalent_signflip"] for v in tost_items.values()),
                    "n_equivalent_crossed_ci90": sum(v["equivalent_crossed_ci90"] for v in tost_items.values())}
    J["rule_simulation"] = simulate_rule(a.nsim)
    J["registration_crossed"] = registration_crossed(load_reg())
    sc, sc_rows, sc_check = step_curves(d)
    J["step_curves"] = sc
    J["provenance"]["checks"]["trajectories"] = sc_check
    rw, rw_rows = reward_curves()
    J["reinforce_reward"] = rw
    J["seg_dice_sim_vs_real"] = seg_dice()
    J["cases"] = case_notes(rows, J["contrasts_q2"])
    J["not_logged"] = [
        "rotation error per step (ang_tr is computed in train_view_match_rl.evaluate() but not saved)",
        "organ-layout IoU per step (seg_tr is computed but not saved)",
        "per-axis translation and rotation errors (pose_error returns only the two norms)",
        "CT-label organ IoU under learned masks (final_seg is the learned-mask IoU in those arms)",
        "a held-out error curve during REINFORCE (only the _imit.pt and final .pt weights are saved)",
        "subject IDs for the real registration targets (Vitale sessions, KidneyUS stems) in any logged output",
        "a mixed model (statsmodels absent)"]
    for r in rows.values():
        r.pop("_ps", None)
    J["rows"] = rows
    J["headline"] = headline(J)

    import csv
    with open(os.path.join(a.out, "step_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sc_rows[0]))
        w.writeheader()
        w.writerows(sc_rows)
    with open(os.path.join(a.out, "reinforce_reward.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rw_rows[0]))
        w.writeheader()
        w.writerows(rw_rows)
    J["provenance"]["outputs"] = {"step_curves.csv": sha256(os.path.join(a.out, "step_curves.csv")),
                                  "reinforce_reward.csv": sha256(os.path.join(a.out, "reinforce_reward.csv"))}
    text = json.dumps(J, indent=1)
    if EMDASH in text or TRIPLE in text:
        raise SystemExit("em-dash or triple hyphen in the output")
    path = os.path.join(a.out, "posthoc_v2.json")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {rel(path)}, step_curves.csv, reinforce_reward.csv")
    for h in J["headline"]:
        print("- " + h)
    return 0


# ===================================================================================== collector hooks
def load_checked(path=OUT_JSON):
    """(json, status). status is 'ok', 'missing' or 'stale: <what changed>' (inputs re-fingerprinted)."""
    if not os.path.exists(path):
        return None, "missing"
    J = json.load(open(path))
    now = input_fingerprint()
    old = J.get("provenance", {}).get("inputs", {})
    # in public mode rel() names a segmenter file segmenter/<path>, as the results archive's recorded fingerprint does,
    # so all three combined digests are recomputed and compared in both modes
    changed = [k for k in now if (now[k].get("sha256") != (old.get(k) or {}).get("sha256"))]
    return J, ("ok" if not changed else "stale: " + ", ".join(changed))


def compact(J, path=OUT_JSON):
    """The RESULTS.json["posthoc"] block: every summary number, without per-patient arrays or step arrays."""
    keep = ("a", "b", "n", "mean", "ci_boot", "ci_t", "ci_crossed_patient_seed", "sign", "holds_rule",
            "p_signflip_exact", "p_t", "holm_signflip", "holm_t", "dissenting_patients")

    def strip(o):
        if isinstance(o, dict):
            if "p_signflip_exact" in o and "ci_boot" in o:          # one contrast: decision-relevant fields only
                return {k: o[k] for k in keep if k in o}
            return {k: strip(v) for k, v in o.items() if k not in ("per_patient", "pos_mean_by_step", "S_mean_by_step")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    rows = {k: {"label": v["label"], "n_episodes": v["n_episodes"], "seeds": v["seeds"],
                "mean": {m: v["metrics"][m]["mean"] for m in v["metrics"]},
                "success_ci_boot_patients": {m: v["metrics"][m]["ci_boot_patients"]
                                             for m in ("succ_5mm_5deg", "succ_10mm_10deg")},
                "per_organ": {o: {m: x["mean"] for m, x in po.items()} for o, po in v["per_organ"].items()},
                "per_seed": v["per_seed"]} for k, v in J["rows"].items()}
    return {"label": J["label"], "source": rel(path), "sha256": sha256(path),
            "provenance": {k: J["provenance"][k] for k in ("script_sha256", "git", "created_utc", "analysis_plan",
                                                         "inputs", "environment", "parameters", "checks")},
            "headline": J["headline"], "pose_metric": J["pose_metric"], "start_lift": J["start_lift"], "rows": rows,
            "contrasts_q2": strip(J["contrasts_q2"]), "contrasts_q2_holm_all": {k: v for k, v in J["contrasts_q2_holm_all"].items()
                                                                                if k != "items"},
            "contrasts_posthoc_rotation": strip(J["contrasts_posthoc_rotation"]),
            "contrasts_posthoc": strip(J["contrasts_posthoc"]),
            "tost_rl": J["tost_rl"], "rule_simulation": J["rule_simulation"],
            "registration_crossed": J["registration_crossed"],
            "step_curves": strip(J["step_curves"]), "reinforce_reward": J["reinforce_reward"]["by_arm_stage"],
            "seg_dice_sim_vs_real": J["seg_dice_sim_vs_real"], "cases": J["cases"], "not_logged": J["not_logged"]}


def _f(x, d=2):
    return f"{x:+.{d}f}"


def _p(x):
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"


def markdown(J, status="ok"):
    """The post hoc section of RESULTS.md."""
    R = J["rows"]
    L = [f"**{J['label']}**", "", f"Source: `{rel(OUT_JSON)}` (written by `scripts/posthoc_v2.py`, {J['provenance']['created_utc']}; "
         f"input fingerprint {status}). Table IV filter: contact on, evaluation DR off; patient is the unit (n = 6).", "",
         "Headline numbers:", ""] + [f"- {h}" for h in J["headline"]]
    pm = J["pose_metric"]
    L += ["", _rule_text("pose metric md") + f"{pm['max_overstatement_bound_pct']:.2f}% at the largest logged angle ({pm['max_final_ang_deg']:.1f} deg; mean bound "
          f"{pm['mean_overstatement_bound_pct']:.3f}%). Per-axis errors are not logged.", ""]
    # (a), (b)
    L += ["### (a, b) Rotation error, success and match S per Table IV row (patient-level means)", "",
          "| row | n ep | translation mm | rotation deg | success 5 mm/5 deg | success 10 mm/10 deg | keep-best 10/10 | final S | contact frac |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in R.items():
        m = {x: v["metrics"][x]["mean"] for x in v["metrics"]}
        L.append(f"| {k} | {v['n_episodes']} | {m['final_pos']:.2f} | {m['final_ang']:.2f} | {100 * m['succ_5mm_5deg']:.1f}% | "
                 f"{100 * m['succ_10mm_10deg']:.1f}% | {100 * m['kb_succ_10mm_10deg']:.1f}% | {m['final_match']:.3f} | "
                 f"{m['contact_frac']:.3f} |")
    L += ["", "Per seed (final translation mm / rotation deg; seeds 0, 1, 2):", "", "| row | seed 0 | seed 1 | seed 2 |",
          "|:--|:--|:--|:--|"]
    for k, v in R.items():
        if len(v["per_seed"]) == 3:
            L.append(f"| {k} | " + " | ".join(f"{v['per_seed'][s]['final_pos']:.2f} / {v['per_seed'][s]['final_ang']:.2f}"
                                             for s in sorted(v["per_seed"])) + " |")
    sl = J["start_lift"]
    L += ["", f"Start lift: {100 * sl['frac_start_clamped']:.1f}% of starts clamped, mean lift {sl['clamp_mm_mean_over_clamped']:.2f} mm "
          f"over clamped starts ({sl['clamp_mm_mean_over_all']:.2f} mm over all)."]
    # (c)
    L += ["", "### (c) Per organ (patient-level means; translation mm / rotation deg / success 10/10)", "",
          "| row | liver | kidney |", "|:--|:--|:--|"]
    for k, v in R.items():
        po = v["per_organ"]
        cell = lambda o: (f"{po[o]['final_pos']['mean']:.2f} / {po[o]['final_ang']['mean']:.2f} / "
                          f"{100 * po[o]['succ_10mm_10deg']['mean']:.0f}%")
        L.append(f"| {k} | {cell('liver')} | {cell('kidney')} |")
    PO = J["per_organ_contrasts"]
    L += ["", "Per organ, key contrasts (translation mm; mean, k/6, t p):", ""]
    for organ in ("liver", "kidney"):
        o = PO[organ]["final_pos"]
        L.append(f"- {organ}: perception gap " + ", ".join(f"{st} {e['mean']:+.2f} ({e['sign']}/{e['n']}, {_p(e['p_t'])})"
                                                          for st, e in o["perception_gap"].items())
                 + "; SonoSPADE minus " + ", ".join(f"{st} {e['mean']:+.2f} ({e['sign']}/{e['n']}, {_p(e['p_t'])})"
                                                   for st, e in o["learned_sonospade_vs"].items()))
    # (d)
    for mt, unit in (("final_pos", "mm"), ("final_ang", "deg")):
        F = J["contrasts_posthoc"][mt]["PH_imitation_minus_base"]["items"]
        L += ["", f"### (d) Imitation minus servo base, {mt} ({unit})", "",
              "| arm/stage | mean | 95% boot CI | 95% t CI | k/6 | exact p | rule |", "|:--|:--|:--|:--|:--|:--|:--|"]
        for n, e in F.items():
            L.append(f"| {n} | {_f(e['mean'])} | [{_f(e['ci_boot'][0])}, {_f(e['ci_boot'][1])}] | "
                     f"[{_f(e['ci_t'][0])}, {_f(e['ci_t'][1])}] | {e['sign']}/{e['n']} | {_p(e['p_signflip_exact'])} | "
                     f"{'holds' if e['holds_rule'] else 'no'} |")
    F = J["contrasts_posthoc_rotation"]["Q2c_rl_contribution"]["items"]
    L += ["", "REINFORCE on rotation (final minus after imitation, deg): " + "; ".join(
        f"{n} {_f(e['mean'])} [{_f(e['ci_t'][0])}, {_f(e['ci_t'][1])}] t-CI, {e['sign']}/{e['n']}" for n, e in F.items())]
    # (e)
    L += ["", "### (e) Error and S against step (logged pos_traj and match_traj; 12 steps)", "",
          "| row | step 0 | min (step) | step 12 | last four deltas (mm) | improving at last step | S step 0 | S step 12 |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in J["step_curves"].items():
        L.append(f"| {k} | {v['pos_start']:.2f} | {v['pos_min']:.2f} ({v['pos_min_step']}) | {v['pos_final']:.2f} | "
                 + ", ".join(f"{x:+.2f}" for x in v["pos_last4_deltas"])
                 + f" | {100 * v['frac_episodes_improving_at_last_step']:.0f}% | {v['S_start']:.3f} | {v['S_final']:.3f} |")
    L += ["", "The scripted sweep is open loop; its step curve shows the sweep, not convergence. 'improving at last "
          "step' is the share of episodes whose error fell on the last step."]
    L += ["", "Not logged: " + "; ".join(J["not_logged"]) + "."]
    # (f)
    L += ["", "### (f) Statistics sensitivity for the planned Q2 contrasts (final translation error, mm)", "",
          "| family / contrast | mean | 95% boot CI | 95% t CI | patient x seed CI | k/6 | exact p | Holm exact | t p | Holm t | rule |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for fid, F in J["contrasts_q2"].items():
        for n, e in F["items"].items():
            cx = e.get("ci_crossed_patient_seed")
            L.append(f"| {fid} / {n} | {_f(e['mean'])} | [{_f(e['ci_boot'][0])}, {_f(e['ci_boot'][1])}] | "
                     f"[{_f(e['ci_t'][0])}, {_f(e['ci_t'][1])}] | "
                     + (f"[{_f(cx[0])}, {_f(cx[1])}]" if cx else "n/a")
                     + f" | {e['sign']}/{e['n']} | {_p(e['p_signflip_exact'])} | {_p(e['holm_signflip'])} | {_p(e['p_t'])} | "
                     f"{_p(e['holm_t'])} | {'holds' if e['holds_rule'] else 'no'} |")
    ha = J["contrasts_q2_holm_all"]
    L += ["", f"Holm over all {ha['m']} Q2 contrasts as one family: {ha['n_survive_signflip']} survive on exact sign-flip p, "
          f"{ha['n_survive_t']} on paired t p.", "", J["rule_simulation"]["note"], ""]
    sim = J["rule_simulation"]
    L += [f"Rule simulation (n = 6, {sim['nsim']} datasets per scenario, the collector's own resampling; verified against "
          f"collect_results_v2.rule on {sim['verified_against_collector_rule']['n_checked']} datasets, "
          f"{sim['verified_against_collector_rule']['n_mismatch']} mismatches):", "",
          "| scenario | CI-and-sign rule | CI excludes 0 | k >= 5/6 | exact sign-flip p <= 0.05 | t-test p <= 0.05 |",
          "|:--|:--|:--|:--|:--|:--|"]
    for k, v in sim["scenarios"].items():
        L.append(f"| {k} | {v['rule']:.4f} | {v['ci_excludes_0']:.4f} | {v['sign_at_least_5_of_6']:.4f} | "
                 f"{v['signflip_exact_p_le_0.05']:.4f} | {v['t_test_p_le_0.05']:.4f} |")
    # (g)
    T = J["tost_rl"]
    L += ["", f"### (g) TOST for the RL null, margin {T['margin_mm']:g} mm, policy rows only", "",
          "| arm/stage | mean | 90% t CI | TOST p (t) | TOST p (sign-flip) | 90% patient x seed CI | equivalent (t) |",
          "|:--|:--|:--|:--|:--|:--|:--|"]
    for n, e in T["items"].items():
        L.append(f"| {n} | {_f(e['mean'])} | [{_f(e['ci90_t'][0])}, {_f(e['ci90_t'][1])}] | {_p(e['p_tost_t'])} | "
                 f"{_p(e['p_tost_signflip'])} | [{_f(e['ci90_crossed_patient_seed'][0])}, {_f(e['ci90_crossed_patient_seed'][1])}] | "
                 f"{'yes' if e['equivalent_t'] else 'no'} |")
    L += ["", f"Margin: {T['margin_source']}. Decision: {T['decision']}."]
    # (h)
    G = J["registration_crossed"]
    L += ["", "### (h) Registration: target bootstrap vs crossed bootstrap (targets x CTs x seeds)", "",
          "| stage/source/organ | n | keep-best gain | target CI | crossed CI | Holm t | final-step gain | crossed CI | static / final / keep-best IoU |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, c in G["cells"].items():
        kb, fg = c["keepbest_gain"], c["final_gain"]
        L.append(f"| {k} | {c['n_targets']} | {_f(kb['mean'], 3)} | [{_f(kb['ci_targets'][0], 3)}, {_f(kb['ci_targets'][1], 3)}] | "
                 f"[{_f(kb['ci_crossed_targets_cts_seeds'][0], 3)}, {_f(kb['ci_crossed_targets_cts_seeds'][1], 3)}] | "
                 f"{_p(kb['holm_t'])} | {_f(fg['mean'], 3)} | [{_f(fg['ci_crossed_targets_cts_seeds'][0], 3)}, "
                 f"{_f(fg['ci_crossed_targets_cts_seeds'][1], 3)}] | {c['static_iou']['mean']:.3f} / {c['final_iou']['mean']:.3f} / "
                 f"{c['keepbest_iou']['mean']:.3f} |")
    gs = G["summary"]
    L += ["", "Q5 cells whose keep-best-gain CI lies above 0: " + "; ".join(
        f"{k}: {gs['q5_ci_above_0'][k]} ({', '.join(v) or 'none'})" for k, v in gs["cells_holding"].items()
        if k in gs["q5_ci_above_0"]) + f"; Holm over 16 cells on the target-level t-test: "
        f"{', '.join(gs['cells_holding']['holm_t_targets']) or 'none'}."]
    lr = G.get("layout_reference") or {}
    L += ["", f"Design: {G['design']}. Bootstrap: {G['bootstrap']}. Subject clustering: {G['subject_clustering']}"
          + (f" Layout reference: median best IoU {lr['median_best_iou']:.3f} ({lr['definition']})." if lr else "")]
    # harvest
    S = J["seg_dice_sim_vs_real"]["per_stage_seed"]
    L += ["", "### Harvest: held-out simulated Dice vs real-frame Dice of the texture-matched segmenters (seeds 0/1/2)", "",
          "| stage | sim held-out mean | Vitale holdout liver | AUL liver | KidneyUS kidney |", "|:--|:--|:--|:--|:--|"]
    for st in STAGES:
        if st in S:
            s = S[st]
            g = lambda k: " / ".join("n/a" if s[x].get(k) is None else f"{s[x][k]:.2f}" for x in sorted(s))
            L.append(f"| {st} | {g('sim_heldout_mean')} | {g('real_vitale_holdout_liver')} | {g('real_aul_liver')} | "
                     f"{g('real_kidneyus_kidney')} |")
    RW = J["reinforce_reward"]["by_arm_stage"]
    L += ["", "REINFORCE training reward (last minus first quarter of the 20 logged points, mean over seeds; mean Spearman "
          "of reward vs iteration): " + "; ".join(f"{k} {_f(v['last_minus_first_quarter_mean'], 4)}, rho {v['spearman_mean']:+.2f}"
                                                  for k, v in RW.items()) + "."]
    C = J["cases"]["dissenting_patients_in_q2_contrasts"]
    L += ["", "Dissenting patients in Q2 contrasts with k < 6: " + "; ".join(
        f"{k} ({v['sign']}: {', '.join(v['dissenting'])})" for k, v in C.items()) + "."]
    text = "\n".join(L)
    return text.replace(EMDASH, ", ")


if __name__ == "__main__":
    raise SystemExit(main())
