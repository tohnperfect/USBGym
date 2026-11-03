#!/usr/bin/env python3
"""Collect every USB Gym v2 result into outputs/v2/RESULTS.md (+ RESULTS.json) under the rules of the v2 plan: the
patient is the unit (n = 6); a contrast counts only if its 95% patient-cluster bootstrap CI excludes 0 and at least
5 of 6 patients agree in sign; each question prints the sentence the plan fixed for its outcome (development repository:
plan_private.V2_SENTENCES; in public mode no sentence is filled and the table build does not compare the ones
RESULTS.json records). Missing inputs are reported as pending, never guessed.

The "compute" section (RESULTS.json["compute"]) reads the CUDA-free cost measurements
outputs/v2/bench_render.json, bench_latency.json (--stages cut cyclegan sonospade) and
bench_closed_loop.json, plus the oracle-ladder runner log (outputs/v2/table4/runner.log: jobs, concurrency,
wall time, hours per trainer run).

The "posthoc" section (RESULTS.json["posthoc"]) is POST HOC, outside the v2 plan: it reports
outputs/v2/posthoc/posthoc_v2.json from scripts/posthoc_v2.py (run that first), after checking that its inputs have not
changed. It comes last and changes no other section.

Usage: PYTHONPATH=src .venv/bin/python scripts/collect_results_v2.py [--notes <the manuscript notes file>]

The table and figure build (make_tables.py, make_figures.py) imports the section collectors only; in public mode
(script_dirs.public()) every data path below follows the result root (script_dirs.results_root()).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
import sys

import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(CODE_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
import script_dirs as SD  # noqa: E402

ROOT = SD.results_root()          # the result files: the code tree, or the result root in public mode
V2 = os.path.join(ROOT, "outputs", "v2")
STAGES = ("physics", "cut", "cyclegan", "sonospade")
NBOOT = 10000


def _v2_text(key):
    """The wording of a rule of the v2 plan in RESULTS.md (development repository; plan_inputs.text)."""
    import plan_inputs as PI
    return PI.text(key)


def _plan_sentence(key, **fields):
    """The sentence that the v2 plan fixed for an outcome, filled with fields (development repository:
    plan_private.V2_SENTENCES). None in public mode, where those sentences are not part of the code."""
    PP = SD.private_module()
    if PP is None:
        return None
    return PP.V2_SENTENCES[key].format(**fields)


def _md(key):
    """A piece of RESULTS.md text (the development build's own wording, plan_inputs.text; neutral in public mode, where
    the collectors' Markdown is never written)."""
    import plan_inputs as PI
    return PI.text("md " + key)


# what the public build prints when it does not compare the outcome sentences RESULTS.json records (make_tables.py)
SENTENCE_SKIP = ("the comparison of the outcome sentences that RESULTS.json records with their source text (table2, "
                 "table3, table4 and the Q3 sentence of table5; the rules that choose them are re-derived)")


def _p(*a):
    return os.path.join(*a)


def boot_ci(per_patient_diff, nboot=NBOOT, seed=0):
    d = np.asarray([x for x in per_patient_diff if np.isfinite(x)], float)
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    return float(d.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def rule(per_patient_diff):
    """(holds, mean, ci, sign_count, n). Holds if the CI excludes 0 and >= 5/6 patients share its sign."""
    d = np.asarray([x for x in per_patient_diff if np.isfinite(x)], float)
    m, ci = boot_ci(d)
    if not len(d):
        return False, m, ci, 0, 0
    sgn = np.sign(m)
    agree = int((np.sign(d) == sgn).sum())
    holds = (ci[0] > 0 or ci[1] < 0) and agree >= 5
    return bool(holds), m, ci, agree, len(d)


def fmt_ci(m, ci, agree, n, unit=""):
    return f"{m:+.2f}{unit} [{ci[0]:+.2f}, {ci[1]:+.2f}], sign {agree}/{n}"


# ----------------------------------------------------------------------------------- the landscape table
def table3(out):
    f = _p(V2, "view_match", "table3_stats.json")
    if not os.path.exists(f):
        return "pending (outputs/v2/view_match/table3_stats.json)"
    t = json.load(open(f))
    lines = [_md("t3 head"), "|---|---|---|---|---|---|"]
    verdict = {}
    for scope, blk in [("pooled", t["pooled_all_organs"])] + [(o, b) for o, b in t["by_organ"].items()]:
        held = []
        for other, c in blk["contrasts"].items():
            pl = c["patient_level"]
            ok, dz, ci, agree, n = rule(list(pl["per_patient_dz"].values()))
            held.append(ok)
            lines.append(f"| {scope} | {c['n_views']} | {other} | {c['mean_ref']:.3f} / {c['mean_other']:.3f} | "
                         f"{dz:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], {agree}/{n} | {'yes' if ok else 'no'} |")
        verdict[scope] = held
    pooled = verdict["pooled"]
    others = list(t["pooled_all_organs"]["contrasts"])
    if all(pooled):
        s = _plan_sentence("q1 all")
    elif any(pooled):
        s = _plan_sentence("q1 some", better=", ".join(o for o, h in zip(others, pooled) if h),
                           rest=", ".join(o for o, h in zip(others, pooled) if not h))
    else:
        s = _plan_sentence("q1 none")
    out["table3"] = {"verdict_pooled": dict(zip(others, pooled)), "sentence": s, "source": f}
    return "\n".join(lines) + f"\n\n**Plan sentence (Q1):** {s}\n\nSource: `{os.path.relpath(f, ROOT)}`"


# ----------------------------------------------------------------------------------- cue ablation
def cue(out):
    d = _p(V2, "view_match", "cue_ablation")
    rows, res = [], {}
    for c in ("none", "shadow", "speckle", "psf", "all"):
        f = _p(d, f"landscape_{c}.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        tr = j.get("translators") or j.get("summary") or {}
        res[c] = {s: tr.get(s, {}).get("app_closeness_per_view_mean") for s in STAGES}
    if not res:
        return "pending"
    out["cue_ablation"] = res
    hdr = "| cue removed | " + " | ".join(STAGES) + " |\n|---|" + "---|" * len(STAGES)
    body = "\n".join(f"| {c} | " + " | ".join("n/a" if v is None or not np.isfinite(v) else f"{v:.3f}" for v in (res[c][s] for s in STAGES)) + " |" for c in res)
    return _md("cue head") + hdr + "\n" + body


# ----------------------------------------------------------------------------------- the realism table
def table2(out):
    f = _p(V2, "realism", "realism_sector.json")
    if not os.path.exists(f):
        return "pending"
    j = json.load(open(f))
    lines = ["| variant | W1 | KID | FID | CNN C2ST AUC |", "|---|---|---|---|---|"]
    for name, v in j["variants"].items():
        w1, kid, fid = v.get("w1_in_sector"), v.get("kid"), v.get("fid_secondary")
        c2s = f"{v.get('c2st_auc_mean', float('nan')):.3f} +/- {v.get('c2st_auc_std', float('nan')):.3f}"
        lines.append(f"| {name} | {'n/a' if w1 is None else f'{w1:.4f}'} | {'n/a' if kid is None else f'{kid:.4f}'} | "
                     f"{'n/a' if fid is None else f'{fid:.1f}'} | {c2s} |")
    pb = j.get("physics_beats_lotus_all_six", {})
    n_ok = sum(1 for a, b in (pb.get("w1_per_case") or {}).values() if a < b)
    pooled_ok = None
    try:
        pooled_ok = j["variants"]["physics"]["w1_in_sector"] < j["variants"]["prior LOTUS"]["w1_in_sector"]
    except Exception:
        pass
    holds = n_ok >= 5 and pooled_ok is not False
    s = _plan_sentence("q4 holds" if holds else "q4 not")
    out["table2"] = {"physics_closer_cases": n_ok, "pooled_closer": pooled_ok, "sentence": s, "source": f}
    return ("\n".join(lines) + _md("t2 w1") + f"{n_ok}/6" + _md("t2 pooled") + f"{pooled_ok}). "
            f"C2ST permutation null on physics: {j.get('c2st_permutation_null_physics')}.\n\n**Plan sentence (Q4):** {s} "
            "No stage is called realistic.\n\nSource: `" + os.path.relpath(f, ROOT) + "`")


# ----------------------------------------------------------------------------------- oracle-ladder
def load_eps():
    f = _p(V2, "table4", "episodes_long.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else []


def per_patient(rows, **flt):
    sel = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in flt.items())]
    acc = collections.defaultdict(list)
    for r in sel:
        try:
            acc[r["patient"]].append(float(r["final_pos"]))
        except (TypeError, ValueError):
            pass
    return {p: float(np.mean(v)) for p, v in acc.items()}, len(sel), sorted({r["seed"] for r in sel})


def diff_pp(a, b):
    ps = sorted(set(a) & set(b))
    return [a[p] - b[p] for p in ps]


def table4(out):
    rows = load_eps()
    if not rows:
        return "pending"
    res, lines = {}, []
    base = dict(contact="on", eval_randomize="off")
    ladder = {}
    for name, flt in [("static", dict(group="ladder", rung="static")),
                      ("scripted", dict(group="ladder", rung="scripted")),
                      ("pose_oracle", dict(group="ladder", rung="pose_oracle"))]:
        pp, n, seeds = per_patient(rows, **flt, **base)
        ladder[name] = pp
    cells = {}
    for st in STAGES:
        for arm, mask in (("ctmask", "ct"), ("matched", "learned")):
            for phase in ("after_imitation", "final"):
                cells[(arm, st, phase)] = per_patient(rows, group=arm, arm=arm, stage=st, phase=phase, **base)
            cells[(arm, st, "base")] = per_patient(rows, group="ladder", rung="ibvs_base", stage=st, mask=mask, **base)
    lines.append(_md("t4 head"))
    lines.append("|---|---|---|---|---|---|---|---|")
    m = lambda pp: float(np.mean(list(pp.values()))) if pp else float("nan")
    for st in STAGES:
        c = cells
        lines.append(f"| {st} | {m(c[('ctmask', st, 'base')][0]):.2f} | {m(c[('ctmask', st, 'after_imitation')][0]):.2f} | "
                     f"{m(c[('ctmask', st, 'final')][0]):.2f} | {m(c[('matched', st, 'base')][0]):.2f} | "
                     f"{m(c[('matched', st, 'after_imitation')][0]):.2f} | {m(c[('matched', st, 'final')][0]):.2f} | "
                     f"{','.join(c[('matched', st, 'final')][2]) or '-'} |")
    lines.append(_md("t4 rungs").format(st=m(ladder["static"]), sc=m(ladder["scripted"]), po=m(ladder["pose_oracle"])))
    # gaps and contrasts (sentences only when every stage has all three seeds in both arms)
    sent = []
    complete = all(len(cells[(arm, st, "final")][2]) == 3 for arm in ("ctmask", "matched") for st in STAGES)
    if not complete:
        sent.append("PENDING: a stage lacks a seed in one of the arms, so the outcome sentences are left out.")
    obs = [rule(diff_pp(cells[("ctmask", st, "final")][0], ladder["pose_oracle"])) for st in STAGES]
    lines.append(_md("t4 obs") +
                 "; ".join(f"{st} {fmt_ci(o[1], o[2], o[3], o[4], ' mm')}" for st, o in zip(STAGES, obs)))
    perc = {st: rule(diff_pp(cells[("matched", st, "final")][0], cells[("ctmask", st, "final")][0])) for st in STAGES}
    lines.append(_md("t4 perc") +
                 "; ".join(f"{st} {fmt_ci(p[1], p[2], p[3], p[4], ' mm')}" for st, p in perc.items()))
    rl = {}
    for arm in ("ctmask", "matched"):
        for st in STAGES:
            rl[(arm, st)] = rule(diff_pp(cells[(arm, st, "final")][0], cells[(arm, st, "after_imitation")][0]))
    lines.append("RL contribution (final minus after-imitation): " +
                 "; ".join(f"{a}/{st} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for (a, st), v in rl.items() if np.isfinite(v[1])))
    if complete and all(abs(v[1]) < 1 for v in rl.values() if np.isfinite(v[1])) and rl:
        sent.append(_plan_sentence("q7 rl"))
    # CT-mask arm across stages, all pairs. This sentence needs every pairwise CI to include 0 (not the CI-plus-sign
    # rule); otherwise the pairs whose CI excludes 0 are reported.
    pair_holds = []
    for i, a in enumerate(STAGES):
        for b in STAGES[i + 1:]:
            r_ = rule(diff_pp(cells[("ctmask", a, "final")][0], cells[("ctmask", b, "final")][0]))
            pair_holds.append(((a, b), r_))
    ci_out = [(ab, r_) for ab, r_ in pair_holds if r_[2] and np.all(np.isfinite(r_[2])) and (r_[2][0] > 0 or r_[2][1] < 0)]
    lines.append(_md("t4 pairs") +
                 "; ".join(f"{a} vs {b} {fmt_ci(r_[1], r_[2], r_[3], r_[4], ' mm')}" for (a, b), r_ in pair_holds))
    if not complete:
        pass
    elif pair_holds and not ci_out:
        sent.append(_plan_sentence("q2 mask fixed"))
    elif pair_holds:
        sent.append(_plan_sentence("q2 mask fixed pairs", pairs=", ".join(
            f"{a} vs {b} ({fmt_ci(r_[1], r_[2], r_[3], r_[4], ' mm')}; "
            f"{_md('rule yes') if r_[0] else _md('rule no')})"
            for (a, b), r_ in ci_out)))
    # learned arm: SonoSPADE vs others
    lc = {o: rule(diff_pp(cells[("matched", "sonospade", "final")][0], cells[("matched", o, "final")][0]))
          for o in ("physics", "cut", "cyclegan")}
    lines.append(_md("t4 learned") +
                 "; ".join(f"vs {o} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for o, v in lc.items()))
    if not complete:
        pass
    elif lc and all(v[0] and v[1] < 0 for v in lc.values()):
        sent.append(_plan_sentence("q2 learned lowest"))
    elif lc and not any(v[0] for v in lc.values()):
        sent.append(_plan_sentence("q2 learned same"))
    elif lc:
        sent.append(_plan_sentence("q2 learned measured", each="; ".join(
            f"SonoSPADE vs {o}: {'lower' if v[0] and v[1] < 0 else ('higher' if v[0] else 'no difference')}" for o, v in lc.items())))
    # contact deltas
    cdelta = {}
    for arm in ("ctmask", "matched"):
        for st in STAGES:
            on, _, _ = per_patient(rows, group="contact", arm=arm, stage=st, phase="final", contact="on", eval_randomize="off")
            off, _, _ = per_patient(rows, group="contact", arm=arm, stage=st, phase="final", contact="off", eval_randomize="off")
            if on and off:
                cdelta[f"{arm}/{st}"] = rule(diff_pp(off, on))
    # Q7 asks for the delta of every rung, each as measured. The reference rungs use
    # no image, so the four stage files hold identical episodes: the delta is taken per stage and stored
    # once when all four agree (per stage otherwise). The servo base is stored per mask and stage.
    for rung in ("static", "scripted", "pose_oracle"):
        per_st = []
        for st in STAGES:
            on, _, _ = per_patient(rows, group="ladder", rung=rung, stage=st, contact="on", eval_randomize="off")
            off, _, _ = per_patient(rows, group="ladder", rung=rung, stage=st, contact="off", eval_randomize="off")
            if on and off:
                per_st.append((st, rule(diff_pp(off, on))))
        if per_st and len(per_st) == len(STAGES) and all(v == per_st[0][1] for _, v in per_st):
            cdelta[rung] = per_st[0][1]
        else:
            for st, v in per_st:
                cdelta[f"{rung}/{st}"] = v
    for mask in ("ct", "learned"):
        for st in STAGES:
            on, _, _ = per_patient(rows, group="ladder", rung="ibvs_base", stage=st, mask=mask, contact="on",
                                   eval_randomize="off")
            off, _, _ = per_patient(rows, group="ladder", rung="ibvs_base", stage=st, mask=mask, contact="off",
                                    eval_randomize="off")
            if on and off:
                cdelta[f"ibvs_base/{mask}/{st}"] = rule(diff_pp(off, on))
    if cdelta:
        lines.append("Contact off minus on (final, mm): " + "; ".join(
            f"{k} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for k, v in cdelta.items()))
    # exploratory, outside the v2 plan: every arm against the static start, and the CT-to-learned-mask gap of
    # SonoSPADE against each other stage (difference in differences)
    vs_static = {}
    for st in STAGES:
        vs_static[f"ctmask/{st}"] = rule(diff_pp(cells[("ctmask", st, "final")][0], ladder["static"]))
        vs_static[f"matched/{st}"] = rule(diff_pp(cells[("matched", st, "final")][0], ladder["static"]))
    vs_static["scripted"] = rule(diff_pp(ladder["scripted"], ladder["static"]))
    did = {}
    for o in ("physics", "cut", "cyclegan"):
        gs = {st: {p: cells[("matched", st, "final")][0][p] - cells[("ctmask", st, "final")][0][p]
                   for p in set(cells[("matched", st, "final")][0]) & set(cells[("ctmask", st, "final")][0])}
              for st in ("sonospade", o)}
        did[o] = rule(diff_pp(gs["sonospade"], gs[o]))
    lines.append(_md("t4 vs static") + "; ".join(
        f"{k} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for k, v in vs_static.items()))
    lines.append(_md("t4 did") + "; ".join(
        f"vs {o} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for o, v in did.items()))
    J = lambda v: {"holds": v[0], "mean": v[1], "ci": v[2], "sign": v[3], "n": v[4]}
    out["table4"] = {"sentences": sent, "source": _p(V2, "table4", "episodes_long.csv"),
                     "cell_means_mm": {f"{arm}/{st}/{ph}": m(cells[(arm, st, ph)][0]) for (arm, st, ph) in cells},
                     "rungs_mm": {k: m(v) for k, v in ladder.items()},
                     "observability_gap": {st: J(o) for st, o in zip(STAGES, obs)},
                     "perception_gap": {st: J(v) for st, v in perc.items()},
                     "rl_contribution": {f"{a}/{st}": J(v) for (a, st), v in rl.items()},
                     "ctmask_pairs": {f"{a}-{b}": J(r_) for (a, b), r_ in pair_holds},
                     "learned_vs": {o: J(v) for o, v in lc.items()},
                     "contact_off_minus_on": {k: J(v) for k, v in cdelta.items()},
                     "exploratory_final_minus_static": {k: J(v) for k, v in vs_static.items()},
                     "exploratory_perception_gap_sonospade_minus": {o: J(v) for o, v in did.items()}}
    return "\n".join(lines) + "\n\n**Plan sentences (Q2, Q7):** " + " ".join(x for x in sent if x is not None)


# ----------------------------------------------------------------------------------- DR policy
def dr_policy(out):
    rows = load_eps()
    sel = [r for r in rows if r.get("group") in ("dr", "dr_eval", "dr_on")]
    if not sel:
        return "pending"
    acc = collections.defaultdict(list)
    for r in sel:
        acc[(r["group"], r["arm"], r.get("stage"), r["phase"], r["eval_randomize"])].append(float(r["final_pos"]))
    lines = [_md("dr head"), "|---|---|---|---|---|---|---|"]
    for k, v in sorted(acc.items()):
        lines.append("| " + " | ".join(str(x) for x in k) + f" | {np.mean(v):.2f} | {len(v)} |")
    # descriptive patient-level deltas against the CT-mask arm without DR (final weights)
    ref = per_patient(rows, group="dr_eval", arm="ctmask", phase="final", eval_randomize="off")[0]
    deltas = {}
    for name, flt in (("eval DR only", dict(arm="ctmask", eval_randomize="episode")),
                      ("training DR only", dict(arm="dr_on", eval_randomize="off")),
                      ("training and eval DR", dict(arm="dr_on", eval_randomize="episode"))):
        pp = per_patient(rows, group="dr_eval", phase="final", **flt)[0]
        if pp and ref:
            deltas[name] = rule(diff_pp(pp, ref))
    if deltas:
        lines.append(_md("dr deltas") + "; ".join(
            f"{k} {fmt_ci(v[1], v[2], v[3], v[4], ' mm')}" for k, v in deltas.items()))
        out["dr_policy"] = {k: {"holds": v[0], "mean": v[1], "ci": v[2], "sign": v[3], "n": v[4]} for k, v in deltas.items()}
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- registration
def registration(out):
    f = _p(V2, "table4", "registration_long.csv")
    if not os.path.exists(f):
        return "pending"
    rows = [r for r in csv.DictReader(open(f)) if r.get("mask") == "seg"]
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        try:
            imp = float(r["keepbest_iou"]) - float(r["static_iou"])
        except (TypeError, ValueError):
            continue
        groups[(r["stage"], r["src"], r["organ"])][r["frame"]].append(imp)
    lines = [_md("reg head"), "|---|---|---|---|---|---|"]
    res = {}
    for k in sorted(groups):
        per_t = [float(np.mean(v)) for v in groups[k].values()]
        mean, ci = boot_ci(per_t)
        holds = ci[0] > 0
        res["/".join(k)] = {"n_targets": len(per_t), "mean": mean, "ci": ci, "holds": holds}
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(per_t)} | {mean:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {'yes' if holds else 'no'} |")
    out["registration"] = res
    return _md("reg intro") + "\n".join(lines) + _v2_text("q5 sentence")


# ----------------------------------------------------------------------------------- Table V
def table5(out):
    res, text = {}, []
    for side in ("vitale", "vitaleho"):
        SEG = SD.segmenter_root()
        jf = _p(SEG, "outputs", "v2", f"tablev_v2_{side}.json")
        cmd = [os.path.join(ROOT, ".venv", "bin", "python"), "scripts/agg_tablev_v2.py", "--dir", "outputs/v2",
               "--real-side", side, "--organs", "liver,gallbladder,vessel,kidney,spleen", "--json-out", jf]
        env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:src", USBG_SEAT="anatomical")
        p = subprocess.run(cmd, cwd=SEG, capture_output=True, text=True, env=env)
        text.append(f"**{side}** (`agg_tablev_v2.py --real-side {side}`):\n\n```\n{(p.stdout or p.stderr).strip()[-2500:]}\n```")
        if os.path.exists(jf):
            res[side] = json.load(open(jf))
    out["table5"] = {k: v for k, v in res.items()}
    verdicts = []
    for side, d in res.items():
        try:
            st = d["stages"]
            # 4-organ total (spleen descriptive): per-seed counts minus spleen hits are not stored per seed, so use
            # the per-organ tallies (of 3 seeds) for liver, gallbladder, vessel, kidney.
            four = {k: sum(v["per_organ_of_3"][o] for o in ("liver", "gallbladder", "vessel", "kidney")) / 3.0 for k, v in st.items()}
            glob_low = all(four[g] <= four[r] - 0.67 for g in ("cut", "cyclegan") for r in ("physics", "sonospade"))
            sent = _plan_sentence("q3 global" if glob_low else "q3 same")
            verdicts.append(f"- {side}: 4-organ mean per seed " + ", ".join(f"{k} {v:.2f}" for k, v in four.items())
                            + f" -> **{sent}**")
            out["table5"][side + "_verdict"] = {"four_organ_mean": four, "sentence": sent}
        except Exception as e:
            verdicts.append(f"- {side}: could not apply the rule ({e!r})")
    return "\n\n".join(text) + (_v2_text("q3 rule") + "\n".join(verdicts))


# ----------------------------------------------------------------------------------- DR landscape
def dr_landscape(out):
    f = _p(V2, "view_match", "landscape_dr.json")
    if not os.path.exists(f):
        return "pending"
    j = json.load(open(f))
    res = j.get("results") or {}
    terms = ("app", "seg_ct", "seg_learned", "combined_ct_0.6/0.4", "combined_learned_0.6/0.4")
    pv = lambda arm, st, t: ((res.get(arm, {}).get(st, {}) or {}).get(t) or {}).get("per_view_mean", float("nan"))
    lines = [_md("drl intro"), "", "| arm | stage | " + " | ".join(terms) + " |", "|---|---|" + "---|" * len(terms)]
    for arm in res:
        for st in res[arm]:
            lines.append(f"| {arm} | {st} | " + " | ".join(f"{pv(arm, st, t):.3f}" for t in terms) + " |")
    verdict = {}
    for st in STAGES:
        for mask in ("ct", "learned"):
            comb = f"combined_{mask}_0.6/0.4"
            keep = pv("cross", st, comb) / pv("off", st, comb)
            app_drop = pv("cross", st, "app") < pv("off", st, "app")
            verdict[f"{st}/{mask}"] = {"combined_retained": keep, "app_drops": bool(app_drop),
                                       "rule_holds": bool(keep >= 0.8 and app_drop)}
    out["dr_landscape"] = {"source": f, "verdict": verdict}
    vl = ["", _v2_text("q6 rule"), ""]
    vl += [f"- {k}: combined retained {v['combined_retained']:.0%}, appearance drops: {v['app_drops']} -> "
           f"{'holds' if v['rule_holds'] else 'does not hold'}" for k, v in verdict.items()]
    return "\n".join(lines + vl)


def misc(out):
    s = []
    f = _p(V2, "fig1", "matched_layout.json")
    if os.path.exists(f):
        j = json.load(open(f))
        s.append(f"Fig. 1 matched-layout search: {json.dumps(j.get('summary', {k: j[k] for k in list(j)[:3]}))[:400]}")
    f = _p(V2, "dr", "tab_dr.tex")
    if os.path.exists(f):
        s.append(f"DR parameter table: `{os.path.relpath(f, ROOT)}` (LaTeX, ready to \\input).")
    w = _p(ROOT, "data", "cache", "windows", "usbg-windows-v1.json")
    if os.path.exists(w):
        d = json.load(open(w))
        hdr = "| case | liver | kidney R | kidney L | gallbladder |\n|---|---|---|---|---|"
        rows = [f"| {c} | " + " | ".join(str(len(r['windows'].get(o, []))) for o in ("liver", "kidney_right", "kidney_left", "gallbladder")) + " |"
                for c, r in sorted(d["cases"].items())]
        s.append("Screened windows per case (usbg-windows-v1):\n\n" + hdr + "\n" + "\n".join(rows))
    return "\n\n".join(s)


# ----------------------------------------------------------------------------------- compute (CUDA-free)
def _iqr(d):
    return {k: d.get(k) for k in ("median_ms", "p25_ms", "p75_ms", "mean_ms")}


def runner_log_stats(path):
    """Jobs, concurrency, wall time and per-trainer hours from the oracle-ladder runner log. Durations come
    from the START/DONE timestamps of each job (the DONE line's own 'x.xx h' is that, rounded)."""
    import datetime as _dt
    import re
    fmt = "%Y-%m-%d %H:%M:%S"
    start, done, rc, k_max, first, last = {}, {}, {}, 0, None, None
    pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+(START|DONE|SKIP|SUMMARY|FAIL\w*|BLOCKED|STALE)\s*(\S*)(.*)$")
    for line in open(path):
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        t = _dt.datetime.strptime(m.group(1), fmt)
        first = first or t
        last = t
        ev, name, rest = m.group(2), m.group(3), m.group(4)
        if ev == "START":
            start[name] = t
            ks = re.search(r"\[\d+/(\d+)\]", rest)
            k_max = max(k_max, int(ks.group(1)) if ks else 0)
        elif ev == "DONE":
            done[name] = t
            r = re.search(r"rc=(-?\d+)", rest)
            rc[name] = int(r.group(1)) if r else None
    dur_h = {n: (done[n] - start[n]).total_seconds() / 3600.0 for n in done if n in start}
    train = {n: h for n, h in dur_h.items() if n.startswith("train:")}
    by_arm, by_arm_stage = collections.defaultdict(list), collections.defaultdict(list)
    for n, h in train.items():
        _, arm, stage, _seed = n.split(":")
        by_arm[arm].append(h)
        by_arm_stage[f"{arm}/{stage}"].append(h)
    med = lambda v: float(np.median(v))
    return {
        "source": os.path.relpath(path, ROOT),
        "jobs_done": len(done), "jobs_rc0": sum(1 for v in rc.values() if v == 0),
        "train_jobs": len(train), "eval_jobs": len(done) - len(train), "concurrent_jobs": k_max,
        "first_event": first.strftime(fmt) if first else None, "last_event": last.strftime(fmt) if last else None,
        "wall_h": (last - first).total_seconds() / 3600.0 if first else None,
        "job_hours_total": float(sum(dur_h.values())), "train_job_hours_total": float(sum(train.values())),
        "train_h_median_all": med(list(train.values())) if train else None,
        "train_h_min": min(train.values()) if train else None, "train_h_max": max(train.values()) if train else None,
        "train_h_median_by_arm": {a: med(v) for a, v in sorted(by_arm.items())},
        "train_n_by_arm": {a: len(v) for a, v in sorted(by_arm.items())},
        "train_h_median_by_arm_stage": {a: med(v) for a, v in sorted(by_arm_stage.items())},
        "train_n_by_arm_stage": {a: len(v) for a, v in sorted(by_arm_stage.items())},
        "train_h_by_job": {n: round(h, 4) for n, h in sorted(train.items())},
    }


def compute(out):
    """CUDA-free cost on the benchmark machine: physics render chain, texture stages, one closed-loop
    step (CPU and MPS, batch 1, one process), and the oracle-ladder run's training cost. Pending, never
    guessed, when a benchmark JSON is missing."""
    fr, fl, fc = _p(V2, "bench_render.json"), _p(V2, "bench_latency.json"), _p(V2, "bench_closed_loop.json")
    flog = _p(V2, "table4", "runner.log")
    miss = [os.path.relpath(f, ROOT) for f in (fr, fl, fc, flog) if not os.path.exists(f)]
    if miss:
        return "pending (" + ", ".join(miss) + ")"
    br, bl, bc = json.load(open(fr)), json.load(open(fl)), json.load(open(fc))
    if not br.get("render_device_explicit"):
        return ("pending: outputs/v2/bench_render.json predates the render-device fix (its cpu row rendered "
                "on MPS); rerun scripts/bench_render.py")
    if "per_stage" not in bl:
        return "pending: outputs/v2/bench_latency.json times SonoSPADE only; rerun with --stages cut cyclegan sonospade"
    devices = [d for d in ("cpu", "mps") if d in bc["per_device"]]
    res = {"machine": bc.get("machine"), "batch": 1, "processes": 1,
           "sources": {"render": os.path.relpath(fr, ROOT), "texture": os.path.relpath(fl, ROOT),
                       "closed_loop": os.path.relpath(fc, ROOT), "training": os.path.relpath(flog, ROOT)},
           "cuda_available": bool((bc.get("machine") or {}).get("cuda_available")),
           "cpu_run_mps_bytes": (bc["per_device"].get("cpu") or {}).get("mps_allocated_bytes_at_end"),
           "torch_threads": {d: bc["per_device"][d].get("torch_threads") for d in devices},
           "render": {}, "texture": {}, "closed_loop": {}}
    res["render_setup"] = {"case": br.get("case"), "n": br.get("n"), "warmup": br.get("warmup"),
                           "seat": br.get("seat")}
    res["texture_setup"] = {"n": bl.get("n"), "pairs": bl.get("pairs"), "warmup": bl.get("warmup_frames")}
    res["closed_loop_setup"] = {"policy_seed": bc.get("seed"), "eval_seed": bc.get("eval_seed")}
    for d in devices:
        pd = br["per_device"][d]
        res["render"][d] = {k: _iqr(pd[k]) for k in ("slice", "render", "warp", "render+warp", "frame")}
        res["render"][d]["n"] = br.get("n")
    for st, v in bl["per_stage"].items():
        res["texture"][st] = {d: dict(_iqr(v["per_device"][d]), n=v["per_device"][d].get("n_timed"))
                              for d in devices if d in v["per_device"]}
    for d in devices:
        res["closed_loop"][d] = {}
        for st, v in bc["per_device"][d]["stages"].items():
            s = v["step"]
            res["closed_loop"][d][st] = {"median_ms": s["median_ms"], "p25_ms": s["p25_ms"], "p75_ms": s["p75_ms"],
                                         "mean_ms": s["mean_ms"], "n_steps": s["n"],
                                         "steps_per_s": s["steps_per_s_from_median"],
                                         "n_timed_episodes": v["n_timed_episodes"],
                                         "steps_per_episode": v["steps_per_episode"],
                                         "n_warmup_steps": v["n_warmup_episodes"] * v["steps_per_episode"],
                                         "components_median_ms": v["components_median_ms"],
                                         "other_median_ms": v["other_median_ms"]}
    res["training"] = runner_log_stats(flog)
    out["compute"] = res

    f2 = lambda x: f"{x:.1f}"
    iq = lambda d: f"{d['median_ms']:.1f} [{d['p25_ms']:.1f}, {d['p75_ms']:.1f}]"
    m = res["machine"] or {}
    lines = [f"Machine: {m.get('chip')}, {m.get('memory_gb')} GB, {m.get('system')}; CUDA available: "
             f"{res['cuda_available']}; torch threads {res['torch_threads']}. " + _md("compute a")
             + f"{res['cpu_run_mps_bytes']}" + _md("compute b"), "",
             "| quantity | " + " | ".join(d.upper() for d in devices) + " |", "|---|" + "---|" * len(devices)]
    for k, lab in (("frame", "physics frame (slice + render + resize + sector warp)"), ("render", "physics render only")):
        lines.append(f"| {lab}, ms | " + " | ".join(iq(res["render"][d][k]) for d in devices) + " |")
    for st in ("cut", "cyclegan", "sonospade"):
        if st in res["texture"]:
            lines.append(f"| texture stage {st}, ms/frame | " + " | ".join(iq(res["texture"][st][d]) for d in devices) + " |")
    for st in STAGES:
        if all(st in res["closed_loop"][d] for d in devices):
            lines.append(f"| closed-loop step, {st}, ms | " + " | ".join(iq(res["closed_loop"][d][st]) for d in devices) + " |")
            lines.append(f"| closed-loop steps/s, {st} | " + " | ".join(f2(res["closed_loop"][d][st]["steps_per_s"]) for d in devices) + " |")
    t = res["training"]
    lines += ["", _md("compute parts"), ""]
    for d in devices:
        for st, v in res["closed_loop"][d].items():
            lines.append(f"- {d} / {st}: " + ", ".join(f"{k} {x:.1f}" for k, x in v["components_median_ms"].items())
                         + f", other {v['other_median_ms']:.1f}")
    lines += ["", _md("compute cost").format(**t)
              + _md("compute run") + _md("compute median").format(**t) +
              ", ".join(f"{a} {h:.2f} h (n={t['train_n_by_arm'][a]})" for a, h in t["train_h_median_by_arm"].items()) + ".",
              "", "Median trainer hours by arm/stage: " +
              ", ".join(f"{k} {h:.2f}" for k, h in t["train_h_median_by_arm_stage"].items()) + "."]
    return "\n".join(lines)


# ==================================================================================== post hoc (v2.1)
def posthoc(out):
    """POST HOC layer, outside the v2 plan: reads outputs/v2/posthoc/posthoc_v2.json (written by
    scripts/posthoc_v2.py), re-fingerprints its inputs and reports nothing from a stale or missing file.
    It only adds RESULTS.json["posthoc"]; no section above reads or changes it."""
    import posthoc_v2 as PH
    J, status = PH.load_checked()
    if J is None:
        return "pending (run scripts/posthoc_v2.py)"
    if status != "ok":
        return f"STALE ({status}): rerun scripts/posthoc_v2.py; no post hoc number is reported."
    out["posthoc"] = PH.compact(J)
    return PH.markdown(J, status)


def main(argv=None):
    ap = argparse.ArgumentParser()
    import plan_inputs as PI
    PP = PI.private()                  # the development repository writes RESULTS.md
    ap.add_argument("--notes", default=_p(ROOT, PP.NOTES_REL))
    ap.add_argument("--out", default=_p(V2, "RESULTS.md"))
    a = ap.parse_args(argv)
    out = {}
    # the section titles and the head of RESULTS.md (development repository only)
    secs = list(zip(PP.RESULTS_TITLES, (table3, cue, table2, table4, dr_policy, registration, table5, dr_landscape,
                                        misc, compute, posthoc)))
    md = PP.results_md_head()
    for title, fn in secs:
        try:
            body = fn(out)
        except Exception as e:                      # never hide a broken section
            body = f"ERROR while collecting: {e!r}"
        md += [f"## {title}", "", body, ""]
    text = "\n".join(md)
    bad = [c for c in ("\u2014",) if c in text]
    if bad:
        text = text.replace("\u2014", ", ")
    open(a.out, "w").write(text)
    json.dump(out, open(a.out.replace(".md", ".json"), "w"), indent=1, default=str)
    updated = False
    if a.notes and os.path.exists(a.notes):
        h = open(a.notes).read()
        key = "## RESULTS (generated)"
        h = h[:h.index(key)] if key in h else h
        open(a.notes, "w").write(h + key + "\n" + text.split("\n", 2)[2])
        updated = True
    print(f"wrote {a.out} and {a.out.replace('.md', '.json')}" + (f"; updated {a.notes}" if updated else ""))


if __name__ == "__main__":
    main()
