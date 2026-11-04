#!/usr/bin/env python3
"""Generate the article's and the supplement's result tables (LaTeX, booktabs) from the v2 result files.

No number in any table is typed by hand: every value, count and caption number is read from the v2
JSON/CSV outputs below, and RESULTS.json is first re-derived from its own sources with the collector's
functions (scripts/collect_results_v2.py) so a stale RESULTS.json stops the run.

Writes <tables>/{ladder,contact_dr,learnability,realism,identity,registration,dr,windows,compute}.tex (the paper's
tables folder; DIR/tables in public mode),
each a complete float with the label that the article's sources reference, and prints every table's numbers
to stdout so the run log documents them.

POST HOC tables (post hoc analyses of the v2.1 plan; outside the v2 plan):
{ladder_rotation_success,imitation_base,ladder_organ,stats_sensitivity,rule_simulation,rl_tost,registration_crossed,
step_error,seg_sim_real}.tex, from outputs/v2/posthoc/posthoc_v2.json (scripts/posthoc_v2.py). They are written only when
RESULTS.json["posthoc"] certifies that file (same sha256) and its inputs are unchanged; otherwise they are skipped
with a log line. They never change the tables above.

EXPLORATORY v2.1 tables (amendment 1; amendment 2 adds SAMUS, LPIPS, the CT-anatomy slot and the torchvision
segmenters pretrained on ImageNet):
pluggability.tex (tab:pluggability, Table 5 of the article; each slot header gives the condition, the reference error
and the post hoc MDD; columns Error, Difference [95% t CI] with k/6, seeds sharing the sign k/3 (post hoc), Holm p, a
post hoc TOST at +/- 1 mm and a Decision column in the plans' outcome words) with figures/plug_rows.json (Fig. 4 data),
patient_seed.tex (tab:patient-seed, Appendix C.8: per-seed differences, the planned patient-by-seed bootstrap and a
crossed bootstrap for every three-seed row of Table 5), leaderboard.tex (tab:leaderboard, Appendix C: the Table 5 rows
of each slot, the reference first and the rest by mean error, printing Table 5's own cells) and pluggability_twin.tex
(tab:pluggability-twin, the supplementary twin in two floats with the Label, Cost, t p, sign-flip p, 90% t CI and Score
columns), from outputs/v2_1/plug/**/ and the segmenter repository's outputs/v2_1/plug/ (--plug-root, --seg-plug-root),
the admission gates in outputs/v2_1/checks/gate/ and the archived v2 files behind RESULTS.json; paired t CI (df 5), Holm
over each slot's full pre-declared family, "secondary" rows, the amendment 2 families (MASK-P, IMN-T, LPIPS-R, CTA-D,
CTA-E; never pooled with amendment 1's; their Appendix constants are loaded through plan_inputs.py, sha256-checked and
asserted before any of their outputs is read), and a provisional section-sign mark on every v2.1 cell. checks.tex
(tab:checks), one row per testbed check and kept v2.1 run, from outputs/v2_1/checks/ and
outputs/v2_1/jobs_final_r3.json. checks_summary.tex (tab:checks-summary, the body table tab:checks-summary): each
check's guard, rule and outcome with absolute values first, from the same outputs, the kept v2.1 analysis files and the
Table 5 record. They are written last and never change the tables above; --only-pluggability and --only-checks write
pluggability or checks.tex alone (tab:checks-summary needs the full run; so do the tables of amendments 3 to 5, written
by amend345_tables()).

Without --read-amend2, the four groups of amendment 2 (LPIPS, SAMUS, the CT-anatomy slot and the segmenters that
start from ImageNet weights) stay unread: no job output or log of theirs is opened, and each group prints one pending
line.

Sources
  outputs/v2/RESULTS.json                       table4, dr_policy, registration, table2, table3, table5
  outputs/v2/table4/episodes_long.csv           episode design (patients, targets, windows, seeds)
  outputs/v2/table4/registration_long.csv       target and repeat counts per registration cell
  outputs/v2/view_match/table3_stats.json       the landscape table per-patient Fisher-z differences
  outputs/v2/view_match/cue_ablation/*.json     renderer-term ablation (via RESULTS.json cue_ablation)
  outputs/v2/view_match/landscape_heldout.json  views per patient and samples per view
  outputs/v2/realism/realism_sector.json        distances to real frames in matched sector geometry
  <segmenter repository>/outputs/v2/tablev_v2_{vitale,vitaleho}.json and consistency_*.json
  outputs/v2/dr/dr_table.json                   DR knobs and measured effects (same data as tab_dr.tex)
  data/cache/windows/usbg-windows-v1.json       screened acoustic windows per case and organ
  outputs/v2/table4/{ctmask,matched,dr_on}/*.json  trainer configs (min_windows, skipped cells)
  usbg.benchmark.TRAIN_PATIENTS / TEST_PATIENTS the patient split
  outputs/v2/RESULTS.json compute               CUDA-free cost (re-derived from outputs/v2/bench_render.json,
                                                bench_latency.json, bench_closed_loop.json, table4/runner.log)

The tables print scientific information and
results only. No table prints plan, amendment, archive, tag, gate, CI, device-parity or deviation-row words, and no
section-sign provisional mark is printed (PROV is empty). PLUG_PROVISIONAL is cleared: the
tag aggregation reproduced all 67 Table 5 rows; plug_rows.json keeps the flags it was written with (PLUG_ROWS_PROVISIONAL).
The rows that are not printed (gates, audits, reruns, CPU against MPS, check records) are still computed, asserted
and logged as not printed. The rule-input table is written to a folder of the
development repository (--process-out); a development test lints every table this script writes.

Usage (development repository): PYTHONPATH=src python <this folder>/make_tables.py [--out <tables folder>]

Public mode (the mirror of the code; the result files of the public deposit under ROOT, laid out as ROOT/outputs/...,
ROOT/data/... and ROOT/segmenter/outputs/...):
  python <this folder>/make_tables.py --public --results ROOT --out DIR
reads the pre-specified constants from public_inputs/plan_constants.json (plan_inputs.py) and nothing outside the code
tree and ROOT, checks every result file it opens or probes against ROOT/MANIFEST.tsv (results_world.py), implies every
--read-amend<n> lift, writes the tables to DIR/tables with every LaTeX comment removed (tex_comments.py) and
figures/plug_rows.json to DIR/figures. The checks that need the development repository (its plan documents, the
manuscript's own text, the absolute paths records cite, the outcome sentences of the v2 plan) are skipped with one
notice each on stderr. The digests this code pins of result files (LATE_PINNED, the records ledger lines) are replaced
by their rows of ROOT/MANIFEST.tsv; the digests that one deposited record holds of another are checked as in the
development build, and a path that a record gives relative to the repository roots (outputs/..., data/...,
segmenter/...) is resolved against ROOT.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys

import numpy as np

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
sys.path.insert(0, os.path.join(CODE_ROOT, "src"))
import script_dirs as SD  # noqa: E402  (code tree, result root, public mode)
if __name__ == "__main__":
    import results_world  # noqa: E402
    results_world.start(sys.argv[1:])     # --public --results ROOT, before any data path below is computed
import plan_inputs as PI  # noqa: E402  (pre-specified constants and record digests: public or development backend)
_T = PI.text   # source notes of the table comments (the development build prints its own; see plan_inputs.PUBLIC_TEXT)
import collect_results_v2 as C  # noqa: E402  (read-only helpers: rule, per_patient, section collectors)
from usbg.benchmark import TEST_PATIENTS, TRAIN_PATIENTS  # noqa: E402
try:   # the testbed's private-path guard; without it (a release of the package that lacks it) public mode relies on
    # the closed world of the result root alone
    from usbg.provenance import PRIVATE_DIRNAMES, is_private_path  # noqa: E402
except ImportError:
    if not SD.public():
        raise
    PRIVATE_DIRNAMES, is_private_path = (), (lambda p: False)

PP = SD.private_module()   # the development backend (plan_private.py): None in public mode and in the public mirror
PAPER = PP.paper_tree(CODE_ROOT) if PP else None    # the paper's sources and tables (development repository only)
ROOT = SD.results_root()     # the result files (outputs/, data/): the code tree, or ROOT of --results in public mode
V2 = os.path.join(ROOT, "outputs", "v2")
SEG = SD.segmenter_root()
STAGES = ("physics", "cut", "cyclegan", "sonospade")
NAME = {"physics": "Physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
GEN = "% generated by " + _T("make_tables script") + " from {src}\n% do not edit by hand: rerun the script\n"
SUMMARY: list[str] = []


def log(s=""):
    SUMMARY.append(s)
    print(s)


def rel(p):
    if SD.public() or PP is None:
        return SD.display(p)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT) else PP.tilde(p)


def _strip_tex(text):
    """The printed text of a table: every LaTeX comment removed, the tokens TeX reads unchanged (tex_comments)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import tex_comments
    return tex_comments.strip_comments(text)[0]


# =================================================================================== formatting
def sg(x, d=2):
    """Signed number in math mode, rounded as RESULTS.md does; a value that rounds to zero prints unsigned."""
    s = f"{x:+.{d}f}"
    return "$" + (f"{0:.{d}f}" if float(s) == 0 else s) + "$"


def ci_txt(ci, d=2):
    return f"$[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]$"


def ks(e):
    return f"{e['sign']}/{e['n']}"


def star(e):
    return "$^{\\ast}$" if e["holds"] else ""


def two_line(e, d=2):
    """Gap cell: mean (asterisk if a difference under the rule), k/6 on line one, CI on line two."""
    return (f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{sg(e['mean'], d)}{star(e)}, {ks(e)}\\\\"
            f"{ci_txt(e['ci'], d)}\\end{{tabular}}")


def plain(e, d=2):
    return f"{e['mean']:+.{d}f} [{e['ci'][0]:+.{d}f}, {e['ci'][1]:+.{d}f}], {e['sign']}/{e['n']}" + (
        " *" if e["holds"] else "")


def write(out_dir, name, text, verbatim=()):
    if "\u2014" in text or "-" * 3 in text:
        raise SystemExit(f"{name}: em-dash or triple hyphen in generated LaTeX")
    scan = text
    for s in verbatim:          # exact plan sentences that print a banned word to deny it (A6_VERBATIM), scan only
        scan = scan.replace(s, "")
    for bad in ("deploy", "realistic", "significant", "differentiable", "state-of-the-art", "novel", "confirmed",
                "comparable", "observability gap", "perception gap", "superset", "sim-to-real", "transfer"):
        if bad in scan.lower():
            raise SystemExit(f"{name}: banned word {bad!r} in generated LaTeX")
    import re
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())
    if re.search(r"\bopen\b", body, flags=re.I):     # the printed text does not use "open" (style rules)
        raise SystemExit(f"{name}: the word 'open' outside a comment; the printed text does not use it")
    if name in BIB_HEADER and not text.startswith("%% NEW BIB KEYS"):   # the floats of the article body
        import re
        keys = sorted({k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", text) for k in m.split(",")})
        if keys:
            raise SystemExit(f"{name}: cites {keys}; write its NEW BIB KEYS line where the keys are checked")
        text = "%% NEW BIB KEYS: none\n" + text
    p = _out_ok(os.path.join(out_dir, name))
    if SD.public():          # the public build writes the printed text only (every check above ran on the full text)
        text = _strip_tex(text)
    with open(p, "w", encoding="utf-8", newline="\n") as f:       # the same bytes on every platform
        f.write(text)
    return p


BIB_HEADER = ("compute.tex", "ladder.tex", "checks_summary.tex")


# =================================================================================== source checks
def _close(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise SystemExit(f"RESULTS.json is stale at {path}: keys {sorted(set(a) ^ set(b))}")
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise SystemExit(f"RESULTS.json is stale at {path}: length")
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))):
            raise SystemExit(f"RESULTS.json is stale at {path}: {a} vs recomputed {b}")
    elif (isinstance(a, str) and isinstance(b, str) and os.path.isabs(a) and os.path.isabs(b)
          and os.path.realpath(SD.map_recorded(a)) == os.path.realpath(SD.map_recorded(b))):
        pass   # one file under two absolute paths (a second checkout whose outputs/ links to the main checkout's)
    elif (SD.public() and isinstance(a, str) and isinstance(b, str) and not os.path.isabs(a) and os.path.isabs(b)
          and SD.map_recorded(a) != a and os.path.realpath(SD.map_recorded(a)) == os.path.realpath(b)):
        pass   # public mode: the results deposit records the path relative to the repository roots
    elif a != b:
        raise SystemExit(f"RESULTS.json is stale at {path}: {a!r} vs recomputed {b!r}")


def load_results():
    R = json.load(open(os.path.join(V2, "RESULTS.json")))
    fresh = {}
    for fn in (C.table3, C.cue, C.table2, C.table4, C.dr_policy, C.registration):
        fn(fresh)
    fresh = json.loads(json.dumps(fresh, default=str))
    rec = R
    if PI.skip_dev(C.SENTENCE_SKIP):     # public mode: the collector fills no plan sentence (_plan_sentence)
        rec = json.loads(json.dumps(R))
        for k, f in (("table3", "sentence"), ("table2", "sentence"), ("table4", "sentences")):
            rec[k].pop(f, None)
            fresh[k].pop(f, None)
    for k in ("table3", "cue_ablation", "table2", "table4", "dr_policy", "registration"):
        _close(rec[k], fresh[k], k)
    log("[check] RESULTS.json table3, cue_ablation, table2, table4, dr_policy and registration "
        "re-derived from their sources: identical")
    return R


def episode_design():
    rows = C.load_eps()
    pol = [r for r in rows if r["group"] in ("ctmask", "matched")]
    seeds = sorted({r["seed"] for r in pol})
    per_run = collections.defaultdict(set)
    for r in rows:   # every run of every group (ladder, policy arms, contact, DR) must share one episode set
        per_run[tuple(r[k] for k in ("group", "arm", "rung", "stage", "mask", "seed", "phase", "contact",
                                     "eval_randomize"))].add((r["patient"], r["target"], r["window"]))
    sets = list(per_run.values())
    if any(s != sets[0] for s in sets):
        raise SystemExit("episode sets differ across arms: the crossed design is broken")
    eps = sets[0]
    pats = sorted({p for p, _, _ in eps})
    targets = sorted({t for _, t, _ in eps})
    per_cell = collections.Counter((p, t) for p, t, _ in eps)
    if len(set(per_cell.values())) != 1:
        raise SystemExit("windows per (patient, target) cell are not constant")
    ref = [r for r in rows if r["group"] == "ladder" and r["rung"] in ("static", "scripted", "pose_oracle")]
    for rung in ("static", "scripted", "pose_oracle"):
        vals = collections.defaultdict(list)
        for r in ref:
            if r["rung"] == rung and r["contact"] == "on":
                vals[r["stage"]].append(float(r["final_pos"]))
        arr = [np.asarray(v) for _, v in sorted(vals.items())]
        if any(a.shape != arr[0].shape or np.abs(a - arr[0]).max() > 0 for a in arr):
            raise SystemExit(f"reference rung {rung} differs across stages; one spanning value is not valid")
    return {"n_eps": len(eps), "patients": pats, "targets": targets, "per_cell": next(iter(per_cell.values())),
            "seeds": seeds}


def _rerun_identity():
    """The v2.1 rerun from the clean tag: every per-episode block of the archived ladder, contact
    and DR evaluations (outputs/v2/table4/{ladder,contact,dr_eval}) against its rerun (outputs/v2_1/s21_rerun/, same
    file name). An episode is bit-identical when every field the two records share, timing excluded, is equal."""
    n = same = blocks = files = 0
    max_pp = 0.0
    for d in ("ladder", "contact", "dr_eval"):
        for f in sorted(glob.glob(os.path.join(V2, "table4", d, "*.json"))):
            g = os.path.join(ROOT, "outputs", "v2_1", "s21_rerun", d, os.path.basename(f))
            a, b = json.load(open(f)), json.load(open(g))
            if _status_of(b) != "exploratory, v2.1":
                raise SystemExit(f"{rel(g)}: analysis_status {_status_of(b)!r}")
            files += 1
            if len(a["results"]) != len(b["results"]):
                raise SystemExit(f"{rel(g)}: block count differs from {rel(f)}")
            for x, y in zip(a["results"], b["results"]):
                if any(x.get(k) != y.get(k) for k in ("weights", "contact", "eval_randomize")):
                    raise SystemExit(f"{rel(g)}: block order differs from {rel(f)}")
                blocks += 1
                pa, pb = collections.defaultdict(list), collections.defaultdict(list)
                for e, h in zip(x["episodes"], y["episodes"]):
                    n += 1
                    ks = [k for k in e if k in h and "sec" not in k and "time" not in k]
                    same += all(e[k] == h[k] for k in ks)
                    pa[e["patient"]].append(float(e["final_pos"]))
                    pb[h["patient"]].append(float(h["final_pos"]))
                for p_ in pa:
                    max_pp = max(max_pp, abs(float(np.mean(pa[p_])) - float(np.mean(pb[p_]))))
    log(f"[rerun] clean-tag rerun: {same} of {n} episodes bit-identical in {blocks} blocks of {files} files; largest "
        f"per-patient |difference| {max_pp:.3g} mm")
    return {"n": n, "same": same, "blocks": blocks, "files": files, "max_pp": max_pp,
            "src": f"outputs/v2_1/s21_rerun/{{ladder,contact,dr_eval}}/*.json vs outputs/v2/table4/{{ladder,contact,"
                   f"dr_eval}}/*.json results[*].episodes[*] (all shared fields, timing excluded): {same} of {n} "
                   f"episodes identical, {blocks} blocks, {files} files, largest per-patient |diff| {max_pp:.3g} mm"}


def _ladder_posthoc_block(J):
    """Table 4 post hoc block, read from the posthoc_v2.json rows (archived data, post hoc, certified by RESULTS.json;
    printed plain): for the five rows static, scripted, pose oracle, CT-mask final and learned-mask final, the final
    rotation error, the success rate at each tolerance and the final match score S. Each metric has one header line,
    shared by the three reference rows (they have no learned weights)."""
    R_ = J["rows"]
    L = ["\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{Post hoc block}} \\\\"]
    mets = [("Final rotation error (deg)", lambda m: f"{m['final_ang']['mean']:.2f}", "final_ang"),
            ("Success, 5~mm and 5$^{\\circ}$ (\\%)", lambda m: f"{100 * m['succ_5mm_5deg']['mean']:.1f}",
             "succ_5mm_5deg"),
            ("Success, 10~mm and 10$^{\\circ}$ (\\%)", lambda m: f"{100 * m['succ_10mm_10deg']['mean']:.1f}",
             "succ_10mm_10deg"),
            ("Final match score $S$", lambda m: f"{m['final_match']['mean']:.3f}", "final_match")]
    for head, fmt, key in mets:
        refs = ", ".join(f"{lab} {fmt(R_[k]['metrics'])}" for k, lab in
                         (("static", "static"), ("scripted", "scripted"), ("pose_oracle", "pose oracle$^{\\dagger}$")))
        L.append(f"{head} & \\multicolumn{{4}}{{l}}{{{refs}}} \\\\ % src: posthoc_v2.json "
                 f"rows.{{static,scripted,pose_oracle}}.metrics.{key}.mean")
        for arm, lab in (("ctmask", "CT mask$^{\\dagger}$, final"), ("matched", "Learned mask, final")):
            L.append(f"\\quad {lab} & " + " & ".join(fmt(R_[f"{arm}/{s}/final"]["metrics"]) for s in STAGES)
                     + f" \\\\ % src: posthoc_v2.json rows.{arm}/<stage>/final.metrics.{key}.mean")
    return L


# =================================================================================== tab:ladder
def ladder(R, D, out):
    t4 = R["table4"]
    cm, rg = t4["cell_means_mm"], t4["rungs_mm"]
    nb = C.NBOOT
    J, why = load_posthoc(R)
    if J is None:
        raise SystemExit(f"Table 4 needs its post hoc block: {why}")
    rr = _rerun_identity()   # kept as a log() check only; its sentence is not printed
    # Caption: the note on the 0.5 mm bound and the estimand sentence; the clean-tag rerun is logged, not printed.
    # One status statement first: the pre-specified cells are primary; the last block (post hoc) is labeled where it
    # appears.
    cap = ("Oracle ladder: final translation error (mm), " + str(len(D["patients"])) + " held-out patients, contact "
           f"on, evaluation domain randomization off; policy rows average {len(D['seeds'])} seeds, each meeting the "
           f"same {D['n_eps']} crossed episodes. "
           "Pre-specified results, primary; the last block is post hoc. "
           "Errors: mean of per-patient means. "
           "Gaps: mean per-patient paired difference, patients of " + str(len(D["patients"]))
           + f" sharing its sign, and 95\\% patient-cluster bootstrap CI ({nb:,} resamples); "
           "$^{\\ast}$a difference under the pre-specified CI-and-sign rule. "
           # the estimand sentence, which every caption carries
           "Every contrast is conditional on the trained artifacts (texture networks, segmenters, bases and policies). "
           "Seeds index artifacts; the patient is the sampling unit. "
           "The servo base accounts for nearly all of the improvement; imitation and REINFORCE each change the "
           "final translation error by less than 0.5~mm. "
           # the L marker of SonoSPADE, with the footnote in the words of Table 5
           "$^{\\mathrm{L}}$Reads the CT label slice. "
           "$^{\\dagger}$Privileged: simulator state that no rig provides. "
           # SonoSPADE, the standard engine's texture stage, is the main benchmark column
           "Shaded: SonoSPADE, the texture stage of the standard engine and the main benchmark; physics, CUT and "
           "CycleGAN are the ladder's other texture settings.")
    cap_src = (f"% src: patients, seeds, episodes: outputs/v2/table4/episodes_long.csv via episode_design() "
               f"({len(D['patients'])} patients, seeds {','.join(D['seeds'])}, {D['n_eps']} episodes per seed); "
               f"resamples collect_results_v2.NBOOT ({nb}); {_T('ladder src notes')}check still runs: {rr['src']}")
    if STAGES[-1] != "sonospade":
        raise SystemExit("tab:ladder: the shaded last column must be SonoSPADE, the standard engine's stage")
    L = [GEN.format(src="outputs/v2/RESULTS.json (table4: cell_means_mm, rungs_mm, observability_gap, "
                        "perception_gap, rl_contribution, learned_vs), outputs/v2/table4/episodes_long.csv, the post "
                        "hoc block from outputs/v2/posthoc/posthoc_v2.json (certified by RESULTS.json posthoc); the "
                        "rerun check reads outputs/v2_1/s21_rerun/ against outputs/v2/table4/ and prints nothing"),
         "\\begin{table*}[tp]",
         "\\caption{" + cap + "} " + cap_src,
         "\\label{tab:ladder}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",  # footnotesize (length)
         # the SonoSPADE column (the last stage) shaded as the standard engine's (colortbl)
         "\\begin{tabular}{@{}l ccc>{\\columncolor{black!8}}c@{}}",
         "\\toprule",
         # SonoSPADE, the only label-conditioned stage, carries the L marker of Tables 5 and A.9
         " & " + " & ".join(NAME[s] + ("$^{\\mathrm{L}}$" if s == "sonospade" else "") for s in STAGES) + " \\\\",
         "\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{Final translation error (mm)}} \\\\"]
    for key, lab in (("static", "Static start (no commanded motion)"), ("scripted", "Scripted sweep (no feedback)"),
                     ("pose_oracle", "Analytic pose oracle$^{\\dagger}$")):
        L.append(f"{lab} & \\multicolumn{{4}}{{c}}{{{rg[key]:.2f}}} \\\\ % src: RESULTS.json table4.rungs_mm.{key}")
    for arm, head in (("ctmask", "CT mask$^{\\dagger}$"), ("matched", "Learned mask (texture-matched segmenter)")):
        L.append("\\addlinespace")
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{head}}} \\\\")
        for ph, lab in (("base", "Servo base"), ("after_imitation", "After imitation"), ("final", "Final")):
            L.append(f"\\quad {lab} & " + " & ".join(f"{cm[f'{arm}/{s}/{ph}']:.2f}" for s in STAGES)
                     + f" \\\\ % src: RESULTS.json table4.cell_means_mm.{arm}/<stage>/{ph}")
    L += ["\\midrule",
          "\\multicolumn{5}{@{}l}{\\textit{Gaps (mm): patient-level mean, k/6 patients, 95\\% CI}} \\\\ % src: "
          "pre-specified plan (95% patient-cluster bootstrap CI, 6 held-out patients)"]
    gap_rows = [("Oracle-to-CT-mask gap", "observability_gap", lambda s: s),
                ("CT-to-learned-mask gap", "perception_gap", lambda s: s),
                ("RL contribution, CT mask", "rl_contribution", lambda s: f"ctmask/{s}"),
                ("RL contribution, learned mask", "rl_contribution", lambda s: f"matched/{s}")]
    for lab, key, kf in gap_rows:
        L.append(f"{lab} & " + " & ".join(two_line(t4[key][kf(s)]) for s in STAGES)
                 + f" \\\\[2pt] % src: RESULTS.json table4.{key}")
    lv = t4["learned_vs"]
    L.append("SonoSPADE minus stage, learned-mask final & " + " & ".join(two_line(lv[s]) for s in STAGES[:3])
             + " & n/a \\\\ % src: RESULTS.json table4.learned_vs")
    L += _ladder_posthoc_block(J)
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    p = write(out, "ladder.tex", "\n".join(L))

    log("\n== tab:ladder (ladder.tex) ==")
    log(f"episodes per seed {D['n_eps']} = {len(D['patients'])} patients x {len(D['targets'])} targets "
        f"x {D['per_cell']} windows; seeds {','.join(D['seeds'])}; bootstrap {nb}")
    log(f"static {rg['static']:.2f} | scripted {rg['scripted']:.2f} | pose oracle {rg['pose_oracle']:.2f}")
    for arm in ("ctmask", "matched"):
        for ph in ("base", "after_imitation", "final"):
            log(f"{arm:7s} {ph:15s} " + "  ".join(f"{s} {cm[f'{arm}/{s}/{ph}']:.2f}" for s in STAGES))
    for lab, key, kf in gap_rows:
        log(f"{lab}: " + "; ".join(f"{s} {plain(t4[key][kf(s)])}" for s in STAGES))
    log("SonoSPADE minus stage (learned final): " + "; ".join(f"{s} {plain(lv[s])}" for s in STAGES[:3]))
    return p


# =================================================================================== tab:contact-dr
def contact_dr(R, D, out):
    t4 = R["table4"]
    co, dp = t4["contact_off_minus_on"], R["dr_policy"]
    rows = C.load_eps()
    ref_pp = C.per_patient(rows, group="dr_eval", arm="ctmask", phase="final", eval_randomize="off")[0]
    ref_mm = float(np.mean(list(ref_pp.values())))
    missing = [k for k in ("static", "scripted", "pose_oracle") if k not in co]
    if missing:
        raise SystemExit(f"contact_off_minus_on lacks the reference rungs {missing}: rerun the collector")

    def zero(e):
        return abs(e["mean"]) < 1e-9 and abs(e["ci"][0]) < 1e-9 and abs(e["ci"][1]) < 1e-9

    # a starred contact row that also passes Holm on the paired t-test over the 16 contact rows (post hoc) carries
    # the superscript H, read from the Table S2.18 record (posthoc_v2_layers.json families.Q7_contact.items[*].holm_t,
    # m 16)
    q7 = _late(LATE_LAYERS)["families"]["Q7_contact"]
    if q7["m"] != 16 or len(q7["items"]) != 16:
        raise SystemExit(f"{LATE_LAYERS}: Q7_contact has m {q7['m']} and {len(q7['items'])} items, the caption says 16")
    q7_pass = {k for k, v in q7["items"].items() if v["holds_rule"] and v["holm_t"] < 0.05}
    q7_star = {k for k, v in q7["items"].items() if v["holds_rule"]}
    if len(q7_pass) != q7["n_holm_t_survive"] or len(q7_star) != q7["n_rule_holds"]:
        raise SystemExit(f"{LATE_LAYERS}: Q7_contact survivor counts disagree with the items")

    def q7_key(key):
        return "ibvs_base/ct" if key.startswith("ibvs_base/ct/") else key

    def cells(e, key=None):
        if zero(e):   # the rung never moves: off and on agree to numerical precision
            return f"$0.00$ & $[0.00, 0.00]$ & n/a"
        if key is not None and (q7_key(key) in q7_star) != bool(star(e)):
            raise SystemExit(f"contact_dr {key}: the star of RESULTS.json and the rule of {LATE_LAYERS} disagree")
        hm = "$^{\\mathrm{H}}$" if key is not None and q7_key(key) in q7_pass else ""
        return f"{sg(e['mean'])}{star(e)}{hm} & {ci_txt(e['ci'])} & {ks(e)}"

    # (row label, [(stage label, RESULTS key)]); a group whose four stages give one identical value prints once
    groups = [("Static start", [("all", "static")]),
              ("Scripted sweep", [("all", "scripted")]),
              ("Pose oracle (privileged)", [("all", "pose_oracle")])]
    for lab, keyf in (("Servo base, CT mask", lambda s: f"ibvs_base/ct/{s}"),
                      ("Servo base, learned", lambda s: f"ibvs_base/learned/{s}"),
                      ("Final servo, CT mask", lambda s: f"ctmask/{s}"),
                      ("Final servo, learned", lambda s: f"matched/{s}")):
        es = [co[keyf(s)] for s in STAGES]
        if all(e == es[0] for e in es):
            groups.append((lab, [("all", keyf(STAGES[0]))]))
        else:
            groups.append((lab, [(NAME[s], keyf(s)) for s in STAGES]))
    one = [lab for lab, items in groups[3:] if len(items) == 1]
    L = [GEN.format(src="outputs/v2/RESULTS.json (table4.contact_off_minus_on, dr_policy) and "
                        "outputs/v2/table4/episodes_long.csv (DR reference arm)"),
         "\\begin{table}[tp]",
         "\\caption{Contact and domain-randomization (DR) contrasts on the final pose error (mm), "
         f"{len(D['patients'])} held-out patients, {len(D['seeds'])} seeds, the crossed episodes of "
         "Table~\\ref{tab:ladder}. Contact rows: contact switched off at evaluation only, minus contact on, "
         "for every rung. The static start never moves, so its "
         "delta is 0 to numerical precision. The reference rungs"
         + (" and the " + " and ".join(x.lower().replace("servo base, ct mask", "CT-mask servo base")
                                        for x in one) if one else "")
         + " do not depend on the texture stage and give one value (all). DR rows: the SonoSPADE CT-mask arm "
         "with DR in evaluation, in training, or in both, minus the same arm without DR "
         f"({ref_mm:.2f}~mm); these rows are descriptive. Each row gives the mean of per-patient paired "
         "differences, a 95\\% patient-cluster bootstrap CI and the number of patients that share the sign of "
         "the mean. $^{\\ast}$: a difference under the CI-and-sign rule. "
         # the Holm marker; src: posthoc_v2_layers.json families.Q7_contact (m, n_rule_holds, n_holm_t_survive)
         f"$^{{\\mathrm{{H}}}}$: also passes Holm on the paired $t$-test over the {q7['m']} contact rows (post hoc; "
         f"Supplementary Table~S2.18); {len(q7_star) - len(q7_pass)} of the {len(q7_star)} starred rows do not.}}"
         f" % src: {LATE_LAYERS} families.Q7_contact m {q7['m']}, n_rule_holds {q7['n_rule_holds']}, n_holm_t_survive "
         f"{q7['n_holm_t_survive']}, items[*].holm_t < 0.05{_T('contact src note')}",
         "\\label{tab:contact-dr}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}llrcc@{}}",
         "\\toprule",
         "Rung & Stage & Mean & 95\\% CI & $k$/6 \\\\",
         "\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{Contact off minus on}} \\\\"]
    for gi, (lab, items) in enumerate(groups):
        if gi in (3, 5):
            L.append("\\addlinespace")
        for j, (stl, key) in enumerate(items):
            L.append(f"{lab if j == 0 else ''} & {stl} & {cells(co[key], key)} \\\\ "
                     f"% src: RESULTS.json table4.contact_off_minus_on.{key}"
                     + (f"; H: {LATE_LAYERS} families.Q7_contact.items[{q7_key(key)}].holm_t "
                        f"{q7['items'][q7_key(key)]['holm_t']:.4g}" if q7_key(key) in q7["items"] else ""))
    L += ["\\addlinespace",
          "\\multicolumn{5}{@{}l}{\\textit{DR minus no DR, SonoSPADE, CT mask (descriptive)}} \\\\"]
    for key, lab in (("eval DR only", "Evaluation only"), ("training DR only", "Training only"),
                     ("training and eval DR", "Training and evaluation")):
        e = dp[key]
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{DR in {lab.lower()}}} & {sg(e['mean'])}{star(e)} & "
                 f"{ci_txt(e['ci'])} & {ks(e)} \\\\ % src: RESULTS.json dr_policy.{key}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p = write(out, "contact_dr.tex", "\n".join(L))

    log("\n== tab:contact-dr (contact_dr.tex) ==")
    for k, e in co.items():
        log(f"contact off-on {k}: {plain(e)}")
    log(f"DR reference (SonoSPADE CT-mask, no DR, final): {ref_mm:.2f} mm")
    for k, e in dp.items():
        log(f"DR {k}: {plain(e)}")
    return p


# =================================================================================== tab:learnability
def learnability(R, out):
    f = os.path.join(V2, "view_match", "table3_stats.json")
    t = json.load(open(f))
    lh = json.load(open(os.path.join(V2, "view_match", "landscape_heldout.json")))
    cue = R["cue_ablation"]
    scopes = [("Pooled", t["pooled_all_organs"]), ("Liver", t["by_organ"]["liver"]),
              ("Kidney", t["by_organ"]["kidney"])]
    L = [GEN.format(src="outputs/v2/view_match/table3_stats.json (per-patient Fisher-z differences, rule via "
                        "scripts/collect_results_v2.rule), outputs/v2/RESULTS.json (table3, cue_ablation), "
                        "outputs/v2/view_match/landscape_heldout.json"),
         "\\begin{table*}[tp]",
         "\\caption{Appearance-term learnability on the " + str(len(lh["cases"])) + " held-out patients. "
         f"Each target view ({lh['targets_per_case']} per patient and organ, {lh['n_target_views']} in all) is "
         f"compared with {lh['samples_per_target']} nearby probe poses; the per-view score is the Spearman "
         "correlation between the sector-interior appearance match and closeness to the target pose. "
         "(a) SonoSPADE against each other stage: mean per-view Spearman of both stages, the mean over patients "
         "of the per-patient difference in mean Fisher-$z$ ($\\Delta z$), its 95\\% patient-cluster bootstrap "
         "CI, and the number of patients that share its sign. Rule: a difference under the CI-and-sign rule. "
         "The pooled rows are the pre-specified primary contrast. (b) Exploratory renderer-term ablation, "
         "not pre-specified: mean per-view Spearman with one renderer term switched off. Interface "
         "reflection off: no transmission loss or specular echo at tissue interfaces, attenuation kept. "
         "Diffuse scattering off: scattering strength 0. Depth variation of the PSF off: a depth-invariant "
         "lateral width. Flat means the appearance term is constant over poses, so the correlation is "
         "undefined. DR is off throughout.}",
         "\\label{tab:learnability}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\textit{(a) SonoSPADE against each other texture stage}\\\\[2pt]",
         "\\begin{tabular}{@{}lrlcccc@{}}",
         "\\toprule",
         "Scope & Views & Other stage & Spearman, SonoSPADE / other & $\\Delta z$ & 95\\% CI & $k$/6, rule \\\\",
         "\\midrule"]
    log("\n== tab:learnability (learnability.tex) ==")
    verdict = {}
    for i, (scope, blk) in enumerate(scopes):
        if i:
            L.append("\\addlinespace")
        held = []
        for other in ("physics", "cut", "cyclegan"):
            c = blk["contrasts"][other]
            ok, dz, ci, agree, n = C.rule(list(c["patient_level"]["per_patient_dz"].values()))
            st = c["patient_level"]["sign_consistency"]
            if st != f"{agree}/{n}" or abs(dz - c["cluster_boot_dz"]["mean"]) > 1e-9:
                raise SystemExit(f"table3_stats sign or mean disagrees with the rule for {scope}/{other}")
            held.append(ok)
            L.append(f"{scope if other == 'physics' else ''} & {c['n_views'] if other == 'physics' else ''} & "
                     f"{NAME[other]} & ${c['mean_ref']:.3f}$ / ${c['mean_other']:.3f}$ & {sg(dz, 3)} & "
                     f"{ci_txt(ci, 3)} & {agree}/{n}, {'yes' if ok else 'no'} \\\\ "
                     f"% src: table3_stats.json {'pooled_all_organs' if scope == 'Pooled' else 'by_organ.' + scope.lower()}"
                     f".contrasts.{other}")
            log(f"{scope:7s} vs {other:9s} views {c['n_views']}: Spearman S {c['mean_ref']:.3f} / other "
                f"{c['mean_other']:.3f}; dz {dz:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], {agree}/{n}, holds {ok}")
        verdict[scope.lower()] = held
    if dict(zip(("physics", "cut", "cyclegan"), verdict["pooled"])) != R["table3"]["verdict_pooled"]:
        raise SystemExit("pooled Table III verdict differs from RESULTS.json table3.verdict_pooled")
    L += ["\\bottomrule", "\\end{tabular}", "",
          "\\vspace{6pt}",
          "\\textit{(b) Renderer-term ablation, mean per-view Spearman}\\\\[2pt]",
          "\\begin{tabular}{@{}lcccc@{}}",
          "\\toprule",
          "Term switched off & " + " & ".join(NAME[s] for s in STAGES) + " \\\\",
          "\\midrule"]
    # what each switch removes (scripts/view_match_landscape.py CUE_ABLATIONS, src/usbg/render_lotus.py
    # render_bmode_physics): shadow_gain=0 zeroes the interface reflection Rc (no transmission loss T_cum,
    # no specular echo; attenuation A_cum kept); speckle_strength=0 zeroes the whole diffuse scattering echo;
    # focal_scale=1e6 makes the lateral PSF width depth-invariant.
    cue_lab = {"none": "None (full renderer)", "shadow": "Interface reflection", "speckle": "Diffuse scattering",
               "psf": "Depth variation of the PSF", "all": "All three"}
    for k in ("none", "shadow", "speckle", "psf", "all"):
        vals = cue[k]
        cells = ["flat" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
                 for v in (vals[s] for s in STAGES)]
        L.append(f"{cue_lab[k]} & " + " & ".join(cells) + f" \\\\ % src: RESULTS.json cue_ablation.{k}")
        log(f"cue {k:8s}: " + "  ".join(f"{s} {c}" for s, c in zip(STAGES, cells)))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    log(f"plan sentence (Q1): {R['table3']['sentence']}")
    return write(out, "learnability.tex", "\n".join(L))


# =================================================================================== tab:realism
REALISM_V21 = [  # (file under outputs/v2_1/plug/renderer/realism/, variant, row label): v2.1, exploratory
    ("torch.json", "lotus_orig", "Original LOTUS code, default parameters"),
    ("simus.json", "simus", "SIMUS via PyMUST"),
]
REALISM_PARITY = ("w1_in_sector", "kid", "kid_std", "fid_secondary", "c2st_auc_mean", "c2st_auc_std")
LOTUS_DEFAULTS_SRC = PI.text("lotus defaults src")


def _kid_design(argv):
    """-> KID subset size from a realism run's argv (--kid-subset) and the subset count from eval_fid_kid._kid."""
    import re
    k = argv.index("--kid-subset") if "--kid-subset" in argv else None
    sub = int(argv[k + 1]) if k is not None else None
    src = open(os.path.join(CODE_ROOT, "scripts", "eval_fid_kid.py")).read()
    m = re.search(r"def _kid\(fr, fg, subset=\d+, n_subsets=(\d+), seed=0\)", src)
    if sub is None or m is None:
        raise SystemExit("realism: KID subset size or subset count not found")
    return sub, int(m.group(1))


def realism(R, out):
    f = os.path.join(V2, "realism", "realism_sector.json")
    j = json.load(open(f))
    v = j["variants"]
    lot, phy = v["prior LOTUS"], v["physics"]
    per = j["physics_beats_lotus_all_six"]["w1_per_case"]
    n_below = sum(1 for c in per if per[c][0] < per[c][1])
    n_below2 = sum(1 for c in phy["w1_per_case"] if phy["w1_per_case"][c] < lot["w1_per_case"][c])
    if n_below != n_below2 or n_below != R["table2"]["physics_closer_cases"]:
        raise SystemExit("physics-vs-LOTUS per-patient count disagrees across fields")
    pooled = phy["w1_in_sector"] < lot["w1_in_sector"]
    null = j["c2st_permutation_null_physics"]
    c2 = j["c2st"]
    n_folds = {k: v[k]["c2st_n"] for k in v}
    if len(set(n_folds.values())) != 1:
        raise SystemExit("C2ST fold counts differ across variants")
    kid_sub, kid_n = _kid_design(j["provenance"]["argv"])
    # the realism table calls this renderer "authors'"; its plug-row label (_pr("lotus_reimpl", ...) below) keeps the
    # wording of figures/plug_rows.json, which is md5-pinned
    groups = [("Baseline renderers", [("prior LOTUS", "LOTUS, authors' reimplementation"),
                                      ("NumPy", "NumPy renderer")]),
              ("USB Gym stages, default render", [("physics", "Physics (no network)"),
                                                  ("physics+CUT", "Physics + CUT"),
                                                  ("physics+CycleGAN", "Physics + CycleGAN"),
                                                  ("physics+SonoSPADE", "Physics + SonoSPADE")]),
              ("USB Gym stages, one full DR draw per view", [("physics [DR full]", "Physics"),
                                                              ("physics+CUT [DR full]", "Physics + CUT"),
                                                              ("physics+CycleGAN [DR full]", "Physics + CycleGAN"),
                                                              ("physics+SonoSPADE [DR full]", "Physics + SonoSPADE")])]
    listed = {k for _, g in groups for k, _ in g}
    if listed != set(v):
        raise SystemExit(f"realism variants not covered: {sorted(set(v) ^ listed)}")
    # v2.1 established-simulator rows (exploratory): same protocol, renderer swapped; their physics (and the
    # reimplementation) must equal the archived file exactly, so the rows sit on the archived scale
    v21, v21_src, orig_cmp = [], [], None
    for name, key, lab in REALISM_V21:
        fp = os.path.join(PLUG_ROOT, "renderer", "realism", name)
        if not os.path.exists(fp):
            continue
        d = _plug_json(fp)
        for pk, ak in (("physics", "physics"), ("lotus_reimpl", "prior LOTUS")):
            if pk in d["variants"]:
                for m_ in REALISM_PARITY:
                    if abs(float(d["variants"][pk][m_]) - float(v[ak][m_])) > 1e-12:
                        raise SystemExit(f"{rel(fp)}: variants[{pk}].{m_} differs from {rel(f)} variants[{ak}]")
        if d["views_per_case"] != j["views_per_case"] or d["n_real"] != j["n_real"] or d["erode_px"] != j["erode_px"] \
                or sorted(d["cases"]) != sorted(j["cases"]) or _kid_design(d["provenance"]["argv"]) != (kid_sub, kid_n):
            raise SystemExit(f"{rel(fp)}: design differs from {rel(f)}")
        x = d["variants"][key]
        v21.append((lab, x, f"{rel(fp)} variants[{key}]"))
        if key == "lotus_orig":
            pc = d["physics_vs_each"]["lotus_orig"]["w1_per_case"]
            orig_cmp = (sum(1 for c in pc if pc[c][0] < pc[c][1]), len(pc),
                        d["variants"]["physics"]["w1_in_sector"] < x["w1_in_sector"], fp)
        v21_src.append(rel(fp))
    cmp_txt = ""
    if orig_cmp is not None:
        cmp_txt = (f" In a post hoc comparison with the original LOTUS code, physics W1 is lower in "
                   f"{orig_cmp[0]}/{orig_cmp[1]} patients and {'' if orig_cmp[2] else 'not '}lower pooled.")
    # the caption keeps only the 200-permutation null of the realism-uncertainty record (the archived 10-permutation null
    # of realism_sector.json c2st_permutation_null_physics is not printed; it stays in the log line below)
    realism_null = _a11()["null"]
    a11_np = _a11()["reading"]["n_perm"]
    a11_m = {f"{n['null_mean']:.2f}" for n in realism_null.values()}
    a11_s = {f"{n['null_sd_ddof1']:.2f}" for n in realism_null.values()}
    if len(a11_m) != 1 or len(a11_s) != 1 or any(n["count_at_or_above"] for n in realism_null.values()):
        raise SystemExit(f"{LATE_REALISM_CI}: the caption's one null sentence needs one rounded mean and SD over the stages "
                         "and no permuted AUC at or above the observed one")
    null_txt = (f"A post hoc label-permutation null with {a11_np} permutations per stage gives a mean AUC of "
                f"{a11_m.pop()} (SD {a11_s.pop()}) for every stage and never reaches the observed AUC of any stage "
                f"($p = 1/{1 + a11_np}$ each). ")
    L = [GEN.format(src="outputs/v2/realism/realism_sector.json (variants, physics_beats_lotus_all_six)"
                        + (", and the v2.1 renderer realism runs "
                                                            + ", ".join(v21_src) if v21_src else "")
                        + f", and {LATE_REALISM_CI} (reading)"),
         "\\begin{table}[tp]",
         "\\caption{Distance to real B-mode frames in matched sector geometry. Simulated side: "
         f"{j['views_per_case']} views per held-out patient ({len(j['cases'])} patients, {j['n_views']} views) "
         "jittered around screened acoustic windows, warped to the real sector. Real side: the "
         f"{j['n_real']} annotated Vitale frames~\\cite{{vitale}}, which no texture stage trained on. All scores "
         f"use in-sector pixels only (sector mask eroded by {j['erode_px']}~px). W1: intensity "
         "Wasserstein-1 distance. KID and FID: kernel and Fr\\'echet Inception distances. C2ST AUC: area under "
         "the ROC curve of a CNN classifier two-sample test, mean $\\pm$ std over "
         f"{c2['repeats']} repeats $\\times$ {c2['folds']} grouped folds (0.5 means the classifier cannot "
         f"separate the domains). KID: mean $\\pm$ std over {kid_n} random subsets of {kid_sub} frames per domain. "
         "Lower W1, KID and FID mean a smaller distance to the real frames. "
         # the FID caveat and the NumPy row (Table S1.3 describes that renderer)
         f"FID from {j['n_real']} real frames is biased upward and is printed for completeness; KID is the unbiased "
         "kernel distance. "
         "NumPy renderer: the five-tissue appearance model of the authors' earlier simulator (Supplementary "
         "Table~S1.3). "
         + null_txt +
         "Physics W1 is below the authors' reimplementation of LOTUS in "
         f"{n_below}/{len(per)} held-out patients and {'is' if pooled else 'is not'} below it pooled."
         + cmp_txt
         # post hoc: pointer to the per-stage null and the frame-bootstrap intervals of Supplementary S2
         # (tables/posthoc_realism.tex); the pooled null prints above
         + (" Supplementary Table~S2.16 gives the null per stage and post hoc frame-bootstrap intervals for W1 and "
            "KID.")
         + (" The established-simulator rows run the same protocol with the renderer swapped (exploratory). "
            "The original LOTUS code runs with its published default parameters." if v21 else "")
         # the physics-row identity is asserted above (REALISM_PARITY), not printed
         + "}"
         + (f" % src: KID subsets: eval_fid_kid._kid n_subsets {kid_n}, --kid-subset {kid_sub} in the provenance argv "
            "of every file read"
            + f"; null sentence: {LATE_REALISM_CI} null[*].null_mean, .null_sd_ddof1 (one value each at 2 decimals), "
              ".count_at_or_above (0 each), reading.n_perm"
            + f"; S2 pointer: {LATE_REALISM_CI} reading.n_perm ({_a11()['reading']['n_perm']}), reading.bootstrap_B "
              f"({_a11()['reading']['bootstrap_B']}, frame bootstrap)"
            + ("" if orig_cmp is None else
                                    f"; published default parameters: {LOTUS_DEFAULTS_SRC}"
                                    f"; original-code comparison {rel(orig_cmp[3])} physics_vs_each.lotus_orig."
                                    f"w1_per_case ({orig_cmp[0]} of {orig_cmp[1]} lower) and variants[*].w1_in_sector "
                                    "(POST HOC" + _T("realism date") + ")")),
         "\\label{tab:realism}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small\\setlength{\\fboxsep}{0.6pt}",
         "\\begin{tabular}{@{}lcccc@{}}",
         "\\toprule",
         "Variant & W1 $\\downarrow$ & KID $\\downarrow$ & FID $\\downarrow$ & C2ST AUC \\\\",
         "\\midrule"]
    log("\n== tab:realism (realism.tex) ==")

    for gi, (head, items) in enumerate(groups):
        if gi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
        for k, lab in items:
            L.append(_realism_row(lab, v[k], f"realism_sector.json variants[{k}]"))
            x = v[k]
            log(f"{k:28s} W1 {x['w1_in_sector']:.4f} KID {x['kid']:.4f} +/- {x['kid_std']:.4f} FID "
                f"{x['fid_secondary']:.1f} AUC {x['c2st_auc_mean']:.3f} +/- {x['c2st_auc_std']:.3f}")
    if v21:
        L += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{Established simulator code (exploratory)}} \\\\"]
        for lab, x, src_ in v21:
            L.append(_realism_row(lab, x, src_, prov=True))
            log(f"{lab:28s} W1 {x['w1_in_sector']:.4f} KID {x['kid']:.4f} +/- {x['kid_std']:.4f} FID "
                f"{x['fid_secondary']:.1f} AUC {x['c2st_auc_mean']:.3f} +/- {x['c2st_auc_std']:.3f} (v2.1)")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    log(f"physics W1 below the LOTUS reimplementation: {n_below}/{len(per)} patients, pooled {pooled}; C2ST null "
        f"{null['mean']:.3f} +/- {null['std']:.3f} (n={null['n']})")
    if orig_cmp is not None:
        log(f"physics W1 below the original LOTUS code (post hoc): {orig_cmp[0]}/{orig_cmp[1]} patients, pooled "
            f"{orig_cmp[2]}")
    log(f"plan sentence (Q4): {R['table2']['sentence']}")
    return write(out, "realism.tex", "\n".join(L))


def _realism_row(lab, x, src_, prov=False):
    """One Table A.4 row: W1, KID with its std over subsets, FID, C2ST AUC with its std; v2.1 rows are provisional."""
    w = _vf if prov else (lambda t: t)
    w1 = w(f"{x['w1_in_sector']:.4f}")
    kid = w(f"${x['kid']:.4f} \\pm {x['kid_std']:.4f}$")
    fid = w(f"{x['fid_secondary']:.1f}")
    auc = w(f"${x['c2st_auc_mean']:.3f} \\pm {x['c2st_auc_std']:.3f}$")
    return f"\\quad {lab} & {w1} & {kid} & {fid} & {auc} \\\\ % src: {src_}"


# =================================================================================== tab:identity
def identity(R, out):
    organs4 = ("liver", "gallbladder", "vessel", "kidney")
    sides = (("vitale", "Real side: all {n} annotated Vitale frames, out-of-fold predictions"),
             ("vitaleho", "Real side: the {n} Vitale holdout frames only"))
    info = {}
    for side, _ in sides:
        tv = json.load(open(os.path.join(SEG, "outputs", "v2", f"tablev_v2_{side}.json")))
        rv = R["table5"][side]
        if json.loads(json.dumps(tv)) != rv:
            raise SystemExit(f"tablev_v2_{side}.json differs from RESULTS.json table5.{side}")
        if tv["missing"] or tv["seat_mode"] != "anatomical" or tv["std"] != "population":
            raise SystemExit(f"tablev_v2_{side}.json: missing inputs, wrong seat or std")
        seeds = tv["seeds"]
        stages = {}
        nreal, nsim, nkid, srcs = set(), set(), set(), set()
        for s in STAGES:
            per_seed = []
            tally = collections.Counter()
            for sd in seeds:
                vit = json.load(open(os.path.join(SEG, "outputs", "v2", f"consistency_{s}_s{sd}_{side}.json")))
                kid = json.load(open(os.path.join(SEG, "outputs", "v2", f"consistency_{s}_s{sd}_kidneyus.json")))
                nreal.add(vit["n_real_frames"]); nsim.add(vit["n_sim_frames"]); nkid.add(kid["n_real_frames"])
                srcs.add(vit["real_prediction_source"].split(" (")[0])
                hits = {o: bool(vit["per_class"][o]["modal_is_correct"]) for o in organs4 if o != "kidney"}
                hits["kidney"] = bool(kid["per_class"]["kidney"]["modal_is_correct"])
                hits["spleen"] = bool(vit["per_class"]["spleen"]["modal_is_correct"])
                tally.update(o for o, h in hits.items() if h)
                per_seed.append(sum(hits[o] for o in organs4))
            agg = tv["stages"][s]["per_organ_of_3"]
            if any(tally[o] != agg[o] for o in organs4 + ("spleen",)):
                raise SystemExit(f"{side}/{s}: per-organ tallies disagree with tablev_v2_{side}.json")
            stages[s] = {"tally": {o: agg[o] for o in organs4 + ("spleen",)}, "per_seed4": per_seed,
                         "mean": statistics.mean(per_seed), "pstd": statistics.pstdev(per_seed)}
        if len(nreal) != 1 or len(nsim) != 1 or len(nkid) != 1:
            raise SystemExit(f"{side}: frame counts differ across stages or seeds")
        four = R["table5"][f"{side}_verdict"]["four_organ_mean"]
        if any(abs(four[s] - stages[s]["mean"]) > 1e-9 for s in STAGES):
            raise SystemExit(f"{side}: 4-organ means disagree with RESULTS.json")
        info[side] = {"stages": stages, "n_real": nreal.pop(), "n_sim": nsim.pop(), "n_kid": nkid.pop(),
                      "seeds": seeds, "src": sorted(srcs), "sentence": R["table5"][f"{side}_verdict"]["sentence"]}
    nv, nh = info["vitale"]["n_real"], info["vitaleho"]["n_real"]
    ns = len(info["vitale"]["seeds"])
    L = [GEN.format(src=PI.text("identity src")),
         "\\begin{table}[tp]",
         "\\caption{Cross-domain organ identity of each stage's texture-matched segmenter. A cell counts the "
         f"segmenter seeds, of {ns}, on which the modal predicted class of the organ is the correct class both "
         f"in simulation ({info['vitale']['n_sim']} frames from the held-out patients) and on real frames. "
         f"Liver, gallbladder (GB), vessel and spleen are scored on Vitale~\\cite{{vitale}}: all {nv} annotated "
         f"frames, each predicted by a model that never trained on it (top), or only the {nh} holdout frames "
         f"that no segmenter trained on (bottom). Kidney is scored on the KidneyUS holdout~\\cite{{kidneyus}} "
         f"({info['vitale']['n_kid']} frames) in both panels. Last column: organs of the four (liver, GB, "
         f"vessel, kidney) per seed, mean $\\pm$ population std over the {ns} seeds. Spleen is not a target and "
         "is descriptive only. The table is descriptive. Under the pre-specified rule, both panels detect no "
         "difference in organ-identity consistency across stages (not powered for equivalence).}",
         "\\label{tab:identity}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}lcccccc@{}}",
         "\\toprule",
         "Stage & Liver & GB & Vessel & Kidney & Spleen & Organs of 4 \\\\",
         "\\midrule"]
    # RESULTS.json keeps the Q3 sentence of the v2 plan; the caption prints the outcome in other words, so the outcome
    # is checked here and the sentence itself never prints.
    if not PI.skip_dev(C.SENTENCE_SKIP) and any(info[s]["sentence"] != PP.V2_SENTENCES["q3 same"] for s in info):
        raise SystemExit("Q3 outcome changed: rewrite the identity caption's last sentence")
    log("\n== tab:identity (identity.tex) ==")
    for i, (side, head) in enumerate(sides):
        if i:
            L.append("\\midrule")
        n_here = info[side]["n_real"]
        head = head.format(n=n_here)
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
        for s in ("sonospade", "physics", "cut", "cyclegan"):
            st = info[side]["stages"][s]
            cells = " & ".join(f"{st['tally'][o]}/{ns}" for o in organs4 + ("spleen",))
            L.append(f"\\quad {NAME[s]} & {cells} & ${st['mean']:.1f} \\pm {st['pstd']:.1f}$ \\\\ "
                     f"% src: tablev_v2_{side}.json stages.{s}")
            log(f"{side:8s} {s:9s} " + " ".join(f"{o} {st['tally'][o]}/{ns}" for o in organs4 + ("spleen",))
                + f" | per-seed 4-organ {st['per_seed4']} -> {st['mean']:.2f} +/- {st['pstd']:.2f}")
        log(f"{side}: real frames {info[side]['n_real']}, sim frames {info[side]['n_sim']}, KidneyUS frames "
            f"{info[side]['n_kid']}, prediction source {info[side]['src']}; Q3: {info[side]['sentence']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "identity.tex", "\n".join(L))


# =================================================================================== tab:registration
def registration(R, out):
    reg = R["registration"]
    f = os.path.join(V2, "table4", "registration_long.csv")
    rows = [r for r in csv.DictReader(open(f)) if r.get("mask") == "seg"]
    reps = collections.Counter((r["stage"], r["src"], r["organ"], r["frame"]) for r in rows)
    if len(set(reps.values())) != 1:
        raise SystemExit("registration: repeats per target are not constant")
    n_rep = next(iter(reps.values()))
    seeds = sorted({r["seed"] for r in rows})
    cts = collections.Counter((r["stage"], r["src"], r["organ"], r["frame"], r["seed"]) for r in rows)
    if len(set(cts.values())) != 1:
        raise SystemExit("registration: CTs per target and seed are not constant")
    n_ct = next(iter(cts.values()))
    if n_ct * len(seeds) != n_rep:
        raise SystemExit("registration: repeats != seeds x CTs")
    groups = [("vitale_holdout", "liver", "Vitale holdout~\\cite{vitale}"),
              ("vitale_holdout", "kidney", "Vitale holdout~\\cite{vitale}"),
              ("aul", "liver", "AUL~\\cite{aul}"),
              ("kidneyus", "kidney", "KidneyUS holdout~\\cite{kidneyus}")]
    covered = {f"{s}/{src}/{o}" for src, o, _ in groups for s in STAGES}
    if covered != set(reg):
        raise SystemExit(f"registration cells not covered: {sorted(covered ^ set(reg))}")
    L = [GEN.format(src="outputs/v2/RESULTS.json (registration) and outputs/v2/table4/registration_long.csv"),
         "\\begin{table}[tp]",
         "\\caption{Offline registration against real target frames (no probe, phantom or robot). Each real "
         f"target is paired with {n_ct} held-out CTs. Each learned-mask servo ({len(seeds)} seeds) takes its "
         "masks of the real frame and of the simulated view, for control and reward, from its own segmenter. "
         "Score: organ-restricted IoU "
         "between the CT label of the simulated view and the annotated organ of the real frame, at the "
         "keep-best step (chosen by that reward, without ground truth) minus at the static start. Mean over "
         f"targets (each averaged over {n_rep} runs), with a 95\\% bootstrap CI over targets. Rule met: yes "
         "only if the CI lies above 0 (pre-specified rule). AUL "
         "and KidneyUS frames are resized screenshots outside the convex-sector geometry of the renders.}",
         "\\label{tab:registration}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}lccl@{}}",
         "\\toprule",
         "Stage & Keep-best minus static IoU & 95\\% CI & Rule met \\\\ % src: " + _T("registration col src"),
         "\\midrule"]
    log("\n== tab:registration (registration.tex) ==")
    log(f"repeats per target {n_rep} = {len(seeds)} seeds x {n_ct} CTs")
    for gi, (src, organ, lab) in enumerate(groups):
        ns = {reg[f"{s}/{src}/{organ}"]["n_targets"] for s in STAGES}
        if len(ns) != 1:
            raise SystemExit(f"registration {src}/{organ}: target counts differ across stages")
        n = ns.pop()
        if gi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{lab}, {organ}, $n = {n}$ targets}}}} \\\\")
        for s in STAGES:
            e = reg[f"{s}/{src}/{organ}"]
            ci = e["ci"]
            if bool(e["holds"]) != (ci[0] > 0):
                raise SystemExit(f"registration {s}/{src}/{organ}: holds flag disagrees with the CI")
            verdict = "yes" if e["holds"] else ("no (CI below 0)" if ci[1] < 0 else "no")
            L.append(f"\\quad {NAME[s]} & {sg(e['mean'], 3)} & {ci_txt(ci, 3)} & {verdict} \\\\ "
                     f"% src: RESULTS.json registration.{s}/{src}/{organ}")
            log(f"{s:9s} {src:14s} {organ:6s} n={e['n_targets']:3d} {e['mean']:+.3f} "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] -> {verdict}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "registration.tex", "\n".join(L))


# =================================================================================== tab:dr
def dr(out):
    f = os.path.join(V2, "dr", "dr_table.json")
    j = json.load(open(f))
    eff, kn = j["effects"], j["knobs"]
    st = [s for s in STAGES if s in eff]
    if st != list(STAGES):
        raise SystemExit("dr_table.json lacks a stage")
    order = j["dr_draw_order"]
    per_label = [k for k in order if kn[k]["per_label"]]
    scalar = [k for k in order if not kn[k]["per_label"]]
    lab = {"gain": "Overall gain", "tgc_slope": "Time-gain slope", "psf_sigma_lat0": "Lateral PSF $\\sigma$ at focus (px)",
           "psf_focus_depth": "Focal depth (m)", "speckle_strength": "Speckle strength",
           "noise_sigma": "Electronic noise $\\sigma$", "dyn_range_db": "Dynamic range (dB)",
           "z_scale": "Impedance multiplier$^{\\dagger}$", "mu_scale": "Attenuation multiplier$^{\\dagger}$",
           "bs_scale": "Backscatter multiplier$^{\\dagger}$"}
    if set(lab) != set(order):
        raise SystemExit(f"DR knobs changed: {sorted(set(lab) ^ set(order))}")

    def key(k, end):
        return f"{k}|per_label_mean|{end}" if kn[k]["per_label"] else f"{k}|{end}"

    def cell(s, k):
        return f"{100 * eff[s][k]['mean']:.2f}"

    n_frames = {eff[s][key(k, e)]["n"] for s in st for k in order for e in ("lo", "hi")}
    if len(n_frames) != 1:
        raise SystemExit("DR effect sample sizes differ")
    nf = n_frames.pop()
    n_full = {eff[s]["ref|full_DR_draw"]["n"] for s in st}   # dr_probe.py --draws full draws per frame
    if len(n_full) != 1 or next(iter(n_full)) % nf:
        raise SystemExit(f"full-DR-draw sample sizes {n_full} are not a constant multiple of {nf} frames")
    n_full = n_full.pop()
    if any(eff[s][k]["n"] != nf for s in st for k in ("ref|speckle_seed_only", "ref|pose_5mm_slide")):
        raise SystemExit("seed-only or slide reference rows are not one value per frame")
    mx ={k: max(100 * eff[s][key(k, e)]["mean"] for s in st for e in ("lo", "hi")) for k in order}
    low2 = sorted(order, key=lambda k: mx[k])[:2]
    L = [GEN.format(src="outputs/v2/dr/dr_table.json (the data behind outputs/v2/dr/tab_dr.tex, "
                        "written by scripts/dr_probe.py)"),
         "\\begin{table*}[tp]",
         "\\caption{Domain-randomization (DR) knobs. When DR is on, each knob is drawn uniformly from its range "
         "once per episode and shared by the target and the observations. DR is off in training and "
         "evaluation for Tables~\\ref{tab:ladder} and~\\ref{tab:learnability}; Table~\\ref{tab:contact-dr} "
         "reports the DR arms. Effect: mean absolute change of "
         "the in-sector pixel intensity (scale 0 to 1, $\\times 100$) when the knob alone moves to the low (lo) "
         "or high (hi) end of its range, with the speckle seed fixed, over "
         f"{nf} frames ({len(j['cases'])} held-out patients $\\times$ {j['poses_per_case']} jittered window "
         "poses), with each texture stage applied (physics: none). $^{\\dagger}$Drawn independently per tissue "
         "label and measured one label at a time (mean over labels covering at least 1\\% of the sector). The "
         "speckle and noise seed is also redrawn per episode. The reference rows give the change from a new "
         "seed alone, a 5~mm in-plane slide and a full DR draw "
         f"({n_full // nf} draws per frame, {n_full} in all). The two weakest knobs, "
         f"{lab[low2[0]].split('$')[0].strip().lower()} and {lab[low2[1]].split('$')[0].strip().lower()}, "
         f"change the frame by at most {mx[low2[0]]:.2f} and {mx[low2[1]]:.2f}.}}",
         "\\label{tab:dr}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}llc" + "@{\\hspace{12pt}}rr" * len(st) + "@{}}",
         "\\toprule",
         " & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAME[s]}}}" for s in st) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(st))),
         "Knob & Default & Range & " + " & ".join("lo & hi" for _ in st) + " \\\\",
         "\\midrule"]
    log("\n== tab:dr (dr.tex) ==")
    log(f"frames per effect {nf}; weakest knobs {low2} (max {mx[low2[0]]:.2f}, {mx[low2[1]]:.2f})")
    for k in scalar + per_label:
        d, r = kn[k]["default"], kn[k]["range"]
        L.append(f"{lab[k]} & {d:g} & [{r[0]:g}, {r[1]:g}] & "
                 + " & ".join(f"{cell(s, key(k, 'lo'))} & {cell(s, key(k, 'hi'))}" for s in st)
                 + f" \\\\ % src: dr_table.json knobs.{k}, effects.<stage>.{key(k, '<end>')}")
        log(f"{k:17s} default {d:g} range [{r[0]:g}, {r[1]:g}] | " + "  ".join(
            f"{s} {cell(s, key(k, 'lo'))}/{cell(s, key(k, 'hi'))}" for s in st))
    L.append("\\midrule")
    for k, txt in (("ref|speckle_seed_only", "New speckle seed only"), ("ref|pose_5mm_slide", "5 mm in-plane slide"),
                   ("ref|full_DR_draw", "Full DR draw")):
        L.append(f"\\textit{{{txt}}} & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cell(s, k)}}}" for s in st)
                 + f" \\\\ % src: dr_table.json effects.<stage>.{k}")
        log(f"{k:22s} " + "  ".join(f"{s} {cell(s, k)}" for s in st))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    # cross-check against the LaTeX table that dr_probe writes
    tex = open(os.path.join(V2, "dr", "tab_dr.tex")).read()
    for s in st:
        for k in scalar:
            for e in ("lo", "hi"):
                if cell(s, key(k, e)) not in tex:
                    raise SystemExit(f"tab_dr.tex lacks {s} {k} {e}: regenerated values diverge")
    log("[check] every scalar-knob cell also appears in outputs/v2/dr/tab_dr.tex")
    return write(out, "dr.tex", "\n".join(L))


# =================================================================================== tab:windows
def windows(out):
    f = os.path.join(ROOT, "data", "cache", "windows", "usbg-windows-v1.json")
    d = json.load(open(f))
    cases = sorted(d["cases"])
    if set(cases) != set(TRAIN_PATIENTS) | set(TEST_PATIENTS) or set(TRAIN_PATIENTS) & set(TEST_PATIENTS):
        raise SystemExit("windows file cases do not match the TRAIN/TEST split")
    # the minimum-window rule actually used by every trainer run, and the cells it skipped
    mins, skipped = set(), set()
    for jf in sorted(glob.glob(os.path.join(V2, "table4", "*", "*.json"))):
        if jf.endswith(".provenance.json") or os.path.basename(jf) == "cells.json":
            continue
        j = json.load(open(jf))
        if "config" not in j or "train_setups" not in j:
            continue
        mins.add(j["config"].get("min_windows"))
        skipped |= set(j["train_setups"].get("skip_draws", {}))
    if len(mins) != 1:
        raise SystemExit(f"trainer runs used different --min-windows: {mins}")
    mw = mins.pop()
    cnt = {c: {o: len(d["cases"][c]["windows"].get(o, [])) for o in ("liver", "kidney_right", "kidney_left")}
           for c in cases}
    rule_skip = {f"{c}:kidney" for c in TRAIN_PATIENTS if cnt[c]["kidney_right"] + cnt[c]["kidney_left"] < mw}
    rule_skip |= {f"{c}:liver" for c in TRAIN_PATIENTS if cnt[c]["liver"] < mw}
    if rule_skip != skipped:
        raise SystemExit(f"skipped cells {sorted(skipped)} differ from the window rule {sorted(rule_skip)}")
    if any(cnt[c]["liver"] < mw or cnt[c]["kidney_right"] + cnt[c]["kidney_left"] < mw for c in TEST_PATIENTS):
        raise SystemExit("a held-out cell falls below the minimum-window rule")
    skip_cases = sorted(s.split(":")[0] for s in skipped)
    p = d["params"]
    L = [GEN.format(src="data/cache/windows/usbg-windows-v1.json, usbg.benchmark.TRAIN_PATIENTS/TEST_PATIENTS "
                        "and the trainer JSONs under outputs/v2/table4/ (config.min_windows, train_setups.skip_draws)"),
         "\\begin{table}[t]",
         # the windows-file name (p["version"]) does not print; the screens are listed in the testbed details of the
         # supplement (app:testbed), where this table prints
         "\\caption{Screened acoustic windows per case. Each "
         "window passes the screens of Section~\\ref{app:testbed}; liver windows must also show a landmark "
         "organ. Kidney targets use flank "
         f"windows of either side. A case and target enter training or evaluation only with at least {mw} "
         "windows (kidney counts both sides), so the kidney targets of "
         + ", ".join(skip_cases[:-1]) + f" and {skip_cases[-1]} are skipped ($^{{\\dagger}}$). "
         f"Train: {len(TRAIN_PATIENTS)} cases; held out: {len(TEST_PATIENTS)} cases, used only for "
         "evaluation.}",
         "\\label{tab:windows}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}llrrr@{}}",
         "\\toprule",
         "Case & Split & Liver & Kidney, right & Kidney, left \\\\",
         "\\midrule"]
    log("\n== tab:windows (windows.tex) ==")
    log(f"min windows {mw}; skipped training cells {sorted(skipped)}")
    tot = {"train": collections.Counter(), "held-out": collections.Counter()}
    for split, group in (("Train", sorted(TRAIN_PATIENTS)), ("Held out", sorted(TEST_PATIENTS))):
        for c in group:
            x = cnt[c]
            mark = "$^{\\dagger}$" if f"{c}:kidney" in skipped else ""
            L.append(f"{c}{mark} & {split.lower()} & {x['liver']} & {x['kidney_right']} & {x['kidney_left']} \\\\ "
                     f"% src: usbg-windows-v1.json cases.{c}.windows")
            tot["train" if split == "Train" else "held-out"].update(x)
            log(f"{c} {split:8s} liver {x['liver']:3d} kidney R {x['kidney_right']:3d} kidney L {x['kidney_left']:3d}"
                + (" (kidney skipped)" if mark else ""))
        L.append("\\midrule")
    for name, lab in (("train", f"Total, train ({len(TRAIN_PATIENTS)})"),
                      ("held-out", f"Total, held out ({len(TEST_PATIENTS)})")):
        x = tot[name]
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{lab}}} & {x['liver']} & {x['kidney_right']} & {x['kidney_left']} \\\\")
        log(f"{lab}: liver {x['liver']} kidney R {x['kidney_right']} kidney L {x['kidney_left']}")
    a = tot["train"] + tot["held-out"]
    L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Total, all ({len(cases)})}} & {a['liver']} & {a['kidney_right']} & "
             f"{a['kidney_left']} \\\\")
    log(f"Total all ({len(cases)}): liver {a['liver']} kidney R {a['kidney_right']} kidney L {a['kidney_left']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "windows.tex", "\n".join(L))


# =================================================================================== tab:compute
def compute_cost(R, out):
    """CUDA-free cost: render chain, texture stages and one closed-loop step on CPU and MPS (batch 1, one
    process), plus the oracle-ladder training cost. RESULTS.json compute is re-derived from the benchmark JSONs
    and the runner log first, so a stale section stops the run."""
    if "compute" not in R:
        raise SystemExit("RESULTS.json has no compute section: rerun scripts/collect_results_v2.py")
    fresh = {}
    C.compute(fresh)
    if "compute" not in fresh:
        raise SystemExit("compute benchmarks missing or stale: see the RESULTS.md Compute section")
    _close(R["compute"], json.loads(json.dumps(fresh["compute"], default=str)), "compute")
    log("[check] RESULTS.json compute re-derived from the benchmark JSONs and runner.log: identical")
    k = R["compute"]
    m, t = k["machine"], k["training"]
    devs = [d for d in ("cpu", "mps") if d in k["closed_loop"]]
    if devs != ["cpu", "mps"]:
        raise SystemExit(f"compute table needs CPU and MPS rows, got {devs}")
    if k["cuda_available"] or k["cpu_run_mps_bytes"] != 0:
        raise SystemExit("compute: CUDA was available or the CPU run touched MPS; the caption would be wrong")
    cl = k["closed_loop"]
    n_steps = {cl[d][st]["n_steps"] for d in devs for st in STAGES}
    n_eps = {cl[d][st]["n_timed_episodes"] for d in devs for st in STAGES}
    spe = {cl[d][st]["steps_per_episode"] for d in devs for st in STAGES}
    n_wu = {cl[d][st]["n_warmup_steps"] for d in devs for st in STAGES}
    thr = set(k["torch_threads"].values())
    seeds = set(t["train_n_by_arm_stage"].values())
    if max(len(x) for x in (n_steps, n_eps, spe, n_wu, thr, seeds)) != 1:
        raise SystemExit("compute: step counts, threads or seeds differ across rows")
    (n_steps,), (n_eps,), (spe,), (n_wu,), (thr,), (seeds,) = n_steps, n_eps, spe, n_wu, thr, seeds
    tx = {st: k["texture"][st] for st in ("cut", "cyclegan", "sonospade")}
    rs, ts_, cs = k["render_setup"], k["texture_setup"], k["closed_loop_setup"]

    def fm(x):
        return f"{x:.2f}" if x < 10 else f"{x:.1f}"

    def iq(e):
        d = 2 if e["median_ms"] < 10 else 1
        return f"{e['median_ms']:.{d}f} [{e['p25_ms']:.{d}f}, {e['p75_ms']:.{d}f}]"

    arm_lab = {"ctmask": "CT mask", "matched": "Learned mask", "dr_on": "CT mask, DR on"}
    # Table 3 prints the tested platform and software; the frame and step times, the steps per second, the
    # texture-alone rows and the training hours print in Supplementary S4 (compute_supp.tex), and the CPU-only
    # agreement (Q14) is computed and logged.
    bp = json.load(open(os.path.join(V2, "bench_closed_loop.json")))["provenance"]
    vers = {k: bp[k] for k in ("python", "torch", "numpy", "mujoco")}
    inst = PI.printed_versions()     # importlib.metadata of the generating .venv; the recorded versions in public mode
    lic_text = open(os.path.join(CODE_ROOT, "LICENSE")).read()
    lic = ("Apache-2.0" if "Apache License" in lic_text and "Version 2.0" in lic_text
           else next(ln.strip() for ln in lic_text.splitlines() if ln.strip()).replace(" License", ""))
    ci = os.path.join(CODE_ROOT, ".github", "workflows", "tests.yml")
    vf_ = (lambda x: f"\\verifyfill{{{x}}}") if PLUG_PROVISIONAL else (lambda x: x)
    # Section 4.4 of the article prints the CPU step medians, the steps per second and the physics frame in prose. The caption names
    # the hardware in one sentence; the CPU-only retrain cell and the license and tests row are written as comments
    # only. Table S4.1 cites this table by \ref.
    cap = (f"Hardware and software of USB Gym's runs: every reported result was computed on one {m['chip']} laptop "
           f"with {m['memory_gb']:.0f}~GB of unified memory.")
    L = [GEN.format(src="outputs/v2/RESULTS.json (compute), re-derived from outputs/v2/bench_render.json, "
                        "bench_latency.json, bench_closed_loop.json and outputs/v2/table4/runner.log; versions from "
                        "outputs/v2/bench_closed_loop.json provenance and the installed .venv; LICENSE; "
                        ".github/workflows/tests.yml"),
         "\\begin{table}[tp]",
         "\\caption{" + cap + "} % src: RESULTS.json compute.machine; "
         f"{_T('compute cut note')}: block A in tables/compute_supp.tex",
         "\\label{tab:compute}",
         "\\centering",
         # a tight box around the verify marks, as in the other generated tables, so a filled cell does not make
         # Table 3 taller (it shares its page with body text)
         "\\renewcommand{\\baselinestretch}{1}\\small\\setlength{\\fboxsep}{0.6pt}",
         "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{0.27\\textwidth} "
         ">{\\raggedright\\arraybackslash}X@{}}",
         "\\toprule"]
    log("\n== tab:compute (compute.tex), Table 3: hardware and software (frame and step times in compute_supp.tex) ==")
    cq = _q_count(_queue(), "v21:cohort:cpu:")   # the CPU-only retrain jobs (Q14), counted, not typed
    for d in devs:
        log(f"physics frame {d}: {iq(k['render'][d]['frame'])} ms")
        log(f"closed-loop step {d}: " + ", ".join(f"{st} {iq(cl[d][st])}" for st in STAGES))
        log(f"steps/s {d}: " + ", ".join(f"{st} {cl[d][st]['steps_per_s']:.1f}" for st in STAGES))
    q14 = _q14()
    # the cell gives the CPU retrain's wall time and its band test (Q14 (i), computed in _q14); no MPS wall time
    # stands beside it, because the archived MPS trainers shared the laptop with other jobs (no speed ordering)
    if q14 is None:
        cpu_cell = _T("cpu pending cell").format(cq[0], cq[1])
    else:
        fi_ = q14["phases"]["final"]
        if not (fi_["within"] and q14["phases"]["imit"]["within"]):
            raise SystemExit("Table 3: the CPU retrain lies outside +/- 1.0 mm; rewrite the CPU-only retrain cell")
        cpu_cell = vf_(f"{q14['H_h']:.2f}") + "~h; " + vf_(f"{fi_['X']:.2f}") + "~mm, " + _T("cpu cell tail")
    log(f"CPU-only retrain cell ({_T('cpu cell tag')}): {cpu_cell}")
    L += [f"% {_T('cpu cell moved')}: CPU-only retrain & {cpu_cell} \\\\ % src: jobs v21:cohort:cpu:* in outputs/v2_1/"
          f"jobs_final_r3.json ({cq[0]} of {cq[1]} outputs)" + ("" if q14 is None else
          f"; H {q14['H_sec']:.1f} s; final {q14['phases']['final']['X']:.4f} against the MPS seed mean "
          f"{q14['phases']['final']['Y']:.4f} mm; " + q14["src"]) + _T("q14 templates"),
          f"Machine & {m['chip']}, {m['memory_gb']:.0f}~GB, {m['ncpu']} CPU cores, macOS "
          f"({m['system']}), Apple GPU through PyTorch MPS \\\\ % src: RESULTS.json compute.machine (chip, "
          "memory_gb, ncpu, system, mps_available)" + _T("cuda line tag"),
          f"Python, PyTorch & {vers['python']}, {vers['torch']} \\\\ % src: "
          "outputs/v2/bench_closed_loop.json provenance.(python, torch); the same versions in every archived v2 "
          "result provenance",
          f"NumPy, MuJoCo & {vers['numpy']}, {vers['mujoco']} \\\\ % src: "
          "outputs/v2/bench_closed_loop.json provenance.(numpy, mujoco)",
          f"Gymnasium, Stable-Baselines3 & {vf_(inst['gymnasium'])}, "
          f"{vf_(inst['stable-baselines3'])} \\\\ % src: importlib.metadata of the generating .venv (not in the run "
          "provenance); verify against the lock file at the release tag",
          f"% {_T('license moved')}: License, tests & {lic}; " + _T("compute lic a") + rel(ci)
          + _T("compute lic b"),
          "\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    log(f"training: median {t['train_h_median_all']:.2f} h (range {t['train_h_min']:.2f} to {t['train_h_max']:.2f}) "
        f"over {t['train_jobs']} runs; versions {vers}; installed {inst}; {lic}")
    p_main = write(out, "compute.tex", "\n".join(L))
    # Supplementary S4, Table S4.1: frame and step times, steps per second, the texture-alone rows and the training
    # hours
    S4 = [GEN.format(src="outputs/v2/RESULTS.json (compute), re-derived from outputs/v2/bench_render.json, "
                         "bench_latency.json, bench_closed_loop.json and outputs/v2/table4/runner.log"),
          "\\begin{table}[t]",
          # the compute detail of Table 3; the training wall hours print as upper bounds (the total wall time of the
          # queue is written as a comment, R5)
          "\\caption{Frame, step and training times on the laptop of Table~\\ref{tab:compute} of the article. "
          "Median [25th, 75th percentile] in ms, batch 1, one process, "
          f"{thr} threads. Physics frame: slice, render, resize and sector warp timed as one block. Closed-loop step: "
          f"each stage's seed-{cs['policy_seed']} learned-mask policy, {n_steps} steps after {n_wu} warm-up steps; the "
          "CPU rows run without the Apple GPU, the MPS rows on it through PyTorch. "
          f"Texture stage: the stage's generator alone on {ts_['n']} rendered sector frames after {ts_['warmup']} "
          "warm-up frames; physics has no texture stage. "
          f"Training: wall hours per trainer run on the laptop's GPU, median over {seeds} seeds, with "
          f"{ {2: 'two', 3: 'three', 4: 'four'}.get(t['concurrent_jobs'], t['concurrent_jobs'])} runs sharing the "
          "machine, so upper bounds.} "
          "% src: RESULTS.json compute.torch_threads, "
          "compute.closed_loop_setup, compute.closed_loop.<dev>.<stage>.(n_steps, n_warmup_steps), compute.texture_setup, "
          "compute.training.(concurrent_jobs, train_jobs, jobs_done, wall_h); scripts/bench_render.py (frame = slice + "
          "render + resize + warp, one block)",
          "\\label{tab:compute-supp}",
          "\\centering",
          "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
          "\\setlength{\\tabcolsep}{3pt}",
          "\\begin{tabular}{@{}l cccc@{}}",
          "\\toprule",
          " & Physics & CUT & CycleGAN & SonoSPADE \\\\",
          "\\midrule",
          "\\multicolumn{5}{@{}l}{\\textit{Physics frame, ms (every stage)}} \\\\"]
    for d in devs:
        e = k["render"][d]["frame"]
        S4.append(f"\\quad {d.upper()} & \\multicolumn{{4}}{{c}}{{{iq(e)}}} \\\\ % src: RESULTS.json compute.render.{d}.frame")
    S4 += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{One closed-loop step, ms}} \\\\"]
    for d in devs:
        S4.append(f"\\quad {d.upper()} & " + " & ".join(iq(cl[d][st]) for st in STAGES)
                  + f" \\\\ % src: RESULTS.json compute.closed_loop.{d}.<stage>")
    S4 += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{Closed-loop steps per second, from the median step}} \\\\"]
    for d in devs:
        S4.append(f"\\quad {d.upper()} & " + " & ".join(f"{cl[d][st]['steps_per_s']:.1f}" for st in STAGES)
                  + f" \\\\ % src: RESULTS.json compute.closed_loop.{d}.<stage>.steps_per_s")
    S4 += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{Texture stage alone, ms per frame}} \\\\"]
    for d in devs:
        S4.append(f"\\quad {d.upper()} & none & " + " & ".join(iq(tx[st][d]) for st in ("cut", "cyclegan", "sonospade"))
                  + f" \\\\ % src: RESULTS.json compute.texture.<stage>.{d}")
        log(f"texture {d}: " + ", ".join(f"{st} {iq(tx[st][d])}" for st in tx))
    # the training rows; the total wall time of the queue is written as a comment only (R5)
    S4 += ["\\addlinespace", "\\multicolumn{5}{@{}l}{\\textit{Training, wall hours per trainer run}} \\\\",
           f"\\quad All {t['train_jobs']} trainer runs & \\multicolumn{{4}}{{c}}{{median {t['train_h_median_all']:.2f}, range "
           f"{t['train_h_min']:.2f} to {t['train_h_max']:.2f}}} \\\\ % src: RESULTS.json compute.training.(train_h_median_all, "
           "train_h_min, train_h_max, train_jobs)"]
    hb = t["train_h_median_by_arm_stage"]
    for arm in ("ctmask", "matched", "dr_on"):
        cells_ = [fm(hb[f"{arm}/{st}"]) if f"{arm}/{st}" in hb else "n/a" for st in STAGES]
        S4.append(f"\\quad {arm_lab[arm]} & " + " & ".join(cells_)
                  + f" \\\\ % src: RESULTS.json compute.training.train_h_median_by_arm_stage.{arm}/<stage>")
        log(f"training wall hours {arm_lab[arm]}: " + ", ".join(cells_))
    S4.append(f"% {_T('wall time tag')}: all {t['jobs_done']} jobs took {t['wall_h']:.1f} h of wall time, "
              f"{t['concurrent_jobs']} at once")
    S4 += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(out, "compute_supp.tex", "\n".join(S4))
    return p_main


# ==================================================================================== POST HOC tables (v2.1)
# Written from outputs/v2/posthoc/posthoc_v2.json (scripts/posthoc_v2.py), only when RESULTS.json["posthoc"]
# certifies that exact file (same sha256) and its inputs are unchanged.
PH_SRC = "outputs/v2/posthoc/posthoc_v2.json (scripts/posthoc_v2.py; POST HOC" + _T("posthoc src tail") + ")"
PH_ROWS = [("ctmask", "CT mask"), ("matched", "Learned mask (texture-matched segmenter)")]
PH_PH = [("base", "Servo base"), ("after_imitation", "After imitation"), ("final", "Final")]
PH_REF = [("static", "Static start (no commanded motion)"), ("scripted", "Scripted sweep (no feedback)"),
          ("pose_oracle", "Analytic pose oracle (privileged)")]


def load_posthoc(R):
    """(posthoc json, None) or (None, reason). Never returns numbers from a stale or uncertified file."""
    import posthoc_v2 as PH
    blk = R.get("posthoc")
    if not blk:
        return None, "RESULTS.json has no posthoc section (run scripts/posthoc_v2.py, then the collector)"
    J, status = PH.load_checked()
    if J is None or status != "ok":
        return None, f"posthoc_v2.json {status}"
    if PH.sha256(PH.OUT_JSON) != blk.get("sha256"):
        return None, "posthoc_v2.json differs from the file RESULTS.json certifies (rerun the collector)"
    return J, None


def pfrac(p, n=6):
    """An exact sign-flip p as k/2^n."""
    k = p * 2 ** n
    if abs(k - round(k)) > 1e-9:
        raise SystemExit(f"exact sign-flip p {p} is not a multiple of 1/{2 ** n}")
    return f"${int(round(k))}/{2 ** n}$"


def pdec(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def ph_grid(J, metric, fmt):
    """Rows of the oracle-ladder layout for one post hoc metric: reference rungs span the four stages."""
    R = J["rows"]
    L = []
    for key, lab in PH_REF:
        L.append(f"{lab} & \\multicolumn{{4}}{{c}}{{{fmt(R[key]['metrics'][metric]['mean'])}}} \\\\ "
                 f"% src: posthoc_v2.json rows.{key}.metrics.{metric}")
    for arm, head in PH_ROWS:
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{head}}} \\\\")
        for ph, lab in PH_PH:
            L.append(f"\\quad {lab} & " + " & ".join(fmt(R[f'{arm}/{s}/{ph}']['metrics'][metric]['mean']) for s in STAGES)
                     + f" \\\\ % src: posthoc_v2.json rows.{arm}/<stage>/{ph}.metrics.{metric}")
    return L


def ladder_rotation_success(J, out):
    R = J["rows"]
    pm = J["pose_metric"]
    n_ep = R["static"]["n_episodes"]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table*}[t]",
         "\\caption{Post hoc, not pre-specified: rotation error, success and final match score on the "
         "crossed episodes of Table~\\ref{tab:ladder}. All rows use the 6 held-out patients, contact on and "
         f"evaluation DR off, with {n_ep} episodes per reference row and 3 seeds per policy row. Rotation error: the geodesic angle between "
         "the final and target probe orientations, the norm of the SO(3) logarithm of the relative rotation. "
         "Success: the share of episodes that end within both bounds, $\\leq$5~mm and $\\leq$5$^{\\circ}$ / "
         "$\\leq$10~mm and $\\leq$10$^{\\circ}$. Final match score $S$: the combined score of the arm at the last "
         "step. Each cell is the mean over patients of per-patient means; rotation contrasts are in "
         "Tables~\\ref{tab:imitation-base} and~\\ref{tab:rl-tost}. The translation error of Table~\\ref{tab:ladder} is the norm of the "
         "translational part of the SE(3) logarithm; it exceeds the probe-face displacement by at most "
         f"{pm['max_overstatement_bound_pct']:.1f}\\% at the largest logged angle.}}",
         "\\label{tab:ladder-rotation-success}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}l cccc@{}}",
         "\\toprule",
         " & " + " & ".join(NAME[s] for s in STAGES) + " \\\\",
         "\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{Final rotation error (deg)}} \\\\"]
    L += ph_grid(J, "final_ang", lambda x: f"{x:.2f}")
    L += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Success (\\%), $\\leq$5~mm and $\\leq$5$^{\\circ}$ / "
          "$\\leq$10~mm and $\\leq$10$^{\\circ}$}} \\\\"]
    L += [ln.replace(".metrics.pair", ".metrics.succ_5mm_5deg|succ_10mm_10deg")
          for ln in ph_grid({"rows": {k: {"metrics": {"pair": {"mean": (v["metrics"]["succ_5mm_5deg"]["mean"],
                                                                         v["metrics"]["succ_10mm_10deg"]["mean"])}}}
                                      for k, v in R.items()}}, "pair",
                            lambda x: f"{100 * x[0]:.1f} / {100 * x[1]:.1f}")]
    L += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Final match score $S$}} \\\\"]
    L += ph_grid(J, "final_match", lambda x: f"{x:.3f}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    log("\n== tab:ladder-rotation-success (ladder_rotation_success.tex), POST HOC ==")
    for k in ("static", "scripted", "pose_oracle") + tuple(f"{a}/{s}/{p}" for a, _ in PH_ROWS for s in STAGES
                                                          for p, _ in PH_PH):
        m = R[k]["metrics"]
        log(f"{k:32s} rot {m['final_ang']['mean']:.2f} | succ 5/5 {100 * m['succ_5mm_5deg']['mean']:.1f}% | "
            f"succ 10/10 {100 * m['succ_10mm_10deg']['mean']:.1f}% | S {m['final_match']['mean']:.3f}")
    return write(out, "ladder_rotation_success.tex", "\n".join(L))


def imitation_base(J, out):
    pos = J["contrasts_posthoc"]["final_pos"]["PH_imitation_minus_base"]["items"]
    ang = J["contrasts_posthoc"]["final_ang"]["PH_imitation_minus_base"]["items"]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table}[t]",
         "\\caption{Post hoc: after imitation minus the servo base alone, per configuration and texture stage (6 held-out "
         "patients, 3 seeds; contact on, evaluation DR off). Mean of per-patient paired differences, the number of "
         "patients (of 6) that share its sign, the 95\\% patient-cluster bootstrap CI and the paired $t$ CI. "
         "$^{\\ast}$: a difference under the CI-and-sign rule, applied here post hoc. Positive: imitation ends "
         "farther from the target than the base. The CT-mask base is the same for every stage.}",
         "\\label{tab:imitation-base}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}ll cccc cccc@{}}",
         "\\toprule",
         " & & \\multicolumn{4}{c}{Translation (mm)} & \\multicolumn{4}{c}{Rotation (deg)} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}",
         "Mask & Stage & Mean & $k$/6 & Bootstrap CI & $t$ CI & Mean & $k$/6 & Bootstrap CI & $t$ CI \\\\",
         "\\midrule"]
    log("\n== tab:imitation-base (imitation_base.tex), POST HOC ==")
    for i, (arm, mlab) in enumerate((("ctmask", "CT"), ("matched", "Learned"))):
        if i:
            L.append("\\addlinespace")
        for j, s in enumerate(STAGES):
            a, b = pos[f"{arm}/{s}"], ang[f"{arm}/{s}"]
            star_ = lambda e: "$^{\\ast}$" if e["holds_rule"] else ""
            L.append(f"{mlab if j == 0 else ''} & {NAME[s]} & {sg(a['mean'])}{star_(a)} & {a['sign']}/{a['n']} & "
                     f"{ci_txt(a['ci_boot'])} & {ci_txt(a['ci_t'])} & {sg(b['mean'])}{star_(b)} & {b['sign']}/{b['n']} & "
                     f"{ci_txt(b['ci_boot'])} & {ci_txt(b['ci_t'])} \\\\ % src: posthoc_v2.json "
                     f"contrasts_posthoc.final_pos|final_ang.PH_imitation_minus_base.items.{arm}/{s}")
            log(f"{arm}/{s}: translation {a['mean']:+.3f} mm {a['sign']}/{a['n']} boot [{a['ci_boot'][0]:+.2f}, "
                f"{a['ci_boot'][1]:+.2f}] rule {a['holds_rule']}; rotation {b['mean']:+.3f} deg {b['sign']}/{b['n']} "
                f"rule {b['holds_rule']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "imitation_base.tex", "\n".join(L))


def ladder_organ(J, out):
    R = J["rows"]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table*}[t]",
         "\\caption{Post hoc, not pre-specified: the oracle ladder of Table~\\ref{tab:ladder} by target "
         "organ. Each cell gives liver / kidney: the mean over the 6 held-out patients of per-patient means over that "
         "organ's episodes (5 acoustic windows per patient and organ, 3 seeds per policy row). Contact on, evaluation "
         "DR off.}",
         "\\label{tab:ladder-organ}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}l cccc@{}}",
         "\\toprule",
         " & " + " & ".join(NAME[s] for s in STAGES) + " \\\\"]
    for metric, head in (("final_pos", "Final translation error (mm), liver / kidney"),
                         ("final_ang", "Final rotation error (deg), liver / kidney")):
        L += ["\\midrule", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\"]

        def cell(k, metric=metric):
            po = R[k]["per_organ"]
            return f"{po['liver'][metric]['mean']:.2f} / {po['kidney'][metric]['mean']:.2f}"
        for key, lab in PH_REF:
            L.append(f"{lab} & \\multicolumn{{4}}{{c}}{{{cell(key)}}} \\\\ % src: posthoc_v2.json rows.{key}.per_organ")
        for arm, head2 in PH_ROWS:
            L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{head2}}} \\\\")
            for ph, lab in PH_PH:
                L.append(f"\\quad {lab} & " + " & ".join(cell(f"{arm}/{s}/{ph}") for s in STAGES)
                         + f" \\\\ % src: posthoc_v2.json rows.{arm}/<stage>/{ph}.per_organ")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    log("\n== tab:ladder-organ (ladder_organ.tex), POST HOC ==")
    for k in ("static", "pose_oracle", "ctmask/physics/base") + tuple(f"matched/{s}/final" for s in STAGES):
        po = R[k]["per_organ"]
        log(f"{k:28s} liver {po['liver']['final_pos']['mean']:.2f} mm / {po['liver']['final_ang']['mean']:.2f} deg | "
            f"kidney {po['kidney']['final_pos']['mean']:.2f} mm / {po['kidney']['final_ang']['mean']:.2f} deg")
    return write(out, "ladder_organ.tex", "\n".join(L))


def stats_sensitivity(J, out):
    Q = J["contrasts_q2"]
    sim = J["rule_simulation"]
    sc = sim["scenarios"]
    ha = J["contrasts_q2_holm_all"]
    # the printed family names (the question codes do not print)
    fam_lab = dict(zip(("Q2a_observability_gap", "Q2b_perception_gap", "Q2c_rl_contribution", "Q2d_ctmask_pairs",
                        "Q2e_learned_sonospade_vs"),
                       ("CT-mask final minus pose oracle (oracle-to-CT-mask gap)",
                        "Learned-mask final minus CT-mask final (CT-to-learned-mask gap)",
                        "Final minus after imitation (RL contribution)", "CT-mask final, first stage minus second",
                        "Learned-mask final, SonoSPADE minus stage")))

    def item_lab(fid, n):
        if fid == "Q2d_ctmask_pairs":
            return " minus ".join(NAME[x] for x in n.split("-"))
        if "/" in n:
            return ("CT mask, " if n.startswith("ctmask/") else "learned, ") + NAME[n.split("/")[1]]
        return NAME[n]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table*}[t]",
         "\\caption{Post hoc statistics sensitivity for every pre-specified ladder contrast (final translation error, mm; 6 "
         "held-out patients, 3 seeds). Mean: mean of per-patient paired differences. Bootstrap: the pre-specified 95\\% "
         f"patient-cluster percentile CI ({sim['nboot']:,} resamples). $t$: 95\\% CI and two-sided $p$ of the paired "
         "$t$-test (5 df). Patient $\\times$ seed: 95\\% crossed bootstrap CI that resamples patients and seeds "
         "independently, so seed variance enters. Exact: two-sided sign-flip $p$ over all $2^6$ sign patterns. "
         "Holm: adjusted within each family (block); over all "
         f"{ha['m']} contrasts as one family, {ha['n_survive_signflip']} survive on the exact $p$ and "
         f"{ha['n_survive_t']} on the $t$ $p$. Rule: the pre-specified CI-and-sign rule (yes = a difference). At $n = 6$ "
         "the smallest two-sided exact $p$ is $2/64 = 0.031$, above Holm's first threshold $0.05/m$ for every "
         # "in a family of two or more": the one-member realism family of Table S2.18 (tab:stats-layers;
         # posthoc_v2_layers.json families.Q4_realism, holm_signflip 0.03125) passes the exact test
         "family of $m \\geq 2$. No exact test on six patients therefore survives Holm in a family of two or more, "
         "and a 5/6 sign pattern has "
         f"exact $p \\geq 4/64$. Over {sim['nsim']:,} simulated null data sets of six differences, the rule declares a "
         f"difference in {100 * sc['null_normal']['rule']:.1f}\\% (normal), {100 * sc['null_t3']['rule']:.1f}\\% "
         f"($t$, 3 df) and {100 * sc['null_centered_exponential']['rule']:.1f}\\% (skewed) of cases. Under the "
         f"normal null, the $t$-test does so in {100 * sc['null_normal']['t_test_p_le_0.05']:.1f}\\% and the "
         f"exact test in {100 * sc['null_normal']['signflip_exact_p_le_0.05']:.1f}\\%.}}",
         "\\label{tab:stats-sensitivity}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
         "\\setlength{\\tabcolsep}{2.8pt}",
         "\\begin{tabular}{@{}l r c c c c c c c c c@{}}",
         "\\toprule",
         "Contrast & Mean & Bootstrap CI & $t$ CI & Patient $\\times$ seed CI & $k$/6 & Exact $p$ & Holm & "
         "$t$ $p$ & Holm & Rule \\\\",
         "\\midrule"]
    log("\n== tab:stats-sensitivity (stats_sensitivity.tex), POST HOC ==")
    for i, (fid, F) in enumerate(Q.items()):
        if i:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{fam_lab[fid]}, $m = {F['m']}$}}}} \\\\")
        for n, e in F["items"].items():
            cx = e.get("ci_crossed_patient_seed")
            L.append(f"\\quad {item_lab(fid, n)} & {sg(e['mean'])} & {ci_txt(e['ci_boot'])} & {ci_txt(e['ci_t'])} & "
                     + (ci_txt(cx) if cx else "n/a")
                     + f" & {e['sign']}/{e['n']} & {pfrac(e['p_signflip_exact'])} & {pdec(e['holm_signflip'])} & "
                     f"{pdec(e['p_t'])} & {pdec(e['holm_t'])} & {'yes' if e['holds_rule'] else 'no'} \\\\ "
                     f"% src: posthoc_v2.json contrasts_q2.{fid}.items.{n}")
            log(f"{fid}/{n}: {e['mean']:+.2f} boot [{e['ci_boot'][0]:+.2f}, {e['ci_boot'][1]:+.2f}] t [{e['ci_t'][0]:+.2f}, "
                f"{e['ci_t'][1]:+.2f}] k {e['sign']}/{e['n']} exact {e['p_signflip_exact']:.4f} holm {e['holm_signflip']:.4f} "
                f"t p {e['p_t']:.4g} holm t {e['holm_t']:.4g} rule {e['holds_rule']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "stats_sensitivity.tex", "\n".join(L))


def rule_simulation(J, out):
    sim = J["rule_simulation"]
    lab = {"null_normal": "Null, normal", "null_t3": "Null, $t$ (3 df)",
           "null_centered_exponential": "Null, centered exponential (skewed)"}
    L = [GEN.format(src=PH_SRC),
         "\\begin{table}[t]",
         # verified_against_collector_rule (the check against the analysis code) is read and logged below
         "\\caption{Post hoc simulation of the pre-specified CI-and-sign rule at $n = 6$ patients: the share of "
         f"{sim['nsim']:,} simulated data sets of six paired differences in which each procedure declares a "
         "difference (two-sided, $\\alpha = 0.05$). The rule is simulated as the analysis applies it: a 95\\% "
         f"percentile bootstrap with the same {sim['nboot']:,} fixed resamples, and at least 5 of 6 patients sharing "
         "the sign of the mean. Null rows give the type-I error; "
         "$\\delta$ rows give the power for normal differences with mean $\\delta$ standard deviations. The exact "
         "sign-flip test assumes a symmetric null, which the skewed row violates.}",
         "\\label{tab:rule-simulation}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\begin{tabular}{@{}lccccc@{}}",
         "\\toprule",
         "Scenario & Rule & CI excludes 0 & $k \\geq 5$/6 & Exact sign-flip & Paired $t$ \\\\",
         "\\midrule"]
    log("\n== tab:rule-simulation (rule_simulation.tex), POST HOC ==")
    log(f"checked against the collector's rule on {sim['verified_against_collector_rule']['n_checked']} data sets "
        "(" + _T("rule sim tag") + ")")
    for k, v in sim["scenarios"].items():
        name = lab.get(k, f"$\\delta = {v['delta_sd']:g}$ SD")
        L.append(f"{name} & {v['rule']:.3f} & {v['ci_excludes_0']:.3f} & {v['sign_at_least_5_of_6']:.3f} & "
                 f"{v['signflip_exact_p_le_0.05']:.3f} & {v['t_test_p_le_0.05']:.3f} \\\\ "
                 f"% src: posthoc_v2.json rule_simulation.scenarios.{k}")
        log(f"{k:26s} rule {v['rule']:.4f} ci {v['ci_excludes_0']:.4f} sign {v['sign_at_least_5_of_6']:.4f} "
            f"exact {v['signflip_exact_p_le_0.05']:.4f} t {v['t_test_p_le_0.05']:.4f}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "rule_simulation.tex", "\n".join(L))


def rl_tost(J, out):
    T = J["tost_rl"]
    rot = J["contrasts_posthoc_rotation"]["Q2c_rl_contribution"]["items"]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table}[t]",
         "\\caption{Post hoc equivalence test of the RL null on the eight trained policy configurations: REINFORCE "
         "fine-tuning at the pre-specified budget, final minus after imitation, 6 held-out patients, 3 seeds. Margin "
         f"$\\pm{T['margin_mm']:g}$~mm, the pre-specified RL threshold, applied as an equivalence margin after the "
         "results were known. TOST: two one-sided paired $t$-tests (5 df); equivalence if $p < 0.05$, that is, if "
         "the 90\\% $t$ CI lies inside the margin. Sign-flip: the exact version (smallest attainable $p$ is "
         "$1/64$). Patient $\\times$ seed: 90\\% crossed bootstrap CI with seed variance included. Rotation: the "
         "same contrast on the final rotation error (deg), with its 95\\% $t$ CI and sign consistency.}",
         "\\label{tab:rl-tost}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}llcccccc@{}}",
         "\\toprule",
         " & & \\multicolumn{4}{c}{Translation (mm)} & \\multicolumn{2}{c}{Rotation (deg)} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(l){7-8}",
         "Mask & Stage & Mean [90\\% $t$ CI] & TOST $p$ & Sign-flip $p$ & Patient $\\times$ seed 90\\% CI & "
         "Mean [95\\% $t$ CI] & $k$/6 \\\\",
         "\\midrule"]
    log("\n== tab:rl-tost (rl_tost.tex), POST HOC ==")
    for i, (arm, mlab) in enumerate((("ctmask", "CT"), ("matched", "Learned"))):
        if i:
            L.append("\\addlinespace")
        for j, s in enumerate(STAGES):
            e, r = T["items"][f"{arm}/{s}"], rot[f"{arm}/{s}"]
            cx = e["ci90_crossed_patient_seed"]
            L.append(f"{mlab if j == 0 else ''} & {NAME[s]} & {sg(e['mean'])} {ci_txt(e['ci90_t'])} & {pdec(e['p_tost_t'])} & "
                     f"{pfrac(e['p_tost_signflip'])} & {ci_txt(cx)} & {sg(r['mean'])} {ci_txt(r['ci_t'])} & "
                     f"{r['sign']}/{r['n']} \\\\ % src: posthoc_v2.json tost_rl.items.{arm}/{s}; "
                     f"contrasts_posthoc_rotation.Q2c_rl_contribution.items.{arm}/{s}")
            log(f"{arm}/{s}: {e['mean']:+.3f} mm, 90% t [{e['ci90_t'][0]:+.2f}, {e['ci90_t'][1]:+.2f}], TOST p "
                f"{e['p_tost_t']:.4g}, sign-flip {e['p_tost_signflip']:.4f}, crossed 90% [{cx[0]:+.2f}, {cx[1]:+.2f}]; "
                f"rotation {r['mean']:+.2f} deg [{r['ci_t'][0]:+.2f}, {r['ci_t'][1]:+.2f}] {r['sign']}/{r['n']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "rl_tost.tex", "\n".join(L))


def registration_crossed_tab(J, out):
    G = J["registration_crossed"]
    cells = G["cells"]
    groups = [("vitale_holdout", "liver", "Vitale holdout~\\cite{vitale}"),
              ("vitale_holdout", "kidney", "Vitale holdout~\\cite{vitale}"),
              ("aul", "liver", "AUL~\\cite{aul}"),
              ("kidneyus", "kidney", "KidneyUS holdout~\\cite{kidneyus}")]
    L = [GEN.format(src=PH_SRC),
         "\\begin{table*}[t]",
         "\\caption{Post hoc reanalysis of the offline registration of Table~\\ref{tab:registration}. Each real target "
         "is run with all 3 seeds and with 3 of the 6 held-out CTs (the same 3 for every seed). Keep-best gain: the "
         "pre-specified metric, keep-best minus static organ-restricted IoU. Final-step gain: final minus static IoU, with "
         "no selection. Targets: the pre-specified bootstrap over targets. Targets $\\times$ seeds and targets $\\times$ "
         "CTs $\\times$ seeds: crossed bootstrap CIs that resample each factor independently (the latter weights the "
         "observed cells, since each target meets 3 of the 6 CTs). With 3 seeds and 6 CTs these CIs are still "
         "optimistic. Holm: target-level paired $t$-test $p$, Holm-adjusted over the 16 cells. Subjects are not "
         "logged, so targets are resampled by frame, not clustered by subject. IoU: static / final / keep-best.}",
         "\\label{tab:registration-crossed}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
         "\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}l cccc c cc c@{}}",
         "\\toprule",
         " & \\multicolumn{5}{c}{Keep-best gain} & \\multicolumn{2}{c}{Final-step gain} & \\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}",
         "Stage & Mean & Targets & Targets $\\times$ seeds & Targets $\\times$ CTs $\\times$ seeds & Holm & Mean & "
         "Targets $\\times$ CTs $\\times$ seeds & IoU \\\\",
         "\\midrule"]
    log("\n== tab:registration-crossed (registration_crossed.tex), POST HOC ==")
    for gi, (src, organ, lab) in enumerate(groups):
        n = {cells[f"{s}/{src}/{organ}"]["n_targets"] for s in STAGES}
        if len(n) != 1:
            raise SystemExit(f"registration_crossed {src}/{organ}: target counts differ across stages")
        if gi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{lab}, {organ}, $n = {n.pop()}$ targets}}}} \\\\")
        for s in STAGES:
            c = cells[f"{s}/{src}/{organ}"]
            kb, fg = c["keepbest_gain"], c["final_gain"]
            L.append(f"\\quad {NAME[s]} & {sg(kb['mean'], 3)} & {ci_txt(kb['ci_targets'], 3)} & "
                     f"{ci_txt(kb['ci_crossed_targets_seeds'], 3)} & {ci_txt(kb['ci_crossed_targets_cts_seeds'], 3)} & "
                     f"{pdec(kb['holm_t'])} & {sg(fg['mean'], 3)} & {ci_txt(fg['ci_crossed_targets_cts_seeds'], 3)} & "
                     f"{c['static_iou']['mean']:.2f} / {c['final_iou']['mean']:.2f} / {c['keepbest_iou']['mean']:.2f} \\\\ "
                     f"% src: posthoc_v2.json registration_crossed.cells.{s}/{src}/{organ}")
            log(f"{s:9s} {src:14s} {organ:6s} kb {kb['mean']:+.3f} t[{kb['ci_targets'][0]:+.3f},{kb['ci_targets'][1]:+.3f}] "
                f"ts[{kb['ci_crossed_targets_seeds'][0]:+.3f},{kb['ci_crossed_targets_seeds'][1]:+.3f}] "
                f"tcs[{kb['ci_crossed_targets_cts_seeds'][0]:+.3f},{kb['ci_crossed_targets_cts_seeds'][1]:+.3f}] "
                f"holm {kb['holm_t']:.4f} | final {fg['mean']:+.3f}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "registration_crossed.tex", "\n".join(L))


def step_error(J, out):
    sc = J["step_curves"]
    steps = (0, 2, 4, 6, 8, 10, 12)
    rows = [("pose_oracle", "Analytic pose oracle (privileged)"), ("static", "Static start"),
            ("ctmask/physics/base", "CT mask, servo base (all stages)")]
    rows += [(f"ctmask/{s}/final", f"CT mask, {NAME[s]}, final") for s in STAGES]
    rows += [(f"matched/{s}/{p}", f"Learned, {NAME[s]}, {pl.lower()}") for s in STAGES for p, pl in (PH_PH[0], PH_PH[2])]
    if any(len(sc[k]["pos_mean_by_step"]) != 13 for k, _ in rows):
        raise SystemExit("step curves are not 13 points long")
    L = [GEN.format(src=PH_SRC + " step_curves"),
         "\\begin{table*}[t]",
         "\\caption{Post hoc: final translation error (mm) against control step on the crossed episodes of "
         "Table~\\ref{tab:ladder}, from the logged per-step trajectories. Each value is the mean over the 6 held-out "
         "patients of per-patient means, with contact on and evaluation DR off. Min: the lowest mean and its step. Improving: the share of "
         "episodes whose error fell on the last step. $S$: the mean match score at steps 0 and 12. Rotation error "
         "per step is not logged.}",
         "\\label{tab:step-error}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\footnotesize",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}l " + "c" * len(steps) + " ccc@{}}",
         "\\toprule",
         "Row & " + " & ".join(f"Step {k}" for k in steps) + " & Min (step) & Improving & $S$, step 0 / 12 \\\\",
         "\\midrule"]
    log("\n== tab:step-error (step_error.tex), POST HOC ==")
    for k, lab in rows:
        v = sc[k]
        mp = v["pos_mean_by_step"]
        L.append(f"{lab} & " + " & ".join(f"{mp[s]:.2f}" for s in steps)
                 + f" & {v['pos_min']:.2f} ({v['pos_min_step']}) & {100 * v['frac_episodes_improving_at_last_step']:.0f}\\% & "
                 f"{v['S_start']:.2f} / {v['S_final']:.2f} \\\\ % src: posthoc_v2.json step_curves.{k}")
        log(f"{k:28s} " + " ".join(f"{mp[s]:.2f}" for s in steps) + f" | min {v['pos_min']:.2f} at {v['pos_min_step']} | "
            f"last deltas " + ", ".join(f"{x:+.2f}" for x in v["pos_last4_deltas"]))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "step_error.tex", "\n".join(L))


def seg_sim_real(J, out):
    S = J["seg_dice_sim_vs_real"]["per_stage_seed"]
    # per-organ simulated columns (posthoc_v2.json keys sim_heldout_liver and sim_heldout_kidney) beside the organ
    # mean, so simulated and real Dice can be read organ by organ
    cols = [("sim_heldout_mean", "Mean over organs"), ("sim_heldout_liver", "Liver"),
            ("sim_heldout_kidney", "Kidney"),
            ("real_vitale_holdout_liver", "Vitale, liver"), ("real_aul_liver", "AUL, liver"),
            ("real_kidneyus_kidney", "KidneyUS, kidney")]
    L = [GEN.format(src=PH_SRC + " seg_dice_sim_vs_real (from the segmenter repository's eval_<stage>_s<seed>.json)"),
         "\\begin{table}[t]",
         "\\caption{Post hoc: Dice of each stage's texture-matched segmenter on held-out simulated frames "
         "(renders of the 6 evaluation CTs, never seen in training, each with its own domain-randomization draw, "
         "not the default evaluation render) and on real holdout frames, seeds 0 / 1 / 2. "
         # AUL is in no segmenter's training (Supplementary S5.1 and S5.3)
         "Vitale-holdout and KidneyUS frames come from datasets whose training splits the segmenters saw; AUL frames "
         "come from a dataset that no segmenter saw. "
         "Compare with the learned-mask rows of Table~\\ref{tab:ladder}.} % src: Supplementary S5.1 (sections/"
         f"supp_eval_details.tex:83, the real training frames) and S5.3 (:152){_T('seg sim note')}",
         "\\label{tab:seg-sim-real}",
         "\\centering",
         "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}",
         "\\toprule",
         " & \\multicolumn{3}{c}{Simulated, held-out CTs} & \\multicolumn{3}{c}{Real holdout frames} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
         "Stage & " + " & ".join(c for _, c in cols) + " \\\\",
         "\\midrule"]
    log("\n== tab:seg-sim-real (seg_sim_real.tex), POST HOC harvest ==")
    for s in STAGES:
        seeds = sorted(S[s])
        if seeds != ["s0", "s1", "s2"]:
            raise SystemExit(f"seg_sim_real {s}: seeds {seeds}")

        def cell(k, s=s, seeds=seeds):
            return " / ".join(f"{S[s][x][k]:.2f}" for x in seeds)
        L.append(f"{NAME[s]} & " + " & ".join(cell(k) for k, _ in cols)
                 + f" \\\\ % src: posthoc_v2.json seg_dice_sim_vs_real.per_stage_seed.{s}")
        log(f"{s:9s} " + " | ".join(f"{k} {cell(k)}" for k, _ in cols))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "seg_sim_real.tex", "\n".join(L))


# =================================================================================== pluggability (v2.1)
# tab:pluggability (pluggability.tex), input by sections/10_pluggability.tex.
# Each slot holds published methods beside the authors' component, its reference row. Inputs: the v2.1 plug outputs
# (outputs/v2_1/plug/<slot>/ and the segmenter repository's outputs/v2_1/plug/, as queued in
# outputs/v2_1/jobs_final_r3.json); for the authors' rows, the archived v2 files behind RESULTS.json (checked
# against RESULTS.json table4.cell_means_mm). Exploratory (amendment 1): no v2 table changes, and every plug file
# carries analysis_status "exploratory, v2.1 pluggability".
# Statistics (amendment 1): each row minus the reference, per patient and paired by seed; interval: 95%
# paired t CI with 5 degrees of freedom; test: two-sided paired t with Holm over every member of the slot's family
# (a member without a complete result counts as p = 1, so a printed Holm p is an upper bound until all members are
# in); a row holds when Holm p < alpha and at least 5 of 6 patients share the sign. Rows outside the family are
# "secondary".
# Later groups use the families of amendment 2 (MASK-P, IMN-T, LPIPS-R; the CT-anatomy block CTA-D and CTA-E), with
# Holm within each family and never pooled with the families of amendment 1 (PLUG_FAMILY_M). Each row's src comment
# and the log also give the sign-flip p (exact) and the bootstrap CI over patients.
PLUG_ROOT = os.path.join(ROOT, "outputs", "v2_1", "plug")
SEG_PLUG_ROOT = os.path.join(SEG, "outputs", "v2_1", "plug")
CHECKS_ROOT = os.path.join(ROOT, "outputs", "v2_1", "checks")
JOBS_FINAL = os.path.join(ROOT, "outputs", "v2_1", "jobs_final_r3.json")
PLUG_STATUS = "exploratory, v2.1 pluggability"
PLUG_SEEDS = (0, 1, 2)
SEED0 = (0,)
PLUG_BOOT = 10000
# A finished v2.1 value is provisional until its identical-episode audit and the aggregation at the v2.1 tag are
# done. Both are done: every audit passed (outputs/v2_1/analysis_plan/pluggability.json checks.<slot>.audit), and the
# tag aggregation recomputed all 67 Table 5 rows (pluggability.json comparison_with_plug_rows: rows_equal 67, findings
# []). The CT-anatomy block and the post hoc columns are outside that aggregation (pluggability.json not_covered).
PLUG_PROVISIONAL = False
# The flag as recorded in figures/plug_rows.json (top-level "provisional" and each row's "provisional"): that file is
# the md5-pinned snapshot amendment 5's rule outcome refers to (PLUG_ROWS_MD5; amend5_combination.json final_table5), so
# its flags keep the value they were written with. Never changed; the printed tables follow PLUG_PROVISIONAL.
PLUG_ROWS_PROVISIONAL = True
# The section-sign mark does not print, whatever PLUG_PROVISIONAL says; the provisional status is kept in
# plug_rows.json ("provisional") and in the \verifyfill boxes, which main.tex prints plain by default.
PROV_MARK = "$^{\\S}$"   # the mark of earlier builds (kept for reference only)
PROV = ""
AST = "$^{\\ast}$"
# Blinding of the amendment 2 groups: no output or log of a job named with imnet, samus, lpips or ctanat is read.
# Their rows print as one line per group; _amend2() still loads, hashes and asserts the Appendix constants (from the
# constants and the queue only). --read-amend2 lifts the blinding.
BLIND_A2 = True
A2_BLIND_KEYS = ("imnet", "samus", "lpips", "ctanat")
A2_BLIND_TXT = "not read in this build (blinded)"
# Blinding of the amendment 3, 4, 5 and 6 outputs (each amendment's blinding rule; amendment 5's and amendment 6's check
# records check that the line "BLIND_A5 = True", or "BLIND_A6 = True", stands here at HEAD and in the working copy):
# no path or job name carrying the amendment's key (outputs/v2_1/amend3, amos and amos_x, amend5, amend6, and their
# analysis files) is read while its flag is on. Each flag is lifted only by its own option (--read-amend3 to
# --read-amend6), once that amendment's analysis has run.
BLIND_A3 = True
BLIND_A4 = True
BLIND_A5 = True
BLIND_A6 = True
A345_BLIND_KEYS = {"BLIND_A3": ("amend3",), "BLIND_A4": ("amos",), "BLIND_A5": ("amend5",), "BLIND_A6": ("amend6",)}
REFS_TEX = os.path.join(PAPER, "sections", "refs.tex") if PAPER else None
# Family sizes of amendment 1: the published methods of each slot; a row list of another size stops the
# build.
PLUG_FAMILY_M = {"texture": 4, "mask": 7, "ctrl": 10, "reward": 2, "renderer": 2}
V2_REAL_NAME = {"physics": "physics", "cut": "physics+CUT", "cyclegan": "physics+CycleGAN",
                "sonospade": "physics+SonoSPADE"}
ROLE_TXT = {"ref": "reference", "family": "family", "secondary": "secondary", "a2": "amend2"}

# =================================================================================== dated amendment 2
# Amendment 2 adds families for four later groups: the CT-anatomy slot (TotalSegmentator v2 labels), SAMUS, the
# torchvision segmenters with backbones pretrained on ImageNet, and LPIPS; never pooled with the families of
# amendment 1. _amend2() loads the Appendix block (plan_inputs.amendment2_appendix), checks its digest against
# AMEND2_SHA256 and asserts every constant below (and every file pattern against the queue) before pluggability()
# reads any output of these groups. Only the ladder and train result files that the Appendix names enter a family;
# no training history, log or checkpoint of these groups is opened, and the SAMUS segeval outputs are not opened.
AMEND2_MD = os.path.join(CODE_ROOT, PP.AMEND2_MD_REL) if PP else None          # development repository only
AMEND2_SHA_FILE = os.path.join(CODE_ROOT, PP.AMEND2_SHA_REL) if PP else None
AMEND2_TXT = PI.text("amend2 file")              # how the tables' source comments name amendment 2
AMEND2_SHA_TXT = PI.text("amend2 sha file")
AMEND2_SHA256 = "8daa46d2071952d845b2282684f4645bf660143a6382f2f296f3c8ca6fe5900d"
AMEND2_FAMILY_M = {"CTA-D": 5, "CTA-E": 5, "MASK-P": 5, "IMN-T": 4, "LPIPS-R": 1}
AMEND2_DECISION = {"test": "paired_t", "df": 5, "sides": "two", "holm": True, "min_sign_agree": 5}
AMEND2_ALPHA = 0.05
AMEND2_MISSING_P = 1.0      # an incomplete member enters Holm at p = 1, so m does not change
PLUG_BOOT_SEED = 0          # the seed of _plug_contrast's patient bootstrap (C.boot_ci(..., seed=0))
SAMUS_BOX_MIN_PX, SAMUS_MAX_BOXES = 20, 5     # recorded by the adapter (masks.boxes_from_mask); asserted, not used here
IMNET_ITERS, IMNET_BATCH, IMNET_WEIGHTS = 4000, 8, "IMAGENET1K_V1"   # the queued train jobs' recipe; asserted
LPIPS_W_SEG, LPIPS_W_APP, LPIPS_ENV_STEPS = 0.6, 0.4, 57600
# CT-anatomy slot: rows R1 to R5 compare each rung on TotalSegmentator labels (CTA_NEW_ROOT) with the same rung on
# the reference labels (CTA_REF_ROOT, the v2.1 rerun), seeds 0 to 2, contact on, DR off; R4 and R5 use final weights.
CTA_NEW_ROOT = os.path.join(ROOT, "outputs", "v2_1", "plug", "ctanat", "fast")
CTA_REF_ROOT = os.path.join(ROOT, "outputs", "v2_1", "s21_rerun")
CTA_SEEDS = (0, 1, 2)
CTA_ROWS = [  # (row, file pattern under both roots, weights of the contact-on block, label, privileged)
    ("R1", "ladder/ibvs_base_physics_ct_s{seed}.json", "zero_residual", "CT-mask servo base, physics", True),
    ("R2", "ladder/ibvs_base_physics_matched_s{seed}.json", "zero_residual", "Learned-mask servo base, physics", False),
    ("R3", "ladder/ibvs_base_sonospade_matched_s{seed}.json", "zero_residual",
     "Learned-mask servo base, per-tissue stage (SonoSPADE)", False),
    ("R4", "contact/ctmask_physics_s{seed}.json", "final", "CT-mask policy, physics, final weights", True),
    ("R5", "contact/ctmask_sonospade_s{seed}.json", "final", "CT-mask policy, SonoSPADE, final weights", True)]
CTA_BLOCK = {"contact": "on", "eval_randomize": "off", "policy_weights": "final"}
CTA_INVARIANCE = {"file": "ladder/ibvs_base_sonospade_ct_s{seed}.json", "equals": "R1", "tol_mm": 1e-6}
CTA_METRIC = "final_pos"
CTA_TOST_MARGIN = 1.0       # CTA-E: equivalence margin (mm); Holm on the TOST p; the 90% t CI prints
CTA_TOST_CI = 0.90
CTA_MODE, CTA_RES_MM = "fast", 3.0
CTA_SETUP_TOL, CTA_POSE_TOL = 1e-9, 1e-6     # the evaluations' own identity checks; asserted, applied by the jobs
CTA_REF_STATUS = "exploratory, v2.1"         # analysis_status of the v2.1 reruns (the reference rungs)
# The pose-oracle and static rungs on TotalSegmentator labels must give the same episodes as on the reference labels;
# if not, no CT-anatomy row is reported. The names are asserted against the queue.
CTA_PREMISE = ("ladder/pose_oracle_physics_ct.json", "ladder/static_physics_ct.json",
               "ladder/pose_oracle_sonospade_ct.json", "ladder/static_sonospade_ct.json")
CTA_SUMMARY = "summary.json"   # scripts/plug/totalseg_labels.py analyze (rows[].metrics.final_pos.per_patient_diff)
CTA_IN_SAMPLE = ("All 19 CTs are in TotalSegmentator's own training data, so these labels are in-sample predictions: the "
              "rows are a label-noise sensitivity check, not a test of an independent CT segmenter.")
# TotalSegmentator's 1.5 mm default mode ran under amendment 6; its evaluation set did not pass the lineage rule of
# the leakage check (lineage incomplete, no leak found), so Table S3.1 lists it as not reported; CUT, CycleGAN and the
# cohort were not run in that mode
CTA_NOT_REPORTED = "TotalSegmentator's 1.5~mm default mode."
CTA_NOT_REPORTED_SRC = ("outputs/v2_1/analysis_plan/amend6_td.json completion.parts.td.state refused (checked "
                        "before this line is written when amendment 6 is read, and again by _est_check); "
                        f"{_T('a6 record file')} {_T('a6 leak part')}leakage refusals")


def _cta_not_reported(mc):
    """The Table S3.1 line of the CT-anatomy part that ran and is not reported (amendment 6 TD). When amendment 6 is
    read, its TD part must still be refused before the line is written (the twin is
    written before est_tables() runs _est_check)."""
    p = os.path.join(AP_DIR, "amend6_td.json")
    if _blinded_a345(p) is None:
        td = _a345("amend6_td.json")["completion"]["parts"]["td"]
        if td.get("state") != "refused" or td.get("completed"):
            raise SystemExit(f"{rel(p)}: the TD part is {td.get('state')!r}, but Table S3.1 lists it as not "
                             "reported")
    return f"{mc}{{\\quad\\textit{{Not reported.}} {CTA_NOT_REPORTED}}} \\\\ % src: {CTA_NOT_REPORTED_SRC}"


# Groups queued after amendment 1. While every row of a group is pending, the group prints as one line.
PLUG_GROUPS = {
    "samus": ("MASK-P ($m = 5$)", "SAMUS~\\cite{lin2024samus}, boxes from the U-Net mask; boxes from the CT "
              "label$^{\\dagger}$, secondary"),
    "imnet": ("MASK-P ($m = 5$) and IMN-T ($m = 4$)", "the four torchvision networks with ImageNet-pretrained "
              "backbones"),
    "lpips": ("LPIPS-R ($m = 1$)", "LPIPS appearance term~\\cite{zhang2018lpips}"),
}
# Plain names that Table 5 of the article prints for the amendment 2 families and the CT-anatomy rows (no codes in
# the tables and figures of the article). The codes stay in the supplementary twin
# (tables/pluggability_twin.tex, PLUG_GROUPS above), in every src comment and in Appendix C, whose key line
# (sections/appendix_decisions.tex) maps each family name to its code and whose CT-anatomy items name each row's key.
# The families are named by the codes that Appendix B defines at first use (only the pending lines of a blinded build
# print these names; Table 5 uses T5_FAMILY).
A2_PLAIN = {"MASK-P": "mask family MASK-P", "IMN-T": "pretrained-backbone family IMN-T",
            "LPIPS-R": "reward family LPIPS-R", "CTA-D": "anatomy difference family CTA-D",
            "CTA-E": "anatomy equivalence family CTA-E"}
# the rung names of the Appendix C.6 item labels (without "privileged"), so each Table 5 name finds its key there
# (one name per stage: SonoSPADE in the printed row names)
CTA_PLAIN = dict(zip(("R1", "R2", "R3", "R4", "R5"),
                     ("CT-mask servo base, physics", "learned-mask servo base, physics",
                      "learned-mask servo base, SonoSPADE", "CT-mask policy, physics, final weights",
                      "CT-mask policy, SonoSPADE, final weights")))


def _a2_plain(fams):
    """The plain names of amendment 2 family codes, joined for one printed phrase."""
    return " and the ".join(A2_PLAIN[f] for f in fams)


def _cap1(s):
    return s[:1].upper() + s[1:]


_AMEND2 = {}
CTA_STATE = {}   # the CT-anatomy block for the body table (blinded flag, reference rungs, contrasts)


def _pr(rid, label, cites=(), role="family", priv=False, rkey=None, v2=False, seeds=PLUG_SEEDS, learned=True,
        bn=None, group=None, fam2=(), twin=None, v21=False, mdd=True, laccess=False, xref=None):
    """One table row.
    role: ref (the slot's reference), family (amend1 family member), secondary (authors' own, prior, privileged or
    optional rows) or a2 (a member of an amendment 2 family, named in fam2; IMN-T rows also name their from-scratch
    twin, an amendment 1 row of the same slot).
    v2: the row reads the archived v2 files (else the v2.1 plug outputs). seeds: the seeds of the row's design
    (seed 0 only for the inference-only SAM-family and SAMUS rows, the look-ahead, SIMUS and the optional 20-epoch
    nnU-Net). learned: a trained method, whose difference is left blank with fewer than three seeds.
    bn: the BatchNorm mode of the primary block (BC and DAgger: eval, amend1).
    v21: a v2.1 file that is not a plug output (status "exploratory, v2.1"), so its cells are provisional.
    mdd: the row enters the slot's post hoc MDD as a method. False for a check of the reference artifact
    and for the rows amend1 keeps off the cohort, whose MDD is the one fixed beforehand (SAM and MedSAM, the CEM
    look-ahead, SIMUS, the optional 20-epoch nnU-Net) and for the privileged CT label of the mask slot.
    laccess: the method reads the CT label slice (Table 5 marks it L).
    xref: (file under ROOT, key path) of a plan-conformant analysis value the row's error must equal (1e-9)."""
    return {"id": rid, "label": label, "cites": list(cites), "role": role, "priv": priv, "rkey": rkey, "v2": v2,
            "seeds": tuple(seeds), "learned": learned, "bn": bn, "group": group, "fam2": tuple(fam2), "twin": twin,
            "v21": v21, "mdd": mdd, "laccess": laccess, "xref": xref}


# The label-painted physics stage (Q10(b), a control of the authors that learns nothing): its learned-mask servo
# base (seeds 0 to 2, the stage's own matched segmenter and calibration seed) is a secondary row of the texture slot,
# and its error must equal the Q10 analysis value.
PAINTED_CL = os.path.join(ROOT, "outputs", "v2_1", "label_controls", "closed_loop",
                          "ibvs_base_label_painted_lit_matched_s{seed}.json")
PAINTED_REAL = os.path.join(ROOT, "outputs", "v2_1", "label_controls", "realism_label_painted.json")
PAINTED_XREF = ("outputs/v2_1/analysis_plan/q10_label_controls_v2.json", ("results", "painting", "patient_means", "P"))


PLUG_SLOTS = [
    {"key": "texture", "title": "Texture stage", "iface": "TextureTranslator",
     "setting": "learned-mask servo base; score KID; cost ms per frame, translator call only, MPS benchmark",
     "rows": [_pr("sonospade", "SonoSPADE (authors')", ["sonospade"], "ref", rkey="matched/sonospade/base", v2=True,
                  laccess=True),
              _pr("physics", "None, physics frame (authors')", [], "secondary", rkey="matched/physics/base", v2=True),
              _pr("cut", "CUT (prior)", ["cut"], "secondary", rkey="matched/cut/base", v2=True),
              _pr("cyclegan", "CycleGAN (prior)", ["cyclegan"], "secondary", rkey="matched/cyclegan/base", v2=True),
              _pr("label_painted", "Label-painted physics stage (authors', not learned)", [], "secondary", v21=True,
                  mdd=False, laccess=True, xref=PAINTED_XREF),
              _pr("fastcut", "FastCUT", ["cut"]),
              _pr("dclgan", "DCLGAN", ["han2021dclgan"]),
              _pr("unsb", "UNSB (Schr\\\"{o}dinger bridge)", ["kim2024unsb"]),
              _pr("histmatch", "Histogram matching", ["gonzalez2018"])],
     "not_run": "Echo from Noise~\\cite{stojanovski2023echo} and SDM~\\cite{zhou2022sdm}: seconds per frame, no "
                "public abdominal weights. ConPres~\\cite{tomar2021conpres}: no public code."},
    {"key": "mask", "title": "Mask source", "iface": "MaskSource",
     "setting": "servo base per mask, SonoSPADE stage; score liver Dice, Vitale holdout; cost ms per step, mask call "
                "timed inside the closed loop, MPS",
     "rows": [_pr("unet", "U-Net, stage-matched (authors')", ["ronneberger2015unet"], "ref",
                  rkey="matched/sonospade/base", v2=True),
              _pr("ct", "CT label (authors')", [], "secondary", priv=True, rkey="ctmask/sonospade/base", v2=True,
                  mdd=False),
              _pr("lraspp_mbv3", "LR-ASPP, MobileNetV3-L", ["howard2019mobilenetv3"]),
              _pr("deeplabv3_mbv3", "DeepLabV3, MobileNetV3-L", ["chen2017deeplabv3", "howard2019mobilenetv3"]),
              _pr("deeplabv3_r50", "DeepLabV3, ResNet-50", ["chen2017deeplabv3", "he2016resnet"]),
              _pr("fcn_r50", "FCN, ResNet-50", ["long2015fcn", "he2016resnet"]),
              _pr("nnunet2d", "nnU-Net 2D, 2 epochs", ["isensee2021nnunet"]),
              _pr("sam_unet", "SAM, boxes from the U-Net mask", ["kirillov2023sam"], seeds=SEED0, learned=False,
                  mdd=False),
              _pr("medsam_unet", "MedSAM, boxes from the U-Net mask", ["ma2024medsam"], seeds=SEED0, learned=False,
                  mdd=False),
              _pr("nnunet2d_e20", "nnU-Net 2D, 20 epochs (optional)", ["isensee2021nnunet"], "secondary",
                  seeds=SEED0, mdd=False),
              _pr("sam_priv", "SAM, boxes from the CT label", ["kirillov2023sam"], "secondary", priv=True,
                  seeds=SEED0, learned=False, mdd=False),
              _pr("medsam_priv", "MedSAM, boxes from the CT label", ["ma2024medsam"], "secondary", priv=True,
                  seeds=SEED0, learned=False, mdd=False),
              # amendment 2: SAMUS with U-Net boxes, seed 0, in MASK-P; the CT-box variant is
              # privileged (secondary); the four ImageNet rows in MASK-P (minus the U-Net) and IMN-T (minus the twin)
              _pr("samus_unet", "SAMUS, boxes from the U-Net mask", ["lin2024samus"], "a2", seeds=SEED0,
                  learned=False, group="samus", fam2=("MASK-P",)),
              # mdd=False, as for the other CT-label rows; no amendment 2 row enters the post hoc MDD (_slot_mdd also
              # drops every row of an amendment 2 group)
              _pr("samus_priv", "SAMUS, boxes from the CT label", ["lin2024samus"], "secondary", priv=True,
                  seeds=SEED0, learned=False, group="samus", mdd=False),
              _pr("lraspp_mbv3_imnet", "LR-ASPP, MobileNetV3-L, ImageNet backbone", ["howard2019mobilenetv3"],
                  "a2", group="imnet", fam2=("MASK-P", "IMN-T"), twin="lraspp_mbv3"),
              _pr("deeplabv3_mbv3_imnet", "DeepLabV3, MobileNetV3-L, ImageNet backbone",
                  ["chen2017deeplabv3", "howard2019mobilenetv3"], "a2", group="imnet", fam2=("MASK-P", "IMN-T"),
                  twin="deeplabv3_mbv3"),
              _pr("deeplabv3_r50_imnet", "DeepLabV3, ResNet-50, ImageNet backbone", ["chen2017deeplabv3",
                                                                                     "he2016resnet"],
                  "a2", group="imnet", fam2=("MASK-P", "IMN-T"), twin="deeplabv3_r50"),
              _pr("fcn_r50_imnet", "FCN, ResNet-50, ImageNet backbone", ["long2015fcn", "he2016resnet"], "a2",
                  group="imnet", fam2=("MASK-P", "IMN-T"), twin="fcn_r50")],
     "not_run": _T("masks not run")},
    {"key": "ctrl", "title": "Controller", "iface": "USBGym/ViewMatch-v0",
     "setting": "SonoSPADE, CT mask$^{\\dagger}$; score success at 10~mm and 10$^\\circ$; cost training wall-clock "
                "hours on MPS (h), or evaluation seconds per episode (/ep)",
     "rows": [_pr("reinforce", "REINFORCE residual, imitation start (authors')", ["reinforce"], "ref",
                  rkey="ctmask/sonospade/final", v2=True),
              _pr("servo_base", "Servo base alone (authors')", [], "secondary", rkey="ctmask/sonospade/base", v2=True),
              # the reference policies re-evaluated with BatchNorm in eval mode (the clean-tag rerun); no citation, so the
              # label keeps to one line (the reference row above cites REINFORCE)
              _pr("reinforce_bneval", "REINFORCE residual, BatchNorm in eval mode (authors')", [],
                  "secondary", v21=True, mdd=False),
              _pr("ppo_res", "PPO, residual", ["schulman2017ppo", "sb3"]),
              _pr("sac_res", "SAC, residual", ["haarnoja2018sac", "sb3"]),
              _pr("td3_res", "TD3, residual", ["fujimoto2018td3", "sb3"]),
              _pr("a2c_res", "A2C, residual", ["mnih2016a3c", "sb3"]),
              _pr("ppo_scratch", "PPO, full twist", ["schulman2017ppo", "sb3"]),
              _pr("dqn_scratch", "DQN, 13 discrete moves", ["mnih2015dqn", "sb3"]),
              _pr("bc_res", "BC, residual", ["pomerleau1991"], bn="eval"),
              _pr("dagger_res", "DAgger, residual", ["ross2011dagger"], bn="eval"),
              _pr("bc_scratch", "BC, full twist", ["pomerleau1991"], bn="eval"),
              _pr("dagger_scratch", "DAgger, full twist", ["ross2011dagger"], bn="eval"),
              _pr("cem", "CEM look-ahead", ["rubinstein1999cem"], "secondary", priv=True, seeds=SEED0,
                  learned=False, mdd=False)],
     "not_run": "SonoGym agents~\\cite{sonogym} on Isaac Lab~\\cite{isaaclab}: need an NVIDIA RTX GPU."},
    # title "Reward", the name of Table 2, Box 1 and Section 7.5 of the article
    {"key": "reward", "title": "Reward", "iface": "Reward",
     "setting": "REINFORCE, imitation start, SonoSPADE, CT mask$^{\\dagger}$; score learnability; cost mean ms per "
                "frame pair over the held-out landscape, CPU",
     "rows": [_pr("paper", "Paper score $S$ (authors')", [], "ref"),
              _pr("seg_only", "Organ overlap only (authors')", [], "secondary"),
              _pr("ncc_only", "Appearance NCC only (authors')", [], "secondary"),
              _pr("ncc_sector", "NCC over the whole sector (authors')", [], "secondary"),
              _pr("ssim", "SSIM appearance term", ["wang2004ssim"]),
              _pr("nmi", "NMI appearance term", ["studholme1999nmi"]),
              _pr("pose", "Change in pose distance", ["sonogym", "hase2020"], "secondary", priv=True),
              _pr("pose_area", "Pose distance plus organ area", ["bi2022vesnet"], "secondary", priv=True),
              # amendment 2: LPIPS-R, the LPIPS term in place of S (wrapper; three seeds)
              _pr("lpips", "LPIPS appearance term", ["zhang2018lpips"], "a2", group="lpips", fam2=("LPIPS-R",))],
     "not_run": None},
    {"key": "renderer", "title": "Renderer", "iface": "Renderer",
     "setting": "physics-trained policy, CT mask$^{\\dagger}$, under each renderer; score KID; cost ms per frame "
                "(slice, render, resize and sector warp), CPU benchmark",
     "rows": [_pr("physics", "Physics renderer (authors')", [], "ref", rkey="ctmask/physics/final", v2=True),
              # one line, in the form of the next row and of the renderer frames figure
              _pr("lotus_orig", "LOTUS, original code, published default parameters",
                  ["velikova2023lotus", "salehi2015"]),
              # the record label of plug_rows.json (PLUG_ROWS_MD5); the tables print "LOTUS, authors'
              # reimplementation" through LABEL_PRINT_RENAME
              _pr("lotus_reimpl", "LOTUS, in-repository reimplementation", ["velikova2023lotus"]),
              _pr("simus", "SIMUS via PyMUST, CPU only", ["garcia2022simus", "cigier2022simus"], "secondary",
                  seeds=SEED0, learned=False, mdd=False)],
     "not_run": _T("renderers not run")},
]


def _check_families():
    for slot in PLUG_SLOTS:
        fam = [r["id"] for r in slot["rows"] if r["role"] == "family"]
        if len(fam) != PLUG_FAMILY_M[slot["key"]]:
            raise SystemExit(f"pluggability {slot['key']}: {len(fam)} family rows {fam}, amend1 fixes "
                             f"m = {PLUG_FAMILY_M[slot['key']]}")
        if sum(r["role"] == "ref" for r in slot["rows"]) != 1:
            raise SystemExit(f"pluggability {slot['key']}: exactly one reference row is required")
        for r in slot["rows"]:
            if (r["role"] == "a2") != bool(r["fam2"]) or any(f not in AMEND2_FAMILY_M for f in r["fam2"]):
                raise SystemExit(f"pluggability {slot['key']}/{r['id']}: an amendment 2 row needs its family names")
            if "IMN-T" in r["fam2"] and not any(t["id"] == r["twin"] and t["role"] == "family" for t in slot["rows"]):
                raise SystemExit(f"pluggability {slot['key']}/{r['id']}: its from-scratch twin {r['twin']!r} is not "
                                 "an amendment 1 family row of the slot")
    # amendment 2: 20 contrasts in 5 families; CTA-D and CTA-E are the CT-anatomy block's R1 to R5
    n2 = collections.Counter(f for s in PLUG_SLOTS for r in s["rows"] for f in r["fam2"])
    n2.update({"CTA-D": len(CTA_ROWS), "CTA-E": len(CTA_ROWS)})
    if dict(n2) != AMEND2_FAMILY_M:
        raise SystemExit(f"pluggability: amendment 2 family members {dict(n2)}, the amendment fixes {AMEND2_FAMILY_M}")


def _status_of(d):
    for blk in (d, d.get("provenance") or {}, d.get("provenance_usbg") or {}):
        if isinstance(blk, dict) and blk.get("analysis_status"):
            return blk["analysis_status"]
    return None


def _blinded(s):
    """True while this build keeps the amendment 2 groups unread (BLIND_A2, the default; --read-amend2 lifts it) for a
    path or job name of one of them: SAMUS, segmenters pretrained on ImageNet, LPIPS or the CT-anatomy slot."""
    return BLIND_A2 and any(k in str(s) for k in A2_BLIND_KEYS)


def _blinded_a345(s):
    """The flag name (BLIND_A3 to BLIND_A6) that keeps a path or job name of amendment 3, 4, 5 or 6 unread in this
    build, or None."""
    flags = {"BLIND_A3": BLIND_A3, "BLIND_A4": BLIND_A4, "BLIND_A5": BLIND_A5, "BLIND_A6": BLIND_A6}
    s = str(s).replace(os.sep, "/").lower()
    for flag, keys in A345_BLIND_KEYS.items():
        if flags[flag] and any(k in s for k in keys):
            return flag
    return None


def _guard(p):
    """Refuse to read any output of a blinded group: no table cell of this build may come from one. A path that no
    option lifts is refused first, whatever the flags: the AMOS data, the private real-frame folders, the ImageNet
    group's weights, logs and histories, and the SAMUS Dice on real frames."""
    why = _hard_refusal(p)
    if why:
        raise SystemExit(f"{rel(str(p))}: {why}; no option of this build lifts it")
    if _blinded(p):
        raise SystemExit(f"{rel(str(p))}: an output of a blinded amendment 2 group (BLIND_A2); this build does not "
                         "read it (rerun with --read-amend2 once the blinding is lifted)")
    flag = _blinded_a345(p)
    if flag:
        n = flag[-1]
        raise SystemExit(f"{rel(str(p))}: an output of amendment {n} ({flag}); this build does not read it (rerun "
                         f"with --read-amend{n} once that amendment's analysis has run and its blinding is lifted)")


def _plug_json(p):
    """A v2.1 plug output: it must carry the pluggability status and must not be a smoke run."""
    _guard(p)
    d = json.load(open(p))
    st = _status_of(d)
    if st != PLUG_STATUS:
        raise SystemExit(f"{rel(p)}: analysis_status {st!r}, expected {PLUG_STATUS!r}")
    prov = d.get("provenance") or {}
    if any(isinstance(x, dict) and x.get("smoke") for x in (d, d.get("plug"), prov.get("plug"), prov.get("v21_plug"))):
        raise SystemExit(f"{rel(p)}: a smoke output, not a held-out result")
    return d


def _fingerprint(eps):
    """The tuple below for each episode, its two float members rounded to 1e-6: what the identical-episode audit of
    amend1 compares."""
    return [(e.get("ep"), e["patient"], e["target"], e["window"], round(float(e["start_pos"]), 6),
             round(float(e["start_ang"]), 6)) for e in eps]


_STATIC_FP = []


def _static_fp():
    """Fingerprint of the archived static rung, contact on: the audit's reference for the original six."""
    if not _STATIC_FP:
        d = json.load(open(os.path.join(V2, "table4", "ladder", "static_physics_ct.json")))
        bs = [r for r in d["results"] if r.get("contact") == "on"]
        if len(bs) != 1:
            raise SystemExit("static_physics_ct.json: expected one contact-on block")
        _STATIC_FP.append(_fingerprint(bs[0]["episodes"]))
    return _STATIC_FP[0]


def _episode_block(d, p, weights, bn=None):
    """-> (block, key path) of the contact-on, DR-off evaluation block of one file in any of the layouts: the trainer
    (d[weights]), eval_view_ladder (results[] by weights), plug_train_sb3 (results[], one controller) and the BC and
    DAgger files (results[] by bn_mode). It must be the crossed held-out design (6 held-out patients, evaluation seed
    999, 60 episodes) on the episode setups of the archived static rung."""
    if weights in ("final", "after_imitation") and isinstance(d.get(weights), dict) and "episodes" in d[weights]:
        b, kp = d[weights], f"{weights}.episodes[*].final_pos"
    else:
        want_bn = bn or "train"
        # control: the label-control files (outputs/v2_1/label_controls/closed_loop/) hold one block per label control;
        # the table reads the uncontrolled block (control "none"); every other layout has no control field
        bs = [r for r in d.get("results", []) if r.get("contact") == "on" and r.get("eval_randomize", "off") == "off"
              and r.get("bn_mode", "train") == want_bn and (weights is None or r.get("weights") == weights)
              and r.get("control", "none") == "none"]
        if len(bs) != 1:
            raise SystemExit(f"{rel(p)}: {len(bs)} contact-on, DR-off blocks with weights {weights!r}, BatchNorm "
                             f"{want_bn}")
        b = bs[0]
        sel = (["contact=on"] + ([f"weights={weights}"] if weights else []) + ([f"bn_mode={bn}"] if bn else [])
               + (["control=none"] if "control" in b else []))
        kp = f"results[{','.join(sel)}].episodes[*].final_pos"
    eps = b.get("episodes") or []
    if (b.get("contact", "on") != "on" or b.get("eval_randomize", "off") != "off"
            or str(b.get("design", "crossed")).split()[0] != "crossed" or b.get("eval_seed", 999) != 999
            or len(eps) != 60 or {e["patient"] for e in eps} != set(TEST_PATIENTS)):
        raise SystemExit(f"{rel(p)}: not the crossed held-out design (6 held-out patients, seed 999, 60 episodes)")
    if _fingerprint(eps) != _static_fp():
        raise SystemExit(f"{rel(p)}: episode setups differ from outputs/v2/table4/ladder/static_physics_ct.json "
                         "(identical-episode check)")
    return b, kp


def _plug_cell(paths, weights, plug, bn=None):
    """seed -> per-patient final translation error and success (10 mm and 10 deg), for the seeds whose file exists."""
    out = {}
    for s, p in paths.items():
        _guard(p)
        if not os.path.exists(p):
            continue
        d = _plug_json(p) if plug else json.load(open(p))
        b, kp = _episode_block(d, p, weights, bn)
        pos, suc = collections.defaultdict(list), collections.defaultdict(list)
        for e in b["episodes"]:
            pos[e["patient"]].append(float(e["final_pos"]))
            suc[e["patient"]].append(float(float(e["final_pos"]) <= 10.0 and float(e["final_ang"]) <= 10.0))
        out[s] = {"pos": {k: float(np.mean(v)) for k, v in pos.items()},
                  "succ": {k: float(np.mean(v)) for k, v in suc.items()}, "d": d, "b": b, "path": p, "kp": kp}
    return out


def _cell_mean(cell, key="pos"):
    return float(np.mean([np.mean(list(c[key].values())) for c in cell.values()])) if cell else None


def _plug_contrast(a, b):
    """Seed-paired, patient-level difference a - b (mean over the common seeds): mean, 95% paired t CI and two-sided
    paired t p, the number of patients sharing the sign of the mean, the exact two-sided sign-flip p, and, as a
    sensitivity interval, the patient bootstrap CI of collect_results_v2.boot_ci."""
    import posthoc_v2 as PH
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    P = sorted(TEST_PATIENTS)
    d = (np.array([[a[s]["pos"][p] for p in P] for s in seeds]).mean(0)
         - np.array([[b[s]["pos"][p] for p in P] for s in seeds]).mean(0))
    m = float(d.mean())
    ts = PH.t_stats(d)
    p_sf, exact = PH.signflip_p(d)
    if not exact:
        raise SystemExit("sign-flip p is not exact at n = 6")
    _, boot = C.boot_ci(d, nboot=PLUG_BOOT, seed=0)
    k = int((np.sign(d) == np.sign(m)).sum())
    # POST HOC columns (they never change a Decision): the per-seed differences and k/3; the TOST at +/- 1 mm (inside
    # the margin when the 90% t CI is, i.e. TOST p < 0.05); and, with two or more seeds, the nested bootstrap of
    # plan_stats.patient_seed_bootstrap_ci beside the crossed one of posthoc_v2.crossed_boot.
    D = np.array([[a[s]["pos"][p] - b[s]["pos"][p] for s in seeds] for p in P])     # [patient, seed]
    per_seed = [float(x) for x in D.mean(0)]
    k_seed = int(sum(np.sign(x) == np.sign(m) for x in per_seed))
    tost = _tost(d, TOST_MARGIN_MM)
    ps_n = ps_c = None
    if len(seeds) >= 2:
        ps_n = [float(x) for x in _plan_stats().patient_seed_bootstrap_ci(D, level=0.95, constants=_plan_consts())]
        ps_c = [float(x) for x in PH.crossed_boot(D, nboot=PLUG_BOOT, seed=PLUG_BOOT_SEED)[0]]
    return {"mean": m, "ci": [float(x) for x in ts["ci_t"]], "p_t": float(ts["p_t"]), "p_sf": float(p_sf),
            "boot": [float(x) for x in boot], "sign": k, "n": len(P), "seeds": seeds,
            "per_patient": {p: float(x) for p, x in zip(P, d)},
            "per_seed": per_seed, "k_seed": k_seed, "n_seed": len(seeds), "sd": float(d.std(ddof=1)),
            "tost_ci90": tost["ci90"], "tost_p": tost["p"], "tost_in": tost["within"],
            "ps_nested": ps_n, "ps_crossed": ps_c}


TOST_MARGIN_MM = 1.0   # post hoc TOST of every Table 5 contrast, at the +/- 1 mm margin of the pre-specified tests
TOST_MARGIN_SRC = _T("tost margin src")
_PS = {}


def _plan_stats():
    """plan_stats.py of the analysis scripts (script_dirs.analysis_dir), the plan-conformant statistics layer
    (imported once)."""
    if "mod" not in _PS:
        p = SD.analysis_dir()
        if p not in sys.path:
            sys.path.insert(0, p)
        import plan_stats  # noqa: E402
        _PS["mod"] = plan_stats
    return _PS["mod"]


def _plan_consts():
    """The bootstrap constants of the v2.1 plan (plan_inputs.plan_v21_bootstrap): 10,000 resamples, seed 0.
    patient_seed_bootstrap_ci reads only C["bootstrap"]."""
    if "C" not in _PS:
        C_ = {"bootstrap": PI.plan_v21_bootstrap()}
        if (int(C_["bootstrap"]["resamples"]), int(C_["bootstrap"]["seed"])) != (PLUG_BOOT, PLUG_BOOT_SEED):
            raise SystemExit(f"v2.1 plan bootstrap constants {C_['bootstrap']} differ from {PLUG_BOOT}, "
                             f"{PLUG_BOOT_SEED}")
        _PS["C"] = C_
    return _PS["C"]


def _tost(d, margin):
    """Equivalence test of per-patient differences d at +/- margin with paired t statistics (n - 1 degrees of
    freedom): the 90% t CI, the TOST p (maximum of the two one-sided p values) and whether the 90% CI lies inside
    (-margin, +margin), computed as in plan_stats.tost_t and posthoc_v2.tost (post hoc for Table 5, unadjusted)."""
    from scipy import stats as st
    x = np.asarray(d, float)
    n = len(x)
    mu, se = float(x.mean()), float(x.std(ddof=1)) / np.sqrt(n)
    if se > 0:
        p_lo, p_hi = float(st.t.sf((mu + margin) / se, n - 1)), float(st.t.cdf((mu - margin) / se, n - 1))
    else:
        p_lo, p_hi = (0.0 if mu > -margin else 1.0), (0.0 if mu < margin else 1.0)
    tq = float(st.t.ppf(0.95, n - 1))
    ci = [mu - tq * se, mu + tq * se]
    p = max(p_lo, p_hi)
    within = bool(-margin < ci[0] and ci[1] < margin)
    if within != (p < 0.05):
        raise SystemExit(f"TOST: 90% CI {ci} and TOST p {p} disagree")
    return {"ci90": [float(v) for v in ci], "p": float(p), "within": within}


def _posthoc_note(c, dd=2):
    """The src-comment string of the post hoc statistics of one contrast (per-seed differences, k/3, TOST, the planned
    patient-by-seed bootstrap and the crossed bootstrap)."""
    f = f"{{:+.{max(dd, 3)}f}}"
    s = (f"; POST HOC{_T('posthoc date')}: per-seed differences " + ", ".join(
        f"s{sd} {f.format(v)}" for sd, v in zip(c["seeds"], c["per_seed"])) + f", k/{c['n_seed']} = {c['k_seed']}; "
         f"TOST +/- {TOST_MARGIN_MM:g} mm: 90% t CI [{c['tost_ci90'][0]:+.3f}, {c['tost_ci90'][1]:+.3f}], TOST p "
         f"{c['tost_p']:.4g}, within the margin: {'yes' if c['tost_in'] else 'no'} (unadjusted; margin of "
         f"{TOST_MARGIN_SRC})")
    if c.get("ps_nested") is not None:
        s += (f"; patient-by-seed bootstrap 95% CI [{c['ps_nested'][0]:+.3f}, {c['ps_nested'][1]:+.3f}] ("
              + _T("ps boot form") + f"plan_stats.patient_seed_bootstrap_ci); crossed patient x seed bootstrap 95% CI "
              f"[{c['ps_crossed'][0]:+.3f}, {c['ps_crossed'][1]:+.3f}] (posthoc_v2.crossed_boot)")
    return s


# what each slot's cost measures (the twin's slot headers name the unit and device; none is a gate timing)
COST_SRC = {
    "texture": "authors' and prior stages RESULTS.json compute.texture.<stage>.mps.median_ms (outputs/v2/"
               "bench_latency.json, 200 calls, batch 1); plugged stages outputs/v2_1/plug/texture/tex/<id>_s0.json "
               "infer_cost_ms.mps.ms_median (30 calls after 5 warm-up calls, translator call only, after training); "
               "both per frame on MPS",
    "mask": "U-Net RESULTS.json compute.closed_loop.mps.sonospade.components_median_ms.segment (outputs/v2/"
            "bench_closed_loop.json, per step); plugged sources the evaluation's own plug.timing.median_ms over every "
            "mask call of the 60 episodes; both per step on MPS",
    "ctrl": "REINFORCE RESULTS.json compute.training.train_h_median_by_arm_stage (outputs/v2/table4/runner.log); plugged "
            "learners result JSON train.sec / 3600 (learner_device mps); CEM results[contact=on].eval_sec_per_episode",
    "reward": "outputs/v2_1/plug/reward/summary.json landscape[*].cost_ms_per_pair, from the landscape JSON "
              "cost.ms_per_pair (1000 x total seconds / pairs scored, CPU)",
    "renderer": "physics RESULTS.json compute.render.cpu.frame.median_ms (outputs/v2/bench_render.json); plugged "
                "renderers outputs/v2_1/plug/renderer/bench/bench.json renderers.<id>.devices.cpu.frame.median_ms "
                "(frame = slice + render + resize + sector warp, the scripts/bench_render.py protocol)",
}


def _cost_src(slot_key, plug_root=None):
    return COST_SRC[slot_key]


# the supplementary twin of Table 5 (tab:pluggability-twin): ten columns, a full-width line spans all of them
TW_NCOL = 10
TW_MC = f"\\multicolumn{{{TW_NCOL}}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.97\\textwidth}}@{{}}}}"
_TW_MC7 = "\\multicolumn{7}{@{}>{\\raggedright\\arraybackslash}p{0.97\\textwidth}@{}}"


def _tw10(x):
    """A line of the CT-anatomy block (_cta_block, seven cells: label, role, cost, error, difference, Holm p, score) in
    the ten columns of the twin (label, role, cost, error, difference, t p, Holm p, sign-flip p, 90% t CI, score)."""
    if x.startswith(_TW_MC7):
        return TW_MC + x[len(_TW_MC7):]
    if not x.strip() or x.startswith(("\\addlinespace", "%")):   # a comment line passes unchanged
        return x
    head, sep, tail = x.partition(" \\\\")
    cells = [y.strip() for y in head.split("&")]
    if not sep or len(cells) != 7:
        raise SystemExit(f"pluggability_twin: a CT-anatomy line with {len(cells)} cells: {x[:120]}")
    lab, role, cost, err, diff, holm, score = cells
    return " & ".join([lab, role, cost, err, diff, "", holm, "", "", score]) + sep + tail


def _srcp(p):
    """A source path for a src comment; the archived trainer files keep their names, written with a wildcard where
    the name holds a word that the printed text does not use."""
    return rel(p).replace("_deploy_", "_*_")


def _tt_path(s):
    """A path for a narrow table column: underscores escaped, a break allowed after each slash and underscore."""
    return s.replace("_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}")


def _ms(x):
    if x is None:
        return ""
    if x >= 1000:
        return f"{x / 1000:.1f}~s"
    return f"{x:.0f}~ms" if x >= 100 else f"{x:.1f}~ms"


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _plug_paths(slot, row, plug_root, seg_plug_root):
    """-> (seed -> result file, weights key, is_plug_file) for the closed-loop error of one row, over its design seeds."""
    lad = os.path.join(V2, "table4", "ladder")
    sid, k, seeds = row["id"], slot["key"], row["seeds"]
    if k == "texture":
        if row["v2"]:
            return {s: os.path.join(lad, f"ibvs_base_{sid}_matched_s{s}.json") for s in seeds}, "zero_residual", False
        if sid == "label_painted":    # Q10(b): a v2.1 file, not a plug output
            return {s: PAINTED_CL.format(seed=s) for s in seeds}, "zero_residual", False
        return ({s: os.path.join(plug_root, "texture", "base", f"ibvs_base_{sid}_matched_s{s}.json") for s in seeds},
                "zero_residual", True)
    if k == "mask":
        if row["v2"]:
            arm = "matched" if sid == "unet" else "ct"
            return {s: os.path.join(lad, f"ibvs_base_sonospade_{arm}_s{s}.json") for s in seeds}, "zero_residual", False
        return ({s: os.path.join(plug_root, "seg", "ladder", f"ibvs_base_sonospade_{sid}_s{s}.json") for s in seeds},
                "zero_residual", True)
    if k == "ctrl":
        if sid == "reinforce":
            return {s: os.path.join(V2, "table4", "ctmask", f"sonospade_deploy_s{s}.json") for s in seeds}, "final", False
        if sid == "servo_base":
            return {s: os.path.join(lad, f"ibvs_base_sonospade_ct_s{s}.json") for s in seeds}, "zero_residual", False
        if sid == "reinforce_bneval":
            return ({s: os.path.join(ROOT, "outputs", "v2_1", "s21_bneval", f"ctmask_sonospade_s{s}.json")
                     for s in seeds}, "final", False)
        return {s: os.path.join(plug_root, "ctrl", sid, f"sonospade_ct_s{s}.json") for s in seeds}, None, True
    if k == "reward":
        return {s: os.path.join(plug_root, "reward", "train", sid, f"sonospade_s{s}.json") for s in seeds}, "final", True
    if k == "renderer":
        if row["v2"]:
            return ({s: os.path.join(V2, "table4", "contact", f"ctmask_physics_s{s}.json") for s in seeds},
                    "final", False)
        return ({s: os.path.join(plug_root, "renderer", "ladder", f"policy_v2physics_under_{sid}_s{s}.json")
                 for s in seeds}, "final", True)
    raise SystemExit(f"unknown slot {k}")


def _gates_of(slot_key, sid):
    """Admission-gate outputs (amend1) that a plug row's numbers depend on (the queue's gate dependencies)."""
    if slot_key == "texture":
        return [f"texture_{sid}.json"]
    if slot_key == "mask":
        for pre, g in (("samus", "mask_samus.json"), ("medsam", "mask_medsam.json"), ("sam_", "mask_sam.json"),
                       ("nnunet", "mask_nnunet.json")):
            if sid.startswith(pre):
                return [g]
        return ["mask_tv.json"]
    if slot_key == "ctrl":
        if sid in ("bc_res", "dagger_res", "bc_scratch", "dagger_scratch"):
            return ["ctrl_goalnet.json"]
        return ["ctrl_reference.json"] if sid == "cem" else ["ctrl_sb3.json"]
    if slot_key == "reward":
        return ["reward_terms.json"] + (["reward_lpips.json"] if sid == "lpips" else [])
    if slot_key == "renderer":
        return [f"renderer_{sid}.json"]
    raise SystemExit(f"unknown slot {slot_key}")


def _gate_json(name):
    p = os.path.join(CHECKS_ROOT, "gate", name)
    _guard(p)
    return _plug_json(p) if os.path.exists(p) else None


def _require_gates(slot, row):
    """Stops the build unless every admission gate of the adapter has an output that admits it."""
    for g in _gates_of(slot["key"], row["id"]):
        j = _gate_json(g)
        if j is None or j.get("admitted") is not True:
            raise SystemExit(f"pluggability {slot['key']}/{row['id']}: results exist but gate {g} is "
                             f"{'missing' if j is None else 'not admitted'}")


def _dice_reportable(slot, row):
    """False when the row's gate records that its real-frame Dice must be withheld (MedSAM, amend1)."""
    if row["v2"] or slot["key"] != "mask":
        return True
    for g in _gates_of("mask", row["id"]):
        j = _gate_json(g)
        if j is not None and _dig(j, "checks", "pretraining_corpus", "real_frame_dice_reportable") is False:
            return False
    return True


def _plug_score_cost(R, slot, row, cell, plug_root, seg_plug_root, seeds=None):
    """-> (score text, cost text, source note, score is v2.1, cost is v2.1) for one row; empty text where the source
    does not exist yet. seeds: restrict the per-seed scores (Dice, success) to these seeds."""
    k, sid, src = slot["key"], row["id"], []
    score = cost = None
    sc_txt = co_txt = ""
    sc21 = co21 = not row["v2"]
    seeds = row["seeds"] if seeds is None else seeds
    sub = {s: c for s, c in cell.items() if s in seeds}
    if k == "texture":
        if sid == "label_painted":
            # Q10(b) realism run (the archived Table A.4 protocol with the painted stage); no per-frame benchmark of the
            # painted stage exists, so its cost prints as not timed
            dp = json.load(open(PAINTED_REAL))
            if _status_of(dp) != "exploratory, v2.1":
                raise SystemExit(f"{rel(PAINTED_REAL)}: analysis_status {_status_of(dp)!r}, expected 'exploratory, v2.1'")
            score = _dig(dp, "variants", "physics+label_painted_lit", "kid")
            src.append(f"score {rel(PAINTED_REAL)} variants[physics+label_painted_lit].kid; cost: no benchmark of the "
                       "painted stage exists")
            co_txt = "not timed"
        elif not row["v2"]:
            f = os.path.join(plug_root, "texture", "realism", "realism_sector_plug.json")
            if os.path.exists(f):
                score = _dig(_plug_json(f), "variants", f"physics+{sid}", "kid")
                src.append(f"score {rel(f)} variants[physics+{sid}].kid")
            f = os.path.join(plug_root, "texture", "tex", f"{sid}_s0.json")
            if os.path.exists(f):
                cost = _dig(_plug_json(f), "infer_cost_ms", "mps", "ms_median")
                src.append(f"cost {rel(f)} infer_cost_ms.mps.ms_median")
        else:
            f = os.path.join(V2, "realism", "realism_sector.json")
            score = _dig(json.load(open(f)), "variants", V2_REAL_NAME[sid], "kid")
            src.append(f"score {rel(f)} variants[{V2_REAL_NAME[sid]}].kid")
            if sid == "physics":
                co_txt = "none"
            else:
                cost = R["compute"]["texture"][sid]["mps"]["median_ms"]
                src.append(f"cost RESULTS.json compute.texture.{sid}.mps.median_ms")
        sc_txt = "" if score is None else f"{score:.4f}"
    elif k == "mask":
        evs, samus_seen = [], False
        for s in seeds:
            f = (os.path.join(SEG, "outputs", "v2", f"eval_sonospade_s{s}.json") if sid == "unet" else
                 os.path.join(seg_plug_root, "seg", sid, f"eval_sonospade_s{s}.json"))
            if sid.startswith("samus"):
                # amendment 2: the real-frame Dice of SAMUS is withheld and its segeval output is not opened
                samus_seen |= os.path.exists(f)
                continue
            if sid != "ct" and os.path.exists(f):
                d = json.load(open(f)) if sid == "unet" else _plug_json(f)
                v = _dig(d, "vitale_holdout_v2.npz", "per_class", "liver")
                if v is not None:
                    evs.append(v)
        if sid == "ct":
            sc_txt, co_txt = "n/a", "none"
        elif samus_seen:
            sc_txt, sc21 = "withheld", False
            src.append(_T("samus withheld src"))
        elif evs and not _dice_reportable(slot, row):
            sc_txt, sc21 = "withheld", False
            src.append("score withheld: outputs/v2_1/checks/gate/" + _gates_of("mask", sid)[0]
                       + " checks.pretraining_corpus.real_frame_dice_reportable = false")
        elif evs:
            score = float(np.mean(evs))
            sc_txt = f"{score:.2f}"
            src.append(f"score {PI.text('unet score dir') if sid == 'unet' else rel(seg_plug_root) + '/seg/' + sid}"
                       f"/eval_sonospade_s<seed>.json vitale_holdout_v2.npz.per_class.liver "
                       + (f"(mean of {len(evs)} seeds)" if len(evs) > 1 else f"(seed {seeds[0]})"))
        if sid == "unet":
            if seeds == PLUG_SEEDS:
                cost = R["compute"]["closed_loop"]["mps"]["sonospade"]["components_median_ms"]["segment"]
                src.append("cost RESULTS.json compute.closed_loop.mps.sonospade.components_median_ms.segment")
        elif sid != "ct" and sub:
            ts = [_dig(c["d"], "plug", "timing", "median_ms") for c in sub.values()]
            ts = [t for t in ts if t is not None]
            cost = float(np.median(ts)) if ts else None
            src.append("cost ladder JSON plug.timing.median_ms (median over seeds)")
    elif k == "ctrl":
        if sub:
            score = 100.0 * _cell_mean(sub, "succ")
            sc_txt = f"{score:.0f}\\%"
            src.append("score success (final_pos <= 10 and final_ang <= 10) from the same episodes")
        if sid == "reinforce":
            if seeds == PLUG_SEEDS:
                h = R["compute"]["training"]["train_h_median_by_arm_stage"]["ctmask/sonospade"]
                co_txt = f"{h:.2f}~h"
                src.append("cost RESULTS.json compute.training.train_h_median_by_arm_stage.ctmask/sonospade (median)")
        elif sid == "servo_base":
            co_txt = "n/a"
        elif sid == "cem" and sub:
            v = [c["b"].get("eval_sec_per_episode") for c in sub.values()]
            v = [x for x in v if x is not None]
            if v:
                co_txt = f"{np.median(v):.0f}~s/ep"
                src.append("cost result JSON results[contact=on].eval_sec_per_episode (median over seeds)")
        elif sub:
            h = [c["d"]["train"]["sec"] / 3600.0 for c in sub.values() if _dig(c["d"], "train", "sec") is not None]
            if h:
                co_txt = f"{np.median(h):.2f}~h"
                src.append("cost result JSON train.sec / 3600 (median over seeds)")
    elif k == "reward":
        term = {"paper": ("ncc_lit", "app"), "ncc_only": ("ncc_lit", "app"), "ncc_sector": ("ncc_sector", "app"),
                "ssim": ("ssim", "app"), "nmi": ("nmi", "app"), "lpips": ("lpips", "app"),
                "pose": ("pose", "potential"), "pose_area": ("pose_area", "potential")}.get(sid)
        for name in ("summary_with_lpips.json", "summary.json"):
            f = os.path.join(plug_root, "reward", name)
            if term is None or _blinded(f) or not os.path.exists(f):   # the LPIPS analysis is blinded in this build
                continue
            for lp, ent in (_plug_json(f).get("landscape") or {}).items():
                if not isinstance(ent, dict) or ent.get("smoke"):
                    continue
                v = _dig(ent, "table", "sonospade", term[0], term[1])
                if v is not None:
                    score = v
                    cost = _dig(ent, "cost_ms_per_pair", term[0])
                    src.append(f"score {rel(f)} landscape[{os.path.basename(lp)}].table.sonospade.{term[0]}.{term[1]}; "
                               f"cost .cost_ms_per_pair.{term[0]}")
                    break
            if score is not None:
                break
        sc_txt = "n/a" if term is None else ("" if score is None else f"{score:.2f}")
        if term is None or sid in ("pose", "pose_area"):
            co_txt = "n/a"
        elif cost is not None:
            co_txt = f"{cost:.2f}~ms"
    elif k == "renderer":
        if row["v2"]:
            if seeds == PLUG_SEEDS:
                f = os.path.join(V2, "realism", "realism_sector.json")
                score = _dig(json.load(open(f)), "variants", "physics", "kid")
                cost = R["compute"]["render"]["cpu"]["frame"]["median_ms"]
                src += [f"score {rel(f)} variants[physics].kid", "cost RESULTS.json compute.render.cpu.frame.median_ms"]
        else:
            for name in ("torch.json", "simus.json"):
                f = os.path.join(plug_root, "renderer", "realism", name)
                if os.path.exists(f):
                    v = _dig(_plug_json(f), "variants", sid, "kid")
                    if v is not None:
                        score = v
                        src.append(f"score {rel(f)} variants[{sid}].kid")
                        break
            f = os.path.join(plug_root, "renderer", "bench", "bench.json")
            if os.path.exists(f):
                cost = _dig(_plug_json(f), "renderers", sid, "devices", "cpu", "frame", "median_ms")
                src.append(f"cost {rel(f)} renderers.{sid}.devices.cpu.frame.median_ms")
        sc_txt = "" if score is None else f"{score:.4f}"
    if not co_txt and cost is not None:
        co_txt = _ms(cost)
    mark = PROV if PLUG_PROVISIONAL else ""
    if sc21 and sc_txt and sc_txt not in ("n/a", "none", "withheld"):
        sc_txt += mark
    if co21 and co_txt and co_txt not in ("n/a", "none", "not timed"):
        co_txt += mark
    return sc_txt, co_txt, "; ".join(src)


PLUG_AUDIT = {"texture": "texture", "mask": "seg", "ctrl": "ctrl", "reward": "reward", "renderer": "renderer"}


def _audit_state(slot_key):
    """The slot's identical-episode audit (amend1): 'passed', 'pending' or a refusal (stops the run)."""
    base = os.path.join(CHECKS_ROOT, "audit", PLUG_AUDIT[slot_key])
    if glob.glob(base + ".failed*.json"):
        raise SystemExit(f"{rel(base)}.failed*.json: the identical-episode audit refused a {slot_key} block; the "
                         "slot's rows are not printed until the file is inspected (amend1)")
    if os.path.exists(base + ".json"):
        if _plug_json(base + ".json").get("passed") is not True:
            raise SystemExit(f"{rel(base)}.json: audit output without passed = true")
        return "passed"
    return "pending"


def _sb3_certified():
    """SB3 rows print a not-certified note unless the positive-control verdict is PASS."""
    f = os.path.join(CHECKS_ROOT, "sb3pos", "verdict.json")
    if not os.path.exists(f):
        return False
    return bool(_plug_json(f).get("rl_slot_certified"))


# the Appendix keys this build asserts (plan_inputs.amendment2_appendix; the development backend checks the block's
# member-completeness definition and its schedule keys itself and gives the rest)
AMEND2_KEYS = {"amendment", "status", "alpha", "n6_decision", "families", "missing_member_p", "cta", "samus", "imnet",
               "lpips", "bootstrap"}


def _slot(key):
    return next(s for s in PLUG_SLOTS if s["key"] == key)


def _row(slot, rid):
    return next(r for r in slot["rows"] if r["id"] == rid)


def _amend2():
    """Amendment 2, before pluggability() reads any output of its four groups: the Appendix block
    (plan_inputs.amendment2_appendix), its digest against AMEND2_SHA256, every constant and file pattern this code uses
    against the block, and every file it reads against the job queue (JOBS_FINAL)."""
    if _AMEND2:
        return _AMEND2["A"]
    A = PI.amendment2_appendix(root=CODE_ROOT, expect=AMEND2_SHA256)
    h = PI.amendment_digest(2)
    if h != AMEND2_SHA256:
        raise SystemExit(f"{AMEND2_TXT}: sha256 {h} differs from the committed digest {AMEND2_SHA256}: refusing to "
                         "read any output of the amendment 2 groups")
    bad = []

    def eq(name, got, want):
        if got != want or type(got) is not type(want):
            bad.append(f"{name}: Appendix {got!r}, code {want!r}")

    eq("keys", set(A), AMEND2_KEYS)
    eq("amendment", A.get("amendment"), "v2.1 amendment 2")
    eq("status", A.get("status"), PLUG_STATUS)
    eq("alpha", A.get("alpha"), AMEND2_ALPHA)
    eq("n6_decision", A.get("n6_decision"), AMEND2_DECISION)
    eq("families", A.get("families"), AMEND2_FAMILY_M)
    eq("missing_member_p", A.get("missing_member_p"), AMEND2_MISSING_P)
    eq("bootstrap", A.get("bootstrap"), {"resamples": PLUG_BOOT, "seed": PLUG_BOOT_SEED})
    # CT-anatomy slot
    c = A.get("cta") or {}
    eq("cta keys", set(c), {"new_root", "reference_root", "rows", "block", "invariance_check", "metric", "seeds",
                            "tost_margin_mm", "tost_min_sign_agree", "tost_ci_printed", "label_mode", "resolution_mm",
                            "setup_tol", "pose_tol", "cohort"})
    eq("cta.new_root", c.get("new_root"), rel(CTA_NEW_ROOT))
    eq("cta.reference_root", c.get("reference_root"), rel(CTA_REF_ROOT))
    eq("cta.rows", c.get("rows"), {r: f for r, f, *_ in CTA_ROWS})
    eq("cta.block", c.get("block"), CTA_BLOCK)
    eq("cta.invariance_check", c.get("invariance_check"), CTA_INVARIANCE)
    eq("cta.metric", c.get("metric"), CTA_METRIC)
    eq("cta.seeds", c.get("seeds"), list(CTA_SEEDS))
    eq("cta.tost_margin_mm", c.get("tost_margin_mm"), CTA_TOST_MARGIN)
    eq("cta.tost_min_sign_agree", c.get("tost_min_sign_agree", "missing"), None)
    eq("cta.tost_ci_printed", c.get("tost_ci_printed"), CTA_TOST_CI)
    eq("cta.label_mode", c.get("label_mode"), CTA_MODE)
    eq("cta.resolution_mm", c.get("resolution_mm"), CTA_RES_MM)
    eq("cta.setup_tol", c.get("setup_tol"), CTA_SETUP_TOL)
    eq("cta.pose_tol", c.get("pose_tol"), CTA_POSE_TOL)
    eq("cta.cohort", c.get("cohort"), "not run")
    for rid, f, w, _, _ in CTA_ROWS:     # ladder rungs use the zero-residual block, the policy rungs the final weights
        eq(f"cta.{rid} weights", w, CTA_BLOCK["policy_weights"] if f.startswith("contact/") else "zero_residual")
    eq("cta.invariance equals a row", CTA_INVARIANCE["equals"] in [r[0] for r in CTA_ROWS], True)
    eq("cta 90% CI (posthoc_v2.tost uses the one-sided 0.95 t quantile)", CTA_TOST_CI, 0.90)
    # SAMUS
    mask, reward = _slot("mask"), _slot("reward")
    unet, samus = _row(mask, "unet"), _row(mask, "samus_unet")
    s = A.get("samus") or {}
    eq("samus keys", set(s), {"family", "file", "reference", "box_min_px", "max_boxes_per_organ", "real_frame_dice"})
    eq("samus.family", s.get("family"), samus["fam2"][0] if samus["fam2"] else None)
    eq("samus seeds", samus["seeds"], SEED0)
    eq("samus.file", s.get("file"), rel(_plug_paths(mask, samus, PLUG_ROOT, SEG_PLUG_ROOT)[0][0]))
    eq("samus.reference", s.get("reference"), rel(_plug_paths(mask, unet, PLUG_ROOT, SEG_PLUG_ROOT)[0][0]))
    eq("samus.box_min_px", s.get("box_min_px"), SAMUS_BOX_MIN_PX)
    eq("samus.max_boxes_per_organ", s.get("max_boxes_per_organ"), SAMUS_MAX_BOXES)
    eq("samus.real_frame_dice", s.get("real_frame_dice"), "withheld")
    # ImageNet-pretrained torchvision segmenters
    i = A.get("imnet") or {}
    eq("imnet keys", set(i), {"families", "file", "reference", "twin", "archs", "seeds", "iters", "batch",
                              "backbone_weights"})
    rows_i = [r for r in mask["rows"] if "IMN-T" in r["fam2"]]
    eq("imnet.archs", i.get("archs"), [r["id"] for r in rows_i])
    eq("imnet.iters", i.get("iters"), IMNET_ITERS)
    eq("imnet.batch", i.get("batch"), IMNET_BATCH)
    eq("imnet.backbone_weights", i.get("backbone_weights"), IMNET_WEIGHTS)
    for r in rows_i:
        eq(f"imnet.families ({r['id']})", i.get("families"), list(r["fam2"]))
        eq(f"imnet.seeds ({r['id']})", i.get("seeds"), list(r["seeds"]))
        eq(f"imnet twin of {r['id']}", r["twin"], r["id"][:-len("_imnet")] if r["id"].endswith("_imnet") else None)
        pa = _plug_paths(mask, r, PLUG_ROOT, SEG_PLUG_ROOT)[0]
        pt = _plug_paths(mask, _row(mask, r["twin"]), PLUG_ROOT, SEG_PLUG_ROOT)[0]
        pu = _plug_paths(mask, unet, PLUG_ROOT, SEG_PLUG_ROOT)[0]
        for sd in r["seeds"]:
            eq(f"imnet.file {r['id']} s{sd}", str(i.get("file")).format(arch=r["id"], seed=sd), rel(pa[sd]))
            eq(f"imnet.reference s{sd}", str(i.get("reference")).format(seed=sd), rel(pu[sd]))
            eq(f"imnet.twin {r['id']} s{sd}",
               str(i.get("twin")).replace(_T("twin arch"), r["twin"]).format(seed=sd), rel(pt[sd]))
    # LPIPS
    lp, paper = _row(reward, "lpips"), _row(reward, "paper")
    e = A.get("lpips") or {}
    eq("lpips keys", set(e), {"file", "reference", "block", "seeds", "w_seg", "w_app", "env_steps"})
    eq("lpips.seeds", e.get("seeds"), list(lp["seeds"]))
    eq("lpips.w_seg", e.get("w_seg"), LPIPS_W_SEG)
    eq("lpips.w_app", e.get("w_app"), LPIPS_W_APP)
    eq("lpips.env_steps", e.get("env_steps"), LPIPS_ENV_STEPS)
    pl, wl, _ = _plug_paths(reward, lp, PLUG_ROOT, SEG_PLUG_ROOT)
    pp, wp, _ = _plug_paths(reward, paper, PLUG_ROOT, SEG_PLUG_ROOT)
    eq("lpips.block", e.get("block"), wl)
    eq("lpips.block (reference)", e.get("block"), wp)
    for sd in lp["seeds"]:
        eq(f"lpips.file s{sd}", str(e.get("file")).format(seed=sd), rel(pl[sd]))
        eq(f"lpips.reference s{sd}", str(e.get("reference")).format(seed=sd), rel(pp[sd]))
    # every file this code reads for the four groups is the out of a queued job; each CT-anatomy evaluation's --ref
    # is the reference rung this code pairs it with
    Qr = {}
    for j in json.load(open(JOBS_FINAL)):
        if j.get("out"):
            Qr[os.path.realpath(SD.map_recorded(j["out"]))] = j

    def queued(path, key, ref=None):
        j = Qr.get(os.path.realpath(path))
        if j is None or key not in j["name"]:
            bad.append(f"{rel(path)}: not the out of a queued job named *{key}* in {rel(JOBS_FINAL)}")
            return
        if ref is not None:
            cmd = j["cmd"] if isinstance(j["cmd"], list) else str(j["cmd"]).split()
            got = cmd[cmd.index("--ref") + 1] if "--ref" in cmd else None
            if got is None or os.path.realpath(SD.map_recorded(got)) != os.path.realpath(ref):
                bad.append(f"{j['name']}: --ref {got!r}, this code pairs it with {rel(ref)}")

    for rid, f, *_ in CTA_ROWS:
        for sd in CTA_SEEDS:
            queued(os.path.join(CTA_NEW_ROOT, f.format(seed=sd)), ":plug:ctanat:fast:",
                   os.path.join(CTA_REF_ROOT, f.format(seed=sd)))
    for sd in CTA_SEEDS:
        f = CTA_INVARIANCE["file"].format(seed=sd)
        queued(os.path.join(CTA_NEW_ROOT, f), ":plug:ctanat:fast:", os.path.join(CTA_REF_ROOT, f))
    for f in CTA_PREMISE:
        queued(os.path.join(CTA_NEW_ROOT, f), ":plug:ctanat:fast:", os.path.join(CTA_REF_ROOT, f))
    queued(os.path.join(CTA_NEW_ROOT, CTA_SUMMARY), ":plug:ctanat:fast:analysis")
    queued(_plug_paths(mask, samus, PLUG_ROOT, SEG_PLUG_ROOT)[0][0], ":samus_unet:sonospade:s0")
    for r in rows_i:
        for sd, p in _plug_paths(mask, r, PLUG_ROOT, SEG_PLUG_ROOT)[0].items():
            queued(p, f":ladder:{r['id']}:sonospade:s{sd}")
        for sd, p in _plug_paths(mask, _row(mask, r["twin"]), PLUG_ROOT, SEG_PLUG_ROOT)[0].items():
            queued(p, f":ladder:{r['twin']}:sonospade:s{sd}")
    for sd in lp["seeds"]:
        queued(pl[sd], f":train:lpips:sonospade:s{sd}")
        queued(pp[sd], f":train:paper:sonospade:s{sd}")
    if bad:
        raise SystemExit(f"{AMEND2_TXT} Appendix and make_tables.py disagree (no output of the amendment 2 groups "
                         "was read):\n  " + "\n  ".join(bad))
    _AMEND2["A"] = A
    log(f"[check] amendment 2: {AMEND2_TXT} sha256 {h} equals {AMEND2_SHA_TXT} and the committed digest; "
        f"Appendix constants (families {A['families']}, seeds, file patterns, TOST margin {CTA_TOST_MARGIN} mm) "
        "asserted; every file read for its groups is a queued job's out")
    return A


def _ps_boot(a, b):
    """Patient-by-seed crossed bootstrap 95% CI of the seed-paired difference a - b (amendment 2, printed
    where seeds are a factor; posthoc_v2.crossed_boot, 10,000 resamples, seed 0)."""
    import posthoc_v2 as PH
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    D = np.array([[a[s]["pos"][p] - b[s]["pos"][p] for s in seeds] for p in sorted(TEST_PATIENTS)])
    return [float(x) for x in PH.crossed_boot(D, nboot=PLUG_BOOT, seed=PLUG_BOOT_SEED)[0]]


def _a2_note(c, fam, m, ps=None):
    """The src-comment statistics of an amendment 2 contrast."""
    return (f"paired t (df 5) p {c['p_t']:.4g}, exact sign-flip p {c['p_sf']:.4g}, patient bootstrap 95% CI "
            f"[{c['boot'][0]:+.2f}, {c['boot'][1]:+.2f}]"
            + ("" if ps is None else f", patient-by-seed bootstrap 95% CI [{ps[0]:+.2f}, {ps[1]:+.2f}]")
            + (f"; Holm within {fam} (m = {m}), members without a complete result at p = {AMEND2_MISSING_P:g}; "
               "exploratory, pre-specified rule (amendment 2)" if fam else ""))


def _cta_contrast(new, ref):
    """CTA-D and CTA-E for one row: per patient, the seed-paired difference of its TotalSegmentator-label run and
    its reference-label run, its 95% paired t CI and two-sided paired t p (CTA-D), and the equivalence test at +/- 1 mm with
    its 90% t CI (CTA-E, posthoc_v2.tost)."""
    import posthoc_v2 as PH
    c = _plug_contrast(new, ref)
    d = np.array([c["per_patient"][p] for p in sorted(TEST_PATIENTS)])
    if float(d.std(ddof=1)) == 0.0:   # equal per-patient differences: the t statistics are undefined
        c["p_t"] = 1.0 if c["mean"] == 0 else 0.0
        c["p_tost"], c["ci90"] = (0.0 if abs(c["mean"]) < CTA_TOST_MARGIN else 1.0), [c["mean"], c["mean"]]
    else:
        t = PH.tost(d, CTA_TOST_MARGIN)
        c["p_tost"], c["ci90"] = float(t["p_tost_t"]), [float(x) for x in t["ci90_t"]]
    c["ps_boot"] = _ps_boot(new, ref)
    return c


def _cta_block(S, prov):
    """The CT-anatomy block of tab:pluggability (amendment 2): rows R1 to R5 in two families, CTA-D (difference) and
    CTA-E (equivalence). -> (lines, rows with results, rows). Reads, for each row and seed, the evaluation on
    TotalSegmentator labels (CTA_NEW_ROOT; a job whose identity checks failed writes a .failed.json instead, which is
    not opened) and its reference rung (CTA_REF_ROOT); the analysis summary, when it exists, must give the same
    per-patient differences. prov: the git-commit counter of the plug result files read (updated in place)."""
    import posthoc_v2 as PH
    mc = "\\multicolumn{7}{@{}>{\\raggedright\\arraybackslash}p{0.97\\textwidth}@{}}"
    L = [f"{mc}{{\\textit{{CT anatomy}} (label volume): TotalSegmentator v2~\\cite{{totalseg}} labels (task total, "
         f"{CTA_MODE} mode, {CTA_RES_MM:.1f}~mm) minus the reference labels, same rung and episodes, "
         "seeds 0 to 2. Holm $p$: CTA-D, paired $t$; equivalence: CTA-E, TOST at $\\pm 1$~mm. "
         f"Families: CTA-D ($m = {AMEND2_FAMILY_M['CTA-D']}$) and CTA-E ($m = {AMEND2_FAMILY_M['CTA-E']}$).}} \\\\ "
         f"% src: {_T('src a2 cta')} and "
         f"Appendix cta (rows R1 to R5, {rel(CTA_NEW_ROOT)} minus {rel(CTA_REF_ROOT)}, contact on, DR off, R4 and R5 "
         "final weights" + _T("cta label volume") + ")"]
    if _blinded(CTA_NEW_ROOT):
        # blinded build: no file under the CT-anatomy root is read or tested for existence; the reference rungs (the
        # v2.1 rerun, not a blinded job) give each row's reference error for the slot header and plug_rows.json
        refs = []
        for rid, pat, w, lab, priv in CTA_ROWS:
            pr = {s: os.path.join(CTA_REF_ROOT, pat.format(seed=s)) for s in CTA_SEEDS}
            ref = _plug_cell(pr, w, False)
            for sd, x in ref.items():
                if _status_of(x["d"]) != CTA_REF_STATUS:
                    raise SystemExit(f"{rel(x['path'])}: analysis_status {_status_of(x['d'])!r}, expected "
                                     f"{CTA_REF_STATUS!r}")
            refs.append((rid, lab, priv, _cell_mean(ref) if len(ref) == len(CTA_SEEDS) else None, pr))
        CTA_STATE.update(blinded=True, refs=refs)
        L.append(f"{mc}{{\\quad\\textit{{Pending.}} CTA-D and CTA-E, {A2_BLIND_TXT}. Rows: "
                 + "; ".join(f"{rid}, {lab}" + ("$^{\\dagger}$" if priv else "") for rid, lab, priv, _, _ in refs)
                 + f".}} \\\\ % src: {_T('src a2 cta rows')} rows R1 to R5; {rel(CTA_NEW_ROOT)} not "
                 "read (BLIND_A2); reference rungs " + "; ".join(
                     f"{rid} {_srcp(pr[0])} (seed 0 shown) err " + ("missing" if e is None else f"{e:.4f} mm")
                     for rid, _, _, e, pr in refs))
        L.append(f"{mc}{{\\quad\\textit{{{CTA_IN_SAMPLE}}}}} \\\\ % src: " + _T("cta in-sample src"))
        L.append(f"{mc}{{\\quad\\textit{{Not run.}} The CUT and CycleGAN stages and the replication cohort.}} \\\\ "
                 "% src: " + _T("cta not run src"))
        L.append(_cta_not_reported(mc))
        log(f"  ctanat: {A2_BLIND_TXT}; reference rungs " + ", ".join(
            f"{rid} {('-' if e is None else f'{e:.3f}')}" for rid, _, _, e, _ in refs))
        return L, 0, len(CTA_ROWS)
    CTA_STATE.update(blinded=False, refs=[], rows=[], prem_fail=[])
    failed = lambda p: (not os.path.exists(p)) and os.path.exists(p[:-5] + ".failed.json")   # noqa: E731
    prem_fail = [f for f in CTA_PREMISE if failed(os.path.join(CTA_NEW_ROOT, f))]
    prem_have = [f for f in CTA_PREMISE if os.path.exists(os.path.join(CTA_NEW_ROOT, f))]
    log(f"  ctanat: pose-oracle and static identity premise: {len(prem_have)} of {len(CTA_PREMISE)} passed"
        + (f", FAILED {prem_fail}" if prem_fail else ""))
    rows, con = [], {}
    for rid, pat, w, lab, priv in CTA_ROWS:
        pn = {s: os.path.join(CTA_NEW_ROOT, pat.format(seed=s)) for s in CTA_SEEDS}
        pr = {s: os.path.join(CTA_REF_ROOT, pat.format(seed=s)) for s in CTA_SEEDS}
        fail = [s for s in CTA_SEEDS if failed(pn[s])]
        ref = _plug_cell(pr, w, False)
        for sd, x in ref.items():
            if _status_of(x["d"]) != CTA_REF_STATUS:
                raise SystemExit(f"{rel(x['path'])}: analysis_status {_status_of(x['d'])!r}, expected {CTA_REF_STATUS!r}")
        new = {} if prem_fail else _plug_cell(pn, w, True)
        for sd, x in new.items():
            g = _dig(x["d"], "provenance", "git") or {}
            prov[(str(g.get("hash"))[:12], bool(g.get("dirty")))] += 1
            tg = x["d"].get("ctanat") or {}
            pv = _dig(x["d"], "provenance", "v21_plug_ctanat") or {}
            if tg.get("smoke") or pv.get("smoke"):
                raise SystemExit(f"{rel(x['path'])}: a smoke output, not a held-out result")
            if tg.get("episode_identity_ok") is not True:
                raise SystemExit(f"{rel(x['path'])}: ctanat.episode_identity_ok is not true")
            if os.path.realpath(SD.map_recorded(str(_dig(pv, "ref", "path")))) != os.path.realpath(pr[sd]):
                raise SystemExit(f"{rel(x['path'])}: checked against {_dig(pv, 'ref', 'path')}, not {rel(pr[sd])}")
        complete = len(new) == len(CTA_SEEDS) and len(ref) == len(CTA_SEEDS) and not fail
        if complete:
            con[rid] = _cta_contrast(new, ref)
        rows.append((rid, lab, priv, w, pn, pr, new, ref, fail))
    ids = [r[0] for r in CTA_ROWS]
    hd = dict(zip(ids, PH.holm([con[r]["p_t"] if r in con else AMEND2_MISSING_P for r in ids])))
    he = dict(zip(ids, PH.holm([con[r]["p_tost"] if r in con else AMEND2_MISSING_P for r in ids])))
    # the analysis job's per-patient, seed-paired differences must equal the ones computed here
    fs = os.path.join(CTA_NEW_ROOT, CTA_SUMMARY)
    if os.path.exists(fs) and not prem_fail:
        sm = _plug_json(fs)
        by = {}
        for r in sm.get("rows") or []:
            by[(os.path.realpath(SD.map_recorded(r["file"])), r.get("weights"), r.get("contact"))] = r
        for rid, lab, priv, w, pn, pr, new, ref, fail in rows:
            for sd, x in new.items():
                r = by.get((os.path.realpath(pn[sd]), w, "on"))
                if r is None:
                    raise SystemExit(f"{rel(fs)}: no row for {rel(pn[sd])} ({w}, contact on)")
                if sd not in ref:
                    continue
                pd_ = r["metrics"][CTA_METRIC]["per_patient_diff"]
                for p in TEST_PATIENTS:
                    mine = x["pos"][p] - ref[sd]["pos"][p]
                    if abs(pd_[p] - mine) > 1e-9 * max(1.0, abs(mine)):
                        raise SystemExit(f"{rel(fs)}: {rid} seed {sd} patient {p} difference {pd_[p]} differs from "
                                         f"{mine} computed here")
        log(f"  ctanat: {rel(fs)} per-patient differences equal the ones computed here")
    CTA_STATE["prem_fail"] = prem_fail
    if prem_fail:
        L.append(f"{mc}{{\\quad No CT-anatomy row is reported: with TotalSegmentator labels, the pose-oracle or static "
                 "rung did not give the episodes of the reference labels.} \\\\ % src: amendment 2 (identity check "
                 "failed): " + ", ".join(rel(os.path.join(CTA_NEW_ROOT, f[:-5] + ".failed.json")) for f in prem_fail))
    n_done, pend = 0, []
    for rid, lab, priv, w, pn, pr, new, ref, fail in rows:
        if prem_fail:
            break
        name = f"{rid}: {lab.replace('per-tissue stage (SonoSPADE)', 'SonoSPADE')}"   # one name per stage
        e_ref = _cell_mean(ref)
        rnote = (f"reference labels err {e_ref:.4f} mm (seeds {sorted(ref)}) {_srcp(pr[0])} (seed 0 shown)"
                 if ref else f"reference {_srcp(pr[0])} missing")
        if not new and not fail:     # nothing to print yet: one 'Pending.' line for the block, as in the slots above
            pend.append((f"{rid}, {lab}" + ("$^{\\dagger}$" if priv else ""), f"{_srcp(pn[0])} ({rnote})"))
            log(f"ctanat   {rid:20s} CTA-D/E   pending | ref {('-' if e_ref is None else f'{e_ref:.3f}')}")
            continue
        if fail:
            L.append(f"{mc}{{\\quad {name}: not reported, since its identity checks with TotalSegmentator labels "
                     f"failed for seed {', '.join(map(str, fail))}.}} \\\\ "
                     f"% src: amendment 2; " + ", ".join(rel(pn[s][:-5] + ".failed.json") for s in fail)
                     + f"; enters CTA-D and CTA-E at p = {AMEND2_MISSING_P:g}")
            log(f"ctanat   {rid:20s} identity check FAILED seeds {fail}: not reported")
            continue
        n_done += bool(new)
        c = con.get(rid)
        err = _cell_mean(new)
        role = "CTA-D/E" + ("$^{\\dagger}$" if priv else "") + ("$^{\\ddagger}$" if new and c is None else "")
        hd_txt = f"${hd[rid]:.3f}${S}" if c else "pending"
        diff = ""
        if c:
            hold = hd[rid] < AMEND2_ALPHA and c["sign"] >= AMEND2_DECISION["min_sign_agree"]
            diff = f"{sg(c['mean'])}{AST if hold else ''} {ci_txt(c['ci'])}, {ks(c)}{S}"
        kp = next(iter(new.values()))["kp"] if new else ""
        note = (f"seeds {sorted(new)}; err {_srcp(pn[min(pn)])} (seed {min(pn)} shown) {kp}" if new else
                f"pending; err {_srcp(pn[min(pn)])} (seed {min(pn)} shown)") + f"; {rnote}"
        if c:
            note += "; diff TotalSegmentator minus reference labels, " + _a2_note(c, "CTA-D", AMEND2_FAMILY_M["CTA-D"],
                                                                                  c["ps_boot"])
        L.append(f"\\quad {name} & {role} & & {'pending' if err is None else f'{err:.2f}{S}'} & {diff} & {hd_txt} & \\\\ "
                 f"% src: {note}")
        CTA_STATE["rows"].append({"rid": rid, "lab": lab, "priv": priv, "err": err, "ref": e_ref, "c": c,
                                  "hd": hd[rid] if c else None, "he": he[rid] if c else None, "src": note})
        if c:
            eqv = he[rid] < AMEND2_ALPHA
            L.append(f"\\quad\\quad Equivalence within $\\pm 1$~mm (TOST) & CTA-E & & & "
                     f"90\\% {ci_txt(c['ci90'])}{AST if eqv else ''}, {ks(c)}{S} & ${he[rid]:.3f}${S} & \\\\ % src: "
                     f"TOST at +/- {CTA_TOST_MARGIN:g} mm{_T('cta tost test')}, TOST p {c['p_tost']:.4g}, "
                     f"Holm within CTA-E (m = {AMEND2_FAMILY_M['CTA-E']}) on the TOST p, members without a complete "
                     f"result at p = {AMEND2_MISSING_P:g}; holds when Holm p < {AMEND2_ALPHA}, no sign count")
        # R1 invariance: the CT-mask servo base of SonoSPADE sees label slices only, so it equals R1 (per seed)
        if rid == CTA_INVARIANCE["equals"] and new:
            pi = {s: os.path.join(CTA_NEW_ROOT, CTA_INVARIANCE["file"].format(seed=s)) for s in CTA_SEEDS}
            inv = _plug_cell({s: p for s, p in pi.items() if s in new}, w, True)
            dm = {s: max(abs(float(a["final_pos"]) - float(b["final_pos"]))
                         for a, b in zip(inv[s]["b"]["episodes"], new[s]["b"]["episodes"])) for s in sorted(inv)}
            if dm:
                over = [s for s, v in dm.items() if v > CTA_INVARIANCE["tol_mm"]]
                L.append("% " + _T("cta invariance moved") + ", the CT-mask base invariance row: "
                         f"\\quad\\quad SonoSPADE CT-mask base minus R1, largest per-episode $|\\Delta|$"
                         + (_T("r1 flag") if over else "")
                         + " & secondary & & & " + ", ".join(f"s{s} {v:.2g}~mm" for s, v in dm.items()) + f"{S} & & "
                         + f" \\\\ % src: {_srcp(pi[min(dm)])} vs R1 (seed {min(dm)} shown), contact on; "
                         f"tolerance {CTA_INVARIANCE['tol_mm']:g} mm" + (f"; exceeded for seeds {over}: {_T('dev listed')}" if over else ""))
                log(f"ctanat   R1 stage invariance max |diff| per seed {dm}" + (f" EXCEEDS for {over}" if over else ""))
        log(f"ctanat   {rid:20s} CTA-D/E   seeds {sorted(new)} err "
            f"{('pending' if err is None else f'{err:.3f}'):>8s} | ref {('-' if e_ref is None else f'{e_ref:.3f}')} | "
            + (f"diff {c['mean']:+.3f} t[{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] {ks(c)} p {c['p_t']:.4f} "
               f"Holm(CTA-D) {hd[rid]:.4f} | TOST p {c['p_tost']:.4f} 90%[{c['ci90'][0]:+.2f}, {c['ci90'][1]:+.2f}] "
               f"Holm(CTA-E) {he[rid]:.4f}" if c else "diff -"))
    if pend:
        L.append(f"{mc}{{\\quad\\textit{{Pending.}} CTA-D and CTA-E: " + "; ".join(n for n, _ in pend)
                 + ".} \\\\ % src: pending; files " + "; ".join(f for _, f in pend) + " (seed 0 shown)")
    L.append(f"{mc}{{\\quad\\textit{{{CTA_IN_SAMPLE}}}}} \\\\ % src: " + _T("cta in-sample src"))
    L.append(f"{mc}{{\\quad\\textit{{Not run.}} The CUT and CycleGAN stages and the replication cohort.}} \\\\ "
             "% src: " + _T("cta not run src"))
    L.append(_cta_not_reported(mc))
    return L, n_done, len(CTA_ROWS)


def pluggability(R, out, plug_root=PLUG_ROOT, seg_plug_root=SEG_PLUG_ROOT):
    """tab:pluggability: each slot with the authors' reference row and the published methods plugged into it."""
    import posthoc_v2 as PH
    log("\n== tab:pluggability (pluggability.tex), EXPLORATORY v2.1 (amendments 1 and 2) ==")
    _check_families()
    _amend2()     # before any output of the amendment 2 groups is read
    PLUG_REC.clear()
    cm = R["table4"]["cell_means_mm"]
    S = PROV if PLUG_PROVISIONAL else ""
    sb3_ok = _sb3_certified()
    body, n_plug_rows, n_plug_done = [], 0, 0
    twin_split = None   # the twin is two floats (one float was 72 pt taller than the page); split as Table 5
    prov = collections.Counter()   # git commit and dirty flag of every plug result file read
    for slot in PLUG_SLOTS:
        rows = slot["rows"]
        cells, extra = {}, {}
        for row in rows:
            paths, w, plug = _plug_paths(slot, row, plug_root, seg_plug_root)
            if row["group"] and _blinded(row["group"]):   # blinded amendment 2 group: nothing is read
                cells[row["id"]] = ({}, paths, w, plug)
                continue
            cell = _plug_cell(paths, w, plug, row["bn"])
            cells[row["id"]] = (cell, paths, w, plug)
            for c_ in (cell.values() if row["v21"] else ()):
                if _status_of(c_["d"]) != "exploratory, v2.1":
                    raise SystemExit(f"{rel(c_['path'])}: analysis_status {_status_of(c_['d'])!r}, expected "
                                     "'exploratory, v2.1'")
            if plug and cell:
                _require_gates(slot, row)
                for c in cell.values():
                    g = _dig(c["d"], "provenance", "git") or {}
                    prov[(str(g.get("hash"))[:12], bool(g.get("dirty")))] += 1
            if row["seeds"] == SEED0:   # optional extra seeds of a seed-0 design are not part of the table
                more = [s for s in (1, 2) if os.path.exists(_plug_paths(slot, dict(row, seeds=(s,)), plug_root,
                                                                         seg_plug_root)[0][s])]
                if more:
                    log(f"  note {slot['key']}/{row['id']}: optional seeds {more} exist; the table keeps seed 0 "
                        "(the amendment's design)")
            if row["bn"] == "eval" and cell:   # BatchNorm train mode, reported beside the eval-mode primary
                extra[row["id"]] = _plug_cell(paths, w, plug, "train")
        ref = next(r for r in rows if r["role"] == "ref")
        ref_cell = cells[ref["id"]][0]
        log(f"  {slot['key']}: identical-episode audit {_audit_state(slot['key'])}")
        # contrasts and the family test
        con = {}
        for row in rows:
            cell = cells[row["id"]][0]
            complete = bool(cell) and len(cell) == len(row["seeds"])
            if row["role"] == "ref" or not complete or not set(row["seeds"]) <= set(ref_cell):
                continue
            if row["learned"] and len(cell) < 3:
                continue
            con[row["id"]] = _plug_contrast(cell, ref_cell)
        fam = [r["id"] for r in rows if r["role"] == "family"]
        p_fam = [con[f]["p_t"] if f in con else 1.0 for f in fam]
        holm = dict(zip(fam, PH.holm(p_fam)))
        # amendment 2 families, never pooled with the family above: MASK-P (each member minus the
        # stage-matched U-Net, i.e. the slot's reference), LPIPS-R (minus the paper's S), and IMN-T (each ImageNet row
        # minus its from-scratch twin, an amendment 1 row of this slot, same stage and seed)
        holm2, con_t, a2_sec = {}, {}, {}
        for fn in dict.fromkeys(r["fam2"][0] for r in rows if r["role"] == "a2"):
            mem = [r["id"] for r in rows if r["role"] == "a2" and r["fam2"][0] == fn]
            if len(mem) != AMEND2_FAMILY_M[fn]:
                raise SystemExit(f"pluggability {slot['key']}: {fn} has {len(mem)} members {mem}, amendment 2 fixes "
                                 f"m = {AMEND2_FAMILY_M[fn]}")
            holm2[fn] = dict(zip(mem, PH.holm([con[x]["p_t"] if x in con else AMEND2_MISSING_P for x in mem])))
        imt = [r for r in rows if "IMN-T" in r["fam2"]]
        for r in imt:
            a, b = cells[r["id"]][0], cells[r["twin"]][0]
            if len(a) == len(r["seeds"]) and set(r["seeds"]) <= set(b):
                con_t[r["id"]] = _plug_contrast(a, b)
        if imt:
            holm2["IMN-T"] = dict(zip([r["id"] for r in imt], PH.holm(
                [con_t[r["id"]]["p_t"] if r["id"] in con_t else AMEND2_MISSING_P for r in imt])))
        # LPIPS-R: the two arms of each seed share one imitation checkpoint; when their errors after imitation differ
        # for a SonoSPADE seed, LPIPS-R prints as secondary (no 'holds') together with that difference
        for r in rows:
            if "LPIPS-R" in r["fam2"] and r["id"] in con:
                ca, cb = cells[r["id"]][0], ref_cell
                imit = {}
                for sd in r["seeds"]:
                    ea, eb = ca[sd]["d"]["after_imitation"], cb[sd]["d"]["after_imitation"]
                    imit[sd] = (ea["pos"] == eb["pos"], max(abs(float(x["final_pos"]) - float(y["final_pos"]))
                                                           for x, y in zip(ea["episodes"], eb["episodes"])))
                a2_sec[r["id"]] = imit
        m = PLUG_FAMILY_M[slot["key"]]
        dd = 3 if slot["key"] == "reward" else 2   # reward differences are below 0.1 mm: 3 decimals
        SR = {"key": slot["key"], "title": slot["title"], "iface": slot["iface"], "m": m, "dd": dd,
              "ref_id": ref["id"], "rows": [], "pending": [], "groups": [], "not_run": slot["not_run"],
              "cells": {rid: v[0] for rid, v in cells.items()}, "defs": rows, "audit": _audit_state(slot["key"]),
              "holm2_m": {fn: AMEND2_FAMILY_M[fn] for fn in holm2}}
        PLUG_REC.append(SR)
        if slot["key"] == T5_SPLIT_BEFORE:
            twin_split = len(body)
        set_txt, set_src = slot["setting"], _cost_src(slot["key"], plug_root)
        if slot["key"] == "renderer":   # the plugged rows and the archived physics row come from different benchmarks
            fb = os.path.join(plug_root, "renderer", "bench", "bench.json")
            pb = _dig(_plug_json(fb), "renderers", "physics", "devices", "cpu", "frame", "median_ms") \
                if os.path.exists(fb) else None
            if pb is not None:
                set_txt += (f"; plugged rows timed under load, so upper bounds (physics took {_ms(pb)}{S} under the "
                            "same load), the physics row from the earlier benchmark")
                set_src += f"; physics in the v2.1 benchmark: {rel(fb)} renderers.physics.devices.cpu.frame.median_ms"
        body += ["\\addlinespace[2pt]" if body else "",
                 f"{TW_MC}"
                 f"{{\\textit{{{slot['title']}}}: {set_txt}; family $m = {m}$}} "
                 f"\\\\ % src: {_T('src a1 stats')} (family sizes texture 4, mask 7, controller 10, reward 2, renderer 2); "
                 f"cost: {set_src}"]
        seed0_rows = [r for r in rows if r["seeds"] == SEED0 and cells[r["id"]][0]]
        has = {g: any(cells[r["id"]][0] for r in rows if r["group"] == g) for g in {r["group"] for r in rows} if g}
        group = None
        pending = []   # rows with nothing to print yet: listed on one line per slot
        tail = []      # the amendment 2 groups print after the slot's pending line
        for row in rows:
            cell, paths, w, plug = cells[row["id"]]
            if row["group"] and row["group"] != group:
                fams, what = PLUG_GROUPS[row["group"]]
                tail.append(f"{TW_MC}"
                            f"{{\\quad\\textit{{{fams}:}} {what}"
                            + ("" if has[row["group"]] else
                               (": " + A2_BLIND_TXT if _blinded(row["group"]) else ": pending")) + ".} \\\\"
                            + f" % src: {_T('src a2 fam')} (families MASK-P 5, IMN-T 4, LPIPS-R 1; never pooled with "
                              f"amend1's)" + ("" if has[row["group"]] else
                                             "; pending; files " + ", ".join(
                                                 rel(_plug_paths(slot, r, plug_root, seg_plug_root)[0][r["seeds"][0]])
                                                 for r in rows if r["group"] == row["group"]) + " (seed 0 shown)"))
            if row["group"] and row["group"] != group:
                SR["groups"].append({"group": row["group"], "fams": PLUG_GROUPS[row["group"]][0],
                                     "fams_plain": _a2_plain(dict.fromkeys(
                                         f for r in rows if r["group"] == row["group"] for f in r["fam2"])),
                                     "what": PLUG_GROUPS[row["group"]][1], "has": has[row["group"]],
                                     "blinded": _blinded(row["group"])})
            group = row["group"]
            if row["group"] and not has[row["group"]]:
                n_plug_rows += 1
                log(f"{slot['key']:8s} {row['id']:20s} {'/'.join(row['fam2']) or 'secondary':9s} pending (group line)")
                continue
            err = _cell_mean(cell)
            if row["rkey"] and err is not None and len(cell) == 3:
                if abs(err - cm[row["rkey"]]) > 1e-9 * max(1.0, abs(err)):
                    raise SystemExit(f"pluggability {slot['key']}/{row['id']}: {err} differs from RESULTS.json "
                                     f"table4.cell_means_mm[{row['rkey']}] = {cm[row['rkey']]}")
            if row["xref"] and err is not None and len(cell) == len(row["seeds"]):
                xf, xk = row["xref"]
                xd = json.load(open(os.path.join(ROOT, xf)))
                if _status_of(xd) != "exploratory, v2.1":
                    raise SystemExit(f"{xf}: analysis_status {_status_of(xd)!r}, expected 'exploratory, v2.1'")
                xv = _dig(xd, *xk)
                if xv is None or abs(err - float(xv)) > 1e-9 * max(1.0, abs(err)):
                    raise SystemExit(f"pluggability {slot['key']}/{row['id']}: {err} differs from {xf} "
                                     f"{'.'.join(xk)} = {xv}")
            if plug:
                n_plug_rows += 1
                n_plug_done += bool(cell)
            mk = S if (plug or row["v21"]) else ""
            # the record's provisional flag keeps its meaning although no mark S is printed
            # (PROV is empty): plug_rows.json depends on it (PLUG_ROWS_MD5), so it reads PLUG_ROWS_PROVISIONAL; the
            # \verifyfill boxes of Table 5 follow PLUG_PROVISIONAL (_vf), now cleared
            prov_ = bool(PLUG_ROWS_PROVISIONAL and PROV_MARK and (plug or row["v21"]))
            sc_txt, co_txt, src = _plug_score_cost(R, slot, row, cell, plug_root, seg_plug_root)
            if not cell and not sc_txt and co_txt in ("", "n/a"):
                pending.append(row)
                SR["pending"].append(row)
                log(f"{slot['key']:8s} {row['id']:20s} {ROLE_TXT[row['role']]:9s} pending")
                continue
            c = con.get(row["id"])
            a2 = row["role"] == "a2"
            fn = row["fam2"][0] if a2 else None
            # LPIPS-R with a post-imitation mismatch prints as secondary (amendment 2)
            demote = a2 and row["id"] in a2_sec and not all(v[0] for v in a2_sec[row["id"]].values())
            diff_txt = holm_txt = tp_txt = sf_txt = c90_txt = ""
            if c is not None:
                tp_txt, sf_txt, c90_txt = (f"{_p3(c['p_t'])}{S}", f"{pfrac(c['p_sf'])}{S}",
                                           f"{ci_txt(c['tost_ci90'], dd)}{S}")
                if a2:
                    hold = (not demote and holm2[fn][row["id"]] < AMEND2_ALPHA
                            and c["sign"] >= AMEND2_DECISION["min_sign_agree"])
                else:
                    hold = row["role"] == "family" and holm[row["id"]] < 0.05 and c["sign"] >= 5
                diff_txt = f"{sg(c['mean'], dd)}{AST if hold else ''} {ci_txt(c['ci'], dd)}, {ks(c)}{S}"
                holm_txt = (f"{_p3(holm[row['id']])}{S}" if row["role"] == "family" else
                            f"{_p3(holm2[fn][row['id']])}{S}" if a2 and not demote else "n/a")
            elif (row["role"] == "family" or (a2 and not demote)) and cell:
                holm_txt = "pending"
            err_txt = "pending" if err is None else f"{err:.2f}{mk}"
            lab = row["label"] + (f"~\\cite{{{','.join(row['cites'])}}}" if row["cites"] else "")
            if slot["key"] == "ctrl" and row["role"] == "family" and not row["bn"] and not sb3_ok:
                lab += ", RL slot without a passing positive control"
            if demote:
                lab += ", post-imitation error differs from the paper arm's"
            short = bool(cell) and (len(cell) < len(row["seeds"]) or (row["learned"] and len(cell) < 3))
            role = (("secondary" if demote else fn) if a2 else ROLE_TXT[row["role"]]) + (
                "$^{\\dagger}$" if row["priv"] else "") + ("$^{\\ddagger}$" if short else "")
            seeds = sorted(cell)
            kp = next(iter(cell.values()))["kp"] if cell else ""
            note = (f"seeds {seeds}; " if cell else "pending; ") + (
                f"err {_srcp(paths[min(paths)])} (seed {min(paths)} shown) {kp}" if plug else
                f"err archived v2 cells {_srcp(paths[min(paths)])} (seed {min(paths)} shown) {kp}"
                + (f", equal to RESULTS.json table4.cell_means_mm[{row['rkey']}]" if row["rkey"] and len(cell) == 3
                   else ""))
            if c is not None:
                note += (f"; diff paired t (df 5) vs the {ref['id']} row on seeds {c['seeds']}, computed here: t p "
                         f"{c['p_t']:.4g}, exact sign-flip p {c['p_sf']:.4g}, patient bootstrap 95% CI "
                         f"[{c['boot'][0]:+.2f}, {c['boot'][1]:+.2f}] (sensitivity)" + _posthoc_note(c, dd))
                if row["role"] == "family":
                    note += f"; Holm over m = {m}, members without a complete result at p = 1"
                if a2:
                    ps = _ps_boot(cell, ref_cell)
                    note += ("" if ps is None else f"; patient-by-seed bootstrap 95% CI [{ps[0]:+.2f}, {ps[1]:+.2f}]") \
                        + (f"; Holm within {fn} (m = {AMEND2_FAMILY_M[fn]}), members without a complete result at "
                           f"p = {AMEND2_MISSING_P:g}; exploratory, pre-specified rule (amendment 2)" if not demote
                           else f"; printed as secondary ({_T('src a2 lpips')}): {_T('dev listed')}")
            if a2 and row["id"] in a2_sec:
                note += "; post-imitation error identical to the paper arm's: " + ", ".join(
                    f"s{sd} {'yes' if v[0] else 'NO'} (max per-episode |diff| {v[1]:.3g} mm)"
                    for sd, v in sorted(a2_sec[row["id"]].items()))
            if row["role"] == "ref" and slot["key"] == "reward":
                note += f"; archived v2 value RESULTS.json table4.cell_means_mm[ctmask/sonospade/final] = " \
                        f"{cm['ctmask/sonospade/final']:.4f} (not a parity check: MPS reruns differ)"
            if row["xref"] and err is not None and len(cell) == len(row["seeds"]):
                note += f"; equal to {row['xref'][0]} {'.'.join(row['xref'][1])}"
            if row["id"] == "lotus_orig":
                note += f"; label 'published default parameters': {LOTUS_DEFAULTS_SRC}"
            if row["id"] == "label_painted" and c is not None:
                # this row has a TOST at +/- 1 mm (Q10, outside the families): the Q10 record must give the same TOST
                # p and the same patient-by-seed interval
                q10 = json.load(open(os.path.join(ROOT, PAINTED_XREF[0])))["results"]["painting"]
                qt, qc = q10["tost_P_minus_So"], q10["P_minus_So"]["contrast"]
                if (abs(qt["p_tost"] - c["tost_p"]) > 1e-9 or abs(qc["mean"] - c["mean"]) > 1e-9
                        or any(abs(x - y) > 1e-9 for x, y in zip(qc["ci_boot_patient_seed"], c["ps_nested"]))):
                    raise SystemExit("pluggability texture/label_painted: TOST or patient-by-seed CI differs from "
                                     f"{PAINTED_XREF[0]} results.painting")
                note += (f"; {_T('painted tost note')}, equal to {PAINTED_XREF[0]} "
                         "results.painting.tost_P_minus_So.p_tost and P_minus_So.contrast.ci_boot_patient_seed")
            (tail if row["group"] else body).append(
                        f"\\quad {_print_label(lab)} & {role} & {co_txt} & {err_txt} & {diff_txt} & {tp_txt} & {holm_txt} & {sf_txt} & "
                        f"{c90_txt} & {sc_txt} \\\\ % src: {note}" + (f"; {src}" if src else ""))
            SR["rows"].append({
                "kind": "row", "id": row["id"], "label": lab, "plain": row["label"], "role": row["role"], "fam": fn,
                "priv": row["priv"], "laccess": row["laccess"], "short": short, "demote": demote, "cost": co_txt,
                "score": sc_txt, "err": err, "v21": prov_, "c": c, "seeds": seeds, "designed": list(row["seeds"]),
                "holm": (holm[row["id"]] if row["role"] == "family" and c is not None else
                         holm2[fn][row["id"]] if a2 and not demote and c is not None else None),
                "hold": (hold if c is not None else None), "group": row["group"],
                "src": note + (f"; {src}" if src else "")})
            log(f"{slot['key']:8s} {row['id']:20s} {ROLE_TXT[row['role']]:9s} seeds {seeds} err "
                f"{('pending' if err is None else f'{err:.3f}'):>8s} | "
                + (f"diff {c['mean']:+.3f} t[{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] {ks(c)} p {c['p_t']:.4f} "
                   f"sf {c['p_sf']:.4f} boot[{c['boot'][0]:+.2f}, {c['boot'][1]:+.2f}]"
                   + (f" Holm(m={m}) {holm[row['id']]:.4f} {'HOLDS' if row['role'] == 'family' and holm[row['id']] < 0.05 and c['sign'] >= 5 else '-'}"
                      if row["role"] == "family" else "")
                   + (f" Holm({fn}, m={AMEND2_FAMILY_M[fn]}) {holm2[fn][row['id']]:.4f} {'HOLDS' if hold else '-'}"
                      + (" SECONDARY (post-imitation mismatch)" if demote else "") if a2 else "") if c else "diff -")
                + f" | score {sc_txt or '-'} | cost {co_txt or '-'}")
            # IMN-T (amendment 2): the same ImageNet row minus its from-scratch twin, same stage and seed
            if "IMN-T" in row["fam2"]:
                ct_, h_ = con_t.get(row["id"]), holm2["IMN-T"][row["id"]]
                tw_paths = cells[row["twin"]][1]
                if ct_ is not None:
                    hold_t = h_ < AMEND2_ALPHA and ct_["sign"] >= AMEND2_DECISION["min_sign_agree"]
                    dt, ht = f"{sg(ct_['mean'], dd)}{AST if hold_t else ''} {ci_txt(ct_['ci'], dd)}, {ks(ct_)}{S}", \
                        f"${h_:.3f}${S}"
                    nt = (f"diff vs the {row['twin']} row (from scratch, amendment 1) on seeds {ct_['seeds']}, "
                          f"computed here: " + _a2_note(ct_, "IMN-T", AMEND2_FAMILY_M["IMN-T"],
                                                        _ps_boot(cell, cells[row["twin"]][0])))
                else:
                    dt, ht, nt = "", "pending", "pending"
                tp_t, sf_t, c9_t = ((f"{_p3(ct_['p_t'])}{S}", f"{pfrac(ct_['p_sf'])}{S}",
                                     f"{ci_txt(ct_['tost_ci90'], dd)}{S}") if ct_ is not None else ("", "", ""))
                tail.append(f"\\quad\\quad Minus the same network trained from scratch & IMN-T & & & {dt} & {tp_t} & "
                            f"{ht} & {sf_t} & {c9_t} & \\\\ % src: {nt}; twin err {_srcp(tw_paths[min(tw_paths)])} "
                            f"(seed {min(tw_paths)} shown)")
                SR["rows"].append({"kind": "imnt", "id": row["id"] + "@IMN-T", "parent": row["id"],
                                   "label": lab + ", minus the same network trained from scratch",
                                   "plain": row["label"] + ", minus from scratch", "role": "a2", "fam": "IMN-T",
                                   "priv": False, "laccess": False, "short": False, "demote": False, "cost": "",
                                   "err": None, "v21": True, "c": ct_, "holm": h_ if ct_ is not None else None,
                                   "hold": hold_t if ct_ is not None else None, "group": row["group"],
                                   "src": f"{nt}; twin err {_srcp(tw_paths[min(tw_paths)])} (seed {min(tw_paths)} shown)"})
                log(f"{slot['key']:8s} {row['id'] + '@IMN-T':20s} IMN-T     "
                    + (f"diff {ct_['mean']:+.3f} t[{ct_['ci'][0]:+.2f}, {ct_['ci'][1]:+.2f}] {ks(ct_)} p "
                       f"{ct_['p_t']:.4f} Holm(IMN-T, m=4) {h_:.4f} {'HOLDS' if hold_t else '-'}" if ct_ else "pending"))
            # SAMUS seeds 0 to 2 against U-Net seeds 0 to 2 (amendment 2): secondary, only if all exist
            if row["group"] == "samus" and cell:
                p3 = _plug_paths(slot, dict(row, seeds=PLUG_SEEDS), plug_root, seg_plug_root)[0]
                if all(os.path.exists(p) for p in p3.values()):
                    c3cell = _plug_cell(p3, w, plug, row["bn"])
                    c3, ps3 = _plug_contrast(c3cell, ref_cell), _ps_boot(c3cell, ref_cell)
                    tail.append(f"\\quad\\quad Seeds 0 to 2, against U-Net seeds 0 to 2 & secondary"
                                + ("$^{\\dagger}$" if row["priv"] else "")
                                + f" & & {_cell_mean(c3cell):.2f}{S} & {sg(c3['mean'], dd)} {ci_txt(c3['ci'], dd)}, "
                                f"{ks(c3)}{S} & {_p3(c3['p_t'])}{S} & n/a & {pfrac(c3['p_sf'])}{S} & "
                                f"{ci_txt(c3['tost_ci90'], dd)}{S} & \\\\ % src: seeds {c3['seeds']}; err "
                                f"{_srcp(p3[0])} (seed 0 shown); diff vs the {ref['id']} row, computed here: "
                                + _a2_note(c3, None, None, ps3) + _posthoc_note(c3, dd))
                    log(f"{slot['key']:8s} {row['id'] + '@s0-2':20s} secondary diff {c3['mean']:+.3f} "
                        f"t[{c3['ci'][0]:+.2f}, {c3['ci'][1]:+.2f}] {ks(c3)}")
                    SR["rows"].append({"kind": "samus3", "id": row["id"] + "@s0-2", "parent": row["id"],
                                       "label": lab + ", seeds 0 to 2 against U-Net seeds 0 to 2",
                                       "plain": row["label"] + ", seeds 0 to 2", "role": "secondary", "fam": None,
                                       "priv": row["priv"], "laccess": False, "short": False, "demote": False,
                                       "cost": "", "err": _cell_mean(c3cell), "v21": True, "c": c3, "holm": None,
                                       "hold": None, "group": row["group"],
                                       "src": f"seeds {c3['seeds']}; err {_srcp(p3[0])} (seed 0 shown); diff vs the "
                                              f"{ref['id']} row, computed here: " + _a2_note(c3, None, None, ps3)})
            # the seed-0 line of the reference, which the seed-0 rows of this slot are paired with
            if row["role"] == "ref" and seed0_rows and 0 in cell:
                c0 = {0: cell[0]}
                s0_sc, _, s0_src = _plug_score_cost(R, slot, row, c0, plug_root, seg_plug_root, seeds=SEED0)
                e0 = _cell_mean(c0)
                body.append(f"\\quad\\quad Seed 0 only, paired with the seed-0 rows & reference & & {e0:.2f}{mk} & & & & "
                            f"& & {s0_sc} \\\\ % src: seed 0; err {_srcp(paths[0])} {cell[0]['kp']}" +
                            (f"; {s0_src}" if s0_src else ""))
                log(f"{slot['key']:8s} {row['id'] + '@seed0':20s} reference err {e0:.3f} score {s0_sc or '-'}")
                SR["rows"].append({"kind": "ref_seed0", "id": row["id"] + "@seed0", "parent": row["id"],
                                   "label": "Seed 0 only, for the seed-0 rows", "plain": "seed 0 only",
                                   "role": "ref", "fam": None, "priv": False, "laccess": False, "short": False,
                                   "demote": False, "cost": "", "err": e0, "v21": prov_, "c": None, "holm": None,
                                   "hold": None, "group": None,
                                   "src": f"seed 0; err {_srcp(paths[0])} {cell[0]['kp']}"})
            # BatchNorm train mode beside the eval-mode primary (amend1): secondary
            if row["id"] in extra and extra[row["id"]]:
                ct = extra[row["id"]]
                et = _cell_mean(ct)
                c2 = _plug_contrast(ct, ref_cell) if (len(ct) == len(row["seeds"]) and ref_cell) else None
                sct, _, _ = _plug_score_cost(R, slot, row, ct, plug_root, seg_plug_root)
                d2 = "" if c2 is None else f"{sg(c2['mean'], dd)} {ci_txt(c2['ci'], dd)}, {ks(c2)}{S}"
                tp2, sf2, c92 = ((f"{_p3(c2['p_t'])}{S}", f"{pfrac(c2['p_sf'])}{S}", f"{ci_txt(c2['tost_ci90'], dd)}{S}")
                                 if c2 is not None else ("", "", ""))
                kpt = next(iter(ct.values()))["kp"]
                n2 = (f"seeds {sorted(ct)}; err {_srcp(paths[min(paths)])} (seed {min(paths)} shown) {kpt}"
                      + ("" if c2 is None else
                         f"; diff paired t (df 5) vs the {ref['id']} row, computed here: t p {c2['p_t']:.4g}, exact "
                         f"sign-flip p {c2['p_sf']:.4g}, patient bootstrap 95% CI [{c2['boot'][0]:+.2f}, "
                         f"{c2['boot'][1]:+.2f}] (sensitivity)" + _posthoc_note(c2, dd))
                      + "; score success from the same episodes")
                body.append(f"\\quad\\quad BatchNorm in train mode & secondary"
                            + ("$^{\\ddagger}$" if len(ct) < 3 else "")
                            + f" & & {et:.2f}{S} & {d2} & {tp2} & {'n/a' if c2 else ''} & {sf2} & {c92} & {sct} "
                              f"\\\\ % src: {n2}")
                SR["rows"].append({"kind": "bn_train", "id": row["id"] + "@bn_train", "parent": row["id"],
                                   "label": lab + ", BatchNorm in train mode",
                                   "plain": row["label"] + ", BatchNorm in train mode", "role": "secondary",
                                   "fam": None, "priv": False, "laccess": False, "short": len(ct) < 3,
                                   "demote": False, "cost": "", "err": et, "v21": True, "c": c2, "holm": None,
                                   "hold": None, "group": None, "src": n2})
                log(f"{slot['key']:8s} {row['id'] + '@bn_train':20s} secondary err {et:.3f} | "
                    + (f"diff {c2['mean']:+.3f} t[{c2['ci'][0]:+.2f}, {c2['ci'][1]:+.2f}] {ks(c2)} p {c2['p_t']:.4f}"
                       if c2 else "diff -") + f" | score {sct or '-'}")
        if pending:
            parts = []
            for role in ("family", "secondary", "a2"):
                items = [r["label"] + (f"~\\cite{{{','.join(r['cites'])}}}" if r["cites"] else "")
                         + ("$^{\\dagger}$" if r["priv"] else "") + (f" ({', '.join(r['fam2'])})" if r["fam2"] else "")
                         for r in pending if r["role"] == role]
                if items:
                    parts.append(("later families" if role == "a2" else ROLE_TXT[role]) + ": " + "; ".join(items))
            files = ", ".join(rel(_plug_paths(slot, r, plug_root, seg_plug_root)[0][r["seeds"][0]]) for r in pending)
            body.append(f"{TW_MC}"
                        f"{{\\quad\\textit{{Pending.}} " + ". ".join(p[0].upper() + p[1:] for p in parts)
                        + f".}} \\\\ % src: pending; files {files} (seed 0 shown)")
        body += tail
        if slot["not_run"]:   # Table 2 holds the one "not evaluated" list; here a comment line
            body.append(f"% Not run{_T('not run note')}: {slot['not_run']} % src: {_T('src a1 notrun')}")
    # the CT-anatomy block (amendment 2), after the five slots of amendment 1
    cta, n_cta_done, n_cta = _cta_block(S, prov)
    body += ["\\addlinespace[2pt]"] + [_tw10(x) for x in cta]
    placeholder = n_plug_done == 0
    head = ("\\textbf{Placeholder: the v2.1 pluggability jobs have not run yet, so the result cells of the plugged "
            "methods are empty.} " if placeholder else "")
    # no tier or amendment words and no provisional mark in the caption; tables are cited by \ref (xr-hyper)
    cap = (head + "Supplementary twin of Table~\\ref{tab:pluggability} of the article, with the columns that "
           "Table~\\ref{tab:pluggability} leaves out: Label, "
           "Cost, the paired $t$ $p$, the exact sign-flip $p$, the 90\\% $t$ interval of the post hoc TOST, and Score. "
           "Exploratory, pre-specified rules. Reference components are marked (authors') or (prior). "
           "Values taken from the primary results stay primary. "
           "Each slot swaps one component of our reference pipeline. "
           "Every run uses the crossed held-out episodes of Table~\\ref{tab:ladder} of the article, contact on. "
           "Error: final translation error, the mean over seeds of patient-level means. "
           "Difference: from the slot's reference component, paired by seed, with its 95\\% paired $t$ interval and the "
           "patients, of 6, that share its sign. "
           "The methods of a family are tested with the two-sided paired $t$-test and Holm's adjustment over all $m$ members. "
           "A member without a complete result enters at $p = 1$. The printed Holm $p$ is then an upper bound. "
           "$^{\\ast}$Holm $p < 0.05$ and at least 5 of 6 patients share the sign. "
           "Secondary comparisons are never tested. "
           "TOST: post hoc and unadjusted, two one-sided paired $t$-tests at $\\pm 1$~mm, the pre-specified margin "
           "for the RL change, the label-painted stage and CTA-E; an interval inside $[-1, +1]$~mm shows "
           "equivalence within 1~mm. "
           "MASK-P (second mask source), IMN-T (pretrained backbone), LPIPS-R (LPIPS reward), CTA-D and CTA-E "
           "(CT-anatomy difference and equivalence): later pre-specified families (exploratory, Holm within each, "
           "never pooled with the slot families). IMN-T: minus the same network trained from scratch. "
           "CTA-E: TOST at $\\pm 1$~mm, Holm on the larger $p$, 90\\% $t$ interval; there $^{\\ast}$ needs no sign "
           "count. "
           "$^{\\dagger}$Privileged: uses simulator state that no physical rig provides. "
           "$^{\\ddagger}$Fewer seeds than designed, or a learned method below three seeds. "
           "Such a row gets no difference. "
           "SAM, MedSAM, SAMUS, the look-ahead and SIMUS run seed 0 by design and pair with seed 0 of the reference. "
           "BC and DAgger learn from the pose oracle and use BatchNorm in eval mode. "
           "Cost: the measurement, unit and device named in each slot header; medians over seeds, except the reward "
           "slot's mean per frame pair. Every cost is a benchmark or a timing inside the closed loop. "
           "KID and learnability as in the realism and learnability tables of \\ref{app:eval} of the article. "
           "MedSAM and SAMUS real-frame Dice are not reported, because their pretraining data may include the real "
           "test frames.")
    # not printed: the provisional footnote of PLUG_PROVISIONAL
    L = [GEN.format(src="outputs/v2_1/plug/ (v2.1 pluggability, exploratory under amendments 1 and 2), outputs/v2_1/checks/ "
                        "(admission gates, positive control), outputs/v2_1/label_controls/ (the label-painted stage), "
                        "the archived v2 files behind outputs/v2/RESULTS.json (table4, compute) and "
                        "outputs/v2/realism/realism_sector.json"),
         f"% plug rows with results: {n_plug_done} of {n_plug_rows}" + (" (PLACEHOLDER)" if placeholder else "")
         + f"; CT-anatomy rows with results: {n_cta_done} of {n_cta}",
         "% plug result files by git commit (provenance.git.hash, .dirty): "
         + "; ".join(f"{h} dirty={d}: {n} files" for (h, d), n in sorted(prov.items())),
         "% statistics: " + _T("twin stats src")
         + f"TOST +/- {TOST_MARGIN_MM:g} mm post hoc (margin of {TOST_MARGIN_SRC})",
         f"% amendment 2: {AMEND2_TXT} sha256 {AMEND2_SHA256} (equal to {AMEND2_SHA_TXT}); its Appendix "
         "constants asserted before any output of its groups was read; families MASK-P 5, IMN-T 4, LPIPS-R 1, CTA-D 5, "
         "CTA-E 5 (TOST +/- 1 mm), Holm within each, incomplete members at p = 1, never pooled with amend1's",
         f"% two floats ({_T('prefix 1')}the one-float version was 72 pt taller than the page), split before the controller slot "
         "as Table 5"]

    def _twhead(first):
        # scriptsize, as tab:checks: with the amendment 2 rows and the CT-anatomy block the footnotesize float is
        # taller than the page
        # [tp] (not [p]): each part is under a page, and a [p] float below \\floatpagefraction blocks the queue
        return (["\\begin{table*}[tp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize"]
                + (["\\caption{" + cap + "} % src: numbers of this caption: " + _T("twin caption src")
                    + "the episode design of outputs/v2/table4/episodes_long.csv; TOST margin: " + TOST_MARGIN_SRC,
                    "\\label{tab:pluggability-twin}"] if first else ["\\ContinuedFloat", "\\caption[]{Continued.}"])
                + ["\\centering", "\\setlength{\\tabcolsep}{2pt}",
                   "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l r r l r r r l r@{}}",
                   "\\toprule",
                   "Slot and method & Label & Cost & \\begin{tabular}[b]{@{}r@{}}Error\\\\(mm)\\end{tabular} & "
                   "\\begin{tabular}[b]{@{}l@{}}Difference (mm),\\\\95\\% $t$ CI, $k$/6\\end{tabular} & $t$ $p$ & "
                   "Holm $p$ & \\begin{tabular}[b]{@{}r@{}}Sign-\\\\flip $p$\\end{tabular} & "
                   "\\begin{tabular}[b]{@{}l@{}}TOST, 90\\%\\\\$t$ CI (mm)\\end{tabular} & Score \\\\ "
                   f"% src: 95% t CI, {_T('src a1 stats')}; k of 6 held-out patients; TOST post hoc",
                   "\\midrule"])
    parts = [body] if twin_split is None else [body[:twin_split], body[twin_split:]]
    for i, part in enumerate(parts):
        b_ = [x for x in part if x != ""]
        while b_ and b_[0].startswith("\\addlinespace"):
            b_ = b_[1:]
        L += _twhead(i == 0) + b_ + ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    for x in L:   # every row of the twin has the ten cells of its header
        if x.startswith(("\\quad", "Slot and method")):
            h_ = x.partition(" % src:")[0]
            h_ = h_.replace("\\\\(mm)", "").replace("\\\\flip", "").replace("\\\\$t$", "")
            if h_.count("&") != TW_NCOL - 1:
                raise SystemExit(f"pluggability_twin: {h_.count('&') + 1} cells in: {x[:140]}")
    # bibliography keys this table cites that sections/refs.tex does not define yet (the refs owner adds them)
    import re
    keys = sorted({k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", "\n".join(L)) for k in m.split(",")})
    new = _new_bib_keys(keys)
    L.insert(0, "%% NEW BIB KEYS: " + (" ".join(new) or "none"))
    log(f"plug rows with results: {n_plug_done} of {n_plug_rows}" + (" -> PLACEHOLDER table" if placeholder else "")
        + f"; CT-anatomy rows with results: {n_cta_done} of {n_cta}")
    log(f"cited keys not yet in refs.tex: {len(new)} of {len(keys)}")
    twin = write(out, "pluggability_twin.tex", "\n".join(L))
    return _plug_body(R, out, prov) + [twin]


# =================================================================================== Table 5 (tab:pluggability), body
# Table 5 of the article: row groups per slot (reference, family, secondary, privileged); columns method (L marks
# label access), cost, final error, difference [95% t CI] with k/6, t p, Holm p, exact sign-flip p and a Decision in
# the outcome words of the rules; each slot header gives the interface, the condition, the absolute error of the
# reference, the family with its size, and the post hoc MDD. Every cell is
# read from the record pluggability() builds while it writes the supplementary twin, so the two tables print the same
# numbers. plug_rows.json (Fig. 4) is written in the same run.
PLUG_REC = []
T5_CAP = {}   # the horizon and episode count Table 5's caption checks
FIG_OUT = os.path.join(PAPER, "figures") if PAPER else None


def _new_bib_keys(keys):
    """The cited keys that the manuscript's refs.tex does not define yet (the NEW BIB KEYS comment line). The public
    build has no manuscript source: it writes 'none', and that comment line is removed with the others."""
    if PI.skip_dev("the list of cited keys that the article's reference list does not define"):
        return []
    import re
    refs = open(REFS_TEX).read()
    return [k for k in keys if not re.search(r"\\bibitem(\[[^\]]*\])?\{" + re.escape(k) + r"\}", refs)]
DEC_LOWER, DEC_HIGHER, DEC_SEC, DEC_NC = "lower", "higher", "secondary", "not completed"
DEC_NULL = _T("dec null")            # amendment 1, otherwise
NOT_POWERED = _T("not powered")      # Table 5 prints it once, in its caption (the cells print DEC_NULL without it)
DEC_IMN_NULL = _T("dec imn null")    # amendment 2, IMN-T otherwise
if not (DEC_NULL.endswith(NOT_POWERED) and DEC_IMN_NULL.endswith(NOT_POWERED)):
    raise SystemExit("DEC_NULL and DEC_IMN_NULL must end in NOT_POWERED")
NOTE_A = "Unadjusted CI excludes 0; not declared a difference after Holm (post hoc note)."
# The condition of every row of a slot (amendment 1 and the setting strings of the twin), in the order frames, mask
# source, residual on or off, seeds.
SLOT_COND = {
    "texture": "frames from the plugged stage; mask from that stage's own matched U-Net; servo base, residual off; "
               "seeds 0 to 2",
    "mask": "SonoSPADE frames; mask from the plugged source; servo base fitted under that mask, residual off; seeds 0 "
            "to 2, and seed 0 for the SAM-type rows, paired with seed 0",
    # (the reward and renderer slots also run under CT masks)
    "ctrl": "SonoSPADE frames; CT mask$^{\\dagger}$; the plugged controller, "
            "residual or full twist as named; seeds 0 to 2, and seed 0 for CEM",
    "reward": "SonoSPADE frames; CT mask$^{\\dagger}$; the REINFORCE residual from the imitation start, trained with the "
              "plugged potential; seeds 0 to 2",
    "renderer": "frames from the plugged renderer, no texture stage; CT mask$^{\\dagger}$; the physics-trained policy, "
                "residual on; seeds 0 to 2, and seed 0 for SIMUS",
}
# (the KID that Table 5 printed beside the texture and renderer method names moved to the twin's Score column)
MDD_ALPHA, MDD_POWER = 0.05, 0.8
T5_SPLIT_BEFORE = "ctrl"   # the slot that opens the continuation float of Table 5
REF_NAME = {"texture": "SonoSPADE", "mask": "the stage-matched U-Net",
            "ctrl": "the REINFORCE residual from the imitation start", "reward": "the match score $S$, retrained",
            "renderer": "the physics renderer"}


def _vf(txt, v21=True):
    """A finished but not final v2.1 cell: boxed by \\verifyfill (no section sign is printed, PROV); archived v2
    cells print plain."""
    if not txt or not v21 or not PLUG_PROVISIONAL:
        return txt
    return f"\\verifyfill{{{txt}}}{PROV}"


def _mdd(sd, n, alpha=MDD_ALPHA, power=MDD_POWER):
    """Minimum detectable difference of a two-sided paired t-test (amend1 and
    scripts/plug/cohort_plug_eval.py _mdd): (t_{1-alpha/2, n-1} + t_{power, n-1}) sd / sqrt(n)."""
    from scipy.stats import t as T
    if n < 2 or sd is None or not np.isfinite(sd):
        return None
    return float((T.ppf(1 - alpha / 2, n - 1) + T.ppf(power, n - 1)) * sd / np.sqrt(n))


MDD_RULE = _T("mdd rule")


def _slot_mdd(SR):
    """Post hoc MDD at n = 6 for one slot. Primary: SD = the median over the contrasts of the slot with its
    reference row (Table 5 tests each row against the reference only) of the standard deviation of the paired
    per-patient differences. Rows: those of amendment 1 with mdd=True, each with all its design seeds; a contrast
    uses the seeds that both rows have. Rows of amendment 2 never enter. For the log and plug_rows.json only: the
    median over all method pairs (the cohort-ordering form of amendment 1) and the median over the Holm-tested
    family contrasts alone (sensitivity)."""
    P = sorted(TEST_PATIENTS)
    # rows of an amendment 2 group (A2_BLIND_KEYS) never enter, whatever their role (so the privileged secondary SAMUS
    # CT-box row stays out of the mask slot)
    meth = [r for r in SR["defs"] if r["mdd"] and r["role"] != "a2" and r.get("group") not in A2_BLIND_KEYS
            and SR["cells"].get(r["id"]) and len(SR["cells"][r["id"]]) == len(r["seeds"])]
    ref = next(r for r in SR["defs"] if r["role"] == "ref")

    def _sd(a, b):
        ca, cb = SR["cells"][a["id"]], SR["cells"][b["id"]]
        sd_ = sorted(set(ca) & set(cb))
        if not sd_:
            return None
        d = (np.array([[ca[s]["pos"][p] for p in P] for s in sd_]).mean(0)
             - np.array([[cb[s]["pos"][p] for p in P] for s in sd_]).mean(0))
        return (a["id"], b["id"], sd_, float(np.std(d, ddof=1)))

    def _summ(pairs, n_methods):
        pairs = [x for x in pairs if x is not None]
        if not pairs:
            return {"n_methods": n_methods, "n_pairs": 0, "sd_mm": None, "mdd_mm": None, "mdd_holm_mm": None,
                    "pairs": []}
        sd = float(np.median([x[3] for x in pairs]))
        return {"n_methods": n_methods, "n_pairs": len(pairs), "sd_mm": sd, "mdd_mm": _mdd(sd, len(P)),
                "mdd_holm_mm": _mdd(sd, len(P), alpha=MDD_ALPHA / SR["m"]), "alpha_holm": MDD_ALPHA / SR["m"],
                "pairs": [{"a": a, "b": b, "seeds": s_, "sd_mm": v} for a, b, s_, v in pairs]}

    has_ref = any(r["id"] == ref["id"] for r in meth)
    out = _summ([_sd(ref, r) for r in meth if r["id"] != ref["id"]] if has_ref else [], len(meth))
    out["rule"] = _T("mdd ref rule")
    out["all_pairs"] = _summ([_sd(a, b) for a, b in itertools.combinations(meth, 2)], len(meth))
    out["all_pairs"]["rule"] = _T("all pairs rule")
    fam = [r for r in SR["defs"] if r["role"] == "family" and SR["cells"].get(r["id"])
           and len(SR["cells"][r["id"]]) == len(r["seeds"]) and not (r["learned"] and len(SR["cells"][r["id"]]) < 3)]
    out["family_only"] = _summ([_sd(ref, r) for r in fam] if SR["cells"].get(ref["id"]) else [], len(fam))
    out["family_only"]["rule"] = "Holm-tested family contrasts only (sensitivity)"
    return out


def _decide(r):
    """-> (Decision cell, post hoc note or None), in the plans' outcome words only."""
    k, c = r["kind"], r["c"]
    if k == "ref_seed0" or r["role"] == "ref":
        return "", None
    if k in ("bn_train", "samus3") or r["role"] == "secondary" or r["demote"]:
        return DEC_SEC, None
    if c is None or r["short"]:
        return DEC_NC, None
    excl = c["ci"][0] > 0 or c["ci"][1] < 0
    if k == "imnt":
        if r["hold"]:
            return ("lowers" if c["mean"] < 0 else "raises"), None
        return DEC_IMN_NULL, (NOTE_A if excl else None)
    if r["hold"]:
        return (DEC_LOWER if c["mean"] < 0 else DEC_HIGHER), None
    return DEC_NULL, (NOTE_A if excl else None)


def _p3(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


# words of a row label that the tables do not print; the record (r["label"], plug_rows.json) keeps the label
LABEL_PRINT_DROP = (" (optional)", ", CPU only")
# printed names of two record labels (print only, so plug_rows.json keeps its labels: PLUG_ROWS_MD5)
LABEL_PRINT_RENAME = (("LOTUS, in-repository reimplementation", "LOTUS, authors' reimplementation"),
                      ("Paper score $S$ (authors')", "Match score $S$ (authors')"))


def _print_label(s):
    """A row label as Table 5 and its twin print it (LABEL_PRINT_DROP removed, LABEL_PRINT_RENAME applied)."""
    for x in LABEL_PRINT_DROP:
        s = s.replace(x, "")
    for a, b in LABEL_PRINT_RENAME:
        s = s.replace(a, b)
    return s


# Family names as Table 5 prints them: the slot family by its slot, the later families by plain names, which the key
# sentence of Appendix B maps to their codes. (header phrase, subheader)
T5_FAMILY = {"MASK-P": ("second mask source", "Second mask-source family"),
             "IMN-T": ("pretrained backbone", "Pretrained-backbone family"),
             "LPIPS-R": ("LPIPS reward", "LPIPS reward family")}
T5_SLOT_FAMILY = {"mask": "mask source", "reward": "reward"}
# the family codes that Table 5 may print (defined at first use in Appendix B); every other plan code stays out
T5_ALLOWED_CODES = ("MASK-P", "IMN-T", "LPIPS-R", "CTA-D", "CTA-E")


def _plain_label(s):
    """A LaTeX row label as plain text for plug_rows.json and the figure."""
    import re
    s = s.replace('\\"{o}', "\u00f6").replace("~", " ").replace("$^{\\dagger}$", "")
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _k3(c):
    """Seeds sharing the sign of the mean, k/n over the contrast's seeds (post hoc); None below two seeds."""
    if c is None or c.get("n_seed", 0) < 2:
        return None
    return f"{c['k_seed']}/{c['n_seed']}"


def _audits_done():
    """True when every identical-episode audit of the final queue has its output, passed, with nothing refused."""
    Q = _queue()
    aud = [n for n in Q if n.startswith("v21:checks:audit:")]
    for n in aud:
        p_, ok = Q[n]
        if not ok or glob.glob(p_[:-5] + ".failed*.json"):
            return False
        j_ = _plug_json(p_)
        if j_.get("passed") is not True or j_.get("refused"):
            return False
    return bool(aud)


def _t5_incomplete():
    """True while a Holm family member of Table 5 lacks a complete result (it then enters at p = 1)."""
    for SR in PLUG_REC:
        if SR["pending"] or any(not g["has"] for g in SR["groups"]):
            return True
        if any(r["kind"] in ("row", "imnt") and r["role"] in ("family", "a2") and (r["c"] is None or r["short"])
               for r in SR["rows"]):
            return True
    return bool(CTA_STATE.get("blinded")) or any(r["c"] is None for r in CTA_STATE.get("rows", []))


def _plug_body(R, out, prov):
    """Write the body Table 5 (tables/pluggability.tex, tab:pluggability) and figures/plug_rows.json.
    Columns: Error, Difference [95% t CI] with k/6, the seeds sharing
    the sign (k/3, post hoc), Holm p, the post hoc TOST at +/- 1 mm and the Decision. Cost, the t p, the
    sign-flip p and the scores print in the supplementary twin (tables/pluggability_twin.tex)."""
    cm, rg = R["table4"]["cell_means_mm"], R["table4"]["rungs_mm"]
    NC = 7
    mcN = f"\\multicolumn{{{NC}}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.985\\textwidth}}@{{}}}}"
    L, J, notes_used, split_at = [], {"slots": []}, [], None
    starts = {}   # where each slot block opens in L: the guide slots print as Table 5, the engine slots in their appendix table
    for SR in PLUG_REC:
        starts[SR["key"]] = len(L)
        if SR["key"] == T5_SPLIT_BEFORE:      # Table 5 is two floats, the second a continuation
            split_at = len(L)
        dd = SR["dd"]
        rows = SR["rows"]
        ref = next(r for r in rows if r["kind"] == "row" and r["role"] == "ref")
        md = _slot_mdd(SR)
        SR["mdd"] = md
        ref_err = (_vf(f"{ref['err']:.2f}", ref["v21"]) if ref["err"] is not None else "pending")
        if md["mdd_mm"] is None:
            mdd_txt = "pending"
        else:
            m1, m2 = _vf(f"{md['mdd_mm']:.{dd}f}"), _vf(f"{md['mdd_holm_mm']:.{dd}f}")   # reward: 3 decimals
            mdd_txt = f"{m1}~mm, {m2}~mm at $\\alpha = 0.05/{SR['m']}$"
        refname = REF_NAME[SR["key"]] + ("$^{\\mathrm{b}}$" if SR["key"] == "reward" else "")
        # "Family: m" for a slot with one family; the slot family and the later families by their Appendix B names
        # (T5_FAMILY), never pooled
        if not SR["holm2_m"]:
            fam_txt = f"Family: $m = {SR['m']}$"
        else:
            fam_txt = "Families: " + _and([f"{T5_SLOT_FAMILY[SR['key']]} ($m = {SR['m']}$)"] + [
                f"{T5_FAMILY[fn][0]} ($m = {m_}$)" for fn, m_ in SR["holm2_m"].items()]) + ", never pooled"

        def _mdd_src(x):
            return ("n/a" if x["sd_mm"] is None else
                    f"median SD {x['sd_mm']:.4f} mm over {x['n_pairs']} contrasts, MDD {x['mdd_mm']:.4f} mm "
                    f"({x['mdd_holm_mm']:.4f} mm at 0.05/{SR['m']})")
        # the condition and reference clauses of the slot header stay on
        # the slot record, so leaderboard_tab() prints the same string; the printed Table 5 is byte-identical
        SR["t5_head"] = f"Condition: {SLOT_COND[SR['key']]}. Reference: {refname}, {ref_err}~mm."
        SR["t5_notes"] = []
        L += ["\\addlinespace[4pt]" if L else "",
              # slot names only, no interface names
              f"{mcN}{{\\textbf{{{SR['title']}}}. {SR['t5_head']} {fam_txt}. Post hoc MDD: {mdd_txt}.}} "
              f"\\\\ % src: reference err {ref['src'].split(';')[1].strip() if ';' in ref['src'] else ref['src']}; "
              f"MDD computed here, {MDD_RULE}: {_mdd_src(md)} with the reference "
              f"({', '.join(pr['b'] for pr in md['pairs'])}); beside it, not printed: {_T('mdd pairs note')} "
              f"{_mdd_src(md['all_pairs'])}; Holm-tested family contrasts "
              f"only {_mdd_src(md['family_only'])}; {_T('mdd sizes note')}"]

        def line(r, indent="\\quad "):
            dec, note = _decide(r)
            if note and note not in notes_used:
                notes_used.append(note)
            lab = _print_label(r["label"]) + ("$^{\\mathrm{L}}$" if r["laccess"] else "") + (
                "$^{\\dagger}$" if r["priv"] and "$^{\\dagger}$" not in r["label"] else "") + (
                "$^{\\ddagger}$" if r["short"] else "")
            err = "" if r["err"] is None else _vf(f"{r['err']:.2f}", r["v21"])
            c = r["c"]
            if c is not None:
                diff = _vf(f"{sg(c['mean'], dd)} {ci_txt(c['ci'], dd)}, {ks(c)}")
                k3 = _vf(_k3(c)) if _k3(c) else "n/a"
                tost = _vf("yes" if c["tost_in"] else "no")
            else:
                diff = k3 = tost = ""
            hp = "" if r["holm"] is None else _vf(_p3(r["holm"]))
            # the null wording ("...; not powered for equivalence") prints as one caption sentence, not as a superscript,
            # so each Decision cell keeps to one line (the full words are
            # in plug_rows.json and App B)
            mk_ = ["a"] if note else []
            dec_t = dec.replace(NOT_POWERED, "") + (f"$^{{\\mathrm{{{','.join(mk_)}}}}}$" if mk_ else "")
            ph = ("" if c is None else
                  f"; TOST +/- {TOST_MARGIN_MM:g} mm (post hoc, unadjusted): 90% t CI [{c['tost_ci90'][0]:+.3f}, "
                  f"{c['tost_ci90'][1]:+.3f}], TOST p {c['tost_p']:.4g} -> {'yes' if c['tost_in'] else 'no'}; seeds "
                  f"sharing the sign (post hoc) {_k3(c) or 'n/a'}, per-seed differences "
                  + ", ".join(f"s{sd} {v:+.3f}" for sd, v in zip(c["seeds"], c["per_seed"])))
            # the printed cells stay on the row record, so leaderboard_tab() prints Table 5's own strings and computes
            # nothing again
            r["t5"] = {"lab": lab, "err": err, "diff": diff, "k3": k3, "holm": hp, "tost": tost, "dec": dec_t,
                       "indent": indent, "pos": len(L)}
            return (f"{indent}{lab} & {err} & {diff} & {k3} & {hp} & {tost} & {dec_t} \\\\ % src: {r['src']}{ph}"
                    + (f"; decision {dec!r}{_T('dec words src')}" if dec else ""))

        # each group carries the role word the leaderboard prints (LB_ROLE);
        # the printed Table 5 is unchanged
        groups = [("Reference", [r for r in rows if r["role"] == "ref"], "ref"),
                  (f"Family ($m = {SR['m']}$)",
                   [r for r in rows if r["kind"] == "row" and r["role"] == "family" and r["err"] is not None], "family")]
        for fn in dict.fromkeys(r["fam"] for r in rows if r["kind"] == "row" and r["role"] == "a2"):
            groups.append((f"{T5_FAMILY[fn][1]} ($m = {SR['holm2_m'][fn]}$)",
                           [r for r in rows if r["kind"] == "row" and r["role"] == "a2" and r["fam"] == fn
                            and r["err"] is not None], "a2"))
        groups.append((f"{T5_FAMILY['IMN-T'][1]}, minus the same network from scratch "
                       f"($m = {AMEND2_FAMILY_M['IMN-T']}$)",
                       [r for r in rows if r["kind"] == "imnt" and r["c"] is not None], "imnt"))
        groups.append(("Secondary", [r for r in rows if r["role"] == "secondary" or r["kind"] in ("bn_train", "samus3")
                                     if not r["priv"] and r["err"] is not None and r["kind"] != "ref_seed0"],
                       "secondary"))
        groups.append(("Privileged, secondary", [r for r in rows if r["priv"] and r["err"] is not None
                                                  and r["role"] != "ref"], "priv"))
        for head, rs, grp_role in groups:
            if not rs:
                continue
            L.append(f"\\multicolumn{{{NC}}}{{@{{}}l}}{{\\quad\\textit{{{head}}}}} \\\\"
                     + (f" % src: family sizes, {_T('src a1 stats')} and {_T('src a2 fam')}" if "$m" in head else ""))
            for r in rs:
                L.append(line(r, "\\quad\\quad " if r["kind"] == "ref_seed0" else "\\quad "))
                r.setdefault("t5_group", []).append(grp_role)
        # pending: rows not yet run, rows whose error is pending, and the blinded or pending amendment 2 groups
        pend = [r["label"] + (f"~\\cite{{{','.join(r['cites'])}}}" if r["cites"] else "")
                + ("$^{\\dagger}$" if r["priv"] else "") for r in SR["pending"]]
        pend += [r["label"] for r in rows if r["kind"] == "row" and r["err"] is None]
        grp = [f"{g['fams_plain']}: {g['what']}, " + (A2_BLIND_TXT if g["blinded"] else "pending")
               for g in SR["groups"] if not g["has"]]
        if pend or grp:
            txt = "; ".join(pend) + ("." if pend else "") + (" " if pend and grp else "") + (
                ". ".join(x[0].upper() + x[1:] for x in grp) + "." if grp else "")
            L.append(f"{mcN}{{\\quad\\textit{{Pending.}} {txt}}} \\\\ % src: pending rows of this slot (files in the "
                     "twin, tables/pluggability_twin.tex)")
        if SR["not_run"]:   # Table 2 holds the one "not evaluated" list; here a comment line
            L.append(f"% Not run{_T('not run note')}: {SR['not_run']} % src: {_T('src a1 notrun')}")
        if SR["key"] == "reward" and ref["err"] is not None:
            arch = cm["ctmask/sonospade/final"]
            drift = _vf(f"{abs(ref['err'] - arch):.2f}")
            # the Table 6 retrain-drift row is not printed; its comparison with the stage spread
            # under CT masks moves here, computed as that row computed it
            ct_f = [cm[f"ctmask/{s_}/final"] for s_ in STAGES]
            spread = max(ct_f) - min(ct_f)
            # the note stays on the slot record for leaderboard_tab()
            SR["t5_notes"].append(f"\\quad (b) The reference is the match score $S$, retrained in the same way as the "
                                  f"plugged rewards. It ends {drift}~mm {'above' if ref['err'] > arch else 'below'} its "
                                  f"{arch:.2f}~mm in Table~\\ref{{tab:ladder}}, "
                                  f"{'more' if abs(ref['err'] - arch) > spread else 'less'} than the {spread:.2f}~mm "
                                  "spread across texture stages under CT masks.")
            L.append(f"{mcN}{{{SR['t5_notes'][-1]}}} \\\\ % src: {ref['src'].split(';')[1].strip()} "
                     f"(retrained, mean {ref['err']:.4f}) minus RESULTS.json table4.cell_means_mm[ctmask/sonospade/final] "
                     f"({arch:.4f}); {_T('prefix 2')}spread: max minus min of table4.cell_means_mm[ctmask/<stage>/final] "
                     f"({spread:.4f}), the Table 6 retrain-drift row's comparison{_T('drift tag')}")
        js = {"key": SR["key"], "title": SR["title"], "m": SR["m"], "audit": SR["audit"],
              "reference": {"id": ref["id"], "label": _plain_label(ref["label"]), "err_mm": ref["err"],
                            "provisional": ref["v21"]},
              "mdd": md, "rows": []}
        for r in rows:
            dec, note = _decide(r)
            c = r["c"]
            js["rows"].append({"id": r["id"], "kind": r["kind"], "label": _plain_label(r["label"]), "role": r["role"],
                               "family": r["fam"], "privileged": r["priv"], "label_access": r["laccess"],
                               "err_mm": r["err"], "provisional": r["v21"], "seeds": r.get("seeds"),
                               "diff_mm": None if c is None else c["mean"], "ci95_t": None if c is None else c["ci"],
                               "k": None if c is None else c["sign"], "p_t": None if c is None else c["p_t"],
                               "p_signflip": None if c is None else c["p_sf"], "p_holm": r["holm"], "decision": dec,
                               "note": note,
                               # post hoc additions
                               "sd_mm": None if c is None else c["sd"],
                               "ci90_t": None if c is None else c["tost_ci90"],
                               "p_tost_1mm": None if c is None else c["tost_p"],
                               "tost_within_1mm": None if c is None else c["tost_in"],
                               "contrast_seeds": None if c is None else c["seeds"],
                               "per_seed_diff_mm": None if c is None else c["per_seed"],
                               "k_seeds": None if c is None or c["n_seed"] < 2 else c["k_seed"],
                               "n_seeds": None if c is None else c["n_seed"],
                               "ps_boot_ci95": None if c is None else c["ps_nested"],
                               "crossed_boot_ci95": None if c is None else c["ps_crossed"]})
        js["pending"] = [_plain_label(r["label"]) for r in SR["pending"]]
        js["groups_pending"] = [{"group": g["group"], "families": g["fams"], "blinded": g["blinded"]}
                                for g in SR["groups"] if not g["has"]]
        J["slots"].append(js)
        log(f"Table 5 {SR['key']:8s} reference {ref['err']:.3f} mm | MDD "
            + ("pending" if md["mdd_mm"] is None else
               f"{md['mdd_mm']:.3f} mm ({md['mdd_holm_mm']:.3f} at 0.05/{SR['m']}), median SD {md['sd_mm']:.3f} over "
               f"{md['n_pairs']} reference contrasts") + " [all pairs "
            + ("-" if md["all_pairs"]["mdd_mm"] is None else f"{md['all_pairs']['mdd_mm']:.3f}") + ", family only "
            + ("-" if md["family_only"]["mdd_mm"] is None else f"{md['family_only']['mdd_mm']:.3f}") + "] | " + "; ".join(
                f"{r['id']} {_decide(r)[0] or 'ref'}" + ("" if r["c"] is None else
                                                         f" TOST {'yes' if r['c']['tost_in'] else 'no'}")
                for r in rows if r["kind"] in ("row", "imnt")))
    # CT anatomy (amendment 2): five rungs, never pooled; blinded in this build
    cs = CTA_STATE
    # the condition, reference and caveat clauses of the CT-anatomy
    # header are built once and stay on CTA_STATE, so leaderboard_tab() prints the same strings; Table 5 is unchanged
    cs["t5_head"] = (
        "Condition: five reference levels of the oracle ladder, each in its "
        "Table~\\ref{tab:ladder} condition with seeds 0 to 2. TotalSegmentator~v2 labels~\\cite{totalseg} (fast "
        "mode, 3.0~mm) replace the reference labels on the same episodes and classes. "
        "Reference: each level rerun on the reference labels" + (": " + "; ".join(
            f"{CTA_PLAIN[rid]}, {_vf(f'{e:.2f}')}" for rid, _, _, e, _ in cs.get("refs", []) if e is not None)
            + "~mm" if cs.get("refs") else "") + ".")
    cs["t5_caveat"] = ("The trained artifacts never saw "
                       "the new labels, so a change mixes label noise with the shift between training and evaluation "
                       "labels.")
    starts["cta"] = len(L)
    L += ["\\addlinespace[4pt]",
          # the CT-anatomy slot header: families by plain names (Appendix B gives their codes)
          f"{mcN}{{\\textbf{{CT anatomy}} (label volume). {cs['t5_head']} Families: CT-anatomy difference, paired $t$, "
          f"and CT-anatomy equivalence, TOST at $\\pm 1$~mm, $m = {AMEND2_FAMILY_M['CTA-D']}$ "
          # no post hoc MDD here: MDD_RULE takes the rows carried to the cohort analysis, never an amendment 2 row
          "each. "
          f"{cs['t5_caveat']} No row "
          "runs on the replication cohort, so the slot has no post hoc MDD.} \\\\ % src: "
          + _T('src a2 cta parts') + "; reference rungs outputs/v2_1/s21_rerun/{ladder,contact}/<row "
          "file> (v2.1 rerun, seeds 0 to 2, contact on): " + "; ".join(f"{rid} {('missing' if e is None else f'{e:.4f}')}"
                                                                     for rid, _, _, e, _ in cs.get("refs", []))]
    if cs.get("blinded"):
        L.append(f"{mcN}{{\\quad\\textit{{Pending.}} The five rows, {A2_BLIND_TXT}.}} \\\\ % src: BLIND_A2 (rows R1 "
                 f"to R5 of {_T('src a2 cta')})")
    else:
        for r in cs.get("rows", []):
            c = r["c"]
            if c is None:
                continue
            dD = r["hd"] < AMEND2_ALPHA and c["sign"] >= AMEND2_DECISION["min_sign_agree"]
            dE = r["he"] < AMEND2_ALPHA
            verb = "falls" if c["mean"] < 0 else "rises"
            # the clause "equivalence within +/- 1 mm not shown" prints as the superscript e, which the caption spells
            # out (the full words: App C)
            dec = (f"{verb}; within $\\pm 1$~mm" if dD and dE else verb if dD else "equivalent within $\\pm 1$~mm"
                   if dE else "no detectable change$^{\\mathrm{e}}$")
            e_txt = _vf(f"{r['err']:.2f}")
            d_txt = _vf(f"{sg(c['mean'])} {ci_txt(c['ci'])}, {ks(c)}")
            k3 = _vf(_k3(c)) if _k3(c) else "n/a"
            # here the TOST column holds CTA-E, the planned TOST at +/- 1 mm with Holm on the TOST p (amendment 2)
            # the row by its plain rung name (CTA_PLAIN); its amendment 2 key stays in the src and in Appendix C
            # the printed cells are built once and stay on the row record, so
            # leaderboard_tab() prints these strings; was the same expressions inline in the L.append below
            t5 = {"lab": _cap1(CTA_PLAIN[r["rid"]]) + ("$^{\\dagger}$" if r["priv"] else ""), "err": e_txt,
                  "diff": d_txt, "k3": k3, "holm": _vf(_p3(r["hd"])), "tost": _vf("yes" if dE else "no"), "dec": dec,
                  "indent": "\\quad ", "pos": len(L)}
            r["t5"], r["t5_group"] = t5, ["cta"]
            L.append(f"\\quad {t5['lab']}"
                     + f" & {t5['err']} & {t5['diff']} & {t5['k3']} & {t5['holm']} & {t5['tost']} & {t5['dec']} "
                       f"\\\\ % src: row {r['rid']}; {r['src']}; TOST column: CTA-E Holm p {r['he']:.4g} (planned, "
                       "amendment 2)")
    L.append(f"{mcN}{{\\quad\\textit{{{CTA_IN_SAMPLE}}}}} \\\\ % src: " + _T("cta in-sample src"))
    J["cta"] = {"blinded": bool(cs.get("blinded")),
                "reference_rungs": [{"row": rid, "label": lab, "privileged": pv, "err_mm": e}
                                    for rid, lab, pv, e, _ in cs.get("refs", [])]}
    # the note a stands in the caption, beside the superscript e (part 2 of the float is a page tall)
    cap = ("Published methods in the three slots of the guide: mask source, controller and reward (exploratory, "
           "pre-specified rule). "
           "Every row: 12 steps, the 60 held-out episodes of Table~\\ref{tab:ladder}, contact on. "
           "Error: final translation error, mean over seeds of patient means. "
           "Difference: from the slot's reference component, paired by seed, with 95\\% paired $t$ CI and patients of 6 "
           "sharing its sign. "
           "Seeds: the seeds, of 3, whose own difference shares that sign (post hoc; n/a for seed-0 rows). "
           # the clause on incomplete members prints only while a member is incomplete
           + ("Holm $p$: within the slot's family, incomplete members at $p = 1$, so an upper bound. " if _t5_incomplete()
              else "Holm $p$: within the slot's family. ")
           + "TOST: post hoc and unadjusted, yes when the 90\\% paired $t$ CI lies inside $\\pm 1$~mm, the "
           "pre-specified equivalence margin. "
           # one caption sentence on the null decisions (plug_rows.json and the twin keep the full decision)
           "Decision: the pre-specified outcome words; $^{\\mathrm{e}}$equivalence within $\\pm 1$~mm not shown. "
           "A decision of no detectable difference (in the pretrained-backbone family, does not detectably change) is not powered for "
           "equivalence; the TOST column shows where equivalence within $\\pm 1$~mm holds"
           + (f". $^{{\\mathrm{{a}}}}${NOTE_A} " if notes_used else ". ")
           + "Slot headers give the condition, the reference error and the post hoc minimum detectable difference "
           "(MDD) from the median SD of the slot's contrasts with its reference, over the rows carried to the cohort "
           "analysis, so not SAM, MedSAM, CEM or the CT label, also at the first Holm level. "
           # which test the MDD describes
           "The MDD is that of the paired $t$-test alone; the 5 of 6 sign condition makes the decision rule slightly "
           "less sensitive. "
           # the estimand sentence, which every caption carries
           "Every contrast is conditional on the trained artifacts (texture networks, segmenters, bases and policies). "
           "Seeds index artifacts; the patient is the sampling unit. "
           "Costs, $t$ and sign-flip $p$, the 90\\% CIs and scores are in the supplementary twin of this table. "
           "$^{\\mathrm{L}}$Reads the CT label slice. $^{\\dagger}$Privileged: simulator state that no rig provides. "
           "$^{\\ddagger}$Fewer seeds than designed.")
    # no provisional sentence (PLUG_PROVISIONAL is False)
    # the horizon of every row read: the caption's "12 steps" is checked, not typed (files without the field are the
    # archived v2 trainer outputs, whose 12-step design RESULTS.json certifies)
    hz = collections.Counter()
    for SR in PLUG_REC:
        for cell in SR["cells"].values():
            for c_ in (cell or {}).values():
                hz[c_["d"].get("n_steps") if isinstance(c_.get("d"), dict) else None] += 1
    if any(k not in (12, None) for k in hz):
        raise SystemExit(f"Table 5: a row was not scored at 12 steps: {dict(hz)}")
    n_eps = len(_static_fp())
    if n_eps != 60:
        raise SystemExit(f"Table 5 caption: the archived static rung has {n_eps} episodes, the caption says 60")
    # the checked horizon and episode count, for the leaderboard caption
    T5_CAP.update(steps=sorted(k for k in hz if k is not None), episodes=n_eps)
    cap_src = (f"% src: 12 steps: n_steps of every plug output read ({hz.get(12, 0)} files with 12, {hz.get(None, 0)} "
               f"archived v2 files without the field){_T('t5 src a')} ({n_eps}){_T('t5 src b')}{TOST_MARGIN_SRC}; MDD: "
               + MDD_RULE
               + "; Supplementary Table S3.1 = tables/pluggability_twin.tex (tab:pluggability-twin)")
    T = [GEN.format(src="the record of pluggability() (the same cells as tables/pluggability_twin.tex): "
                        "outputs/v2_1/plug/, outputs/v2_1/checks/, outputs/v2_1/s21_bneval/, "
                        "outputs/v2_1/label_controls/ and the archived v2 files behind outputs/v2/RESULTS.json"),
         "% plug result files by git commit (provenance.git.hash, .dirty): "
         + "; ".join(f"{h} dirty={d}: {n} files" for (h, d), n in sorted(prov.items())),
         f"% amendment 2: {AMEND2_TXT} sha256 {AMEND2_SHA256}; Appendix constants asserted; "
         + (f"its groups {A2_BLIND_TXT}" if BLIND_A2 else "its groups read"),
         f"% oracle {rg['pose_oracle']:.4f} mm and static {rg['static']:.4f} mm: RESULTS.json table4.rungs_mm",
         ]

    # the renderer and texture stage (with the CT anatomy) form the standard simulation engine,
    # still pluggable; Table 5 prints the guide's slots and pluggability_engine.tex (tab:pluggability-engine) the engine's
    if list(starts) != ["texture", "mask", "ctrl", "reward", "renderer", "cta"]:
        raise SystemExit(f"Table 5: slot order {list(starts)}, the guide/engine split expects texture, mask, ctrl, "
                         "reward, renderer, CT anatomy")
    G = L[starts["mask"]:starts["renderer"]]
    E = L[starts["texture"]:starts["mask"]] + L[starts["renderer"]:]
    split_g = starts[T5_SPLIT_BEFORE] - starts["mask"]
    cap_e = ("Published methods in the three slots of the standard engine: texture stage, renderer and CT anatomy "
             "(exploratory, pre-specified rule). The engine is the physics renderer with the SonoSPADE texture stage "
             "and the reference CT labels; every slot stays pluggable. Columns, slot headers, marks and notes as in "
             "Table~\\ref{tab:pluggability}; the MDD leaves out SIMUS and the painted stage.")

    def head(first, cap_, label):
        # [tp]: with [p] only, part 1 (about 0.6 page) sat below \\floatpagefraction and blocked the float queue
        return (["\\begin{table*}[tp]",
                 "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}"]
                + (["\\caption{" + cap_ + "} " + cap_src, "\\label{" + label + "}"] if first else
                   ["\\ContinuedFloat", "\\caption[]{Continued.}"])
                + ["\\centering", "\\setlength{\\tabcolsep}{2.4pt}",
                   "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X r l c r c "
                   ">{\\raggedright\\arraybackslash}p{0.2\\textwidth}@{}}",
                   "\\toprule",
                   "Slot and method & \\begin{tabular}[b]{@{}r@{}}Error\\\\(mm)\\end{tabular} & "
                   "Difference (mm), 95\\% $t$ CI, $k$/6 & Seeds & Holm $p$ & "
                   f"TOST & Decision \\\\ % src: 95% t CI, {_T('src a1 stats')}; k of 6 held-out patients; Seeds = k of 3 "
                   "seeds and TOST at +/- 1 mm, post hoc",
                   "\\midrule"])
    foot = ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    TE = list(T)
    for T_, rows_, cap_, label, cut in ((T, G, cap, "tab:pluggability", split_g),
                                        (TE, E, cap_e, "tab:pluggability-engine", None)):
        parts = [rows_] if cut is None else [rows_[:cut], rows_[cut:]]
        for i, part in enumerate(parts):
            body_ = [x for x in part if x != ""]
            while body_ and body_[0].startswith("\\addlinespace"):
                body_ = body_[1:]
            T_ += head(i == 0, cap_, label) + body_ + foot
    import re
    for name, T_ in (("pluggability.tex", T), ("pluggability_engine.tex", TE)):
        for x in T_:   # every row of Table 5 and the engine table has the cells of its header
            if x.startswith(("\\quad", "Slot and method")) and x.partition(" % src:")[0].count("&") != NC - 1:
                raise SystemExit(f"{name}: {x.partition(' % src:')[0].count('&') + 1} cells in: {x[:140]}")
        keys = sorted({k.strip() for m_ in re.findall(r"\\cite\{([^}]*)\}", "\n".join(T_)) for k in m_.split(",")})
        new = _new_bib_keys(keys)
        T_.insert(0, "%% NEW BIB KEYS: " + (" ".join(new) or "none"))
        # Table 5 and the engine table print no code other than the family codes of Appendix B (T5_ALLOWED_CODES)
        _no_plan_codes(name, T_, allow=T5_ALLOWED_CODES)
    J.update({"generated_by": _T("plug_rows generated_by"), "status": PLUG_STATUS,
              "provisional": PLUG_ROWS_PROVISIONAL, "blinded_amendment2": BLIND_A2,
              "oracle_mm": rg["pose_oracle"], "static_mm": rg["static"],
              "oracle_src": "outputs/v2/RESULTS.json table4.rungs_mm.pose_oracle",
              "static_src": "outputs/v2/RESULTS.json table4.rungs_mm.static",
              "mdd_rule": MDD_RULE, "tost_margin_mm": TOST_MARGIN_MM, "tost_margin_src": TOST_MARGIN_SRC,
              PI.key("posthoc_rule"): _T("posthoc rule text")})
    os.makedirs(FIG_OUT, exist_ok=True)
    with open(_out_ok(os.path.join(FIG_OUT, "plug_rows.json")), "w", encoding="utf-8", newline="\n") as fh:
        # public mode: keys in sorted order, as every figure data file of the public build
        # (script_dirs.public_json_view)
        json.dump(J, fh, indent=1, default=float, sort_keys=SD.public())
    log(f"wrote {rel(os.path.join(FIG_OUT, 'plug_rows.json'))}")
    return [write(out, "pluggability.tex", "\n".join(T)), write(out, "pluggability_engine.tex", "\n".join(TE))]


# =================================================================================== tab:patient-seed (App C.8)
def patient_seed_tab(out):
    """tab:patient-seed (tables/patient_seed.tex, Appendix C.8): for every Table 5 row whose contrast has a seed factor
    (two or more seeds), the per-seed differences, the seeds sharing the sign (post hoc), the patient-by-seed
    bootstrap 95% CI of the v2.1 plan (plan_stats.patient_seed_bootstrap_ci) and a crossed patient x seed bootstrap
    95% CI beside it (post hoc, posthoc_v2.crossed_boot). Read from the Table 5 record (PLUG_REC)."""
    if not PLUG_REC:
        raise SystemExit("patient_seed_tab() reads the Table 5 record: run pluggability() first")
    log("\n== tab:patient-seed (patient_seed.tex), App C.8: patient-by-seed intervals of the three-seed Table 5 rows ==")
    NC = 6
    L, n_rows, n_incl0, split_at = [], 0, [], None
    for SR in PLUG_REC:
        dd = SR["dd"]
        rs = [r for r in SR["rows"] if r["c"] is not None and r["c"]["n_seed"] >= 2
              and r["kind"] in ("row", "bn_train", "imnt", "samus3")]
        if not rs:
            continue
        if SR["key"] == T5_SPLIT_BEFORE:   # two floats (one is taller than the page)
            split_at = len(L)
        L += (["\\addlinespace"] if L and split_at != len(L) else []) + [f"\\multicolumn{{{NC}}}{{@{{}}l}}{{\\textit{{{SR['title']}}}, minus "
                                                  f"{REF_NAME[SR['key']]}}} \\\\"]
        for r in rs:
            c = r["c"]
            lab = _print_label(r["label"]) + ("$^{\\dagger}$" if r["priv"] and "$^{\\dagger}$" not in r["label"] else "")
            seeds_txt = ", ".join(sg(v, dd).strip("$") for v in c["per_seed"])
            dtxt = _vf(f"{sg(c['mean'], dd)} {ci_txt(c['ci'], dd)}, {ks(c)}")
            row = (f"\\quad {lab} & {dtxt} & "
                   f"{_vf(f'${seeds_txt}$')} & {_vf(_k3(c))} & {_vf(ci_txt(c['ps_nested'], dd))} & "
                   f"{_vf(ci_txt(c['ps_crossed'], dd))} \\\\")
            inc = c["ps_nested"][0] <= 0 <= c["ps_nested"][1]
            if inc and (c["ci"][0] > 0 or c["ci"][1] < 0):
                n_incl0.append(f"{SR['key']}/{r['id']}")
            L.append(row + f" % src: {SR['key']}/{r['id']} vs the {SR['ref_id']} row on seeds {c['seeds']}, computed "
                           f"here from the Table 5 record (POST HOC except the patient-by-seed bootstrap, whose form is "
                           f"planned): mean {c['mean']:+.4f} t CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] {ks(c)}; per-seed "
                           + ", ".join(f"s{sd} {v:+.4f}" for sd, v in zip(c["seeds"], c["per_seed"]))
                           + f"; k/{c['n_seed']} {c['k_seed']}; patient-by-seed [{c['ps_nested'][0]:+.4f}, "
                           f"{c['ps_nested'][1]:+.4f}] " + "(plan_stats.patient_seed_bootstrap_ci, " + _T('ps boot consts')
                           + f"bootstrap 10000, seed 0); crossed [{c['ps_crossed'][0]:+.4f}, {c['ps_crossed'][1]:+.4f}] "
                           f"(posthoc_v2.crossed_boot, 10000, seed 0)")
            log(f"{SR['key']:8s} {r['id']:22s} {c['mean']:+.3f} t[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] per-seed "
                + ", ".join(f"{v:+.3f}" for v in c["per_seed"]) + f" k {c['k_seed']}/{c['n_seed']} ps "
                f"[{c['ps_nested'][0]:+.3f}, {c['ps_nested'][1]:+.3f}] crossed [{c['ps_crossed'][0]:+.3f}, "
                f"{c['ps_crossed'][1]:+.3f}]")
            n_rows += 1
    cap = ("Patient-by-seed intervals of the Table~\\ref{tab:pluggability} rows with three seeds (exploratory). "
           "Difference and $k$/6 as in Table~\\ref{tab:pluggability}. "
           "Per seed: the difference of patient means within one seed, seed $s$ of the row minus seed $s$ of the "
           "reference, for seeds 0, 1 and 2. "
           "Seeds: those whose difference shares the sign of the mean (post hoc). "
           "Patient-by-seed: the pre-specified bootstrap, which resamples patients, then seeds within each sampled patient, "
           "with the seed index paired across arms; 10,000 resamples, seed 0. "
           "Crossed: patients and seeds resampled independently, a post hoc sensitivity check with the same resamples "
           "and seed. "
           "Rows run at seed 0 only have no seed factor and are not listed. "
           # the estimand sentence, which every caption carries
           "Every contrast is conditional on the trained artifacts (texture networks, segmenters, bases and policies). "
           "$^{\\dagger}$Privileged.")
    T = [GEN.format(src="the Table 5 record of pluggability() (the same contrasts as tables/pluggability.tex)"),
         "% " + _T("ps planned form"),
         "\\begin{table*}[tp]",
         "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}",
         "\\caption{" + cap + "} % src: " + _T("ps caption src"),
         "\\label{tab:patient-seed}"]

    def _ps_head():
        return ["\\centering",
                "\\setlength{\\tabcolsep}{3pt}",
                "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l l c l l@{}}",
                "\\toprule",
                "Slot and row & Difference (mm), 95\\% $t$ CI, $k$/6 & Per seed (mm) & Seeds & Patient-by-seed 95\\% CI & "
                "Crossed 95\\% CI \\\\",
                "\\midrule"]
    foot = ["\\bottomrule", "\\end{tabularx}", "\\end{table*}"]
    if split_at is None:
        T += _ps_head() + L + foot + [""]
    else:   # the second float continues the first (\ContinuedFloat, as Table 5), split before the controller slot
        T += (_ps_head() + L[:split_at] + foot
              + ["\\begin{table*}[tp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}",
                 "\\ContinuedFloat", "\\caption[]{Continued.}"] + _ps_head() + L[split_at:] + foot + [""])
    for x in T:
        if x.startswith(("\\quad", "Slot and row")) and x.partition(" % src:")[0].count("&") != NC - 1:
            raise SystemExit(f"tab:patient-seed: wrong cell count in {x[:120]}")
    log(f"tab:patient-seed: {n_rows} rows; patient-by-seed CI includes 0 while the t CI excludes it: "
        + (", ".join(n_incl0) or "none"))
    return write(out, "patient_seed.tex", "\n".join(T))


# =================================================================================== tab:leaderboard (Appendix C)
# Appendix C, every method plugged under the protocol
# of Section 4.2 (sec:plug-protocol), slot by slot, from the runs of Table 5, ordered by mean error. The order is not a
# ranking (Section 9.2): each row was tested against its slot's reference only. Every cell is the string Table 5
# printed for the same row (r["t5"], kept by _plug_body() as it writes tables/pluggability.tex), and every slot header
# is Table 5's own condition and reference clauses (SR["t5_head"], CTA_STATE["t5_head"]), with Table 5's slot notes
# (SR["t5_notes"]) and the CT-anatomy sentence CTA_IN_SAMPLE, so nothing is computed again and no number is typed.
# Order: the slots in Table 5's order; in each, the
# reference, then the rows whose difference is taken from it by unrounded mean error, lowest first (Table 5's position
# breaks a tie), then Table 5's privileged rows in the same way; then the reference's seed-0 line, where Table 5 has
# one, and the rows run at seed 0 only (their differences are taken from that line), indented under it, again with the
# privileged rows last. Within each block every row has the same reference, so the error order is the order of the
# printed differences. The CT-anatomy rows keep Table 5's order: each is scored against its own reference level, so
# they are one method at five levels, not methods to order. Rows without an error of their own (the pretrained-backbone
# differences) stay out, and the caption says so. Role: the Table 5 group a row printed under.
LB_ROLE = {"ref": "reference", "family": "family", "a2": "second family", "secondary": "secondary",
           "priv": "privileged", "cta": "family"}
LB_MARKS = (("$^{\\mathrm{L}}$", "L"), ("$^{\\dagger}$", "$\\dagger$"), ("$^{\\ddagger}$", "$\\ddagger$"),
            ("$^{\\mathrm{a}}$", "a"), ("$^{\\mathrm{e}}$", "e"))
LB_IND = ("\\quad ", "\\quad\\quad ")   # Table 5's indents of a row and of the reference's seed-0 line


def _lb_seeds(r):
    """The seeds behind a Table 5 row's error: the row's own, or, for a row that Table 5 prints beside its base (the
    BatchNorm-train and three-seed SAMUS rows), those of its contrast."""
    if r.get("seeds"):
        return list(r["seeds"])
    return list(r["c"]["seeds"]) if r.get("c") else None


def leaderboard_tab(out):
    """tab:leaderboard (tables/leaderboard.tex, Appendix C): the Table 5 rows of each slot under Table 5's condition and
    reference clauses, in blocks that share one reference (the reference, then the seed-0 line), each block by mean
    error with the privileged rows last, with Table 5's method label and marks, error, difference [95% t CI] with k/6
    and decision, and the row's Table 5 group as its role. Read from the Table 5 record (PLUG_REC, CTA_STATE) after
    _plug_body()."""
    if not PLUG_REC or not T5_CAP:
        raise SystemExit("leaderboard_tab() reads the Table 5 record: run pluggability() first")
    log("\n== tab:leaderboard (leaderboard.tex), Appendix C: the Table 5 rows of each slot, in blocks that share a "
        "reference, each by mean error ==")
    steps = T5_CAP["steps"]
    if len(steps) != 1:
        raise SystemExit(f"leaderboard caption: the Table 5 rows were scored at horizons {steps}, one is expected")
    NC = 5
    mcN = f"\\multicolumn{{{NC}}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.985\\textwidth}}@{{}}}}"
    L, split_at, left_out, a2_names, seed0, n_rows, n_priv = [], None, [], [], [], 0, 0

    def row(key, r, role, indent):
        t = r["t5"]
        if "&" in t["lab"] + t["err"] + t["diff"] + t["dec"]:
            raise SystemExit(f"leaderboard {key}/{r.get('id', r.get('rid'))}: a cell holds '&'")
        err_src = "" if r["err"] is None else f"; err {r['err']:.4f} mm (the order key)"
        return (f"{indent}{t['lab']} & {LB_ROLE[role]} & {t['err']} & {t['diff']} & {t['dec']} \\\\ % src: Table 5 "
                f"row {key}/{r.get('id', r.get('rid'))}, its printed cells (the record of pluggability()){err_src}")

    def by_err(rs):
        return sorted(rs, key=lambda r: (r["err"], r["t5"]["pos"]))

    for SR in PLUG_REC:
        printed = sorted((r for r in SR["rows"] if "t5" in r), key=lambda r: r["t5"]["pos"])
        for r in printed:
            if len(r.get("t5_group", ())) != 1:
                raise SystemExit(f"leaderboard {SR['key']}/{r['id']}: printed under {r.get('t5_group')} in Table 5, "
                                 "one group is expected")
        refs = [r for r in printed if r["t5_group"][0] == "ref"]
        if not refs or refs[0]["kind"] != "row" or any(r["kind"] != "ref_seed0" for r in refs[1:]) or len(refs) > 2:
            raise SystemExit(f"leaderboard {SR['key']}: the reference row and at most its seed-0 line are expected")
        ref, ref0 = refs[0], (refs[1] if len(refs) == 2 else None)
        rest = [r for r in printed if r["t5_group"][0] not in ("ref", "imnt")]
        if any(r["err"] is None for r in rest):
            raise SystemExit(f"leaderboard {SR['key']}: a Table 5 row of a family, secondary or privileged group has no "
                             "error")
        s0 = [r for r in rest if _lb_seeds(r) == [0]]
        if s0 and ref0 is None:
            raise SystemExit(f"leaderboard {SR['key']}: rows run at seed 0 only, but Table 5 prints no seed-0 line")
        for r in rest:   # the block a row sits in is the reference its difference is taken from
            base = ref0 if r in s0 else ref
            if r["c"] is not None and abs(r["err"] - base["err"] - r["c"]["mean"]) > 1e-6:
                raise SystemExit(f"leaderboard {SR['key']}/{r['id']}: its difference {r['c']['mean']:+.6f} is not its "
                                 f"error minus the {'seed-0 line' if base is ref0 else 'reference'} "
                                 f"({r['err'] - base['err']:+.6f})")
        left_out += [(SR["key"], r) for r in printed if r["t5_group"][0] == "imnt"]
        a2_names += [T5_FAMILY[r["fam"]][1] for r in rest if r["t5_group"][0] == "a2"]
        seed0 += s0
        three = [r for r in rest if r not in s0]
        order = ([(ref, LB_IND[0])]
                 + [(r, LB_IND[0]) for r in by_err([r for r in three if r["t5_group"][0] != "priv"])]
                 + [(r, LB_IND[0]) for r in by_err([r for r in three if r["t5_group"][0] == "priv"])])
        if ref0 is not None:
            order += ([(ref0, LB_IND[1])]
                      + [(r, LB_IND[1]) for r in by_err([r for r in s0 if r["t5_group"][0] != "priv"])]
                      + [(r, LB_IND[1]) for r in by_err([r for r in s0 if r["t5_group"][0] == "priv"])])
        if sorted(id(r) for r, _ in order) != sorted(id(r) for r in refs + rest):
            raise SystemExit(f"leaderboard {SR['key']}: a Table 5 row is missing from, or repeated in, the order")
        if SR["key"] == T5_SPLIT_BEFORE:      # two floats, split where Table 5 splits
            split_at = len(L)
        L += (["\\addlinespace[4pt]"] if L else []) + [
            f"{mcN}{{\\textbf{{{SR['title']}}}. {SR['t5_head']}}} \\\\ % src: the slot title and the condition and "
            "reference clauses of the Table 5 slot header (SR['t5_head'], the same string)"]
        for r, ind in order:
            L.append(row(SR["key"], r, r["t5_group"][0], ind))
            n_rows += 1
            n_priv += r["t5_group"][0] == "priv"
        for note in SR.get("t5_notes", []):
            L.append(f"{mcN}{{{note}}} \\\\ % src: the note of the Table 5 slot (SR['t5_notes'], the same string)")
        log(f"leaderboard {SR['key']:8s} " + "; ".join(
            f"{r['id']} {r['err']:.3f}" for r, _ in order) + " | left out: "
            + (", ".join(r["id"] for k, r in left_out if k == SR["key"]) or "none"))
    cs = CTA_STATE
    cta = [] if cs.get("blinded") else sorted((r for r in cs.get("rows", []) if "t5" in r), key=lambda r: r["t5"]["pos"])
    if cta:
        L += ["\\addlinespace[4pt]",
              f"{mcN}{{\\textbf{{CT anatomy}} (label volume). {cs['t5_head']} {cs['t5_caveat']}}} \\\\ % src: the "
              "condition, reference and caveat clauses of the CT-anatomy header of Table 5 (CTA_STATE t5_head and "
              f"t5_caveat, the same strings; {_T('src a2 cta')}); one method at five reference levels, each against its "
              "own level on the reference labels, so the rows keep Table 5's order"]
        for r in cta:
            if r["err"] is None:
                raise SystemExit(f"leaderboard CT anatomy/{r['rid']}: a printed row without an error")
            L.append(row("ctanat", r, "cta", r["t5"]["indent"]))
            n_rows += 1
        # amend2 prints this sentence wherever the CT-anatomy rows print
        L.append(f"{mcN}{{\\quad\\textit{{{CTA_IN_SAMPLE}}}}} \\\\ % src: " + _T("cta in-sample src"))
        log("leaderboard ctanat   " + "; ".join(f"{r['rid']} {r['err']:.3f}" for r in cta) + " (Table 5 order)")
    vis = "\n".join(re.split(r"(?<!\\)%", x, maxsplit=1)[0] for x in L if x.startswith("\\quad"))
    marks = [m for s, m in LB_MARKS if s in vis]
    pend = sum(len(SR["pending"]) + sum(1 for r in SR["rows"] if r["kind"] == "row" and r["err"] is None)
               for SR in PLUG_REC) + (len(CTA_ROWS) if cs.get("blinded") else 0)
    a2_names = list(dict.fromkeys(a2_names))
    cap = ("Leaderboard of the methods plugged into each slot (exploratory). "
           # the engine slots (texture stage, renderer, CT anatomy) print in their own table
           "Each row is the row of Table~\\ref{tab:pluggability} (the guide's slots) or "
           "Table~\\ref{tab:pluggability-engine} (the engine's slots) for the same method, run in its slot's condition "
           "(slot headers, as in those tables), with "
           f"{steps[0]} steps, the {T5_CAP['episodes']} held-out episodes of Table~\\ref{{tab:ladder}} and contact on. "
           "In each slot the reference comes first, then the rows whose difference is taken from it, by mean error, "
           "lowest first" + (", with the privileged rows last" if n_priv else "") + ". "
           + ("Rows run at seed 0 only follow the reference's seed-0 line in the same order, since their differences "
              "are taken from that line and their error is that of one seed. " if seed0 else "")
           + "The order is by mean error only and is not a ranking: each row was tested against its slot's reference, "
           "not against the other rows, and where the decision is no detectable difference, the methods are not "
           "distinguished from the reference or from each other. "
           + ("The CT-anatomy rows keep the order of Table~\\ref{tab:pluggability-engine}, since each is scored against its "
              "own reference level. " if cta else "")
           # the authors' rows are components of the USB Gym release, cited as the software record (usbgym)
           + "Authors': our published component~\\cite{usbgym}. "
           + "Role: the row's group in those tables"
           + (f"; second family: the {' or the '.join(x if x[:2].isupper() else x[:1].lower() + x[1:] for x in a2_names)}"
              if a2_names else "")
           + ("; privileged: its group of privileged rows, which are secondary" if n_priv else "") + ". "
           + "Error, difference with its 95\\% paired $t$ CI and $k$/6, and decision"
           + (f", with the marks {_and(marks)}," if marks else "") + " as in those tables. "
           + ("" if not left_out else
              "The pretrained-backbone differences of Table~\\ref{tab:pluggability}, which have no error of their own, "
              "are left out." if all(r["kind"] == "imnt" for _, r in left_out) else
              "Rows of Table~\\ref{tab:pluggability} without an error of their own are left out.")
           + (" Rows still pending in Table~\\ref{tab:pluggability} are not listed." if pend else ""))
    cap_src = (f"% src: the Table 5 record (r['t5'], the cells _plug_body() printed; SR['t5_head'] and SR['t5_notes'], "
               f"its slot-header clauses and notes); {steps[0]} steps and {T5_CAP['episodes']} episodes: the values "
               "Table 5's caption checks (T5_CAP); left out: "
               + (", ".join(f"{k}/{r['id']}" for k, r in left_out) or "none") + f"; pending rows: {pend}; seed-0 rows: "
               + (", ".join(r["id"] for r in seed0) or "none") + f"; privileged-group rows: {n_priv}")
    T = [GEN.format(src="the record of pluggability() (the cells of tables/pluggability.tex, the same row objects)")]

    def head(first):
        # [htp] for the first float: Appendix C is a few lines of text, so the table follows its heading on the same
        # page instead of floating above it; the continuation goes to the top of the next page. A wrapped
        # method name hangs 3em in (2em is the indent of the seed-0 rows), so it does not
        # read as a heading or as a seed-0 row
        return (["\\begin{table*}[htp]" if first else "\\begin{table*}[tp]",
                 "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}"]
                + (["\\caption{" + cap + "} " + cap_src, "\\label{tab:leaderboard}"] if first else
                   ["\\ContinuedFloat", "\\caption[]{Continued.}"])
                + ["\\centering", "\\setlength{\\tabcolsep}{3pt}",
                   "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash\\hangindent=3em\\hangafter=1}X l r l "
                   ">{\\raggedright\\arraybackslash}p{0.2\\textwidth}@{}}",
                   "\\toprule",
                   "Slot and method & Role & \\begin{tabular}[b]{@{}r@{}}Error\\\\(mm)\\end{tabular} & "
                   "Difference (mm), 95\\% $t$ CI, $k$/6 & Decision \\\\ % src: the Table 5 columns of the same names",
                   "\\midrule"])
    foot = ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    parts = [L] if split_at is None else [L[:split_at], L[split_at:]]
    for i, part in enumerate(parts):
        body_ = list(part)
        while body_ and body_[0].startswith("\\addlinespace"):
            body_ = body_[1:]
        T += head(i == 0) + body_ + foot
    for x in T:   # every row has the cells of the header
        if x.startswith(("\\quad", "Slot and method")) and x.partition(" % src:")[0].count("&") != NC - 1:
            raise SystemExit(f"tab:leaderboard: {x.partition(' % src:')[0].count('&') + 1} cells in: {x[:140]}")
    keys = sorted({k.strip() for m_ in re.findall(r"\\cite\{([^}]*)\}", "\n".join(T)) for k in m_.split(",")})
    new = _new_bib_keys(keys)
    T.insert(0, "%% NEW BIB KEYS: " + (" ".join(new) or "none"))
    _no_plan_codes("leaderboard.tex", T)
    log(f"tab:leaderboard: {n_rows} rows in {len(PLUG_REC) + bool(cta)} slots; left out "
        + (", ".join(f"{k}/{r['id']}" for k, r in left_out) or "none") + f"; cites {len(keys)} keys, {len(new)} not in "
        f"{rel(REFS_TEX)}")
    return write(out, "leaderboard.tex", "\n".join(T))


# =================================================================================== tab:plug-cohort (Supplementary S6)
# Supplementary Table S6.7, the plug-cohort rows as measured (Appendix B.10: the rows that change one slot at a time
# on the 20 cohort CTs are reported as measured): per slot
# and row, the six-patient mean and the 20-CT mean of analysis_plug_cohort.json slots[*].six_means_mm and
# .cohort_means_mm (scripts/plug/cohort_plug_eval.py analyze, amend1, descriptive), and the 20-CT mean minus
# the slot reference's 20-CT mean, a difference of two archived means. No interval, k, p or decision is printed, because
# the record holds none for these rows (the cohort analysis is estimation only). The seeds of each row are counted from
# the record's pairs[]; a row with fewer seeds than its slot reference carries the double dagger. Labels are the Table 5
# row names (PLUG_SLOTS, print form), without citations.
PC_SLOT_ORDER = ("texture", "mask", "ctrl", "reward", "renderer")
PC_SLOT_HEAD = {
    "texture": "Texture stage: learned masks from each stage's matched U-Net; servo base, residual off",
    "mask": "Mask source: SonoSPADE frames; servo base fitted under each mask, residual off",
    "ctrl": "Controller: SonoSPADE frames; CT mask$^{\\dagger}$; each plugged controller",
    "reward": "Reward: SonoSPADE frames; CT mask$^{\\dagger}$; REINFORCE residual trained with each potential",
    "renderer": "Renderer: CT mask$^{\\dagger}$; policy trained and evaluated under each renderer"}
PC_DECIMALS = {"reward": 3}   # the reward rows differ by hundredths of a mm (Table 5 prints them to 3 decimals)


def _pc_row_id(method):
    """A plug-cohort method id -> its Table 5 row id ("sonospade (authors')" -> "sonospade")."""
    return method.replace(" (authors')", "")


def plug_cohort_tab(out):
    """tables/plug_cohort.tex (tab:plug-cohort, Supplementary Table S6.7)."""
    if _plug_cohort() is None:     # the leakage record must have passed (checked there)
        raise SystemExit(f"plug_cohort.tex: {rel(PC_FILE)} is missing")
    d = _plug_json(PC_FILE)
    S_ = d["slots"]
    seeds = collections.defaultdict(set)
    for p_ in d["pairs"]:
        seeds[(p_["slot"], p_["method"])].add(int(p_["seed"]))
    nc = {int(v["n_cohort"]) for v in S_.values()}
    n6 = {int(v["n_six"]) for v in S_.values()}
    if nc != {20} or n6 != {6}:
        raise SystemExit(f"plug_cohort.tex: n_cohort {nc} and n_six {n6}, the caption says 20 and 6")
    frel = rel(PC_FILE)
    L = [GEN.format(src=f"{frel} (slots.<slot>.six_means_mm, .cohort_means_mm, pairs[]) and the Table 5 row names"),
         "\\begin{table}[tp]",
         "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{Plug-cohort rows: each plugged method rerun with its frozen artifacts on the 20 cohort CTs "
         "(exploratory; no decision). Mask-source rows are servo bases with the residual off, as on the six patients, "
         "and so are the texture rows. Renderer rows are policies trained and evaluated under each renderer, so their "
         "six-patient errors differ from Table~\\ref{tab:pluggability} of the article. Errors: mean over patients or "
         "CTs of the seed-mean final translation error. The last column is a difference of two means, with no "
         "interval and no test. $^{\\dagger}$Privileged: simulator state that no rig provides. $^{\\ddagger}$Fewer "
         "seeds than designed: seed 0 only.}"
         f" % src: {frel} slots[*].n_six 6, n_cohort 20; mask and texture rows: plug_cohort/<slot>/<row>/ibvs_base_*.json "
         "results[0].weights zero_residual" + _T("cohort check date") + "; renderer rows: Appendix B.7 of the article, "
         f"sections/appendix_decisions.tex:360{_T('cohort note')}",
         "\\label{tab:plug-cohort}",
         "\\centering",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X r r r@{}}",
         "\\toprule",
         "Slot and row & \\begin{tabular}[b]{@{}r@{}}Six patients\\\\(mm)\\end{tabular} & "
         "\\begin{tabular}[b]{@{}r@{}}20 CTs\\\\(mm)\\end{tabular} & "
         "\\begin{tabular}[b]{@{}r@{}}20 CTs, minus\\\\the reference (mm)\\end{tabular} \\\\",
         "\\midrule"]
    log("\n== tab:plug-cohort (plug_cohort.tex), Supplementary S6: the plug-cohort rows as measured ==")
    if set(S_) != set(PC_SLOT_ORDER):
        raise SystemExit(f"plug_cohort.tex: slots {sorted(S_)} differ from {PC_SLOT_ORDER}")
    for i, key in enumerate(PC_SLOT_ORDER):
        v = S_[key]
        dd = PC_DECIMALS.get(key, 2)
        slot = _slot(key)
        ids = {_pc_row_id(m): m for m in v["methods_with_both_sides"]}
        known = [r["id"] for r in slot["rows"]]
        if set(ids) - set(known):
            raise SystemExit(f"plug_cohort.tex {key}: rows {sorted(set(ids) - set(known))} have no Table 5 name")
        ref = next(r for r in slot["rows"] if r["role"] == "ref")
        if ref["id"] not in ids:
            raise SystemExit(f"plug_cohort.tex {key}: the reference {ref['id']} has no cohort row")
        mref = ids[ref["id"]]
        ref_c = v["cohort_means_mm"][mref]
        ref_seeds = seeds[(key, mref)]
        L += (["\\addlinespace"] if i else []) + [f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{PC_SLOT_HEAD[key]}}}}} \\\\"]
        for r in slot["rows"]:
            if r["id"] not in ids:
                continue
            m_ = ids[r["id"]]
            e6, ec = v["six_means_mm"][m_], v["cohort_means_mm"][m_]
            few = len(seeds[(key, m_)]) < len(ref_seeds)
            lab = (_print_label(r["label"]) + ("$^{\\dagger}$" if r["priv"] else "")
                   + ("$^{\\ddagger}$" if few else ""))
            if r["id"] == ref["id"]:
                lab, dtxt = lab + ", reference", ""
            else:
                dtxt = sg(ec - ref_c, dd)
            L.append(f"\\quad {lab} & {e6:.{dd}f} & {ec:.{dd}f} & {dtxt} \\\\ % src: {frel} "
                     f"slots.{key}.six_means_mm[{m_}] = {e6}, slots.{key}.cohort_means_mm[{m_}] = {ec}"
                     + ("" if r["id"] == ref["id"] else
                        f", minus slots.{key}.cohort_means_mm[{mref}] = {ref_c} (difference of two archived means)")
                     + f"; seeds {sorted(seeds[(key, m_)])} (pairs[])")
            log(f"{key:8s} {r['id']:22s} six {e6:.3f} cohort {ec:.3f}"
                + ("" if r["id"] == ref["id"] else f" minus ref {ec - ref_c:+.3f}") + f" seeds {sorted(seeds[(key, m_)])}")
    L += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return write(out, "plug_cohort.tex", "\n".join(L))


# =================================================================================== tab:checks (v2.1)
# One row per testbed check (amendment 1) and per kept v2.1 run (the clean-tag rerun, Q8 to Q14, Q2R): the question it answers,
# the criterion fixed before the data, its outcome and its source. A row prints an outcome only from a finished output:
# the check outputs (admission gates, audits, target speckle, SB3 positive control, plug cohort), the Q14 band test
# computed here (_q14), and, for the kept runs without an analysis job in the final queue, the outputs of the
# committed analysis scripts in outputs/v2_1/analysis_plan/. A row without such an output says "Pending" with the
# reason. Every outcome is exploratory.
def _queue():
    """name -> (out path, exists) from the final v2.1 queue (each recorded path mapped to the result root in public
    mode, script_dirs.map_recorded)."""
    jobs = json.load(open(JOBS_FINAL))
    outs = {j["name"]: (SD.map_recorded(j["out"]) if j["out"] else j["out"]) for j in jobs}
    return {n: (p, bool(p) and os.path.exists(p)) for n, p in outs.items()}


def _q_count(Q, prefix):
    names = [n for n in Q if n.startswith(prefix)]
    return sum(Q[n][1] for n in names), len(names)


# Q14 of the v2.1 plan, CPU against MPS. (i) The CPU retrain: the band and the
# archived MPS cell and seed means come from plan_inputs.q14_tolerance(), and after imitation and final are tested
# each on its own; the per-patient CPU values are counted against the per-patient MPS seed range, descriptively. (ii)
# The neural and deterministic rows: the per-patient and per-episode tolerances of Q14_TOL. The cpu-agree job records
# tolerance_mm null and compares with MPS seed 0 only, so the band test of (i) is computed here from the raw outputs.
# No wall-time ordering of CPU against MPS is made: the archived MPS trainers shared the machine with other jobs
# (RESULTS.json compute.training.concurrent_jobs), and no rule for it was fixed.
PC_FILE = os.path.join(ROOT, "outputs", "v2_1", "checks", "plug_cohort", "analysis_plug_cohort.json")
PC_LEAK = os.path.join(ROOT, "outputs", "v2_1", "checks", "plug_cohort", "leakage_plug_cohort.json")
PC_NAME = {"ctrl": "controller", "reward": "reward", "renderer": "renderer", "mask": "mask source", "texture": "texture"}
_PC = {}


def _mm_s(x):
    """An MDD in mm: two decimals, three below 0.1 mm (the reward slot)."""
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def _plug_cohort():
    """amend1 (descriptive, no decision): per slot, the pairwise orderings of the original six kept on the
    cohort, Kendall tau-b with its bootstrap interval, and the planned MDD at 6, 20 and 26 patients, from
    scripts/plug/cohort_plug_eval.py analyze. None while the analysis has not written (cached)."""
    if _PC:
        return _PC
    if not os.path.exists(PC_FILE):
        return None
    d = _plug_json(PC_FILE)
    lk = json.load(open(PC_LEAK))
    if lk.get("passed") is not True:
        raise SystemExit(f"{rel(PC_LEAK)}: the plug-cohort leakage gate did not pass")
    S_ = d["slots"]
    if set(S_) != set(PC_NAME):
        raise SystemExit(f"{rel(PC_FILE)}: slots {sorted(S_)} differ from {sorted(PC_NAME)}")
    nc = {int(v["n_cohort"]) for v in S_.values()}
    if len(nc) != 1:
        raise SystemExit(f"{rel(PC_FILE)}: the slots differ in n_cohort")
    sl = {k_: {"kept": int(S_[k_]["sign_keeping"]["kept"]), "pairs": int(S_[k_]["sign_keeping"]["pairs"]),
               "tau": float(S_[k_]["kendall_tau_b"]), "ci": [float(x) for x in S_[k_]["tau_b_ci95"]],
               "mdd": {n_: float(S_[k_]["mdd_mm"][n_]) for n_ in ("n6", "n_cohort", "n_pooled")}}
          for k_ in PC_NAME}
    lo = min(sl.values(), key=lambda e: e["tau"])
    hi = max(sl.values(), key=lambda e: e["tau"])
    tau_s = (lambda e: f"{e['tau']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]")   # noqa: E731
    rng = {n_: f"{_mm_s(min(e['mdd'][n_] for e in sl.values()))} to {_mm_s(max(e['mdd'][n_] for e in sl.values()))}"
           for n_ in ("n6", "n_cohort", "n_pooled")}
    _PC.update({"n_c": nc.pop(), "slots": sl, "tau_lo": tau_s(lo), "tau_hi": tau_s(hi), "mdd_rng": rng,
                "src": (f"{rel(PC_FILE)} slots.<slot>.sign_keeping (kept, pairs), kendall_tau_b, tau_b_ci95, mdd_mm "
                        "(n6, n_cohort, n_pooled): " + "; ".join(
                            f"{k_} {e['kept']}/{e['pairs']}, tau-b {e['tau']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
                            f", MDD {e['mdd']['n6']:.3f}/{e['mdd']['n_cohort']:.3f}/{e['mdd']['n_pooled']:.3f}"
                            for k_, e in sl.items()) + f"; {rel(PC_LEAK)} passed true")})
    log("plug cohort: " + "; ".join(f"{k_} {e['kept']}/{e['pairs']} tau {e['tau']:.2f}" for k_, e in sl.items()))
    return _PC


Q14_BAND_MM = 1.0
Q14_PLAN_Y ={"imit": (11.78, (11.51, 11.55, 12.28)), "final": (11.81, (11.91, 11.51, 12.02))}
Q14_NEURAL_PP_MM, Q14_NEURAL_EP_MM, Q14_NEURAL_EP_FRAC = 0.10, 1.0, 0.95
Q14_CPU_EVAL = os.path.join(ROOT, "outputs", "v2_1", "cpu_repro", "contact", "ctmask_physics_s0.json")
Q14_MPS_EVAL = os.path.join(V2, "table4", "contact", "ctmask_physics_s{seed}.json")
Q14_AGREE = os.path.join(ROOT, "outputs", "v2_1", "cpu_repro", "agreement.json")
_Q14 = {}


def _q14():
    """Q14 (i) and (ii) from the raw outputs (cached): the CPU retrain against the mean of the three archived
    MPS seed means, after imitation and final, contact on, DR off; the CPU evaluation of the archived MPS policy; the
    queued CPU jobs. Returns None while the agreement job has not written."""
    if _Q14:
        return _Q14
    if not os.path.exists(Q14_AGREE):
        if SD.public():      # a public build reads finished results only: a missing file is not a pending job
            raise SystemExit(f"{rel(Q14_AGREE)}: not in the result root (MANIFEST.tsv): the Q14 rows of Table 3 and "
                             "of the checks summary need it")
        return None
    ag = json.load(open(Q14_AGREE))
    if _status_of(ag) != "exploratory, v2.1":
        raise SystemExit("cpu_repro/agreement.json: unexpected analysis_status")
    tol = PI.q14_tolerance(root=CODE_ROOT)    # row (i) of the v2.1 plan, checked verbatim in the development repository
    if tol != {"band_mm": Q14_BAND_MM, **{w: [y, list(ys)] for w, (y, ys) in Q14_PLAN_Y.items()}}:
        raise SystemExit("the Q14 tolerance row (i) differs: update Q14_BAND_MM and Q14_PLAN_Y")
    tae = next(v for k, v in ag["pairs"].items() if k.startswith("train_and_eval_device"))
    evd = next(v for k, v in ag["pairs"].items() if k.startswith("eval_device"))
    P = sorted(TEST_PATIENTS)
    out = {"phases": {}}
    for w in ("imit", "final"):
        sel = {"weights": w, "contact": "on"}
        cpu = _ppm([Q14_CPU_EVAL], sel)
        mps = [_ppm([Q14_MPS_EVAL.format(seed=s_)], sel) for s_ in (0, 1, 2)]
        X, Ys = _mean(cpu), [_mean(m_) for m_ in mps]
        Y = float(np.mean(Ys))
        y_add, ys_add = Q14_PLAN_Y[w]
        if round(Y, 2) != y_add or tuple(round(v, 2) for v in Ys) != ys_add:
            raise SystemExit(f"Q14 {w}: MPS seed means {Ys} differ from those of the v2.1 plan, {ys_add}")
        c_ag = tae["cells"][f"{w}/contact_on"]
        if abs(c_ag["cell_mean_cpu"] - X) > 1e-5 or abs(c_ag["cell_mean_mps"] - Ys[0]) > 1e-5:
            raise SystemExit(f"Q14 {w}: agreement.json cell means differ from the raw files")
        inside = [p_ for p_ in P if min(m_[p_] for m_ in mps) <= cpu[p_] <= max(m_[p_] for m_ in mps)]
        out["phases"][w] = {"X": X, "Y": Y, "Y_seeds": Ys, "diff": X - Y, "within": abs(X - Y) <= Q14_BAND_MM,
                            "k": len(inside), "n": len(P), "outside": [p_ for p_ in P if p_ not in inside],
                            "cpu_pp": cpu, "mps_range": {p_: (min(m_[p_] for m_ in mps), max(m_[p_] for m_ in mps))
                                                         for p_ in P}}
    tr = _out_path(ag["wall"]["cpu_trainer"]["path"])
    trj = json.load(open(tr))
    H = float(ag["wall"]["cpu_trainer"]["wall_sec"])
    if abs(float(trj.get("wall_sec", H)) - H) > 0.5:
        raise SystemExit("Q14: the CPU trainer's wall time differs between its file and agreement.json")
    pp = max(c_["per_patient_abs_diff_max"] for c_ in evd["cells"].values())
    ep = max(c_["episode_abs_diff_max"] for c_ in evd["cells"].values())
    Q = _queue()
    cpu_jobs = sorted(n for n in Q if n.startswith("v21:cohort:cpu:"))
    det = [n for n in cpu_jobs if any(k_ in n for k_ in ("static", "scripted", "pose_oracle", "ibvs_base"))]
    out.update({"H_sec": H, "H_h": H / 3600.0, "trainer": rel(tr), "ii_pp_max": pp, "ii_ep_max": ep,
                "ii_within": pp <= Q14_NEURAL_PP_MM and ep <= Q14_NEURAL_EP_MM,
                "jobs": cpu_jobs, "jobs_done": sum(Q[n][1] for n in cpu_jobs), "det_jobs": det,
                "agree_tol": ag.get("tolerance_mm"), "src": (
                    f"{rel(Q14_CPU_EVAL)} and {rel(Q14_MPS_EVAL.format(seed='{0,1,2}'))}, blocks [weights=imit|final, "
                    "contact=on, eval_randomize=off], per-patient means of episodes[*].final_pos (identical-episode "
                    "fingerprint checked); band: " + _T("q14 band src") + "; k: CPU per-patient mean inside [min, "
                    "max] of the three MPS seeds' per-patient "
                    f"means; H: {rel(Q14_AGREE)} wall.cpu_trainer.wall_sec (= wall_sec of the CPU trainer file that "
                    "wall.cpu_trainer.path names); (ii): "
                    "agreement.json pairs[eval_device ...].cells[*].per_patient_abs_diff_max and episode_abs_diff_max; "
                    "agreement.json tolerance_mm is null (the job applies no tolerance), so the band test is made here")})
    for w, e in out["phases"].items():
        log(f"Q14 (i) {w}: CPU {e['X']:.4f} mm against MPS seed mean {e['Y']:.4f} (seeds "
            + ", ".join(f"{v:.4f}" for v in e["Y_seeds"]) + f"): {e['diff']:+.4f} mm, "
            + ("within" if e["within"] else "OUTSIDE") + f" +/- {Q14_BAND_MM} mm; {e['k']}/{e['n']} patients inside the "
            f"MPS seed range (outside: {', '.join(e['outside']) or 'none'})")
    log(f"Q14 (i) H {H:.1f} s = {H / 3600.0:.3f} h; (ii) policy checkpoint: per-patient max {pp:.2g} mm, per-episode "
        f"max {ep:.2g} mm; CPU jobs {out['jobs_done']} of {len(cpu_jobs)}; deterministic-rung CPU evaluations queued: "
        f"{len(det)}")
    _Q14.update(out)
    return _Q14


def checks(out):
    """tab:checks: the testbed checks and the kept v2.1 runs, one row each."""
    log("\n== tab:checks (checks.tex), EXPLORATORY v2.1 ==")
    S = PROV if PLUG_PROVISIONAL else ""
    Q = _queue()
    rows = []   # (check, guards, rule, outcome, source, src comment)

    # 1. admission gates (amend1)
    gates_all = sorted(n for n in Q if n.startswith("v21:checks:gate:"))
    blind_g = [n for n in gates_all if _blinded(n)]    # amendment 2 gates (SAMUS, LPIPS): not read in this build
    gates = [n for n in gates_all if n not in blind_g]
    adm, failed, pend, cuda = [], [], [], 0
    for n in gates:
        p, ok = Q[n]
        if os.path.exists(p[:-5] + ".failed.json"):
            failed.append(n)
        elif ok:
            j = _plug_json(p)
            if j.get("admitted") is not True:
                raise SystemExit(f"{rel(p)}: gate output without admitted = true")
            adm.append(n)
            cuda += int(j.get("cuda_moves_total", 0))
        else:
            pend.append(n.split(":", 3)[3])
    pend_txt = (" Pending: " + ", ".join(x.replace("_", "\\_") for x in pend) + ".") if pend else ""
    if pend == ["mask:nnunet"]:
        pend_txt = " The nnU-Net gate is pending."
    out_g = (f"{len(adm)} of {len(gates)} adapters admitted{S}. "
             + ("None was refused. " if not failed else f"{len(failed)} refused{S}. ")
             + f"The admitted adapters made {cuda} CUDA moves{S}.{pend_txt}"
             + (f" The {len(blind_g)} amendment 2 gates are {A2_BLIND_TXT}." if blind_g else ""))
    rows.append((_T("row a1 gate"),
                 "An adapter that reads privileged state, needs CUDA, or acts differently on CPU and MPS.",
                 "A pose-free adapter must give bit-identical outputs with its privileged channels removed or "
                 "poisoned. Every privileged adapter must be detected. No CUDA move is allowed. CPU and MPS must agree "
                 "within preset tolerances.",
                 out_g, "checks/gate/*.json",
                 f"checks/gate/*.json admitted, cuda_moves_total; gate jobs v21:checks:gate:* in "
                 f"outputs/v2_1/jobs_final_r3.json ({len(adm)} admitted, {len(failed)} failed, pending {pend})"))

    # 2. identical-episode audit (amend1)
    audits = sorted(n for n in Q if n.startswith("v21:checks:audit:"))
    a_pass, a_fail, a_blocks = [], [], 0
    for n in audits:
        p, ok = Q[n]
        fl = sorted(glob.glob(p[:-5] + ".failed*.json"))
        if fl:
            a_fail.append(n.split(":")[-1])
        elif ok:
            j = _plug_json(p)
            if j.get("passed") is not True or j.get("refused"):
                raise SystemExit(f"{rel(p)}: audit output without passed = true and an empty refused list")
            a_pass.append(n.split(":")[-1])
            # n_blocks is not read or printed: the blocks of the reward audit include those of the blinded LPIPS row
    if a_fail:
        out_a = (f"Refused in {len(a_fail)} of {len(audits)} slots ({', '.join(a_fail)}); their aggregators are "
                 f"blocked{S}.")
    elif a_pass:
        AUD_ = {"texture": "texture", "seg": "mask source", "ctrl": "controller", "reward": "reward",
                "renderer": "renderer", "plugcohort": "plug cohort"}
        a_pass = sorted(a_pass, key=lambda x: list(AUD_).index(x) if x in AUD_ else 99)
        out_a = (f"Passed in {len(a_pass)} of {len(audits)} slots ({', '.join(AUD_.get(x, x) for x in a_pass)}), "
                 f"none refused{S}."
                 + (" The reward audit also covers the blocks of the blinded LPIPS row, whose results this build "
                    "does not read." if "reward" in a_pass and BLIND_A2 else "")
                 + ("" if len(a_pass) == len(audits) else f" The other {len(audits) - len(a_pass)} are pending."))
    else:
        out_a = f"Pending: 0 of {len(audits)} audits."
    rows.append((_T("row a1 audit"),
                 "A plugged row scored on episodes other than its reference's.",
                 "Every per-episode setup (episode, case, organ, window and start errors, to $10^{-6}$) must equal "
                 "that of the archived static rung. A mismatch blocks the slot's aggregator.",
                 out_a, "checks/audit/<slot>.json",
                 f"checks/audit/<slot>.json passed and refused (n_blocks not printed; the reward audit includes the "
                 f"LPIPS blocks{_T('a2 audit note')}, {'blinded in this build' if BLIND_A2 else 'read in this build'}); "
                 f"audit jobs v21:checks:audit:* in outputs/v2_1/jobs_final_r3.json (passed {a_pass}, refused "
                 f"{a_fail})"))

    # 3. independent target speckle (amend1)
    f = os.path.join(CHECKS_ROOT, "tspeckle", "summary.json")
    if os.path.exists(f):
        t = json.load(open(f))
        if t.get("analysis_status") not in ("exploratory, v2.1", PLUG_STATUS):
            raise SystemExit(f"{rel(f)}: unexpected analysis_status {t.get('analysis_status')!r}")
        ch = [float(r["final_pos"]["mean_change"]) for r in t["rows"]]
        n_cav = sum(abs(x) > 1.0 for x in ch)
        out_t = ((f"All {len(ch)} rows change by less than 1~mm (from {sg(min(ch))} to {sg(max(ch))}~mm){S}. "
                  "No row carries the caveat.") if n_cav == 0 else
                 f"{n_cav} of {len(ch)} rows change by more than 1~mm{S} and carry the caveat.")
        src_t = "checks/tspeckle/summary.json rows[*].final_pos.mean_change (" + ", ".join(f"{x:+.4f}" for x in ch) + ")"
    else:
        out_t, src_t = "Pending.", "checks/tspeckle/summary.json (missing)"
    rows.append(("Independent target speckle",
                 "Does the appearance score match the shared speckle field instead of the anatomy?",
                 "Report the change in final error when the target's speckle is independent of the observation's, "
                 "with its paired $t$ interval. A change above 1~mm adds a caveat to that row.",
                 out_t, "checks/tspeckle/summary.json", src_t))

    # 4. SB3 positive control (amend1)
    f = os.path.join(CHECKS_ROOT, "sb3pos", "verdict.json")
    if os.path.exists(f):
        v = _plug_json(f)
        sd = v["seeds"]
        lm = [float(s["learner_mm"]) for s in sd]
        om = sorted({round(float(s["oracle_mm"]), 6) for s in sd})
        n_pass = sum(bool(s["seed_passes"]) for s in sd)
        if len(om) != 1:
            raise SystemExit(f"{rel(f)}: the oracle differs across seeds; rewrite the checks row")
        bound = om[0] + 3.0
        out_s = (f"{v['verdict']} in {n_pass} of {len(sd)} seeds{S}. The learner ends at {min(lm):.2f} to "
                 f"{max(lm):.2f}~mm, against a pass bound of {bound:.2f}~mm (oracle {om[0]:.2f}~mm plus 3~mm){S}.")
        src_s = (f"checks/sb3pos/verdict.json verdict ({v['verdict']}), rl_slot_certified ({v['rl_slot_certified']}), "
                 f"seeds[*].learner_mm ({', '.join(f'{x:.4f}' for x in lm)}), seeds[*].oracle_mm ({om[0]:.4f}), "
                 f"seeds[*].seed_passes")
    else:
        out_s, src_s = "Pending.", "checks/sb3pos/verdict.json (missing)"
    rows.append(("Positive control of the RL slot",
                 "Can a Stable-Baselines3 (SB3) learner learn an easy variant of the task in the "
                 "reinforcement-learning slot?",
                 "Pass if SB3 PPO ends within 3~mm of the pose oracle in all 3 seeds at 57,600 steps. The test uses "
                 "an easy privileged variant with pose observations and a pose reward, on training patients.",
                 out_s, "checks/sb3pos/verdict.json", src_s))

    # 5. plugged methods on the cohort (amend1): the analysis exists, so its descriptive
    # outcome prints (per slot: kept pairwise orderings, Kendall tau-b with its bootstrap interval, MDD at 6, 20, 26)
    dn, tot = _q_count(Q, "v21:checks:plugcohort:")
    pc = _plug_cohort()
    if pc is None:
        out_pc, src_pc = (f"Pending: {dn} of {tot} runs done.",
                          f"jobs v21:checks:plugcohort:* in outputs/v2_1/jobs_final_r3.json ({dn} of {tot} outputs exist)")
    else:
        # the job count does not print
        out_pc = (f"Pairwise orderings of the original six kept on {pc['n_c']} new CTs: "
                  + ", ".join(f"{PC_NAME[k_]} {e['kept']} of {e['pairs']}" for k_, e in pc["slots"].items())
                  + f". Kendall $\\tau_b$ ranges from {pc['tau_lo']} to {pc['tau_hi']}{S}. Pre-specified MDDs: "
                  + f"{pc['mdd_rng']['n6']}~mm at $n = 6$, {pc['mdd_rng']['n_cohort']}~mm at 20 and "
                    f"{pc['mdd_rng']['n_pooled']}~mm at 26{S} (\\ref{{app:decisions-cohort}}).")
        src_pc = (f"jobs v21:checks:plugcohort:* in outputs/v2_1/jobs_final_r3.json ({dn} of {tot} outputs exist); "
                  + pc["src"])
    rows.append(("Plugged methods on the 20-CT cohort",
                 "Does the ordering of methods recur beyond the six held-out patients?",
                 "Descriptive: Kendall $\\tau_b$ against the original six with a bootstrap interval, the sign-keeping "
                 "of pairwise differences, and the minimum detectable difference. No decision is attached.",
                 out_pc, "checks/plug_cohort/analysis_plug_cohort.json", src_pc))

    # kept v2.1 runs: the final queue holds no analysis job for them; the committed analysis scripts wrote their
    # outcomes to outputs/v2_1/analysis_plan/, and each row prints the outcome from that file
    APD = os.path.join(ROOT, "outputs", "v2_1", "analysis_plan")

    def _ap(name):
        d_ = json.load(open(os.path.join(APD, name)))
        if _status_of(d_) != "exploratory, v2.1":
            raise SystemExit(f"outputs/v2_1/analysis_plan/{name}: unexpected analysis_status")
        return d_
    se, q10, q11, q12, q2r = (_ap(x) for x in ("section_e_v2.json", "q10_label_controls_v2.json",
                                                "q11_q13_v3.json", "q12_registration_nulls_v2.json",
                                                "q2r_cohort_v2.json"))
    q8 = _q8_all()   # every condition rescored; asserted against q8_gap_cause_v2.json
    # AS(row) prints nothing
    AS = lambda r_: ""   # noqa: E731
    dse = se["decisions"]
    if (dse["rerun"]["key"], dse["batchnorm"]["key"], dse["horizon_36"]["key"]) != ("within", "plan_rule_change",
                                                                                    "otherwise"):
        raise SystemExit("section_e_v2.json: a decision of the clean-tag rerun changed; rewrite its row of checks()")
    er = se["rerun"]
    hz = se["horizon_36"]["ctmask_base"]["result"]
    # the rerun clause is not printed; it read: "Rerun within the preset tolerance
    # ({er['n_episodes_all_fields_identical']} of {er['n_episodes']} episodes bit-identical)."
    log(f"{_T('se log name')} rerun ({_T('rerun tag')}): {er['n_episodes_all_fields_identical']:,} of "
        f"{er['n_episodes']:,} episodes bit-identical; {_T('rerun script row')}")
    out_e = ("BatchNorm in eval mode changes pre-specified outcomes, so both modes are reported. "
             f"No horizon effect: at 36 steps the CT-mask servo base changes by {sg(hz['mean'])}~mm "
             f"{ci_txt(hz['ci_t'])}, {hz['sign_k']}/{hz.get('sign_n', 6)}.")
    d8 = q8["decisions"]
    if d8["Q8_class"] != "mixed" or not d8["Q8_class_decided_without_rescoring"]:
        raise SystemExit("q8_gap_cause_all_conditions.json: the Q8 class changed; rewrite the Q8 row of checks()")
    resc = q8["q8"]["rescoring"]
    inc8 = [re.search(r"rescoring of (C\d)", x["item"]).group(1) for x in q8["deviations"]
            if x.get("status") == "not completed" and re.search(r"rescoring of (C\d)", x.get("item", ""))]
    if inc8 or not q8["deviations"][0]["item"].startswith("Q8 (b) primary look-ahead bound, sampling seeds [1, 2]"):
        raise SystemExit("q8_gap_cause_all_conditions.json: a rescoring is incomplete again, or deviations[0] is not "
                         "the seed-0 bound; rewrite the Q8 row of checks()")
    # the primary bound is the seed mean (sampling seeds 0 to 2) of q8_gap_cause_look_s012.json; _q8_look asserts
    # every part that the seeds cannot change against the file above
    q8L = _q8_look()
    if q8L["q8"]["classification"]["bound_seeds_used"] != [0, 1, 2] or q8L["decisions"]["Q8_class"] != "mixed":
        raise SystemExit("q8_gap_cause_look_s012.json: the bound seeds or the class changed; rewrite the Q8 row")
    dt8 = _A345["q8_dates"]
    # the rescoring dates ({dt8}) and the script clause are not printed
    out_8 = (f"Mixed class: both single-cause classes fail on their bound clauses. Rescoring on fresh speckle "
             f"covers all {len(resc['conditions_rescored'])} conditions, and "
             "the class stays mixed, decided without the rescoring. The primary bound is the mean over sampling "
             "seeds 0 to 2.")
    d9 = d8["Q9_A"]
    if d9 != {"R2_minus_base_ct": "reversed", "R2_minus_base_learned": "no_difference"}:
        raise SystemExit("q8_gap_cause_v2.json: a Q9-A decision changed; rewrite the Q9-A row of checks()")
    f9 = q8["q9"]["family_Q9_A"]
    m9 = f9["members"]["R2_minus_base_ct"]
    m9 = m9.get("contrast", m9)
    out_9 = (f"Under CT masks the hill-climb ends {m9['mean']:.2f}~mm above the servo base, {m9['sign_k']}/6 (Holm "
             f"$p$ {f9['p_holm']['R2_minus_base_ct']:.3f}); under learned masks no difference is detected.")
    d10 = q10["decisions"]
    if (set(d10["Q10_L_keys"].values()) != {"holds"} or d10["Q10_P_P_minus_Ph"] != "holds"
            or d10["Q10_CL_keys"]["permute_total_error_change"] != "holds"
            or any(d10["Q10_CL_keys"][k_] != "no_difference" for k_ in ("x_p(3)", "x_p(6)"))):
        raise SystemExit("q10_label_controls_v2.json: a Q10 decision changed; rewrite the Q10 row of checks()")
    out_10 = ("Rendered masks follow the offset and permuted labels; the closed loop does not detectably follow the "
              "offsets, and permuted organ identities raise the error. The label-painted stage lowers the error from "
              "physics, but its equivalence with SonoSPADE is not shown, so its class is unresolved.")
    d11 = q11["Q11"]["decision"]
    if set(d11["keys"].values()) != {"holds"}:
        raise SystemExit("q11_q13_v3.json: a Q11 decision changed; rewrite the Q11 row of checks()")
    out_11 = (f"SonoSPADE stays lower than budget-matched CUT, CycleGAN and physics in all {len(d11['keys'])} contrasts "
              f"(Holm $p$ {max(d11['holm_p'].values()):.3f} each).")
    d12 = q12["decisions"]
    n12a, n12b, m12 = d12["Q12a"]["n_holding"], d12["Q12b"]["n_holding"], q12["families"]["Q12a"]["m"]
    hold12 = [k_ for k_, v_ in d12["Q12a"].items() if v_ == "holds" or (isinstance(v_, dict) and v_.get("key") == "holds")]
    if n12a != 1 or n12b != 0 or hold12 != ["sonospade/kidneyus/kidney"]:
        raise SystemExit("q12_registration_nulls_v2.json: a Q12 decision changed; rewrite the Q12 row of checks()")
    out_12 = (f"Selection by the score beats a random state of the same trajectory in {n12a} of {m12} cells "
              f"(SonoSPADE, kidney); the policy's keep-best gain exceeds matched undirected motion in {n12b} of {m12}.")
    d2 = q2r["decisions"]
    if (any(d2["Q2R-P"][k_]["key"] != "holds" for k_ in ("C1", "C2", "C3")) or d2["ordering"] != "recurs"):
        raise SystemExit("q2r_cohort_v2.json: a Q2R decision changed; rewrite the Q2R row of checks()")
    out_2r = (f"On {d2['n_c']} new CTs, both contrasts replicate, RL changes the error by less than 1~mm in every "
              "arm, and the ordering recurs.")
    q13 = q11["Q13"]["decision"]["lateral"]
    if q13["verdict_key"] != "none_passes" or q13["passing_variants"]:
        raise SystemExit("q11_q13_v3.json: the Q13 lateral verdict changed; rewrite the Q13 row of checks()")
    # the secondary family Q13-S (the gain deficit) prints in this row too
    q13g = q11["Q13"]["decision"]["gain"]
    if q13g["verdict_key"] != "none_passes" or q13g["passing_variants"]:
        raise SystemExit("q11_q13_v3.json: the Q13 gain verdict changed; rewrite the Q13 row of checks()")
    out_13 = (f"None of the {len(q13['holm_p'])} settings removes half of the planted deficit (Holm $p$ "
              f"{min(q13['holm_p'].values()):.3f} each). The planted bias failed the reward-visibility rule, so the "
              "outcome is inconclusive. Nor does any setting remove half of the secondary deficit, which halves the "
              f"servo base's command at execution (Holm $p$ {min(q13g['holm_p'].values()):.3f} each).")
    q14 = _q14()
    if q14 is None:
        out_14 = f"Pending: {_q_count(Q, 'v21:cohort:cpu:')[0]} of {_q_count(Q, 'v21:cohort:cpu:')[1]} jobs done."
        src_14 = "jobs v21:cohort:cpu:* in outputs/v2_1/jobs_final_r3.json"
    else:
        fi, im = q14["phases"]["final"], q14["phases"]["imit"]
        if not (fi["within"] and im["within"]):
            raise SystemExit("Q14 (i): a CPU retrain cell lies outside +/- 1.0 mm; print the Outside sentence")
        out_14 = (f"{q14['jobs_done']} of {len(q14['jobs'])} jobs done. The CPU retrain ends at {fi['X']:.2f}~mm, and "
                  f"{im['X']:.2f}~mm after imitation, against MPS seed means of {fi['Y']:.2f} and {im['Y']:.2f}~mm: "
                  f"within $\\pm 1.0$~mm{S}. The CPU evaluation of the archived policy changes no patient by more "
                  f"than {q14['ii_pp_max']:.4f}~mm{S}.")
        src_14 = (f"jobs v21:cohort:cpu:* in outputs/v2_1/jobs_final_r3.json ({q14['jobs_done']} of {len(q14['jobs'])} "
                  f"outputs: {', '.join(q14['jobs'])}); final {fi['X']:.4f} against {fi['Y']:.4f} (seeds "
                  + ", ".join(f"{v:.4f}" for v in fi["Y_seeds"]) + f"), k {fi['k']}/6; after imitation {im['X']:.4f} "
                  f"against {im['Y']:.4f}, k {im['k']}/6; H {q14['H_sec']:.1f} s; " + q14["src"])
        q2 = _q14_ii()   # Q14 (ii), run later at tag v2.1.2
        if q2 is None:
            out_14 += " Part (ii) was not queued."
        else:
            nb = {k.split(":")[-1].replace("_", " "): v for k, v in q2["neu"].items()}
            vv = list(nb.items())
            out_14 += (f" (ii){_T('q14 ii note')}: the deterministic rungs differ "
                       f"by at most {q2['det_max']:.2f}~mm per episode, within 0.01~mm. The learned-mask servo base "
                       "(physics, seed~0) differs by up to "
                       + " and ".join(f"{v['largest_patient_mm']:.2f}~mm per patient with {k}" for k, v in vv)
                       + ", outside the 0.10~mm tolerance; "
                       + " and ".join(f"{100 * v['frac_within_episode']:.1f}\\%" for _, v in vv)
                       + " of its episodes lie within 1~mm.")
            src_14 += ("; (ii): outputs/v2_1/cpu_repro/ladder/agreement_rungs.json checks.<cell>:<weights>:<contact>."
                       "{largest_difference, largest_patient_mm, frac_within_episode} ("
                       + "; ".join(f"{k} {v['largest_difference']:.4g} / {v['largest_patient_mm']:.4f} / "
                                   f"{v['frac_within_episode']:.4f}" for k, v in {**q2['det'], **q2['neu']}.items())
                       + "), largest_patient_mm_neural " + f"{q2['X']:.4f}, tolerances (Q14 table), "
                       "sentence_key outside, re-derived here from the five CPU rung files and outputs/v2/table4/ladder "
                       "within 1e-9")
    kept = [
        ("BatchNorm in eval mode and 36-step experiments",
         "Do the results depend on BatchNorm in train mode, and is the servo's error a horizon limit?",
         "Eval minus train mode per arm, with its $t$ interval. "
         "A horizon effect if the CT-mask servo's 36-step minus 12-step change is at most $-1$~mm and below 0 under a "
         "paired $t$-test, with at least 5 of 6 patients.",
         out_e, "analysis_plan/section_e_v2.json",
         "outputs/v2_1/analysis_plan/section_e_v2.json decisions.rerun.key (within), rerun.n_episodes_all_fields_"
         "identical and .n_episodes, decisions.batchnorm.key (plan_rule_change), decisions.horizon_36.key (otherwise) "
         "and horizon_36.ctmask_base.result (mean, ci_t, sign_k); " + _T("section_e script") + "; runs "
         f"outputs/v2_1/s21_rerun/, s21_bneval/, s21_h36/; no {_T('se job name')} analysis job in outputs/v2_1/jobs_final_r3.json "
         + _T("rerun row tail")),
        ("Q8, cause of the oracle-to-CT-mask gap (the gap between the pose oracle and the CT-mask servo)",
         "Does the gap come from the controller or from the mask information?",
         "Classify the gap as controller-limited, information-limited or mixed. The classes use the poses that "
         "score within 0.01 of the target, rescored on fresh speckle. They also use one-sided $t$ bounds on a "
         "privileged look-ahead that queries the simulator, with margins of 3~mm and 1~mm.",
         out_8, "analysis_plan/q8_gap_cause_all_conditions.json",
         "outputs/v2_1/analysis_plan/q8_gap_cause_all_conditions.json decisions.Q8_class (mixed), "
         "Q8_class_decided_without_rescoring (true), q8.rescoring.conditions_rescored ("
         + ", ".join(resc["conditions_rescored"]) + "), q8.rescoring.files[*] provenance.created_utc (" + ", ".join(dt8)
         + "); bound seeds: outputs/v2_1/analysis_plan/q8_gap_cause_look_s012.json q8.classification.bound_seeds_used "
         "[0, 1, 2] and decisions.Q8_class (mixed), asserted part by part against q8_gap_cause_all_conditions.json "
         "(_q8_look); q8_gap_cause_all_conditions.json asserted against q8_gap_cause_v2.json (classification, "
         "families, reference means, C0 and C1 level sets, Q9, printed_sentences [0..8] and v2 [9..22] = [14..27])"
         + _T("rows q8")),
        ("Q9-A, hill-climb on the match score",
         "Is the servo base's error the limit of pose-free control?",
         "Paired $t$-test with Holm over 2 contrasts (CT and learned masks), and at least 5 of 6 patients sharing "
         "the sign.",
         out_9, "analysis_plan/q8_gap_cause_v2.json",
         "outputs/v2_1/analysis_plan/q8_gap_cause_v2.json decisions.Q9_A (R2_minus_base_ct reversed, "
         "R2_minus_base_learned no_difference) and q9.family_Q9_A (p_holm, members); runs "
         "outputs/v2_1/s23_rungs/hillclimb_physics_{ct,matched}_s{0,1,2}.json; no Q9-A analysis job in "
         "outputs/v2_1/jobs_final_r3.json" + _T("rerun row")),
        ("Q10, label-information controls",
         "Does a learned-mask lead come from label information given to a label-conditioned stage?",
         "Paired $t$-tests with Holm in three families, one-sided for the closed loop. A stage that paints the "
         "labels onto the physics frame is tested for equivalence at 1~mm.",
         out_10, "analysis_plan/q10_label_controls_v2.json",
         "outputs/v2_1/analysis_plan/q10_label_controls_v2.json decisions.Q10_L_keys (holds x3), Q10_CL_keys "
         "(x_p(3), x_p(6) no_difference; permute_total_error_change holds), Q10_P_P_minus_Ph (holds), painting_class "
         "(" + d10["painting_class"] + ")"),
        ("Q11, texture budget and texture seeds",
         "Does a stage lead come from an unequal training budget or from a single texture seed?",
         "Paired $t$-test with Holm over 3 contrasts, and at least 5 of 6 patients sharing the sign.",
         out_11, "analysis_plan/q11_q13_v3.json",
         "outputs/v2_1/analysis_plan/q11_q13_v3.json Q11.decision.keys (holds x3) and .holm_p; the job aggregate "
         "outputs/v2_1/texture/q11_texture_budget.json applies the v2 rule (cross-check only)"),
        ("Q12, keep-best nulls on real frames",
         "Does a keep-best gain on real frames come from selection by the score rather than from directed control?",
         "Subject-level exact sign-flip test with Holm over 8 testable cells, plus a crossed bootstrap interval.",
         out_12, "analysis_plan/q12_registration_nulls_v2.json",
         "outputs/v2_1/analysis_plan/q12_registration_nulls_v2.json decisions.Q12a.n_holding (1: "
         "sonospade/kidneyus/kidney) and Q12b.n_holding (0), families.Q12a.m (8); the job aggregate "
         "outputs/v2_1/regnull/Q12_registration_nulls.json uses 16-cell families (cross-check only)"),
        ("Q13, positive control of REINFORCE",
         "Can REINFORCE remove a planted deficit when one exists?",
         "One-sided paired $t$-test with Holm over 3 variants, at least 5 of 6 patients, and every seed at 50\\% "
         "recovery or more.",
         out_13, "analysis_plan/q11_q13_v3.json",
         "outputs/v2_1/analysis_plan/q11_q13_v3.json Q13.decision.lateral (verdict_key none_passes, holm_p) and "
         "Q13.bias_selection_pilot (final S change -0.008 against the -0.03 visibility rule" + _T("row q13") + "); "
         "outputs/v2_1/rlpos/rlpos_summary.json overall.key (none_pass, cross-check)"),
        ("Q14, CPU against MPS",
         "Results that hold on the Apple GPU only.",
         "(i) CPU retrain: mean error at most 1.0~mm from the MPS seed mean. (ii) CPU evaluation of the archived "
         "checkpoints: per episode at most 0.01~mm (deterministic rungs); per patient at most 0.10~mm, and per episode at "
         "most 1.0~mm in 95\\% of the episodes (neural rows).",
         out_14, "cpu_repro/agreement.json", src_14),
        ("Q2R, replication on 20 new CTs",
         "Do the main contrasts recur beyond the six held-out patients?",
         "Exact sign-flip test with Holm over 3 contrasts. The claim that RL changes the error by less than 1~mm "
         "in every arm uses an intersection-union equivalence test.",
         out_2r, "analysis_plan/q2r_cohort_v2.json",
         "outputs/v2_1/analysis_plan/q2r_cohort_v2.json decisions.Q2R-P (C1, C2, C3 holds), decisions.ordering "
         "(recurs), n_c (20); the job aggregate outputs/v2_1/cohort/analysis_q2r.json (cross-check only)"),
    ]
    rl = json.load(open(os.path.join(ROOT, "outputs", "v2_1", "rlpos", "rlpos_summary.json")))
    if _dig(rl, "overall", "key") != "none_pass":
        raise SystemExit("rlpos_summary.json overall.key changed: update the Q13 row of checks()")
    a345 = _checks_rows_a345()   # the check records of amendments 3 to 5 (flags lifted only)
    # the admission-gate and identical-episode audit rows, the Q14 row and the check records of amendments 3 to 5 are
    # computed and asserted above and logged here; they do not print
    moved = rows[:2] + [k_ for k_ in kept if k_[0].startswith("Q14")] + list(a345)
    rows = rows[2:]
    kept = [k_ for k_ in kept if not k_[0].startswith("Q14")]
    for a_, _, _, d_, e_, _ in moved:
        log(f"[{_T('not printed tag')}] {a_}: {d_} (source {e_})")
    a345 = []

    L = [GEN.format(src="outputs/v2_1/checks/ (gate, audit, tspeckle, sb3pos, plug_cohort), outputs/v2_1/jobs_final_r3.json "
                        "and the kept v2.1 analysis outputs named in each row"),
         "\\begin{table*}[tp]",
         "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{Controls and exploratory tests (exploratory, pre-specified rules). Each row gives "
         "the question that the analysis answers, its criterion fixed before the data and its outcome. Q8 to Q13 and "
         "Q2R are questions of the pre-specified exploratory analyses."
         + (" Pending: no output that applies the pre-specified rule yet." if any(str(r_[3]).startswith("Pending")
                                                                          for r_ in rows + kept) else "") + "}",
         "\\label{tab:checks}"]
    # the kept v2.1 runs continue in a second float (\\ContinuedFloat, as Table 5): one float is taller than the page
    head_ = ["\\centering",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{0.19\\textwidth} "
             ">{\\raggedright\\arraybackslash\\hsize=0.75\\hsize}X >{\\raggedright\\arraybackslash\\hsize=1.25\\hsize}X "
             ">{\\raggedright\\arraybackslash}X@{}}",
             "\\toprule",
             "Analysis & Question & Criterion fixed before the data & Outcome \\\\",
             "\\midrule"]
    foot_ = ["\\bottomrule", "\\end{tabularx}", "\\end{table*}"]
    L += head_ + ["\\multicolumn{4}{@{}l}{\\textit{Controls}} \\\\"]
    cont_ = ["\\begin{table*}[tp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize", "\\ContinuedFloat",
             "\\caption[]{Continued.}"]
    for i, (a, b, c, d, e, g) in enumerate(rows + kept + a345):
        if i == len(rows):
            L += foot_ + cont_ + head_ + ["\\multicolumn{4}{@{}l}{\\textit{Exploratory tests}} \\\\"]
        if a345 and i == len(rows) + len(kept):
            L += foot_ + cont_ + head_ + ["\\multicolumn{4}{@{}l}{\\textit{Check records of amendments 3 to 5}} \\\\"]
        L.append(f"{a} & {b} & {c} & {d} \\\\[2pt] % src: {e}; {g}")   # source path in the src comment only
        log(f"{a}: {d}")
    L += foot_ + [""]
    return write(out, "checks.tex", "\n".join(L))


# =================================================================================== tab:checks-summary (body)
# The table tab:checks-summary of the article (Table 7): one row per
# testbed check, grouped as guards, controls, repeatability, replication and one cross-reference, with what it guards
# against, its rule fixed before the data, its outcome with absolute values first, whether the outcome meets the rule,
# and the plan item that fixed the rule. Plan codes print only in the last column. Every v2.1 number is computed here
# from its output file (no number is typed); v2.1 cells are boxed by \verifyfill and marked with a section sign until
# the tag-v2.1 aggregation, and archived v2 cells print plain. The full rules and sources stay in tables/checks.tex.
# No output of a blinded amendment 2 job is opened: gate jobs are taken from the queue by name and blinded names are
# skipped before any path is touched.
CS_TOL_MM = 1e-9
# rows that are computed but not printed (the admission and leakage gates, the audit and the reruns, CPU against
# MPS, the retrain drift, whose comparison is note (b) of Table 5, and the established-simulator cross-reference)
CS_MOVED = ("Admission gate", "Leakage gate", "Identical-episode audit", "Clean-tag rerun", "Registration rerun",
            "CPU against MPS", "Retrain drift", "Established simulator code")
CS_REGROUP = {"Independent target speckle": "Controls", "Plugged methods on the cohort": "Replication"}
CS_GROUP_PRINT = {"Repeatability": "Sensitivity"}
CS_GROUP_ORDER = ("Controls", "Sensitivity", "Replication")
CS_ROW_FIRST = {"Independent target speckle": 0, "20-CT cohort": 0}   # the first row of its group; others keep order


def _out_path(p):
    """A path recorded by a job in the v2.1 checkout, mapped to this checkout (outputs/ is shared between the two);
    to the result root in public mode (script_dirs.map_recorded, which also maps the segmenter repository's)."""
    if SD.public():
        return SD.map_recorded(p)
    i = str(p).find("/outputs/")
    return os.path.join(ROOT, str(p)[i + 1:]) if i >= 0 else p


def _ppm(files, sel):
    """Per-patient mean final translation error, averaged over the files (seeds), of the one block of each file whose
    keys match sel (missing eval_randomize reads as off). Each block must be the crossed held-out design on the
    episode setups of the archived static rung (the identical-episode fingerprint)."""
    acc = collections.defaultdict(list)
    for f in files:
        _guard(f)
        d = json.load(open(f))
        bs = [r for r in d["results"] if all(r.get(k, "off" if k == "eval_randomize" else None) == v
                                             for k, v in sel.items())]
        if len(bs) != 1:
            raise SystemExit(f"{rel(f)}: {len(bs)} blocks match {sel}")
        eps = bs[0]["episodes"]
        if len(eps) != 60 or {e["patient"] for e in eps} != set(TEST_PATIENTS) or _fingerprint(eps) != _static_fp():
            raise SystemExit(f"{rel(f)}: not the crossed held-out design on the archived episode setups")
        pp = collections.defaultdict(list)
        for e in eps:
            pp[e["patient"]].append(float(e["final_pos"]))
        for p_, v in pp.items():
            acc[p_].append(float(np.mean(v)))
    return {p_: float(np.mean(v)) for p_, v in acc.items()}


def _tci(d):
    """Mean, 95% paired t CI (df n - 1), two-sided t p and the patients sharing the sign of a per-patient dict."""
    import posthoc_v2 as PH
    P = sorted(d)
    x = np.array([d[p_] for p_ in P])
    ts = PH.t_stats(x)
    m = float(x.mean())
    return {"mean": m, "ci": [float(v) for v in ts["ci_t"]], "p_t": float(ts["p_t"]),
            "k": int((np.sign(x) == np.sign(m)).sum()), "n": len(P)}


def _ci_s(e, d=2):
    return f"{sg(e['mean'], d)} {ci_txt(e['ci'], d)}, {e['k']}/{e['n']}"


def _mean(d):
    return float(np.mean(list(d.values())))


# the section of the plan that fixed each check's rule, for the src comment of tab:checks-summary (the codes stay
# there and in the supplementary twin tables/checks.tex; Supplementary S8, tab:codes, maps every code)
ITEM_PLAIN = {_T("rule src q2r"): _T("rule src q2r plain"), _T("rule src q13"): _T("rule src q13 plain"),
              "Q10": _T("rule src plan plain"), "Q11": _T("rule src plan plain"), "Q2R": _T("rule src plan plain"),
              _T("rule src a1 gate q14"): _T("rule src a1 q14 plain")}
# the pattern of the plan codes, which print only in Appendix C and the supplement
PLAN_CODE_RE = re.compile(r"\b(?:[FNQWIG]\d+[a-z]?(?:-[A-Z]+)?|D\.\d+|CTA-[A-Z]|MASK-P|IMN-T|LPIPS-R|"
                          r"Q\d+[A-Z]?(?:-[A-Z]+)?|A\d{1,2}|C\d{1,2}|R\d|"
                          r"LM-[DE]|SKIN-[DE]|AMOS-[PS0-9]|CMB-[DEG]|[DBG]-[PS]|Q8-[PS]|Q9-A|arm [A-HX]|[FN]\d\d)\b")


def _no_plan_codes(name, lines, allow=()):
    """A body table prints no plan code outside its src comments. allow: codes
    that this table may print (Table 5: the family codes that Appendix B defines at first use, T5_ALLOWED_CODES)."""
    for ln in lines:
        vis = re.split(r"(?<!\\)%", ln, maxsplit=1)[0]
        vis = re.sub(r"\\(?:ref|label|cite|input)\{[^}]*\}", "", vis)
        for a_ in allow:
            vis = re.sub(r"\b" + re.escape(a_) + r"\b", "", vis)
        m = PLAN_CODE_RE.search(vis)
        if m:
            raise SystemExit(f"{name}: plan code {m.group(0)!r} would print: {vis[:160]}")


def _and(xs):
    """'a', 'a and b', 'a, b and c'."""
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def checks_summary(R, out):
    """tab:checks-summary (body): what each check guards against, its outcome (absolute first) and whether it meets
    its rule; the rules print in tab:checks (S8.2)."""
    log("\n== tab:checks-summary (checks_summary.tex): exploratory v2.1 except the archived leakage gate ==")
    if not PLUG_REC:
        raise SystemExit("checks_summary() reads the Table 5 record: run pluggability() first")
    V21 = os.path.join(ROOT, "outputs", "v2_1")
    v = (lambda x: f"\\verifyfill{{{x}}}") if PLUG_PROVISIONAL else (lambda x: x)
    S = PROV if PLUG_PROVISIONAL else ""
    Q = _queue()
    cm = R["table4"]["cell_means_mm"]
    rows = []   # (group, check, guards, rule, outcome, met, plan item, src)
    # The plan-conformant analyses of the kept v2.1 questions (the analysis folder) are the
    # source of the cells of the checks they cover; those cells print plain, without the provisional mark. The job
    # outputs read below for the same checks only cross-check them.
    APD = os.path.join(V21, "analysis_plan")

    def _ap(name):
        d_ = json.load(open(os.path.join(APD, name)))
        if _status_of(d_) != "exploratory, v2.1":
            raise SystemExit(f"outputs/v2_1/analysis_plan/{name}: unexpected analysis_status")
        return d_

    def _apc(c_):
        """An analysis_plan contrast block in the {mean, ci, k, n} form of _ci_s."""
        return {"mean": float(c_["mean"]), "ci": [float(x) for x in c_["ci_t"]], "k": int(c_["sign_k"]),
                "n": int(c_.get("sign_n", c_["n"]))}

    APR = {"q2r": _ap("q2r_cohort_v2.json"), "q8": _q8_all(), "q10": _ap("q10_label_controls_v2.json"),
           "q11": _ap("q11_q13_v3.json"), "e": _ap("section_e_v2.json")}
    APS = "outputs/v2_1/analysis_plan/"
    VER = set()   # checks whose rule outcome comes from the decision records: no provisional mark on "Rule met"

    # Guards ===================================================================================================
    gates_all = sorted(n for n in Q if n.startswith("v21:checks:gate:"))
    blind_g = [n for n in gates_all if _blinded(n)]          # SAMUS and LPIPS gates: never opened in this build
    adm, refused, pend, cuda = [], [], [], 0
    for n in (x for x in gates_all if x not in blind_g):
        p_, ok = Q[n]
        if os.path.exists(p_[:-5] + ".failed.json"):
            refused.append(n)
        elif ok:
            j = _plug_json(p_)
            if j.get("admitted") is not True:
                raise SystemExit(f"{rel(p_)}: gate output without admitted = true")
            adm.append(n)
            cuda += int(j.get("cuda_moves_total", 0))
        else:
            pend.append(n.split(":", 3)[3])
    n_g = len(gates_all) - len(blind_g)
    # one gate count from one source (the queue), the blinding stated once; amendment 2 records the blinded gates
    # (SAMUS, LPIPS) as admitted (the development build reads that from the amendment, not from a job output)
    bl_names = sorted({nm for n in blind_g for k_, nm in (("samus", "SAMUS"), ("lpips", "LPIPS"), ("imnet", "ImageNet"),
                                                          ("ctanat", "CT-anatomy")) if k_ in n})
    a2_rec = (None if PI.skip_dev("the source text on the admitted gates (printed only while those results were blinded)")
              else _T("a2 admitted") in open(AMEND2_MD).read())
    out_g = (f"{len(gates_all)} gates; {v(f'{len(adm)} of {n_g}')}" + (" read in this build" if blind_g else "")
             + f" admitted, {v(len(refused))} refused, {v(cuda)} CUDA moves{S}"
             + (f"; pending: {', '.join(pend)}" if pend else "")
             + (f"; the {' and '.join(bl_names)} gates not read in this build (blinded)"
                + (", recorded as admitted in amendment~2" if a2_rec and bl_names == ["LPIPS", "SAMUS"] else "")
                if blind_g else ""))
    rows.append(("Guards", "Admission gate",
                 "Adapters that read privileged state, need CUDA or differ across devices",
                 "Pose-free outputs bit-identical without privileged inputs; privileged adapters detected; no CUDA; "
                 "CPU and MPS agree",
                 out_g, "yes" if adm and not refused and not pend else ("no" if refused else "pending"),
                 _T("rule src a1 gate"),
                 f"jobs v21:checks:gate:* in outputs/v2_1/jobs_final_r3.json: outputs/v2_1/checks/gate/<name>.json "
                 f"admitted and cuda_moves_total ({len(adm)} admitted, {len(refused)} refused, pending {pend}; not read: "
                 f"{', '.join(n.split(':', 3)[3] for n in blind_g)}); {len(gates_all)} gate jobs in the queue; "
                 f"{AMEND2_TXT}, the record of the admitted gates "
                 f"({'not checked' if a2_rec is None else 'found' if a2_rec else 'NOT found'})"))
    lg = json.load(open(os.path.join(V2, "leakage_gate.json")))
    lc = json.load(open(os.path.join(V21, "cohort", "leakage_cohort.json")))
    if _status_of(lc) != "exploratory, v2.1":
        raise SystemExit("outputs/v2_1/cohort/leakage_cohort.json: unexpected analysis_status")
    lk = APR["q2r"]["cohort_acceptance"]["leakage_gate"]
    if lk["n_findings"] != len(lc["findings"]) or lk["passed"] is not lc["passed"]:
        raise SystemExit("q2r_cohort_v2.json cohort_acceptance.leakage_gate differs from leakage_cohort.json")
    VER.add("Leakage gate")
    lr_ = _leak_rerun()
    ok_lg = not lg["findings"] and not lc["findings"] and lc["passed"]
    rows.append(("Guards", "Leakage gate",
                 "A test patient in any training set",
                 "No held-out or cohort case in any transitive training set",
                 f"No finding in {lg['n_records']} archived records; cohort: {lk['n_findings']} in "
                 f"{v(lc['n_sidecars'])}{S}" + (lr_[0] if lr_ else ""),
                 ("yes, after one rerun each" if lr_ else "yes") if ok_lg else "no", _T("rule src q2r"),
                 f"outputs/v2/leakage_gate.json n_records ({lg['n_records']}), findings ({len(lg['findings'])}), "
                 f"results_seat_mode ({lg['results_seat_mode']}); outputs/v2_1/cohort/leakage_cohort.json findings "
                 f"({len(lc['findings'])}), n_sidecars ({lc['n_sidecars']}), passed ({lc['passed']}); cohort findings: "
                 f"{APS}q2r_cohort_v2.json cohort_acceptance.leakage_gate (n_findings {lk['n_findings']}, passed "
                 f"{lk['passed']})" + (f"; reruns: {lr_[1]}" if lr_ else "")))
    aud = sorted(n for n in Q if n.startswith("v21:checks:audit:"))
    a_done = [n for n in aud if Q[n][1]]
    a_ref = [n for n in aud if glob.glob(Q[n][0][:-5] + ".failed*.json")]
    for n in a_done:
        j_ = _plug_json(Q[n][0])
        if j_.get("passed") is not True or j_.get("refused"):
            a_ref.append(n)
    AUD_NAME = {"texture": "texture", "seg": "mask source", "ctrl": "controller", "reward": "reward",
                "renderer": "renderer", "plugcohort": "plug cohort"}
    a_ord = sorted(a_done, key=lambda n: list(AUD_NAME).index(n.split(":")[-1]))
    out_a = (f"Refused in {len(a_ref)} of {len(aud)} slots" if a_ref else
             (f"{v(f'{len(a_done)} of {len(aud)}')} passed ({', '.join(AUD_NAME[n.split(':')[-1]] for n in a_ord)}), "
              f"none refused{S}") if len(a_done) == len(aud) else
             f"{len(a_done)} of {len(aud)} audits done" + (
                 f": {_and([n.split(':')[-1] for n in a_done])} slot{'s' if len(a_done) > 1 else ''}, "
                 "none refused" if a_done else ""))
    rows.append(("Guards", "Identical-episode audit",
                 "Rows scored on other episodes than their reference",
                 "Every episode setup equals the archived static rung's, to $10^{-6}$; a mismatch blocks the slot",
                 out_a, "pending" if len(a_done) < len(aud) else ("no" if a_ref else "yes"), _T("rule src a1 audit"),
                 f"jobs v21:checks:audit:* in outputs/v2_1/jobs_final_r3.json ({len(a_done)} of {len(aud)} outputs, "
                 f"{len(a_ref)} refused)"))
    t = json.load(open(os.path.join(CHECKS_ROOT, "tspeckle", "summary.json")))
    ch = [(r_["row"], r_["final_pos"], r_) for r_ in t["rows"]]
    big = max(ch, key=lambda x: abs(x[1]["mean_change"]))
    lo_, hi_ = min(x[1]["mean_change"] for x in ch), max(x[1]["mean_change"] for x in ch)
    sel_b = {"weights": "zero_residual", "contact": "on"}
    e_ind, e_sh = (_mean(_ppm([_out_path(big[2][k])], sel_b)) for k in ("independent", "shared"))
    if abs((e_ind - e_sh) - big[1]["mean_change"]) > 1e-6:
        raise SystemExit("tspeckle: the largest row's files do not reproduce its mean_change")
    n_cav = sum(abs(x[1]["mean_change"]) > 1.0 for x in ch)
    k_, n_ = (int(x) for x in big[1]["sign_count"].split("/"))
    big_ci = _ci_s({"mean": big[1]["mean_change"], "ci": big[1]["t_ci95"], "k": k_, "n": n_})
    t8k = ("tspk_base_ct", "tspk_base_learned", "tspk_R2_ct", "tspk_R2_learned")   # the order of summary.json rows
    t8 = [APR["q8"]["q9"]["independent_target_speckle"][k]["contrast"] for k in t8k]
    if len(t8) != len(ch) or any(abs(c_["mean"] - x[1]["mean_change"]) > 1e-9 for c_, x in zip(t8, ch)):
        raise SystemExit("tspeckle: summary.json rows differ from q8_gap_cause_v2.json q9.independent_target_speckle")
    ib = max(range(len(t8)), key=lambda i_: abs(t8[i_]["mean"]))
    if ch[ib][0] != big[0]:
        raise SystemExit("tspeckle: the largest row differs between summary.json and q8_gap_cause_v2.json")
    big_ci, lo_, hi_ = _ci_s(_apc(t8[ib])), min(c_["mean"] for c_ in t8), max(c_["mean"] for c_ in t8)
    n_cav = sum(abs(c_["mean"]) > 1.0 for c_ in t8)
    VER.add("Independent target speckle")
    rows.append(("Guards", "Independent target speckle",
                 "Are appearance scores insensitive to shared speckle?",
                 "Change with an independent target speckle field; a point change above 1~mm adds a caveat",
                 f"Largest: learned-mask servo base, physics, seed~0, {v(f'{e_ind:.2f}')} against "
                 f"{v(f'{e_sh:.2f}')}~mm{S}, {big_ci}; {len(ch)} conditions: {sg(lo_)} to {sg(hi_)}~mm",
                 "yes" if n_cav == 0 else "no", _T("rule src a1 speckle"),
                 "outputs/v2_1/checks/tspeckle/summary.json rows[*].final_pos.mean_change ("
                 + ", ".join(f"{x[1]['mean_change']:+.4f}" for x in ch) + f"); largest row {big[0]!r}: t_ci95 "
                 f"{big[1]['t_ci95']}, sign_count {big[1]['sign_count']}; its independent and shared files "
                 f"({rel(_out_path(big[2]['independent']))}, {rel(_out_path(big[2]['shared']))}) contact on, "
                 f"zero_residual: {e_ind:.4f} and {e_sh:.4f} mm; contrasts: {APS}q8_gap_cause_all_conditions.json "
                 "(Q9 asserted equal to q8_gap_cause_v2.json) "
                 f"q9.independent_target_speckle.{{{', '.join(t8k)}}}.contrast (mean, ci_t, sign_k)"))
    dn, tot = _q_count(Q, "v21:checks:plugcohort:")
    pc = _plug_cohort()
    rows.append(("Guards", "Plugged methods on the cohort",
                 "Do plugged-method orderings recur on the cohort?",
                 "Descriptive: Kendall $\\tau_b$, kept pairwise signs, MDD at 6, 20 and 26 patients",
                 (f"{dn} of {tot} jobs done" if pc is None else
                  "Orderings kept: " + ", ".join(PC_NAME[k_] + " " + v(f"{e['kept']} of {e['pairs']}")
                                                 for k_, e in pc["slots"].items())
                  + f"; Kendall $\\tau_b$ {v(pc['tau_lo'])} to {v(pc['tau_hi'])}{S}"),
                 "pending" if pc is None else "descriptive", _T("rule src a1 cohort"),
                 f"jobs v21:checks:plugcohort:* in outputs/v2_1/jobs_final_r3.json ({dn} of {tot} outputs)"
                 + ("" if pc is None else "; " + pc["src"])))

    # Controls =================================================================================================
    vd = _plug_json(os.path.join(CHECKS_ROOT, "sb3pos", "verdict.json"))
    seeds_ = [_plug_json(_out_path(x["path"])) for x in vd["seeds"]]
    lm = [float(x["learner_mm"]) for x in vd["seeds"]]
    om = {round(float(x["oracle_mm"]), 9) for x in vd["seeds"]}
    sm = {round(float(x["static_mm"]), 9) for x in vd["seeds"]}
    mg = {float(x["pass_rule"]["margin_mm"]) for x in seeds_}
    st_ = {int(x["budget"]["env_steps"]) for x in seeds_}
    if len(om) != 1 or len(sm) != 1 or len(mg) != 1 or len(st_) != 1:
        raise SystemExit("sb3pos: the three seeds do not share oracle, static, margin and budget")
    om, sm, mg, st_ = om.pop(), sm.pop(), mg.pop(), st_.pop()
    rows.append(("Controls", "Positive control, SB3 learner",
                 "Can the RL training loop learn a correction?",
                 f"PPO within {mg:.0f}~mm of the pose oracle in all {len(lm)} seeds at {st_:,} steps; privileged easy "
                 "variant, TRAIN patients",
                 f"{v(f'{min(lm):.2f}')} to {v(f'{max(lm):.2f}')}~mm in {v(sum(x['seed_passes'] for x in vd['seeds']))} "
                 f"of {len(lm)} seeds; bound {v(f'{om + mg:.2f}')}, pose oracle {v(f'{om:.2f}')}, static start "
                 f"{v(f'{sm:.2f}')}~mm{S}",
                 "yes" if vd["verdict"] == "PASS" else "no", _T("rule src a1 sb3"),
                 f"outputs/v2_1/checks/sb3pos/verdict.json verdict ({vd['verdict']}), seeds[*].learner_mm "
                 f"({', '.join(f'{x:.4f}' for x in lm)}), oracle_mm ({om:.4f}), static_mm ({sm:.4f}); "
                 f"ppo_poseobs_s{{0,1,2}}.json pass_rule.margin_mm ({mg}) and budget.env_steps ({st_})"))
    rl = json.load(open(os.path.join(V21, "rlpos", "rlpos_summary.json")))
    prim = [c for c in rl["cells"] if c["deficit"] == "bias05"]
    if len(prim) != 3 or any(c["primary_metric"] != "final_dlat_mm" for c in prim):
        raise SystemExit("rlpos: expected three lateral-deficit cells on final_dlat_mm")
    dl = {round(c["metrics"]["final_dlat_mm"]["deficit_rule"]["mean"], 9) for c in prim}
    fr = [c["metrics"]["final_dlat_mm"]["recovered"]["fraction"] for c in prim]
    if len(dl) != 1:
        raise SystemExit("rlpos: the three variants do not share the planted deficit")
    dl = dl.pop()
    q13 = APR["q11"]["Q13"]
    lat = q13["deficits"]["lateral"]
    q13v = ("shipped", "explore", "reward")
    d13 = {round(lat["variants"][k]["precondition"]["contrast"]["mean"], 9) for k in q13v}
    frac13 = [lat["variants"][k]["recovered_fraction"]["estimate"] for k in q13v]
    k13 = [lat["family_result"]["members"][k]["sign_k"] for k in q13v]
    if len(d13) != 1 or abs(d13.copy().pop() - dl) > 1e-6 or sorted(round(x, 6) for x in frac13) != sorted(round(x, 6) for x in fr):
        raise SystemExit("rlpos: rlpos_summary.json differs from q11_q13_v3.json Q13")
    if len(set(k13)) != 1:
        raise SystemExit("Q13: the settings differ in their positive-patient counts; rewrite the REINFORCE row")
    dl, fr = d13.pop(), frac13
    if q13["decision"]["lateral"]["verdict_key"] != "none_passes":
        raise SystemExit("Q13: the lateral verdict is not none_passes; rewrite the REINFORCE row")
    VER.add("Positive control, REINFORCE")
    rows.append(("Controls", "Positive control, REINFORCE",
                 "Can REINFORCE remove a planted deficit?",
                 "Half of a planted deficit removed: one-sided paired $t$, Holm over 3 settings, 5 of 6, every seed",
                 f"Planted lateral deficit {dl:.2f}~mm; {min(fr):.2f} to {max(fr):.2f} of it removed, positive in "
                 f"{k13[0]} of 6 patients for every setting; the planted bias did not meet the reward-visibility rule",
                 "inconclusive", _T("rule src q13"),
                 f"outputs/v2_1/rlpos/rlpos_summary.json overall.key ({rl['overall']['key']}), cells[deficit=bias05]"
                 f".metrics.final_dlat_mm.deficit_rule.mean ({dl:.4f}) and .recovered.fraction "
                 f"({', '.join(f'{x:.4f}' for x in fr)}); {APS}q11_q13_v3.json Q13.deficits.lateral.variants.<setting>"
                 f".precondition.contrast.mean ({dl:.4f}) and .recovered_fraction.estimate, family_result.members.<setting>"
                 f".sign_k ({k13}), Q13.decision.lateral.verdict_key (none_passes, Holm p 1.000 each); reward-visibility "
                 "rule: Q13.bias_selection_pilot.grid.0.5.final_S_change (-0.008, fails <= -0.03" + _T("bias pilot note")
                 + ")"))
    lcd = os.path.join(V21, "label_controls")
    lsum = json.load(open(os.path.join(lcd, "summary.json")))
    so_f = [os.path.join(lcd, "closed_loop", f"ibvs_base_sonospade_matched_s{s_}.json") for s_ in (0, 1, 2)]
    pa_f = [os.path.join(lcd, "closed_loop", f"ibvs_base_label_painted_lit_matched_s{s_}.json") for s_ in (0, 1, 2)]
    ph_f = [os.path.join(V2, "table4", "ladder", f"ibvs_base_physics_matched_s{s_}.json") for s_ in (0, 1, 2)]
    base = {"weights": "zero_residual", "contact": "on"}
    so_n, so_p = _ppm(so_f, dict(base, control="none")), _ppm(so_f, dict(base, control="permute"))
    perm = _tci({p_: so_p[p_] - so_n[p_] for p_ in so_n})
    pa_n, ph = _ppm(pa_f, dict(base, control="none")), _ppm(ph_f, base)
    pnt = _tci({p_: pa_n[p_] - ph[p_] for p_ in ph})
    r1 = lsum["rules"]["R1_reproduced_by_label_painting_lit"]
    e_pa, e_ph, e_so, e_sop = _mean(pa_n), _mean(ph), cm["matched/sonospade/base"], _mean(so_p)
    if abs(e_pa - r1["painted"]) > 1e-4 or abs(e_ph - r1["physics_v2"]) > 1e-4 or abs(e_so - r1["sonospade"]) > 1e-4:
        raise SystemExit("label_controls: summary.json R1 values differ from the closed-loop files")
    frac = (e_ph - e_pa) / (e_ph - e_so)
    q10 = APR["q10"]
    pm10 = q10["results"]["painting"]["patient_means"]
    pnt10 = _apc(q10["results"]["painting"]["family_Q10_P"]["members"]["P_minus_Ph"])
    frac10 = q10["results"]["painting"]["fraction_f"]
    perm10 = _apc(q10["results"]["closed_loop"]["sonospade"]["family_Q10_CL"]["members"]["permute_total_error_change"])
    if (abs(pm10["P"] - e_pa) > 1e-4 or abs(pm10["Ph"] - e_ph) > 1e-4 or abs(pm10["So"] - e_so) > 1e-4
            or abs(pnt10["mean"] - pnt["mean"]) > 1e-4 or abs(perm10["mean"] - perm["mean"]) > 1e-4
            or abs(frac10["estimate"] - frac) > 1e-4):
        raise SystemExit("label_controls: closed-loop files differ from q10_label_controls_v2.json")
    if set(q10["decisions"]["Q10_L_keys"].values()) != {"holds"} or q10["decisions"]["Q10_P_P_minus_Ph"] != "holds":
        raise SystemExit("Q10: a render or painting decision changed; rewrite the label-channel row")
    VER.add("Label-channel controls")
    rows.append(("Controls", "Label-channel controls",
                 "Does the SonoSPADE lead come from label access rather than texture?",
                 "Holm-adjusted paired $t$ in three families; label-painted stage tested for equivalence within 1~mm",
                 f"Label-painted stage {pm10['P']:.2f}~mm against physics {pm10['Ph']:.2f} and SonoSPADE "
                 f"{pm10['So']:.2f}~mm: {_ci_s(pnt10)} from physics, {frac10['estimate']:.2f} [{frac10['ci'][0]:.2f}, "
                 f"{frac10['ci'][1]:.2f}] of the lead; equivalence with SonoSPADE not shown. Rendered masks follow offset "
                 f"and permuted labels, 6/6 each. Permuted organ identities {pm10['So'] + perm10['mean']:.2f}~mm, "
                 f"{_ci_s(perm10)}",
                 "inconclusive", "Q10",
                 "outputs/v2_1/label_controls/closed_loop/ibvs_base_label_painted_lit_matched_s{0,1,2}.json [control="
                 f"none, contact on, zero_residual] ({e_pa:.4f}) minus outputs/v2/table4/ladder/ibvs_base_physics_"
                 f"matched_s{{0,1,2}}.json ({e_ph:.4f}): {pnt['mean']:+.4f}, t p {pnt['p_t']:.4g}; fraction of the lead "
                 f"over RESULTS.json table4.cell_means_mm[matched/sonospade/base] ({e_so:.4f}), equal to summary.json "
                 "rules.R1_reproduced_by_label_painting_lit; ibvs_base_sonospade_matched_s{0,1,2}.json [control=permute] "
                 f"({e_sop:.4f}) minus [control=none]: {perm['mean']:+.4f}, t p {perm['p_t']:.4g}; cells: {APS}"
                 "q10_label_controls_v2.json results.painting.patient_means, family_Q10_P.members.P_minus_Ph, fraction_f "
                 "(estimate, patient bootstrap ci), results.closed_loop.sonospade.family_Q10_CL.members."
                 "permute_total_error_change, decisions.Q10_L_keys (holds) and decisions.painting_class (unresolved: "
                 "equivalence not shown, results.painting.tost_P_minus_So)"))
    q11 = json.load(open(os.path.join(V21, "texture", "q11_texture_budget.json")))
    qc = q11["contrasts"]["primary_final_pos_contact_on"]
    if [c["b"] for c in qc] != ["cut", "cyclegan", "physics"] or any(c["a"] != "sonospade" for c in qc):
        raise SystemExit("q11: unexpected contrasts")
    q = [_tci(c["per_patient_diff"]) for c in qc]
    ar = {k: q11["arms"][k]["final_pos"]["mean"] for k in ("sonospade", "cut", "cyclegan", "physics")}
    q11 = APR["q11"]["Q11"]
    g11 = {k: q11["per_stage"][j]["grand_mean_mm"] for k, j in
           (("sonospade", "per_tissue_t"), ("cut", "cut_bm"), ("cyclegan", "cyclegan_bm"), ("physics", "physics"))}
    m11k = (PI.key("sonospade_minus_cut_bm"), PI.key("sonospade_minus_cyclegan_bm"), "per_tissue_minus_physics")
    q = [_apc(q11["Q11-P"]["members"][k]) for k in m11k]
    if any(abs(g11[k] - ar[k]) > 1e-3 for k in ar):
        raise SystemExit("q11: q11_texture_budget.json arm means differ from q11_q13_v3.json Q11.per_stage")
    a_so, a_cu, a_cy, a_ph = (f"{g11[k]:.2f}" for k in ("sonospade", "cut", "cyclegan", "physics"))
    VER.add("Texture-budget test")
    rows.append(("Controls", "Texture-budget test",
                 "Does the SonoSPADE lead persist with equal budgets and three texture seeds?",
                 "Paired $t$, Holm over 3 contrasts, 5 of 6 sharing the sign",
                 f"SonoSPADE, three texture seeds, {a_so}~mm against budget-matched CUT {a_cu}, CycleGAN {a_cy} and "
                 f"physics {a_ph}~mm: "
                 + "; ".join(_ci_s(e) for e in q) + "~mm",
                 "yes" if all(q11["decision"]["keys"][k] == "holds" for k in m11k) else "no", "Q11",
                 "outputs/v2_1/texture/q11_texture_budget.json arms.{sonospade,cut,cyclegan,physics}.final_pos.mean ("
                 + ", ".join(f"{ar[k]:.4f}" for k in ("sonospade", "cut", "cyclegan", "physics")) + "); "
                 f"cross-check only; cells: {APS}q11_q13_v3.json Q11.per_stage.<stage>.grand_mean_mm ("
                 + ", ".join(f"{g11[k]:.4f}" for k in ("sonospade", "cut", "cyclegan", "physics")) + ") and Q11.Q11-P.members."
                 "{" + ", ".join(m11k) + "} (" + ", ".join(f"{e['mean']:+.4f}" for e in q) + "; decision.keys "
                 + ", ".join(q11["decision"]["keys"][k] for k in m11k)
                 + f"; Holm p {q11['decision']['holm_p'][m11k[0]]:.3f})"))

    # Repeatability ============================================================================================
    rr = _rerun_identity()
    n_same, n_all, mpp = rr["same"], rr["n"], rr["max_pp"]
    er = APR["e"]["rerun"]
    if (er["n_episodes_all_fields_identical"], er["n_episodes"]) != (n_same, n_all) or abs(er["max_abs_patient_diff_mm"] - mpp) > 1e-9:
        raise SystemExit("rerun: the episode comparison differs from section_e_v2.json rerun")
    n_same, n_all, mpp = er["n_episodes_all_fields_identical"], er["n_episodes"], er["max_abs_patient_diff_mm"]
    VER.add("Clean-tag rerun")
    rows.append(("Repeatability", "Clean-tag rerun",
                 "Drift when the archive is recomputed from the tag",
                 "Within 0.01~mm per episode for deterministic rungs, 0.10~mm per patient for neural rows",
                 _T("rerun outcome").format(n_same=n_same, n_all=n_all, mpp=mpp),
                 "yes" if APR["e"]["decisions"]["rerun"]["key"] == "within" else "no", _T("rule src rerun"),
                 rr["src"] + f"; cells: {APS}section_e_v2.json rerun.n_episodes_all_fields_identical ({n_same}), "
                 f".n_episodes ({n_all}), .max_abs_patient_diff_mm ({mpp}), decisions.rerun.key (within)"))
    rg_ = json.load(open(os.path.join(V21, "regnull", "Q12_registration_nulls.json")))["repeatability_rerun_vs_archived"]
    n_id, n_cmp, n_f = rg_["n_identical_all_fields"], rg_["n_compared"], len(rg_["fields"])
    eg = APR["e"]["registration_rerun"]
    if (eg["n_all_fields_identical"], eg["n_compared"]) != (n_id, n_cmp):
        raise SystemExit("registration rerun: Q12_registration_nulls.json differs from section_e_v2.json")
    n_id, n_cmp = eg["n_all_fields_identical"], eg["n_compared"]
    VER.add("Registration rerun")
    rows.append(("Repeatability", "Registration rerun",
                 "Drift of the real-frame runs",
                 "Every logged field of every episode compared",
                 f"{n_id:,} of {n_cmp:,} episodes identical in every logged field",
                 "yes" if n_id == n_cmp else "no", _T("rule src rerun"),
                 "outputs/v2_1/regnull/Q12_registration_nulls.json repeatability_rerun_vs_archived.n_identical_all_fields "
                 f"({n_id}), .n_compared ({n_cmp}), .fields ({n_f}); cells: {APS}section_e_v2.json registration_rerun"
                 f".n_all_fields_identical ({n_id}), .n_compared ({n_cmp}), .fields_that_differ "
                 f"({eg['fields_that_differ'] or 'none'})"))
    par, tol = [], set()
    for n in (x for x in gates_all if x.startswith("v21:checks:gate:ctrl") and x not in blind_g):
        p_, ok = Q[n]
        if not ok:
            continue
        dp_ = _plug_json(p_).get("checks", {}).get("device_parity", {})
        tol.add(float(dp_["tolerances"]["action_cpu_mps_max_abs"]))
        for r_ in (dp_.get("results") or []):   # continuous actions; a discrete action (DQN) must match exactly
            if r_.get("max_abs_action_diff") is not None and not r_.get("discrete"):
                par.append(float(r_["max_abs_action_diff"]))
    dq, tq = _q_count(Q, "v21:cohort:cpu:")
    q14 = _q14()
    if not par or len(tol) != 1:
        raise SystemExit("gate device parity: no controller action differences or more than one tolerance")
    mx, tl = max(par), tol.pop()
    e_ = int(np.floor(np.log10(mx))) if mx > 0 else 0
    mx_txt = f"${mx / 10 ** e_:.1f} \\times 10^{{{e_}}}$" if mx > 0 else "$0$"
    tl_e = int(round(np.log10(tl)))
    q2 = _q14_ii()
    met_i = ("pending" if q14 is None else "yes" if (q14["phases"]["final"]["within"]
                                                      and q14["phases"]["imit"]["within"]) else "no")
    ii_txt = ii_src = ""
    if q2 is not None:
        S2 = _S(q2["file"], q2["d"])
        ii_txt = (f"; (ii) deterministic rungs {_plan_stats().fmt_num(q2['det_max'], 2, False)}~mm per episode, policy "
                  f"{q14['ii_pp_max']:.4f}~mm per patient, learned-mask servo base up to "
                  f"{S2.plain('largest_patient_mm_neural')}~mm per patient (tolerance "
                  f"{S2.plain('tolerances.neural_patient_mm')}~mm)")
        ii_src = ("; (ii): " + " | ".join(S2.e) + f" | run:max(checks[deterministic].largest_difference) = "
                  f"{json.dumps(q2['det_max'])} => {_plan_stats().fmt_num(q2['det_max'], 2, False)}; policy per patient: "
                  "outputs/v2_1/cpu_repro/agreement.json pairs[eval_device ...].cells[*].per_patient_abs_diff_max "
                  f"({q14['ii_pp_max']:.4g})")
    rows.append(("Repeatability", "CPU against MPS",
                 "Results that hold on MPS only",
                 f"Gate actions within $10^{{{tl_e}}}$; CPU-only retrain at most 1.0~mm from the MPS seed mean",
                 f"Actions differ by at most {v(mx_txt)}{S}; " + (
                     f"retrain: {dq} of {tq} jobs done" if q14 is None else
                     ("(i) " if q2 is not None else "") + "CPU retrain " + v(f"{q14['phases']['final']['X']:.2f}")
                     + " against the MPS seed mean "
                     f"{q14['phases']['final']['Y']:.2f}~mm, and " + v(f"{q14['phases']['imit']['X']:.2f}")
                     + f" against {q14['phases']['imit']['Y']:.2f}~mm after imitation{S}") + ii_txt,
                 met_i if q2 is None else f"(i) {met_i}; (ii) no",
                 _T("rule src a1 gate q14"),
                 f"outputs/v2_1/checks/gate/ctrl_*.json (non-blinded queue jobs) checks.device_parity.results[*]"
                 f".max_abs_action_diff of the continuous components (max {mx:.3g} over {len(par)}) and "
                 f"checks.device_parity.tolerances.action_cpu_mps_max_abs ({tl:g}); jobs "
                 f"v21:cohort:cpu:* ({dq} of {tq} outputs); Q14 rule: Q14 (i)"
                 + ("" if q14 is None else
                    f"; final {q14['phases']['final']['X']:.4f} against {q14['phases']['final']['Y']:.4f} "
                    f"({q14['phases']['final']['diff']:+.4f}, k {q14['phases']['final']['k']}/6), after imitation "
                    f"{q14['phases']['imit']['X']:.4f} against {q14['phases']['imit']['Y']:.4f} "
                    f"({q14['phases']['imit']['diff']:+.4f}, k {q14['phases']['imit']['k']}/6), both within +/- "
                    f"{Q14_BAND_MM} mm; no wall-time ordering printed (the archived MPS trainers shared the laptop); "
                    + q14["src"]) + ii_src))
    pr_cell = _plug_cell({s_: os.path.join(PLUG_ROOT, "reward", "train", "paper", f"sonospade_s{s_}.json")
                          for s_ in (0, 1, 2)}, "final", True)
    if len(pr_cell) != 3:
        raise SystemExit("retrain drift: the retrained S needs all three seeds")
    e_pr, arch = _cell_mean(pr_cell), cm["ctmask/sonospade/final"]
    ct_f = [cm[f"ctmask/{s_}/final"] for s_ in STAGES]
    spread = max(ct_f) - min(ct_f)
    rows.append(("Repeatability", "Retrain drift",
                 "A retrained reference read as the archived one",
                 "None; reported beside the reward slot",
                 f"Retrained $S$ {v(f'{e_pr:.2f}')}~mm, {v(f'{e_pr - arch:+.2f}')}~mm from the archived {arch:.2f}~mm"
                 + (f", above the {spread:.2f}~mm stage spread under CT masks" if abs(e_pr - arch) > spread else "")
                 + S, "descriptive", _T("rule src a1 audit"),
                 "outputs/v2_1/plug/reward/train/paper/sonospade_s{0,1,2}.json final.episodes[*].final_pos "
                 f"({e_pr:.4f}) minus outputs/v2/RESULTS.json table4.cell_means_mm[ctmask/sonospade/final] ({arch:.4f}); "
                 f"CT-mask stage spread max minus min of table4.cell_means_mm[ctmask/<stage>/final] ({spread:.4f})"))
    bn = []
    for arm in ("ctmask", "matched"):
        for st in STAGES:
            fe = [os.path.join(V21, "s21_bneval", f"{arm}_{st}_s{s_}.json") for s_ in (0, 1, 2)]
            ft = [os.path.join(V21, "s21_rerun", "contact", f"{arm}_{st}_s{s_}.json") for s_ in (0, 1, 2)]
            a_, b_ = _ppm(fe, {"weights": "final", "contact": "on"}), _ppm(ft, {"weights": "final", "contact": "on"})
            bn.append((f"{arm}/{st}", _tci({p_: a_[p_] - b_[p_] for p_ in a_})))
    fe = [os.path.join(V21, "s21_bneval", f"dr_on_sonospade_s{s_}.json") for s_ in (0, 1, 2)]
    ft = [os.path.join(V21, "s21_rerun", "dr_eval", f"dr_on_sonospade_s{s_}.json") for s_ in (0, 1, 2)]
    sel = {"weights": "final", "contact": "on", "eval_randomize": "off"}
    a_, b_ = _ppm(fe, sel), _ppm(ft, sel)
    bn.append(("dr_on/sonospade", _tci({p_: a_[p_] - b_[p_] for p_ in a_})))
    be = APR["e"]["batchnorm_eval"]
    tiv = [a_ for a_ in be["arms"] if a_["set"].startswith("Table IV policy row")]
    for k_, e_ in bn:   # the final-weight arms computed here must equal the analysis
        m_ = [a_ for a_ in tiv if a_["arm"] == k_ and a_["phase"] == "final"]
        if len(m_) != 1 or abs(m_[0]["contrast_eval_minus_train"]["mean"] - e_["mean"]) > 1e-6:
            raise SystemExit(f"BatchNorm: {k_} final differs from section_e_v2.json batchnorm_eval")
    bmin = min(a_["contrast_eval_minus_train"]["mean"] for a_ in tiv)
    bmax = max(a_["contrast_eval_minus_train"]["mean"] for a_ in tiv)
    bab = be["largest_abs_arm_mean_table_iv_settings"]
    chg = APR["e"]["decisions"]["batchnorm"]["plan_rule_outcomes_changed"]
    if not (any(c_.startswith(_T("rl clause key")) for c_ in chg)
            and any(c_.startswith("Q2 CT-mask stage sentence") for c_ in chg)):
        raise SystemExit("BatchNorm: the changed plan-rule outcomes differ from the printed ones; rewrite this row")
    VER.add("BatchNorm in eval mode")
    rows.append(("Repeatability", "BatchNorm in eval mode",
                 "Are the results unchanged with eval-mode BatchNorm?",
                 f"Eval minus train mode per arm; above {spread:.2f}~mm, the stage spread under CT masks, both modes "
                 "are reported",
                 f"{sg(bmin)} to {sg(bmax)}~mm over {len(tiv)} policies; the outcomes of the 0.5~mm "
                 "bound on imitation and REINFORCE and of the CT-mask stage-pair test change",
                 "no" if abs(bab["mean_mm"]) > spread else "yes", _T("rule src rerun"),
                 "outputs/v2_1/s21_bneval/{ctmask,matched}_<stage>_s{0,1,2}.json and dr_on_sonospade minus "
                 "outputs/v2_1/s21_rerun/{contact,dr_eval}/ same files, final weights, contact on, DR off, seeds "
                 "averaged per patient: " + ", ".join(f"{k} {e['mean']:+.3f}" for k, e in bn)
                 + f"; threshold{_T('bn threshold src')}, here {spread:.4f} from RESULTS.json; cells: {APS}section_e_v2.json "
                   f"batchnorm_eval.arms (set 'Table IV policy row', {len(tiv)} rows: after imitation and final weights), "
                   f"largest_abs_arm_mean_table_iv_settings ({bab['arm']} {bab['phase']}, {bab['mean_mm']:+.3f}) and "
                   f"decisions.batchnorm.plan_rule_outcomes_changed ({'; '.join(chg)})"))
    h36 = _ppm([os.path.join(V21, "s21_h36", f"ibvs_base_physics_ct_s{s_}_n36.json") for s_ in (0, 1, 2)],
               {"weights": "zero_residual", "contact": "on"})
    h12 = _ppm([os.path.join(V2, "table4", "ladder", f"ibvs_base_physics_ct_s{s_}.json") for s_ in (0, 1, 2)],
               {"weights": "zero_residual", "contact": "on"})
    hz = _tci({p_: h36[p_] - h12[p_] for p_ in h36})
    o36 = _ppm([os.path.join(V21, "s21_h36", "pose_oracle_physics_ct_n36.json")],
               {"weights": "zero_residual", "contact": "on"})
    e36, e12, eo = _mean(h36), _mean(h12), _mean(o36)
    if abs(e12 - cm["ctmask/physics/base"]) > 1e-9:
        raise SystemExit("36-step check: the 12-step CT-mask base differs from RESULTS.json")
    h36e = APR["e"]["horizon_36"]
    hz36 = _apc(h36e["ctmask_base"]["result"])
    if (abs(h36e["ctmask_base"]["mean_36_mm"] - e36) > 1e-6 or abs(h36e["ctmask_base"]["mean_12_mm"] - e12) > 1e-6
            or abs(hz36["mean"] - hz["mean"]) > 1e-6 or abs(h36e["oracle"]["mean_36_mm"] - eo) > 1e-6):
        raise SystemExit("36-step check: the files differ from section_e_v2.json horizon_36")
    hz, e36, e12, eo = hz36, h36e["ctmask_base"]["mean_36_mm"], h36e["ctmask_base"]["mean_12_mm"], h36e["oracle"]["mean_36_mm"]
    hold = bool(APR["e"]["decisions"]["horizon_36"]["lowers_rule_met"])
    VER.add("36-step horizon")
    rows.append(("Repeatability", "36-step horizon",
                 "Is the servo's error a horizon limit?",
                 "Horizon effect if the CT-mask servo base's 36-step change is below 0 (paired $t$, 5 of 6) and at "
                 "most $-1$~mm",
                 f"Servo base {e36:.2f}~mm at 36 steps against {e12:.2f}~mm at 12, {_ci_s(hz)}; pose oracle "
                 f"{eo:.2f}~mm", "yes" if hold else "no", _T("rule src rerun"),
                 _T("h36 src") + f" ({e36:.4f} minus {e12:.4f}: "
                 f"{hz['mean']:+.4f}); pose_oracle_physics_ct_n36.json ({eo:.4f}); cells: {APS}section_e_v2.json "
                 "horizon_36.ctmask_base (mean_36_mm, mean_12_mm, result), horizon_36.oracle.mean_36_mm and "
                 "decisions.horizon_36.lowers_rule_met"))

    # Replication ==============================================================================================
    co = json.load(open(os.path.join(V21, "cohort", "analysis_q2r.json")))
    if _status_of(co) != "exploratory, v2.1":
        raise SystemExit("analysis_q2r.json: unexpected analysis_status")
    c1, c2 = co["primary_cohort"]["C1_physics_perception_gap"], co["primary_cohort"]["C2_sonospade_minus_physics_learned_base"]
    o2 = co["reference_original_six"]["C2_sonospade_minus_physics_learned_base"]
    pg = R["table4"]["perception_gap"]["physics"]
    if abs(co["reference_original_six"]["C1_physics_perception_gap"]["mean"] - pg["mean"]) > 1e-3:
        raise SystemExit("analysis_q2r.json original-six physics gap differs from RESULTS.json table4.perception_gap")
    cr = list(csv.DictReader(open(_out_path(co["inputs"]["cohort_csv"]["path"]))))
    md5 = hashlib.md5(open(_out_path(co["inputs"]["cohort_csv"]["path"]), "rb").read()).hexdigest()
    if md5 != co["inputs"]["cohort_csv"]["md5"]:
        raise SystemExit("cohort episodes_long.csv md5 differs from the one analysis_q2r.json read")
    flt = dict(group="contact", rung="policy", stage="physics", phase="final", contact="on", eval_randomize="off")
    a_ct, _, s_ct = C.per_patient(cr, arm="ctmask", **flt)
    a_lm, _, s_lm = C.per_patient(cr, arm="matched", **flt)
    if any(abs((a_lm[p_] - a_ct[p_]) - c1["per_patient"][p_]) > 1e-3 for p_ in c1["per_patient"]) or \
            len(a_ct) != c1["n"]:
        raise SystemExit("cohort: the policy arms of episodes_long.csv do not reproduce analysis_q2r.json C1")
    co_txt = lambda c: f"{sg(c['mean'])} {ci_txt(c['ci95_t'])}, {c['sign_consistency']}"   # noqa: E731
    q2 = APR["q2r"]
    q2m = q2["q2r_p"]["family"]["members"]
    q2c1, q2c2 = _apc(q2m["C1"]), _apc(q2m["C2"])
    lm2 = q2["ordering"]["cells"]["7"]
    if lm2["name"] != "learned final, physics":
        raise SystemExit("q2r_cohort_v2.json ordering cell 7 is not the learned-mask physics final")
    lm2, o2c = lm2["cohort"], q2["reference_original_six"]["C2"]
    ct2 = lm2 - q2c1["mean"]
    if abs(ct2 - _mean(a_ct)) > 1e-3 or abs(lm2 - _mean(a_lm)) > 1e-3 or abs(q2c1["mean"] - c1["mean"]) > 1e-6:
        raise SystemExit("cohort: episodes_long.csv arm means differ from q2r_cohort_v2.json")
    dq2 = q2["decisions"]["Q2R-P"]
    met2 = "yes" if all(dq2[k]["key"] == "holds" for k in ("C1", "C2", "C3")) else "no"
    VER.add("20-CT cohort")
    rows.append(("Replication", "20-CT cohort",
                 "Do the main contrasts recur beyond the six held-out patients?",
                 "Exact sign-flip test, Holm over 3; RL change within 1~mm by an intersection-union test",
                 f"Physics servo: CT mask {ct2:.2f}, learned mask {lm2:.2f}~mm; gap {_ci_s(q2c1)}, against "
                 f"{sg(pg['mean'])}, {pg['sign']}/{pg['n']}, on the six. SonoSPADE minus physics, learned-mask servo "
                 f"base: {_ci_s(q2c2)}, against {sg(o2c['mean'])}, {o2c['sign_k']}/{o2c['n']}. RL change within 1~mm in "
                 f"all {len(q2['q2r_p']['C3_intersection_union']['arms'])} configurations; the ordering "
                 f"{q2['ordering']['class']}", met2, "Q2R",
                 f"outputs/v2_1/cohort/analysis_q2r.json primary_cohort.C1_physics_perception_gap and "
                 f"C2_sonospade_minus_physics_learned_base (mean, ci95_t, sign_consistency; p_holm {c1['p_holm']:.3g}, "
                 f"{c2['p_holm']:.3g}); arm means from {rel(_out_path(co['inputs']['cohort_csv']['path']))} (md5 "
                 f"{md5[:8]}, the file the analysis read) group contact, rung policy, stage physics, final, contact on, "
                 f"seeds {s_ct}: ctmask {_mean(a_ct):.4f}, matched {_mean(a_lm):.4f}, per-patient differences equal to C1 "
                 f"within 1e-3; original six: RESULTS.json table4.perception_gap.physics ({pg['mean']:.4f}, "
                 f"{pg['sign']}/{pg['n']}) and analysis_q2r.json reference_original_six.C2 ({o2['mean']:.4f}, "
                 f"{o2['sign_consistency']}); cells: {APS}q2r_cohort_v2.json ordering.cells.7.cohort ({lm2:.4f}) minus "
                 f"q2r_p.family.members.C1.mean for the CT mask ({ct2:.4f}), q2r_p.family.members.C1 and C2 (mean, ci_t, "
                 f"sign_k), reference_original_six.C2, decisions.Q2R-P (C1, C2, C3 holds; Holm p "
                 f"{dq2['C1']['p_holm']:.2g}), q2r_p.C3_intersection_union (p {q2['q2r_p']['C3_intersection_union']['p']:.2g}) "
                 f"and ordering.class"))

    # Cross-reference ==========================================================================================
    rd = next(SR for SR in PLUG_REC if SR["key"] == "renderer")
    lo_r = next(r_ for r_ in rd["rows"] if r_["id"] == "lotus_orig")
    si_r = next(r_ for r_ in rd["rows"] if r_["id"] == "simus")

    def _cost(r_):
        c_ = r_["cost"] or ""
        return v(c_[:-len(PROV)]) + S if PROV and c_.endswith(PROV) else c_   # PROV is empty
    lo_e, si_e = ("closed-loop row pending" if r_["err"] is None else v(f"{r_['err']:.2f}") + "~mm"
                  for r_ in (lo_r, si_r))
    rows.append(("Cross-reference", "Established simulator code",
                 "A slot that takes only the authors' renderer",
                 "Descriptive: original LOTUS code and SIMUS behind the renderer interface",
                 # the benchmark ran under load (bench.json note and load_before), so its times are labeled as upper
                 # bounds here
                 f"Original LOTUS code {_cost(lo_r)} per CPU frame in a renderer benchmark run under load, an upper "
                 f"bound, {lo_e}; SIMUS {_cost(si_r)} per CPU frame in the same benchmark, "
                 f"{si_e}, secondary (Table~\\ref{{tab:pluggability}})", "descriptive",
                 _T("rule src a1 slots"),
                 f"the renderer rows of Table 5 (this run's record): lotus_orig {lo_r['src']}; simus {si_r['src']}"))

    # the checks of amendments 3 to 5 enter their groups; the criterion column is not printed (Supplementary Table
    # S8.2, tables/checks.tex, keeps every criterion) so the float stays on one page
    new_ = _cs_rows_a345()
    for r_ in new_:
        VER.add(r_[1])
        at = max(i_ for i_, x in enumerate(rows) if x[0] == r_[0]) + 1
        rows.insert(at, r_)
    # these rows are computed and asserted above and logged; they do not print
    for r_ in [x for x in rows if x[1] in CS_MOVED]:
        log(f"[{_T('not printed tag')}] {r_[1]}: {r_[4]} | {r_[5]}")
    rows = [x for x in rows if x[1] not in CS_MOVED]
    # the "Guards" group dissolves: its two remaining rows join Controls and Replication; "Repeatability" is retitled
    rows = [((CS_REGROUP.get(x[1]) or CS_GROUP_PRINT.get(x[0], x[0])),) + tuple(x[1:]) for x in rows]
    rows.sort(key=lambda x: (CS_GROUP_ORDER.index(x[0]), CS_ROW_FIRST.get(x[1], 1)))   # stable within a group
    fam_ = any(r_[5].endswith(" differ") for r_ in rows)
    cap = ("Controls, sensitivity analyses and replication: the question each answers and its outcome, absolute values "
           "first (exploratory). Supplementary Table~S4.2 gives each criterion. "
           # the estimand sentence, which every caption carries
           "Every contrast is conditional on the trained artifacts (texture networks, segmenters, bases and policies). "
           "Seeds index artifacts; the patient or CT is the sampling unit. Intervals: 95\\% paired $t$ with the "
           "patients sharing the sign. Criterion met: yes or no, "
           + ("pending a plan-conformant analysis, " if any(r_[5] == "pending" for r_ in rows) else "")
           + "inconclusive, " + ("descriptive, or, for a family of changes, how many differ." if fam_ else
                                 "or descriptive."))
    L = [GEN.format(src="outputs/v2_1/analysis_plan/ (the plan-conformant analyses, for the checks they cover), "
                        "outputs/v2_1/checks/, outputs/v2_1/{rlpos,label_controls,texture,regnull,s21_rerun,s21_bneval,"
                        "s21_h36,cohort}/, outputs/v2/leakage_gate.json, outputs/v2/RESULTS.json and the Table 5 record "
                        "(each row's src)"),
         "\\begin{table*}[tp]",
         "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}",
         "\\caption{" + cap + "} % src: " + _T("cs caption src"),
         "\\label{tab:checks-summary}",
         "\\centering",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{0.13\\textwidth} "
         ">{\\raggedright\\arraybackslash}p{0.18\\textwidth} "
         ">{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}p{0.085\\textwidth}@{}}",
         "\\toprule",
         "Analysis & Question & Outcome & Criterion met \\\\",
         "\\midrule"]
    grp = None
    for g, a, b, c, d, met, item, src in rows:
        if g != grp:
            if grp is not None:
                L.append("\\addlinespace")
            L.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
            grp = g
        met_ = met + (S if met in ("yes", "no") and a not in VER else "")   # v2.1 outcomes not yet verified carry the mark
        # the plan section goes to the src comment with the code
        item_p = ITEM_PLAIN.get(item, item)
        if PLAN_CODE_RE.search(item_p):
            raise SystemExit(f"tab:checks-summary: a plan code would print under {a!r}: {item_p!r} (add it to "
                             "ITEM_PLAIN)")
        L.append(f"{a} & {b} & {d} & {met_} \\\\[1pt] % src: "
                 + f"plan item '{item}' (plan section: {item_p}); "
                 + (f"rule (Supplementary Table S4.2): {re.sub(r'[$~{}]', '', c)}; " if c else "") + src)
        log(f"{a}: {d} | {met}")
    L += ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    _no_plan_codes("checks_summary.tex", L)
    log(f"tab:checks-summary: {len(rows)} rows in {len(dict.fromkeys(r_[0] for r_ in rows))} groups")
    return write(out, "checks_summary.tex", "\n".join(L))


# =================================================================================== amendments 3 to 5
# The tables of amendment 3 (arm A, Table [T]; the
# earlier learned-mask rows, Supplementary Table [S]; arm B; the phantom metric table), amendment 4 (Tables AMOS-1 and
# AMOS-2, its CT-anatomy context table [Tab] and the AMOS results), amendment 5 ([Tab] and the rule
# input) and the supplementary detail tables of S11, from the committed analysis outputs in
# outputs/v2_1/analysis_plan/ and the check records, written by the committed analysis scripts; this build only
# prints. Every number carries one src entry
#   <file>:<key path> = <unrounded JSON value> => <printed text>
# (entries of one line joined by " | "; <file> without a slash lies in outputs/v2_1/analysis_plan/, "plan:amend<n>" is
# the amendment's Appendix block, "run:" a value of this run's own record, "note:" a source without a key), and
# a development test resolves each key and formats each value again with plan_stats.fmt_num.
# Nothing is written outside the paper tree (outputs/v2_1 is shared with the v2.1 checkout), or DIR in public mode.
AP_DIR = os.path.join(ROOT, "outputs", "v2_1", "analysis_plan")
PLAN_DIR = PP.PLAN_DIR if PP else None          # the folder of the plan documents (development repository only)
OUT_ALLOWED = (PAPER,) if PAPER else ()
# tables outside the article are written to a folder of the development repository (main() --process-out)
PROCESS_OUT = os.path.join(PAPER, *PP.OUTSIDE_TABLES_REL) if PAPER else None
STATUS_V21 = "exploratory, v2.1"
STATUS_A4 = _T("status a4")
STATUS_A5 = _T("status a5")
STATUS_A6 = _T("status a6")
PLAN_V21_SHA12 = "d62fdb75aff4"
# file -> (provenance.analysis_status, amendment whose sha256 the file must cite)
A345_STATUS = {"amend3_arm_a.json": (PLUG_STATUS, 3), "amend3_earlier.json": (PLUG_STATUS, 3),
               "amend3_arm_b.json": (STATUS_V21, 3), "amend3_arm_c.json": (STATUS_V21, 3),
               "amos_q2r.json": (STATUS_A4, 4), "amend5_combination.json": (STATUS_A5, 5),
               "amend5_armh.json": (STATUS_A5, 5), "amend5_amos.json": (STATUS_A5, 5),
               **{f"amend6_{i}.json": (STATUS_A6, 6) for i in ("g2", "p9", "l10", "c30", "r13", "look", "sp", "td")}}
# the check records each read flag needs (amendment 3; amendment 4 and its Appendix blinding.read_after;
# amendment 5)
LIFT_RECORDS = {3: ("outputs/v2_1/amend3/checks/arm_a.json", "outputs/v2_1/amend3/checks/arm_b.json",
                    "outputs/v2_1/amend3/checks/arm_b_post.json", "outputs/v2_1/amend3/checks/arm_c.json"),
                4: ("outputs/v2_1/amos/checks/amos_checks.json", "outputs/v2_1/amos/checks/read_order.json"),
                5: ("outputs/v2_1/amend5/checks/amend5_checks.json",),
                6: ("outputs/v2_1/amend6/checks/amend6_checks.json",)}
# the gate of --read-amend4 on the TotalSegmentator items record, a file of the development repository's plan tree
# (plan_private): skipped with a notice in public mode
LIFT_DEV_OUTPUTS = {4: (PP.A4_ITEMS_REL,) if PP else ()}
LIFT_DEV_GATES = {4: "the existence of a development record of the TotalSegmentator label items (the gate of one "
                      "read option)"}
LIFT_OUTPUTS = {3: ("outputs/v2_1/analysis_plan/amend3_arm_a.json",),
                4: ("outputs/v2_1/analysis_plan/amos_q2r.json",
                    *LIFT_DEV_OUTPUTS[4],
                    "outputs/v2_1/plug/ctanat/fast/summary.json"),
                5: ("outputs/v2_1/analysis_plan/amend5_combination.json",),
                6: tuple(f"outputs/v2_1/analysis_plan/amend6_{i}.json"
                         for i in ("g2", "p9", "l10", "c30", "r13", "look", "sp", "td"))}
# amendment 6: its check record must exist, cite the plan and be in the committed ledger; its passed field is not
# required, since amend6.py applies the check of each item to that item alone
LIFT_NEEDS_PASSED = {3: True, 4: True, 5: True, 6: False}
PLUG_ROWS_MD5 = "600f73ff4cb238cbfc6d137fc9dce357"   # the Table 5 snapshot that amendment 5's rule outcome refers to
# the same snapshot as the public build writes it (its generated_by field names the public script; keys sorted)
PLUG_ROWS_MD5_PUBLIC = "29d24b2a53556ea711e8ee109faaf3ce"
# Paths no option lifts (the build's hard refusals): the AMOS data, the private real-frame pools and their derivatives (any path
# component starting with one of usbg.provenance.PRIVATE_DIRNAMES, or usbg.provenance.is_private_path), the training
# artifacts (logs, histories, checkpoints) of the ImageNet-pretrained group and SAMUS's real-frame Dice.
HARD_REFUSE = tuple(PP.HARD_REFUSE_DIRS) if PP is not None else ()   # the AMOS data folder (development backend)
IMNET_TOKENS = ("imnet", "imagenet")
TRAIN_SUFFIXES = (".pt", ".pth", ".ckpt", ".log")


def _hard_refusal(p):
    """Why no option lets this build read p, or None. In public mode the words are matched against the path below the
    result root, as written and resolved (script_dirs.guard_forms): the location of ROOT itself never matters."""
    for form in (SD.guard_forms(p) if SD.public() else (str(p),)):
        why = _hard_refusal_form(form)
        if why:
            return why
    return None


def _hard_refusal_form(form):
    s = form.replace(os.sep, "/").lower()
    if is_private_path(form) or any(x.startswith(n) for x in s.split("/") for n in PRIVATE_DIRNAMES):
        return "a private path (never read by this build)"
    for t in HARD_REFUSE:
        if t in s:
            return f"a path with {t!r} (never read by this build)"
    base = os.path.basename(s)
    if any(t in s for t in IMNET_TOKENS) and (base.endswith(TRAIN_SUFFIXES) or "history" in base or "/train/" in s):
        return "a training artifact (log, history or checkpoint) of the ImageNet-pretrained group"
    if "samus" in s and base.startswith("eval_"):
        return "SAMUS real-frame Dice (withheld)"
    return None


def _out_ok(p):
    """Every file this build writes lies under the paper tree, or DIR in public mode (outputs/ is shared with the
    v2.1 checkout)."""
    rp = os.path.realpath(p)
    if not any(rp.startswith(os.path.realpath(a) + os.sep) for a in OUT_ALLOWED):
        raise SystemExit(f"{p}: outside {', '.join(rel(a) for a in OUT_ALLOWED)}; this build writes nothing else")
    return p


def _plan_sha(n):
    """sha256 of amendment n as its digest file records it (plan_inputs.amendment_digest)."""
    return PI.amendment_digest(n, plan_dir=PLAN_DIR)


def _ledger_head():
    """The records ledger as committed at HEAD (development repository only: plan_private.ledger_head, a read-only
    git show; plan_stats.ledger_at_head)."""
    if "ledger" not in _A345:
        _A345["ledger"] = PI.private().ledger_head(ROOT)
    return _A345["ledger"]


def _in_ledger(rid, sha):
    """True when the records ledger committed with the analyses lists sha for rid (a path relative to the result root,
    or the neutral id of a plan-tree record): plan_inputs.ledger_has, with this build's ledger reader. Public mode: a
    result file is checked against the deposit's MANIFEST.tsv instead (the ledger lines name the development files'
    bytes), a plan-tree record by its recorded digest."""
    if SD.public() and rid.startswith(("outputs/", "data/", "segmenter/")):
        import results_world
        PI.manifest_instead("the records ledger lines of the result files")
        return results_world.listed(rid) == sha
    return PI.ledger_has(rid, sha, head=_ledger_head)


def _ledger_name(rid=""):
    """How messages name the authority that _in_ledger consults for rid."""
    if not SD.public():
        return "the records ledger committed at HEAD"
    return ("MANIFEST.tsv of the result root" if rid.startswith(("outputs/", "data/", "segmenter/"))
            else "the recorded records ledger")


def _rec_name(rid):
    """How messages name a plan-tree record: its path in the development repository, its neutral id otherwise."""
    return PP.record_path(rid) if (PP is not None and not SD.public()) else rid


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lift_ok(n):
    """The condition its plan attaches to --read-amend<n>; SystemExit naming what is missing. Check records are read
    under the plans' readable checks/ folders; no output of the amendment is opened here."""
    miss = []
    dev_skip = set(LIFT_DEV_OUTPUTS.get(n, ())) if n in LIFT_DEV_GATES and PI.skip_dev(LIFT_DEV_GATES[n]) else set()
    for r_ in LIFT_OUTPUTS[n]:
        if r_ in dev_skip:
            continue
        p = os.path.join(ROOT, r_)
        if _hard_refusal(p) or not os.path.exists(p):
            miss.append(f"{r_} does not exist")
    led = None
    for r_ in LIFT_RECORDS[n]:
        p = os.path.join(ROOT, r_)
        if "/checks/" not in r_ or _hard_refusal(p):
            raise SystemExit(f"_lift_ok: {r_} is not a check record")
        if not os.path.exists(p):
            miss.append(f"{r_} does not exist")
            continue
        d = json.load(open(p))
        if LIFT_NEEDS_PASSED[n] and d.get("passed") is not True:
            miss.append(f"{r_} has passed = {d.get('passed')!r}")
        if d.get("cites") != _plan_sha(n):
            miss.append(f"{r_} cites {str(d.get('cites'))[:12]}, not amendment {n} ({_plan_sha(n)[:12]})")
        if not _in_ledger(r_, _sha256(p)):
            miss.append(f"{r_} (sha256 {_sha256(p)[:12]}) is not in {_ledger_name(r_)}")
    if miss:
        raise SystemExit(f"--read-amend{n} refused: " + "; ".join(miss))
    log(f"[blinding] --read-amend{n}: " + ", ".join(LIFT_OUTPUTS[n][:1] + LIFT_RECORDS[n])
        + " present; records " + ("passed, " if LIFT_NEEDS_PASSED[n] else "") + "cite the plan and are in the "
        + ("committed ledger" if not SD.public() else "result root's MANIFEST.tsv"))


_A345 = {}


def _a345(name):
    """One amendment 3 to 5 analysis output (cached): blinding guard, status string and the amendment's sha256."""
    if name in _A345:
        return _A345[name]
    p = os.path.join(AP_DIR, name)
    _guard(p)
    d = json.load(open(p))
    st, n = A345_STATUS[name]
    if _status_of(d) != st:
        raise SystemExit(f"{rel(p)}: analysis_status {_status_of(d)!r}, expected {st!r}")
    am = d.get("amendment") if isinstance(d.get("amendment"), dict) else {}
    got = {x for x in (_dig(d, "provenance", "amendments", f"amend{n}", "sha256"), am.get("sha256")) if x}
    if got != {_plan_sha(n)}:
        raise SystemExit(f"{rel(p)}: cites amendment {n} sha256 {sorted(got)}, the recorded digest is {_plan_sha(n)}")
    if _dig(d, "provenance", PI.key("plan_v21"), "sha12") not in (None, PLAN_V21_SHA12):
        raise SystemExit(f"{rel(p)}: v2.1 plan sha12 {_dig(d, 'provenance', PI.key('plan_v21'), 'sha12')}")
    _A345[name] = d
    return d


def _a345_ready(*names):
    """True if every named output may be read in this build (its flags lifted); else False (the table is skipped)."""
    return all(_blinded_a345(os.path.join(AP_DIR, x)) is None for x in names)


# --------------------------------------------------------------------------------- key paths and src entries
_KP_TOKEN = re.compile(r'\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\["([^"]*)"\]')


def _kp_parts(kp):
    parts, i = [], 0
    while i < len(kp):
        m = _KP_TOKEN.match(kp, i)
        if not m or m.end() == i:
            raise SystemExit(f"bad key path {kp!r} at {i}")
        parts.append(m.group(1) if m.group(1) is not None else int(m.group(2)) if m.group(2) is not None
                     else m.group(3))
        i = m.end()
    return parts


def _kp_get(d, kp):
    for k in _kp_parts(kp):
        d = d[k]
    return d


def _q(k):
    """A dict key in a key path: bare if it is an identifier, else quoted."""
    return f".{k}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k)) else f'["{k}"]'


def _p_tex(p):
    """plan_stats.fmt_p (3 decimals, two leading digits below 0.001) with e-notation as a power of ten."""
    s = _plan_stats().fmt_p(p, 3)
    if "e" in s:
        m_, e_ = s.split("e")
        return f"{m_} \\times 10^{{{int(e_)}}}"
    return s


class _S:
    """The src entries of one printed line. Each formatter reads the value at a key path of the current file, returns
    the printed LaTeX and records '<file>:<key path> = <JSON value> => <printed text>'."""

    def __init__(self, f=None, d=None):
        self.f, self.d, self.e = f, d, []

    def at(self, f, d):
        self.f, self.d = f, d
        return self

    def get(self, kp):
        return _kp_get(self.d, kp)

    def rec(self, kp, val, txt, f=None):
        self.e.append(f"{f or self.f}:{kp} = {json.dumps(val)} => {txt}")
        return txt

    def num(self, kp, d=2, signed=True, math=True):
        v = self.get(kp)
        t = self.rec(kp, v, _plan_stats().fmt_num(v, d, signed))
        return f"${t}$" if (math or t.startswith(("+", "-"))) else t

    def plain(self, kp, d=2):
        """An unsigned value (an error, a count of mm), printed without math mode unless negative."""
        return self.num(kp, d, signed=False, math=False)

    def ci(self, kp, d=2, signed=True):
        v = self.get(kp)
        a, b = (self.rec(f"{kp}[{i}]", v[i], _plan_stats().fmt_num(v[i], d, signed)) for i in (0, 1))
        return f"$[{a}, {b}]$"

    def p(self, kp):
        v = self.get(kp)
        return "$" + self.rec(kp, v, _p_tex(v)) + "$"

    def int(self, kp):
        v = self.get(kp)
        if float(v) != int(v):
            raise SystemExit(f"{self.f}:{kp} = {v} is not an integer")
        return self.rec(kp, v, str(int(v)))

    def pct(self, kp, d=1):
        v = self.get(kp)
        t = self.rec(kp, v, _plan_stats().fmt_num(100.0 * float(v), d, False) + "%")
        return t[:-1] + "\\%"

    def word(self, kp, mapping):
        v = self.get(kp)
        if v not in mapping:
            raise SystemExit(f"{self.f}:{kp} = {v!r}: no printed word for it")
        self.rec(kp, v, "word")
        return mapping[v]

    def run(self, name, val, d=2, signed=False):
        """A value of this run's own record (no JSON key): printed with the same formatter."""
        t = _plan_stats().fmt_num(val, d, signed)
        self.e.append(f"run:{name} = {json.dumps(val)} => {t}")
        return f"${t}$" if t.startswith(("+", "-")) else t

    def note(self, text):
        self.e.append("note: " + text.replace(" | ", "; "))
        return self

    def plan(self, n, kp, d=2):
        """A constant of amendment n's Appendix block (plan_inputs.amendment_value), printed as a number."""
        v = PI.amendment_value(n, kp)
        t = _plan_stats().fmt_num(v, d, False)
        self.e.append(f"plan:amend{n}:{kp} = {json.dumps(v)} => {t}")
        return t

    def tail(self):
        return (" % src: " + " | ".join(self.e)) if self.e else ""

    def line(self, visible):
        out = visible + self.tail()
        self.e = []
        return out


def _src_entries(comment):
    """[(file, key path, value, printed)] of a src comment written by _S (note entries skipped)."""
    out = []
    for e in comment.split(" | "):
        e = e.strip()
        if not e or e.startswith("note:"):
            continue
        f_, _, rest = e.partition(":")
        if f_ == "plan" or f_ == "run":
            g_, _, rest = rest.partition(":") if f_ == "plan" else ("", "", rest)
            f_ = f"plan:{g_}" if f_ == "plan" else "run"
        kp, _, rest = rest.partition(" = ")
        val, _, txt = rest.rpartition(" => ")
        out.append((f_, kp, json.loads(val), txt))
    return out


def _table_head(label, caption, colspec, header, star=False, cont=False, pos="tp"):
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[{pos}]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize"]
    if cont:
        L += ["\\ContinuedFloat", "\\caption[]{Continued.}"]
    else:
        L += [caption, f"\\label{{{label}}}"]
    L += ["\\centering", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}",
          "\\toprule", header, "\\midrule"]
    return L


def _table_foot(star=False):
    return ["\\bottomrule", "\\end{tabularx}", "\\end{table*}" if star else "\\end{table}"]


def _mc(n, text, style="l"):
    return f"\\multicolumn{{{n}}}{{@{{}}{style}}}{{{text}}}"


def _mcp(n, text):
    """A full-width paragraph row."""
    return f"\\multicolumn{{{n}}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{0.98\\textwidth}}@{{}}}}{{{text}}}"


def _tex_plain(s):
    """A verbatim analysis string as LaTeX: +/- as \\pm, e-notation as a power of ten, % escaped."""
    s = s.replace("+/-", "$\\pm$").replace("±", "$\\pm$").replace("%", "\\%")
    s = re.sub(r"(\d\.\d)e-0?(\d+)", lambda m: f"${m.group(1)} \\times 10^{{-{m.group(2)}}}$", s)
    s = s.replace("_", "\\_")
    s = re.sub(r"(?<![\w$.])([+-]\d+\.\d+)", r"$\1$", s)
    return s


# --------------------------------------------------------------------------------- amendment 3, arm A ([T])
ARM_A_MEMBERS = [("dagger_res:physics", "B1", "DAgger residual minus learned-mask servo base, physics frames"),
                 ("dagger_res:sonospade", "B2", "DAgger residual minus learned-mask servo base, SonoSPADE frames"),
                 ("bc_res:physics", "B3", "BC residual minus learned-mask servo base, physics frames"),
                 ("bc_res:sonospade", "B4", "BC residual minus learned-mask servo base, SonoSPADE frames"),
                 ("pair:physics", "P1", "DAgger residual minus BC residual, physics frames"),
                 ("pair:sonospade", "P2", "DAgger residual minus BC residual, SonoSPADE frames")]
DIFF_WORDS = {"no_difference": "no detectable difference"}
EQ_WORDS = {"no_difference": "equivalence not shown", "holds": "equivalent within $\\pm 1$~mm"}


def _diff_word(m, kp_key, kp_dir, S, none="no detectable difference"):
    """The decision word of a difference member: 'holds lower' / 'holds higher' / the null word (none)."""
    k = S.get(kp_key)
    S.rec(kp_key, k, "word")
    if k == "holds":
        dr = S.get(kp_dir)
        S.rec(kp_dir, dr, "word")
        return "holds lower" if dr < 0 else "holds higher"
    if k != "no_difference":
        raise SystemExit(f"{S.f}:{kp_key} = {k!r}: no printed word")
    return none


def arm_a_tab(out):
    """tables/arm_a.tex (tab:arm-a, Table [T] of amendment 3; Appendix C.3)."""
    if not _a345_ready("amend3_arm_a.json", "amend3_earlier.json"):
        log("arm_a.tex: amendment 3 blinded in this build; not written")
        return None
    F, FE = "amend3_arm_a.json", "amend3_earlier.json"
    A, E = _a345(F), _a345(FE)
    if A["result"]["status"] != "completed" or A["result"]["families"]["LM-D"]["missing"] or \
            A["result"]["families"]["LM-E"]["missing"]:
        raise SystemExit("amend3_arm_a.json: arm A is not complete; print the incomplete sentence instead")
    S = _S(F, A)
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend3_arm_a.json and amend3_earlier.json (amendment 3, arm A)")]
    cap = S.at(FE, E)
    b_p = cap.plain("result.ranges.physics_matched.base")
    b_s = cap.plain("result.ranges.sonospade_matched.base")
    if E["result"]["rows"][7]["row"] != "pose_oracle" or E["result"]["rows"][7]["condition"] != "physics_matched":
        raise SystemExit("amend3_earlier.json rows[7] is not the pose oracle")
    orc = cap.plain("result.rows[7].mean_mm")
    cap.at(F, A)
    m1a, m1h = cap.plain('result.mdd["LM-D_M1"].alpha_005_mm'), cap.plain('result.mdd["LM-D_M1"].holm_first_step_mm')
    m2a, m2h = cap.plain('result.mdd["LM-D_M2"].alpha_005_mm'), cap.plain('result.mdd["LM-D_M2"].holm_first_step_mm')
    met = cap.get("result.separation_criterion.met")
    cap.rec("result.separation_criterion.met", met, "word")
    nm = cap.int('result.families["LM-D"].m')
    ne = cap.int('result.families["LM-E"].m')
    caption = ("\\caption{Learned-mask controller experiment on the six held-out patients (exploratory, pre-specified "
               f"rule). Families LM-D (difference) and LM-E (equivalence within $\\pm 1$~mm), "
               + (f"$m = {nm}$ each" if nm == ne else f"$m = {nm}$ and $m = {ne}$") + ", Holm within each family. "
               f"A difference holds if its Holm $p$ is below {cap.plan(3, 'alpha')} and at least 5 "
               "of 6 patients share its sign (paired $t$, df 5); the exact sign-flip $p$ is printed and never decides. "
               # (the learners' absolute errors print in their B rows)
               "Absolute errors are the base's mean plus the paired difference. "
               f"The servo base ends at {b_p}~mm with physics frames and {b_s}~mm with SonoSPADE frames, and the pose "
               f"oracle at {orc}~mm. MDD from the observed SDs: {m1a} and {m1h}~mm for the base "
               f"contrasts, {m2a} and {m2h}~mm for the learner pair, at $\\alpha = {cap.plan(3, 'alpha')}$ and at "
               "Holm's first step. "
               "The separation criterion (a physics-frame difference of B1, B3 or P1 holds) is "
               + ("met" if met else "not met") + ". Crossed bootstrap: case by seed, a sensitivity check.}"
               + cap.tail())
    S.e = []   # cap is S: the caption's entries stay on the caption line
    head = ("Contrast & Family & Difference or 90\\% CI (mm) & $k$/6 & Test $p$ & Sign-flip $p$ & Holm $p$ & "
            "Decision & Crossed bootstrap 95\\% CI \\\\")
    L += _table_head("tab:arm-a", caption, "@{}>{\\raggedright\\arraybackslash}X l l c c c c "
                     ">{\\raggedright\\arraybackslash}p{0.13\\textwidth} l@{}", head)
    for mid, code, name in ARM_A_MEMBERS:
        d_ = f'result.families["LM-D"].members["{mid}"]'
        e_ = f'result.families["LM-E"].members["{mid}"]'
        S.at(F, A)
        # a B row (learner minus the learned-mask servo base) prints the learner's absolute final error: the base's
        # mean (amend3_earlier.json result.ranges.<stage>_matched.base, the value the caption prints) plus the paired
        # mean of this member
        lname = name
        if not mid.startswith("pair:"):
            lrn, stg = mid.split(":")
            bkp = f"result.ranges.{stg}_matched.base"
            bval, dval = _kp_get(E, bkp), _kp_get(A, d_ + ".mean")
            S.note(f"learner final error = {FE}:{bkp} {bval:.4f} + {F}:{d_}.mean {dval:+.4f} (sum of two archived "
                   f"means{_T('learner sum note')})")
            lname = (f"{name}; the {'DAgger' if lrn == 'dagger_res' else 'BC'} residual ends at "
                     f"{S.run(f'{code} learner final error, base mean plus paired mean', bval + dval)}~mm")
        k_ = f"{S.int(d_ + '.sign_k')}/{S.int(d_ + '.sign_n')}"
        row = (f"\\quad {code}: {lname} & LM-D & {S.num(d_ + '.mean')} {S.ci(d_ + '.ci_t')} & {k_} & "
               f"{S.p(d_ + '.p_t')} (paired $t$) & {S.p(d_ + '.p_signflip_two_sided')} & {S.p(d_ + '.p_holm')} & "
               f"{_diff_word(None, d_ + '.key', d_ + '.sign_direction', S)} & "
               f"{S.ci(f'result.members[\"{mid}\"].crossed_bootstrap_sensitivity.ci95')} \\\\")
        L.append(S.line(row))
        row = (f"\\quad\\quad Equivalence within $\\pm 1$~mm & LM-E & 90\\% {S.ci(e_ + '.ci90')} & & "
               f"{S.p(e_ + '.p_decision')} (TOST) & & {S.p(e_ + '.p_holm')} & "
               f"{S.word(e_ + '.key', EQ_WORDS)} & \\\\")
        L.append(S.line(row))
    L += _table_foot() + [""]
    return write(out, "arm_a.tex", "\n".join(L))


# --------------------------------------------------------------------------------- amendment 3, earlier rows ([S])
EARLIER_COND = [("physics_matched", "Learned masks, physics frames"),
                ("sonospade_matched", "Learned masks, SonoSPADE frames"),
                ("physics_ct", "CT masks, physics frames")]
# the v2_final and v2_imit rows are v2.1 controller jobs
# (v21:plug:ctrl:v2_final|v2_imit:<condition>:s<k>, outputs/v2_1/jobs_final_r3.json, --algo v2_policy on the archived
# weights), not the archived Table 4 cells, which they need not equal; the labels and the caption say so
EARLIER_REF_NAME = {"static": "Static start", "random": "Random actions", "pose_oracle": "Pose oracle (privileged)",
                    "v2_final": "REINFORCE residual of the primary results, final weights, rerun",
                    "v2_imit": "REINFORCE residual of the primary results, after imitation, rerun",
                    "servo_base": "Servo base alone",
                    "cem": "CEM look-ahead (privileged simulator rollouts, CT masks)"}


def _ctrl_name(rid):
    if rid in EARLIER_REF_NAME:
        return EARLIER_REF_NAME[rid]
    lab = {r["id"]: r["label"] for r in _slot("ctrl")["rows"]}
    if rid not in lab:
        raise SystemExit(f"no plain name for controller row {rid!r}")
    return lab[rid]


def earlier_ctrl_tab(out):
    """tables/earlier_ctrl.tex (tab:earlier-ctrl, Supplementary Table [S] of amendment 3)."""
    if not _a345_ready("amend3_earlier.json"):
        log("earlier_ctrl.tex: amendment 3 blinded in this build; not written")
        return None
    F = "amend3_earlier.json"
    E = _a345(F)
    S = _S(F, E)
    rows = E["result"]["rows"]
    lab = S.get("result.label")
    if lab != _T("earlier label") or any(r["label"] != lab for r in rows):
        raise SystemExit("amend3_earlier.json: the rows do not all carry the label of the earlier runs")
    S.rec("result.label", lab, "word")
    caption = ("\\caption{Earlier learned-mask controller runs, seeds 0 to 2 (exploratory, no pre-specified rule; "
               "run before the learned-mask controller arm was specified). None of these rows enters a decision, and "
               "none is read as a separation "
               "of two methods. Error: final translation error over the 60 held-out episodes, mean over seeds of "
               "patient means. Difference: learner minus the servo base of the same condition, paired by patient, "
               "95\\% $t$ CI, with the patients of 6 sharing its sign. Reference rows print their error only. "
               "The rerun rows re-evaluate the primary REINFORCE weights in the same evaluation as these learners, "
               "so they need not equal the cells of Table~\\ref{tab:ladder}.}"
               + S.tail())
    S.e = []
    head = "Row & Seeds & Error (mm) & Minus servo base (mm), 95\\% $t$ CI, $k$/6 \\\\"
    L = [GEN.format(src=f"outputs/v2_1/analysis_plan/amend3_earlier.json result.rows ({_T('src a3 earlier')})")]
    L += _table_head("tab:earlier-ctrl", caption, "@{}>{\\raggedright\\arraybackslash}X c r l@{}", head)
    seen = 0
    for cond, title in EARLIER_COND:
        if seen:
            L.append("\\addlinespace")
        L.append(_mc(4, f"\\textit{{{title}}}") + " \\\\")
        for i, r in enumerate(rows):
            if r["condition"] != cond:
                continue
            seen += 1
            kp = f"result.rows[{i}]"
            S.rec(kp + ".row", r["row"], "word")
            if r["kind"] == "learner":
                k_ = S.int(kp + ".sign_k")
                diff = f"{S.num(kp + '.difference_mm')} {S.ci(kp + '.ci_t')}, {k_}/6"
                S.note(f"{F}:{kp}.string {r['string']!r} prints k/6")
            else:
                diff = ""
            L.append(S.line(f"\\quad {_ctrl_name(r['row'])} & {S.int(kp + '.n_seeds')} & {S.plain(kp + '.mean_mm')} "
                            f"& {diff} \\\\"))
    if seen != len(rows):
        raise SystemExit(f"amend3_earlier.json: {len(rows)} rows, {seen} printed")
    L += _table_foot() + [""]
    return write(out, "earlier_ctrl.tex", "\n".join(L))


# --------------------------------------------------------------------------------- amendment 3, arm B
ARM_B_RUNGS = [("static", "Static start"), ("scripted", "Scripted sweep"), ("pose_oracle", "Pose oracle (privileged)"),
               ("ct_base", "CT-mask servo base, physics frames, seeds 0 to 2"),
               ("lm_base", "Learned-mask servo base, SonoSPADE frames, seeds 0 to 2")]


def arm_b_tab(out):
    """tables/arm_b.tex (tab:arm-b; Appendix C.8): SKIN-D and SKIN-E per rung, then the plane quantities per rung
    (amendment 3, always printed: the angle between the planes and the target-face offset per rung)."""
    if not _a345_ready("amend3_arm_b.json"):
        log("arm_b.tex: amendment 3 blinded in this build; not written")
        return None
    F = "amend3_arm_b.json"
    B = _a345(F)
    if B["result"]["status"] != "completed" or B["result"]["families"]["SKIN-D"]["missing"] or \
            B["result"]["families"]["SKIN-E"]["missing"] or B["result"].get("post_record_passed") is not True:
        raise SystemExit("amend3_arm_b.json: arm B is not complete or its post record did not pass")
    S = _S(F, B)
    mda, mdh = S.plain('result.mdd["SKIN-D"].alpha_005_mm'), S.plain('result.mdd["SKIN-D"].holm_first_step_mm')
    nm, ne = S.int('result.families["SKIN-D"].m'), S.int('result.families["SKIN-E"].m')
    caption = ("\\caption{Skin-tangent contact experiment (exploratory, pre-specified rule): skin-tangent plane minus "
               "target-anchored plane, same rung and seed, on the six held-out patients. Families SKIN-D (change) and "
               "SKIN-E (equivalence within $\\pm 1$~mm), " + (f"$m = {nm}$ each" if nm == ne else f"$m = {nm}$ and $m = {ne}$")
               + f", Holm within each family; a change holds if its Holm $p$ is below {S.plan(3, 'alpha')} and at least 5 "
               "of 6 patients "
               "share its sign (paired $t$, df 5); the exact sign-flip $p$ is printed and never decides. Both servo bases keep "
               "the calibrations fit with the target-anchored plane. MDD from the observed "
               f"SDs: {mda}~mm at $\\alpha = {S.plan(3, 'alpha')}$ and {mdh}~mm at Holm's first step. Lower block: the "
               "plane quantities "
               "over the same sampled setups, per rung (n episodes: 60 setups, times 3 seeds for the bases).}"
               + S.tail())
    S.e = []
    head = ("Rung & Family & Change or 90\\% CI (mm) & $k$/6 & Test $p$ & Sign-flip $p$ & Holm $p$ & Decision & "
            "Crossed bootstrap 95\\% CI \\\\")
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend3_arm_b.json (amendment 3, arm B)")]
    L += _table_head("tab:arm-b", caption, "@{}>{\\raggedright\\arraybackslash}X l l c c c c "
                     ">{\\raggedright\\arraybackslash}p{0.12\\textwidth} l@{}", head)
    for rid, name in ARM_B_RUNGS:
        d_, e_ = f"result.members.{rid}.skin_d", f"result.members.{rid}.skin_e"
        k_ = f"{S.int(d_ + '.sign_k')}/{S.int(d_ + '.sign_n')}"
        cb = (S.ci(f"result.members.{rid}.crossed_bootstrap_sensitivity.ci95")
              if B["result"]["members"][rid].get("crossed_bootstrap_sensitivity") else "n/a")
        L.append(S.line(f"\\quad {name} & SKIN-D & {S.num(d_ + '.mean')} {S.ci(d_ + '.ci_t')} & {k_} & "
                        f"{S.p(d_ + '.p_t')} (paired $t$) & {S.p(d_ + '.p_signflip_two_sided')} & "
                        f"{S.p(d_ + '.p_holm')} & "
                        f"{_diff_word(None, d_ + '.key', d_ + '.mean', S, none='no detectable change')} & {cb} \\\\"))
        L.append(S.line(f"\\quad\\quad Equivalence within $\\pm 1$~mm & SKIN-E & 90\\% {S.ci(e_ + '.ci90')} & & "
                        f"{S.p(e_ + '.p_decision')} (TOST) & & {S.p(e_ + '.p_holm')} & {S.word(e_ + '.key', EQ_WORDS)} "
                        "& \\\\"))
    L += ["\\addlinespace", _mc(9, "\\textit{Plane quantities per rung: median (range); lift: median (largest)}")
          + " \\\\",
          " & \\multicolumn{3}{l}{Angle to the target beam (deg)} & \\multicolumn{3}{l}{Signed target-face offset (mm)} "
          "& Lift (mm) & Episodes \\\\"]
    for rid, name in ARM_B_RUNGS:
        p_ = f"result.plane.{rid}"
        L.append(S.line(f"\\quad {name} & \\multicolumn{{3}}{{l}}{{{S.plain(p_ + '.angle_to_target_beam_deg.median')} "
                        f"({S.plain(p_ + '.angle_to_target_beam_deg.min')} to "
                        f"{S.plain(p_ + '.angle_to_target_beam_deg.max')})}} & \\multicolumn{{3}}{{l}}{{"
                        f"{S.num(p_ + '.target_face_signed_mm.median')} ({S.num(p_ + '.target_face_signed_mm.min')} to "
                        f"{S.num(p_ + '.target_face_signed_mm.max')})}} & {S.plain(p_ + '.lift_mm.median')} "
                        f"({S.plain(p_ + '.lift_mm.max')}) & {S.int(p_ + '.angle_to_target_beam_deg.n')} \\\\"))
    L += _table_foot() + [""]
    return write(out, "arm_b.tex", "\n".join(L))


# --------------------------------------------------------------------------------- amendment 3, arm C (phantoms)
# Band words from amendment 3 and its Appendix block (bands): G's map for the writers.
PHANTOM_RENDERER = {"physics": "Physics renderer", "lotus_orig": "Original LOTUS code", "simus": "SIMUS"}
BAND_WORDS = {
    "psf_physics_lateral_50_le_10_and_110": "lateral PSF width in lines at the 50~mm wire no larger than at the 10 and "
                                            "110~mm wires (wire phantom W; design check)",
    "psf_axial_mm_factor_1_5": "axial PSF width in mm within a factor of 1.5 across the six wires (wire phantom W)",
    "psf_lotus_lateral_lines_within_10pct": "lateral PSF width in lines within 10\\% of its mean across depth (wire "
                                            "phantom W; design check)",
    "psf_lotus_axial_samples_within_10pct": "axial PSF width in samples within 10\\% of its mean across depth (wire "
                                            "phantom W; design check)",
    "psf_simus_lateral_110_ge_30": "lateral PSF width in mm at the 110~mm wire at least that at the 30~mm wire (wire "
                                   "phantom W)",
    "speckle_snr_physics_2_6_to_4_0": "speckle SNR in phantom H from 2.6 to 4.0",
    "speckle_snr_simus_below_1_6": "speckle SNR in phantom H below 1.6",
    "cnr_over_h_snr_at_least_0_90": "CNR of the anechoic disc in phantom C over the speckle SNR of phantom H, at least "
                                    "0.90",
    "slope_H_db_per_cm:physics": "attenuation slope in phantom H within $\\pm 1.0$~dB/cm of $+0.16$~dB/cm (design "
                                 "check)",
    "slope_H_db_per_cm:simus": "attenuation slope in phantom H within $\\pm 1.0$~dB/cm of 0 (design check)",
    "slope_L_muscle_minus_soft_below_0": "muscle minus soft-tissue attenuation slope in phantom L, below 0 (design "
                                         "check)",
    "shadow_A_to_bottom_at_least_80mm": "shadow below the air slab of phantom A to the image bottom, at least 80~mm",
    "shadow_B_at_least_10mm": "shadow below the 5~mm bone slab of phantom B, at least 10~mm",
}
# the field words of the arm C sentence (metric, phantom and value, for each)
BAND_FIELD = {"cnr_over_h_snr_at_least_0_90": "the CNR of the anechoic disc over the speckle SNR of phantom H",
              "shadow_B_at_least_10mm": "the shadow depth below the slab of phantom B",
              "speckle_snr_physics_2_6_to_4_0": "the speckle SNR of phantom H",
              "speckle_snr_simus_below_1_6": "the speckle SNR of phantom H",
              "slope_H_db_per_cm": "the attenuation slope of phantom H",
              "slope_L_muscle_minus_soft_below_0": "the muscle minus soft-tissue slope of phantom L",
              "shadow_A_to_bottom_at_least_80mm": "the shadow depth below the slab of phantom A"}
# metric of the metric table -> (printed name, unit, band id of each renderer)
PHANTOM_METRICS = [("snr", "Speckle SNR, phantom H", "",
                    {"physics": "speckle_snr_physics_2_6_to_4_0", "simus": "speckle_snr_simus_below_1_6"}),
                   ("cnr", "CNR of the anechoic disc, phantom C", "", {}),
                   ("cnr_over_snr", "CNR over the speckle SNR", "", {r: "cnr_over_h_snr_at_least_0_90"
                                                                    for r in PHANTOM_RENDERER}),
                   ("slope_H", "Attenuation slope, phantom H", "dB/cm", {"physics": "slope_H_db_per_cm",
                                                                         "simus": "slope_H_db_per_cm"}),
                   ("slope_L_soft", "Slope, soft tissue, phantom L", "dB/cm", {}),
                   ("slope_L_muscle", "Slope, muscle, phantom L", "dB/cm", {}),
                   ("slope_L_diff", "Muscle minus soft-tissue slope, phantom L", "dB/cm",
                    {r: "slope_L_muscle_minus_soft_below_0" for r in PHANTOM_RENDERER}),
                   ("shadow_B_mm", "Shadow depth below the bone slab, phantom B", "mm",
                    {"physics": "shadow_B_at_least_10mm", "simus": "shadow_B_at_least_10mm"}),
                   ("shadow_A_mm", "Shadow depth below the air slab, phantom A", "mm",
                    {"physics": "shadow_A_to_bottom_at_least_80mm", "simus": "shadow_A_to_bottom_at_least_80mm"})]
PSF_BANDS = ("psf_physics_lateral_50_le_10_and_110", "psf_axial_mm_factor_1_5", "psf_lotus_lateral_lines_within_10pct",
             "psf_lotus_axial_samples_within_10pct", "psf_simus_lateral_110_ge_30")
LABEL_NAME = {"2": "gallbladder disc", "8": "muscle", "10": "bone", "11": "air (lung)", "12": "soft tissue"}


def _band_words(rid, band):
    return BAND_WORDS.get(f"{band}:{rid}") or BAND_WORDS[band]


def _design_check(rid, band):
    """A band that holds by the renderer's construction (its words say "design check")."""
    return "design check" in _band_words(rid, band)


def _band_words_printed(rid, band):
    """The band words as printed: a design-check band prints only when it is not met (not scorable or
    outside), and then without its "design check" tag."""
    return _band_words(rid, band).replace("; design check)", ")").replace(" (design check)", "")


def _inside(v):
    return "not scorable" if v is None else ("inside" if v else "outside")


def phantoms_tab(out):
    """tables/phantoms.tex (tab:phantoms; Supplementary S1), amendment 3: per metric the seed mean with its SD, the
    band and whether the value lies in it, with the Rayleigh reference and the clipped fractions."""
    if not _a345_ready("amend3_arm_c.json"):
        log("phantoms.tex: amendment 3 blinded in this build; not written")
        return None
    F = "amend3_arm_c.json"
    C_ = _a345(F)
    R_ = C_["result"]
    ids = {b["band"] for r in R_["renderers"].values() for b in r["bands"]}
    if ids - {k.split(":")[0] for k in BAND_WORDS}:
        raise SystemExit(f"amend3_arm_c.json: band ids without words: {sorted(ids - set(BAND_WORDS))}")
    S = _S(F, C_)
    ray = S.plain("result.rayleigh_reference")
    # no amendment label; the rows of the design-check bands (which hold by construction) are logged, not printed;
    # the renderer headers keep their counts over every band
    caption = ("\\caption{Canonical digital phantoms, renderers only (descriptive; no family, no "
               "decision). Each metric: mean (SD) over the 10 speckle seeds on each renderer's native output. Bands "
               "fixed before any phantom was rendered; not scorable: a wire was not measurable in every seed. "
               # the domain of the speckle SNR and of the frames, as scripts/plug/phantoms_v21.py of the v2.1 checkout
               # computes them (amplitude: each display map inverted, the original LOTUS map as returned; speckle SNR
               # over depths 30 to 90 mm and lines 40 to 119, each depth row divided by its mean; grid 160 lines by 512
               # samples)
               "Speckle SNR: mean over SD of the amplitude, after each renderer's display map is inverted (the original "
               "LOTUS map is linear and is used as returned), in phantom H at 30 to 90~mm depth on the central 80 lines, each depth row divided "
               "by its mean, on the 512 by 160 frame before resizing and the sector warp. "
               f"Rayleigh reference line of the speckle SNR, that of a Rayleigh envelope: {ray}. The band counts in the "
               "renderer headers also "
               "include design-check bands that the renderer meets by its construction; those bands are not listed. "
               "The continuation gives the wire widths and the clipped fractions.}"
               # a design-check band that is not scorable or outside prints (it does not hold by construction), so
               # every not-scorable count in a header has its row
               + S.tail())
    S.e = []
    head = "Metric & Mean (SD) & Band fixed before rendering & Outcome \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend3_arm_c.json result (amendment 3, arm C)")]
    L += _table_head("tab:phantoms", caption, "@{}>{\\raggedright\\arraybackslash}p{0.3\\textwidth} l "
                     ">{\\raggedright\\arraybackslash}X l@{}", head)
    for j, (rid, rname) in enumerate(PHANTOM_RENDERER.items()):
        if j:
            L.append("\\addlinespace")
        r_ = f"result.renderers.{rid}"
        L.append(S.line(_mc(4, f"\\textit{{{rname}: meets {S.int(r_ + '.k')} of {S.int(r_ + '.K')} bands, "
                                f"{S.int(r_ + '.u')} not scorable}}") + " \\\\"))
        bi = {b["band"]: i for i, b in enumerate(R_["renderers"][rid]["bands"])}
        used = set()
        for mk, mname, unit, bands in PHANTOM_METRICS:
            m_ = f"result.metric_table.{rid}.{mk}"
            ent = R_["metric_table"][rid][mk]
            val = (f"{S.num(m_ + '.value', 2, signed=False, math=False)}" if "value" in ent else
                   f"{S.num(m_ + '.mean', 2, signed=False, math=False)} ({S.plain(m_ + '.sd')})")
            band = bands.get(rid)
            if band:
                i = bi[band]
                used.add(band)
                b_ = f"{r_}.bands[{i}]"
                S.rec(b_ + ".band", band, "word")
                iv = S.get(b_ + ".inside")
                S.rec(b_ + ".inside", iv, "word")
                bw, oc = _band_words_printed(rid, band), _inside(iv)
                S.note(f"band words: {_T('src a3 bands')} and Appendix arm_c bands (BAND_WORDS[{band}])")
            else:
                bw, oc = "no band", ""
            row_ = f"\\quad {mname}" + (f" ({unit})" if unit else "") + f" & {val} & {bw} & {oc} \\\\"
            if band and _design_check(rid, band) and iv is True:
                log(f"[{_T('phantom tag')}] phantoms {rid}: {S.line(row_)}")
                continue
            L.append(S.line(row_))
        for band in PSF_BANDS:
            if band in bi:
                i = bi[band]
                used.add(band)
                b_ = f"{r_}.bands[{i}]"
                S.rec(b_ + ".band", band, "word")
                iv = S.get(b_ + ".inside")
                S.rec(b_ + ".inside", iv, "word")
                v_ = R_["renderers"][rid]["bands"][i]["value"]
                vtxt = ("" if v_ is None else
                        f"{S.plain(b_ + '.value[0]')} and {S.plain(b_ + '.value[1]')}" if isinstance(v_, list) else
                        S.plain(b_ + ".value"))
                S.note(f"band words: {_T('src a3 bands')} and Appendix arm_c bands (BAND_WORDS[{band}])")
                row_ = (f"\\quad PSF width, wire phantom W & {vtxt} & {_band_words_printed(rid, band)} & "
                        f"{_inside(iv)} \\\\")
                if _design_check(rid, band) and iv is True:
                    log(f"[{_T('phantom tag')}] phantoms {rid}: {S.line(row_)}")
                    continue
                L.append(S.line(row_))
        if used != set(bi):
            raise SystemExit(f"phantoms: bands of {rid} not printed: {sorted(set(bi) - used)}")
    L += _table_foot()
    # continuation: wires and clipped fractions
    head2 = ("Wire depth (mm) & Measurable seeds & Lateral width (lines) & Lateral width (mm) & Axial width (samples) "
             "& Axial width (mm) \\\\")
    L += _table_head("tab:phantoms", "", "@{}>{\\raggedright\\arraybackslash}X c c c c c@{}", head2, cont=True)
    for j, (rid, rname) in enumerate(PHANTOM_RENDERER.items()):
        if j:
            L.append("\\addlinespace")
        L.append(_mc(6, f"\\textit{{{rname}: wire phantom W, FWHM, mean (SD) over 10 seeds}}") + " \\\\")
        for i, w in enumerate(R_["metric_table"][rid]["wires"]):
            w_ = f"result.metric_table.{rid}.wires[{i}]"
            L.append(S.line(f"\\quad {S.num(w_ + '.depth_mm', 0, False, False)} & {S.int(w_ + '.n_measurable')}/10 & "
                            f"{S.plain(w_ + '.lat_lines.mean')} ({S.plain(w_ + '.lat_lines.sd')}) & "
                            f"{S.plain(w_ + '.lat_mm.mean')} ({S.plain(w_ + '.lat_mm.sd')}) & "
                            f"{S.plain(w_ + '.ax_samples.mean')} ({S.plain(w_ + '.ax_samples.sd')}) & "
                            f"{S.plain(w_ + '.ax_mm.mean')} ({S.plain(w_ + '.ax_mm.sd')}) \\\\"))
    L += ["\\addlinespace", _mc(6, "\\textit{Clipped fractions of pixels at 0 and at 1, per phantom region (label)}")
          + " \\\\", "Phantom, region & \\multicolumn{2}{c}{Physics renderer} & \\multicolumn{2}{c}{Original LOTUS code} "
          "& SIMUS \\\\"]
    cf = R_["clip_fractions"]
    regions = []
    for ph in ("H", "L", "C", "W", "B", "A"):
        labs = sorted({lb for rid in PHANTOM_RENDERER for lb in (cf.get(rid, {}).get(ph) or {})}, key=int)
        regions += [(ph, lb) for lb in labs]
    for ph, lb in regions:
        cells = []
        for rid in PHANTOM_RENDERER:
            k_ = f'result.clip_fractions.{rid}.{ph}["{lb}"]'
            if lb in (cf.get(rid, {}).get(ph) or {}):
                cells.append(f"{S.num(k_ + '.at0', 3, False, False)} / {S.num(k_ + '.at1', 3, False, False)}")
            else:
                cells.append("n/a")
        L.append(S.line(f"\\quad {ph}, {LABEL_NAME.get(lb, 'label ' + lb)} & \\multicolumn{{2}}{{c}}{{{cells[0]}}} & "
                        f"\\multicolumn{{2}}{{c}}{{{cells[1]}}} & {cells[2]} \\\\"))
    L += _table_foot() + [""]
    return write(out, "phantoms.tex", "\n".join(L))


# --------------------------------------------------------------------------------- amendment 4 (AMOS)
AMOS_FROZEN = "outputs/v2_1/amos/amos_frozen.json"     # the ledgered frozen manifest (a check record; no image read)
YIELD_COLS = [("screened", "Screened"), ("pass", "Pass"), ("rejected", "Rejected"), ("error", "Error"),
              ("label_gate_rejected", "Label check failed"), ("rejected_at_draw", "Rejected at draw"),
              ("median_liver_windows", "Liver windows, median"), ("zero_liver_windows", "No liver window"),
              ("median_kidney_windows", "Kidney windows, median"), ("zero_kidney_windows", "No kidney window")]
YIELD_DROP = ("error", "rejected_at_draw")   # all zero; not printed
STRUCT_NAME = {"spleen": "Spleen", "kidney_right": "Right kidney", "kidney_left": "Left kidney",
               "gallbladder": "Gallbladder", "esophagus": "Esophagus", "liver": "Liver", "stomach": "Stomach",
               "aorta": "Aorta", "inferior_vena_cava": "Inferior vena cava", "pancreas": "Pancreas",
               "duodenum": "Duodenum", "urinary_bladder": "Urinary bladder",
               "adrenal_gland_right": "Right adrenal gland", "adrenal_gland_left": "Left adrenal gland"}
AMOS_ARM_NAME = {"ctmask/physics": "CT masks, physics", "ctmask/cut": "CT masks, CUT",
                 "ctmask/cyclegan": "CT masks, CycleGAN", "ctmask/sonospade": "CT masks, SonoSPADE",
                 "matched/physics": "learned masks, physics", "matched/cut": "learned masks, CUT",
                 "matched/cyclegan": "learned masks, CycleGAN", "matched/sonospade": "learned masks, SonoSPADE"}
AMOS_CON_NAME = {"C1": "C1: learned minus CT masks, physics, final policy",
                 "C2": "C2: SonoSPADE minus physics, learned-mask servo base",
                 "gap": "Gap: CT-mask servo base minus pose oracle",
                 "bound": "Bound: privileged look-ahead (seed 0) minus CT-mask servo base",
                 **{k: f"RL change, {v}" for k, v in AMOS_ARM_NAME.items()}}


def _amos_frozen():
    if "frozen" not in _A345:
        p = os.path.join(ROOT, AMOS_FROZEN)
        _guard(p)
        d = json.load(open(p))
        if _status_of(d) != STATUS_A4 or d.get("frozen") is not True or d.get("smoke") or d.get("dry_run"):
            raise SystemExit(f"{AMOS_FROZEN}: not the frozen AMOS manifest")
        man = _a345("amos_q2r.json")["manifest"]
        sha_ok = man.get("sha256") == _sha256(p)
        if man.get("path") != AMOS_FROZEN or not sha_ok or not man.get("in_records_ledger"):
            raise SystemExit(f"{AMOS_FROZEN}: sha256 differs from amos_q2r.json manifest, or it is not ledgered")
        _A345["frozen"] = d
    return _A345["frozen"]


def amos_yield_tab(out):
    """tables/amos_yield.tex (tab:amos-yield, Table AMOS-1 of amendment 4; Appendix C.9)."""
    if not _a345_ready("amos_q2r.json"):
        log("amos_yield.tex: amendment 4 blinded in this build; not written")
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    M = _amos_frozen()
    S = _S(F, A)
    ys = A["yield_by_split_and_slice_spacing"]
    if ys != M["yield_by_split_and_slice_spacing"]:
        raise SystemExit("amos_q2r.json yield table differs from the frozen manifest's")
    sl = S.at(AMOS_FROZEN, M).plain("eligibility.pool_rules.max_slice_mm", 1)
    S.at(F, A)
    # no amendment or v2.1 label; the draw and printed-pool counts and the all-zero columns "Error" and "Rejected at
    # draw" are not printed; the label-source sentence stands here, its thresholds from amendment 4
    n_rad = S.get("design_counts.n_rejected_at_draw")
    zero_cols = [c for c in YIELD_DROP if any(int(ys[k][c]) != 0 for k in ys)]
    if n_rad or zero_cols:
        raise SystemExit(f"amos_yield: rejected at the draw ({n_rad}) or a dropped column is not all zero "
                         f"({zero_cols}); print it")
    n_elig = S.int('design_counts.n_E')
    caption = ("\\caption{Yield of the AMOS external cohort (exploratory). "
               f"Of {n_elig} eligible AMOS CTs (slice spacing at most {sl}~mm), "
               f"{S.int('design_counts.n_G')} passed the label check and " + f"{S.int('design_counts.n_W')} the window "
               f"screen. The first {S.int('n_A')} passes in the seeded order form "
               "the cohort" + ("" if S.get("reduced") is False else "; the fallback was applied") + ". Rows: AMOS "
               "split (Tr, training; Va, validation) and native slice spacing. Windows: screened acoustic windows per "
               "CT. Every AMOS structure comes from TotalSegmentator v2 fast-mode predictions; the AMOS manual labels "
               "serve only to accept a case, which needs a liver Dice of at least "
               f"{S.plan(4, 'amos.gate.liver_dice_min')} and a kidney Dice of at least "
               f"{S.plan(4, 'amos.gate.kidney_dice_min')}, with {S.plan(4, 'amos.gate.absent_kidney_ml_max', 0)}~mL as "
               "the absent-kidney volume.}")
    S.rec("reduced", A["reduced"], "word")
    S.note("label source: amendment 4 Appendix amos.label_rule (ts_only), amos.totalsegmentator.mode (fast)"
           + _T("amos label note"))
    caption += S.tail()
    S.e = []
    cols_ = [(c, n_) for c, n_ in YIELD_COLS if c not in YIELD_DROP]
    head = ("Split, slice spacing & " + " & ".join(f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{n_}\\end{{tabular}}"
                                                   for _, n_ in cols_) + " \\\\")
    head = head.replace(", median", ",\\\\median").replace("Label check failed", "Label check\\\\failed").replace(
        "Rejected at draw", "Rejected\\\\at draw").replace("No liver window", "No liver\\\\window").replace(
        "No kidney window", "No kidney\\\\window")
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amos_q2r.json design_counts, n_A and "
                        f"yield_by_split_and_slice_spacing ({_T('src a4 yield')})")]
    L += _table_head("tab:amos-yield", caption, "@{}l " + "c " * len(cols_) + "@{}", head)
    for k in ys:
        y_ = f'yield_by_split_and_slice_spacing["{k}"]'
        cells = [S.num(f"{y_}.{c}", 1, False, False) if c.startswith("median") else S.int(f"{y_}.{c}")
                 for c, _ in cols_]
        L.append(S.line(f"{k.replace(', slice', ', slice')} & " + " & ".join(cells) + " \\\\"))
    L += _table_foot() + [""]
    return write(out, "amos_yield.tex", "\n".join(L))


def amos_labels_tab(out):
    """tables/amos_labels.tex (tab:amos-labels, Table AMOS-2 of amendment 4; Appendix C.9)."""
    if not _a345_ready("amos_q2r.json"):
        log("amos_labels.tex: amendment 4 blinded in this build; not written")
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    S = _S(F, A)
    la = A["label_agreement"]
    if set(la["accepted"]) - set(STRUCT_NAME):
        raise SystemExit(f"amos_q2r.json label_agreement: structures without a name {set(la['accepted']) - set(STRUCT_NAME)}")
    na = {v["n"] for v in la["accepted"].values()}
    nr = {v["n"] for v in la["rejected"].values()}
    if len(na) != 1 or len(nr) != 1:
        raise SystemExit("amos_q2r.json label_agreement: n differs across structures")
    s0 = next(iter(la["accepted"]))
    r0 = next(iter(la["rejected"]))
    caption = ("\\caption{Agreement of the TotalSegmentator v2 labels with the AMOS manual organ labels, Dice per "
               f"structure (descriptive). Accepted: the {S.int(f'label_agreement.accepted.{s0}.n')} cohort "
               f"CTs; rejected: the {S.int(f'label_agreement.rejected.{r0}.n')} screened CTs rejected by the label check or "
               "the window screen. "
               f"Median over the accepted CTs: liver {S.plain('label_agreement.median_liver_accepted')}, kidneys "
               f"{S.plain('label_agreement.median_kidneys_accepted')}.}}" + S.tail())
    S.e = []
    head = ("Structure & \\multicolumn{3}{c}{Accepted: median (min, max)} & \\multicolumn{3}{c}{Rejected: median "
            "(min, max)} \\\\")
    L = [GEN.format(src=f"outputs/v2_1/analysis_plan/amos_q2r.json label_agreement ({_T('src a4 yield')})")]
    L += _table_head("tab:amos-labels", caption, "@{}>{\\raggedright\\arraybackslash}X r l l r l l@{}", head)
    for k in la["accepted"]:
        a_, r_ = f"label_agreement.accepted.{k}", f"label_agreement.rejected.{k}"
        rej = (f"{S.plain(r_ + '.median')} & ({S.plain(r_ + '.min')}, & {S.plain(r_ + '.max')})"
               if k in la["rejected"] else "n/a & &")
        L.append(S.line(f"\\quad {STRUCT_NAME[k]} & {S.plain(a_ + '.median')} & ({S.plain(a_ + '.min')}, & "
                        f"{S.plain(a_ + '.max')}) & {rej} \\\\"))
    L += _table_foot() + [""]
    return write(out, "amos_labels.tex", "\n".join(L))


def amos_ctanat_tab(out):
    """tables/amos_ctanat.tex (tab:amos-ctanat, [Tab] of amendment 4; Appendix C.9): the five CT-anatomy
    rungs of the original six on TotalSegmentator labels (amendment 2, this run's Table 5 record) beside AMOS."""
    if not _a345_ready("amos_q2r.json"):
        log("amos_ctanat.tex: amendment 4 blinded in this build; not written")
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    S = _S(F, A)
    col = A["secondary"]["ct_anatomy_amos_column"]
    six = {r["rid"]: r for r in CTA_STATE.get("rows", [])} if not CTA_STATE.get("blinded") else {}
    if [rid for rid, *_ in CTA_ROWS] != sorted(col):
        raise SystemExit("amos_q2r.json ct_anatomy_amos_column: rows differ from R1 to R5")
    caption = ("\\caption{CT-anatomy context (descriptive): the five CT-anatomy rungs of the original "
               "six on TotalSegmentator labels beside the same rungs on the 30 AMOS CTs. The two columns "
               "share the label model and differ in CT source, patients, $n$, the labels of the calibration fit and "
               "the window source; they are never pooled, and no difference across them is printed. Error: final "
               "translation error, mean over seeds of case means"
               + ("." if six else "; the original six column is not read in this build, so the AMOS column prints "
                                  "alone.") + "}")
    head = ("Rung & \\begin{tabular}[b]{@{}r@{}}Original six,\\\\TotalSegmentator labels (mm)\\end{tabular} & "
            "\\begin{tabular}[b]{@{}l@{}}AMOS (mm),\\\\95\\% $t$ CI\\end{tabular} & AMOS $n$ \\\\")
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amos_q2r.json secondary.ct_anatomy_amos_column and the CT-anatomy "
                        "block of this run's Table 5 record (amendment 2)")]
    L += _table_head("tab:amos-ctanat", caption, "@{}>{\\raggedright\\arraybackslash}X r l c@{}", head)
    for rid, *_ in CTA_ROWS:
        c_ = f"secondary.ct_anatomy_amos_column.{rid}"
        if six and six.get(rid, {}).get("err") is not None:
            v6 = S.run(f"CTA_STATE.rows[{rid}].err", six[rid]["err"])
            S.note(f"original six: {six[rid]['src'][:160]}")
        else:
            v6 = "not read"
        S.rec(c_ + ".label", S.get(c_ + ".label"), "word")
        L.append(S.line(f"\\quad {_cap1(CTA_PLAIN[rid])} & {v6} & {S.plain(c_ + '.mean')} {S.ci(c_ + '.ci_t', 2, False)} "
                        f"& {S.int(c_ + '.n')} \\\\"))
    L += _table_foot() + [""]
    return write(out, "amos_ctanat.tex", "\n".join(L))


def _rep_word(S, kp, six):
    """C1 and C2 of AMOS-P: keyed on the sign of the original six (amendment 4)."""
    k = S.get(kp + ".key")
    S.rec(kp + ".key", k, "word")
    m = S.get(kp + ".mean")
    if k == "holds":
        return "replicates" if (m > 0) == (six > 0) else "reverses"
    return "not replicated"


def amos_results_tab(out):
    """tables/amos_results.tex (tab:amos-results; Appendix C.9): the AMOS decisions with the cohort string of each
    contrast (amendment 4 number format), the comparison with the replication cohort and the relabeling
    control (amendment 4; estimation only)."""
    if not _a345_ready("amos_q2r.json"):
        log("amos_results.tex: amendment 4 blinded in this build; not written")
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    if A["deviations"] or not A["decisions"]["decisions_allowed"] or A["families"]["AMOS-P"]["missing"] or \
            A["families"]["AMOS-S"]["missing"]:
        raise SystemExit("amos_q2r.json: incomplete families or deviations; print the not-run sentence of amendment 4 instead")
    S = _S(F, A)
    st = A["statistics_lines"]
    if [x.split(":")[0] for x in st] != ["C1", "C2", "gap", "bound"]:
        raise SystemExit("amos_q2r.json statistics_lines: unexpected order")
    for fam in ("AMOS-P", "AMOS-S"):
        for m_ in A["families"][fam]["members"].values():
            if m_.get("signflip_method", "monte carlo") != "monte carlo":
                raise SystemExit("amos_q2r.json: a decision member's sign-flip is not the Monte Carlo branch")
    caption = ("\\caption{AMOS external cohort on 30 CTs (exploratory, pre-specified rule; external "
               "replication in simulation only). Top: the decisions. Families AMOS-P ($m = 3$) and AMOS-S ($m = 2$), "
               "two-sided Monte Carlo sign-flip test (1,000,000 sign vectors, seed 0), Holm within each family; the CT "
               "is the unit. C1 and C2 replicate when they hold with the sign of the original six. Middle: AMOS beside "
               "the 20 "
               "replication-cohort CTs and the original six (estimation only; no $p$, no decision; 95\\% $t$ CIs). An "
               "interval that covers 0 does not show that a value is unchanged, and one that excludes 0 is not a "
               "difference that holds. Bottom: the relabeling control on the 20 cohort CTs (estimation only), the "
               "paired change and the AMOS difference side by side, never added.}")
    S.note("Monte Carlo wording: amos_q2r.json statistics_lines (verbatim); signflip_method 'monte carlo' of every "
           "decision member; members' decision_test 'exact sign-flip' is not printed (design 2.1 f)")
    caption += S.tail()
    S.e = []
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amos_q2r.json families, C3_per_arm, ordering, statistics_lines, "
                        f"comparison and arm_x ({_T('src a4 results')})"),
         "\\begin{table}[tp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize", caption, "\\label{tab:amos-results}",
         "\\centering", "\\setlength{\\tabcolsep}{2.5pt}"]
    # block 1: decisions
    L += ["\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l c c c "
          ">{\\raggedright\\arraybackslash}p{0.1\\textwidth} l l r@{}}", "\\toprule",
          "Contrast & Estimate (mm), 95\\% $t$ CI & $k$/$n$ & Sign-flip $p$ & Holm $p$ & Decision & Case bootstrap & "
          "Case-by-seed bootstrap & Six \\\\", "\\midrule"]
    for i, (fam, mid) in enumerate((("AMOS-P", "C1"), ("AMOS-P", "C2"), ("AMOS-S", "gap"), ("AMOS-S", "bound"))):
        m_ = f'families["{fam}"].members.{mid}'
        six = A["original_six"]["values"][mid]
        six_t = S.num(f"original_six.values.{mid}")
        if fam == "AMOS-P":
            dec = _rep_word(S, m_, six)
        else:
            dec = _diff_word(None, m_ + ".key", m_ + ".mean", S)
        L.append(S.line(f"\\quad {AMOS_CON_NAME[mid]} & {S.num(m_ + '.mean')} {S.ci(m_ + '.ci_t')} & "
                        f"{S.int(m_ + '.sign_k')}/{S.int(m_ + '.sign_n')} & {S.p(m_ + '.p_signflip_two_sided')} & "
                        f"{S.p(m_ + '.p_holm')} & {dec} & {S.ci(m_ + '.ci_boot_patient')} & "
                        f"{S.ci(m_ + '.ci_boot_patient_seed')} & {six_t} \\\\"))
        # the italic statistics line would repeat the decision row: it is logged, not printed
        log(f"amos_results: statistics line {i} (not printed): {st[i]}")
    m_ = 'families["AMOS-P"].members.C3'
    mx = A["families"]["AMOS-P"]["members"]["C3"]["max_arm"]
    S.rec(m_ + ".max_arm", mx, "word")
    c3w = {"holds": "equivalence within $\\pm 1$~mm shown in every arm"}
    L.append(S.line(f"\\quad C3: RL change within $\\pm 1$~mm in all 8 policy arms (intersection-union TOST) & & & "
                    f"{S.p(m_ + '.p_decision')} & {S.p(m_ + '.p_holm')} & {S.word(m_ + '.key', c3w)} & & & \\\\"))
    a_ = f'C3_per_arm["{mx}"]'
    L.append(S.line(f"\\quad\\quad Largest per-arm TOST $p$: {AMOS_ARM_NAME[mx]} & {S.num(a_ + '.mean')}, 90\\% "
                    f"{S.ci(a_ + '.ci90_t')} & {S.int(a_ + '.sign_k')}/{S.int(a_ + '.n')} & "
                    f"{S.p(a_ + '.p_tost_signflip')} (TOST) & & & & & \\\\"))
    o_ = "ordering"
    L.append(S.line(f"\\quad Ordering of the rungs and stages & $\\tau_b$ {S.plain(o_ + '.tau_b')} "
                    f"{S.ci(o_ + '.tau_b_ci95', 2, False)} & {S.int(o_ + '.sign_keeping.k')}/"
                    f"{S.int(o_ + '.sign_keeping.n')} & & & {S.word(o_ + '.class', {'recurs': 'recurs'})} & & & \\\\"))
    L += ["\\bottomrule", "\\end{tabularx}", "", "\\medskip"]
    # block 2: comparison with the replication cohort
    L += ["\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l l r l l@{}}", "\\toprule",
          "Contrast & AMOS (mm), 95\\% $t$ CI & 20-CT cohort (mm), 95\\% $t$ CI & Six & AMOS minus cohort, Welch 95\\% "
          "CI & Two-sample bootstrap 95\\% CI \\\\", "\\midrule"]
    for k in A["comparison"]["contrasts"]:
        c_ = f'comparison.contrasts["{k}"]'
        L.append(S.line(f"\\quad {AMOS_CON_NAME[k]} & {S.num(c_ + '.amos.mean')} {S.ci(c_ + '.amos.ci_t')} & "
                        f"{S.num(c_ + '.cohort.mean')} {S.ci(c_ + '.cohort.ci_t')} & {S.num(c_ + '.original_six')} & "
                        f"{S.num(c_ + '.difference')} {S.ci(c_ + '.welch_ci95')} & "
                        f"{S.ci(c_ + '.boot_ci95_two_sample')} \\\\"))
    t_ = "comparison.tau_b_between_cohorts"
    L.append(S.line(f"\\quad Kendall $\\tau_b$ between the cohorts' 12 cell means & & & & {S.plain(t_ + '.tau_b')} "
                    f"{S.ci(t_ + '.ci95', 2, False)} & \\\\"))
    L += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", "\\begin{table}[tp]",
          "\\renewcommand{\\baselinestretch}{1}\\scriptsize", "\\ContinuedFloat", "\\caption[]{Continued.}", "\\centering",
          "\\setlength{\\tabcolsep}{2.5pt}"]
    # block 3: the relabeling control (arm X)
    if S.get("arm_x.status") != "run":
        raise SystemExit("amos_q2r.json arm_x: not run; print the not-run sentence of amendment 4 instead")
    L += ["\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l c l@{}}", "\\toprule",
          f"Contrast (relabeling control on {S.int('arm_x.n_X')} cohort CTs) & Relabeled minus cohort, paired 95\\% $t$ "
          "CI & $k$/$n$ & AMOS minus relabeled cohort, Welch 95\\% CI \\\\", "\\midrule"]
    L[-2] = S.line(L[-2])
    for k in A["arm_x"]["contrasts"]:
        x_ = f'arm_x.contrasts["{k}"]'
        L.append(S.line(f"\\quad {AMOS_CON_NAME[k]} & {S.num(x_ + '.paired_vs_cohort.mean')} "
                        f"{S.ci(x_ + '.paired_vs_cohort.ci_t')} & {S.int(x_ + '.paired_vs_cohort.k')}/"
                        f"{S.int(x_ + '.paired_vs_cohort.n')} & {S.num(x_ + '.amos_minus_arm_x.D')} "
                        f"{S.ci(x_ + '.amos_minus_arm_x.welch_ci95')} \\\\"))
    L += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return write(out, "amos_results.tex", "\n".join(L))


# --------------------------------------------------------------------------------- amendment 5 ([Tab], rule input)
CMB_D = (("D-P", "Combination minus authors' original pipeline, physics frames"),
         ("D-S", "Combination minus authors' original pipeline, SonoSPADE frames"),
         ("B-P", "Combination minus its learned-mask servo base, physics frames"),
         ("B-S", "Combination minus its learned-mask servo base, SonoSPADE frames"))
CMB_G = (("G-P", "Combination minus pose oracle, physics frames"),
         ("G-S", "Combination minus pose oracle, SonoSPADE frames"))
Q2R_COHORT = "q2r_cohort_v2.json"


def _q2r_cohort():
    if "q2r" not in _A345:
        p = os.path.join(AP_DIR, Q2R_COHORT)
        _guard(p)
        d = json.load(open(p))
        if _status_of(d) != STATUS_V21:
            raise SystemExit(f"{Q2R_COHORT}: unexpected analysis_status")
        _A345["q2r"] = d
    return _A345["q2r"]


def combination_tab(out):
    """tables/combination.tex (tab:combination, [Tab] of amendment 5 others:P and others:S; Appendix C.10)."""
    if not _a345_ready("amend5_combination.json", "amend5_armh.json"):
        log("combination.tex: amendment 5 blinded in this build; not written")
        return None
    F, FH, FA = "amend5_combination.json", "amend5_armh.json", "amend5_amos.json"
    Cb, H = _a345(F), _a345(FH)
    Am = _a345(FA) if _a345_ready(FA) else None
    if any(Cb["cohort"]["families"][f]["missing"] for f in ("CMB-D", "CMB-E", "CMB-G")) or \
            not all(Cb["record_states"].values()) or Cb["hand_run"] or not Cb["reference_recompute"]["passed"]:
        raise SystemExit("amend5_combination.json: incomplete families, a record state not true, a hand run or a "
                         "failed reference recompute; print the not-run sentences of amendment 5 instead")
    if H["arm_h"]["status"] != "complete":
        raise SystemExit("amend5_armh.json: arm H not complete")
    Q = _q2r_cohort()
    S = _S(F, Cb)
    ncase = S.int("cohort.n_cases")
    # "the authors' original pipeline", the one name of the authors' baseline in the text (the rows below say it)
    caption = ("\\caption{Combination chosen by the pre-specified rule (exploratory): the authors' original "
               "pipeline with the SSIM appearance term in the REINFORCE reward, the only slot the rule changed. "
               f"Primary: the {ncase} replication-cohort CTs, exact sign-flip test, Holm within families CMB-D "
               "($m = 4$, difference), CMB-E ($m = 4$, equivalence within $\\pm 1$~mm, TOST) and CMB-G ($m = 2$, gap to "
               "the pose oracle, conditional on this renderer: grid-fixed speckle, no elevational beam width, rigid "
               "tissue). Each D difference compares three trained policies per arm and is conditional on those six "
               "RL runs. Other pipelines measured on these CTs print as measured and none is ranked against the "
               f"combination. MDD from the observed SDs, at $\\alpha = {S.plan(5, 'alpha')}$ and at "
               "Holm's first step. "
               "The continuation prints the six held-out patients (selection-biased; estimation only), the "
               "same-procedure retraining of the match score $S$ (estimation only) and AMOS (secondary; estimation only).}")
    S.rec("rule_rederivation.slots_changed", Cb["rule_rederivation"]["slots_changed"], "word")
    S.rec("rule_rederivation.reward", Cb["rule_rederivation"]["reward"], "word")
    caption += S.tail()
    S.e = []
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend5_combination.json, amend5_armh.json, amend5_amos.json and "
                        f"q2r_cohort_v2.json ({_T('src a5 results')})")]
    head = ("Contrast & Family & Difference or 90\\% CI (mm) & $k$/20 & Sign-flip $p$ & Holm $p$ & Decision & "
            "Crossed 95\\% CI & MDD (mm) \\\\")
    L += _table_head("tab:combination", caption, "@{}>{\\raggedright\\arraybackslash}X l l c c c "
                     ">{\\raggedright\\arraybackslash}p{0.11\\textwidth} l l@{}", head)
    # (1) the pipelines on the cohort, as measured
    L.append(_mc(9, f"\\textit{{Pipelines on the {ncase} cohort CTs, final translation error (mm), as measured and "
                    "unranked}") + " \\\\")
    rf = "cohort.references_mm"
    L.append(S.line(f"\\quad Static start {S.plain(rf + '.static')}; pose oracle (privileged) "
                    f"{S.plain(rf + '.pose_oracle')} & & & & & & & & \\\\"))
    for comp, frames in (("physics", "physics frames"), ("sonospade", "SonoSPADE frames")):
        L.append(S.line(f"\\quad With {frames}: authors' original pipeline {S.plain(f'{rf}.authors_{comp}')}; its "
                        f"learned-mask servo base {S.plain(f'{rf}.base_{comp}')}; the combination "
                        f"{S.plain(f'{rf}.combination_{comp}')} & & & & & & & & \\\\"))
    cells = Q["ordering"]["cells"]
    want = {"2": "scripted", "8": "learned base, CUT", "9": "learned final, CUT", "10": "learned base, CycleGAN",
            "11": "learned final, CycleGAN"}
    if any(cells[k]["name"] != v for k, v in want.items()):
        raise SystemExit(f"{Q2R_COHORT}: ordering cells 2 and 8 to 11 are not the expected rungs")
    S.at(Q2R_COHORT, Q)
    L.append(S.line(f"\\quad Scripted sweep {S.plain('ordering.cells[\"2\"].cohort')}; CUT frames: learned-mask servo "
                    f"base {S.plain('ordering.cells[\"8\"].cohort')}, REINFORCE final "
                    f"{S.plain('ordering.cells[\"9\"].cohort')}; CycleGAN frames: base "
                    f"{S.plain('ordering.cells[\"10\"].cohort')}, final {S.plain('ordering.cells[\"11\"].cohort')} "
                    "& & & & & & & & \\\\"))
    S.at(F, Cb)
    # (2) the families
    L += ["\\addlinespace", _mc(9, "\\textit{Families on the cohort CTs}") + " \\\\"]
    for mid, name in CMB_D:
        d_, e_ = f'cohort.families["CMB-D"].members["{mid}"]', f'cohort.families["CMB-E"].members["{mid}"]'
        md_ = f'cohort.mdd["{mid}"]'
        L.append(S.line(f"\\quad {mid}: {name} & CMB-D & {S.num(d_ + '.mean')} {S.ci(d_ + '.ci_t')} & "
                        f"{S.int(d_ + '.sign_k')} & {S.p(d_ + '.p_signflip_two_sided')} & {S.p(d_ + '.p_holm')} & "
                        f"{S.word(d_ + '.decision_word', {'holds higher': 'holds higher', 'holds lower': 'holds lower', 'no difference detected': 'no difference detected'})} & "
                        f"{S.ci(d_ + '.crossed_case_by_seed_ci95')} & {S.plain(md_ + '.mdd_alpha_005_mm')}, "
                        f"{S.plain(md_ + '.mdd_holm_first_mm')} \\\\"))
        L.append(S.line(f"\\quad\\quad Equivalence within $\\pm 1$~mm & CMB-E & 90\\% {S.ci(e_ + '.ci90_t')} & & "
                        f"{S.p(e_ + '.p_decision')} (TOST) & {S.p(e_ + '.p_holm')} & "
                        f"{S.word(e_ + '.decision_word', {'equivalent within +/- 1 mm': 'equivalent within $\\pm 1$~mm', 'equivalence not shown': 'equivalence not shown'})} "
                        "& & \\\\"))
    for mid, name in CMB_G:
        g_, md_ = f'cohort.families["CMB-G"].members["{mid}"]', f'cohort.mdd["{mid}"]'
        L.append(S.line(f"\\quad {mid}: {name} & CMB-G & {S.num(g_ + '.mean')} {S.ci(g_ + '.ci_t')} & "
                        f"{S.int(g_ + '.sign_k')} & {S.p(g_ + '.p_signflip_two_sided')} & {S.p(g_ + '.p_holm')} & "
                        f"{S.word(g_ + '.decision_word', {'holds higher': 'holds higher'})}; conditional on this "
                        f"renderer & & {S.plain(md_ + '.mdd_alpha_005_mm')}, {S.plain(md_ + '.mdd_holm_first_mm')} "
                        "\\\\"))
    L += _table_foot()
    # continuation
    head2 = "Row & Estimate (mm), 95\\% $t$ CI & $k$/$n$ & Sign-flip $p$ & Interval (sensitivity) \\\\"
    L += _table_head("tab:combination", "", "@{}>{\\raggedright\\arraybackslash}X l c c l@{}", head2, cont=True)
    # (3) secondary
    L.append(_mc(5, "\\textit{Secondary, cohort CTs (no family, no decision)}") + " \\\\")
    for comp, frames in (("physics", "physics frames"), ("sonospade", "SonoSPADE frames")):
        s_ = f"cohort.secondary.{comp}"
        L.append(S.line(f"\\quad Combination, {frames}: mean final error & {S.plain(s_ + '.mean_final_mm.mean')} "
                        f"{S.ci(s_ + '.mean_final_mm.ci_t', 2, False)} & & & \\\\"))
        L.append(S.line(f"\\quad\\quad minus the static start & {S.num(s_ + '.minus_static.mean')} "
                        f"{S.ci(s_ + '.minus_static.ci_t')} & {S.int(s_ + '.minus_static.sign_k')}/"
                        f"{S.int(s_ + '.minus_static.sign_n')} & & \\\\"))
        L.append(S.line(f"\\quad\\quad share of the static-to-oracle distance closed & "
                        f"{S.num(s_ + '.share_static_to_oracle_closed.estimate', 3, False, False)} "
                        f"{S.ci(s_ + '.share_static_to_oracle_closed.ci', 3, False)} (case bootstrap) & & & \\\\"))
    # (4) six held-out patients
    lab6 = S.get("six.label")
    S.rec("six.label", lab6, "word")
    L.append("\\addlinespace")
    L.append(S.line(_mc(5, f"\\textit{{Six held-out patients ({lab6})}}") + " \\\\"))
    r6 = "six.references_mm"
    for comp, frames in (("physics", "physics frames"), ("sonospade", "SonoSPADE frames")):
        L.append(S.line(f"\\quad With {frames}: authors' original pipeline {S.plain(f'{r6}.authors_{comp}')}, servo "
                        f"base {S.plain(f'{r6}.base_{comp}')}, combination {S.plain(f'{r6}.combination_{comp}')} "
                        "& & & & \\\\"))
    for mid, name in CMB_D + CMB_G:
        x_ = f'six.estimates["{mid}"]'
        L.append(S.line(f"\\quad {mid}: {name} & {S.num(x_ + '.mean')} {S.ci(x_ + '.ci_t')} & "
                        f"{S.int(x_ + '.sign_k')}/{S.int(x_ + '.sign_n')} & {S.p(x_ + '.p_signflip_two_sided')} & "
                        f"{S.ci(x_ + '.ci_boot_patient_seed')} (patient by seed) \\\\"))
    # (5) arm H
    S.at(FH, H)
    labh = S.get("label")
    S.rec("label", labh, "word")
    L.append("\\addlinespace")
    L.append(S.line(_mc(5, f"\\textit{{Same-procedure retraining of the match score $S$ ({labh}): the two differences side by "
                           "side, never added}") + " \\\\"))
    for comp, frames in (("physics", "physics frames"), ("sonospade", "SonoSPADE frames")):
        for set_, sname in (("cohort", f"{ncase} cohort CTs"), ("six", "six held-out patients")):
            h_ = f"arm_h.per_stage.{comp}.{set_}"
            L.append(S.line(f"\\quad {frames}, {sname}: combination minus rerun & {S.num(h_ + '.combination_minus_armh.mean')} "
                            f"{S.ci(h_ + '.combination_minus_armh.ci_t')} & {S.int(h_ + '.combination_minus_armh.sign_k')}/"
                            f"{S.int(h_ + '.combination_minus_armh.sign_n')} & & \\\\"))
            L.append(S.line(f"\\quad\\quad rerun minus the authors' original pipeline & "
                            f"{S.num(h_ + '.armh_minus_authors.mean')} {S.ci(h_ + '.armh_minus_authors.ci_t')} & "
                            f"{S.int(h_ + '.armh_minus_authors.sign_k')}/{S.int(h_ + '.armh_minus_authors.sign_n')} & & "
                            "\\\\"))
    # (6) AMOS
    L.append("\\addlinespace")
    if Am is None:
        L.append(_mc(5, "\\textit{AMOS rows: not read in this build (amendment 4 blinded)}") + " \\\\")
    else:
        S.at(FA, Am)
        laba = S.get("label")
        S.rec("label", laba, "word")
        L.append(S.line(_mc(5, f"\\textit{{{S.int('n_A')} AMOS CTs ({laba}; no Holm, no decision)}}") + " \\\\"))
        ra = "references_mm"
        L.append(S.line(f"\\quad Static start {S.plain(ra + '.static')}; pose oracle (privileged) "
                        f"{S.plain(ra + '.pose_oracle')} & & & & \\\\"))
        for comp, frames in (("physics", "physics frames"), ("sonospade", "SonoSPADE frames")):
            L.append(S.line(f"\\quad With {frames}: authors' original pipeline {S.plain(f'{ra}.authors_{comp}')}, servo "
                            f"base {S.plain(f'{ra}.base_{comp}')}, combination {S.plain(f'{ra}.combination_{comp}')} "
                            "& & & & \\\\"))
        for mid, name in CMB_D + CMB_G:
            x_ = f'estimates["{mid}"]'
            meth = S.get(x_ + ".signflip_method_printed")
            S.rec(x_ + ".signflip_method_printed", meth, "word")
            L.append(S.line(f"\\quad {mid}: {name} & {S.num(x_ + '.mean')} {S.ci(x_ + '.ci_t')} & "
                            f"{S.int(x_ + '.sign_k')}/{S.int(x_ + '.sign_n')} & {S.p(x_ + '.p_signflip_two_sided')}"
                            + ("$^{\\mathrm{MC}}$" if meth.startswith("Monte Carlo") else "")
                            + f" & {S.ci(x_ + '.crossed_case_by_seed_ci95')} (crossed) \\\\"))
        L.append(_mcp(5, "\\quad $^{\\mathrm{MC}}$Monte Carlo sign-flip $p$ (1,000,000 sign vectors, seed 0).")
                 + " \\\\ % src: note: amend5_amos.json estimates[*].signflip_method_printed")
    # (7) start lifts
    S.at(F, Cb)
    for kp in ("lifts.six", "lifts.cohort"):
        if S.get(kp) is not None:
            raise SystemExit(f"{F}:{kp} is recorded now; print it")
        S.rec(kp, None, "word")
    if Am is not None:
        if Am.get("lift") is not None:
            raise SystemExit(f"{FA}:lift is recorded now; print it")
        S.rec("lift", None, "word", f=FA)
    # the "not recorded" start-lift line is not printed; the check above refuses a build in which the lifts are
    # recorded
    log("combination: " + S.line("start-lift share and mean lift per set: not recorded by the analysis ("
                                 + _T("lift tag") + ")"))
    L += _table_foot() + [""]
    return write(out, "combination.tex", "\n".join(L))


RULE_SLOT_NAME = {"renderer": "Renderer", "texture": "Texture stage", "mask": "Mask source", "controller": "Controller",
                  "reward": "Reward"}
# the rows that amendment 5's rule did not read, by the label of its Appendix (development repository only: the
# rule-input table is written there)
NOT_READ_ROWS = PP.NOT_READ_ROWS if PP else {}


def _short_dec(w):
    return {DEC_NULL: "no detectable difference"}.get(w, w)


def rule_input_tab(out):
    """tables/rule_input.tex (tab:rule-input; Appendix C.10): the rows amendment 5's rule read, with the decision word
    it read, beside the word of the final Table 5 (figures/plug_rows.json), every change marked (amendment 5: the
    rule was applied once; no later decision word reopens it)."""
    if not _a345_ready("amend5_combination.json"):
        log("rule_input.tex: amendment 5 blinded in this build; not written")
        return None
    PS_ = _plan_stats()
    A5 = PS_.load_amendment(5)
    sel = A5["selection"]
    Cb = _a345("amend5_combination.json")
    pr = os.path.join(FIG_OUT, "plug_rows.json")
    if hashlib.md5(open(pr, "rb").read()).hexdigest() != Cb["final_table5"]["md5"]:
        raise SystemExit("figures/plug_rows.json md5 differs from amend5_combination.json final_table5.md5")
    J = json.load(open(pr))
    fin = {SL["key"]: {r["id"]: r for r in SL["rows"]} for SL in J["slots"]}
    S = _S("plan:amend5", A5)
    # while final_table5.provisional is true, Table 5 is the current, provisional table, not the final one
    prov5 = bool(Cb["final_table5"]["provisional"])
    t5now = ("the current Table~\\ref{tab:pluggability}, provisional until the tag aggregation" if prov5
             else "the final Table~\\ref{tab:pluggability}")
    caption = (_T("rule input caption")
               + t5now + ". A row replaces the authors' component only if its word is "
               f"lower. The rule was applied once, and no later word reopens it ({_T('a5 rule item')}).}}"
               + S.tail())
    head = ("Slot and row & Read by the rule: error (mm), Holm $p$, word & " + ("Current" if prov5 else "Final")
            + " Table~5: error (mm), Holm $p$, word & Change \\\\")
    L = [GEN.format(src=PI.text("rule input src"))]
    L += _table_head("tab:rule-input", caption, "@{}>{\\raggedright\\arraybackslash}X l l l@{}", head)
    lab = {SL["key"]: {r["id"]: r["label"] for r in SL["rows"]} for SL in PLUG_SLOTS}
    changes, lower_now = [], []
    for slot, rr in sel["rows_read"].items():
        sk = sel["snapshot_slot_key"][slot]
        L.append(_mc(4, f"\\textit{{{RULE_SLOT_NAME[slot]}}}") + " \\\\")
        for i, r in enumerate(rr["rows"]):
            S.at("plan:amend5", A5)
            k_ = f"selection.rows_read.{slot}.rows[{i}]"
            f_ = fin[sk][r["id"]]
            j_ = [x["id"] for x in J["slots"][[s_["key"] for s_ in J["slots"]].index(sk)]["rows"]].index(r["id"])
            jk = f"slots[{[s_['key'] for s_ in J['slots']].index(sk)}].rows[{j_}]"
            read = (f"{S.plain(k_ + '.err_mm')}, ${S.num(k_ + '.p_holm', 3, False, False)}$, "
                    f"{_short_dec(r['decision'])}")   # Holm p as the plan recorded it (field_compare.p_holm_decimals 3)
            S.rec(k_ + ".decision", r["decision"], "word")
            S.at(_T("plug_rows path"), J)
            now = f"{S.plain(jk + '.err_mm')}, {S.p(jk + '.p_holm')}, {_short_dec(f_['decision'])}"
            S.rec(jk + ".decision", f_["decision"], "word")
            ch = "word changed" if f_["decision"] != r["decision"] else ""
            if ch:
                changes.append((r["id"], r["decision"], f_["decision"]))
            if f_["decision"] == "lower":
                lower_now.append(r["id"])
            L.append(S.line(f"\\quad {lab[sk][r['id']]} & {read} & {now} & {ch} \\\\"))
        for nr in rr.get("not_read", []):
            m_ = NOT_READ_ROWS.get(nr, "missing")
            if m_ == "missing":
                raise SystemExit(f"amendment 5 rows_read.{slot}.not_read: no row map for {nr!r}")
            if m_ is None:
                S.at("plan:amend5", A5).rec(f"selection.rows_read.{slot}.not_read", rr["not_read"], "word")
                L.append(S.line(f"\\quad {_cap1(nr)} & not read & not a Table~5 family row & \\\\"))
                continue
            for rid in m_[1]:
                f_ = fin[sk][rid]
                si = [s_["key"] for s_ in J["slots"]].index(sk)
                j_ = [x["id"] for x in J["slots"][si]["rows"]].index(rid)
                jk = f"slots[{si}].rows[{j_}]"
                S.at("plan:amend5", A5).rec(f"selection.rows_read.{slot}.not_read", rr["not_read"], "word")
                S.at(_T("plug_rows path"), J)
                why = "blinded" if "blinded" in nr else ("optional, pending" if "optional" in nr else "pending")
                cand = f_["role"] == "family"
                now = (f"{S.plain(jk + '.err_mm')}, {S.p(jk + '.p_holm')}, {_short_dec(f_['decision'])}"
                       if f_.get("p_holm") is not None else f"{S.plain(jk + '.err_mm')}, {f_['role']} row")
                S.rec(jk + ".role", f_["role"], "word")
                if cand and f_["decision"] == "lower":
                    lower_now.append(rid)
                L.append(S.line(f"\\quad {lab[sk][rid]} & not read ({why}) & {now} & "
                                + ("" if cand else "not a candidate") + " \\\\"))
    ssim = fin["reward"]["ssim"]["decision"]
    S.at("amend5_combination.json", Cb)
    sd = S.get("final_table5.ssim_decision")
    S.rec("final_table5.ssim_decision", sd, "word")
    S.rec("final_table5.provisional", Cb["final_table5"]["provisional"], "word")
    if sd != ssim:
        raise SystemExit("SSIM decision of the final Table 5 differs from amend5_combination.json final_table5")
    if lower_now != ["ssim"]:
        raise SystemExit(f"rows of the final Table 5 that print lower: {lower_now}; the rule outcome sentence needs "
                         "a new reading")
    # the non-provisional branch prints "final. "
    L.append(S.line(_mcp(4, "\\quad The SSIM row's word is " + ("provisional until the tag aggregation. "
                                                               if prov5 else "final. ")
                         + "The rule replaces a component only on a lower decision, so the choice stands; no other "
                           "candidate row of the " + ("current" if prov5 else "final") + " Table~5 prints lower.")
                    + " \\\\"))
    L += _table_foot() + [""]
    log("rule input: word changes " + "; ".join(f"{a}: {b!r} -> {c!r}" for a, b, c in changes))
    _A345["rule_changes"] = changes
    return write(out, "rule_input.tex", "\n".join(L))


# --------------------------------------------------------------------------------- Supplementary S11 detail tables
SEC_LEARNER = {"dagger_res": "DAgger residual", "bc_res": "BC residual"}
SEC_RUNG = dict(ARM_B_RUNGS)
# (the token of the record's secondary row, its printed words, unit); the first two tokens as the records at hand
# word them (plan_inputs.text)
SEC_METRIC = [(_T("sec bn token"), "BatchNorm in train mode minus the reference", "mm"),
              (_T("sec reinforce token"), "minus the learned-mask REINFORCE final of the primary results", "mm"),
              ("minus its own CT-mask row", "minus its own CT-mask row", "mm"),
              ("minus the earlier row of the same learner (learner-seed replication, descriptive)",
               "minus the earlier row of the same learner (learner-seed replication, descriptive)", "mm"),
              ("kb_success_5mm_5deg", "keep-best success within 5~mm and 5$^\\circ$", "fraction"),
              ("kb_success_10mm_10deg", "keep-best success within 10~mm and 10$^\\circ$", "fraction"),
              ("success_5mm_5deg", "success within 5~mm and 5$^\\circ$", "fraction"),
              ("success_10mm_10deg", "success within 10~mm and 10$^\\circ$", "fraction"),
              ("final_ang", "rotation error", "deg"), ("final_match", "final match score $S$", "score"),
              ("|lat_mm|", "lateral error", "mm"), ("|elev_mm|", "elevational error", "mm"),
              ("|ax_mm|", "axial error", "mm"), ("|tilt_out_deg|", "out-of-plane tilt", "deg"),
              ("|tilt_in_deg|", "in-plane tilt", "deg"), ("|roll_deg|", "roll", "deg")]


def _sec_metric(row):
    """(printed metric words, unit) of a secondary row name; the unit comes from the metric, never from .string,
    which prints 'mm' for every row."""
    for tok, words, unit in SEC_METRIC:
        if tok in row:
            return words, unit
    raise SystemExit(f"secondary row {row!r}: no metric words")


_CI_IN_STRING = re.compile(r"([+-]?\d+\.\d+) mm \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]")


def s11_arm_a_secondary(out):
    """tables/s11_arm_a_secondary.tex (Supplementary S11): amend3_arm_a.json result.secondary, label 'secondary'."""
    if not _a345_ready("amend3_arm_a.json"):
        return None
    F = "amend3_arm_a.json"
    A = _a345(F)
    S = _S(F, A)
    caption = ("\\caption{Secondary comparisons of the learned-mask controller experiment (secondary; no family, no "
               "decision). Learner minus the learned-mask servo base unless the row names another reference; mean "
               "over the six held-out patients with the 95\\% $t$ CI and the patients of 6 sharing its sign. Units "
               "from the metric: mm, deg, the match score $S$, or a fraction of episodes.}")
    head = "Learner, frames & Row & Unit & Mean, 95\\% $t$ CI & $k$/6 \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend3_arm_a.json result.secondary (amendment 3, arm A)")]
    L += _table_head("tab:s11-arm-a-secondary", caption, "@{}l >{\\raggedright\\arraybackslash}X l l c@{}", head)
    for i, r in enumerate(A["result"]["secondary"]):
        kp = f"result.secondary[{i}]"
        lr, st = r["row"].split()[:2]
        words, unit = _sec_metric(r["row"])
        rest = r["row"].split(" ", 2)[2]
        if words.startswith(("minus", "BatchNorm")):
            text = words
        else:
            text = f"{words} minus the reference"
            if not rest.endswith("minus the reference"):
                raise SystemExit(f"{F}:{kp}.row {r['row']!r}: unexpected form")
        d_ = 3 if unit in ("fraction", "score") else 2
        S.rec(kp + ".label", r["label"], "word")
        S.rec(kp + ".row", r["row"], "word")
        fr = "physics frames" if st == "physics" else "SonoSPADE frames"
        L.append(S.line(f"{SEC_LEARNER[lr]}, {fr} & {text} & {unit} & {S.num(kp + '.mean', d_)} "
                        f"{S.ci(kp + '.ci_t', d_)} & {S.int(kp + '.sign_k')}/6 \\\\"))
    L += _table_foot() + [""]
    return write(out, "s11_arm_a_secondary.tex", "\n".join(L))


def s11_arm_b_secondary(out):
    """tables/s11_arm_b_secondary.tex (Supplementary S11): amend3_arm_b.json result.secondary, with the tangent and
    tilted windows (episode level, descriptive)."""
    if not _a345_ready("amend3_arm_b.json"):
        return None
    F = "amend3_arm_b.json"
    B = _a345(F)
    S = _S(F, B)
    caption = ("\\caption{Secondary comparisons of the skin-tangent contact experiment (secondary; no family, no "
               "decision): skin-tangent minus target-anchored plane, same rung and seed. Patient-level rows: mean "
               "over the six held-out patients with its 95\\% $t$ CI; units from the "
               "metric. Window rows: the mean episode difference of final translation error (mm) over the windows "
               "where the target-anchored plane is tangent to the skin (window tilt below 1$^\\circ$) and over the "
               "tilted windows (episode level, descriptive).}")
    head = "Rung & Row & Unit & Mean, 95\\% CI or episodes \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend3_arm_b.json result.secondary (amendment 3, arm B)")]
    sec = B["result"]["secondary"]
    half = next(i for i, r in enumerate(sec) if r["row"].startswith("ct_base"))
    for part, (a_, b_) in enumerate(((0, half), (half, len(sec)))):
        L += _table_head("tab:s11-arm-b-secondary", caption, "@{}l >{\\raggedright\\arraybackslash}X l l@{}", head,
                         cont=part == 1)
        for i in range(a_, b_):
            r = sec[i]
            kp = f"result.secondary[{i}]"
            rung = r["row"].split()[0]
            S.rec(kp + ".row", r["row"], "word")
            if "windows" in r["row"]:
                kind = "tangent windows" if "tangent" in r["row"] else "tilted windows"
                L.append(S.line(f"{SEC_RUNG[rung]} & {kind} & mm & {S.num(kp + '.mean_episode_difference_mm')} "
                                f"over {S.int(kp + '.n_episodes')} episodes \\\\"))
                continue
            words, unit = _sec_metric(r["row"])
            d_ = 2   # the analysis printed these intervals to 2 decimals (.string)
            m = _CI_IN_STRING.match(r["string"])
            if not m:
                raise SystemExit(f"{F}:{kp}.string {r['string']!r}: no interval")
            mean = S.num(kp + ".mean", d_)
            lo, hi = m.group(2), m.group(3)
            S.rec(kp + ".string", r["string"], f"[{lo}, {hi}]")
            civ = f"$[{lo}, {hi}]$"
            L.append(S.line(f"{SEC_RUNG[rung]} & {words} & {unit} & {mean} {civ} \\\\"))
        L += _table_foot()
    L.append("")
    return write(out, "s11_arm_b_secondary.tex", "\n".join(L))


def s11_amos_cells(out):
    """tables/s11_amos_cells.tex (Supplementary S11): the 13 cells of the ordering, original six, AMOS and the
    replication cohort (descriptive; never pooled)."""
    if not _a345_ready("amos_q2r.json"):
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    S = _S(F, A)
    caption = ("\\caption{The 13 rung and stage cells of the AMOS ordering criterion: final translation error (mm), mean "
               "over seeds of case means, on the original six held-out patients, the 30 AMOS CTs and the 20 "
               "replication-cohort CTs (descriptive; the three sets are printed side by side and never pooled).}")
    head = "Cell & Original six & AMOS & 20-CT cohort \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amos_q2r.json ordering.cells and comparison.cells")]
    L += _table_head("tab:s11-amos-cells", caption, "@{}>{\\raggedright\\arraybackslash}X r r r@{}", head)
    for k, c in A["ordering"]["cells"].items():
        o_, c_ = f'ordering.cells["{k}"]', f'comparison.cells["{k}"]'
        if A["comparison"]["cells"][k]["name"] != c["name"]:
            raise SystemExit(f"amos_q2r.json: cell {k} names differ")
        S.rec(o_ + ".name", c["name"], "word")
        L.append(S.line(f"\\quad {_cap1(c['name'])} & {S.plain(o_ + '.original_six')} & {S.plain(o_ + '.amos')} & "
                        f"{S.plain(c_ + '.cohort.mean')} \\\\"))
    L += _table_foot() + [""]
    return write(out, "s11_amos_cells.tex", "\n".join(L))


def s11_amos_cases(out):
    """tables/s11_amos_cases.tex (Supplementary S11): per-case descriptors of amendment 4 from the
    ledgered frozen manifest (screen records; no image or label file is opened)."""
    if not _a345_ready("amos_q2r.json"):
        return None
    M = _amos_frozen()
    F = AMOS_FROZEN
    S = _S(F, M)
    caption = ("\\caption{Per-case descriptors of the 30 AMOS cohort CTs (descriptive). AMOS "
               "records no per-case age, sex, institute, study type, vendor or pathology. Split and native slice "
               "spacing; axial extent; median HU of the liver and the aorta; fraction of kidney voxels at or above "
               "300~HU; lateral truncation of the kidney slices; screened liver and kidney windows; Dice of the "
               "TotalSegmentator v2 labels against the AMOS manual labels (liver; right and left kidney).}")
    head = ("Case & Split & Slice (mm) & Extent (mm) & Liver HU & Aorta HU & Kidney $\\ge$300~HU & Truncation & "
            "Liver win. & Kidney win. & Liver Dice & Kidney Dice \\\\")
    L = [GEN.format(src="outputs/v2_1/amos/amos_frozen.json screen[<case>] (the ledgered frozen manifest)")]
    L += _table_head("tab:s11-amos-cases", caption, "@{}l l r r r r r r r r r l@{}", head)
    if sorted(M["cases"]) != sorted(_a345("amos_q2r.json")["cases"]):
        raise SystemExit("amos_frozen.json cases differ from amos_q2r.json cases")
    for case in M["cases"]:
        e_ = f'screen["{case}"]'
        ed = e_ + ".eligibility_descriptors"
        S.rec(ed + ".split", M["screen"][case]["eligibility_descriptors"]["split"], "word")
        L.append(S.line(f"{case.replace('_', '\\_')} & {M['screen'][case]['eligibility_descriptors']['split']} & "
                        f"{S.plain(ed + '.slice_spacing_mm')} & {S.num(ed + '.axial_extent_mm', 0, False, False)} & "
                        f"{S.num(ed + '.liver_median_hu', 0, False, False)} & "
                        f"{S.num(ed + '.aorta_median_hu', 0, False, False)} & "
                        f"{S.num(ed + '.kidney_frac_ge_300', 3, False, False)} & "
                        f"{S.num(ed + '.kidney_lateral_truncation', 3, False, False)} & "
                        f"{S.int(e_ + '.n_windows.liver')} & {S.int(e_ + '.n_kidney_windows')} & "
                        f"{S.plain(e_ + '.label.dice.liver')} & {S.plain(e_ + '.label.dice.kidney_right')}, "
                        f"{S.plain(e_ + '.label.dice.kidney_left')} \\\\"))
    L += _table_foot() + [""]
    return write(out, "s11_amos_cases.tex", "\n".join(L))


def s11_amos_secondary(out):
    """tables/s11_amos_secondary.tex (Supplementary S11): C3 per arm, the arm-averaged C3 and the contrasts without the
    CTs whose aorta median is at least 300 HU (secondary; no decision)."""
    if not _a345_ready("amos_q2r.json"):
        return None
    F = "amos_q2r.json"
    A = _a345(F)
    S = _S(F, A)
    n9 = len(A["secondary"]["aorta_ge_300"])
    caption = ("\\caption{Secondary comparisons of the AMOS cohort (secondary; no family beyond C3, no "
               "decision). Top: REINFORCE final minus after imitation per policy arm, mean with its 90\\% $t$ CI. "
               "Each arm prints its sign-flip TOST $p$ at $\\pm 1$~mm (Monte Carlo, 1,000,000 sign vectors, seed 0); "
               "the last row averages the 8 arms. Bottom: the four contrasts without the "
               f"{n9} CTs whose aorta median is at least 300~HU.}}")
    head = "Row & Estimate (mm), CI & $k$/$n$ & TOST $p$ or $n$ \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amos_q2r.json C3_per_arm, C3_arm_averaged_secondary and "
                        "secondary.contrasts")]
    L += _table_head("tab:s11-amos-secondary", caption, "@{}>{\\raggedright\\arraybackslash}X l c l@{}", head)
    for arm in A["C3_per_arm"]:
        a_ = f'C3_per_arm["{arm}"]'
        L.append(S.line(f"\\quad RL change, {AMOS_ARM_NAME[arm]} & {S.num(a_ + '.mean')}, 90\\% {S.ci(a_ + '.ci90_t')} & "
                        f"{S.int(a_ + '.sign_k')}/{S.int(a_ + '.n')} & {S.p(a_ + '.p_tost_signflip')} \\\\"))
    a_ = "C3_arm_averaged_secondary"
    L.append(S.line(f"\\quad Average over the 8 arms & {S.num(a_ + '.mean')}, 95\\% {S.ci(a_ + '.ci95_t')} & "
                    f"{S.int(a_ + '.sign_k')}/{S.int(a_ + '.n')} & {S.p(a_ + '.p_tost_signflip')} \\\\"))
    L.append("\\addlinespace")
    for k in ("C1", "C2", "gap", "bound"):
        c_ = f"secondary.contrasts.{k}"
        L.append(S.line(f"\\quad {AMOS_CON_NAME[k]}, without the opacified aortas & {S.num(c_ + '.mean')}, 95\\% "
                        f"{S.ci(c_ + '.ci_t')} & {S.int(c_ + '.sign_k')}/{S.int(c_ + '.sign_n')} & "
                        f"$n = {S.int(c_ + '.n')}$ \\\\"))
    L += _table_foot() + [""]
    return write(out, "s11_amos_secondary.tex", "\n".join(L))


def s11_combination_seeds(out):
    """tables/s11_combination_seeds.tex (Supplementary S11): per-seed differences and the crossed or patient-by-seed
    intervals of every D and B member of amendment 5, cohort, six and AMOS."""
    if not _a345_ready("amend5_combination.json"):
        return None
    F, FA = "amend5_combination.json", "amend5_amos.json"
    Cb = _a345(F)
    Am = _a345(FA) if _a345_ready(FA) else None
    S = _S(F, Cb)
    caption = ("\\caption{Per-seed differences of the combination's D and B members (mm; seed $s$ of the combination minus "
               "seed $s$ of its reference, for seeds 0, 1 and 2), with the case-by-seed interval: crossed on the 20 "
               "cohort CTs and on AMOS, patient by seed on the six (selection-biased; estimation only). AMOS: "
               "secondary; estimation only.}")
    head = "Member & Set & Per seed (mm) & Interval, 95\\% \\\\"
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend5_combination.json cohort.families[\"CMB-D\"] and "
                        "six.estimates; amend5_amos.json estimates")]
    L += _table_head("tab:s11-combination-seeds", caption, "@{}>{\\raggedright\\arraybackslash}X l l l@{}", head)
    for mid, name in CMB_D:
        S.at(F, Cb)
        c_ = f'cohort.families["CMB-D"].members["{mid}"]'
        ps_ = ", ".join(S.num(f"{c_}.per_seed_differences_mm[{i}]") for i in range(3))
        L.append(S.line(f"\\quad {mid}: {name} & 20 cohort CTs & {ps_} & {S.ci(c_ + '.crossed_case_by_seed_ci95')} "
                        "(crossed) \\\\"))
        s_ = f'six.estimates["{mid}"]'
        ps_ = ", ".join(S.num(f"{s_}.per_seed_differences_mm[{i}]") for i in range(3))
        L.append(S.line(f"\\quad\\quad & six held-out & {ps_} & {S.ci(s_ + '.ci_boot_patient_seed')} (patient by seed) "
                        "\\\\"))
        if Am is not None:
            S.at(FA, Am)
            a_ = f'estimates["{mid}"]'
            ps_ = ", ".join(S.num(f"{a_}.per_seed_differences_mm[{i}]") for i in range(3))
            L.append(S.line(f"\\quad\\quad & 30 AMOS CTs & {ps_} & {S.ci(a_ + '.crossed_case_by_seed_ci95')} "
                            "(crossed) \\\\"))
    L += _table_foot() + [""]
    return write(out, "s11_combination_seeds.tex", "\n".join(L))


# --------------------------------------------------------------------------------- Q14 (ii) and the cross-checks
Q14_RUNGS = os.path.join(ROOT, "outputs", "v2_1", "cpu_repro", "ladder", "agreement_rungs.json")
Q14_TOL = {"deterministic_episode_mm": 0.01, "neural_patient_mm": 0.1, "neural_episode_mm": 1.0,
           "neural_episode_frac": 0.95}
Q14_GIT = "6bc6ac644362d0f92d7e03c1a03418e90f534272"   # tag v2.1.2
_Q14II = {}


def _q14_ii():
    """Q14 (ii) (cached): the CPU evaluations of the archived MPS rungs, from agreement_rungs.json (run at tag
    v2.1.2), re-derived here from the five CPU rung files and the archived v2 ladder files, episode by episode and
    patient by patient, within 1e-9. None while the record does not exist."""
    if _Q14II:
        return _Q14II
    if not os.path.exists(Q14_RUNGS):
        return None
    _guard(Q14_RUNGS)
    d = json.load(open(Q14_RUNGS))
    if _status_of(d) != STATUS_V21 or not str(_dig(d, "plan", PI.key("plan_sha256"))).startswith(PLAN_V21_SHA12):
        raise SystemExit("agreement_rungs.json: unexpected status or v2.1 plan sha256")
    if d["tolerances"] != Q14_TOL:
        raise SystemExit(f"agreement_rungs.json tolerances {d['tolerances']} differ from those of the v2.1 plan, "
                         f"{Q14_TOL}")
    g = _dig(d, "provenance", "git") or {}
    if g.get("hash") != Q14_GIT or g.get("dirty") is not False:
        raise SystemExit("agreement_rungs.json: not run at tag v2.1.2 from a clean tree")
    seen = set()
    for pr in d["pairs"]:
        cpu_p, ref_p = _out_path(pr["cpu"]["path"]), _out_path(pr["reference"]["path"])
        for p_, m_ in ((cpu_p, pr["cpu"]["md5"]), (ref_p, pr["reference"]["md5"])):
            _guard(p_)
            if hashlib.md5(open(p_, "rb").read()).hexdigest() != m_:
                raise SystemExit(f"Q14 (ii): {rel(p_)} differs from the file agreement_rungs.json paired")
        cj, rj = json.load(open(cpu_p)), json.load(open(ref_p))
        tol_ep = Q14_TOL["deterministic_episode_mm"] if pr["kind"] == "deterministic" else Q14_TOL["neural_episode_mm"]
        for cb in cj["results"]:
            if cb.get("eval_randomize", "off") != "off":
                continue
            key = f"{pr['cell']}:{cb['weights']}:contact_{cb['contact']}"
            if key not in d["checks"]:
                continue
            rb = [r for r in rj["results"] if r.get("weights") == cb["weights"] and r.get("contact") == cb["contact"]
                  and r.get("eval_randomize", "off") == "off"]
            if len(rb) != 1 or _fingerprint(cb["episodes"]) != _fingerprint(rb[0]["episodes"]):
                raise SystemExit(f"Q14 (ii) {key}: no single archived block on the same episode setups")
            de = [abs(float(a["final_pos"]) - float(b["final_pos"])) for a, b in zip(cb["episodes"], rb[0]["episodes"])]
            pa, pb = collections.defaultdict(list), collections.defaultdict(list)
            for a, b in zip(cb["episodes"], rb[0]["episodes"]):
                pa[a["patient"]].append(float(a["final_pos"]))
                pb[b["patient"]].append(float(b["final_pos"]))
            dp = max(abs(np.mean(pa[k]) - np.mean(pb[k])) for k in pa)
            fr = sum(x <= tol_ep for x in de) / len(de)
            c_ = d["checks"][key]
            if (abs(max(de) - c_["largest_difference"]) > 1e-9 or abs(dp - c_["largest_patient_mm"]) > 1e-9
                    or abs(fr - c_["frac_within_episode"]) > 1e-9 or c_["kind"] != pr["kind"]):
                raise SystemExit(f"Q14 (ii) {key}: the raw files do not reproduce agreement_rungs.json")
            seen.add(key)
    if seen != set(d["checks"]):
        raise SystemExit(f"Q14 (ii): checks not re-derived: {sorted(set(d['checks']) - seen)}")
    neu = {k: v for k, v in d["checks"].items() if v["kind"] == "neural"}
    det = {k: v for k, v in d["checks"].items() if v["kind"] == "deterministic"}
    X = max(v["largest_patient_mm"] for v in neu.values())
    if abs(X - d["largest_patient_mm_neural"]) > 1e-12 or d["sentence_key"] != "outside" or d["passed"] is not False:
        raise SystemExit("Q14 (ii): the record's sentence key or largest neural per-patient value changed")
    _Q14II.update({"d": d, "X": X, "Y": Q14_TOL["neural_patient_mm"], "neu": neu, "det": det,
                   "det_max": max(v["largest_difference"] for v in det.values()),
                   "file": "outputs/v2_1/cpu_repro/ladder/agreement_rungs.json"})
    log(f"Q14 (ii): deterministic rungs largest {_Q14II['det_max']:.3g} mm over {len(det)} blocks; learned-mask base "
        + "; ".join(f"{k.split(':')[-1]} per patient {v['largest_patient_mm']:.4f} mm, {100 * v['frac_within_episode']:.1f}% "
                    f"of episodes within 1 mm" for k, v in neu.items()) + f"; outside (X {X:.4f} against 0.10 mm)")
    return _Q14II


Q8_ALL, Q8_V2 = "q8_gap_cause_all_conditions.json", "q8_gap_cause_v2.json"


def _q8_all():
    """The Q8 output with every condition rescored, asserted key by key against q8_gap_cause_v2.json
    for everything the rescoring of C2 to C6 cannot change: classification, both families,
    the reference means, the C0 and C1 level sets, all of Q9 and printed_sentences [0..8]; v2 [9..22] are
    all_conditions [14..27] and [9..13] are the new C2 to C6 level-set lines. Older src comments cite
    the v2 file, which is unchanged."""
    if "q8" in _A345:
        return _A345["q8"]
    a, b = (json.load(open(os.path.join(AP_DIR, x))) for x in (Q8_ALL, Q8_V2))
    for d_, n_ in ((a, Q8_ALL), (b, Q8_V2)):
        if _status_of(d_) != STATUS_V21:
            raise SystemExit(f"{n_}: unexpected analysis_status")
    for k in ("classification", "family_Q8_P", "family_Q8_S", "reference_means"):
        if a["q8"][k] != b["q8"][k]:
            raise SystemExit(f"{Q8_ALL}: q8.{k} differs from {Q8_V2}")
    if a["q9"] != b["q9"] or any(a["q8"]["levelsets"][c] != b["q8"]["levelsets"][c] for c in ("C0", "C1")):
        raise SystemExit(f"{Q8_ALL}: Q9 or the C0 and C1 level sets differ from {Q8_V2}")
    pa, pb = a["printed_sentences"], b["printed_sentences"]
    if pa[:9] != pb[:9] or pa[14:] != pb[9:] or len(pa) != len(pb) + 5:
        raise SystemExit(f"{Q8_ALL}: the printed-sentence index map [0..8], v2 [9..22] -> [14..27] does not hold")
    if [x["slot"] for x in pa[9:14]] != [f"Q8 level set C{i}" for i in range(2, 7)]:
        raise SystemExit(f"{Q8_ALL}: printed_sentences[9..13] are not the C2 to C6 level-set lines")
    for k in ("Q8_class", "Q8_class_decided_without_rescoring", "Q8_P", "Q8_S", "Q9_A"):
        if a["decisions"][k] != b["decisions"][k]:
            raise SystemExit(f"{Q8_ALL}: decisions.{k} differs from {Q8_V2}")
    r = a["q8"]["rescoring"]
    if r["conditions_rescored"] != [f"C{i}" for i in range(7)] or r["conditions_incomplete"]:
        raise SystemExit(f"{Q8_ALL}: not every condition is rescored")
    dates = []
    for f_ in r["files"]:
        d_ = json.load(open(_out_path(os.path.join(ROOT, f_["path"]))))
        dates.append(str(d_["provenance"]["created_utc"])[:10])
    _A345["q8"] = a
    _A345["q8_dates"] = dates
    return a


def _q8_look():
    """The Q8 analysis whose primary look-ahead bound is the mean over sampling seeds 0 to 2 (the runner job
    v21:q8:analyze:look_seeds), cached. Pinned by sha256 (the records ledger holds it too), its default-path
    check record (ledgered) passed, and asserted against _q8_all() for every part the seeds cannot change, as
    q8_look_parts_same.py of the late analyses checks (run read-only: every part equal): Q9, the level sets,
    the rescoring, the secondary look-ahead, the reference means but the bound, the other two Q8-S members (all but
    their Holm-tied fields) and the reference keys of the settings checks and descriptive means. Its seed set is
    [0, 1, 2], the bound is the mean of the three seed means, and the class and decisions are unchanged."""
    if "q8_look" in _A345:
        return _A345["q8_look"]
    d, a = _late(LATE_Q8), _q8_all()
    if _late(LATE_Q8_CHECK).get("passed") is not True:
        raise SystemExit(f"{LATE_Q8_CHECK}: the default-path check did not pass")
    q, r = d["q8"], a["q8"]
    probs = []
    if q["classification"].get("bound_seeds_used") != [0, 1, 2]:
        probs.append("classification.bound_seeds_used is not [0, 1, 2]")
    for k in ("Q8_class", "Q8_class_decided_without_rescoring", "Q8_P", "Q8_S", "Q9_A"):
        if d["decisions"][k] != a["decisions"][k]:
            probs.append(f"decisions.{k} differs from {Q8_ALL}")
    if d["q9"] != a["q9"]:
        probs.append("q9 differs")
    for k in ("levelsets", "rescoring", "secondary_lookahead"):
        if q[k] != r[k]:
            probs.append(f"q8.{k} differs")
    for k, v in r["reference_means"].items():
        if k != "bound_C1_12_mean_mm" and q["reference_means"][k] != v:
            probs.append(f"q8.reference_means.{k} differs")
    for blk_ in ("lookahead_settings_checks", "lookahead_descriptive_means_mm"):
        if any(q[blk_].get(k) != v for k, v in r[blk_].items()):
            probs.append(f"q8.{blk_} differs on a reference key")
    tied = ("p_holm", "holds", "decision", "p_decision")
    for m in ("bound_C2_minus_servo_12", "bound_C1_minus_servo_36"):
        x, y = q["family_Q8_S"]["members"][m], r["family_Q8_S"]["members"][m]
        if {k: v for k, v in x.items() if k not in tied} != {k: v for k, v in y.items() if k not in tied}:
            probs.append(f"q8.family_Q8_S.members.{m} differs")
    lm = q["lookahead_descriptive_means_mm"]
    if abs(float(np.mean([lm[f"bound_C1_s{i}"] for i in range(3)])) - q["reference_means"]["bound_C1_12_mean_mm"]) > 1e-9:
        probs.append("the bound is not the mean of the three seed means")
    if probs:
        raise SystemExit(f"{LATE_Q8}: " + "; ".join(probs))
    log(f"[q8 seeds 0 to 2] {LATE_Q8} (sha256 {LATE_PINNED[LATE_Q8][:12]}): bound {q['reference_means']['bound_C1_12_mean_mm']:.4f} "
        f"mm (seed 0 alone {r['reference_means']['bound_C1_12_mean_mm']:.4f}); Q8-P "
        f"{q['family_Q8_P']['members']['bound_C1_minus_servo_12']['mean']:+.4f}, class {d['decisions']['Q8_class']}")
    _A345["q8_look"] = d
    return d


# ------------------------------------------------------------------------------- Q8-G1 (tag v2.1.4.1)
# The Q8-G1 reader, committed before the start-12 merge so that the reading path is fixed before any Q8-G1 output
# exists. It reads only the analysis record (ledgered after the run) and the two dated records it cites (both
# ledgered before the merge). The record's provenance names the v2.1.4.1 commit, which plan_stats' successor record
# does not cover; this reader accepts exactly that commit and nothing else.
Q8G1_REL = "outputs/v2_1/q8_g1/q8_g1_biplane.json"
Q8G1_COMMIT = "4a96c5caf3148323be26fa7056b0c6405b018020"     # tag v2.1.4.1
Q8G1_READ_REC = PP.Q8G1_READ_REL if PP else None      # plan-tree records (development repository only)
Q8G1_CHECK_REL = PP.Q8G1_CHECK_REL if PP else None
# (key of the record that cites the file, neutral id of the file)
Q8G1_RECORDS = ((PI.key("q8g1_read_record"), "q8g1 reading record"), (PI.key("q8g1_check_record"), "q8g1 gate record"))
Q8G1_STATUS_LABEL = _T("q8g1 status")


def _q8_g1():
    """The Q8-G1 analysis record (cached), checked before any field is used: its name, status strings and label; its
    code state (the v2.1.4.1 commit, clean); the record, the reading and the gate record in the records ledger committed
    at HEAD, with the record citing the same reading and gate files byte for byte; the printed form and search the
    reading fixes; and its single-plane count equal to the printed Q8 analysis (C1, 5 mm). Returns the record; the
    sentence is rec['printed_sentence']['text']. SystemExit names the first problem; no field is read around it."""
    if "q8_g1" in _A345:
        return _A345["q8_g1"]
    p = os.path.join(ROOT, Q8G1_REL)
    why = _hard_refusal(p)
    if why or not os.path.isfile(p):
        raise SystemExit(f"{Q8G1_REL}: {why or 'does not exist (Q8-G1 not completed)'}")
    rec = json.load(open(p))
    probs = []
    if rec.get("record") != "Q8-G1 analysis: biplane gated extra of Q8" or rec.get("question") != "Q8-G1":
        probs.append("not the Q8-G1 analysis record")
    if _status_of(rec) != STATUS_V21 or rec.get("status_label") != Q8G1_STATUS_LABEL or rec.get("smoke"):
        probs.append("status string, status label or smoke flag differ")
    g = (rec.get("provenance") or {}).get("git") or {}
    if g.get("hash") != Q8G1_COMMIT or g.get("dirty") is not False:
        probs.append(f"code state {str(g.get('hash'))[:12]} dirty={g.get('dirty')} is not the clean v2.1.4.1 commit")
    if not _in_ledger(Q8G1_REL, _sha256(p)):
        probs.append(f"{Q8G1_REL} is not in {_ledger_name(Q8G1_REL)}")
    shas = {rid: PI.record_digest(rid, root=ROOT) for _, rid in Q8G1_RECORDS}
    for _, rid in Q8G1_RECORDS:
        if not _in_ledger(rid, shas[rid]):
            probs.append(f"{_rec_name(rid)} is not in {_ledger_name(rid)}")
    for key, rid in Q8G1_RECORDS:
        cited = rec.get(key) or {}
        # the record cites each file by its absolute path in the development repository too (realpath-compared there)
        path_ok = (PI.skip_dev("the absolute paths that one analysis record cites (their sha256 is checked)")
                   or os.path.realpath(cited.get("path", "")) == os.path.realpath(os.path.join(ROOT, _rec_name(rid))))
        if not path_ok or shas[rid] is None or cited.get("sha256") != shas[rid]:
            probs.append(f"the record's {key} is not {_rec_name(rid)} as committed")
    sent, rd = rec.get("printed_sentence") or {}, rec.get(PI.key("q8g1_read_record")) or {}
    if sent.get("form") != rd.get("form") or (rec.get("search") or {}).get("biplane") != rd.get("search"):
        probs.append("printed form or search differ from the reading record")
    one = ((rec.get("single_plane_levelset_d5") or {}).get("A_patients_ambiguous"),
           (rec.get("single_plane_levelset_d5") or {}).get("n_targets_ambiguous"))
    q8 = _q8_all()["q8"]["levelsets"]["C1"]["rescored"]["5.0"]
    if one != (q8["A_patients_ambiguous"], q8["n_targets_ambiguous"]):
        probs.append(f"single-plane count {one} differs from the printed Q8 analysis")
    if probs:
        raise SystemExit(f"{Q8G1_REL}: " + "; ".join(probs))
    log(f"[q8_g1] {rel(p)} (sha256 {_sha256(p)[:12]}, v2.1.4.1 {Q8G1_COMMIT[:12]}): {sent['text']}")
    _A345["q8_g1"] = rec
    return rec


# Use: the Q8-G1 sentence of Section 6.5 or App A.2 prints _q8_g1()["printed_sentence"]["text"] verbatim, labeled
# exploratory; the default-path check prints from _q8_g1()["default_path_check"] (passed, the two largest differences)
# in the deviation row. Until the record exists, the build keeps the "not run" fill (no sentence).
# Test (development test): a synthetic record in a tmp ROOT with a fake ledger passes; each listed problem refuses.


def _plug_rows_check():
    """figures/plug_rows.json must still be the snapshot amendment 5's rule outcome refers to (md5), and the reward
    slot's SSIM word must equal amend5_combination.json final_table5.ssim_decision (amendment 5)."""
    pr = os.path.join(FIG_OUT, "plug_rows.json")
    md = hashlib.md5(open(pr, "rb").read()).hexdigest()
    want = PLUG_ROWS_MD5_PUBLIC if SD.public() else PLUG_ROWS_MD5
    if md != want:
        raise SystemExit(f"figures/plug_rows.json md5 {md} differs from {want}: stop and report")
    J = json.load(open(pr))
    w = next(r["decision"] for SL in J["slots"] if SL["key"] == "reward" for r in SL["rows"] if r["id"] == "ssim")
    if _a345_ready("amend5_combination.json"):
        sd = _a345("amend5_combination.json")["final_table5"]["ssim_decision"]
        if w != sd:
            raise SystemExit(f"Table 5 SSIM word {w!r} differs from amend5_combination.json final_table5 {sd!r}: the "
                             "changed-word preface of amendment 5 must print")
    log(f"plug_rows.json md5 {md} (the amendment 5 snapshot); Table 5 SSIM word {w!r}")
    return w


# --------------------------------------------------------------------------------- Table 7 and S8.2 rows
def _a345_check(rel_):
    p = os.path.join(ROOT, rel_)
    _guard(p)
    d = json.load(open(p))
    return d


PH_LOWER = {"physics": "physics renderer", "lotus_orig": "original LOTUS code", "simus": "SIMUS"}
# The identity, parity and recompute rows print in tab:checks (S8.2) only (the arm A and amendment 5 record rows
# there): with them the float of Table 7 does not fit its page.
CS_IDENTITY_ROWS = False


def _cs_rows_a345():
    """The new rows of tab:checks-summary (body): (group, check, guards, rule, outcome, met, plan section, src)."""
    rows = []
    if _a345_ready("amend3_arm_b.json"):
        F = "amend3_arm_b.json"
        B = _a345(F)
        S = _S(F, B)
        mem = B["result"]["members"]
        lo = min(mem, key=lambda k: mem[k]["skin_d"]["mean"])
        hi = max(mem, key=lambda k: mem[k]["skin_d"]["mean"])
        plo = min(mem, key=lambda k: mem[k]["skin_d"]["p_holm"])
        phi = max(mem, key=lambda k: mem[k]["skin_d"]["p_holm"])
        n_hold = sum(m["skin_d"]["key"] == "holds" for m in mem.values())
        n_eq = sum(m["skin_e"]["key"] == "holds" for m in mem.values())
        if n_hold or n_eq:
            raise SystemExit("arm B: a SKIN-D or SKIN-E member holds; rewrite the skin-tangent row")
        both = ("ct_base", "lm_base")
        if any(mem[k]["skin_d"]["sign_k"] != mem[k]["skin_d"]["sign_n"] or mem[k]["skin_d"]["mean"] <= 0 for k in both):
            raise SystemExit("arm B: a servo base does not rise in every patient; rewrite the skin-tangent row")
        out_b = (f"No level of the oracle ladder changes detectably ({S.num(f'result.members.{lo}.skin_d.mean')} to "
                 f"{S.num(f'result.members.{hi}.skin_d.mean')}~mm, Holm $p$ {S.p(f'result.members.{plo}.skin_d.p_holm')} "
                 f"to {S.p(f'result.members.{phi}.skin_d.p_holm')}); equivalence within $\\pm 1$~mm is not shown. The "
                 # "rise by" is the SKIN-D holds form of amendment 3; both members show no detectable change, so the
                 # row prints the estimate form, and "differ" keeps one meaning in the row (the criterion cell "0 of 5
                 # differ")
                 "CT-mask and learned-mask servo-base differences are "
                 + " and ".join(S.num(f"result.members.{k}.skin_d.mean") for k in both)
                 + "~mm, "
                 + (lambda ks_: ks_[0] + " each" if len(set(ks_)) == 1 else " and ".join(ks_))(
                     [f"{S.int(f'result.members.{k}.skin_d.sign_k')}/{S.int(f'result.members.{k}.skin_d.sign_n')}"
                      for k in both])
                 + ". The plane lies near the target window (median angle "
                 f"{S.plain('result.plane.static.angle_to_target_beam_deg.median')}$^\\circ$)")
        # the arm's MDD and the pose oracle's loss of success at 5 mm and 5 deg, its largest secondary change
        # (Supplementary Table S6.2)
        sec_ = B["result"]["secondary"][26]
        if sec_["row"] != "pose_oracle success_5mm_5deg skin minus target plane":
            raise SystemExit(f"{F}: result.secondary[26] is {sec_['row']!r}, not the pose oracle's 5 mm, 5 deg success")
        # the MDD names its basis (the median SD of the arm's five rungs, result.mdd["SKIN-D"].median_sd_mm), and the
        # success change prints in percentage points, as every other success value of the body (unit conversion of
        # result.secondary[26].mean)
        pts_ = S.rec("result.secondary[26].mean", sec_["mean"], _plan_stats().fmt_num(100.0 * float(sec_["mean"]), 0, True))
        out_b += (f". MDD {S.plain('result.mdd[\"SKIN-D\"].alpha_005_mm')}~mm at the median SD; pose-oracle success "
                  f"at 5~mm and 5$^{{\\circ}}$, ${pts_}$ points (secondary)")
        rows.append(("Controls", "Skin-tangent contact", "Do the results depend on the target-anchored contact plane?", "",
                     out_b, f"0 of {len(mem)} differ", _T("rule src a3 b"), " | ".join(S.e)))
    if _a345_ready("amend3_arm_c.json"):
        F = "amend3_arm_c.json"
        C_ = _a345(F)
        S = _S(F, C_)
        R_ = C_["result"]["renderers"]
        kk = ", ".join(f"{PH_LOWER[r]} {S.int(f'result.renderers.{r}.k')} "
                       f"of {S.int(f'result.renderers.{r}.K')}" for r in PHANTOM_RENDERER)
        uu = _and([S.int(f"result.renderers.{r}.u") for r in PHANTOM_RENDERER])
        outs = []
        for r in PHANTOM_RENDERER:
            for i, b in enumerate(R_[r]["bands"]):
                if b["inside"] is False:
                    outs.append((r, i, b["band"]))
        cnr = [(r, i) for r, i, b in outs if b == "cnr_over_h_snr_at_least_0_90"]
        shd = [(r, i) for r, i, b in outs if b == "shadow_B_at_least_10mm"]
        if len(cnr) + len(shd) != len(outs):
            raise SystemExit("phantoms: a band outside other than the CNR ratio and the B shadow; rewrite the row")
        out_c = (f"Bands met: {kk} ({uu} not scorable). Outside: the disc CNR over the speckle SNR, "
                 + " and ".join(S.plain(f"result.renderers.{r}.bands[{i}].value") for r, i in cnr)
                 + " (band at least 0.90)"
                 + "".join(f"; the SIMUS shadow below the bone slab, {S.plain(f'result.renderers.{r}.bands[{i}].value')}~mm "
                           "(band at least 10~mm)" for r, i in shd)
                 + ". Several bands are design checks")
        # the output speckle SNR of the three renderers against the Rayleigh line of Table S1.5 (amend3_arm_c.json
        # result.metric_table.<renderer>.snr.mean, result.rayleigh_reference)
        snr_ = ("Output speckle SNR " + _and([S.plain(f"result.metric_table.{r}.snr.mean") for r in PHANTOM_RENDERER])
                + f", in that order (Rayleigh {S.plain('result.rayleigh_reference')})")
        if not out_c.endswith(". Several bands are design checks"):
            raise SystemExit("phantoms: the Table 6 cell no longer ends with its design-check sentence")
        out_c = out_c[:-len("Several bands are design checks")] + snr_ + ". Several bands are design checks"
        S.note(f"bands 0.90 and 10 mm: {_T('src a3 bands')} and Appendix bands (cnr_over_h_snr_min, shadow_mm)")
        rows.append(("Repeatability", "Phantom bands", "Do the renderers' properties lie in their designed ranges?", "",
                     out_c, "descriptive", _T("rule src a3 c"), " | ".join(S.e)))
    if CS_IDENTITY_ROWS and _a345_ready("amend3_arm_a.json"):
        F = "amend3_arm_a.json"
        A = _a345(F)
        S = _S(F, A)
        hc = A["result"]["handled_checks"]
        if A["result"]["handled_failed"] or not all(v["passed"] for v in hc.values()):
            raise SystemExit("arm A: a handled check failed; rewrite the identity row")
        mx = max(v["largest_difference"] for v in hc.values())
        S.e.append(f"run:max(result.handled_checks[*].largest_difference) = {json.dumps(mx)} => "
                   f"{_plan_stats().fmt_num(mx, 1, False)}")
        out_i = (f"Reference identity and parity per stage and seed: {sum(v['passed'] for v in hc.values())} of "
                 f"{len(hc)} checks pass, largest difference {_plan_stats().fmt_num(mx, 1, False)}~mm")
        rows.append(("Repeatability", "Learned-mask arm identity and parity", "Learners scored against another "
                     "reference or on another device", "", out_i, "yes", _T("rule src a3 checks"), " | ".join(S.e)))
    if CS_IDENTITY_ROWS and _a345_ready("amend5_combination.json") and _a345_ready("amend5_amos.json"):
        F = "amend5_combination.json"
        Cb = _a345(F)
        S = _S(F, Cb)
        ids = []
        for s_ in ("six", "cohort", "armh", "amos"):
            r_ = f"outputs/v2_1/amend5/checks/identity_{s_}.json"
            d_ = _a345_check(r_)
            if d_.get("passed") is not True:
                raise SystemExit(f"{r_}: not passed")
            ids += [v["largest_difference"] for v in d_["checks"].values() if isinstance(v, dict) and v.get("unit") == "mm"]
            S.e.append(f"{r_}:passed = true => word")
        mx = max(ids)
        S.e.append(f"run:max(identity_*.json checks[*].largest_difference) = {json.dumps(mx)} => "
                   f"{_plan_stats().fmt_num(mx, 1, False)}")
        out_r = (f"Imitation starts equal to the archived ones in the six, cohort, rerun and AMOS sets (largest "
                 f"difference {_plan_stats().fmt_num(mx, 1, False)}~mm); references recomputed within "
                 f"{S.num('reference_recompute.cohort_vs_q2r_max_abs', 1, False, False)}~mm on the cohort and "
                 f"{S.num('reference_recompute.six_vs_table4_max_abs_mm', 4, False, False)}~mm on the six")
        if not Cb["reference_recompute"]["passed"]:
            raise SystemExit("amendment 5 reference recompute did not pass")
        rows.append(("Repeatability", "Combination identity and recompute", "Starts or references that differ from "
                     "the archived ones", "", out_r, "yes", _T("rule src a5 checks"), " | ".join(S.e)))
    if _a345_ready("amos_q2r.json"):
        F = "amos_q2r.json"
        A = _a345(F)
        S = _S(F, A)
        P_ = 'families["AMOS-P"].members'
        dch = A["design_checks"]
        if dch["identical_episodes"]["cell_mismatches"] or dch["windows"]["mismatches"] or \
                A["frozen_identity"]["artifacts"]["n_changed_or_missing"]:
            raise SystemExit("amos_q2r.json: a design check failed; rewrite the AMOS row")
        c3 = S.word(f"{P_}.C3.key", {"holds": "holds"})
        ok = all(A["decisions"]["AMOS-P"][k]["key"] == "holds" for k in ("C1", "C2", "C3"))
        out_a = (f"Learned minus CT masks, physics: {S.num(P_ + '.C1.mean')} {S.ci(P_ + '.C1.ci_t')}, "
                 f"{S.int(P_ + '.C1.sign_k')}/{S.int(P_ + '.C1.sign_n')}, against {S.num('original_six.values.C1')} on "
                 f"the six. SonoSPADE minus physics, learned-mask servo base: {S.num(P_ + '.C2.mean')} "
                 f"{S.ci(P_ + '.C2.ci_t')}, {S.int(P_ + '.C2.sign_k')}/{S.int(P_ + '.C2.sign_n')}, against "
                 f"{S.num('original_six.values.C2')}. RL change within 1~mm in all "
                 f"{len(A['C3_intersection_union']['arms'])} configurations; the ordering "
                 f"{S.word('ordering.class', {'recurs': 'recurs'})}. Simulation only")
        # the design-check clause is computed and logged, not printed
        dc_txt = (f"{S.int('design_checks.identical_episodes.cell_mismatches')} episode and "
                  f"{S.int('design_checks.windows.mismatches')} window mismatches; "
                  f"{S.int('frozen_identity.artifacts.n')} frozen artifacts unchanged")
        log(f"30 external CTs, design checks ({_T('not printed tag')}): {dc_txt}")
        del c3
        rows.append(("Replication", "30 external CTs", "Do the main contrasts recur on a second public CT dataset?", "",
                     out_a, "yes" if ok else "no", _T("rule src a4"), " | ".join(S.e)))
    return rows


def _leak_rerun():
    """The leakage reruns of step 3 (tag v2.1.3), from the check records: -> (outcome clause, src) or None."""
    if not (_a345_ready("amend3_arm_a.json") and _a345_ready("amend5_combination.json")):
        return None
    recs = [("outputs/v2_1/amend3/checks/arm_a.json", "checks.leakage"),
            ("outputs/v2_1/amend3/checks/arm_b_post.json", "checks.leakage"),
            ("outputs/v2_1/amend5/checks/identity_six.json", "checks.six_leakage_gate"),
            ("outputs/v2_1/amend5/checks/identity_armh.json", "checks.six_leakage_gate")]
    src = []
    for r_, kp in recs:
        d_ = _a345_check(r_)
        g = _kp_get(d_, kp)
        if d_.get("passed") is not True or g.get("passed") is not True or g.get("n_leak") != 0:
            raise SystemExit(f"{r_}: {kp} did not pass with 0 leaks")
        if not str(_dig(d_, "code_at_run", "tag") or "v2.1.3").startswith("v2.1.3"):
            raise SystemExit(f"{r_}: not run at v2.1.3")
        src.append(f"{r_}:{kp}.n_leak = 0 => 0 | {r_}:provenance.time_utc = {json.dumps(d_['provenance']['time_utc'])} "
                   "=> word")
    for r_ in ("outputs/v2_1/amend3/checks/leakage_arm_a.1792928217.json",
               "outputs/v2_1/amend3/checks/leakage_arm_b.1792928297.json"):
        d_ = _a345_check(r_)
        if not d_["findings"] or any(not str(f_).startswith("STALE") for f_ in d_["findings"]):
            raise SystemExit(f"{r_}: the kept first attempt is not STALE findings only")
        src.append(f"note: {r_} (kept first attempt): {len(d_['findings'])} findings, all STALE, no LEAK")
    src.append(f"note: {_T('v213 record file')}{_T('v213 parts')} (the four failed checks; 30 SCHEMA findings of the "
               "two amendment 5 gates); leakage_rule of arm_a.json (v2.1.3 reading)")
    return ("; four records of amendments~3 and 5 failed once on their leakage gates, with no leak, and passed on one "
            "rerun at tag v2.1.3", " | ".join(src))


def _checks_rows_a345():
    """The rows of tab:checks (S8.2) for the check records of amendments 3 to 5: (check, guards, rule, outcome,
    source, src)."""
    rows = []
    if _a345_ready("amend3_arm_a.json"):
        d_ = _a345_check("outputs/v2_1/amend3/checks/arm_a.json")
        ch = d_["checks"]
        ident = [k for k in ch if k.startswith(("reference_identity", "parity"))]
        rows.append((_T("row a3 a"),
                     "Learner rows read before their gates, outputs, seeds, status strings, model md5s, leakage gate, "
                     "reference identity and audit are checked.",
                     "Every item passes before arm A is read; the record is committed to the records ledger first.",
                     f"Passed: {len(ch)} items; {ch['outputs_present']['n']} of {ch['outputs_present']['n_expected']} "
                     f"outputs; reference identity and parity {sum(ch[k]['passed'] for k in ident)} of {len(ident)} "
                     f"at {max(ch[k]['largest_difference'] for k in ident):.1f}~mm; leakage {ch['leakage']['n_leak']} "
                     f"leaks, {ch['leakage']['n_findings']} STALE findings outside the lineage listed. The first "
                     "attempt failed on the leakage item alone; one rerun at tag v2.1.3 passed" + _T("tag rerun note") + ".",
                     "amend3/checks/arm_a.json",
                     "outputs/v2_1/amend3/checks/arm_a.json passed, checks.* (outputs_present, reference_identity:*, "
                     "parity:*, leakage.n_leak, leakage.n_findings), leakage_rule; first attempt "
                     "outputs/v2_1/amend3/checks/leakage_arm_a.1792928217.json"))
    if _a345_ready("amend3_arm_b.json"):
        a_ = _a345_check("outputs/v2_1/amend3/checks/arm_b.json")["checks"]
        b_ = _a345_check("outputs/v2_1/amend3/checks/arm_b_post.json")["checks"]
        dflt = [k for k in a_ if k.startswith("default:")]
        cal = [k for k in a_ if k.startswith("calibration:")]
        pu = [k for k in a_ if k.startswith("plane_unit:")]
        rows.append((_T("row a3 b"),
                     "A contact surface that changes the default episodes, the servo calibrations or the setups.",
                     "Defaults identical to the archived rungs; calibrations identical; the plane's unit tests; "
                     "identical setups; leakage gate after the run.",
                     f"Passed. Defaults: {sum(a_[k]['passed'] for k in dflt)} of {len(dflt)} at "
                     f"{max(a_[k]['largest_difference'] for k in dflt):.1f}~mm; calibrations "
                     f"{sum(a_[k]['passed'] for k in cal)} of {len(cal)} identical; plane unit tests "
                     f"{sum(a_[k]['passed'] for k in pu)} of {len(pu)}, the normal test {_T('plane normal note')}; post "
                     f"record: leakage {b_['leakage']['n_leak']} leaks, first attempt failed on the leakage item "
                     "alone, one rerun at tag v2.1.3 passed" + _T("tag rerun note") + ".",
                     "amend3/checks/arm_b.json, arm_b_post.json",
                     "outputs/v2_1/amend3/checks/arm_b.json checks.default:*, calibration:*, plane_unit:* ("
                     + _T("plane normal src") + "on the normal test); arm_b_post.json checks.leakage.n_leak and rung:*; first attempt "
                     "leakage_arm_b.1792928297.json"))
    if _a345_ready("amend3_arm_c.json"):
        c_ = _a345_check("outputs/v2_1/amend3/checks/arm_c.json")["checks"]
        rows.append((_T("row a3 c"),
                     "Metrics or phantoms that do not measure what they claim.",
                     "Self-tests of the metrics on synthetic inputs; phantom geometry; renderer gates admitted; the "
                     "physics adapter bit-identical to the built-in renderer.",
                     f"Passed: {sum(v['passed'] for v in c_.values() if isinstance(v, dict))} of "
                     f"{sum(isinstance(v, dict) for v in c_.values())} items; Rayleigh SNR self-test "
                     f"{c_['rayleigh_snr']['value']:.3f} against 1.91; physics adapter "
                     f"{c_['physics_adapter_bit_identical']['largest_difference']:.1f} display units from the built-in "
                     "renderer.",
                     "amend3/checks/arm_c.json",
                     "outputs/v2_1/amend3/checks/arm_c.json checks.* (rayleigh_snr.value, "
                     "physics_adapter_bit_identical.largest_difference)"))
    if _a345_ready("amos_q2r.json"):
        A = _a345("amos_q2r.json")
        dch = A["design_checks"]
        rec = {r_: _a345_check(f"outputs/v2_1/{r_}") for r_ in ("amos/checks/amos_checks.json",
                                                                  "amos_x/checks/arm_x_checks.json",
                                                                  "amos/checks/read_order.json")}
        if not all(v.get("passed") is True for v in rec.values()):
            raise SystemExit("an AMOS check record did not pass")
        inv = A["invariance"]["max_abs_mm"]
        rows.append((_T("row a4"),
                     "AMOS rows read before the frozen manifest, the labels, the episodes or the frozen artifacts are "
                     "checked.",
                     "Check records pass and are committed before any AMOS output is read; identical episodes and "
                     "windows; frozen artifacts unchanged; the replication cohort recomputed.",
                     f"Passed: the AMOS, arm X and read-order records. Identical episodes: "
                     f"{dch['identical_episodes']['blocks_checked']} blocks, {dch['identical_episodes']['cell_mismatches']} "
                     f"mismatches; windows: {dch['windows']['episodes_checked']:,} episodes, {dch['windows']['mismatches']} "
                     f"mismatches; {A['frozen_identity']['artifacts']['n']} frozen artifacts, "
                     f"{A['frozen_identity']['artifacts']['n_changed_or_missing']} changed; cohort recompute "
                     f"{A['comparison']['reference_recompute']['largest_abs_diff_mm']:.1f}~mm over "
                     f"{A['comparison']['reference_recompute']['n_cases']} cases; CT-mask base invariance "
                     f"{inv:.1f}~mm, so its line does not print.",
                     "amos/checks/amos_checks.json, amos_x/checks/arm_x_checks.json",
                     "outputs/v2_1/amos/checks/amos_checks.json, outputs/v2_1/amos_x/checks/arm_x_checks.json, "
                     "outputs/v2_1/amos/checks/read_order.json passed; amos_q2r.json design_checks.identical_episodes, "
                     "design_checks.windows, frozen_identity.artifacts, comparison.reference_recompute, "
                     "invariance.max_abs_mm"))
    if _a345_ready("amend5_combination.json") and _a345_ready("amend5_amos.json"):
        Cb = _a345("amend5_combination.json")
        rec = _a345_check("outputs/v2_1/amend5/checks/amend5_checks.json")
        gate = _a345_check("outputs/v2_1/amend5/checks/gate_reward_terms.json")
        if rec.get("passed") is not True or not all(v["passed"] for v in rec["sets"].values()) or \
                gate.get("admitted") is not True:
            raise SystemExit("amend5_checks.json or the reward-term gate did not pass")
        idm = []
        for s_ in ("six", "cohort", "armh", "amos"):
            d_ = _a345_check(f"outputs/v2_1/amend5/checks/identity_{s_}.json")
            if d_.get("passed") is not True:
                raise SystemExit(f"identity_{s_}.json did not pass")
            idm += [v["largest_difference"] for v in d_["checks"].values() if isinstance(v, dict) and v.get("unit") == "mm"]
        SET_NAME = {"six": "six", "cohort": "cohort", "armh": "arm H", "amos": "AMOS"}
        rows.append((_T("row a5"),
                     "Combination rows read before their identity, seeds, audits, leakage gates and the reward gate "
                     "are checked.",
                     "Each set's checks pass and the append-only record is committed before the set is read.",
                     f"Passed: sets {_and([SET_NAME[x] for x in rec['sets']])}; {len(rec['versions'])} earlier versions "
                     f"kept; reward-term gate admitted; imitation starts identical in the four sets (largest "
                     f"difference {max(idm):.1f}~mm); references recomputed within "
                     f"{Cb['reference_recompute']['cohort_vs_q2r_max_abs']:.1f}~mm on the cohort and "
                     f"{Cb['reference_recompute']['six_vs_table4_max_abs_mm']:.4f}~mm on the six. The identity checks "
                     "of the six-patient and arm H sets failed once on their leakage gates alone; one rerun each at tag "
                     "v2.1.3 passed" + _T("identity rerun note") + ".",
                     "amend5/checks/amend5_checks.json",
                     "outputs/v2_1/amend5/checks/amend5_checks.json passed, sets.*.passed, versions; identity_{six,cohort,"
                     "armh,amos}.json checks[*].largest_difference (unit mm); "
                     "gate_reward_terms.json admitted; identity_six.json and identity_armh.json checks.six_leakage_gate; "
                     "amend5_combination.json reference_recompute"))
    return rows


def amend345_tables(out):
    """The tables of amendments 3 to 5 (each skipped, with a log line, while its amendment stays blinded). The rule-input
    table is written to a folder of the development repository, not to the article."""
    log("\n== amendments 3 to 5: tables (exploratory; each analysis ran under its plan after its records) ==")
    ws = [arm_a_tab, earlier_ctrl_tab, arm_b_tab, phantoms_tab, amos_yield_tab, amos_labels_tab, amos_ctanat_tab,
          amos_results_tab, combination_tab, s11_arm_a_secondary, s11_arm_b_secondary, s11_amos_cells,
          s11_amos_cases, s11_amos_secondary, s11_combination_seeds]
    written = [p for p in (w(out) for w in ws) if p]
    if PI.skip_dev("tables/rule_input.tex (not input by the article; it needs a whole block of a plan document)"):
        return written
    os.makedirs(PROCESS_OUT, exist_ok=True)
    p_ = rule_input_tab(PROCESS_OUT)   # not input by the article
    return written + ([p_] if p_ else [])


# =================================================================================== late results
# Post hoc and secondary analyses that finished last: the realism uncertainty, the DR landscape over
# segmenter seeds 0 to 2, the s0513 case study, the post hoc statistics layers (questions Q1, Q4, Q6, Q7),
# the mixed model and the Q8 look-ahead bound over sampling seeds 0 to 2. Each record is read
# once (cached) and refused unless it is the file that the records ledger committed at HEAD lists (the realism and DR records, both
# mixed-model files and the fit list), or, for a record the ledger does not hold (the case study, the statistics layers and the
# Q8 seed-mean analysis; the analysis runs append to the ledger, this build never does), unless its sha256 equals the
# one pinned below. Every table is written into the tables folder only (_out_ok).
LATE_REALISM_CI = "outputs/v2/posthoc/a11/realism_uncertainty.json"
LATE_DR_SEEDS = "outputs/v2/posthoc/a12/a12_dr_landscape_ci.json"
LATE_CASE_STUDY = "outputs/v2/posthoc/a13/s0513_case_study.json"
LATE_LAYERS = "outputs/v2/posthoc/posthoc_v2_layers.json"
LATE_MM = ("outputs/v2/posthoc/mixed_model_a14.json", "outputs/v2_1/analysis_plan/mixed_model_38.json")
LATE_FIT_LIST = PP.FIT_LIST_REL if PP else None   # a plan-tree record (development repository only)
FIT_LIST_ID = "mixed-model fit list"                       # its neutral id (plan_inputs.record_digest, ledger_has)
LATE_Q8 = "outputs/v2_1/analysis_plan/q8_gap_cause_look_s012.json"
LATE_Q8_CHECK = "outputs/v2_1/checks/q8_look_defpath/agreement.json"
LATE_SIGN_CHECK = "outputs/v2_1/analysis_plan/q10_label_controls_v3.json"
# sha256 pinned in code (computed from the files the scripts wrote), among them the sign-check record (the
# sign check printed by hand in Section 8.4); the records ledger holds the same sha256 for all four
LATE_PINNED = {LATE_CASE_STUDY: "51cbedd94a5fc3b7ea5ca3d80f84b0f38d9e7f303d3b2803a2f834f9fd0f1c26",
               LATE_LAYERS: "f0965958493cac764a67ec0f1e63cc8ae6731981f2f59b6b654f168a04ca902e",
               LATE_Q8: "003614d9aa071cd4a266dbd721cec3694ebefee48daeebaf84174a58887815d2",
               LATE_SIGN_CHECK: "f14755a199448bf5d5cf4c2a719700d3adac3a4392db86de26f0d1d3169fd74b"}
LATE_STATUS = {LATE_REALISM_CI: "post hoc, v2", LATE_DR_SEEDS: "post hoc, v2", LATE_CASE_STUDY: "post hoc, v2", LATE_LAYERS: "post hoc, v2",
               LATE_MM[0]: "post hoc, v2", LATE_MM[1]: STATUS_V21, LATE_Q8: STATUS_V21, LATE_Q8_CHECK: STATUS_V21,
               LATE_SIGN_CHECK: STATUS_V21}
LATE_LABEL = {LATE_REALISM_CI: "post hoc", LATE_DR_SEEDS: "post hoc", LATE_CASE_STUDY: "post hoc", LATE_LAYERS: "post hoc"}


def _late(rel_):
    """One late record (cached): no hard-refused path, the status string, the label word, and either its line in the
    records ledger committed at HEAD (same path) or its pinned sha256. SystemExit names the first problem."""
    key = "late:" + rel_
    if key in _A345:
        return _A345[key]
    p = os.path.join(ROOT, rel_)
    why = _hard_refusal(p)
    if why or not os.path.isfile(p):
        raise SystemExit(f"{rel_}: {why or 'does not exist'}")
    sha = _sha256(p)
    if rel_ in LATE_PINNED:
        # the pins name the development files' bytes; a public build checks the file against ROOT/MANIFEST.tsv
        want = _listed_sha256(rel_) if SD.public() else LATE_PINNED[rel_]
        if sha != want:
            raise SystemExit(f"{rel_}: sha256 {sha[:12]} differs from the pinned {str(want)[:12]}")
    else:
        if not _in_ledger(FIT_LIST_ID if rel_ == LATE_FIT_LIST else rel_, sha):
            raise SystemExit(f"{rel_} (sha256 {sha[:12]}) is not in {_ledger_name(rel_)}")
    if rel_.endswith(".json"):
        d = json.load(open(p))
        if rel_ in LATE_STATUS and _status_of(d) != LATE_STATUS[rel_]:
            raise SystemExit(f"{rel_}: analysis_status {_status_of(d)!r}, expected {LATE_STATUS[rel_]!r}")
        if rel_ in LATE_LABEL and LATE_LABEL[rel_] not in str(d.get("label", "")).lower():
            raise SystemExit(f"{rel_}: its label does not say {LATE_LABEL[rel_]!r}")
    else:
        d = open(p, encoding="utf-8").read()
    _A345[key] = d
    return d


def _listed_sha256(rel_):
    """The sha256 that ROOT/MANIFEST.tsv lists for a result file (public mode)."""
    import results_world
    sha = results_world.listed(rel_)
    if sha is None:
        raise SystemExit(f"{rel_}: not listed in MANIFEST.tsv of the result root, so its pinned record cannot be checked")
    return sha


def _fit_list():
    """The mixed-model fit list (a plan-tree record): (its text, or None in public mode, and its sha256), refused
    unless the committed ledger lists it."""
    if SD.public():
        sha = PI.record_digest(FIT_LIST_ID)
        if not sha or not _in_ledger(FIT_LIST_ID, sha):
            raise SystemExit(f"{FIT_LIST_ID}: not in the recorded records ledger")
        PI.skip_dev("the header line of the mixed-model fit list (the file's sha256 is checked)")
        return None, sha
    fl = _late(LATE_FIT_LIST)
    return fl, PI.record_digest(FIT_LIST_ID, root=ROOT)


def _late_head(src):
    return [f"%% NEW BIB KEYS: none\n% generated by {_T('make_tables script')} from {src}\n"
            "% do not edit by hand: rerun the script"]


# ================================================================================= realism uncertainty
REALISM_CI_ROWS = [("Baseline renderers", [("prior LOTUS", "LOTUS, authors' reimplementation"), ("NumPy", "NumPy renderer")]),
            ("USB Gym stages, default render", [("physics", "Physics (no network)"), ("physics+CUT", "Physics + CUT"),
                                                ("physics+CycleGAN", "Physics + CycleGAN"),
                                                ("physics+SonoSPADE", "Physics + SonoSPADE")]),
            ("USB Gym stages, one full DR draw per view", [("physics [DR full]", "Physics"),
                                                            ("physics+CUT [DR full]", "Physics + CUT"),
                                                            ("physics+CycleGAN [DR full]", "Physics + CycleGAN"),
                                                            ("physics+SonoSPADE [DR full]", "Physics + SonoSPADE")])]


def _a11():
    """realism_uncertainty.json, checked against realism_sector.json (the values of Table A.6, tab:realism)."""
    A = _late(LATE_REALISM_CI)
    if A["default_path_check"].get("passed") is not True or A["print_rule"].get("point_values_printed") != "archived":
        raise SystemExit(f"{LATE_REALISM_CI}: the default-path check did not pass or the print rule is not the archived one: "
                         "this reader prints only the archived point values")
    if A["reading"]["null_statistic"] != "fold-mean" or A["reading"]["bootstrap_B"] != 10000:
        raise SystemExit(f"{LATE_REALISM_CI}: null statistic or bootstrap size differ from the reading")
    v = json.load(open(os.path.join(V2, "realism", "realism_sector.json")))["variants"]
    listed = {k for _, g in REALISM_CI_ROWS for k, _ in g}
    if listed != set(A["bootstrap"]) or listed != set(v):
        raise SystemExit(f"{LATE_REALISM_CI}: bootstrap rows {sorted(A['bootstrap'])} differ from Table A.6's variants")
    for k, b in A["bootstrap"].items():
        if (abs(b["point"]["w1"] - v[k]["w1_in_sector"]) > 1e-12 or abs(b["point"]["kid"] - v[k]["kid"]) > 1e-12
                or b["B"] != 10000):
            raise SystemExit(f"{LATE_REALISM_CI}: bootstrap[{k}] point values differ from realism_sector.json")
    for k, n in A["null"].items():
        if (abs(n["observed_auc"] - v[k]["c2st_auc_mean"]) > 1e-12 or n["n_perm"] != 200
                or abs(n["p_perm"] - (1 + n["count_at_or_above"]) / (1 + n["n_perm"])) > 1e-15):
            raise SystemExit(f"{LATE_REALISM_CI}: null[{k}] differs from realism_sector.json or from the p rule")
    return A


REALISM_SECTOR_REL = "outputs/v2/realism/realism_sector.json"


def posthoc_realism_tab(out):
    A = _a11()
    S = _S(LATE_REALISM_CI, A)
    nulls = A["null"]
    # the rows without a null print their observed C2ST AUC (Table A.6 prints it; _a11 asserts that the null rows'
    # observed_auc equals realism_sector.json variants[*].c2st_auc_mean)
    v_sec = json.load(open(os.path.join(ROOT, REALISM_SECTOR_REL)))["variants"]
    L = _late_head(LATE_REALISM_CI + " (post hoc; frame bootstrap and C2ST label-permutation null) and "
                   + REALISM_SECTOR_REL + " (variants[*].c2st_auc_mean of the rows without a null)") + [
        "\\begin{table*}[tp]",
        "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
        "\\caption{Uncertainty of the distances to real frames of Table~\\ref{tab:realism} (post hoc). Point values are "
        "those of Table~\\ref{tab:realism}. W1 and KID: 95\\% percentile intervals of a bootstrap over frames ("
        f"{A['reading']['bootstrap_B']:,} resamples; simulated and real frames resampled independently). Resampling "
        "biases W1 upward where it is "
        "small, so a point value can lie near the lower end of its interval. Null: the classifier two-sample test "
        "with the domain labels permuted, "
        f"{A['reading']['n_perm']} permutations per stage. Each permutation is one five-fold "
        "run, so its spread is that of a five-fold mean and $p$ is expected to be conservative. "
        f"$p = (1 + \\#\\{{\\text{{null}} \\geq \\text{{observed}}\\}})/{1 + A['reading']['n_perm']}$, so "
        f"$1/{1 + A['reading']['n_perm']}$ is the smallest attainable. n/r: no null run. The established-simulator "
        "rows of Table~\\ref{tab:realism} have no intervals.}"
        f" % src: {LATE_REALISM_CI} reading.bootstrap_B {A['reading']['bootstrap_B']}, reading.n_perm {A['reading']['n_perm']}, "
        "reading.null_statistic (fold-mean), reading.null_statistic_description, reading.p_rule, reading.bootstrap_unit",
        "\\label{tab:posthoc-realism}",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l l l c c c@{}}",
        "\\toprule",
        "Variant & W1 [95\\% CI] & KID [95\\% CI] & C2ST AUC & Null AUC, mean (SD) & $p$ \\\\",
        "\\midrule"]
    log("\n== tab:posthoc-realism (posthoc_realism.tex), POST HOC ==")
    for gi, (head, items) in enumerate(REALISM_CI_ROWS):
        if gi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
        for k, lab in items:
            b = f'bootstrap["{k}"]'
            w1 = (S.plain(f"{b}.point.w1", 4) + " " + S.ci(f"{b}.w1.ci95", 4, signed=False))
            kid = (S.plain(f"{b}.point.kid", 4) + " " + S.ci(f"{b}.kid.ci95", 4, signed=False))
            if k in nulls:
                n_ = f'null["{k}"]'
                auc = S.plain(f"{n_}.observed_auc", 3)
                nm = S.plain(f"{n_}.null_mean", 3) + " (" + S.plain(f"{n_}.null_sd_ddof1", 3) + ")"
                cnt = S.int(f"{n_}.count_at_or_above")
                pp = f"${int(cnt) + 1}/{1 + nulls[k]['n_perm']}$"
                S.rec(f"{n_}.p_perm", nulls[k]["p_perm"], pp)
            else:
                auc = S.rec(f'variants["{k}"].c2st_auc_mean', v_sec[k]["c2st_auc_mean"],
                            _plan_stats().fmt_num(v_sec[k]["c2st_auc_mean"], 3, False), f=REALISM_SECTOR_REL)
                nm, pp = "n/r", "n/r"
            L.append(S.line(f"\\quad {lab} & {w1} & {kid} & {auc} & {nm} & {pp} \\\\"))
            log(f"{k:28s} W1 {A['bootstrap'][k]['point']['w1']:.4f} {A['bootstrap'][k]['w1']['ci95']} KID "
                f"{A['bootstrap'][k]['point']['kid']:.4f} {A['bootstrap'][k]['kid']['ci95']}"
                + (f" null {nulls[k]['null_mean']:.4f} ({nulls[k]['null_sd_ddof1']:.4f}) p {nulls[k]['p_perm']:.5f}"
                   if k in nulls else ""))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "posthoc_realism.tex", "\n".join(L))


# ================================================================================= DR landscape over seeds
DR_SEEDS_ARM = [("off", "Off"), ("shared", "Shared"), ("cross", "Cross"), ("crossseed", "Cross-seed")]
DR_SEEDS_W = "0.6/0.4"


def _a12(R):
    """a12_dr_landscape_ci.json, checked: the seed-0 check passed, the seed set and the once rule as recorded, the
    archived landscape file unchanged, and its seed-0 retention equal to RESULTS.json dr_landscape.verdict."""
    A = _late(LATE_DR_SEEDS)
    if A["s0_check"].get("passed") is not True or A["seed_set"] != ["archived s0", "new s1", "new s2"]:
        raise SystemExit(f"{LATE_DR_SEEDS}: the seed-0 check failed, or the seeds differ from the archived seed 0 plus "
                         "the new seeds 1 and 2")
    if A["fixed_terms_rule_applied"].get("rule") != "once" or A["reading"]["interval"] != "seeds":
        raise SystemExit(f"{LATE_DR_SEEDS}: CT-mask terms not printed once, or the interval is not over seeds")
    lp = os.path.join(ROOT, A["archived"]["path"])
    if hashlib.md5(open(lp, "rb").read()).hexdigest() != A["archived"]["md5"]:
        raise SystemExit(f"{LATE_DR_SEEDS}: {A['archived']['path']} changed since the aggregation")
    if [a for a, _ in DR_SEEDS_ARM] != A["arms"] or list(STAGES) != A["stages"]:
        raise SystemExit(f"{LATE_DR_SEEDS}: arms or stages differ")
    ver = R["dr_landscape"]["verdict"]
    for s in STAGES:
        rr = A["retention_rule_q6"][s]
        if (abs(rr["learned_per_seed"][0]["combined_retained"] - ver[f"{s}/learned"]["combined_retained"]) > 1e-12
                or abs(rr["ct"]["combined_retained"] - ver[f"{s}/ct"]["combined_retained"]) > 1e-12):
            raise SystemExit(f"{LATE_DR_SEEDS}: seed-0 retention of {s} differs from RESULTS.json dr_landscape.verdict")
        for a, _ in DR_SEEDS_ARM:
            for t in ("seg_learned", f"combined_learned_{DR_SEEDS_W}"):
                c = A["table"][a][s][t]
                if c["n"] != 3 or c["df"] != 2 or len(c["per_seed"]) != 3:
                    raise SystemExit(f"{LATE_DR_SEEDS}: table.{a}.{s}.{t} is not over three seeds")
    return A


def posthoc_dr_landscape_tab(R, out):
    A = _a12(R)
    S = _S(LATE_DR_SEEDS, A)
    W = DR_SEEDS_W
    L = _late_head(LATE_DR_SEEDS + " (post hoc; segmenter seeds 0 to 2)") + [
        "\\begin{table*}[tp]",
        "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
        "\\caption{DR landscape over segmenter seeds 0 to 2 (post hoc). Mean per-view Spearman correlation of each term "
        "with closeness to the target over the 60 held-out views. Arms: DR off (off), one draw shared by target and "
        "samples (shared), independent draws (cross) or a new speckle seed only (cross-seed). Combined: the match "
        "score at weights " + W + ". CT-mask terms do not depend on the segmenter seed and print once. Learned-mask "
        "terms: the value of each segmenter seed, and their mean with the 95\\% $t$ interval over the three seeds (2 "
        "df). Seed 0 is the run of \\ref{app:dr}. The lower block gives the share of the off arm's combined "
        "correlation that the cross arm keeps; the pre-specified rule asks for at least 80\\%. "
        "$^{\\dagger}$Privileged.}",
        "\\label{tab:posthoc-dr-landscape}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l l c c c l l l l@{}}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{CT mask$^{\\dagger}$} & \\multicolumn{2}{c}{Segmentation term, learned mask} & "
        "\\multicolumn{2}{c}{Combined, learned mask} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "Arm & Stage & Appearance & Segmentation & Combined & Seeds 0, 1, 2 & Mean [95\\% CI] & Seeds 0, 1, 2 & "
        "Mean [95\\% CI] \\\\",
        "\\midrule"]
    log("\n== tab:posthoc-dr-landscape (posthoc_dr_landscape.tex), POST HOC ==")
    for ai, (a, alab) in enumerate(DR_SEEDS_ARM):
        if ai:
            L.append("\\addlinespace")
        for si, s in enumerate(STAGES):
            b = f"table.{a}.{s}"
            ct = [S.plain(f'{b}.ct_mask_terms_once["{t}"]', 3) for t in ("app", "seg_ct", f"combined_ct_{W}")]
            cells = []
            for t in ("seg_learned", f"combined_learned_{W}"):
                tk = f'{b}["{t}"]'
                cells.append(", ".join(S.plain(f"{tk}.per_seed[{i}]", 3) for i in range(3)))
                cells.append(S.plain(f"{tk}.mean", 3) + " " + S.ci(f"{tk}.ci95_t", 3, signed=False))
            L.append(S.line(f"{alab if si == 0 else ''} & {NAME[s]} & " + " & ".join(ct + cells) + " \\\\"))
            log(f"{a:9s} {s:9s} ct {[round(A['table'][a][s]['ct_mask_terms_once'][t], 3) for t in ('app', 'seg_ct', f'combined_ct_{W}')]} "
                f"learned seg {A['table'][a][s]['seg_learned']['mean']:.3f} {A['table'][a][s]['seg_learned']['ci95_t']} "
                f"combined {A['table'][a][s][f'combined_learned_{W}']['mean']:.3f}")
    L += ["\\midrule",
          "\\multicolumn{9}{@{}l}{\\textit{Share of the off arm's combined correlation kept by the cross arm}} \\\\"]
    for s in STAGES:
        b = f"retention_rule_q6.{s}"
        rr = A["retention_rule_q6"][s]
        ct = S.pct(f"{b}.ct.combined_retained", 0)
        ps_ = ", ".join(S.pct(f"{b}.learned_per_seed[{i}].combined_retained", 0) for i in range(3))
        mn = S.pct(f"{b}.learned_on_seed_means.combined_retained", 0)
        k3 = S.int(f"{b}.learned_k_of_3")
        S.rec(f"{b}.learned_on_seed_means.rule_holds", rr["learned_on_seed_means"]["rule_holds"], "word")
        S.rec(f"{b}.ct.rule_holds", rr["ct"]["rule_holds"], "word")
        on_m = "and on the seed means" if rr["learned_on_seed_means"]["rule_holds"] else "not on the seed means"
        L.append(S.line(f" & {NAME[s]} & & & {ct} & \\multicolumn{{2}}{{l}}{{rule holds in {k3} of 3 seeds, {on_m}}} & "
                        f"{ps_} & {mn} \\\\"))
        log(f"retention {s}: ct {rr['ct']['combined_retained']:.3f}; learned per seed "
            f"{[round(x['combined_retained'], 3) for x in rr['learned_per_seed']]} k {rr['learned_k_of_3']}; seed means "
            f"{rr['learned_on_seed_means']['combined_retained']:.3f} holds {rr['learned_on_seed_means']['rule_holds']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "posthoc_dr_landscape.tex", "\n".join(L))


# ================================================================================= statistics layers, Q1 Q4 Q6 Q7
LAYER_BLOCKS = [  # (family, block title, unit, decimals, item labels)
    ("Q1_pooled", "Learnability, pooled organs: SonoSPADE minus stage, per-patient mean Fisher $z$", 3,
     {"physics": "Physics", "cut": "CUT", "cyclegan": "CycleGAN"}),
    ("Q1_liver", "Learnability, liver views", 3, {"physics": "Physics", "cut": "CUT", "cyclegan": "CycleGAN"}),
    ("Q1_kidney", "Learnability, kidney views", 3, {"physics": "Physics", "cut": "CUT", "cyclegan": "CycleGAN"}),
    ("Q4_realism", "Realism: physics minus the authors' reimplementation of LOTUS, W1 per patient", 3,
     {"physics_minus_lotus": "Physics minus LOTUS, authors' reimplementation"}),
    ("Q6_policy", "Domain-randomization policy arms minus the CT-mask arm without DR (SonoSPADE), mm", 2,
     {"eval DR only": "Evaluation DR only", "training DR only": "Training DR only",
      "training and eval DR": "Training and evaluation DR"}),
    ("Q7_contact", "Contact off minus on, final translation error, mm", 2,
     {"ctmask/physics": "CT-mask final$^{\\dagger}$, Physics", "ctmask/cut": "CT-mask final$^{\\dagger}$, CUT",
      "ctmask/cyclegan": "CT-mask final$^{\\dagger}$, CycleGAN",
      "ctmask/sonospade": "CT-mask final$^{\\dagger}$, SonoSPADE",
      "matched/physics": "Learned-mask final, Physics", "matched/cut": "Learned-mask final, CUT",
      "matched/cyclegan": "Learned-mask final, CycleGAN", "matched/sonospade": "Learned-mask final, SonoSPADE",
      "static": "Static start", "scripted": "Scripted sweep", "pose_oracle": "Pose oracle$^{\\dagger}$",
      "ibvs_base/ct": "CT-mask servo base$^{\\dagger}$, four stages",
      "ibvs_base/learned/physics": "Learned-mask servo base, Physics",
      "ibvs_base/learned/cut": "Learned-mask servo base, CUT",
      "ibvs_base/learned/cyclegan": "Learned-mask servo base, CycleGAN",
      "ibvs_base/learned/sonospade": "Learned-mask servo base, SonoSPADE"}),
]
LAYER_COUNTS = {"Q1_pooled": (3, 3, 0), "Q1_liver": (3, 3, 0), "Q1_kidney": (3, 3, 0), "Q4_realism": (1, 1, 1),
              "Q6_policy": (3, 0, 0), "Q7_contact": (16, 9, 0)}   # (m, Holm t passes, Holm exact passes): asserted


def _layers():
    F = _late(LATE_LAYERS)
    fams = F["families"]
    if [f for f, *_ in LAYER_BLOCKS] != list(fams):
        raise SystemExit(f"{LATE_LAYERS}: families {list(fams)} differ from the table blocks")
    for f, _, _, labs in LAYER_BLOCKS:
        fm = fams[f]
        if set(labs) != set(fm["items"]) or fm["m"] != len(fm["items"]):
            raise SystemExit(f"{LATE_LAYERS}: {f} items differ from the table rows")
        got = (fm["m"], sum(1 for it in fm["items"].values() if it["holm_t"] < 0.05),
               sum(1 for it in fm["items"].values() if it["holm_signflip"] < 0.05))
        if got != LAYER_COUNTS[f] or got[1:] != (fm["n_holm_t_survive"], fm["n_holm_signflip_survive"]):
            raise SystemExit(f"{LATE_LAYERS}: {f} counts {got} differ from {LAYER_COUNTS[f]} or from the record")
    a7 = {x["family"]: (x["m"], x["n_holm_t_pass"], x["n_holm_signflip_pass"]) for x in F["a7_sentence"]["families"]}
    if a7 != LAYER_COUNTS:
        raise SystemExit(f"{LATE_LAYERS}: a7_sentence.families {a7} differ from {LAYER_COUNTS}")
    return F


def stats_layers_tab(out):
    F = _layers()
    L = _late_head(LATE_LAYERS + " (post hoc)") + [
        "\\begin{table*}[tp]",
        "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
        "\\caption{Post hoc statistics layer of Table~\\ref{tab:stats-sensitivity} for the learnability, realism, "
        "domain-randomization policy and contact contrasts of the primary analysis (6 held-out patients). Columns as "
        "in Table~\\ref{tab:stats-sensitivity}; Holm within each block. Learnability: difference of the per-patient "
        "mean Fisher $z$ of the per-view Spearman correlation. Realism: difference of the per-patient in-sector W1. "
        "The four CT-mask servo-base rows of the contact block hold identical episodes by construction and count "
        "once; the static start, whose differences are all 0, enters at $p = 1$. Patient $\\times$ seed: n/a where "
        "a row has no seed factor. $^{\\dagger}$Privileged.}",
        "\\label{tab:stats-layers}",
        "\\centering",
        "\\setlength{\\tabcolsep}{2.2pt}",   # 2.8pt (as Table S2.5) ran 5.6pt past the text width
        "\\begin{tabular}{@{}l r c c c c c c c c c@{}}",
        "\\toprule",
        "Contrast & Mean & Bootstrap CI & $t$ CI & Patient $\\times$ seed CI & $k$/6 & Exact $p$ & Holm & "
        "$t$ $p$ & Holm & Rule \\\\",
        "\\midrule"]
    log("\n== tab:stats-layers (stats_layers.tex), POST HOC ==")
    for bi, (f, title, dd, labs) in enumerate(LAYER_BLOCKS):
        fm = F["families"][f]
        if bi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\textit{{{title}, $m = {fm['m']}$}}}} \\\\")
        for n, lab in labs.items():
            e = fm["items"][n]
            cx = e.get("ci_crossed_patient_seed")
            # no k/6 for differences that are all 0 by construction (the static start), as Table A.5 (contact_dr.tex)
            # prints that row
            kk = "n/a" if e.get("identical_by_construction") is True else f"{e['sign']}/{e['n']}"
            L.append(f"\\quad {lab} & {sg(e['mean'], dd)} & {ci_txt(e['ci_boot'], dd)} & {ci_txt(e['ci_t'], dd)} & "
                     + (ci_txt(cx, dd) if cx else "n/a")
                     + f" & {kk} & {pfrac(e['p_signflip_exact'])} & {pdec(e['holm_signflip'])} & "
                     f"{pdec(e['p_t'])} & {pdec(e['holm_t'])} & {'yes' if e['holds_rule'] else 'no'} \\\\ "
                     f'% src: {LATE_LAYERS} families.{f}.items["{n}"].{{mean, ci_boot, ci_t, ci_crossed_patient_seed, '
                     "sign, p_signflip_exact, holm_signflip, p_t, holm_t, holds_rule}")
            log(f"{f}/{n}: {e['mean']:+.4f} t [{e['ci_t'][0]:+.4f}, {e['ci_t'][1]:+.4f}] k {e['sign']}/{e['n']} "
                f"exact {e['p_signflip_exact']:.4f} holm {e['holm_signflip']:.4f} t p {e['p_t']:.4g} holm t "
                f"{e['holm_t']:.4g} rule {e['holds_rule']}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return write(out, "stats_layers.tex", "\n".join(L))


# ================================================================================= the s0513 case study
CASE_STUDY_ROWS = [("static", "Static start"), ("pose_oracle", "Analytic pose oracle$^{\\dagger}$"),
            ("ct_base", "CT-mask servo base$^{\\dagger}$")] + \
    [(f"ctmask/{s}/final", f"CT-mask final policy$^{{\\dagger}}$, {NAME[s]}") for s in STAGES] + \
    [(f"matched/{s}/final", f"Learned-mask final policy, {NAME[s]}") for s in STAGES]


def _a13():
    A = _late(LATE_CASE_STUDY)
    if A["case"] != "s0513" or list(A["item1_rows"]) != [k for k, _ in CASE_STUDY_ROWS]:
        raise SystemExit(f"{LATE_CASE_STUDY}: case or rows differ from the listed rows")
    for k, _ in CASE_STUDY_ROWS:
        r = A["item1_rows"][k]
        want = 10 if k in ("static", "pose_oracle") else 30
        if r["n_episodes_case"] != want or len(r["episodes"]) != want:
            raise SystemExit(f"{LATE_CASE_STUDY}: {k} holds {r['n_episodes_case']} s0513 episodes, not {want}")
    cc = A["cross_check"]
    if cc.get("agree") is not True:
        raise SystemExit(f"{LATE_CASE_STUDY}: cross_check with posthoc_logs.json does not agree")
    return A


def _a13_split(A, row):
    by = collections.defaultdict(list)
    for e in A["item1_rows"][row]["episodes"]:
        by[e["target"]].append(float(e["final_pos"]))
    return {t: float(np.mean(x)) for t, x in by.items()}


def posthoc_s0513_tab(out):
    A = _a13()
    S = _S(LATE_CASE_STUDY, A)
    st = A["item1_rows"]["static"]["patient_level"]["start_pos"]
    L = _late_head(LATE_CASE_STUDY + " (post hoc, descriptive)") + [
        "\\begin{table}[tp]",
        "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
        "\\caption{Held-out patient s0513 against the other five (post hoc, descriptive). Its start error is "
        + S.plain("item1_rows.static.patient_level.start_pos.case") + "~mm (other five "
        + S.plain("item1_rows.static.patient_level.start_pos.others_min") + " to "
        + S.plain("item1_rows.static.patient_level.start_pos.others_max") + "~mm). Final: patient mean of the final "
        "translation error; other five: their range. $S$: match score of the final view against the target; other "
        "five: their mean. Static start and pose oracle: 10 episodes of s0513; the other rows pool seeds 0 to 2 (30 "
        "episodes). $^{\\dagger}$Privileged.}" + S.tail(),
        "\\label{tab:posthoc-s0513}",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l r r r r r@{}}",
        "\\toprule",
        "Row & Episodes & \\begin{tabular}[b]{@{}r@{}}s0513 final\\\\(mm)\\end{tabular} & "
        "\\begin{tabular}[b]{@{}r@{}}Other five,\\\\range (mm)\\end{tabular} & "
        "\\begin{tabular}[b]{@{}r@{}}s0513\\\\final $S$\\end{tabular} & "
        "\\begin{tabular}[b]{@{}r@{}}Other five,\\\\mean $S$\\end{tabular} \\\\",
        "\\midrule"]
    S.e = []
    log("\n== tab:posthoc-s0513 (posthoc_s0513.tex), POST HOC ==")
    for k, lab in CASE_STUDY_ROWS:
        b = f'item1_rows["{k}"]'
        n = S.int(f"{b}.n_episodes_case")
        fp = f"{b}.patient_level.final_pos"
        fm = f"{b}.patient_level.final_match"
        L.append(S.line(f"{lab} & {n} & {S.plain(fp + '.case')} & {S.plain(fp + '.others_min')} to "
                        f"{S.plain(fp + '.others_max')} & {S.plain(fm + '.case', 3)} & "
                        f"{S.plain(fm + '.others_mean', 3)} \\\\"))
        pl = A["item1_rows"][k]["patient_level"]
        log(f"{k:24s} final {pl['final_pos']['case']:.3f} (others {pl['final_pos']['others_min']:.3f} to "
            f"{pl['final_pos']['others_max']:.3f}, rank {pl['final_pos']['rank_highest_first']}) S "
            f"{pl['final_match']['case']:.3f} vs {pl['final_match']['others_mean']:.3f}")
    for row in ("static", "ct_base"):
        log(f"s0513 {row} by target (from episodes[*].final_pos): {_a13_split(A, row)}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return write(out, "posthoc_s0513.tex", "\n".join(L))


# ================================================================================= mixed model
MM_FIT_NAME = {   # plain question names (no plan codes print)
    "a14-Q2-ct": ("Primary ladder contrasts, CT masks", True),
    "a14-Q2-lm": ("Primary ladder contrasts, learned masks", True),
    "a14-Q6-ct": ("Domain-randomization policy arms, CT masks", True),
    "a14-Q7-ct": ("Contact off minus on, reference rungs and CT masks", True),
    "a14-Q7-lm": ("Contact off minus on, learned masks", True),
    "s38-Q8-ct": ("Cause of the oracle-to-CT-mask gap: privileged look-ahead, CT masks", False),
    "s38-Q9-ct": ("Hill-climb on the match score, CT masks", False),
    "s38-Q9-lm": ("Hill-climb on the match score, learned masks", False),
    "s38-Q10-lm": ("Label-information controls, learned masks", False),
    "s38-Q11-lm": ("Texture budget and texture seeds, learned masks", False),
    "s38-Q13-ct": ("Planted-deficit control of REINFORCE, CT masks", False),
    "s38-Q2R-ct": ("Replication cohort of 20 CTs, CT masks", False),
    "s38-Q2R-lm": ("Replication cohort of 20 CTs, learned masks", False),
    "s38-E-BN-ct": ("BatchNorm in evaluation minus train mode, CT masks", False),
    "s38-E-BN-lm": ("BatchNorm in evaluation minus train mode, learned masks", False),
    "s38-E-H36-ct": ("36 against 12 steps, CT masks", False),
}
def _mm_label(s):
    """The printed form of a mixed-model record label: the development build derives it from its record's label
    (plan_private.mm_label); the results archive holds the printed label itself."""
    return PP.mm_label(s) if PP is not None and not SD.public() else s


def _mm():
    """Both mixed-model result files and the fit list, from the committed ledger; every fit converged; the record
    cites the fit list by its sha256; the 16 fits and 89 rows of the fit list."""
    fl, sha_fl = _fit_list()
    out_ = []
    for r_ in LATE_MM:
        d = _late(r_)
        cite_ok = (PI.skip_dev("the plan path that the mixed-model records cite (their sha256 is checked)")
                   or os.path.normpath(d["fit_list_record"]["path"]) == os.path.normpath(LATE_FIT_LIST))
        if d["fit_list_record"]["sha256"] != sha_fl or not cite_ok:
            raise SystemExit(f"{r_}: does not cite the fit list {_rec_name(FIT_LIST_ID)} as committed")
        bad = [k for k, f in d["fits"].items() if f["status"] != "fitted" or f["converged"] is not True]
        if bad:
            raise SystemExit(f"{r_}: fits not fitted or not converged: {bad}")
        if d["model"]["formula"] != "final_pos ~ C(arm)" or d["model"]["reml"] is not True:
            raise SystemExit(f"{r_}: model differs from the pre-specified mixed model")
        out_.append(d)
    fits = {k: f for d in out_ for k, f in d["fits"].items()}
    rows = [r for d in out_ for r in d["table_rows"]]
    if set(fits) != set(MM_FIT_NAME) or len(rows) != 89 or (fl is not None and
                                                              not fl.startswith("USB Gym v2.1: fit list")):
        raise SystemExit(f"mixed model: {len(fits)} fits and {len(rows)} rows, expected 16 and 89")
    return out_, fits, rows


def mixed_model_tab(out):
    D, fits, rows = _mm()
    by_fit = collections.OrderedDict((k, []) for d in D for k in d["fits"])
    for d, r_ in zip(D, LATE_MM):
        for i, r in enumerate(d["table_rows"]):
            by_fit[r["fit"]].append((r, r_, i))
    pkgs = {d["environment"]["installed"]["packages"]["statsmodels"] for d in D}
    if pkgs != {"0.15.0"}:
        raise SystemExit(f"mixed model: statsmodels {pkgs}, expected 0.15.0")
    n_powell = sum(1 for f in fits.values() if f["optimizer"] == "powell")
    n_bound_patient = sum(1 for f in fits.values() if "patient" in f["boundary"])
    head_ = ["\\centering", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X r l@{}}", "\\toprule",
             "Contrast & Estimate (mm) & Wald 95\\% CI (mm) \\\\", "\\midrule"]
    foot_ = ["\\bottomrule", "\\end{tabularx}", "\\end{table*}"]
    L = _late_head(", ".join(LATE_MM) + " (secondary; the primary-result fits post hoc)") + [
        "\\begin{table*}[tp]",
        "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
        "\\caption{Mixed-model estimates of the final translation error (secondary; never overriding a pre-specified "
        "rule; the fits of the primary results are post hoc). Final translation error by arm, with a random intercept "
        "for patient and variance components for window and seed within patient. Each model is fitted per question "
        "and mask source by restricted maximum likelihood (statsmodels 0.15.0 MixedLM; Nelder-Mead, then Powell in "
        f"{n_powell} fits). All {len(fits)} fits converged. Wald 95\\% CIs use the normal quantile, since statsmodels "
        "provides no Satterthwaite or Kenward-Roger degrees of freedom. The estimand is conditional on the trained "
        "artifacts, with seeds indexing artifacts and the patient as the sampling unit; these intervals do not rest "
        "on the patient alone. The model has no arm-by-patient term and one residual variance for all arms of a "
        "model, reference rows included. In a contrast of two arms with the same seeds, the patient, window and seed "
        "effects cancel. Its interval then rests on that pooled episode-level residual variance and does not carry "
        "the between-patient spread of the contrast that the paired $t$ interval carries. In a contrast of an arm with "
        "seeds against an arm without seeds or with fewer seeds, the seed variance component enters the interval, "
        "which then treats the seeds as sampled. Contrasts between the two mask sources are not modeled, since each "
        "model holds one mask source. The planted lateral-bias rows apply the contrasts of the REINFORCE control to "
        "the final translation error. The control's rule reads the lateral displacement, which no model here holds, "
        "so these rows need not agree with Table~\\ref{tab:checks-summary}. "
        "Each header gives the fit's variance components as SDs in mm; a component at "
        f"the boundary was estimated at 0 (the patient component in {n_bound_patient} of {len(fits)} fits). "
        "$^{\\ddagger}$Includes runs that began before their decision rules were fixed. $^{\\dagger}$Privileged.}",
        "\\label{tab:mixed-model}"] + head_
    log("\n== tab:mixed-model (mixed_model.tex), SECONDARY ==")
    n_lines, split_at = 0, 50   # 58 lines ran 36pt past one page; headers wrap to two lines
    for fi, (fid, rs) in enumerate(by_fit.items()):
        f = fits[fid]
        name, ph = MM_FIT_NAME[fid]
        sd = f["sd_components"]
        rec = LATE_MM[0] if fid.startswith("a14") else LATE_MM[1]
        notes = []
        if f["boundary"]:
            notes.append("at the boundary: " + ", ".join(f["boundary"]))
        if f["se_check"]["flag"]:
            notes.append(f"GLS standard errors differ from the model's by up to "
                         f"{100 * f['se_check']['max_rel_gap']:.1f}\\%")
        if any("Hessian matrix" in w for w in f["warnings"]):
            notes.append("Hessian not positive definite at the estimate")
        mark = "$^{\\ddagger}$" if f[PI.key("early_inputs")] or f.get(PI.key("early_artifacts")) else ""
        hdr = (f"\\textit{{{name}{' (post hoc)' if ph else ''}}}{mark}. SD (mm): patient {sd['patient']:.2f}, window "
               f"{sd['window']:.2f}, seed {sd['seed']:.2f}, residual {sd['residual']:.2f}"
               + ("; " + "; ".join(notes) if notes else ""))
        if n_lines + 1 + len(rs) > split_at and n_lines:
            L += foot_ + ["\\begin{table*}[tp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize", "\\ContinuedFloat",
                          "\\caption[]{Continued.}"] + head_
            n_lines = 0
        elif fi:
            L.append("\\addlinespace")
        L.append(f"\\multicolumn{{3}}{{@{{}}p{{0.97\\textwidth}}@{{}}}}{{{hdr}}} \\\\ % src: {rec} fits"
                 f'["{fid}"].sd_components, .boundary, .se_check, .warnings, .{PI.key("early_inputs")}, .optimizer '
                 f'{f["optimizer"]}')
        n_lines += 1
        fmt = _plan_stats().fmt_num
        for r, r_, i in rs:
            lab = _mm_label(r["label"])
            est = f"${fmt(r['estimate_mm'], 2, True)}$"
            ci = f"$[{fmt(r['ci95_mm'][0], 2, True)}, {fmt(r['ci95_mm'][1], 2, True)}]$"
            rm = "$^{\\ddagger}$" if _T("mm early clause") in r["printed"] else ""
            if bool(rm) != bool(mark):
                raise SystemExit(f"mixed model {fid}: the earlier-runs clause of a row and of its fit disagree")
            if not r["printed"].startswith(r["label"] + ": " + fmt(r["estimate_mm"], 2, True) + " mm "):
                raise SystemExit(f"mixed model {fid}: table_rows[{i}] printed form differs from its estimate")
            L.append(f"\\quad {lab}{rm} & {est} & {ci} \\\\ % src: {r_} table_rows[{i}].estimate_mm "
                     f"{r['estimate_mm']:.4f}, .ci95_mm [{r['ci95_mm'][0]:.4f}, {r['ci95_mm'][1]:.4f}], .member "
                     f"{r['member']}")
            n_lines += 1
            log(f"{fid:13s} {lab}: {r['estimate_mm']:+.3f} [{r['ci95_mm'][0]:+.3f}, {r['ci95_mm'][1]:+.3f}]"
                f"{_T('mm early log') if rm else ''}")
    L += foot_ + [""]
    return write(out, "mixed_model.tex", "\n".join(L))


SIGN_CHECK_TEX = os.path.join(PAPER, "sections", "11_checks.tex") if PAPER else None


def _sign_check():
    """The sign check (reversed label offset, -3 mm) is printed by hand in Section 8.4 (11_checks.tex). Its record
    is pinned (LATE_PINNED); the printed sentence must be the record's printed form, with the values the record's
    mean and t interval round to. SystemExit names the first problem."""
    d = _late(LATE_SIGN_CHECK)
    e = d["results"]["closed_loop"]["sonospade"]["elev_m3_sign_check"]
    fmt = _plan_stats().fmt_num
    m, (lo, hi) = e["Delta_y_minus3"]["mean"], e["Delta_y_minus3"]["ci_t"]
    plain = (f"With the label offset reversed (-3 mm), the shift is {fmt(m, 2, True)} mm [{fmt(lo, 2, True)}, "
             f"{fmt(hi, 2, True)}] (secondary).")
    tex = (f"With the label offset reversed ($-3$~mm), the shift is ${fmt(m, 2, True)}$~mm [${fmt(lo, 2, True)}$, "
           f"${fmt(hi, 2, True)}$] (secondary).")
    if e["printed"] != plain:
        raise SystemExit(f"{LATE_SIGN_CHECK}: elev_m3_sign_check.printed {e['printed']!r} differs from {plain!r}")
    # a missing file or a commented-out sentence does not pass; the comment strip is the one write() uses
    if not PI.skip_dev("a sentence of the article's source text (Section 8.4)"):
        if not os.path.isfile(SIGN_CHECK_TEX):
            raise SystemExit(f"{rel(SIGN_CHECK_TEX)}: missing, so the sign-check sentence cannot be checked")
        body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in open(SIGN_CHECK_TEX, encoding="utf-8").read().splitlines())
        if tex not in body:
            raise SystemExit(f"{rel(SIGN_CHECK_TEX)}: the sign-check sentence {tex!r} is not printed as the record gives it")
    log(f"[{_T('sign check tag')}] {LATE_SIGN_CHECK} (sha256 {LATE_PINNED[LATE_SIGN_CHECK][:12]}): {plain}")


Q8G1_TEX = os.path.join(PAPER, "sections", "appendix_eval.tex") if PAPER else None


def _q8_g1_check():
    """The Q8-G1 sentence (the biplane extra of Q8) is printed by hand in Appendix A.2 (appendix_eval.tex, paragraph
    "Second imaging plane"), verbatim as the analysis record prints it, with "~mm" for " mm". The record passes
    _q8_g1() first (code state, ledger, the two cited records, k1). SystemExit names the first
    problem; a missing file or a commented-out sentence does not pass (the comment strip of _sign_check)."""
    rec = _q8_g1()
    text = rec["printed_sentence"]["text"]
    if rec["printed_sentence"].get("form") != "verbatim" or rec["printed_sentence"].get("add_on_printed") is not False:
        raise SystemExit(f"{Q8G1_REL}: printed form or add-on differ from the reading (verbatim, no add-on)")
    tex = re.sub(r"(\d) mm\b", r"\1~mm", text)
    if PI.skip_dev("a sentence of the article's source text (Appendix A.2)"):
        return
    if not os.path.isfile(Q8G1_TEX):
        raise SystemExit(f"{rel(Q8G1_TEX)}: missing, so the Q8-G1 sentence cannot be checked")
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in open(Q8G1_TEX, encoding="utf-8").read().splitlines())
    if body.count(tex) != 1:
        raise SystemExit(f"{rel(Q8G1_TEX)}: the Q8-G1 sentence {tex!r} is not printed once as the record gives it")


def late_tables(R, out):
    """The tables of the late results (Supplementary S2); the hand-written sentences cite the same
    records."""
    log("\n== late results" + _T("late date") + ": tables (post hoc and secondary) ==")
    _sign_check()
    _q8_g1_check()   # late result (Q8-G1, Appendix A.2)
    return [posthoc_realism_tab(out), posthoc_dr_landscape_tab(R, out), stats_layers_tab(out), posthoc_s0513_tab(out),
            mixed_model_tab(out)]


# ================================================================================== amendment 6
# The eight item analyses of amendment 6 (amend6.py of the analysis folder, outputs/v2_1/analysis_plan/amend6_<item>.json)
# and its record sentences (development build only). While BLIND_A6 is on, no amendment 6 output is read: each item
# file holds one blinded line, and each record macro of tables/amend6_records.tex holds its earlier sentence (from
# the amendment's Appendix). With --read-amend6 (the check record exists, cites the amendment and is ledgered; the
# eight outputs exist), each item file holds its printed sentences and tables, and the record macros hold the printed
# sentences and history rows of the analyses. [Tab] and [section] of a sentence become \ref{tab:amend6-<table_ref>}
# and \ref{sec:amend6-<item>}, labels the manuscript defines; in the preface \AmendSixPrefaceRank the [Tab] marker
# stays as the macro \AmendSixTab, which the article's sources define and position.
A6_ITEMS = ("g2", "p9", "l10", "c30", "r13", "look", "sp", "td")
A6_RECORD_IDS = "ABCDEFGHIJKLM"
A6_PRINTED_LABEL = _T("a6 printed label")
# the label of a caption: the result sentences carry the label of the records; a caption is the build's and prints
# the scientific part only
A6_CAPTION_LABEL = "exploratory; estimation only"
_A6_ESC = re.compile(r"(?<!\\)([%&#_])")
# The plan sentences an item file prints verbatim although they hold a word that write() refuses, because the
# sentence denies it (with the x2 rows; the article prints it by hand as well): item -> sentences
A6_VERBATIM = {"l10": ("No stage is called realistic.",)}
# (development build only: the record tables need the record sentences of the amendment) the titles of the item
# files and of their tables, {table id: (title, label prefix, closing clause)}; a table id outside A6_TABLES is
# refused
A6_ITEM_TITLES = PP.A6_ITEM_TITLES if PP else {}
A6_TABLES = PP.A6_TABLES if PP else {}
# The printed columns of an amendment 6 table row (amend6.py estimate(), paired() and held_contrast(), plus the row's
# own keys), in printed order: key -> (header, kind), formatted by the number format of amendment 6. Kinds: text (a word;
# amend6.py's codes printed through A6_WORDS), mm (a signed mean, 2 decimals), mm_abs (an error, unsigned, 2
# decimals), ci (a signed interval, 2 decimals), kn (k/n as counts: sign_k over sign_n), p (the two-sided sign-flip p
# as plan_stats.fmt_p formats it; a Monte Carlo p carries MC), seeds (the per-seed differences, signed, 2 decimals),
# dice (3 decimals) and note (a row's own note, blank where it has none).
# Every key of a row is in A6_COLUMNS or A6_NOT_PRINTED, else the table is refused: no value prints with a default
# format.
A6_COLUMNS = {
    "row": ("Row", "text"), "set": ("CTs", "text"), "contrast": ("Contrast", "text"), "rung": ("Rung", "text"),
    "stage": ("Stage", "text"), "mask": ("Masks", "text"), "organ": ("Organ", "text"),
    "E": ("Final error (mm)", "mm_abs"), "mean": ("Mean (mm)", "mm"), "ci_t": ("95\\% $t$ CI (mm)", "ci"),
    "sign_k": ("$k/n$", "kn"), "p_signflip": ("Sign-flip $p$ (unadjusted; descriptive)", "p"),
    "ci_boot_unit": ("Bootstrap CI over units (mm)", "ci"), "per_seed": ("Per seed (mm)", "seeds"),
    "ci_boot_unit_seed": ("Bootstrap CI over units and seeds (mm)", "ci"),
    "ci_boot_held": ("Bootstrap CI over units and seeds, servo seed mean held (mm)", "ci"),
    "default_vs_reference": ("Dice, default against reference", "dice"),
    "default_vs_fast": ("Dice, default against fast mode", "dice"), "note": ("Note", "note")}
# keys read into another column (n and sign_n by k/n, p_signflip_label by the p header, signflip_method by the MC
# mark) or never printed (the per-unit values, and the seed lists that the per-seed column shows)
A6_NOT_PRINTED = ("n", "sign_n", "p_signflip_label", "signflip_method", "per_unit", "seeds", "seeds_used")
A6_P_LABEL = "unadjusted; descriptive"
A6_P_EXACT, A6_P_MC = "exact sign-flip p", "Monte Carlo sign-flip p"
# The codes amend6.py writes into a row's stage, mask, rung, set, row and contrast cells, each printed as its words
# (LaTeX, math in $...$); the build stops on a code that has no words here, and on a key that has no format. The organ
# and note cells are text, printed escaped.
_A6_STAGE = {"physics": "physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
_A6_ARM = {"ctmask": "CT-mask", "matched": "learned-mask"}
_A6_FIELDS = {"onefield": "one speckle field", "indep": "independent speckle fields"}
_A6_OFFSET = {"elev3": "3 mm label offset", "elev6": "6 mm label offset"}
_A6_L10_ARM = {"x2": "x2 base", "policy": "final policy"}
_A6_BIAS_SET = {"shipped": "shipped", "explore": "learned-exploration", "reward": "privileged-shaping"}
# the 25 rungs of the contact-off part (d) (amend6.group_rungs): the ladder rungs, the servo bases and the policies at both weights
_A6_CONTACT_RUNG = dict(
    {"static": "static", "scripted": "scripted", "pose_oracle": "pose oracle",
     "base_ct_physics": "CT-mask servo base, physics", "base_ct_sonospade_s0": "CT-mask servo base, SonoSPADE, seed 0"},
    **{f"base_matched_{st}": f"learned-mask servo base, {w}" for st, w in _A6_STAGE.items()},
    **{f"{a}_{st}/{wt}": f"{aw} policy, {w}, {ww}" for a, aw in _A6_ARM.items() for st, w in _A6_STAGE.items()
       for wt, ww in (("imit", "imitation start"), ("final", "final weights"))})
A6_CONTRASTS = dict(
    {"indep bound minus grid-fixed bound": "independent speckle fields: bound minus grid-fixed bound",
     "x2 minus lit": "x2 base minus lit painted stage", "x2 minus physics": "x2 base minus physics",
     "x2 minus SonoSPADE": "x2 base minus SonoSPADE",
     "C1-type": "learned minus CT masks, physics (C1-type)",
     "C2-type": "SonoSPADE minus physics, learned-mask servo base (C2-type)",
     "(i)": "(i)", "(ii)": "(ii)", "(i) minus (ii)": "(i) minus (ii)", "(iii)": "(iii)", "(iv)": "(iv)",
     "default minus reference": "default minus reference labels",
     "default minus fast": "default minus fast-mode labels"},
    **{f"{o} bound minus CT-mask base": f"{w}: bound minus linear servo" for o, w in _A6_FIELDS.items()},
    **{f"{o} bound minus oracle": f"{w}: bound minus pose oracle" for o, w in _A6_FIELDS.items()},
    **{f"{a} Delta_y {c}": f"{aw}, {cw}: shift $\\Delta y$" for a, aw in _A6_L10_ARM.items()
       for c, cw in _A6_OFFSET.items()},
    **{f"{a} x_p {c}": f"{aw}, {cw}: $x_p$" for a, aw in _A6_L10_ARM.items() for c, cw in _A6_OFFSET.items()},
    **{f"{a} permute total error change": f"{aw}, organ identities permuted: total error change"
       for a, aw in _A6_L10_ARM.items()},
    **{f"C3-type {a}/{st}": f"REINFORCE change, {aw} policy, {w} (C3-type)" for a, aw in _A6_ARM.items()
       for st, w in _A6_STAGE.items()},
    **{f"{v} deficit D_p": f"{w} settings: deficit $D_p$" for v, w in _A6_BIAS_SET.items()},
    **{f"{v} x_p": f"{w} settings: recovery $x_p$" for v, w in _A6_BIAS_SET.items()})
A6_WORDS = {"stage": _A6_STAGE, "mask": {"ct": "CT", "matched": "learned"},
            "rung": dict({"sguard": "S-guarded servo", "mlp_base": "MLP base", "cnn_base": "CNN base",
                          "hillclimb": "hill-climb"}, **_A6_CONTACT_RUNG),
            "set": {"cohort": "replication cohort", "amos": "AMOS"},
            "row": {"R1": "CT-mask servo base, physics (R1)", "R2": "learned-mask servo base, physics (R2)",
                    "R3": "learned-mask servo base, SonoSPADE (R3)",
                    "R4": "CT-mask policy, physics, final weights (R4)",
                    "R5": "CT-mask policy, SonoSPADE, final weights (R5)"},
            "contrast": A6_CONTRASTS}


def _a6_tex(text, table_ref=None, item=None):
    """A sentence of amendment 6 as LaTeX text: %, &, # and _ escaped; [Tab] and [section] (amend6.py writes them as
    \\AmendSixTab and \\AmendSixSection) resolved to the labels the manuscript defines, when the sentence names them."""
    t = _A6_ESC.sub(r"\\\1", str(text))
    if table_ref:
        t = t.replace("\\AmendSixTab", "\\ref{tab:amend6-" + str(table_ref).replace("_", "-") + "}")
    if item:
        t = t.replace("\\AmendSixSection", "\\ref{sec:amend6-" + item + "}")
    return t


def _a6_plan_rows():
    """The record sentences of the Appendix of amendment 6 (plan_stats.load_amendment, sha256-checked), ids A to M by
    position."""
    rows = _plan_stats().load_amendment(6)["record_sentences"]
    if len(rows) != len(A6_RECORD_IDS):
        raise SystemExit(f"amendment 6 Appendix record_sentences: {len(rows)} rows, not {len(A6_RECORD_IDS)}")
    return dict(zip(A6_RECORD_IDS, rows))


def _a6_earlier(row):
    """The earlier sentence as the manuscript prints it: the Appendix's printed_as when it gives one, else its text."""
    return row.get("printed_as") or row["text"]


def _a6_number(tid, key, v):
    """A finite number of an amendment 6 row, or None (printed n/a); a bool, a string or another type is refused."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SystemExit(f"amendment 6 table {tid}: {key} = {v!r} is not a number")
    return float(v) if math.isfinite(v) else None


def _a6_signed(tid, key, v, d=2):
    x = _a6_number(tid, key, v)
    return "n/a" if x is None else f"${_plan_stats().fmt_num(x, d, True)}$"


def _a6_cell(tid, key, row, words=None, ci_break=False):
    """One printed cell of an amendment 6 table row, formatted by its key's kind (A6_COLUMNS). words: the printed
    words of the codes (default A6_WORDS); ci_break: an interval prints as $[lo$, $hi]$, so that a narrow column can
    break it after the comma (the values are unchanged)."""
    words = A6_WORDS if words is None else words
    kind, v = A6_COLUMNS[key][1], row.get(key)
    ps = _plan_stats()
    if v is None:
        return "" if kind == "note" else "n/a"
    if kind in ("text", "note"):
        if not isinstance(v, str):
            raise SystemExit(f"amendment 6 table {tid}: {key} = {v!r} is not a word")
        if key not in words:
            return _a6_tex(v)
        if v not in words[key]:
            raise SystemExit(f"amendment 6 table {tid}: {key} code {v!r} has no printed words (make_tables A6_WORDS)")
        return words[key][v]
    if kind == "kn":
        k, n = v, row.get("sign_n")
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (k, n)) or k > n:
            raise SystemExit(f"amendment 6 table {tid}: k/n = {k!r}/{n!r} is not a count")
        return ps.fmt_k(k, n)
    if kind == "ci":
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise SystemExit(f"amendment 6 table {tid}: {key} = {v!r} is not an interval")
        lo, hi = (_a6_number(tid, key, x) for x in v)
        if lo is None or hi is None:
            return "n/a"
        if ci_break:
            return f"$[{ps.fmt_num(lo, 2, True)}$, ${ps.fmt_num(hi, 2, True)}]$"
        return f"${ps.fmt_ci([lo, hi], 2, True)}$"
    if kind == "seeds":
        if not isinstance(v, dict) or not all(re.fullmatch(r"[0-9]+", str(s)) for s in v):
            raise SystemExit(f"amendment 6 table {tid}: {key} = {v!r} is not a value per seed")
        return ", ".join(f"s{int(s)} {_a6_signed(tid, key, x)}" for s, x in sorted(v.items(), key=lambda e: int(e[0])))
    if kind == "mm":
        return _a6_signed(tid, key, v)
    x = _a6_number(tid, key, v)
    if x is None:
        return "n/a"
    if kind == "mm_abs":
        return ps.fmt_num(x, 2, False) if x >= 0 else f"${ps.fmt_num(x, 2, True)}$"
    if kind == "dice":
        return ps.fmt_num(x, 3, False)
    # p: two-sided, labeled "unadjusted; descriptive" (the header), exact or marked Monte Carlo
    if not 0.0 <= x <= 1.0 or row.get("p_signflip_label") != A6_P_LABEL:
        raise SystemExit(f"amendment 6 table {tid}: p = {v!r} with label {row.get('p_signflip_label')!r}")
    return f"${_p_tex(x)}$" + ("$^{\\mathrm{MC}}$" if _a6_p_method(tid, row) == A6_P_MC else "")


def _a6_p_method(tid, row):
    m = str(row.get("signflip_method"))
    if m == A6_P_EXACT:
        return A6_P_EXACT
    if m.startswith(A6_P_MC + " ("):
        return A6_P_MC
    raise SystemExit(f"amendment 6 table {tid}: sign-flip method {m!r}")


def _a6_table(item, tid, tab, words=None, titles=None, label_prefix="tab:amend6-", ci_break=False, widths=None):
    """One amendment 6 table (amend6.py 'tables' entry: note and rows) as a float: the A6_COLUMNS of its rows in
    A6_COLUMNS order, each cell formatted by its kind; the caption is the table's title, its label and
    the note. A key outside A6_COLUMNS and A6_NOT_PRINTED, or a table id outside A6_TABLES, is refused. The keywords
    default to the amendment 6 item files (A6_WORDS, A6_TABLES, tab:amend6-, unbroken intervals); est_tables() passes
    the floats-only words, titles and labels of the paper (EST_WORDS, EST_TABLES, tab:est-, ci_break). A title of
    four parts adds its fourth to the caption, and a float that prints a dagger says what it marks. widths gives
    the X columns relative widths per key (est_tables: EST_WIDTHS); the default keeps them equal."""
    words = A6_WORDS if words is None else words
    titles = A6_TABLES if titles is None else titles
    rows = [r for r in tab.get("rows") or [] if isinstance(r, dict)]
    if not rows:
        return []
    if tid not in titles:
        raise SystemExit(f"amendment 6 table {tid} has no printed title (make_tables A6_TABLES)")
    seen = {k for r in rows for k in r}
    unknown = sorted(seen - set(A6_COLUMNS) - set(A6_NOT_PRINTED))
    if unknown:
        raise SystemExit(f"amendment 6 table {tid}: no printed format for {unknown} (make_tables A6_COLUMNS)")
    for r in rows:
        if "n" in r and r.get("sign_n") != r["n"]:
            raise SystemExit(f"amendment 6 table {tid}: n = {r['n']!r} is printed as the n of k/n "
                             f"({r.get('sign_n')!r})")
    keys = [k for k in A6_COLUMNS if k in seen]
    title, prefix, clause = titles[tid][:3]
    suffix = titles[tid][3] if len(titles[tid]) > 3 else ""
    mc = sorted({str(r["signflip_method"]) for r in rows if r.get("p_signflip") is not None and
                 _a6_p_method(tid, r) == A6_P_MC})
    body = [" & ".join(_a6_cell(tid, k, r, words=words, ci_break=ci_break) for k in keys) + " \\\\" for r in rows]
    priv = " " + EST_DAGGER + "Privileged." if EST_DAGGER in title + suffix + "".join(body) else ""
    caption = (f"\\caption{{{title} ({prefix}{A6_CAPTION_LABEL}{'; ' + clause if clause else ''}): "
               f"{_a6_tex(tab.get('note') or '')}."
               + "".join(f" $^{{\\mathrm{{MC}}}}${_a6_tex(m).replace('sign-flip p', 'sign-flip $p$')}." for m in mc)
               + suffix + priv + "}")
    head = " & ".join(A6_COLUMNS[k][0] for k in keys) + " \\\\"
    if widths is None:
        colspec = "@{}" + ">{\\raggedright\\arraybackslash}X " * len(keys) + "@{}"
    else:   # relative X widths, scaled so that they sum to the column count (the tabularx \hsize rule)
        w = [float(widths.get(k, 1.0)) for k in keys]
        w = [round(x * len(w) / sum(w), 4) for x in w]
        w[-1] = round(len(w) - sum(w[:-1]), 4)
        colspec = "@{}" + "".join(f">{{\\raggedright\\arraybackslash\\hsize={x:.4f}\\hsize}}X " for x in w) + "@{}"
    L = _table_head(label_prefix + tid.replace("_", "-"), caption, colspec, head, star=True)
    return L + body + _table_foot(star=True)


def amend6_tables(out):
    """tables/amend6_<item>.tex (eight) and tables/amend6_records.tex."""
    if PI.skip_dev("the record tables of the last follow-up analyses (not input by the article; they need "
                   "sentences of the plan documents)"):
        return []
    log("\n== amendment 6: tables and record sentences (exploratory; planned after results; estimation only) ==")
    plan_rows = _a6_plan_rows()
    written = []
    blinded = any(_blinded_a345(os.path.join(AP_DIR, f"amend6_{i}.json")) for i in A6_ITEMS)
    if blinded:
        for i in A6_ITEMS:
            L = [GEN.format(src=f"nothing (amendment 6 blinded in this build; outputs/v2_1/analysis_plan/"
                                f"amend6_{i}.json is not read)"),
                 f"\\textit{{{A6_ITEM_TITLES[i]}: {A2_BLIND_TXT}.}}"]
            written.append(write(out, f"amend6_{i}.tex", "\n".join(L) + "\n"))
        log("amend6_<item>.tex: amendment 6 blinded in this build; one line per item")
        L = [GEN.format(src=_T("a6 records src"))]
        for rid, row in plan_rows.items():
            L.append(f"\\newcommand{{\\AmendSixRec{rid}}}{{{_a6_tex(_a6_earlier(row))}}}")
            L.append(f"\\newcommand{{\\AmendSixHist{rid}}}{{}}")
            L.append(f"\\newcommand{{\\AmendSixRecForm{rid}}}{{}}")
        written.append(write(out, "amend6_records.tex", "\n".join(L) + "\n"))
        return written
    outs = {i: _a345(f"amend6_{i}.json") for i in A6_ITEMS}
    for i, d in outs.items():
        if d.get("item") != ("LC+LA" if i == "look" else i.upper()):
            raise SystemExit(f"amend6_{i}.json: item {d.get('item')!r}")
        L = [GEN.format(src=f"outputs/v2_1/analysis_plan/amend6_{i}.json ({_T('src a6 record')})")]
        for s in d.get("printed_sentences") or []:
            if s.get("id") == "preface_cohort_rank":
                continue                                   # \AmendSixPrefaceRank in amend6_records.tex
            L.append(f"% {s.get('id')} ({s.get('kind')})")
            L.append(_a6_tex(s["text"], s.get("table_ref"), i))
        for tid in sorted(d.get("tables") or {}):
            L += _a6_table(i, tid, d["tables"][tid])
        written.append(write(out, f"amend6_{i}.tex", "\n".join(L) + "\n", verbatim=A6_VERBATIM.get(i, ())))
    by_id = {}
    for i, d in outs.items():
        for r in d.get("record_sentences") or []:
            if r.get("id") in by_id:
                raise SystemExit(f"record sentence {r.get('id')} appears in two amendment 6 outputs")
            by_id[r.get("id")] = (i, r)
    if sorted(by_id) != sorted(A6_RECORD_IDS):
        raise SystemExit(f"amendment 6 record sentences: {sorted(by_id)}, not {list(A6_RECORD_IDS)}")
    L = [GEN.format(src="outputs/v2_1/analysis_plan/amend6_<item>.json record_sentences and printed_sentences "
                        f"({_T('src a6 sentences')})")]
    for rid in A6_RECORD_IDS:
        i, r = by_id[rid]
        if r.get("earlier_text") != _a6_earlier(plan_rows[rid]):
            raise SystemExit(f"record sentence {rid}: its earlier text differs from the amendment's Appendix")
        hist = r.get("history") if r.get("superseded") else None
        L.append(f"\\newcommand{{\\AmendSixRec{rid}}}{{{_a6_tex(r.get('printed') or '', item=i)}}}")
        L.append(f"\\newcommand{{\\AmendSixHist{rid}}}{{{_a6_tex(hist, item=i) if hist else ''}}}")
        form = r.get("not_run_form") if not r.get("superseded") else None
        L.append(f"\\newcommand{{\\AmendSixRecForm{rid}}}{{{_a6_tex(form, item=i) if form else ''}}}")
    ranks = [s for d in outs.values() for s in d.get("printed_sentences") or [] if s.get("id") == "preface_cohort_rank"]
    if ranks:
        L.append(f"\\newcommand{{\\AmendSixPrefaceRank}}{{{_a6_tex(ranks[0]['text'])}}}")
    written.append(write(out, "amend6_records.tex", "\n".join(L) + "\n"))
    return written


# ================================================================ estimation-only floats of the paper
# Supplementary S7 (supplementary.tex, label supp:est) inputs tables/est_<item>.tex: the floats of the amendment 6
# outputs in the words of the article ("servo base" for the linear servo, a dagger on every privileged term, no codes),
# labels tab:est-<table id>. The amend6_<item>.tex files above are not input by the article. The hand-written
# sentences that print these outputs are checked by _est_check() first.
EST_DAGGER = "$^{\\dagger}$"
_D = EST_DAGGER
EST_FLOAT_ORDER = {"g2": ("g2_contrasts",), "p9": ("p9_rung12", "p9_rung36", "p9_cohort"), "l10": ("l10_contrasts",),
                   "c30": ("c30_c", "c30_d", "c30_e"), "r13": ("r13_contrasts",), "look": ("look_contrasts",)}
# the line of a blinded build, per item (the S7 subsection titles)
EST_ITEM_TITLES = {"g2": "Tissue-anchored speckle in the gap-cause test", "p9": "Further pose-free rungs",
                   "l10": "Label controls of the per-tissue final policy",
                   "c30": "Policies at 36 steps, contact off on the cohort and angiography CTs",
                   "r13": "Planted-deficit control with a larger bias",
                   "look": "Look-ahead sampling seeds 1 and 2 on the cohorts"}
# Table S7.1 holds a contrast with the grid-fixed bound, so its renderer clause covers the anchored mode only; the
# clauses of S7.1 and S7.6 print as caption sentences of their own, so that no caption sentence reaches 36 words
_EST_ANCHORED_CLAUSE = (" Every result under the anchored mode is conditional on this renderer with tissue-anchored speckle: no "
                  "elevational beam width, rigid tissue.")
_EST_LOOK_CLAUSE = " Both rows are conditional on this renderer: grid-fixed speckle, no elevational beam width, rigid tissue."
# table id -> (title, label prefix, closing clause, caption suffix); {dropped} is the number of rows EST_DROP leaves out
EST_TABLES = {
    "g2_contrasts": (f"Tissue-anchored speckle: the look-ahead bound{_D} against the grid-fixed bound{_D}, the CT-mask "
                     f"servo base{_D} and the pose oracle{_D}, six held-out patients", "", "", _EST_ANCHORED_CLAUSE),
    "p9_rung12": ("Pose-free rungs at 12 steps minus the servo base of the same stage, masks and seeds, six held-out "
                  "patients", "", "",
                  " Under CT masks, the MLP base is read for physics only: its fit for each other stage starts from "
                  "other random weights, so its {dropped} rows for those stages are not shown."),
    "p9_rung36": ("Pose-free rungs at 36 steps minus the 36-step servo base, physics frames, six held-out patients",
                  "", "", ""),
    "p9_cohort": ("Pose-free rungs on 20 new CTs minus the servo base of the same stage, masks and seeds", "", "", ""),
    "l10_contrasts": ("Learned-mask final policy of the per-tissue stage: label controls in the closed loop, six "
                      "held-out patients", "secondary; ", "", ""),
    "c30_c": (f"CT-mask policies{_D} evaluated at 36 steps minus the 36-step CT-mask servo base{_D}, six held-out "
              "patients", "descriptive, out of distribution; ", "", ""),
    "c30_d": ("Contact off minus contact on, per rung, 20 new CTs", "", "", ""),
    "c30_e": (f"Angiography CTs (a descriptive pool, never combined with the cohort): learned minus CT{_D} masks, "
              "SonoSPADE minus physics and the change under REINFORCE per policy arm", "", "", ""),
    "r13_contrasts": ("REINFORCE control with the bias selected on training patients: the planted lateral deficit at "
                      "iteration 0 ($D_p$) and the recovery statistic $x_p$ per setting, six held-out patients",
                      "", "", f" Every run starts from the CT-mask policy{_D} of the physics stage after imitation."),
    "look_contrasts": (f"Look-ahead sampling seeds 0 to 2 averaged: the look-ahead bound{_D} minus the mean of the "
                       f"CT-mask servo base{_D} over its calibration seeds, on both cohorts", "", "", _EST_LOOK_CLAUSE)}
# rows left out of the printed tables: with CT masks only the physics row of the MLP base is printed, and the rows
# of the other stages appear only in the invariance sentence of Supplementary S7.2
EST_DROP = {"p9_rung12": lambda r: r.get("rung") == "mlp_base" and r.get("mask") == "ct" and r.get("stage") != "physics"}
_EST_ARM = {"ctmask": ("CT-mask", _D), "matched": ("learned-mask", "")}
_EST_BIAS = {"shipped": "default", "explore": "learned-exploration", "reward": f"pose-shaping{_D}"}
EST_WORDS = {
    "stage": _A6_STAGE, "mask": {"ct": f"CT{_D}", "matched": "learned"},
    "rung": dict({"sguard": "$S$-guarded servo", "mlp_base": "MLP base", "cnn_base": "CNN base", "hillclimb": "hill-climb",
                  "static": "static", "scripted": "scripted", "pose_oracle": f"pose oracle{_D}",
                  "base_ct_physics": f"CT-mask servo base{_D}, physics",
                  "base_ct_sonospade_s0": f"CT-mask servo base{_D}, SonoSPADE, seed 0"},
                 **{f"base_matched_{st}": f"learned-mask servo base, {w}" for st, w in _A6_STAGE.items()},
                 **{f"{a}_{st}/{wt}": f"{aw} policy{d}, {w}, {ww}" for a, (aw, d) in _EST_ARM.items()
                    for st, w in _A6_STAGE.items()
                    for wt, ww in (("imit", "imitation start"), ("final", "final weights"))}),
    "set": {"cohort": "replication cohort", "amos": "AMOS"},
    "row": A6_WORDS["row"],
    "contrast": dict(
        {k: v for k, v in A6_CONTRASTS.items() if "x2" not in k and "type" not in k and " bound minus " not in k
         and not k.startswith(tuple(_A6_BIAS_SET))},
        **{"indep bound minus grid-fixed bound":
           f"independent speckle fields: look-ahead bound{_D} minus grid-fixed bound{_D}",
           **dict(zip(("C1-type", "C2-type"), (f"learned minus CT{_D} masks, physics",
                                               "SonoSPADE minus physics, learned-mask servo base")))},
        **{f"{o} bound minus CT-mask base": f"{w}: look-ahead bound{_D} minus CT-mask servo base{_D}"
           for o, w in _A6_FIELDS.items()},
        **{f"{o} bound minus oracle": f"{w}: look-ahead bound{_D} minus pose oracle{_D}" for o, w in _A6_FIELDS.items()},
        **{f"C3-type {a}/{st}": f"REINFORCE change, {aw} policy{d}, {w}" for a, (aw, d) in _EST_ARM.items()
           for st, w in _A6_STAGE.items()},
        **{f"{v} deficit D_p": f"{w} settings: deficit $D_p$" for v, w in _EST_BIAS.items()},
        **{f"{v} x_p": f"{w} settings: recovery $x_p$" for v, w in _EST_BIAS.items()})}
EST_ROOT = PAPER
# relative X-column widths of the est floats (with equal columns the per-seed and
# code cells wrapped to 3 or 4 lines, and Tables S7.2 and S7.7 were taller than the text block; the Note cell of
# Table S7.1 gets its own weight, or the scaling narrows it to 8 lines)
EST_WIDTHS = {"contrast": 2.4, "rung": 1.5, "stage": 1.05, "mask": 0.8, "set": 1.2, "E": 0.7, "mean": 0.7,
              "sign_k": 0.6, "p_signflip": 0.9, "per_seed": 2.0, "note": 1.8}


def est_tables(out):
    """tables/est_<item>.tex (six): the floats of the amendment 6 outputs that Supplementary S7 inputs, in
    EST_FLOAT_ORDER, with EST_WORDS and EST_TABLES. While BLIND_A6 is on, each holds one blinded line. Otherwise
    _est_check() runs first: the paper's sentences that print these outputs must print what the outputs give."""
    written = []
    if any(_blinded_a345(os.path.join(AP_DIR, f"amend6_{i}.json")) for i in A6_ITEMS):
        for i in EST_FLOAT_ORDER:
            L = [GEN.format(src=f"nothing (amendment 6 blinded in this build; outputs/v2_1/analysis_plan/"
                                f"amend6_{i}.json is not read)"),
                 f"\\textit{{{EST_ITEM_TITLES[i]}: {A2_BLIND_TXT}.}}"]
            written.append(write(out, f"est_{i}.tex", "\n".join(L) + "\n"))
        log("est_<item>.tex: amendment 6 blinded in this build; one line per item")
        return written
    outs = {i: _a345(f"amend6_{i}.json") for i in A6_ITEMS}
    _est_check(outs)
    for i, tids in EST_FLOAT_ORDER.items():
        L = [GEN.format(src=f"outputs/v2_1/analysis_plan/amend6_{i}.json (tables only{_T('est src note')})")]
        for tid in tids:
            tab = _est_rows(tid, outs[i]["tables"][tid])
            L += _a6_table(i, tid, tab, words=EST_WORDS, titles=tab["titles"], label_prefix="tab:est-",
                           ci_break=True, widths=EST_WIDTHS)
        written.append(write(out, f"est_{i}.tex", "\n".join(L) + "\n"))
    log("est_<item>.tex: " + ", ".join(f"{i} {len(t)}" for i, t in EST_FLOAT_ORDER.items()) + " floats")
    return written


def _est_rows(tid, tab):
    """The rows of a table as the paper prints them (EST_DROP), its note with the count of the rows shown, and the
    titles with the caption suffix filled."""
    rows = [r for r in tab.get("rows") or [] if isinstance(r, dict)]
    keep = [r for r in rows if not EST_DROP.get(tid, lambda r: False)(r)]
    note = tab.get("note") or ""
    title = EST_TABLES[tid]
    if len(keep) != len(rows):
        m = re.fullmatch(r"(\d+) contrasts; none adjusted", note)
        if not m or int(m.group(1)) != len(rows):
            raise SystemExit(f"est table {tid}: note {note!r} does not count its {len(rows)} rows")
        note = f"{len(keep)} contrasts shown; none adjusted"
        title = title[:3] + (title[3].replace("{dropped}", str(len(rows) - len(keep))),)   # no str.format: a suffix may hold TeX braces
    elif "{dropped}" in title[3]:
        raise SystemExit(f"est table {tid}: no row left out, but its caption says some are")
    return {"note": note, "rows": keep, "titles": {tid: title}}


# ================================================================== the guard of the hand-written sentences
def _est_fmt():
    """Number forms of the sentences (plan_stats.fmt_num): signed in math, magnitudes plain, an unsigned value in math
    only when negative, a percent, and a grid constant as the paper prints it."""
    f = _plan_stats().fmt_num

    def s(x, d=2):                       # signed difference
        return f"${f(x, d, True)}$"

    def m(x, d=2):                       # magnitude whose direction the words carry
        return f(abs(x), d, False)

    def u(x, d=2):                       # unsigned value, in math when negative
        t = f(x, d, False)
        return f"${f(x, d, True)}$" if t.startswith("-") else t

    def pc(x):                           # a fraction as a whole percent, as the outputs print it
        return f"{100.0 * x:.0f}"

    def c(x):                            # a grid constant: 0, 0.005, ..., 1.0
        return "1.0" if float(x) == 1.0 else f"{float(x):g}"
    return s, m, u, pc, c


def _est_need(ok, what):
    if not ok:
        raise SystemExit(f"est check: {what}")


def _est_sentences(outs):
    """The sentences of the paper that print the amendment 6 outputs, rebuilt from their keys: (file under the paper tree,
    sentence, count). Part 1 (article) and Part 2 (Supplementary S7 and S1.5). A second form of a sentence is a
    contingency of the page budget: (file, sentence, count, [(other file, other sentence)])."""
    s, m, u, pc, c = _est_fmt()
    ST = ("physics", "cut", "cyclegan", "sonospade")
    W = _A6_STAGE
    S7, AE, CH, LR = "supplementary.tex", "sections/appendix_eval.tex", "sections/11_checks.tex", \
        "sections/05_results_ladder.tex"
    E = []
    # == completion: the three parts the paper reports as not reported
    comp = {"x2": outs["l10"]["completion"]["parts"]["x2"], "sp": outs["sp"]["completion"]["parts"]["sp"],
            "td": outs["td"]["completion"]["parts"]["td"]}
    for k, p in comp.items():
        _est_need(p.get("state") == "refused" and not p.get("completed"), f"{k} is {p.get('state')!r}, but the paper "
                  "reports it as not reported")
    # and no other part is open, so the S7 lead names every part that is not reported
    open_ = {(i, k) for i, d in outs.items() for k, p in ((d.get("completion") or {}).get("parts") or {}).items()
             if p.get("state") != "completed" or not p.get("completed") or not p.get("checks_passed")}
    _est_need(open_ == {("l10", "x2"), ("sp", "sp"), ("td", "td")}, f"the parts not completed are {sorted(open_)}, "
              "but the S7 lead names only l10 x2, sp and td as not reported")
    E += [("sections/appendix_decisions.tex", "\\item Its independent-speckle condition is not reported.", 1),
          ("sections/appendix_decisions.tex", "\\item Fast mode is coarser than the 1.5~mm default, which is not "
                                              "reported.", 1),
          (S7, "Of this set, the painted stage's variant that doubles both painted contrasts, the independent-speckle "
               "condition of the LPIPS reward and the 1.5~mm default mode of TotalSegmentator are not reported.", 1),
          (S7, "The variant of the painted stage that doubles both painted contrasts is not reported.", 0)]
    # == contact off on the cohort (body, Section 6.4 of the article; S7.4)
    C = outs["c30"]["results"]
    b = C["d"]["base_ct_physics"]
    pol = [C["d"][f"ctmask_{st}/final"]["mean"] for st in ST]
    _est_need(b["mean"] > 0 and b["ci_t"][0] > 0 and min(pol) > 0, "contact off is printed as a rise")
    tail = (f"raises the final error by {m(b['mean'])}~mm [{m(b['ci_t'][0])}, {m(b['ci_t'][1])}] "
            f"({b['sign_k']}/{b['sign_n']}) for the CT-mask servo base and {m(min(pol))} to {m(max(pol))}~mm for "
            "CT-mask final policies")                       # the range is of the final weights (results.d .../final)
    E.append((LR, f"On {b['n']} new CTs, contact off {tail} (exploratory; estimation only; Supplementary S7).", 1,
              [(S7, f"On these CTs, contact off {tail} (exploratory; estimation only).")]))
    # == the larger bias (body, Section 8.2)
    R = outs["r13"]["results"]
    grid = R["selection"]["grid"]
    gk = sorted(grid, key=float)
    B = R["selection"]["B"]
    _est_need(float(gk[-1]) == B and not R["selection"]["qualified"] and not any(g["qualifies"] for g in grid.values()),
              "the bias is the largest grid value, none qualifying")
    dS = grid[gk[-1]]["final_S_change"]
    lat = R["lateral"]
    D_ = lat["shipped"]["D"]
    _est_need(all(abs(lat[v]["D"]["mean"] - D_["mean"]) < 1e-9 for v in lat), "one deficit for the three settings")
    fr = {v: lat[v]["fraction"] for v in ("shipped", "explore", "reward")}
    kpos = {v: sum(1 for y in lat[v]["x"]["per_unit"].values() if y > 0) for v in fr}
    _est_need(len(set(kpos.values())) == 1 and len({lat[v]["x"]["n"] for v in fr}) == 1,
              "one count of patients with a positive recovery statistic for the three settings")
    E += [(CH, f"Raised to {c(B)}~mm per step, the largest value tried on TRAIN, the planted bias still lowers the "
               f"final match score by only {m(dS, 3)}.", 1),
          (CH, f"At this bias, REINFORCE removes {pc(fr['shipped']['estimate'])}\\%, {pc(fr['explore']['estimate'])}\\% "
               f"and {pc(fr['reward']['estimate'])}\\% of the {u(D_['mean'])}~mm lateral deficit with default settings, "
               "learned exploration and privileged pose shaping (exploratory; estimation only; Supplementary S7).", 1),
          (CH, "Their patient-bootstrap intervals are " + ", ".join(
              f"[{pc(fr[v]['ci'][0])}\\%, {pc(fr[v]['ci'][1])}\\%]" for v in ("shipped", "explore"))
           + f" and [{pc(fr['reward']['ci'][0])}\\%, {pc(fr['reward']['ci'][1])}\\%], and the recovery statistic is "
             f"positive in {kpos['shipped']} of {lat['shipped']['x']['n']} patients for every setting.", 1)]
    # == look-ahead seeds 0 to 2 (body, Section 8.5 of the article; App. B.9)
    K = outs["look"]["results"]
    sd = sorted(K["cohort"]["seeds_used"])
    _est_need(sd == sorted(K["amos"]["seeds_used"]) and sd == list(range(sd[0], sd[-1] + 1)), "look-ahead seeds")
    for st_, n_ in (("cohort", 20), ("amos", 30)):
        x = K[st_]["contrast"]
        _est_need(x["mean"] < 0 and x["ci_t"][1] < 0 and x["sign_n"] == n_ and
                  abs(K[st_]["bound_seed_mean"] - K[st_]["base_mean"] - x["mean"]) < 1e-6, f"look-ahead {st_}")
        E.append((CH, f"With sampling seeds {sd[0]} to {sd[-1]} averaged, it ends at {u(K[st_]['bound_seed_mean'])}~mm, "
                      f"{m(x['mean'])}~mm [{m(x['ci_t'][1])}, {m(x['ci_t'][0])}] below the servo base "
                      f"({x['sign_k']}/{x['sign_n']}; exploratory; estimation only; conditional on this renderer; "
                      "Supplementary S7).", 1))
    E += [(CH, f"This bound uses look-ahead sampling seed {sd[0]}, as pre-specified.", 2),
          ("sections/appendix_decisions.tex", f"This bound uses look-ahead sampling seed {sd[0]}, as pre-specified; "
           f"Section~\\ref{{sec:checks-cohort}} gives it with sampling seeds {sd[0]} to {sd[-1]} averaged (estimation "
           "only).", 1)]
    # == tissue-anchored speckle (App. A.2; S1.5; S7.1)
    G = outs["g2"]["results"]
    ls, bd = G["levelset"], G["bounds"]
    lat_mm = u(ls["lattice_h_mm"])
    npat = len(ls["shared"]["per_patient"])
    gm = {o: sum(bd[o]["bound_seed_mean_per_patient"].values()) / len(bd[o]["bound_seed_mean_per_patient"])
          for o in ("onefield", "indep")}
    gf = bd["indep"]["bound_minus_grid_fixed"]
    _est_need(gf["mean"] < 0, "the independent-field bound is printed below the grid-fixed one")
    add = (f"With speckle anchored to tissue, {ls['shared']['A']}/{npat} patients are ambiguous at 5~mm and the bound "
           f"ends at {u(gm['onefield'])}~mm (exploratory).")
    printed = {x["id"]: x["text"] for x in outs["g2"]["printed_sentences"]}
    _est_need(re.sub(r"(\d) mm\b", r"\1~mm", printed["g2_add"]) == add, "App. A.2 prints g2_add verbatim")
    E += [(AE, f"The gap-cause test was also run with speckle fixed to tissue on a {lat_mm}~mm lattice, with observation and "
               "target sharing one anchored field or drawing independent ones (estimation only; Supplementary S7).", 1),
          (AE, add, 1),
          (AE, "With the target drawn from an independent anchored field, so that speckle carries no match information, "
               f"{ls['independent']['A']}/{npat} patients are ambiguous at 5~mm and the bound ends at "
               f"{u(gm['indep'])}~mm.", 1),
          (AE, f"With grid-fixed speckle, {G['grid_fixed']['k1']}/{npat} were ambiguous and the bound ended at "
               f"{u(G['grid_fixed']['mean_mm'])}~mm, and the independent-field bound ends {m(gf['mean'])}~mm "
               f"[{u(-gf['ci_t'][1])}, {u(-gf['ci_t'][0])}] below it ({gf['sign_k']}/{gf['sign_n']}; exploratory; "
               "estimation only; Supplementary S7).", 1),
          ("sections/appendix_renderer.tex", "An opt-in tissue-anchored mode, used only in one variant of the gap-cause "
           f"test (Section~\\ref{{supp:est-g2}}), instead fixes the speckle to tissue on a {lat_mm}~mm lattice.", 1)]
    gsd = sorted(bd["onefield"]["bound_minus_base"]["seeds_used"])
    _est_need(gsd == sorted(bd["indep"]["bound_minus_base"]["seeds_used"]), "anchored-speckle bound seeds")
    sh, ind = ls["shared"], ls["independent"]
    nt = sum(p["n_targets"] for p in sh["per_patient"].values())
    _est_need(sh["n_targets_ambiguous"] == ind["n_targets_ambiguous"] and sh["F"] == ind["F"] and sh["A"] == ind["A"],
              "the two anchored level sets are printed as equal")
    E += [(S7, "The bound is privileged look-ahead ascent on the CT-mask match score, averaged over look-ahead sampling "
               f"seeds {gsd[0]} to {gsd[-1]}.", 1),
          (S7, "With one speckle field anchored to tissue, poses 5~mm or more from the target score within 0.01 of the "
               f"target's own score for {sh['n_targets_ambiguous']}/{nt} ({pc(sh['F'])}\\%) of targets, as with "
               "independent anchored fields.", 1),
          (S7, f"Both have {sh['A']}/{npat} patients with a majority, with margins of {u(sh['margin_mean'], 3)} "
               f"[{u(sh['margin_ci_t'][0], 3)}, {u(sh['margin_ci_t'][1], 3)}] and {u(ind['margin_mean'], 3)} "
               f"[{u(ind['margin_ci_t'][0], 3)}, {u(ind['margin_ci_t'][1], 3)}] (exploratory; estimation only).", 1)]
    for o, lead in (("onefield", "With one speckle field anchored to tissue"),
                    ("indep", "With independent speckle fields anchored to tissue")):
        bb, bo = bd[o]["bound_minus_base"], bd[o]["bound_minus_oracle"]
        _est_need(bb["mean"] < 0 and bo["mean"] > 0 and bb["sign_k"] == bo["sign_k"] and bb["sign_n"] == bo["sign_n"],
                  f"anchored speckle {o}: below the servo base and above the pose oracle, one count")
        E.append((S7, f"{lead}, the bound ends {m(bb['mean'])}~mm [{u(-bb['ci_t'][1])}, {u(-bb['ci_t'][0])}] below the "
                      f"servo base and {m(bo['mean'])}~mm [{u(bo['ci_t'][0])}, {u(bo['ci_t'][1])}] above the pose oracle "
                      f"({bb['sign_k']}/{bb['sign_n']} each; exploratory; estimation only).", 1))
    # == further pose-free rungs (S7.2)
    P = outs["p9"]["results"]
    r12, r36, coh = P["rung12"], P["rung36"], P["cohort"]
    sel = {(x["stage"], x["mask"]): x["selected"] for x in P["pilot_selection"]["guard_eps"]}
    sig = {(x["stage"], x["mask"]): x["selected"] for x in P["pilot_selection"]["es_sigma0"]}
    _est_need(sel[("physics", "ct")] == sel[("cut", "ct")] == sel[("cyclegan", "ct")] and
              sig[("cut", "ct")] == sig[("cyclegan", "ct")] == sig[("sonospade", "ct")], "the grouped pilot selections")
    per = lambda cells: ", ".join(f"{s(cells[i])} ({W[st]})" for i, st in enumerate(ST[:-1])) + \
        f" and {s(cells[-1])} ({W[ST[-1]]})"                                  # noqa: E731
    lst = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1]                     # noqa: E731
    mean = lambda d: d["mean"]                                                  # noqa: E731
    E += [(S7, f"Under CT masks, the selected $\\epsilon$ is {c(sel[('physics', 'ct')])} with physics, CUT and CycleGAN "
               f"frames and {c(sel[('sonospade', 'ct')])} with SonoSPADE frames.", 1),
          (S7, "Under learned masks, it is " + lst([c(sel[(st, 'matched')]) for st in ST])
           + " with physics, CUT, CycleGAN and SonoSPADE frames.", 1),
          (S7, f"The selected $\\sigma_0$ is {c(sig[('cut', 'ct')])} under CT masks with each of the three stages, and "
               + lst([c(sig[(st, 'matched')]) for st in ST[1:]]) + " under learned masks with CUT, CycleGAN and "
               "SonoSPADE frames.", 1),
          (S7, "Under CT masks, the $S$-guarded servo ends " + per([mean(r12[f"sguard/{st}/ct"]) for st in ST])
           + "~mm from the servo base of the same stage and seeds (Table~\\ref{tab:est-p9-rung12}; exploratory; "
             "estimation only).", 1),
          (S7, "Under learned masks, it ends " + per([mean(r12[f"sguard/{st}/matched"]) for st in ST])
           + "~mm from it (exploratory; estimation only).", 1),
          (S7, f"Under CT masks, the MLP base ends {s(mean(r12['mlp_base/physics/ct']))}~mm from the servo base with "
               "physics frames (exploratory; estimation only).", 1),
          (S7, "Under learned masks, it ends " + per([mean(r12[f"mlp_base/{st}/matched"]) for st in ST])
           + "~mm from it (exploratory; estimation only).", 1),
          (S7, "Under CT masks, the CNN base ends " + per([mean(r12[f"cnn_base/{st}/ct"]) for st in ST])
           + "~mm from the servo base (exploratory; estimation only).", 1),
          (S7, "Under CT masks, the hill-climb ends " + ", ".join(
              f"{s(mean(r12[f'hillclimb/{st}/ct']))} ({W[st]})" for st in ST[1:-1])
           + f" and {s(mean(r12['hillclimb/sonospade/ct']))} ({W['sonospade']})~mm from the servo base, and under "
             "learned masks " + lst([s(mean(r12[f"hillclimb/{st}/matched"])) for st in ST[1:]])
           + "~mm (exploratory; estimation only).", 1)]
    inv = P["mlp_ct_invariance"]
    _est_need(set(inv) == set(ST[1:]) and all(x["n_above"] == x["n"] == 60 for v in inv.values() for x in v.values()),
              "the MLP base differs from physics in every episode")
    E += [(S7, "Under CT masks, the MLP base is not stage-invariant: with CUT, CycleGAN and SonoSPADE frames, its final "
               "error differs from physics by up to " + lst([u(max(x["largest"] for x in inv[st].values()))
                                                             for st in ST[1:]]) + "~mm per episode.", 1),
          (S7, "All 60 episodes of each stage and seed differ by more than $10^{-6}$~mm, because its fit starts from "
               "random weights drawn after the stage's translator is built.", 1)]
    t36 = {(x["rung"], x["mask"]): x for x in outs["p9"]["tables"]["p9_rung36"]["rows"]}
    base36 = {}
    for (rg, mk), x in t36.items():
        _est_need(abs(x["mean"] - r36[f"{rg}/{mk}"]["mean"]) < 1e-9, f"p9_rung36 {rg}/{mk}")
        base36.setdefault(mk, []).append(x["E"] - x["mean"])
    _est_need(all(max(v) - min(v) < 1e-6 for v in base36.values()), "one 36-step servo base per mask source")
    ct4, lm3 = ("sguard", "hillclimb", "mlp_base", "cnn_base"), ("sguard", "hillclimb", "mlp_base")
    ci = lambda d: f"{s(d['mean'])}~mm [{s(d['ci_t'][0])}, {s(d['ci_t'][1])}]"  # noqa: E731
    E += [(S7, "At 36 steps with physics frames, under CT masks, the $S$-guarded servo, the hill-climb, the MLP base and "
               "the CNN base end at " + lst([u(t36[(rg, 'ct')]["E"]) for rg in ct4])
           + "~mm (Table~\\ref{tab:est-p9-rung36}).", 1),
          (S7, "These are " + lst([ci(r36[f"{rg}/ct"]) for rg in ct4]) + " from the 36-step servo base ("
           + lst([f"{r36[f'{rg}/ct']['sign_k']}/{r36[f'{rg}/ct']['sign_n']}" for rg in ct4])
           + "; exploratory; estimation only).", 1),
          (S7, "Under learned masks, the $S$-guarded servo, the hill-climb and the MLP base end at "
           + lst([u(t36[(rg, 'matched')]["E"]) for rg in lm3]) + "~mm.", 1),
          (S7, "These are " + lst([ci(r36[f"{rg}/matched"]) for rg in lm3]) + " from the 36-step servo base ("
           + lst([f"{r36[f'{rg}/matched']['sign_k']}/{r36[f'{rg}/matched']['sign_n']}" for rg in lm3])
           + "; exploratory; estimation only).", 1),
          (S7, f"The 36-step servo base ends at {u(base36['ct'][0])}~mm under CT masks and {u(base36['matched'][0])}~mm "
               "under learned masks (Section~\\ref{sec:checks-horizon} of the article).", 1)]
    nc = coh["sguard/physics/ct"]["n"]
    E += [(S7, f"On {nc} new CTs, under CT masks, the $S$-guarded servo ends "
           + per([mean(coh[f"sguard/{st}/ct"]) for st in ST]) + "~mm from the servo base of the same stage "
             "(Table~\\ref{tab:est-p9-cohort}; exploratory; estimation only).", 1),
          (S7, "On these CTs, under learned masks, it ends " + per([mean(coh[f"sguard/{st}/matched"]) for st in ST])
           + "~mm from it (exploratory; estimation only).", 1),
          (S7, "On these CTs, under CT masks, the CNN base ends " + per([mean(coh[f"cnn_base/{st}/ct"]) for st in ST])
           + "~mm from the servo base (exploratory; estimation only).", 1)]
    # the base fit (R5): bands, R^2 and gains over the named cells
    r5 = P["r5"]
    WY = r5["mlp_base_physics_ct_s0"]["dof"].index("wy")
    band = lambda k: list(r5[k]["bands"].values())                              # noqa: E731
    vals = lambda ks, f, skip=None: [x for k in ks for j, x in enumerate(r5[k][f]) if (k, j) != skip]  # noqa: E731
    rg = lambda xs, d: f"{u(min(xs), d)} to {u(max(xs), d)}"                    # noqa: E731
    mct = [f"mlp_base_physics_ct_s{i}" for i in range(3)]
    _est_need(all(b_ == "barely predicts" for k in mct[:2] for b_ in band(k)) and
              [b_ for b_ in band(mct[2])].count("partly predicts") == 1 and band(mct[2])[WY] == "partly predicts",
              "MLP base, CT masks: barely at seeds 0 and 1, wy partly at seed 2")
    mlm = [f"mlp_base_{st}_matched_s0" for st in ("physics", "sonospade", "cut", "cyclegan")]
    _est_need(all(b_ == "barely predicts" for k in mlm for b_ in band(k)), "MLP base, learned masks: barely")
    lin = [f"ibvs_base_{st}_matched_s0" for st in ("sonospade", "cut", "cyclegan")]
    wyk = ("ibvs_base_sonospade_matched_s0", WY)
    _est_need(band(lin[0])[WY] == "partly predicts" and all(
        b_ == "barely predicts" for k in lin for j, b_ in enumerate(band(k)) if (k, j) != wyk),
        "servo base, learned masks: wy partly with SonoSPADE frames only")
    E += [(S7, "Under CT masks with physics frames, on held-out pairs the MLP base barely predicts every channel at "
               "calibration seeds 0 and 1, and partly predicts wy at seed 2 (R$^2$ "
               f"{u(r5[mct[2]]['heldout_r2'][WY], 3)}; fresh TRAIN pairs {u(r5[mct[2]]['train_fresh_r2'][WY], 3)}).", 1),
          (S7, f"At seed 2, its gain on wy shrinks to b = {u(r5[mct[2]]['heldout_gain'][WY])}.", 1),
          (S7, f"Over the three seeds, its held-out R$^2$ is {rg(vals(mct, 'heldout_r2'), 3)} (fresh TRAIN pairs "
               f"{rg(vals(mct, 'train_fresh_r2'), 3)}), and its gains shrink to b = {rg(vals(mct, 'heldout_gain'), 2)} "
               "(exploratory; estimation only).", 1),
          (S7, "Under learned masks at calibration seed 0, on held-out pairs the MLP base barely predicts all six "
               f"channels with each of the four stages (R$^2$ {rg(vals(mlm, 'heldout_r2'), 3)}; fresh TRAIN pairs "
               f"{rg(vals(mlm, 'train_fresh_r2'), 3)}).", 1),
          (S7, f"Its gains shrink to b = {rg(vals(mlm, 'heldout_gain'), 2)} (exploratory; estimation only).", 1),
          (S7, "Under learned masks at calibration seed 0, on held-out pairs the servo base partly predicts wy with "
               f"SonoSPADE frames (R$^2$ {u(r5[lin[0]]['heldout_r2'][WY], 3)}; fresh TRAIN pairs "
               f"{u(r5[lin[0]]['train_fresh_r2'][WY], 3)}) and its gain shrinks to b = "
               f"{u(r5[lin[0]]['heldout_gain'][WY])}.", 1),
          (S7, "It barely predicts every other channel with SonoSPADE, CUT and CycleGAN frames (R$^2$ "
               f"{rg(vals(lin, 'heldout_r2', wyk), 3)}; fresh TRAIN pairs {rg(vals(lin, 'train_fresh_r2', wyk), 3)}).", 1),
          (S7, f"Its gains on them shrink to b = {rg(vals(lin, 'heldout_gain', wyk), 2)} (exploratory; estimation only).",
           1)]
    # == L10: the final policy's label controls (S7.3)
    pcl = outs["l10"]["results"]["policy_controls"]
    e3, e6, pm = pcl["elev3"]["Delta_y"], pcl["elev6"]["Delta_y"], pcl["permute"]
    _est_need(e3["sign_k"] == e6["sign_k"] and e3["sign_n"] == e6["sign_n"], "one count for the two offsets")
    lp = {x["id"]: x["text"] for x in outs["l10"]["printed_sentences"]}
    base = [lp[f"l10_pol_off:{e}"].split(" mm for its learned-mask servo base")[0].rsplit("against ", 1)[1]
            for e in ("elev3", "elev6")]
    _est_need(all(re.fullmatch(r"[-+]\d+\.\d+", x) for x in base), "the servo base shifts of l10_pol_off")
    E += [(S7, f"With the 3 and 6~mm label offsets applied to the observations only, its final elevational displacement "
               f"shifts by {ci(e3)} and {ci(e6)} ({e3['sign_k']}/{e3['sign_n']} each; secondary; exploratory; estimation "
               "only).", 1),
          (S7, f"For the servo base, the same shifts are ${base[0]}$ and ${base[1]}$~mm "
               "(Section~\\ref{sec:checks-label} of the article).", 1),
          (S7, f"With organ identities permuted, its total error changes by {ci(pm)} ({pm['sign_k']}/{pm['sign_n']}; "
               "secondary; exploratory; estimation only).", 1)]
    # == contact off, parts (c), (d) and (e) (S7.4)
    tc = {x["stage"]: x["E"] for x in outs["c30"]["tables"]["c30_c"]["rows"]}
    dv = [x["mean"] for x in C["d"].values()]
    fin = {st: C["d"][f"matched_{st}/final"] for st in ST}
    ef = lambda d: f"{s(d['mean'])}~mm [{s(d['ci_t'][0])}, {s(d['ci_t'][1])}]"  # noqa: E731
    e_ = C["e"]
    na = e_["n"]
    c3 = [x["mean"] for x in e_["C3_type"].values()]
    dmin, dmax = s(min([x["mean"] for x in C["d"].values()])), s(max([x["mean"] for x in C["d"].values()]))
    c1k, c1n, c2k, c2n = (e_["C1_type"]["sign_k"], e_["C1_type"]["sign_n"], e_["C2_type"]["sign_k"],
                          e_["C2_type"]["sign_n"])
    ap = {x["id"]: x["text"] for x in outs["c30"]["printed_sentences"]}
    angio = (f"These {na} angiography CTs come from the same public dataset as the replication cohort "
             f"({e_['m_institute_E']} of {na} from institute E); they are a descriptive pool, not a replication.")
    _est_need(ap["preface_angio"] == angio and e_["never_pooled"] is True, "S7.4 prints preface_angio verbatim")
    # the unit cell of Table S4.1's row Estimation-only runs (its 17 is an output)
    E.append(("sections/supp_index.tex", f"patient ({npat}); CT ({e_['n']}, {b['n']} and "
                                         f"{K['amos']['contrast']['sign_n']})", 1))
    E += [(S7, "Evaluated at 36 steps, outside the 12-step horizon they were trained for, the CT-mask policies end at "
           + ", ".join(f"{u(tc[st])} ({W[st]})" for st in ST[:-1]) + f" and {u(tc['sonospade'])} (SonoSPADE)~mm.", 1),
          (S7, "These are " + lst([s(C["c"][st]["mean"]) for st in ST]) + "~mm from the 36-step CT-mask servo base, "
               "which is stage-invariant (Table~\\ref{tab:est-c30-c}; descriptive, out of distribution; exploratory; "
               "estimation only).", 1),
          (S7, f"On {b['n']} new CTs, turning contact off changes the final error by {dmin} to {dmax}~mm "
               f"across the {len(dv)} rows of Table~\\ref{{tab:est-c30-d}} (exploratory; estimation only).", 1),
          (S7, f"For the learned-mask final policies, it changes the final error by {ef(fin['physics'])} with physics "
               f"frames ({fin['physics']['sign_k']}/{fin['physics']['sign_n']}) and {ef(fin['cut'])} with CUT frames "
               f"({fin['cut']['sign_k']}/{fin['cut']['sign_n']}).", 1),
          (S7, f"With CycleGAN and SonoSPADE frames, the changes are {ef(fin['cyclegan'])} "
               f"({fin['cyclegan']['sign_k']}/{fin['cyclegan']['sign_n']}) and {ef(fin['sonospade'])} "
               f"({fin['sonospade']['sign_k']}/{fin['sonospade']['sign_n']}; exploratory; estimation only).", 1),
          (S7, f"Angiography studies are excluded from the replication cohort by study type, and the {e_['n']} "
               "angiography CTs of the dataset that pass the cohort's acceptance rule were evaluated as a separate "
               "pool.", 1),
          (S7, angio, 1),
          (S7, f"On {na} angiography CTs, learned masks with the physics stage change the final error by "
               f"{ef(e_['C1_type'])} over CT masks ({c1k}/{c1n}; exploratory; "
               "estimation only).", 1),
          (S7, f"SonoSPADE minus physics at the learned-mask servo base is {ef(e_['C2_type'])} "
               f"({c2k}/{c2n}), and REINFORCE changes the final error by "
               f"{s(min(c3))} to {s(max(c3))}~mm across the {len(c3)} policy arms (Table~\\ref{{tab:est-c30-e}}; "
               "exploratory; estimation only).", 1),
          (S7, f"On the {b['n']} cohort CTs, the same three values are {s(e_['cohort']['C1'])} and "
               f"{s(e_['cohort']['C2'])}~mm (Section~\\ref{{sec:checks-cohort}} of the article) and "
               f"{s(e_['cohort']['C3_range'][0])} to {s(e_['cohort']['C3_range'][1])}~mm (descriptive).", 1)]
    # == the larger bias (S7.5)
    E += [(S7, f"In Section~\\ref{{sec:checks-controls}} of the article, the {c(gk[0])}~mm per step bias of the "
               "planted-deficit control did not meet its reward-visibility rule on TRAIN, so the outcome is "
               "inconclusive.", 1),
          (S7, f"On TRAIN, no grid value lowered the final $S$ by 0.03 with an interval below 0, so $B = {c(B)}$~mm per "
               "step (exploratory; estimation only).", 1),
          (S7, "The final $S$ changes of the grid values " + lst([c(k) for k in gk]) + "~mm per step are "
           + lst([s(grid[k]["final_S_change"], 3) for k in gk]) + ".", 1),
          (S7, f"At iteration 0, the planted bias shifts the held-out lateral displacement by {ef(D_)} "
               f"({D_['sign_k']}/{D_['sign_n']}; Table~\\ref{{tab:est-r13-contrasts}}; exploratory; estimation "
               "only).", 1)]
    for v, lead, end in (("shipped", "With default settings, REINFORCE removes", " of the planted lateral deficit"),
                         ("explore", "With learned exploration, it removes", ""),
                         ("reward", "With privileged pose shaping, it removes", "")):
        x, f_ = lat[v]["x"], lat[v]["fraction"]
        npos = sum(1 for y in x["per_unit"].values() if y > 0)
        ps_ = [pc(lat[v]["fraction_per_seed"][k]) for k in sorted(lat[v]["fraction_per_seed"], key=int)]
        E.append((S7, f"{lead} {pc(f_['estimate'])}\\% [{pc(f_['ci'][0])}\\%, {pc(f_['ci'][1])}\\%]{end} (patient "
                      f"bootstrap; mean recovery statistic {ef(x)}, positive in {npos}/{x['n']}; per seed "
                      + lst(ps_) + "\\%" + ("; exploratory; estimation only" if v == "reward" else "") + ").", 1))
    E.append((S7, f"The planted deficit is only weakly visible in the reward (final $S$ change {s(dS, 3)}).", 1))
    # == LC and LA per seed (S7.6)
    pss = lambda st_: [s(K[st_]["contrast"]["per_seed"][str(k)]) for k in sd]  # noqa: E731
    E += [(S7, f"Section~\\ref{{sec:checks-cohort}} of the article gives the look-ahead bound of each cohort at sampling "
               f"seed {sd[0]} and with seeds {sd[0]} to {sd[-1]} averaged.", 1),
          (S7, "Per seed, the bound ends " + lst(pss("cohort")) + "~mm from the servo base on the replication cohort "
               "and " + lst(pss("amos")) + "~mm on AMOS (Table~\\ref{tab:est-look-contrasts}; exploratory; estimation "
               "only).", 1)]
    return E


# Lines of Supplementary S7 whose numbers are method constants or names, not outputs (Part 2 admits them as written)
EST_CONSTANT_LINES = (
    "\\subsection{Policies at 36 steps, contact off on the cohort and angiography CTs}\\label{supp:est-c30}",
    "\\subsection{Look-ahead sampling seeds 1 and 2 on the cohorts}\\label{supp:est-look}",
    "Differences are rung or condition minus its reference, signed unless the words give the direction, with the 95\\% "
    "paired $t$ CI and the count of units that share the sign of the mean.",
    "The MLP base replaces the least-squares map of the servo base with a multilayer perceptron of two hidden layers of "
    "64 units, fit to the same mask features and calibration pairs.",
    "The threshold $\\epsilon$ was chosen from 0, 0.005, 0.01, 0.02 and 0.04, and the hill-climb's initial step "
    "$\\sigma_0$ from 0.1, 0.25, 0.5 and 1.0.",
    "At 12 steps, each rung is compared with the servo base of the same stage, mask source and seeds; at 36 steps, with "
    "physics frames only, with the 36-step servo base.",
    "The 300~HU rule relabels opacified vessels as bone in these angiography CTs, a documented ingestion limit "
    "(Table~\\ref{tab:renderer-cohort}), so their rows measure the pipeline under that limit.",
    "By construction, the imitation anchor (weight 0.3) pulls the policy toward its teacher, the servo base without the "
    "bias, which works against recovery.")
_EST_REFS = re.compile(r"\\(?:ref|label|input|cite)\{[^}]*\}")


def _est_check(outs, root=None):
    """The guard of the hand-written sentences that print amendment 6 (the _sign_check pattern). Part 1: each
    sentence of _est_sentences() prints in its file, outside comments, as often as it says (an article sentence with a
    contingency form passes when that form prints instead). Part 2: every line of Supplementary S7 (from
    \\label{supp:est} to the end of the supplement's sections) that holds a digit outside a reference is one of those
    sentences or of EST_CONSTANT_LINES, so no number of S7 is left without its keys. SystemExit names the first
    problem."""
    if root is None and PI.skip_dev("the sentences of Supplementary S7 in the article's source text"):
        return
    root = EST_ROOT if root is None else root
    texts = {}

    def body(rel_):
        if rel_ not in texts:
            p = os.path.join(root, rel_)
            if not os.path.isfile(p):
                raise SystemExit(f"est check: {rel_} is missing, so its sentences cannot be checked")
            texts[rel_] = [re.sub(r"(?<!\\)%.*$", "", ln).strip() for ln in open(p, encoding="utf-8").read().splitlines()]
        return texts[rel_]

    def count(rel_, t):
        return "\n".join(body(rel_)).count(t)

    want = _est_sentences(outs)
    expect = set()
    for e in want:
        rel_, t, n = e[:3]
        got = count(rel_, t)
        if got != n and len(e) > 3 and got == 0 and all(count(r2, t2) == 1 for r2, t2 in e[3]):
            expect.update(t2 for _r2, t2 in e[3])            # a contingency form of the page budget
            continue
        if got != n:
            raise SystemExit(f"est check: {rel_} prints {got} times, not {n}: {t!r}")
        expect.add(t)
    sup = body("supplementary.tex")
    i = next((j for j, ln in enumerate(sup) if "\\label{supp:est}" in ln), None)
    if i is None:
        raise SystemExit("est check: supplementary.tex has no \\label{supp:est}")
    k = next((j for j in range(i + 1, len(sup)) if sup[j].startswith("\\section") or
              sup[j].startswith("\\input{sections/refs}")), len(sup))
    for ln in sup[i:k]:
        if re.search(r"\d", _EST_REFS.sub("", ln)) and ln not in expect and ln not in EST_CONSTANT_LINES:
            raise SystemExit(f"est check: Supplementary S7 prints a number that no output key gives: {ln!r}")
    log(f"[est] {len(want)} sentences print what the amendment 6 outputs give; Supplementary S7 holds no other number")


def posthoc_tables(R, out):
    J, why = load_posthoc(R)
    if J is None:
        log(f"\n[post hoc] tables skipped: {why}")
        return []
    log("\n[post hoc] posthoc_v2.json certified by RESULTS.json (sha256 match) and its inputs are unchanged")
    return [ladder_rotation_success(J, out), imitation_base(J, out), ladder_organ(J, out), stats_sensitivity(J, out),
            rule_simulation(J, out),
            rl_tost(J, out), registration_crossed_tab(J, out), step_error(J, out), seg_sim_real(J, out)]


def _plan_digest_checks():
    """The digests this code pins (AMEND2_SHA256, PLAN_V21_SHA12, compared with result provenance) equal the ones the
    plan inputs give."""
    if PI.amendment_digest(2) != AMEND2_SHA256 or PI.plan_v21_digest()["sha12"] != PLAN_V21_SHA12:
        raise SystemExit("the plan digests differ from AMEND2_SHA256 or PLAN_V21_SHA12 of this code")


def _public_out(a):
    """Public mode: tables to DIR/tables, figures/plug_rows.json to DIR/figures, nothing written elsewhere; every
    --read-amend<n> lift implied (each still checks its records)."""
    global OUT_ALLOWED
    if not SD.public():
        raise SystemExit("--public: start this script from its command line (results_world.start reads --public and "
                         "--results before any data path is computed)")
    if not a.out:
        raise SystemExit("--public needs --out DIR")
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    OUT_ALLOWED = (d,)
    a.out, a.fig_out = os.path.join(d, "tables"), os.path.join(d, "figures")
    for n in (2, 3, 4, 5, 6):
        setattr(a, f"read_amend{n}", True)


def main(argv=None):
    global BLIND_A2, BLIND_A3, BLIND_A4, BLIND_A5, BLIND_A6, REFS_TEX, FIG_OUT, PROCESS_OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="the tables folder (default: the paper's tables folder of the development "
                    "repository); with --public the output root DIR (DIR/tables, DIR/figures)")
    ap.add_argument("--plug-root", default=PLUG_ROOT, help="v2.1 pluggability outputs of the gym (tab:pluggability)")
    ap.add_argument("--seg-plug-root", default=SEG_PLUG_ROOT, help="v2.1 pluggability outputs of the segmenter repo")
    ap.add_argument("--only-pluggability", action="store_true",
                    help="write only the Table 5 files (pluggability.tex, its twin, patient_seed.tex and "
                         "leaderboard.tex)")
    ap.add_argument("--only-checks", action="store_true", help="write only checks.tex")
    ap.add_argument("--fig-out", default=FIG_OUT, help="directory of figures/plug_rows.json (Fig. 4 data)")
    ap.add_argument("--process-out", default=PROCESS_OUT,
                    help="directory of the tables that the article does not input (in the development "
                         "repository)")
    ap.add_argument("--read-amend2", action="store_true",
                    help="read the outputs of the amendment 2 groups (default: blinded, none of them is read)")
    for n in (3, 4, 5, 6):
        ap.add_argument(f"--read-amend{n}", action="store_true",
                        help=f"read the outputs of amendment {n} (default: blinded, none of them is read)")
    ap.add_argument("--refs", default=REFS_TEX, help="refs.tex whose \\bibitem keys the NEW BIB KEYS line is checked "
                    "against (default: the manuscript's sections/refs.tex)")
    ap.add_argument("--public", action="store_true", help="public mode: the result files under --results ROOT and "
                    "the committed plan constants; see the module docstring")
    ap.add_argument("--results", help="public mode: the result root (outputs/, data/, segmenter/outputs/)")
    a = ap.parse_args(argv)
    if a.public:
        _public_out(a)
    elif a.out is None:
        a.out = os.path.join(PAPER, "tables")
    _plan_digest_checks()
    BLIND_A2 = not a.read_amend2
    for n in (3, 4, 5, 6):   # each lift is checked against its amendment's condition before the flag is rebound
        if getattr(a, f"read_amend{n}"):
            _lift_ok(n)
    BLIND_A3, BLIND_A4, BLIND_A5 = not a.read_amend3, not a.read_amend4, not a.read_amend5
    BLIND_A6 = not a.read_amend6
    REFS_TEX = a.refs
    FIG_OUT = a.fig_out
    PROCESS_OUT = a.process_out
    log(f"[blinding] amendment 2 groups {', '.join(A2_BLIND_KEYS)}: " + (A2_BLIND_TXT if BLIND_A2 else "read"))
    log("[blinding] " + "; ".join(f"amendment {f[-1]} ({', '.join(k)}): " + (A2_BLIND_TXT if globals()[f] else "read")
                                  for f, k in A345_BLIND_KEYS.items()))
    os.makedirs(a.out, exist_ok=True)
    if a.only_checks:
        log("\nwrote " + rel(checks(a.out)))
        return 0
    R = load_results()
    if a.only_pluggability:
        log("\nwrote " + ", ".join(rel(x) for x in pluggability(R, a.out, a.plug_root, a.seg_plug_root)
                                    + [patient_seed_tab(a.out), leaderboard_tab(a.out)]))
        return 0
    D = episode_design()
    written = [ladder(R, D, a.out), contact_dr(R, D, a.out), learnability(R, a.out), realism(R, a.out),
               identity(R, a.out), registration(R, a.out), dr(a.out), windows(a.out), compute_cost(R, a.out)]
    written += posthoc_tables(R, a.out)
    written += late_tables(R, a.out)          # post hoc and secondary results (Supplementary S2)
    written += pluggability(R, a.out, a.plug_root, a.seg_plug_root)
    _plug_rows_check()                          # the amendment 5 snapshot and the Table 5 SSIM word
    written.append(patient_seed_tab(a.out))     # App C.8, from the Table 5 record
    written.append(leaderboard_tab(a.out))      # Appendix C, the leaderboard, from the Table 5 record
    written.append(plug_cohort_tab(a.out))      # Supplementary Table S6.7
    written += amend345_tables(a.out)           # after pluggability(): the CT-anatomy context table reads its record
    written += amend6_tables(a.out)             # amendment 6: one line per item while blinded, its records table
    written += est_tables(a.out)                # amendment 6 floats of Supplementary S7, after the sentence guard
    written.append(checks(a.out))
    written.append(checks_summary(R, a.out))   # tab:checks-summary reads the Table 5 record, so it is written last
    log("\nwrote " + ", ".join(rel(p) for p in written if p))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PI.PlanInputsError as e:      # a missing or incomplete public_inputs/plan_constants.json: one line
        raise SystemExit(f"plan inputs: {e}")
