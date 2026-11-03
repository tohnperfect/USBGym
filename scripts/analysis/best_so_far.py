#!/usr/bin/env python3
"""Best measured complete pipeline so far, per run-time input class (POST HOC).

POST HOC: no pre-specified analysis of v2 or v2.1 defines this selection. It was defined after every input below
existed, it decides nothing, and every number it adds is labeled post hoc. Archived v2 results stay primary and v2.1
results exploratory; the selection uses the six held-out patients that also measured the errors, so the selected
error is optimistic. No episode is run, nothing renders, no network or git runs and no input is modified.

Question. Among the complete closed-loop pipelines measured so far (frames, mask source, controller, and for the RL
rows the training reward and renderer), which one ends closest to the target, for each kind of run-time input?
  a1  no privileged input at run time, frames that do not read the CT label slice (physics renders and the label-free
      translators on them), learned mask;
  a2  no privileged input at run time, frames rendered from the CT label slice (SonoSPADE, the label-painted stage),
      learned mask: these frames carry label information into the image;
  b   a privileged mask at run time (the CT mask, or SAM-family boxes from the CT label), no simulator look-ahead;
  c   privileged simulator look-ahead (the CEM look-ahead): a reference, never selected.
Anchors: the static start, the scripted sweep and the analytic pose oracle (archived v2 ladder). Every class shares
one episode design (contact plane through the face of the target window, starts drawn near the target pose), so no
row is free of target information.

Per candidate: the mean final translation error (patient means of the seed averages, as in Tables 4 and 5); its
paired difference from the static start and from the pose oracle per patient (two-sided paired t with 5 degrees of
freedom, its 95% interval, the number of patients of 6 sharing the sign, and the bootstrap interval over patients of
NBOOT resamples as a sensitivity check); and the fraction of the gap between static start and pose oracle that it
closes (a ratio of patient means, with a bootstrap interval over patients). The best so far is the lowest error among
the eligible candidates of a class. Beside it, the same selection made leave one patient out (choose on 5 patients,
score the sixth) gives a selection-adjusted estimate. A configuration that also ran on the 20-CT replication cohort
gets its cohort value as an unselected check (only the archived Table 4 configurations ran there).

Version 3: the Table 5 data file is the one make_tables.py --read-amend2 wrote after the v2.1 queue ended
(md5 PLUG_ROWS_MD5 below). Five Table 5 rows that were still pending in the earlier snapshot (TD3 residual, BC full
twist, DAgger full twist, histogram matching, the optional 20-epoch nnU-Net) enter from their own result files
(LATE_T5), checked against that file's err_mm; the TD3 residual and the full-twist BC and DAgger learners of the
physics stage under CT masks enter with the earlier controller runs of amendment 3. The amendment 2 groups are not
candidates: only make_tables.py reads their files, and their Table 5 errors are listed from plug_rows.json beside each
class's best (not_entered.amendment2_rows). Version 3 first ran on an earlier copy of the Table 5 data (md5
2180ea05c223f36baaad695c632a643d, output best_so_far_v3.json); a later edit of that file changed two row labels and the
MDD of the mask slot (every err_mm unchanged), so the pin moved to the final file and the output is best_so_far_v4.json,
whose values equal those of v3 (only that label and the provenance differ).

What it reads (read-only; every path passes plan_stats.check_input_path, which refuses imnet, samus, lpips, ctanat
and the private real-frame folders, and this script's own guard, which also refuses outputs/v2_1/amend3/ and the
controller collector and audit outputs):
  outputs/v2/RESULTS.json, outputs/v2/table4/episodes_long.csv, outputs/v2/table4/ladder/static_physics_ct.json:
      v2 ladder files of the archive (primary).
  figures/plug_rows.json (md5 PLUG_ROWS_MD5) and figures/forest.json of the paper tree: the Table 5 data written by
      make_tables.py after the queue ended; outputs/v2_1/analysis_plan/posthoc_logs_v2.json (m8_horizon.keepbest_plug):
      the files and blocks behind each Table 5 row it lists as complete; those row files themselves (completed,
      non-blinded); for the five rows it lists as pending, the files and blocks of LATE_T5 (make_tables._plug_paths).
  outputs/v2_1/label_controls/closed_loop/ibvs_base_label_painted_lit_matched_s*.json and
      outputs/v2_1/analysis_plan/q10_label_controls_v2.json: the label-painted row.
  outputs/v2_1/plug/ctrl/<learner>/<stage>_matched_s<seed>.json (the residual learners PPO, SAC, BC and
      DAgger, and DQN; physics and SonoSPADE) and outputs/v2_1/plug/ctrl/<learner>/physics_ct_s<seed>.json (the ten
      completed learners and the CEM look-ahead): the earlier controller runs of amendment 3, labeled LABEL_14. They are
      never used to state or deny that two methods separate.
  outputs/v2_1/analysis_plan/q2r_cohort_v2.json and outputs/v2_1/cohort/table4/episodes_long.csv: the 20-CT cohort.
  amendment 3 and its digest file (digest recorded only; development repository).

What it writes: ONE JSON, outputs/v2_1/analysis_plan/best_so_far.json, through plan_stats.write_result (refuses to
overwrite). The "table" subcommand reads that JSON and writes tables/best_so_far.tex of the paper tree.

Usage (development repository, from the repository root, one process, CPU only, seconds):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python <this folder>/best_so_far.py run \
      [--dry-run] [--out outputs/v2_1/analysis_plan/best_so_far.json]
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python <this folder>/best_so_far.py table \
      [--json outputs/v2_1/analysis_plan/best_so_far.json] [--tex <tables folder>/best_so_far.tex]
Public mode (the result files of the public deposit under ROOT; see script_dirs.py):
  python <this folder>/best_so_far.py table --public --results ROOT --out DIR     # DIR/tables/best_so_far.tex
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
if __name__ == "__main__":
    import results_world  # noqa: E402
    results_world.start(sys.argv[1:])     # --public --results ROOT, before plan_stats computes its data paths
import plan_inputs as PI  # noqa: E402
import plan_stats as ps  # noqa: E402
import script_dirs as SD  # noqa: E402

REPO = ps.REPO_ROOT
DATA = ps.DATA_ROOT          # the result files: the code tree, or the result root in public mode
V2 = os.path.join(DATA, "outputs", "v2")
V21 = os.path.join(DATA, "outputs", "v2_1")
PLUG = os.path.join(V21, "plug")
AP = os.path.join(V21, "analysis_plan")
PAPER = ps._PP.paper_tree(REPO) if ps._PP else None    # the paper tree (development repository only)
FIG = os.path.join(PAPER, "figures") if PAPER else None
DEFAULT_OUT = os.path.join(AP, "best_so_far.json")
DEFAULT_TEX = os.path.join(PAPER, "tables", "best_so_far.tex") if PAPER else None
PLUG_ROWS_REL = PI.text("plug_rows path")    # how the records name the Table 5 data file
# Render only: the table renders the existing Version 4 result; the run subcommand writes nothing (outputs/ is shared
# with the v2.1 checkout), and the arm A rows of amendment 3 are not candidates (they print in Table tab:arm-a; a new
# selection would be one more post hoc look). Amendment 5 fixes the label of this table and its [posthoc] sentence,
# which the caption's src comment cites (read from amend5_combination.json).
RENDER_JSON = os.path.join(AP, "best_so_far_v4.json")
RENDER_JSON_MD5 = "54b77a0f9691dc2ccfd24765d4a47a49"
RENDER_BEST = {"a1": "s14:bc_res/physics_matched", "a2": "s14:bc_res/sonospade_matched", "b": "t5:ctrl/dagger_res"}
AMEND5_JSON = os.path.join(AP, "amend5_combination.json")
# the random-action anchor of the caption, from the earlier learned-mask runs
# (amend3_earlier.json result.rows[6], the row Supplementary Table S3.2 prints), read-only; the module copy serves the
# synthetic tests
AMEND3_EARLIER_JSON = os.path.join(AP, "amend3_earlier.json")
RANDOM_ANCHOR_COPY = {"mean_mm": 17.45190828782803, "n_seeds": 3, "src": "best_so_far.py RANDOM_ANCHOR_COPY (module copy)"}
# the status strings and texts that the records hold (the development records and the results archive word some of
# them differently; plan_inputs.text gives the form of the records at hand)
AMEND5_STATUS = PI.text("status a5")
POSTHOC_LABEL = PI.text("a5 label text")
POSTHOC_SENTENCE = PI.text("posthoc sentence")
STATUS = "post hoc, v2"
STATUS_NOTE = PI.text("bsf status note")
LABEL_14 = PI.text("earlier label")
LABEL_T5 = PI.text("t5 label")
LABEL_V2 = "archived v2, primary"
LABEL_V21 = "exploratory, v2.1"
PLUG_STATUS = "exploratory, v2.1 pluggability"

TEST_PATIENTS = ("s0358", "s0513", "s0519", "s0591", "s0649", "s0762")
SEEDS = (0, 1, 2)
STAGES = ("physics", "cut", "cyclegan", "sonospade")
STAGE_NAME = {"physics": "physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
# v3: the Table 5 data written by make_tables.py --read-amend2 after the queue ended (the 2026-10-20 snapshot,
# md5 de5d539ec925b38d1328f0664439bd55 of 07:31:57 UTC, was the input of versions 1 and 2)
PLUG_ROWS_MD5 = "600f73ff4cb238cbfc6d137fc9dce357"
PLUG_ROWS_TIME = "2026-10-24 16:29:24 UTC"
AMEND3 = ps._PP.AMEND3_STEM if ps._PP else None     # the dated amendment 3 (development repository; run path)
TOL = 1e-9
NBOOT, BOOT_SEED = 10000, 0

# Paths this script never opens, beyond plan_stats.FORBIDDEN_INPUT_PATTERNS: the output root of amendment 3, and the
# controller collector and audit outputs (not needed here). v3: the TD3 residual and the full-twist BC and DAgger
# learners have finished and enter through LATE_T5 and S14_PHYS_CT.
LOCAL_FORBIDDEN = ("/outputs/v2_1/amend3/", "/outputs/v2_1/plug/ctrl/summary", "/outputs/v2_1/checks/audit/ctrl")
# v3: the Table 5 rows that posthoc_logs_v2.json keepbest_plug lists as pending (it was written before they finished):
# their result files and block, as make_tables._plug_paths and _episode_block read them (the GoalPolicy nets in
# BatchNorm eval mode), checked against plug_rows.json err_mm.
LATE_T5 = {
    ("ctrl", "td3_res"): ("outputs/v2_1/plug/ctrl/td3_res/sonospade_ct_s{s}.json", SEEDS,
                          "results[contact=on].episodes"),
    ("ctrl", "bc_scratch"): ("outputs/v2_1/plug/ctrl/bc_scratch/sonospade_ct_s{s}.json", SEEDS,
                             "results[contact=on,bn_mode=eval].episodes"),
    ("ctrl", "dagger_scratch"): ("outputs/v2_1/plug/ctrl/dagger_scratch/sonospade_ct_s{s}.json", SEEDS,
                                 "results[contact=on,bn_mode=eval].episodes"),
    ("texture", "histmatch"): ("outputs/v2_1/plug/texture/base/ibvs_base_histmatch_matched_s{s}.json", SEEDS,
                               "results[contact=on,weights=zero_residual].episodes"),
    ("mask", "nnunet2d_e20"): ("outputs/v2_1/plug/seg/ladder/ibvs_base_sonospade_nnunet2d_e20_s{s}.json", (0,),
                               "results[contact=on,weights=zero_residual].episodes"),
}

CLASSES = {
    "a1": {"title": "No privileged input at run time; frames that do not read the CT label slice; learned mask",
           "short": "learned mask, label-free frames"},
    "a2": {"title": "No privileged input at run time; frames rendered from the CT label slice; learned mask",
           "short": "learned mask, label-rendered frames"},
    "b": {"title": "Privileged mask at run time (CT mask or CT-label boxes); no simulator look-ahead",
          "short": "privileged mask"},
    "c": {"title": "Privileged simulator look-ahead at run time (a reference, never selected)",
          "short": "privileged look-ahead"},
}
SELECT_CLASSES = ("a1", "a2", "b")
REFERENCE_OF_CLASS = {"a1": "v2:matched/physics/base", "a2": "v2:matched/sonospade/base", "b": "v2:ctmask/base"}

INPUTS = set()


# ===================================================================================== guarded reading
class Refused(ps.PlanStatsError):
    pass


def guard(path):
    ps.check_input_path(path)
    p = SD.guard_forms(path)[0] if SD.public() else os.path.abspath(str(path))   # public mode: below the result root
    p = ps._slashes(p)                    # '/' separators, so that the patterns match Windows path forms too
    hits = [pat for pat in LOCAL_FORBIDDEN if pat in p]
    if hits:
        raise Refused(f"refusing to read {path}: matches this script's forbidden pattern(s) {hits}")
    return path


def plug_file(path):
    """True for a result file under outputs/v2_1/plug/ (matched with '/' separators, Windows path forms included)."""
    return "/outputs/v2_1/plug/" in ps._slashes(os.path.abspath(str(path)))


def read_json(path, record=True):
    guard(path)
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if record:
        INPUTS.add(os.path.abspath(path))
    return d


def read_csv(path):
    guard(path)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    INPUTS.add(os.path.abspath(path))
    return rows


def md5(path):
    guard(path)
    return ps.md5(path)


def amend3_info():
    """sha256 record of amendment 3, verified against its own .sha256 file (plan_stats.amendment_info covers 1 and 2)."""
    p, sp = AMEND3 + ".md", AMEND3 + ".sha256"
    guard(p)
    guard(sp)
    with open(sp, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    rec = lines[0].split()[0]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    got = h.hexdigest()
    INPUTS.update({os.path.abspath(p), os.path.abspath(sp)})
    if got != rec:
        raise ps.PlanHashMismatch(f"sha256 of {p} is {got}, but {sp} records {rec}")
    return {"path": ps.rel(p), "sha256": got, "sha12": got[:12], "sha256_file": ps.rel(sp),
            "hashed_utc": lines[1].strip() if len(lines) > 1 else None, "verified": True,
            "used_for": PI.text("bsf a3 used for")}


# ===================================================================================== per-patient cells
def per_patient_csv(rows, **flt):
    """collect_results_v2.per_patient: per patient, the mean final_pos over every matching row (all seeds and stage
    files pooled; the design is balanced, 10 episodes per patient per seed, so this equals the seed average)."""
    acc = collections.defaultdict(list)
    seeds = set()
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in flt.items()):
            acc[r["patient"]].append(float(r["final_pos"]))
            seeds.add(int(r["seed"]))
    return {p: float(np.mean(v)) for p, v in acc.items()}, sorted(seeds), {p: len(v) for p, v in acc.items()}


def fingerprint(eps):
    return [(e.get("ep"), e["patient"], e["target"], e["window"], round(float(e["start_pos"]), 6),
             round(float(e["start_ang"]), 6)) for e in eps]


_FP = {}


def static_fp():
    if "fp" not in _FP:
        d = read_json(os.path.join(V2, "table4", "ladder", "static_physics_ct.json"))
        bs = [r for r in d["results"] if r.get("contact") == "on"]
        if len(bs) != 1:
            raise ps.PlanStatsError("static_physics_ct.json: expected one contact-on block")
        _FP["fp"] = fingerprint(bs[0]["episodes"])
    return _FP["fp"]


def parse_block(spec):
    """'results[contact=on,weights=zero_residual].episodes' -> ('results', {...}); 'final.episodes' -> ('final', {})."""
    m = re.fullmatch(r"results\[([^\]]*)\]\.episodes", spec)
    if m:
        kv = dict(x.split("=", 1) for x in m.group(1).split(",") if x)
        return "results", kv
    m = re.fullmatch(r"(final|after_imitation)\.episodes", spec)
    if m:
        return m.group(1), {}
    raise ps.PlanStatsError(f"unknown block spec {spec!r}")


def select_block(d, path, spec, fp=None):
    """The one contact-on, DR-off, crossed held-out block named by spec, checked against the archived static rung's
    episode setups (the identical-episode check of make_tables._episode_block)."""
    where, kv = parse_block(spec)
    if where != "results":
        b = d[where]
    else:
        want_bn = kv.get("bn_mode", "train")
        bs = [r for r in d.get("results", [])
              if str(r.get("contact")) == kv.get("contact", "on") and r.get("eval_randomize", "off") == "off"
              and (r.get("bn_mode") or "train") == want_bn
              and ("weights" not in kv or r.get("weights") == kv["weights"])
              and r.get("control", "none") == kv.get("control", "none")]
        if len(bs) != 1:
            raise ps.PlanStatsError(f"{ps.rel(path)}: {len(bs)} blocks match {spec}")
        b = bs[0]
    eps = b.get("episodes") or []
    if (b.get("contact", "on") != "on" or b.get("eval_randomize", "off") != "off"
            or str(b.get("design", "crossed")).split()[0] != "crossed" or b.get("eval_seed", 999) != 999
            or len(eps) != 60 or {e["patient"] for e in eps} != set(TEST_PATIENTS)):
        raise ps.PlanStatsError(f"{ps.rel(path)}: not the crossed held-out design (6 patients, seed 999, 60 episodes)")
    if fingerprint(eps) != (static_fp() if fp is None else fp):
        raise ps.PlanStatsError(f"{ps.rel(path)}: episode setups differ from the archived static rung")
    if b.get("pos") is not None and abs(float(np.mean([float(e["final_pos"]) for e in eps])) - float(b["pos"])) > 1e-6:
        raise ps.PlanStatsError(f"{ps.rel(path)}: block 'pos' differs from the mean of its episodes")
    return b


def per_patient_block(b):
    acc = collections.defaultdict(list)
    for e in b["episodes"]:
        acc[e["patient"]].append(float(e["final_pos"]))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def check_plug_file(d, path):
    st = d.get("analysis_status") or (d.get("provenance") or {}).get("analysis_status")
    if st != PLUG_STATUS:
        raise ps.PlanStatsError(f"{ps.rel(path)}: analysis_status {st!r}, expected {PLUG_STATUS!r}")
    prov = d.get("provenance") or {}
    if any(isinstance(x, dict) and x.get("smoke") for x in (d, d.get("plug"), prov.get("plug"), prov.get("v21_plug"))) \
            or d.get("smoke") is True:
        raise ps.PlanStatsError(f"{ps.rel(path)}: a smoke output")


def cell_from_files(files, spec, plug=False, condition=None):
    """seed -> per-patient mean final translation error, from one result file per seed."""
    out = {}
    for f in files:
        d = read_json(f)
        if plug:
            check_plug_file(d, f)
        if condition:
            cond = d.get("condition") or {}
            for k, v in condition.items():
                if cond.get(k) != v:
                    raise ps.PlanStatsError(f"{ps.rel(f)}: condition.{k} = {cond.get(k)!r}, expected {v!r}")
        m = re.search(r"_s(\d)(?:_|\.json)", os.path.basename(f))
        seed = int(m.group(1)) if m else int(d.get("seed", 0))
        if seed in out:
            raise ps.PlanStatsError(f"two files for seed {seed}: {files}")
        out[seed] = per_patient_block(select_block(d, f, spec))
    return out


def seed_average(cell):
    """{patient: mean over seeds of that patient's per-seed means}."""
    return {p: float(np.mean([cell[s][p] for s in sorted(cell)])) for p in TEST_PATIENTS}


def pp_vector(pp, patients=TEST_PATIENTS):
    missing = [p for p in patients if p not in pp]
    if missing:
        raise ps.IncompleteData(f"missing patients {missing}")
    return np.array([pp[p] for p in patients], dtype=float)


# ===================================================================================== statistics
def diff_summary(a, b, units):
    """Paired difference a - b over units: mean, 95% t CI (df n - 1), k/n sharing the sign of the mean, two-sided
    paired t p (reported only), 95% patient bootstrap CI (sensitivity), per-unit values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    t = ps.paired_t(d)
    sc = ps.sign_count(d)
    return {"mean": t["mean"], "ci95_t": t["ci_t"], "sd": t["sd"], "k": sc["k"], "n": sc["n"],
            "p_t_two_sided": t["p_two_sided"], "ci95_boot_patient": ps.patient_bootstrap_ci(d),
            "per_unit": {u: float(x) for u, x in zip(units, d)}}


def closed_share(cand, static, oracle):
    """Share of the static-to-oracle distance a candidate closes: (mean static - mean cand) / (mean static - mean
    oracle), patients resampled jointly (plan_stats.ratio_bootstrap_ci)."""
    num = np.asarray(static, float) - np.asarray(cand, float)
    den = np.asarray(static, float) - np.asarray(oracle, float)
    r = ps.ratio_bootstrap_ci(num, den)
    return {"estimate": r["estimate"], "ci95_boot_patient": r["ci"], "n_dropped": r.get("n_dropped")}


def lopo_select(cands, patients=TEST_PATIENTS):
    """Leave-one-patient-out selection: for each patient, pick the candidate with the lowest mean error over the other
    patients and score it on the left-out patient. -> {estimate, per_patient, picks}."""
    ids = [c["id"] for c in cands]
    M = np.array([pp_vector(c["pp"], patients) for c in cands])   # [cand, patient]
    out, picks = [], []
    for j, p in enumerate(patients):
        keep = [i for i in range(len(patients)) if i != j]
        i_best = int(np.argmin(M[:, keep].mean(1)))
        out.append(float(M[i_best, j]))
        picks.append(ids[i_best])
    return {"estimate": float(np.mean(out)), "per_patient": dict(zip(patients, out)),
            "picks": dict(zip(patients, picks)), "n_candidates": len(cands)}


# ===================================================================================== candidates
def cand(cid, cls, label, frames, mask, controller, source, status_label, seeds, cell=None, pp=None, *,
         label_access=False, privileged=(), eligible=True, note=None, xref=None, files=(), block=None, reward=None,
         renderer="physics renderer", cohort=None, verify=False):
    if pp is None:
        pp = seed_average(cell)
    c = {"id": cid, "class": cls, "label": label, "frames": frames, "mask": mask, "controller": controller,
         "reward": reward, "renderer": renderer, "label_access": label_access, "privileged_at_run_time": list(privileged),
         "source": source, "status_label": status_label, "seeds": list(seeds), "eligible": eligible, "note": note,
         "files": [ps.rel(f) for f in files], "block": block, "pp": pp, "cohort_selector": cohort,
         "provisional": verify}
    if cell is not None:
        c["per_seed_err_mm"] = {str(s): float(np.mean(list(cell[s].values()))) for s in sorted(cell)}
    c["err_mm"] = float(np.mean(pp_vector(pp)))
    if xref is not None:
        src, val = xref
        c["xref"] = {"source": src, "value": float(val), "abs_diff": abs(c["err_mm"] - float(val))}
        if c["xref"]["abs_diff"] > TOL:
            raise ps.PlanStatsError(f"{cid}: error {c['err_mm']!r} differs from {src} = {val!r}")
    return c


def archived_candidates(rows, R):
    """Table 4 cells (archived v2, primary) and the reference rungs, from episodes_long.csv as the collector reads it."""
    base = dict(contact="on", eval_randomize="off")
    cm = R["table4"]["cell_means_mm"]
    rung = {}
    for name in ("static", "scripted", "pose_oracle"):
        pp, seeds, n = per_patient_csv(rows, group="ladder", rung=name, **base)
        val = R["table4"]["rungs_mm"][name]
        if abs(float(np.mean(pp_vector(pp))) - val) > TOL:
            raise ps.PlanStatsError(f"rung {name}: recomputed mean differs from RESULTS.json")
        rung[name] = {"pp": pp, "seeds": seeds, "err_mm": float(np.mean(pp_vector(pp))),
                      "xref": {"source": f"outputs/v2/RESULTS.json table4.rungs_mm.{name}", "value": val},
                      "episodes_per_patient": n}
    out = []
    nice = {"base": "servo base", "after_imitation": "servo base + imitation residual (GoalPolicy)",
            "final": "servo base + REINFORCE residual (final weights)"}
    for st in STAGES:
        for arm, mask in (("matched", "learned"), ("ctmask", "ct")):
            for ph in ("base", "after_imitation", "final"):
                if arm == "ctmask" and ph == "base" and st != "physics":
                    continue       # the CT-mask servo base reads only the CT mask: one pipeline for all four stages
                if ph == "base":
                    pp, seeds, _ = per_patient_csv(rows, group="ladder", rung="ibvs_base", stage=st, mask=mask, **base)
                else:
                    pp, seeds, _ = per_patient_csv(rows, group=arm, arm=arm, stage=st, phase=ph, **base)
                key = f"{arm}/{st}/{ph}"
                if arm == "ctmask" and ph == "base":
                    cid, frames = "v2:ctmask/base", "any stage (the servo base reads only the mask)"
                    cls = "b"
                else:
                    cid, frames = f"v2:{key}", STAGE_NAME[st]
                    cls = "b" if arm == "ctmask" else ("a2" if st == "sonospade" else "a1")
                coh = None
                if ph == "base":
                    coh = {"group": "ladder", "rung": "ibvs_base", "mask": mask, "stage": st}
                else:
                    coh = {"group": "contact", "rung": "policy", "arm": arm, "stage": st, "phase": ph}
                out.append(cand(cid, cls, f"{frames} frames, {'stage-matched U-Net' if mask == 'learned' else 'CT'} "
                                f"mask, {nice[ph]}", frames, "stage-matched U-Net" if mask == "learned" else "CT mask",
                                nice[ph], "outputs/v2/table4/episodes_long.csv (Table 4 cell)", LABEL_V2, seeds,
                                pp=pp, label_access=(st == "sonospade" and cid != "v2:ctmask/base"),
                                privileged=(["CT mask"] if arm == "ctmask" else []),
                                xref=(f"outputs/v2/RESULTS.json table4.cell_means_mm[{key}]", cm[key]),
                                reward=("paper score S" if ph == "final" else None), cohort=coh))
    return rung, out


def table5_candidates(plug_rows, kbp):
    """Table 5 rows (the snapshot), re-read from the files posthoc_logs_v2 lists for each row."""
    rows = {(s["key"], r["id"]): (s, r) for s in plug_rows["slots"] for r in s["rows"]}
    out, skipped = [], []
    # duplicates of archived Table 4 cells: the archived cell is the candidate
    dup = {("texture", "sonospade"), ("texture", "physics"), ("texture", "cut"), ("texture", "cyclegan"),
           ("mask", "unet"), ("mask", "ct"), ("ctrl", "reinforce"), ("ctrl", "servo_base"), ("renderer", "physics")}
    spec = {
        # texture slot: learned-mask servo base, each stage with its own matched U-Net
        ("texture", "fastcut"): ("a1", "FastCUT", "stage-matched U-Net", "servo base"),
        ("texture", "dclgan"): ("a1", "DCLGAN", "stage-matched U-Net", "servo base"),
        ("texture", "unsb"): ("a1", "UNSB", "stage-matched U-Net", "servo base"),
        ("texture", "histmatch"): ("a1", "Histogram matching", "stage-matched U-Net", "servo base"),   # v3, LATE_T5
        # mask slot: servo base per mask, SonoSPADE frames
        ("mask", "lraspp_mbv3"): ("a2", "SonoSPADE", "LR-ASPP, MobileNetV3-L", "servo base"),
        ("mask", "deeplabv3_mbv3"): ("a2", "SonoSPADE", "DeepLabV3, MobileNetV3-L", "servo base"),
        ("mask", "deeplabv3_r50"): ("a2", "SonoSPADE", "DeepLabV3, ResNet-50", "servo base"),
        ("mask", "fcn_r50"): ("a2", "SonoSPADE", "FCN, ResNet-50", "servo base"),
        ("mask", "nnunet2d"): ("a2", "SonoSPADE", "nnU-Net 2D, 2 epochs", "servo base"),
        ("mask", "sam_unet"): ("a2", "SonoSPADE", "SAM, boxes from the U-Net mask", "servo base"),
        ("mask", "medsam_unet"): ("a2", "SonoSPADE", "MedSAM, boxes from the U-Net mask", "servo base"),
        ("mask", "nnunet2d_e20"): ("a2", "SonoSPADE", "nnU-Net 2D, 20 epochs", "servo base"),   # v3, LATE_T5
        ("mask", "sam_priv"): ("b", "SonoSPADE", "SAM, boxes from the CT label", "servo base"),
        ("mask", "medsam_priv"): ("b", "SonoSPADE", "MedSAM, boxes from the CT label", "servo base"),
        # controller slot: SonoSPADE frames, CT mask
        ("ctrl", "reinforce_bneval"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual, BatchNorm eval"),
        ("ctrl", "ppo_res"): ("b", "SonoSPADE", "CT mask", "servo base + PPO residual"),
        ("ctrl", "sac_res"): ("b", "SonoSPADE", "CT mask", "servo base + SAC residual"),
        ("ctrl", "a2c_res"): ("b", "SonoSPADE", "CT mask", "servo base + A2C residual"),
        ("ctrl", "td3_res"): ("b", "SonoSPADE", "CT mask", "servo base + TD3 residual"),   # v3, LATE_T5
        ("ctrl", "bc_scratch"): ("b", "SonoSPADE", "CT mask", "BC, full twist (BatchNorm eval)"),   # v3, LATE_T5
        ("ctrl", "dagger_scratch"): ("b", "SonoSPADE", "CT mask", "DAgger, full twist (BatchNorm eval)"),   # v3
        ("ctrl", "ppo_scratch"): ("b", "SonoSPADE", "CT mask", "PPO, full twist"),
        ("ctrl", "dqn_scratch"): ("b", "SonoSPADE", "CT mask", "DQN, 13 discrete moves"),
        ("ctrl", "bc_res"): ("b", "SonoSPADE", "CT mask", "servo base + BC residual (BatchNorm eval)"),
        ("ctrl", "dagger_res"): ("b", "SonoSPADE", "CT mask", "servo base + DAgger residual (BatchNorm eval)"),
        ("ctrl", "cem"): ("c", "SonoSPADE", "CT mask", "CEM look-ahead (simulator rollouts)"),
        # reward slot: REINFORCE residual, SonoSPADE frames, CT mask, trained with each reward
        ("reward", "paper"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "seg_only"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "ncc_only"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "ncc_sector"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "ssim"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "nmi"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "pose"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        ("reward", "pose_area"): ("b", "SonoSPADE", "CT mask", "servo base + REINFORCE residual"),
        # renderer slot: the physics-trained CT-mask policy under each renderer
        ("renderer", "lotus_orig"): ("b", "physics-stage policy frames", "CT mask", "servo base + REINFORCE residual"),
        ("renderer", "lotus_reimpl"): ("b", "physics-stage policy frames", "CT mask",
                                       "servo base + REINFORCE residual"),
        ("renderer", "simus"): ("b", "physics-stage policy frames", "CT mask", "servo base + REINFORCE residual"),
    }
    renderer_of = {"lotus_orig": "LOTUS forward model, original code", "lotus_reimpl": "LOTUS, authors' reimplementation",
                   "simus": "SIMUS via PyMUST"}
    for (slot, rid), (cls, frames, mask, ctrl) in spec.items():
        s, r = rows[(slot, rid)]
        kp = kbp.get(f"{slot}.{rid}")
        late = (slot, rid) in LATE_T5 and (not kp or kp.get("status") != "complete")
        if late:   # v3: pending when posthoc_logs_v2 was written; files and block from LATE_T5
            pat, sds, blk = LATE_T5[(slot, rid)]
            kp = {"files": [pat.format(s=x) for x in sds], "block": blk, "final_mean_mm": r["err_mm"]}
        elif not kp or kp.get("status") != "complete":
            skipped.append({"id": f"t5:{slot}/{rid}", "reason": "not complete in posthoc_logs_v2 keepbest_plug"})
            continue
        files = [os.path.join(DATA, f) for f in kp["files"]]
        plug = any(plug_file(f) for f in files)
        cell = cell_from_files(files, kp["block"], plug=plug)
        if sorted(cell) != sorted(r["seeds"] or []):
            raise ps.PlanStatsError(f"{slot}/{rid}: seeds {sorted(cell)} differ from plug_rows {r['seeds']}")
        priv = []
        if r["privileged"]:
            priv = ["CT-label boxes"] if "boxes" in mask else (["simulator look-ahead", "CT mask"] if cls == "c"
                                                              else ["CT mask"])
        elif cls == "b":
            priv = ["CT mask"]
        reward = {"paper": "paper score S", "seg_only": "organ overlap only", "ncc_only": "appearance NCC only",
                  "ncc_sector": "NCC over the whole sector", "ssim": "SSIM appearance term",
                  "nmi": "NMI appearance term", "pose": "change in pose distance (privileged, training only)",
                  "pose_area": "pose distance plus organ area (privileged, training only)"}.get(rid) \
            if slot == "reward" else None
        lab = r["label"].replace("$S$", "S").replace("\\\"{o}", "o")
        out.append(cand(f"t5:{slot}/{rid}", cls, f"Table 5 {slot} slot: {lab}", frames, mask, ctrl,
                        "Table 5 data (" + PLUG_ROWS_REL + "), row files via "
                        + ("this script's LATE_T5 (pending in posthoc_logs_v2.json)" if late else
                           "posthoc_logs_v2.json"),
                        LABEL_T5 if plug else LABEL_V21, sorted(cell), cell=cell,
                        label_access=bool(r["label_access"]) or frames == "SonoSPADE",
                        privileged=priv, eligible=(cls in SELECT_CLASSES),
                        note=("seed 0 only (inference-only or single-seed design)" if len(cell) == 1 else None),
                        xref=(f"{PLUG_ROWS_REL} slots[{slot}].rows[{rid}].err_mm", r["err_mm"]),
                        files=files, block=kp["block"], reward=reward,
                        renderer=renderer_of.get(rid, "physics renderer") if slot == "renderer" else "physics renderer",
                        verify=bool(r.get("provisional", True))))
        if abs(out[-1]["err_mm"] - float(kp["final_mean_mm"])) > TOL:
            raise ps.PlanStatsError(f"{slot}/{rid}: differs from posthoc_logs_v2 keepbest_plug final_mean_mm")
    return out, skipped, dup


def painted_candidate(plug_rows, q10):
    files = [os.path.join(V21, "label_controls", "closed_loop", f"ibvs_base_label_painted_lit_matched_s{s}.json")
             for s in SEEDS]
    cell = cell_from_files(files, "results[contact=on,weights=zero_residual,control=none].episodes")
    r = next(r for s in plug_rows["slots"] for r in s["rows"] if s["key"] == "texture" and r["id"] == "label_painted")
    c = cand("t5:texture/label_painted", "a2", "Table 5 texture slot: label-painted physics stage",
             "label-painted physics", "stage-matched U-Net", "servo base",
             "outputs/v2_1/label_controls/closed_loop (Table 5 texture row; Q10)", LABEL_V21, sorted(cell),
             cell=cell, label_access=True, files=files,
             block="results[contact=on,weights=zero_residual,control=none].episodes",
             xref=(PLUG_ROWS_REL + " slots[texture].rows[label_painted].err_mm", r["err_mm"]),
             verify=True)
    p = q10["results"]["painting"]["patient_means"]["P"]
    if abs(c["err_mm"] - float(p)) > TOL:
        raise ps.PlanStatsError("label-painted row differs from q10_label_controls_v2.json patient_means.P")
    c["xref2"] = {"source": "outputs/v2_1/analysis_plan/q10_label_controls_v2.json results.painting.patient_means.P",
                  "value": float(p)}
    return c


S14_LEARNED = ("ppo_res", "sac_res", "bc_res", "dagger_res", "dqn_scratch")
# v3: the TD3 residual and the full-twist BC and DAgger learners on physics frames with CT masks (finished before the
# candidate set was fixed)
S14_PHYS_CT = ("a2c_res", "bc_res", "dagger_res", "dqn_scratch", "ppo_res", "ppo_scratch", "sac_res", "td3_res",
               "bc_scratch", "dagger_scratch")
S14_BN_EVAL = ("bc_res", "dagger_res", "bc_scratch", "dagger_scratch")
S14_NAME = {"ppo_res": "servo base + PPO residual", "sac_res": "servo base + SAC residual",
            "a2c_res": "servo base + A2C residual", "bc_res": "servo base + BC residual (BatchNorm eval)",
            "dagger_res": "servo base + DAgger residual (BatchNorm eval)", "dqn_scratch": "DQN, 13 discrete moves",
            "ppo_scratch": "PPO, full twist", "td3_res": "servo base + TD3 residual",
            "bc_scratch": "BC, full twist (BatchNorm eval)", "dagger_scratch": "DAgger, full twist (BatchNorm eval)",
            "cem": "CEM look-ahead (simulator rollouts)"}


def s14_candidates():
    """Candidate rows from the controller runs of amendment 3: learners trained under learned masks on physics and
    SonoSPADE frames, and, for the physics stage, learners and look-ahead under CT masks. Primary BatchNorm block for BC
    and DAgger: eval (amendment 1's primary mode)."""
    out = []
    for learner in S14_LEARNED:
        for st in ("physics", "sonospade"):
            files = [os.path.join(PLUG, "ctrl", learner, f"{st}_matched_s{s}.json") for s in SEEDS]
            blk = "results[contact=on,bn_mode=eval].episodes" if learner in ("bc_res", "dagger_res") \
                else "results[contact=on].episodes"
            cell = cell_from_files(files, blk, plug=True, condition={"texture": st, "mask": "learned"})
            out.append(cand(f"s14:{learner}/{st}_matched", "a2" if st == "sonospade" else "a1",
                            f"{STAGE_NAME[st]} frames, stage-matched U-Net mask, {S14_NAME[learner]}",
                            STAGE_NAME[st], "stage-matched U-Net", S14_NAME[learner],
                            PI.text("bsf s14 src"), LABEL_14, sorted(cell), cell=cell,
                            label_access=(st == "sonospade"), files=files, block=blk, verify=True))
    for learner in S14_PHYS_CT + ("cem",):
        seeds = (0,) if learner == "cem" else SEEDS
        files = [os.path.join(PLUG, "ctrl", learner, f"physics_ct_s{s}.json") for s in seeds]
        blk = "results[contact=on,bn_mode=eval].episodes" if learner in S14_BN_EVAL \
            else "results[contact=on].episodes"
        cell = cell_from_files(files, blk, plug=True, condition={"texture": "physics", "mask": "ct"})
        cls = "c" if learner == "cem" else "b"
        out.append(cand(f"s14:{learner}/physics_ct", cls, f"physics frames, CT mask, {S14_NAME[learner]}",
                        "physics", "CT mask", S14_NAME[learner], PI.text("bsf s14 src"),
                        LABEL_14, sorted(cell), cell=cell,
                        privileged=(["simulator look-ahead", "CT mask"] if cls == "c" else ["CT mask"]),
                        eligible=(cls in SELECT_CLASSES), files=files, block=blk,
                        note=("seed 0 only" if len(cell) == 1 else None), verify=True))
    return out


# ===================================================================================== cohort
def cohort_values(rows_c, q2r):
    """Cohort per-patient values of the anchors and the archived configurations (q2r cell_matrix rule)."""
    patients = sorted({r["patient"] for r in rows_c})
    if len(patients) != q2r["n_c"]:
        raise ps.PlanStatsError(f"cohort: {len(patients)} patients, q2r_cohort_v2 says {q2r['n_c']}")

    def cell(**flt):
        groups = {}
        for r in rows_c:
            if r.get("contact") == "on" and r.get("eval_randomize", "off") == "off" and \
                    all(r.get(k) == v for k, v in flt.items()):
                groups.setdefault((r["stage"], r["seed"]), {}).setdefault(r["patient"], []).append(float(r["final_pos"]))
        if not groups:
            return None
        stages = sorted({g[0] for g in groups})
        seeds = sorted({g[1] for g in groups}, key=int)
        for st in stages:
            for s in seeds:
                if (st, s) not in groups or set(groups[(st, s)]) != set(patients):
                    raise ps.PlanStatsError(f"cohort cell {flt} incomplete at stage {st} seed {s}")
        return {p: float(np.mean([np.mean([np.mean(groups[(st, s)][p]) for st in stages]) for s in seeds]))
                for p in patients}, [int(s) for s in seeds]

    anchors = {}
    oc = q2r["ordering"]["cells"]
    names = {"static": "static", "scripted": "scripted", "pose_oracle": "pose oracle"}
    for key, nm in names.items():
        pp, seeds = cell(group="ladder", rung=key, stage="physics")
        ref = next(v for v in oc.values() if v["name"] == nm)["cohort"]
        if abs(float(np.mean([pp[p] for p in patients])) - ref) > TOL:
            raise ps.PlanStatsError(f"cohort {key}: differs from q2r_cohort_v2 ordering cell")
        anchors[key] = {"pp": pp, "seeds": seeds, "mean": float(np.mean([pp[p] for p in patients])),
                        "xref": f"outputs/v2_1/analysis_plan/q2r_cohort_v2.json ordering.cells[{nm}].cohort"}
    return patients, anchors, cell


def attach_cohort(c, patients, anchors, cellf, q2r):
    sel = c.get("cohort_selector")
    if not sel:
        c["cohort"] = {"status": "not run on the cohort (only the archived Table 4 configurations ran there)"}
        return
    got = cellf(**sel)
    if got is None:
        c["cohort"] = {"status": "not run on the cohort"}
        return
    pp, seeds = got
    v = pp_vector(pp, patients)
    st, orc = pp_vector(anchors["static"]["pp"], patients), pp_vector(anchors["pose_oracle"]["pp"], patients)
    c["cohort"] = {"status": "complete", "n": len(patients), "seeds": seeds, "mean_mm": float(v.mean()),
                   "minus_static": diff_summary(v, st, patients), "minus_oracle": diff_summary(v, orc, patients),
                   "per_patient": {p: float(x) for p, x in zip(patients, v)},
                   "selector": sel, "source": "outputs/v2_1/cohort/table4/episodes_long.csv"}
    # cross-check with the q2r ordering cells where the same cell is listed there
    names = {("ladder", "ibvs_base", "ct", "physics"): "CT-mask base",
             ("ladder", "ibvs_base", "learned", "physics"): "learned base, physics",
             ("ladder", "ibvs_base", "learned", "cut"): "learned base, CUT",
             ("ladder", "ibvs_base", "learned", "cyclegan"): "learned base, CycleGAN",
             ("ladder", "ibvs_base", "learned", "sonospade"): "learned base, per-tissue",
             ("contact", "matched", "physics", "final"): "learned final, physics",
             ("contact", "matched", "cut", "final"): "learned final, CUT",
             ("contact", "matched", "cyclegan", "final"): "learned final, CycleGAN",
             ("contact", "matched", "sonospade", "final"): "learned final, per-tissue"}
    key = (sel["group"], sel.get("rung"), sel.get("mask"), sel.get("stage")) if sel["group"] == "ladder" else \
        (sel["group"], sel.get("arm"), sel.get("stage"), sel.get("phase"))
    nm = names.get(key)
    if nm:
        ref = next(v for v in q2r["ordering"]["cells"].values() if v["name"] == nm)["cohort"]
        if abs(c["cohort"]["mean_mm"] - ref) > TOL:
            raise ps.PlanStatsError(f"{c['id']}: cohort mean differs from q2r_cohort_v2 ordering cell {nm}")
        c["cohort"]["xref"] = f"outputs/v2_1/analysis_plan/q2r_cohort_v2.json ordering.cells[{nm}].cohort"


def cohort_selection_check(classes, cands, patients):
    """POST HOC check of the selection rule on data it did not see: within each class, among the configurations that
    also ran on the 20-CT cohort (the archived Table 4 cells), take the lowest error on the six held-out patients and
    score it on the cohort, beside the class's reference pipeline."""
    for k in SELECT_CLASSES:
        rec = classes[k]
        pool = sorted([c for c in cands if c["class"] == k and c["eligible"] and c["cohort"].get("status") == "complete"],
                      key=lambda c: c["err_mm"])
        pick = pool[0]
        ref = next(c for c in cands if c["id"] == rec["reference"])
        a, b = pp_vector(pick["cohort"]["per_patient"], patients), pp_vector(ref["cohort"]["per_patient"], patients)
        rec["cohort_check"] = {
            "label": "post hoc; the selection rule restricted to the configurations that ran on the cohort (archived "
                     "Table 4 cells), applied on the six held-out patients and scored on the 20 cohort CTs, which it "
                     "did not see",
            "n_configs": len(pool), "selected_on_six": pick["id"], "six_err_mm": pick["err_mm"],
            "six_minus_reference": diff_summary(pp_vector(pick["pp"]), pp_vector(ref["pp"]), TEST_PATIENTS),
            "cohort_mean_mm": pick["cohort"]["mean_mm"], "reference_cohort_mean_mm": ref["cohort"]["mean_mm"],
            "cohort_minus_reference": diff_summary(a, b, patients),
            "ranking_on_cohort": [{"id": c["id"], "six_err_mm": c["err_mm"], "cohort_mean_mm": c["cohort"]["mean_mm"]}
                                  for c in sorted(pool, key=lambda c: c["cohort"]["mean_mm"])]}


# ===================================================================================== analysis
def analyse(cands, rung):
    st = pp_vector(rung["static"]["pp"])
    orc = pp_vector(rung["pose_oracle"]["pp"])
    for c in cands:
        v = pp_vector(c["pp"])
        c["minus_static"] = diff_summary(v, st, TEST_PATIENTS)
        c["minus_oracle"] = diff_summary(v, orc, TEST_PATIENTS)
        c["share_closed"] = closed_share(v, st, orc)
    classes = {}
    for k in SELECT_CLASSES + ("c",):
        members = sorted([c for c in cands if c["class"] == k], key=lambda c: c["err_mm"])
        elig = [c for c in members if c["eligible"]]
        rec = {"title": CLASSES[k]["title"], "n_candidates": len(members), "n_eligible": len(elig),
               "ranking": [{"id": c["id"], "err_mm": c["err_mm"], "seeds": c["seeds"], "status_label": c["status_label"]}
                           for c in members]}
        if k in SELECT_CLASSES:
            best = elig[0]
            rec["best"] = best["id"]
            rec["runner_up"] = elig[1]["id"] if len(elig) > 1 else None
            rec["best_minus_runner_up_mm"] = (best["err_mm"] - elig[1]["err_mm"]) if len(elig) > 1 else None
            rec["reference"] = REFERENCE_OF_CLASS[k]
            ref = next(c for c in cands if c["id"] == REFERENCE_OF_CLASS[k])
            rec["best_minus_reference"] = diff_summary(pp_vector(best["pp"]), pp_vector(ref["pp"]), TEST_PATIENTS)
            rec["best_minus_reference"]["label"] = PI.text("bsf s14 label")
            pre = [c for c in elig if c["status_label"] != LABEL_14]
            rec[PI.key("best_without_earlier_rows")] = {
                "id": pre[0]["id"], "err_mm": pre[0]["err_mm"], "status_label": pre[0]["status_label"],
                "minus_static": pre[0]["minus_static"], "minus_oracle": pre[0]["minus_oracle"],
                "note": "the same rule over the candidates printed under an archived or pre-specified rule only"}
            rec["lopo_selection"] = lopo_select(elig)
            rec["lopo_selection"]["minus_static"] = diff_summary(
                pp_vector(rec["lopo_selection"]["per_patient"]), st, TEST_PATIENTS)
            rec["lopo_selection"]["minus_oracle"] = diff_summary(
                pp_vector(rec["lopo_selection"]["per_patient"]), orc, TEST_PATIENTS)
            rec["selection_label"] = ("post hoc; lowest mean final translation error among the class's eligible "
                                      f"candidates ({len(elig)}), selected on the same six held-out patients that "
                                      "measured it, so optimistic")
        else:
            rec["note"] = "reference only; the look-ahead reads the simulator at run time and is never selected"
        classes[k] = rec
    return classes


A2_CLASS = {("mask", "samus_unet"): "a2", ("mask", "samus_priv"): "b", ("mask", "lraspp_mbv3_imnet"): "a2",
            ("mask", "deeplabv3_mbv3_imnet"): "a2", ("mask", "deeplabv3_r50_imnet"): "a2",
            ("mask", "fcn_r50_imnet"): "a2", ("reward", "lpips"): "b"}


def a2_rows(plug_rows, classes):
    """v3, for comparison only (not candidates): the amendment 2 rows' Table 5 errors (plug_rows.json, written by
    make_tables.py --read-amend2) beside the best error of the class each would join."""
    best = {k: next(r["err_mm"] for r in classes[k]["ranking"] if r["id"] == classes[k]["best"])
            for k in SELECT_CLASSES}
    out = []
    for s in plug_rows["slots"]:
        for r in s["rows"]:
            k = A2_CLASS.get((s["key"], r["id"]))
            if k is None or r.get("err_mm") is None:
                continue
            out.append({"id": f"t5:{s['key']}/{r['id']}", "class": k, "err_mm": float(r["err_mm"]),
                        "seeds": r.get("seeds"), "class_best": classes[k]["best"], "class_best_err_mm": best[k],
                        "below_class_best": float(r["err_mm"]) < best[k],
                        "source": f"{PLUG_ROWS_REL} slots[{s['key']}].rows[{r['id']}].err_mm"})
    if len(out) != len(A2_CLASS):
        raise ps.PlanStatsError(f"amendment 2 rows: {len(out)} of {len(A2_CLASS)} found in plug_rows.json")
    return out


def build(args):
    INPUTS.clear()
    C = ps.load_constants()
    asserted = ps.assert_constants({"alpha": 0.05, "bootstrap": {"resamples": NBOOT, "seed": BOOT_SEED}}, C)
    R = read_json(os.path.join(V2, "RESULTS.json"))
    rows = read_csv(os.path.join(V2, "table4", "episodes_long.csv"))
    pr_path = os.path.join(FIG, "plug_rows.json")
    if md5(pr_path) != PLUG_ROWS_MD5:
        raise ps.PlanStatsError(f"plug_rows.json md5 {md5(pr_path)} is not the Table 5 snapshot {PLUG_ROWS_MD5}")
    plug_rows = read_json(pr_path)
    forest = read_json(os.path.join(FIG, "forest.json"))
    phl = read_json(os.path.join(AP, "posthoc_logs_v2.json"))
    q10 = read_json(os.path.join(AP, "q10_label_controls_v2.json"))
    q2r = read_json(os.path.join(AP, "q2r_cohort_v2.json"))
    rows_c = read_csv(os.path.join(V21, "cohort", "table4", "episodes_long.csv"))
    a3 = amend3_info()

    rung, cands = archived_candidates(rows, R)
    # anchors cross-checked with the per-patient values q2r_cohort_v2 printed for the original six
    for key, nm in (("static", "static"), ("scripted", "scripted"), ("pose_oracle", "pose oracle")):
        ref = next(v for v in q2r["ordering"]["cells"].values() if v["name"] == nm)["original_six_per_patient"]
        if any(abs(rung[key]["pp"][p] - ref[p]) > TOL for p in TEST_PATIENTS):
            raise ps.PlanStatsError(f"anchor {key}: per-patient values differ from q2r_cohort_v2")
    t5, skipped, dup = table5_candidates(plug_rows, phl["m8_horizon"]["keepbest_plug"])
    cands += t5
    cands.append(painted_candidate(plug_rows, q10))
    cands += s14_candidates()
    # forest.json carries the same Table 5 error for the family rows it draws
    for fr in forest["rows"]:
        c = next((c for c in cands if c["id"] == f"t5:{fr['slot']}/{fr['id']}"), None)
        if c is not None and abs(c["err_mm"] - float(fr["err_mm"])) > TOL:
            raise ps.PlanStatsError(f"{c['id']}: differs from forest.json")
    ids = [c["id"] for c in cands]
    if len(ids) != len(set(ids)):
        raise ps.PlanStatsError("duplicate candidate ids")

    classes = analyse(cands, rung)
    patients_c, anchors_c, cellf = cohort_values(rows_c, q2r)
    for c in cands:
        attach_cohort(c, patients_c, anchors_c, cellf, q2r)
    cohort_selection_check(classes, cands, patients_c)
    st_c, or_c = pp_vector(anchors_c["static"]["pp"], patients_c), pp_vector(anchors_c["pose_oracle"]["pp"], patients_c)
    for k in ("static", "scripted", "pose_oracle"):
        v = pp_vector(anchors_c[k]["pp"], patients_c)
        anchors_c[k]["minus_static"] = diff_summary(v, st_c, patients_c) if k != "static" else None
        anchors_c[k]["minus_oracle"] = diff_summary(v, or_c, patients_c) if k != "pose_oracle" else None
    st6, or6 = pp_vector(rung["static"]["pp"]), pp_vector(rung["pose_oracle"]["pp"])
    anchors = {}
    for k in ("static", "scripted", "pose_oracle"):
        v = pp_vector(rung[k]["pp"])
        coh = {kk: vv for kk, vv in anchors_c[k].items() if kk != "pp"}
        coh["per_patient"] = anchors_c[k]["pp"]
        anchors[k] = {"err_mm": rung[k]["err_mm"], "seeds": rung[k]["seeds"], "per_patient_mm": rung[k]["pp"],
                      "episodes_per_patient": rung[k]["episodes_per_patient"], "xref": rung[k]["xref"],
                      "status_label": LABEL_V2,
                      "minus_static": diff_summary(v, st6, TEST_PATIENTS) if k != "static" else None,
                      "minus_oracle": diff_summary(v, or6, TEST_PATIENTS) if k != "pose_oracle" else None,
                      "cohort": coh}

    for c in cands:
        c["per_patient_mm"] = c.pop("pp")
    payload = {
        "question": "best measured complete pipeline so far, per run-time input class (post hoc)",
        "label": PI.text("bsf run label"),
        "status_note": STATUS_NOTE,
        "classes_defined": CLASSES,
        "anchors": anchors,
        "classes": classes,
        "candidates": cands,
        "not_entered": {
            "duplicates_of_archived_cells": [f"t5:{s}/{r}" for s, r in sorted(dup)],
            "pending_in_table5_snapshot_not_read": [],
            "entered_in_v3": ["t5:" + "/".join(k) for k in LATE_T5]
                             + [f"s14:{x}/physics_ct" for x in ("td3_res", "bc_scratch", "dagger_scratch")],
            "secondary_batchnorm_mode": "BC and DAgger residuals and full twists in BatchNorm train mode (secondary "
                                        "in Table 5 and amendment 1); the primary eval block is the candidate",
            "amendment2_groups": ["samus", "imnet", "lpips", "ctanat"],
            "amendment2_note": "amendment 2 groups are not candidates: their files are read only by make_tables.py "
                               "(--read-amend2); their Table 5 errors are listed from plug_rows.json beside the best "
                               "of the class they would join, for comparison only",
            "amendment2_rows": a2_rows(plug_rows, classes),
            "amendment3_arms": "arm A (learned-mask DAgger and BC residuals, learner seeds 3 to 5), arms B and C: "
                               "run after the queue; outputs/v2_1/amend3/ is never read here (pending)",
            "re_evaluations": "28 jobs re-evaluate archived learned-mask artifacts (the rungs random actions, static start "
                              "and servo base, the imitation and final policies of v2 and the pose oracle), and so does "
                              "outputs/v2_1/s21_bneval; the candidate is the archived cell",
            "other_horizons_and_metrics": "36-step arms and keep-best errors (posthoc_logs_v2 m8_horizon) use another "
                                          "horizon or metric; not candidates",
            "skipped": skipped,
        },
        "not_combined": "Every slot varied one component against fixed defaults; no candidate combines the winners "
                        "of two slots (for example the best mask source with the best controller). That combination "
                        "was not run.",
        "pending": "amendment 3 arm A (learned-mask DAgger and BC residuals with new learner seeds) runs after the "
                   "queue and will update classes a1 and a2",
        "amendment3": a3,
        "table5_snapshot": {"path": PLUG_ROWS_REL, "md5": PLUG_ROWS_MD5,
                            "written": PLUG_ROWS_TIME,
                            "note": "v3: written by make_tables.py --read-amend2 after the queue ended; versions 1 and "
                                    "2 read the 2026-10-20 07:31:57 UTC snapshot (md5 "
                                    "de5d539ec925b38d1328f0664439bd55)"},
        "cohort": {"n": len(patients_c), "patients": patients_c,
                   "note": "only the archived Table 4 configurations ran on the cohort; unselected check"},
        "design": {"patients": list(TEST_PATIENTS), "unit": "patient (n = 6); seeds averaged within patient",
                   "error": "final translation error, mean over patients of seed-averaged patient means",
                   "ci": PI.text("bsf design ci"),
                   "share_closed": "(static - candidate) / (static - pose oracle), ratio of patient means, patient "
                                   "bootstrap CI",
                   "lopo": "leave-one-patient-out selection over the class's eligible candidates (select on 5, score "
                           "the sixth), post hoc"},
    }
    return payload, asserted


# ===================================================================================== LaTeX table
def f2(x, signed=False):
    return ps.fmt_num(x, 2, signed)


def tex_diff(dd, vf):
    s = f"${f2(dd['mean'], True)}$ [${f2(dd['ci95_t'][0], True)}$, ${f2(dd['ci95_t'][1], True)}$], {dd['k']}/{dd['n']}"
    return vf(s)


def _a5_texts(path=AMEND5_JSON):
    """amendment 5's label of this table and its [posthoc] sentence (amend5_combination.json, read-only), asserted
    against this script's copies and, in the development repository, against the amendment's Appendix value."""
    d = read_json(path, record=False)
    if (d.get("provenance") or {}).get("analysis_status") != AMEND5_STATUS:
        raise ps.PlanStatsError(f"{ps.rel(path)}: unexpected analysis_status")
    lab = d["posthoc_table_label"]
    sen = [x["text"] for x in d["sentences"] if x["id"] == "posthoc"]
    if lab != POSTHOC_LABEL or sen != [POSTHOC_SENTENCE] or (
            not ps._public_plan() and PI.amendment_value(5, "posthoc_table_label") != lab):
        raise ps.PlanStatsError("amendment 5's post hoc label or [posthoc] sentence differs from this script's copy")
    return {"label": lab, "posthoc": sen[0], "src": f"{ps.rel(path)} posthoc_table_label and sentences[id=posthoc] "
                                                     f"(index {[x['id'] for x in d['sentences']].index('posthoc')})"}


def _random_anchor(path=AMEND3_EARLIER_JSON):
    """The random-action controller of the earlier learned-mask runs (amend3_earlier.json result.rows[6], physics
    frames under learned masks; the same value prints in every condition of Supplementary Table S3.2), checked."""
    d = read_json(path, record=False)
    r = d["result"]["rows"][6]
    if r.get("row") != "random" or r.get("kind") != "reference" or int(r.get("n_seeds", 0)) != 3:
        raise ps.PlanStatsError(f"{ps.rel(path)}: result.rows[6] is not the three-seed random-action reference")
    return {"mean_mm": float(r["mean_mm"]), "n_seeds": int(r["n_seeds"]),
            "src": f"{ps.rel(path)} result.rows[6] (row random, n_seeds 3, mean_mm {r['mean_mm']:.4f})"}


def _best_so_far_check(text):
    """'best so far' prints only after 'post hoc' in the same sentence (amendment 5)."""
    vis = "\n".join(re.split(r"(?<!\\)%", ln, maxsplit=1)[0] for ln in text.splitlines())
    for sent in re.split(r"(?<=[.;])\s+", vis):
        low = sent.lower()
        i = low.find("best so far")
        if i >= 0 and "post hoc" not in low[:i]:
            raise ps.PlanStatsError(f"'best so far' without 'post hoc' before it: {sent[:160]!r}")


# The printed caption carries no amendment wording and no provisional mark; amendment 5's label and [posthoc] sentence
# are read and asserted (_a5_texts) and cited in the src comment; these are their printed forms.
SCI_LABEL_PRINT = ("Post hoc: lowest point estimate on the same six held-out patients that measured it, so optimistic; "
                   "not the pre-specified combination rule.")
SCI_POSTHOC_PRINT = ("The pre-specified combination selects by the slot decisions and tests its choice on 20 CTs that "
                     "played no part in it, so the two can name different pipelines.")
SCI_NOTE = PI.text("best_so_far note")     # a source comment of the development table


def render_table(J, json_path, a5=None, rand=None):
    """tables/best_so_far.tex from the result JSON (booktabs, scriptsize, column width, [tp]). a5: the
    amendment 5 texts (_a5_texts); rand: the random-action anchor (_random_anchor); the module copies are used when
    they are None (synthetic tests)."""
    a5 = a5 or {"label": POSTHOC_LABEL, "posthoc": POSTHOC_SENTENCE, "src": "best_so_far.py POSTHOC_* (module copy)"}
    rand = rand or RANDOM_ANCHOR_COPY
    cands = {c["id"]: c for c in J["candidates"]}
    A = J["anchors"]
    K = J["classes"]
    jrel = ps.rel(json_path)
    vfy = lambda s: f"\\verifyfill{{{s}}}"   # noqa: E731
    plain = lambda s: s                       # noqa: E731
    L, D, E, S = "$^{\\mathrm{L}}$", "$^{\\dagger}$", "$^{\\mathrm{E}}$", "$^{\\S}$"
    short = {"v2:matched/physics/base": "Physics; U-Net; servo base",
             "v2:matched/sonospade/base": f"SonoSPADE{L}; U-Net; servo base",
             "v2:ctmask/base": f"Any stage; CT mask{D}; servo base"}

    def describe(c):
        if c["id"] in short:
            return short[c["id"]]
        fr = c["frames"].replace("physics-stage policy frames", "physics")
        fr = {"physics": "Physics", "label-painted physics": f"Label-painted{L}", "SonoSPADE": f"SonoSPADE{L}"}.get(fr, fr)
        mk = c["mask"].replace("stage-matched U-Net", "U-Net").replace("CT mask", f"CT mask{D}")
        if "boxes from the CT label" in mk:
            mk = mk + D
        ct = c["controller"].replace(" (BatchNorm eval)", "")
        ct = ct.replace("CEM look-ahead (simulator rollouts)", f"CEM look-ahead{D}")
        mark = E if c["status_label"] == LABEL_14 else ""
        prov = ""   # the section-sign mark does not print
        seeds = "" if len(c["seeds"]) == 3 else ", seed " + ", ".join(str(x) for x in c["seeds"])
        return f"{fr}; {mk}; {ct}{seeds}{mark}{prov}"

    def row(name, err, ms, mo, coh, vf, src):
        return f"{name} & {vf(err)} & {ms} & {mo} & {coh} \\\\ % src: {src}"

    def crow(tag, c):
        vf = vfy if c["provisional"] else plain
        coh = f2(c["cohort"]["mean_mm"]) if c["cohort"].get("status") == "complete" else "n/r"
        src = (f"{jrel} candidates[{c['id']}] (source {c['source']}; status {c['status_label']}; "
               + (f"xref {c['xref']['source']} = {c['xref']['value']:.4f}; " if c.get("xref") else "")
               + f"seeds {c['seeds']}; per-seed errors "
               + ("n/a (archived cell)" if not c.get("per_seed_err_mm") else
                  ", ".join(f"s{k} {v:.2f}" for k, v in c["per_seed_err_mm"].items()))
               + "; cohort " + (c["cohort"].get("xref") or c["cohort"].get("source") or c["cohort"].get("status"))
               + ")")
        name = f"\\quad {tag}: {describe(c)}" if tag else f"\\quad {describe(c)}"
        return row(name, f2(c["err_mm"]), tex_diff(c["minus_static"], vf), tex_diff(c["minus_oracle"], vf), coh, vf,
                   src)

    nE = sum(1 for c in J["candidates"] if c["status_label"] == LABEL_14)
    n = {k: K[k]["n_eligible"] for k in SELECT_CLASSES}
    bc = [k for k in ("a1", "a2") if "BC residual" in cands[K[k]["best"]]["controller"]]
    cap = ("\\caption{Lowest measured error per kind of run-time input, post hoc. "
           + SCI_LABEL_PRINT + " " + SCI_POSTHOC_PRINT + " "
           "Pipelines name the frames, the mask source and the controller. "
           "Error: final translation error over the 60 held-out episodes of Table~\\ref{tab:ladder}, mean over seeds "
           "of patient means. "
           "Differences are paired by patient, with the 95\\% $t$ CI and the patients of 6 sharing the sign. "
           # the random-action anchor as a caption sentence (the candidate counts of the blocks stay)
           f"A random-action controller, from earlier runs with no pre-specified rule, ends at {f2(rand['mean_mm'])}~mm "
           f"({rand['n_seeds']} seeds; Supplementary Table~S3.2). "
           f"Each post hoc lowest row is the lowest error among the {n['a1']}, {n['a2']} and {n['b']} measured "
           "candidates of the second, third and fourth blocks. "
           "No candidate combines the winners of two slots; the pre-specified combination is reported in "
           "Section~\\ref{sec:plug-best}. "
           "The learned-mask controller arm with new learner seeds is not a candidate; its rows are in "
           "Table~\\ref{tab:arm-a}. "
           + ("Both lowest rows without privileged input are BC residuals (exploratory, no pre-specified rule). "
              if bc == ["a1", "a2"] else "")
           + "None of these rows enters a decision, and none is read as a separation of two methods. "
           "Cohort: the same configuration on the 20-CT replication cohort, an unselected check; n/r, not run there. "
           "Every episode keeps the contact plane through the target window face and a start drawn around the "
           "target. "
           "No row is therefore free of target information. "
           f"{L}Label-conditioned stage: its network also reads the CT label slice. "
           f"{D}Privileged: simulator state that no rig provides. "
           f"{E}Exploratory, no pre-specified rule.}}")
    src_cap = (f" % src: {jrel} classes.<a1,a2,b>.best and .n_eligible ({n['a1']}, {n['a2']}, {n['b']}); design; "
               f"not_combined; label E on {nE} candidates" + PI.text("bsf src a") + f"{a5['src']}"
               + PI.text("bsf src b") + "controller ("
               + ", ".join(cands[K[k]['best']]['controller'] for k in ('a1', 'a2')) + ")" + PI.text("bsf src c")
               + f"{rand['src']}" + PI.text("bsf src d"))
    lines = ["%% NEW BIB KEYS: none",
             f"% generated by {PI.text('best_so_far script')} table from {jrel} (POST HOC" + PI.text("bsf gen note")
             + ")",
             "% do not edit by hand: rerun the script's table subcommand",
             SCI_NOTE,
             "\\begin{table}[tp]",
             "\\renewcommand{\\baselinestretch}{1}\\scriptsize\\setlength{\\fboxsep}{0.6pt}",
             cap + src_cap, "\\label{tab:best-so-far}", "\\centering", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}X r l l r@{}}", "\\toprule",
             "Pipeline & \\begin{tabular}[b]{@{}r@{}}Error\\\\(mm)\\end{tabular} & "
             "\\begin{tabular}[b]{@{}l@{}}Minus static start (mm),\\\\95\\% $t$ CI, $k$/6\\end{tabular} & "
             "\\begin{tabular}[b]{@{}l@{}}Minus pose oracle (mm),\\\\95\\% $t$ CI, $k$/6\\end{tabular} & "
             "\\begin{tabular}[b]{@{}r@{}}20-CT\\\\cohort (mm)\\end{tabular} \\\\",
             "\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Anchors (reference ladder, Table~\\ref{tab:ladder})}} \\\\"]
    s, sc, o = A["static"], A["scripted"], A["pose_oracle"]
    lines.append(row("\\quad Static start (no commanded motion)", f2(s["err_mm"]), "n/a",
                     tex_diff(s["minus_oracle"], plain), f2(s["cohort"]["mean"]), plain,
                     f"{jrel} anchors.static (err_mm = outputs/v2/RESULTS.json table4.rungs_mm.static; cohort "
                     "outputs/v2_1/analysis_plan/q2r_cohort_v2.json ordering.cells[static].cohort)"))
    lines.append(row("\\quad Scripted sweep (no feedback)", f2(sc["err_mm"]), tex_diff(sc["minus_static"], plain),
                     tex_diff(sc["minus_oracle"], plain), f2(sc["cohort"]["mean"]), plain,
                     f"{jrel} anchors.scripted (RESULTS.json table4.rungs_mm.scripted; cohort q2r_cohort_v2.json "
                     "ordering.cells[scripted].cohort)"))
    lines.append(row(f"\\quad Analytic pose oracle{D}", f2(o["err_mm"]), tex_diff(o["minus_static"], plain), "n/a",
                     f2(o["cohort"]["mean"]), plain,
                     f"{jrel} anchors.pose_oracle (RESULTS.json table4.rungs_mm.pose_oracle; cohort q2r_cohort_v2.json "
                     "ordering.cells[pose oracle].cohort)"))
    # the frame-class terms of the block headers
    heads = {"a1": "No privileged input at run time: frames without a label-conditioned stage, learned mask",
             "a2": f"No privileged input at run time: label-conditioned frames{L}, learned mask",
             "b": f"Privileged mask at run time{D}"}
    for k in SELECT_CLASSES:
        lines += ["\\addlinespace", f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{heads[k]}}}}} \\\\"]
        lines.append(crow("Reference", cands[K[k]["reference"]]))
        if K[k]["best"] != K[k]["reference"]:
            lines.append(crow("Post hoc lowest", cands[K[k]["best"]]))
    # the look-ahead reference: the lowest class-c row already printed elsewhere (not a row of the earlier runs)
    look = [r["id"] for r in K["c"]["ranking"] if r["status_label"] != LABEL_14]
    if look:
        lines += ["\\addlinespace",
                  f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{Privileged simulator look-ahead{D}, a reference never "
                  "selected}} \\\\"]
        lines.append(crow("", cands[look[0]]))
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    text = "\n".join(lines) + "\n"
    ps.check_text(text)
    _best_so_far_check(text)
    words = ("confirmed", "significant", "comparable", "deployable", "superset", "differentiable")
    banned = re.compile(r"\b(" + "|".join(words) + r")\b|observability gap|perception gap|\bopen\b", re.IGNORECASE)
    hits = sorted({m.group(0) for m in banned.finditer(text)})
    if hits:
        raise ps.PlanStatsError(f"banned words in the table: {hits}")
    return text


# ===================================================================================== CLI
def cmd_run(args):
    if not args.dry_run:
        raise ps.PlanStatsError("best_so_far.py is render-only: the run subcommand writes "
                                "no JSON; render best_so_far_v4.json with the table subcommand")
    payload, asserted = build(args)
    # console summary
    for k in SELECT_CLASSES:
        rec = payload["classes"][k]
        c = next(c for c in payload["candidates"] if c["id"] == rec["best"])
        print(f"[{k}] best {c['id']}: {c['err_mm']:.2f} mm; minus static {c['minus_static']['mean']:+.2f} "
              f"[{c['minus_static']['ci95_t'][0]:+.2f}, {c['minus_static']['ci95_t'][1]:+.2f}] "
              f"{c['minus_static']['k']}/6; minus oracle {c['minus_oracle']['mean']:+.2f}; "
              f"share closed {c['share_closed']['estimate']:.2f}; LOPO {rec['lopo_selection']['estimate']:.2f} mm "
              f"(of {rec['n_eligible']})")
    if args.dry_run:
        print("dry run: nothing written")
        return payload
    out = ps.write_result(args.out, payload, analysis_status=STATUS, inputs=sorted(INPUTS),
                          script=os.path.abspath(__file__), constants_asserted=asserted, status_note=STATUS_NOTE,
                          amendments=(1, 2), root=args.root)
    print("wrote", ps.rel(out))
    return payload


def cmd_table(args):
    pub = SD.public()
    if not pub and args.out:
        raise ps.PlanStatsError("--out DIR is an option of public mode: add --public --results ROOT")
    if pub:     # the pin names the development file's bytes; the public build checks it against ROOT/MANIFEST.tsv
        if not args.out:
            raise ps.PlanStatsError("--public needs --out DIR (the table is written to DIR/tables/best_so_far.tex)")
        args.tex = os.path.join(os.path.abspath(args.out), "tables", "best_so_far.tex")
        if os.path.abspath(args.json) == os.path.abspath(RENDER_JSON):
            import results_world
            PI.manifest_instead("the pinned md5 of the Version 4 result (RENDER_JSON_MD5)")
            if results_world.listed(args.json) is None:
                raise ps.PlanStatsError(f"{ps.rel(args.json)}: not listed in MANIFEST.tsv of the result root, so the "
                                        "pinned Version 4 result cannot be checked")
    elif os.path.abspath(args.json) == os.path.abspath(RENDER_JSON) and md5(args.json) != RENDER_JSON_MD5:
        raise ps.PlanStatsError(f"{ps.rel(args.json)}: md5 differs from the pinned Version 4 result")
    J = read_json(args.json, record=False)
    if os.path.abspath(args.json) == os.path.abspath(RENDER_JSON):
        got = {k: J["classes"][k]["best"] for k in RENDER_BEST}
        if got != RENDER_BEST:
            raise ps.PlanStatsError(f"classes {got} differ from Version 4's {RENDER_BEST}")
    text = render_table(J, args.json, _a5_texts(args.amend5), _random_anchor())
    if not pub and not os.path.realpath(args.tex).startswith(os.path.realpath(PAPER) + os.sep) and not args.allow_out:
        raise ps.PlanStatsError(f"{args.tex}: outside {ps.rel(PAPER)}/; this table is written nowhere else")
    if pub:     # the printed text only: every LaTeX comment removed (tex_comments.py of the paper build)
        sys.path.insert(0, SD.paper_dir())
        import tex_comments
        text = tex_comments.strip_comments(text)[0]
        os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    with open(args.tex, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", ps.rel(args.tex))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--out", default=DEFAULT_OUT)
    a.add_argument("--root", default=ps.RESULTS_ROOT, help=argparse.SUPPRESS)
    a.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("table")
    b.add_argument("--json", default=RENDER_JSON)
    b.add_argument("--tex", default=DEFAULT_TEX)
    b.add_argument("--amend5", default=AMEND5_JSON, help="amend5_combination.json (the post hoc label and sentence)")
    b.add_argument("--allow-out", action="store_true", help=argparse.SUPPRESS)
    b.add_argument("--public", action="store_true", help="public mode: result files under --results ROOT")
    b.add_argument("--results", help="public mode: the result root (outputs/, data/, segmenter/outputs/)")
    b.add_argument("--out", help="public mode: the output root DIR (DIR/tables/best_so_far.tex)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_table(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PI.PlanInputsError as e:      # a missing or incomplete public_inputs/plan_constants.json: one line
        raise SystemExit(f"plan inputs: {e}")
