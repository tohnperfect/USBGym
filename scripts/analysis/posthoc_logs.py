#!/usr/bin/env python3
"""Post hoc analyses of EXISTING USB Gym logs.

POST HOC: no pre-specified analysis of v2 or v2.1 defines them. They were defined after every input below existed,
they decide nothing, and every number they produce is labeled post hoc. Archived v2 results stay primary; v2.1 outputs
stay exploratory. No episode is run, nothing renders, no network runs, git is never run, and no input is modified.

What it reads (read-only; every path passes plan_stats.check_input_path, which refuses the blinded amendment-2
groups imnet, samus, lpips and ctanat and the private real-frame folders):
  outputs/v2_1/s21_rerun/{ladder,contact}/*.json and their .traj.npz sidecars: the v2.1 clean-tag rerun with the
      extended log. The v2.1 clean-tag rerun found every shared episode field bit-identical to the archived v2
      episodes (11,520 of 11,520), so the extended log describes the archived episodes.
  outputs/v2_1/s21_h36/*.json and .traj.npz: the v2.1 36-step arms (exploratory).
  outputs/v2/table4/episodes_long.csv and outputs/v2/table4/ladder/static_physics_ct.json: archived v2 episodes.
  outputs/v2/RESULTS.json: archived cell means and contrasts, used only to check the recomputed per-patient values.
  outputs/v2_1/analysis_plan/q8_gap_cause_v2.json: verified per-patient values of two printed v2.1 contrasts.
  The Table 5 row files (completed, non-blinded): outputs/v2_1/plug/{texture,seg,ctrl,reward,renderer}/...,
      outputs/v2_1/s21_bneval/ and the archived v2 files behind the reference rows.
  data/raw/totalseg_small/meta.csv: TotalSegmentator metadata (age, gender, study type) of the CT cases.
  outputs/v2_1/cohort/cohort_frozen.json: case ids of the 20-case replication cohort.
  Source text (constants only, parsed with regular expressions, never imported): src/usbg/mujoco_env.py,
      src/usbg/goal_rl.py, src/usbg/placement.py, src/usbg/_par/contracts.py, the sibling par/contracts.py and
      scripts/train_view_match_rl.py of the v2.1 checkout (make_env passes no settle_steps).

What it writes: ONE JSON, outputs/v2_1/analysis_plan/posthoc_logs.json, through plan_stats.write_result (refuses to
overwrite). The "tables" subcommand reads that JSON and writes tables/posthoc_*.tex of the paper tree.

Definitions (from src/usbg/controllers_v21.py, the extended-log evaluator):
  axes      signed SE(3)-log error of the ACTUAL probe pose in the TARGET probe frame: lateral (probe x, along the
            array), elevational (probe y, out of plane), axial (probe z, along the beam) in mm; out-of-plane tilt
            (rotation about x, fanning), in-plane tilt (about y, rocking) and roll (about z) in deg. Per-axis
            "error" here is |component|, averaged per patient (patient-level mean = mean of per-patient means).
  cmd_lag   distance between the commanded (mocap, press-limited) and the actual probe-face position after each
            step (mm); index 0 is the start (0 by construction) and is excluded.
  gap_min   height of the lowest of the face centre and four face corners above the contact plane z = 0 (mm);
            > 0 hovers, < 0 pressed in. Steps 1..N.
  twist     the executed 6-DoF body twist [vx, vy, vz, wx, wy, wz] (clipped to goal_rl.MAX_TWIST); the commanded
            fraction of the bound is twist / MAX_TWIST per channel. vz > 0 points into the tissue.
  keep-best the pose-free keep-best of the evaluator: the step with the highest match score S (argmax over steps
            0..N, no pose read), scored by its translation error (episode field keepbest_pos).
  start lift placement.clamp_start_outside_skin lifts a perturbed start along world +z, the outward normal of the
            target-anchored plane (the target beam is world -z), until the face centre and corners clear it.
            In the target frame the lift lowers the axial offset by the lift, so the pre-lift axial offset is the
            logged start offset plus clamp_mm.

Usage (development repository, from the repository root, one process, CPU only):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python <this folder>/posthoc_logs.py run \
      [--dry-run] [--out outputs/v2_1/analysis_plan/posthoc_logs.json]
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python <this folder>/posthoc_logs.py tables \
      [--json outputs/v2_1/analysis_plan/posthoc_logs.json] [--tables-dir <tables folder>]
Public mode (the result files of the public deposit under ROOT; see script_dirs.py):
  python <this folder>/posthoc_logs.py tables --public --results ROOT --out DIR     # DIR/tables/posthoc_*.tex
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
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
RERUN = os.path.join(V21, "s21_rerun")
H36 = os.path.join(V21, "s21_h36")
PLUG = os.path.join(V21, "plug")
# the sibling checkouts whose source text the run subcommand reads (development repository only)
V21_TREE = ps._PP.sibling(REPO, "v21 tree") if ps._PP else None
PAR_SIBLING = ps._PP.sibling(REPO, "par contracts") if ps._PP else None
PAPER = ps._PP.paper_tree(REPO) if ps._PP else None    # the paper tree (development repository only)
META_CSV = os.path.join(DATA, "data", "raw", "totalseg_small", "meta.csv")
COHORT_FROZEN = os.path.join(V21, "cohort", "cohort_frozen.json")
DEFAULT_OUT = os.path.join(ps.RESULTS_ROOT, "posthoc_logs.json")
DEFAULT_TABLES = os.path.join(PAPER, "tables") if PAPER else None
JSON_REL = "outputs/v2_1/analysis_plan/posthoc_logs.json"

STATUS = "post hoc, v2"
STATUS_NOTE = PI.text("phl status note")
LABEL = PI.text("phl label")

STAGES = ("physics", "cut", "cyclegan", "sonospade")
SEEDS = (0, 1, 2)
PATIENTS = ("s0358", "s0513", "s0519", "s0591", "s0649", "s0762")
TRAIN_PATIENTS = ("s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727",
                  "s0777", "s1086", "s1373")
AXES = ("lat_mm", "elev_mm", "ax_mm", "tilt_out_deg", "tilt_in_deg", "roll_deg")
TWIST_CH = ("vx", "vy", "vz", "wx", "wy", "wz")          # same order as AXES (probe body frame)
BLIND_KEYS = ("imnet", "samus", "lpips", "ctanat")
PLUG_STATUS = "exploratory, v2.1 pluggability"
TANGENT_MAX_DEG = 1.0      # window_tilt_deg below this: a beam along the skin normal (logged 0.00 to 0.03 deg)
USED_CONSTANTS = {"alpha": 0.05, "bootstrap": {"resamples": 10000, "seed": 0},
                  "n6_decision": {"test": "paired_t", "df": 5}}

INPUTS: list = []          # every file read, for the provenance block


# ============================================================================================ low-level reading
def _blinded(path):
    forms = SD.guard_forms(path) if SD.public() else (str(path),)   # public mode: the path below the result root
    return any(k in f.lower() for f in forms for k in BLIND_KEYS)


def _note(path):
    ps.check_input_path(path)
    if _blinded(path):
        raise ps.ForbiddenPath(f"refusing a blinded amendment-2 path: {path}")
    if path not in INPUTS:
        INPUTS.append(path)
    return path


def read_json(path):
    _note(path)
    return ps.read_json(path)


def read_text(path):
    _note(path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def read_npz(path):
    """-> list of (meta, {key: array}) blocks of a save_traj_npz sidecar."""
    _note(path)
    z = np.load(path, allow_pickle=False)
    metas = json.loads(str(z["index"]))
    out = []
    for m in metas:
        i = int(m["block"])
        pre = f"b{i}__"
        out.append((m, {k[len(pre):]: np.asarray(z[k]) for k in z.files if k.startswith(pre)}))
    return out


def traj_of(json_path):
    return json_path[:-5] + ".traj.npz"


def blocks_with_traj(json_path):
    """-> (doc, [(result block, traj dict)]) with the npz blocks matched to the JSON results by (weights, contact,
    eval_randomize) and checked episode for episode (ep order, final translation error)."""
    d = read_json(json_path)
    nb = read_npz(traj_of(json_path))
    out = []
    for r in d["results"]:
        key = (r.get("weights"), r.get("contact"), r.get("eval_randomize", "off"))
        hits = [tr for m, tr in nb if (m.get("weights"), m.get("contact"), m.get("eval_randomize", "off")) == key]
        if len(hits) != 1:
            raise ps.PlanStatsError(f"{ps.rel(json_path)}: {len(hits)} npz blocks for {key}")
        tr = hits[0]
        eps = r["episodes"]
        if [int(e["ep"]) for e in eps] != [int(x) for x in tr["ep"]]:
            raise ps.PlanStatsError(f"{ps.rel(json_path)}: npz episode order differs from the JSON for {key}")
        fin = np.asarray([float(e["final_pos"]) for e in eps])
        if np.max(np.abs(fin - tr["pos"][:, -1].astype(np.float64))) > 1e-3:
            raise ps.PlanStatsError(f"{ps.rel(json_path)}: npz final pos differs from the JSON for {key}")
        out.append((r, tr))
    return d, out


def pick(blocks, contact, weights=None):
    hits = [(r, tr) for r, tr in blocks if r.get("contact") == contact and r.get("eval_randomize", "off") == "off"
            and (weights is None or r.get("weights") == weights)]
    if len(hits) != 1:
        raise ps.PlanStatsError(f"{len(hits)} blocks for contact={contact} weights={weights}")
    return hits[0]


# ============================================================================================ pure helpers (tested)
def patient_means(values, patients, order=PATIENTS):
    """Per-patient mean of per-episode values (episodes of every seed pooled; equal counts per patient and seed make
    this the mean of per-seed patient means)."""
    acc = collections.defaultdict(list)
    for v, p in zip(np.asarray(values, np.float64).ravel(), list(patients)):
        acc[str(p)].append(float(v))
    missing = [p for p in order if p not in acc]
    if missing:
        raise ps.IncompleteData(f"no episodes for patients {missing}")
    return {p: float(np.mean(acc[p])) for p in order}


def level_mean(pm):
    """Patient-level mean: the mean of the per-patient means."""
    return float(np.mean([pm[p] for p in PATIENTS]))


def vec(pm, order=PATIENTS):
    return np.asarray([pm[p] for p in order], np.float64)


def lopo(per_patient):
    """Leave-one-patient-out check of a paired contrast given its per-patient differences {patient: d}.
    Returns the full mean and sign count, the mean with each patient left out, their range, whether every
    left-out mean keeps the sign of the full mean, and the patients whose difference opposes that sign."""
    keys = list(per_patient)
    d = np.asarray([float(per_patient[k]) for k in keys], np.float64)
    n = len(d)
    if n < 3:
        raise ps.IncompleteData("leave-one-out needs at least 3 units")
    m = float(d.mean())
    sgn = float(np.sign(m))
    k = int(np.sum(np.sign(d) == sgn)) if sgn != 0 else 0
    loo = {keys[i]: float((d.sum() - d[i]) / (n - 1)) for i in range(n)}
    vals = np.asarray(list(loo.values()))
    holds = bool(sgn != 0 and np.all(np.sign(vals) == sgn))
    dissent = [keys[i] for i in range(n) if np.sign(d[i]) != sgn]
    most = max(loo, key=lambda kk: abs(loo[kk] - m))
    return {"n": n, "mean": m, "sign_k": k, "lopo_means": loo, "lopo_min": float(vals.min()),
            "lopo_max": float(vals.max()), "sign_kept_in_all": holds, "dissenting_patients": dissent,
            "most_influential_patient": most, "shift_when_left_out": float(loo[most] - m)}


def step_stats(arr, skip_first=True):
    """mean, 95th percentile (numpy linear interpolation) and max over every step value (index 0 skipped)."""
    a = np.asarray(arr, np.float64)
    if skip_first:
        a = a[:, 1:]
    a = a.ravel()
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "n_values": int(a.size)}


def twist_fraction(twist, bound):
    """Per channel: signed mean of twist / bound, mean of |twist| / bound and the share of steps at the bound."""
    tw = np.asarray(twist, np.float64).reshape(-1, 6)
    b = np.asarray(bound, np.float64).reshape(6)
    f = tw / b
    sat = np.abs(np.abs(tw) - b) <= 1e-6 * b
    return {ch: {"signed_mean_frac": float(f[:, j].mean()), "mean_abs_frac": float(np.abs(f[:, j]).mean()),
                 "at_bound_frac": float(sat[:, j].mean())} for j, ch in enumerate(TWIST_CH)}


def step_length_mm(twist, dt):
    """Commanded translation per step, |(vx, vy, vz)| * dt in mm (first order in the rotation)."""
    tw = np.asarray(twist, np.float64)
    return np.linalg.norm(tw[..., :3], axis=-1) * dt * 1000.0


def keepbest_index(match_traj):
    """The pose-free keep-best step: argmax of S over steps 0..N (first maximum)."""
    return int(np.argmax(np.asarray(match_traj, np.float64)))


def lift_reconstruct(exyz0_z, clamp_mm):
    """Pre-lift axial offset in the target frame: the logged (post-lift) offset plus the lift."""
    return np.asarray(exyz0_z, np.float64) + np.asarray(clamp_mm, np.float64)


def dist(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0, "min": float(x.min()),
            "max": float(x.max()), "n": int(x.size)}


def fingerprint(eps):
    """The identical-episode setup fingerprint of make_tables.py (episode, case, organ, window, start errors)."""
    return [(e.get("ep"), e["patient"], e["target"], e["window"], round(float(e["start_pos"]), 6),
             round(float(e["start_ang"]), 6)) for e in eps]


def parse_meta(text):
    """TotalSegmentator meta.csv (semicolon separated, optional BOM) -> {image_id: row dict}."""
    text = text.lstrip("﻿")
    rd = csv.DictReader(text.splitlines(), delimiter=";")
    out = {}
    for r in rd:
        out[r["image_id"].strip()] = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
    return out


def sex_age_summary(meta, ids):
    rows = [meta[i] for i in ids]
    ages = [int(r["age"]) for r in rows if str(r.get("age", "")).strip().isdigit()]
    sex = collections.Counter((r.get("gender") or "").strip().lower() or "not recorded" for r in rows)
    return {"n": len(ids), "n_age": len(ages),
            "age_median": float(np.median(ages)) if ages else None,
            "age_min": int(min(ages)) if ages else None, "age_max": int(max(ages)) if ages else None,
            "age_mean": float(np.mean(ages)) if ages else None,
            "female": int(sex.get("f", 0)), "male": int(sex.get("m", 0)),
            "sex_not_recorded": int(sum(v for k, v in sex.items() if k not in ("f", "m")))}


def code_constants(env_src, contracts_src, goal_src, place_src):
    """Actuation constants parsed from source text (never imported)."""
    def one(pat, s, what):
        m = re.search(pat, s)
        if not m:
            raise ps.PlanStatsError(f"constant not found: {what}")
        return m
    ts = float(one(r'<option timestep="([0-9.]+)"', env_src, "timestep").group(1))
    settle = int(one(r"settle_steps=(\d+)", env_src, "settle_steps default").group(1))
    w = one(r"weld_solref=\(([0-9.]+),\s*([0-9.]+)\)", env_src, "weld_solref default")
    press = float(one(r'USBG_MAX_PRESS_MM",\s*"([0-9.]+)"', env_src, "press cap").group(1))
    dt = float(one(r"(?m)^DT_C\s*=\s*([0-9.]+)", contracts_src, "DT_C").group(1))
    mt = one(r"MAX_TWIST\s*=\s*np\.array\(\[([^\]]+)\]", goal_src, "MAX_TWIST").group(1)
    max_twist = [float(x) for x in mt.split(",")]
    ph = one(r"PROBE_HALF\s*=\s*\(([^)]+)\)", place_src, "PROBE_HALF").group(1)
    probe_half = [float(x) for x in ph.split(",")]
    return {"timestep_s": ts, "settle_steps": settle, "physics_time_per_step_s": ts * settle,
            "weld_solref_timeconst_s": float(w.group(1)), "weld_solref_dampratio": float(w.group(2)),
            "control_step_s": dt, "physics_share_of_control_step": ts * settle / dt,
            "press_cap_mm": press, "max_twist": max_twist, "probe_half_m": probe_half}


# ============================================================================================ and (rerun)
def _ladder_path(name):
    return os.path.join(RERUN, "ladder", name + ".json")


def _contact_path(name):
    return os.path.join(RERUN, "contact", name + ".json")


def rung_files():
    """rung key -> list of (json path, weights) for the extended-log rerun (contact on and off in each file)."""
    R = {"static": [(_ladder_path("static_physics_ct"), None)],
         "scripted": [(_ladder_path("scripted_physics_ct"), None)],
         "pose_oracle": [(_ladder_path("pose_oracle_physics_ct"), None)],
         "ct_base": [(_ladder_path(f"ibvs_base_physics_ct_s{s}"), None) for s in SEEDS]}
    for st in STAGES:
        R[f"learned_base_{st}"] = [(_ladder_path(f"ibvs_base_{st}_matched_s{s}"), None) for s in SEEDS]
        for arm, mk in (("ctmask", "ct"), ("matched", "learned")):
            for w, ph in (("imit", "imit"), ("final", "final")):
                R[f"{mk}_{ph}_{st}"] = [(_contact_path(f"{arm}_{st}_s{s}"), w) for s in SEEDS]
    for mk in ("ct", "learned"):
        for ph in ("imit", "final"):
            R[f"{mk}_{ph}_4stage"] = [x for st in STAGES for x in R[f"{mk}_{ph}_{st}"]]
    R["learned_base_4stage"] = [x for st in STAGES for x in R[f"learned_base_{st}"]]
    return R


RUNG_TEXT = {"static": "static start", "scripted": "scripted sweep", "pose_oracle": "pose oracle (privileged)",
             "ct_base": "CT-mask servo base (privileged mask)"}


def _cache_blocks():
    cache = {}

    def get(path):
        if path not in cache:
            cache[path] = blocks_with_traj(path)
        return cache[path]
    get.cache = cache
    return get


def summarize_block_set(sel, n_steps_expected):
    """sel: list of (result block, traj) for one rung and contact setting, all seeds/stages pooled."""
    pats = np.concatenate([np.asarray(tr["patient"]) for _, tr in sel])
    axes_fin = np.concatenate([tr["axes"][:, -1, :] for _, tr in sel]).astype(np.float64)
    axes_0 = np.concatenate([tr["axes"][:, 0, :] for _, tr in sel]).astype(np.float64)
    pos_fin = np.concatenate([tr["pos"][:, -1] for _, tr in sel]).astype(np.float64)
    eps = [e for r, _ in sel for e in r["episodes"]]
    tilt = np.asarray([float(e.get("window_tilt_deg") or 0.0) for e in eps])
    for _, tr in sel:
        if tr["pos"].shape[1] != n_steps_expected + 1:
            raise ps.PlanStatsError("unexpected horizon in a rerun block")
    out = {"n_episodes": int(len(eps)), "n_blocks": len(sel), "axes": {}, "tilt_split": {}}
    for j, ax in enumerate(AXES):
        pm = patient_means(np.abs(axes_fin[:, j]), pats)
        out["axes"][ax] = {"patient_mean_abs": level_mean(pm), "per_patient_abs": pm,
                           "signed_patient_mean": level_mean(patient_means(axes_fin[:, j], pats)),
                           "start_patient_mean_abs": level_mean(patient_means(np.abs(axes_0[:, j]), pats))}
    out["final_pos_patient_mean"] = level_mean(patient_means(pos_fin, pats))
    tang = tilt < TANGENT_MAX_DEG
    out["tilt_split"] = {"n_tangent_episodes": int(tang.sum()), "n_tilted_episodes": int((~tang).sum()),
                         "tilt_values_deg": sorted({round(float(t), 2) for t in tilt})}
    for j, ax in enumerate(AXES):
        out["tilt_split"][ax] = {"tangent_mean_abs": float(np.abs(axes_fin[tang, j]).mean()),
                                 "tilted_mean_abs": float(np.abs(axes_fin[~tang, j]).mean())}
    cf = np.asarray([float(e["contact_frac"]) for e in eps])
    out["contact_frac_mean"] = float(cf.mean())
    pos_all = np.concatenate([tr["pos"] for _, tr in sel]).astype(np.float64)
    out["keepbest"] = {"pos_patient_mean": level_mean(patient_means([float(e["keepbest_pos"]) for e in eps], pats)),
                       "step_mean": float(np.mean([int(e["keepbest_step"]) for e in eps])),
                       "min_mean_pos_step": int(np.argmin(pos_all.mean(0)))}
    gm = np.concatenate([tr["gap_min"] for _, tr in sel]).astype(np.float64)
    out["gap_min_mm"] = {"mean_steps": float(gm[:, 1:].mean()), "final_mean": float(gm[:, -1].mean()),
                         "min": float(gm[:, 1:].min()), "hover_frac": float((gm[:, 1:] > 0).mean())}
    lag = np.concatenate([tr["cmd_lag"] for _, tr in sel]).astype(np.float64)
    out["cmd_lag_mm"] = step_stats(lag)
    out["cmd_lag_mm"]["patient_mean"] = level_mean(patient_means(lag[:, 1:].mean(1), pats))
    tw = np.concatenate([tr["twist"] for _, tr in sel]).astype(np.float64)
    return out, pats, axes_fin, tw, lag, np.concatenate([tr["pos"] for _, tr in sel]).astype(np.float64)


def m6_m7(get, C, dt, bound):
    R = rung_files()
    rungs = {}
    raw = {}
    for key, files in R.items():
        rec = {"files": sorted({ps.rel(p) for p, _ in files}), "weights": sorted({str(w) for _, w in files})}
        for c in ("on", "off"):
            sel = [pick(get(p)[1], c, w) for p, w in files]
            s, pats, axf, tw, lag, pos = summarize_block_set(sel, 12)
            s["twist_fraction"] = twist_fraction(tw, bound)
            sl = step_length_mm(tw, dt)
            s["commanded_step_mm"] = {"mean": float(sl.mean()), "step1_mean": float(sl[:, 0].mean())}
            s["lag_over_step"] = float(s["cmd_lag_mm"]["mean"] / sl.mean()) if sl.mean() > 0 else None
            rec[c] = s
            raw[(key, c)] = (pats, axf, pos, lag, tw)
        # paired contrasts, contact off minus on, per axis and for the final translation error
        oc = {}
        for j, ax in enumerate(AXES):
            on = patient_means(np.abs(raw[(key, "on")][1][:, j]), raw[(key, "on")][0])
            off = patient_means(np.abs(raw[(key, "off")][1][:, j]), raw[(key, "off")][0])
            d = vec(off) - vec(on)
            oc[ax] = _contrast(d, C)
        on = patient_means(raw[(key, "on")][2][:, -1], raw[(key, "on")][0])
        off = patient_means(raw[(key, "off")][2][:, -1], raw[(key, "off")][0])
        oc["final_pos_mm"] = _contrast(vec(off) - vec(on), C)
        rec["off_minus_on"] = oc
        # tilted-window episodes: off minus on, episode-paired
        rec["off_minus_on_tilt_split"] = {
            ax: {"tangent": rec["off"]["tilt_split"][ax]["tangent_mean_abs"] - rec["on"]["tilt_split"][ax]["tangent_mean_abs"],
                 "tilted": rec["off"]["tilt_split"][ax]["tilted_mean_abs"] - rec["on"]["tilt_split"][ax]["tilted_mean_abs"]}
            for ax in AXES}
        rungs[key] = rec
    # CT-mask base: pose trajectories do not depend on the texture stage (masks are CT label slices)
    ident = {}
    for s in SEEDS:
        ref = get(_ladder_path(f"ibvs_base_physics_ct_s{s}"))[1]
        for st in STAGES[1:]:
            oth = get(_ladder_path(f"ibvs_base_{st}_ct_s{s}"))[1]
            for c in ("on", "off"):
                a, b = pick(ref, c)[1]["axes"], pick(oth, c)[1]["axes"]
                ident[f"s{s}_{st}_{c}"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    for name in ("scripted", "pose_oracle", "static"):
        ref = get(_ladder_path(f"{name}_physics_ct"))[1]
        for st in STAGES[1:]:
            oth = get(_ladder_path(f"{name}_{st}_ct"))[1]
            for c in ("on", "off"):
                a, b = pick(ref, c)[1]["axes"], pick(oth, c)[1]["axes"]
                ident[f"{name}_{st}_{c}"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    stage_identity = {"max_abs_axes_diff_vs_physics": ident, "max_over_all": float(max(ident.values())),
                      "note": "under CT masks and for the open-loop and oracle rungs the pose trajectory does not "
                              "depend on the texture stage, so the physics files stand for all four stages; keep-best "
                              "does depend on the stage because S includes the appearance term"}
    # oracle in step 1: commanded step against closed distance and lag
    orc = {}
    for c in ("on", "off"):
        pats, axf, pos, lag, tw = raw[("pose_oracle", c)]
        sl = step_length_mm(tw, dt)
        orc[c] = {"commanded_step1_mm_mean": float(sl[:, 0].mean()),
                  "closed_step1_mm_mean": float((pos[:, 0] - pos[:, 1]).mean()),
                  "cmd_lag_step1_mm_mean": float(lag[:, 1].mean()),
                  "start_pos_mm_mean": float(pos[:, 0].mean())}
    return rungs, stage_identity, orc


def _contrast(d, C):
    c = ps.contrast(d, design="n6", constants=C, units=PATIENTS)
    keep = ("n", "mean", "ci_t", "sd", "p_t", "sign_k", "sign_n", "p_signflip_two_sided", "ci_boot_patient",
            "per_unit")
    out = {k: c[k] for k in keep}
    out["test"] = "paired t (df 5), two-sided, post hoc, no multiplicity adjustment"
    out["lopo"] = lopo(c["per_unit"])
    return out


# ============================================================================================ starts
def starts(get):
    p = _ladder_path("static_physics_ct")
    r, tr = pick(get(p)[1], "on")
    eps = r["episodes"]
    lifted = np.asarray([bool(e["start_clamped"]) for e in eps])
    clamp = np.asarray([float(e["clamp_mm"]) for e in eps])
    exyz0 = tr["euclid_xyz"][:, 0, :].astype(np.float64)
    ax0 = tr["axes"][:, 0, :].astype(np.float64)
    pre_z = lift_reconstruct(exyz0[:, 2], clamp)
    pre = exyz0.copy()
    pre[:, 2] = pre_z
    tilt = np.asarray([float(e.get("window_tilt_deg") or 0.0) for e in eps])
    out = {"source": ps.rel(p) + " results[contact=on] and its .traj.npz (b0 euclid_xyz, axes at step 0)",
           "n_starts": int(len(eps)), "n_lifted": int(lifted.sum()),
           "lift_mm_lifted": dist(clamp[lifted]),
           "lift_mm_all_mean": float(clamp.mean()),
           "lifted_by_window": {"tangent": int(lifted[tilt < TANGENT_MAX_DEG].sum()),
                                "tilted": int(lifted[tilt >= TANGENT_MAX_DEG].sum()),
                                "n_tangent": int((tilt < TANGENT_MAX_DEG).sum()),
                                "n_tilted": int((tilt >= TANGENT_MAX_DEG).sum()),
                                "rule": "tangent when window_tilt_deg < 1 (beam along the skin normal; logged values "
                                        "0.00 to 0.03 deg), tilted otherwise (5.1 to 10.0 deg)"},
           "axial_offset_mm_after_lift": dist(exyz0[:, 2]),
           "axial_offset_mm_before_lift": dist(pre_z),
           "axial_deeper_than_target_frac_after": float((exyz0[:, 2] > 0).mean()),
           "axial_deeper_than_target_frac_before": float((pre_z > 0).mean()),
           "lateral_offset_mm": dist(exyz0[:, 0]), "elevational_offset_mm": dist(exyz0[:, 1]),
           "euclid_start_mm_after_lift": dist(np.linalg.norm(exyz0, axis=1)),
           "euclid_start_mm_before_lift": dist(np.linalg.norm(pre, axis=1)),
           "start_axes_patient_mean_abs": {ax: level_mean(patient_means(np.abs(ax0[:, j]), tr["patient"]))
                                           for j, ax in enumerate(AXES)},
           "start_pos_patient_mean": level_mean(patient_means(tr["pos"][:, 0], tr["patient"])),
           "nominal": {"lateral_elevational_mm": "uniform +/-20", "axial_mm": "uniform +/-10",
                       "tilt_deg": "uniform +/-18 about x and y", "roll_deg": "uniform +/-4",
                       "source": "scripts/eval_view_ladder.py defaults (lateral 0.02 m under the anatomical seat, "
                                 "--axial-m 0.01, --fan-deg 18) and src/usbg/goal_rl.py sample_setup (roll_deg 4)"},
           "check_pre_lift_axial_within_nominal": bool(np.all(np.abs(pre_z) <= 10.0 + 1e-3)),
           "check_lateral_within_nominal": bool(np.all(np.abs(exyz0[:, :2]) <= 20.0 + 1e-3))}
    if out["n_lifted"] != int(round(float(r["start_clamped_frac"]) * len(eps))):
        raise ps.PlanStatsError("lift count differs from the block's start_clamped_frac")
    return out


def shared_starts(get, extra_blocks):
    """Every rerun, 36-step and plug block read here against the archived static rung's setup fingerprint."""
    ref_path = os.path.join(V2, "table4", "ladder", "static_physics_ct.json")
    d = read_json(ref_path)
    ref = fingerprint([r for r in d["results"] if r.get("contact") == "on"][0]["episodes"])
    n, same, bad = 0, 0, []
    seen_paths = set()
    for path, (doc, blocks) in list(get.cache.items()):
        for r, _ in blocks:
            n += 1
            ok = fingerprint(r["episodes"]) == ref
            same += ok
            if not ok:
                bad.append(f"{ps.rel(path)}:{r.get('weights')}:{r.get('contact')}")
        seen_paths.add(path)
    for path, eps in extra_blocks:
        n += 1
        ok = fingerprint(eps) == ref
        same += ok
        if not ok:
            bad.append(ps.rel(path))
    # archived CSV: one start per (patient, organ, window) across every arm, seed and phase
    rows = ARCHIVED_ROWS()
    starts_by_cell = collections.defaultdict(set)
    for r_ in rows:
        starts_by_cell[(r_["patient"], r_["target"], r_["window"])].add((round(float(r_["start_pos"]), 9),
                                                                         round(float(r_["start_ang"]), 9)))
    return {"reference": ps.rel(ref_path) + " results[contact=on].episodes (fingerprint: episode, case, organ, window, "
                                            "start errors to 1e-6)",
            "n_blocks_checked": n, "n_identical": int(same), "differing": bad,
            "archived_csv_cells": len(starts_by_cell),
            "archived_csv_cells_with_one_start": int(sum(len(v) == 1 for v in starts_by_cell.values())),
            "archived_csv_rows": len(rows),
            "by_construction": "the setup of evaluation episode k is drawn from numpy default_rng(eval_seed + k) with "
                               "eval_seed 999 for every rung, mask, stage and calibration or training seed "
                               "(src/usbg/controllers_v21.py evaluate_ext; scripts/train_view_match_rl.py evaluate)",
            "shared_across_seeds": bool(same == n and not bad)}


_ARCH = []


def ARCHIVED_ROWS():
    if not _ARCH:
        p = os.path.join(V2, "table4", "episodes_long.csv")
        _note(p)
        with open(p, newline="") as fh:
            _ARCH.extend(csv.DictReader(fh))
    return _ARCH


# ============================================================================================ 
def h36_files():
    F = {"ct_base": [os.path.join(H36, f"ibvs_base_physics_ct_s{s}_n36.json") for s in SEEDS],
         "pose_oracle": [os.path.join(H36, "pose_oracle_physics_ct_n36.json")],
         "scripted": [os.path.join(H36, "scripted_physics_ct_n36.json")],
         "static": [os.path.join(H36, "static_physics_ct_n36.json")]}
    for st in STAGES:
        F[f"learned_base_{st}"] = [os.path.join(H36, f"ibvs_base_{st}_matched_s{s}_n36.json") for s in SEEDS]
    return F


def m8(get, C, bound, rungs12):
    F = h36_files()
    hz = {}
    for key, files in F.items():
        sel = [pick(get(p)[1], "on") for p in files]
        pats = np.concatenate([np.asarray(tr["patient"]) for _, tr in sel])
        A = np.concatenate([tr["axes"] for _, tr in sel]).astype(np.float64)        # (E, 37, 6)
        pos = np.concatenate([tr["pos"] for _, tr in sel]).astype(np.float64)
        S = np.concatenate([tr["S"] for _, tr in sel]).astype(np.float64)
        if A.shape[1] != 37:
            raise ps.PlanStatsError(f"{key}: expected 37 logged steps")
        rec = {"files": [ps.rel(p) for p in files], "steps": {}}
        for t in (0, 12, 36):
            rec["steps"][f"t{t}"] = {ax: level_mean(patient_means(np.abs(A[:, t, j]), pats)) for j, ax in enumerate(AXES)}
            rec["steps"][f"t{t}"]["pos_mm"] = level_mean(patient_means(pos[:, t], pats))
        rec["per_seed_t36"] = {}
        for p, (r, tr) in zip(files, sel):
            a = tr["axes"].astype(np.float64)
            rec["per_seed_t36"][os.path.basename(p)] = {
                **{ax: level_mean(patient_means(np.abs(a[:, 36, j]), tr["patient"])) for j, ax in enumerate(AXES)},
                "pos_mm": level_mean(patient_means(tr["pos"][:, 36], tr["patient"])),
                "keepbest_pos_mm": level_mean(patient_means([e["keepbest_pos"] for e in r["episodes"]], tr["patient"]))}
        if key == "ct_base":
            ch = {}
            for j, ax in enumerate(AXES):
                d = vec(patient_means(np.abs(A[:, 36, j]), pats)) - vec(patient_means(np.abs(A[:, 12, j]), pats))
                ch[ax] = _contrast(d, C)
            d = vec(patient_means(pos[:, 36], pats)) - vec(patient_means(pos[:, 12], pats))
            ch["pos_mm"] = _contrast(d, C)
            rec["t36_minus_t12"] = ch
        # keep-best: over 0..36 (logged field) and recomputed over 0..12 from the same trajectories
        eps = [e for r, _ in sel for e in r["episodes"]]
        kb36 = np.asarray([float(e["keepbest_pos"]) for e in eps])
        kb_chk = np.asarray([pos[i, keepbest_index(S[i])] for i in range(len(eps))])
        if np.max(np.abs(kb36 - kb_chk)) > 1e-3:
            raise ps.PlanStatsError(f"{key}: keep-best field differs from argmax S of the trajectory")
        kb12 = np.asarray([pos[i, keepbest_index(S[i, :13])] for i in range(len(eps))])
        rec["keepbest"] = {"t36_patient_mean": level_mean(patient_means(kb36, pats)),
                           "t12_within_36step_run_patient_mean": level_mean(patient_means(kb12, pats)),
                           "final36_patient_mean": level_mean(patient_means(pos[:, 36], pats)),
                           "keepbest_step36_mean": float(np.mean([int(e["keepbest_step"]) for e in eps])),
                           "min_mean_pos_step": int(np.argmin(pos.mean(0)))}
        d = vec(patient_means(kb36, pats)) - vec(patient_means(pos[:, 36], pats))
        rec["keepbest"]["t36_keepbest_minus_final"] = _contrast(d, C)
        rec["twist_fraction_36"] = twist_fraction(np.concatenate([tr["twist"] for _, tr in sel]), bound)
        hz[key] = rec
    # the 36-step run reproduces the 12-step run up to step 12 (same episodes, deterministic)
    rep, rep_kb = {}, {}
    for key in ("ct_base", "pose_oracle", "scripted", "static") + tuple(f"learned_base_{st}" for st in STAGES):
        r12 = rungs12[key]["on"]["axes"]
        rep[key] = max(abs(r12[ax]["patient_mean_abs"] - _t12_from(hz[key], ax)) for ax in AXES)
        rep_kb[key] = abs(rungs12[key]["on"]["keepbest"]["pos_patient_mean"]
                          - hz[key]["keepbest"]["t12_within_36step_run_patient_mean"])
    if max(rep.values()) > 1e-4 or max(rep_kb.values()) > 1e-3:
        raise ps.PlanStatsError(f"the 36-step arms do not reproduce the 12-step run up to step 12: {rep} {rep_kb}")
    return hz, {"max_abs_diff_t12_vs_12step_run": rep, "max_abs_diff_keepbest_first12_vs_12step_run": rep_kb}


def _t12_from(rec, ax):
    return rec["steps"]["t12"][ax]


def keepbest_archived(C):
    """12-step keep-best of every Table 4 rung from the archived episodes (contact on and off)."""
    rows = ARCHIVED_ROWS()

    def sel(contact, **flt):
        out = [r for r in rows if r["contact"] == contact and r["eval_randomize"] == "off"
               and all(r[k] == v for k, v in flt.items())]
        return out
    spec = {"static": dict(group="ladder", rung="static"), "scripted": dict(group="ladder", rung="scripted"),
            "pose_oracle": dict(group="ladder", rung="pose_oracle")}
    for st in STAGES:
        spec[f"ct_base_{st}"] = dict(group="ladder", rung="ibvs_base", mask="ct", stage=st)
        spec[f"learned_base_{st}"] = dict(group="ladder", rung="ibvs_base", mask="learned", stage=st)
        for arm, mk in (("ctmask", "ct"), ("matched", "learned")):
            for ph, pk in (("after_imitation", "imit"), ("final", "final")):
                spec[f"{mk}_{pk}_{st}"] = dict(group=arm, arm=arm, stage=st, phase=ph)
    off_spec = {}
    for k, f in spec.items():
        if f.get("group") in ("ctmask", "matched"):
            off_spec[k] = dict(f, group="contact")
        else:
            off_spec[k] = f
    out = {}
    for k in spec:
        rec = {}
        for c, sp in (("on", spec[k]), ("off", off_spec[k])):
            rs = sel(c, **sp)
            if not rs:
                continue
            pats = [r["patient"] for r in rs]
            fin = patient_means([float(r["final_pos"]) for r in rs], pats)
            kb = patient_means([float(r["keepbest_pos"]) for r in rs], pats)
            rec[c] = {"n_episodes": len(rs), "seeds": sorted({r["seed"] for r in rs}),
                      "stages": sorted({r["stage"] for r in rs}),
                      "final_patient_mean": level_mean(fin), "keepbest_patient_mean": level_mean(kb),
                      "keepbest_minus_final": _contrast(vec(kb) - vec(fin), C)}
        out[k] = rec
    return out


# ============================================================================================ plug rows (Table 5)
def _row(slot, rid, role, paths, weights, plug, bn=None, seeds=SEEDS, label=None):
    return {"slot": slot, "id": rid, "role": role, "paths": {s: paths(s) for s in seeds}, "weights": weights,
            "plug": plug, "bn": bn, "seeds": tuple(seeds), "label": label or rid}


def plug_rows():
    lad = os.path.join(V2, "table4", "ladder")
    S0 = (0,)
    rows = [
        _row("texture", "sonospade", "ref", lambda s: os.path.join(lad, f"ibvs_base_sonospade_matched_s{s}.json"),
             "zero_residual", False),
        _row("texture", "physics", "secondary", lambda s: os.path.join(lad, f"ibvs_base_physics_matched_s{s}.json"),
             "zero_residual", False),
        _row("texture", "cut", "secondary", lambda s: os.path.join(lad, f"ibvs_base_cut_matched_s{s}.json"),
             "zero_residual", False),
        _row("texture", "cyclegan", "secondary", lambda s: os.path.join(lad, f"ibvs_base_cyclegan_matched_s{s}.json"),
             "zero_residual", False)]
    for rid in ("fastcut", "dclgan", "unsb", "histmatch"):
        rows.append(_row("texture", rid, "family",
                         lambda s, r=rid: os.path.join(PLUG, "texture", "base", f"ibvs_base_{r}_matched_s{s}.json"),
                         "zero_residual", True))
    rows += [_row("mask", "unet", "ref", lambda s: os.path.join(lad, f"ibvs_base_sonospade_matched_s{s}.json"),
                  "zero_residual", False),
             _row("mask", "ct", "secondary", lambda s: os.path.join(lad, f"ibvs_base_sonospade_ct_s{s}.json"),
                  "zero_residual", False)]
    for rid, role, sd in (("lraspp_mbv3", "family", SEEDS), ("deeplabv3_mbv3", "family", SEEDS),
                          ("deeplabv3_r50", "family", SEEDS), ("fcn_r50", "family", SEEDS),
                          ("nnunet2d", "family", SEEDS), ("sam_unet", "family", S0), ("medsam_unet", "family", S0),
                          ("nnunet2d_e20", "secondary", S0), ("sam_priv", "secondary", S0),
                          ("medsam_priv", "secondary", S0)):
        rows.append(_row("mask", rid, role,
                         lambda s, r=rid: os.path.join(PLUG, "seg", "ladder", f"ibvs_base_sonospade_{r}_s{s}.json"),
                         "zero_residual", True, seeds=sd))
    rows += [_row("ctrl", "reinforce", "ref",
                  lambda s: os.path.join(V2, "table4", "ctmask", f"sonospade_deploy_s{s}.json"), "final", False),
             _row("ctrl", "servo_base", "secondary", lambda s: os.path.join(lad, f"ibvs_base_sonospade_ct_s{s}.json"),
                  "zero_residual", False),
             _row("ctrl", "reinforce_bneval", "secondary",
                  lambda s: os.path.join(V21, "s21_bneval", f"ctmask_sonospade_s{s}.json"), "final", False)]
    for rid in ("ppo_res", "sac_res", "td3_res", "a2c_res", "ppo_scratch", "dqn_scratch"):
        rows.append(_row("ctrl", rid, "family",
                         lambda s, r=rid: os.path.join(PLUG, "ctrl", r, f"sonospade_ct_s{s}.json"), None, True))
    for rid in ("bc_res", "dagger_res", "bc_scratch", "dagger_scratch"):
        rows.append(_row("ctrl", rid, "family",
                         lambda s, r=rid: os.path.join(PLUG, "ctrl", r, f"sonospade_ct_s{s}.json"), None, True,
                         bn="eval"))
    for rid in ("bc_res", "dagger_res"):
        rows.append(_row("ctrl", rid + "_bntrain", "secondary",
                         lambda s, r=rid: os.path.join(PLUG, "ctrl", r, f"sonospade_ct_s{s}.json"), None, True,
                         bn="train"))
    rows.append(_row("ctrl", "cem", "secondary", lambda s: os.path.join(PLUG, "ctrl", "cem", f"sonospade_ct_s{s}.json"),
                     None, True, seeds=S0))
    for rid, role in (("paper", "ref"), ("seg_only", "secondary"), ("ncc_only", "secondary"),
                      ("ncc_sector", "secondary"), ("ssim", "family"), ("nmi", "family"), ("pose", "secondary"),
                      ("pose_area", "secondary")):
        rows.append(_row("reward", rid, role,
                         lambda s, r=rid: os.path.join(PLUG, "reward", "train", r, f"sonospade_s{s}.json"),
                         "final", True))
    rows += [_row("renderer", "physics", "ref",
                  lambda s: os.path.join(V2, "table4", "contact", f"ctmask_physics_s{s}.json"), "final", False)]
    for rid, role, sd in (("lotus_orig", "family", SEEDS), ("lotus_reimpl", "family", SEEDS),
                          ("simus", "secondary", S0)):
        rows.append(_row("renderer", rid, role,
                         lambda s, r=rid: os.path.join(PLUG, "renderer", "ladder",
                                                       f"policy_v2physics_under_{r}_s{s}.json"),
                         "final", True, seeds=sd))
    for r in rows:
        if any(_blinded(p) for p in r["paths"].values()) or _blinded(r["id"]):
            raise ps.PlanStatsError(f"a blinded row slipped into the row list: {r['id']}")
    return rows


def _status_of(d):
    for blk in (d, d.get("provenance") or {}, d.get("provenance_usbg") or {}):
        if isinstance(blk, dict) and blk.get("analysis_status"):
            return blk["analysis_status"]
    return None


def episode_block(d, weights, bn=None):
    """make_tables._episode_block: the contact-on, DR-off block of one file in any of its layouts."""
    if weights in ("final", "after_imitation") and isinstance(d.get(weights), dict) and "episodes" in d[weights]:
        return d[weights], f"{weights}.episodes"
    want = bn or "train"
    bs = [r for r in d.get("results", []) if r.get("contact") == "on" and r.get("eval_randomize", "off") == "off"
          and r.get("bn_mode", "train") == want and (weights is None or r.get("weights") == weights)]
    if len(bs) != 1:
        raise ps.PlanStatsError(f"{len(bs)} contact-on, DR-off blocks (weights {weights}, BatchNorm {want})")
    sel = ["contact=on"] + ([f"weights={weights}"] if weights else []) + ([f"bn_mode={bn}"] if bn else [])
    return bs[0], f"results[{','.join(sel)}].episodes"


def plug_cells(static_fp):
    rows = plug_rows()
    cells, fps = {}, []
    for r in rows:
        cell = {}
        for s, p in r["paths"].items():
            if not os.path.exists(p):
                continue
            d = read_json(p)
            if r["plug"]:
                if _status_of(d) != PLUG_STATUS:
                    raise ps.PlanStatsError(f"{ps.rel(p)}: status {_status_of(d)!r}")
                prov = d.get("provenance") or {}
                if any(isinstance(x, dict) and x.get("smoke") for x in (d, d.get("plug"), prov.get("plug"),
                                                                         prov.get("v21_plug"))):
                    raise ps.PlanStatsError(f"{ps.rel(p)}: smoke run")
            b, kp = episode_block(d, r["weights"], r["bn"])
            eps = b["episodes"]
            if fingerprint(eps) != static_fp:
                raise ps.PlanStatsError(f"{ps.rel(p)}: episode setups differ from the archived static rung")
            fps.append((p, eps))
            pats = [e["patient"] for e in eps]
            cell[s] = {"path": ps.rel(p), "key": kp,
                       "final": patient_means([float(e["final_pos"]) for e in eps], pats),
                       "keepbest": patient_means([float(e["keepbest_pos"]) for e in eps], pats),
                       "keepbest_step_mean": float(np.mean([int(e["keepbest_step"]) for e in eps])),
                       "residual_frac": ((d.get("condition") or {}).get("residual_frac")),
                       "base": ((d.get("condition") or {}).get("base"))}
        cells[(r["slot"], r["id"])] = (r, cell)
    return cells, fps


TABLE5_TEX = os.path.join(PAPER, "tables", "pluggability.tex") if PAPER else None


def table5_printed(cells):
    """Rows whose difference the current Table 5 build prints: a src comment that names the row's file and reports
    'diff paired t'. BC and DAgger rows are told apart by the BatchNorm mode of the block."""
    lines = [ln for ln in read_text(TABLE5_TEX).splitlines() if "diff paired t" in ln]
    out = {}
    for (sl, rid), (r, cell) in cells.items():
        s0 = min(r["paths"])
        rp = ps.rel(r["paths"][s0])
        hit = [ln for ln in lines if rp in ln]
        if r["bn"]:
            hit = [ln for ln in hit if f"bn_mode={r['bn']}" in ln]
        elif r["slot"] == "ctrl" and r["plug"]:
            hit = [ln for ln in hit if "bn_mode=" not in ln]
        out[f"{sl}.{rid}"] = bool(hit)
    return out


def plug_summary(cells, C):
    refs = {r["slot"]: (r, c) for (sl, rid), (r, c) in cells.items() if r["role"] == "ref"}
    printed = table5_printed(cells)
    out = {}
    for (sl, rid), (r, cell) in cells.items():
        if not cell:
            out[f"{sl}.{rid}"] = {"status": "pending (no output file yet)", "design_seeds": list(r["seeds"])}
            continue
        seeds = sorted(cell)
        fin = float(np.mean([level_mean(cell[s]["final"]) for s in seeds]))
        kb = float(np.mean([level_mean(cell[s]["keepbest"]) for s in seeds]))
        rec = {"status": "complete" if len(seeds) == len(r["seeds"]) else "partial", "seeds": seeds,
               "design_seeds": list(r["seeds"]), "role": r["role"], "files": [cell[s]["path"] for s in seeds],
               "block": cell[seeds[0]]["key"], "final_mean_mm": fin, "keepbest_mean_mm": kb,
               "keepbest_minus_final_mm": kb - fin,
               "keepbest_step_mean": float(np.mean([cell[s]["keepbest_step_mean"] for s in seeds])),
               "residual_frac": sorted({str(cell[s]["residual_frac"]) for s in seeds}),
               "base": sorted({str(cell[s]["base"]) for s in seeds}),
               "difference_printed_in_table5_build": printed[f"{sl}.{rid}"]}
        rr, rc = refs[sl]
        common = sorted(set(seeds) & set(rc))
        if r["role"] != "ref" and common:
            for what in ("final", "keepbest"):
                d = (np.mean([vec(cell[s][what]) for s in common], axis=0)
                     - np.mean([vec(rc[s][what]) for s in common], axis=0))
                rec[f"minus_ref_{what}"] = _contrast(d, C)
                rec[f"minus_ref_{what}"]["seeds"] = common
                rec[f"minus_ref_{what}"]["reference"] = rr["id"]
        out[f"{sl}.{rid}"] = rec
    return out


# ============================================================================================ 
def c2_meta():
    meta = parse_meta(read_text(META_CSV))
    coh = read_json(COHORT_FROZEN)
    cohort = list(coh["cases"])
    if len(cohort) != 20:
        raise ps.PlanStatsError("the frozen cohort must have 20 cases")
    groups = {"held_out": list(PATIENTS), "train": list(TRAIN_PATIENTS), "cohort": cohort,
              "original_19": list(TRAIN_PATIENTS) + list(PATIENTS)}
    missing = [i for g in groups.values() for i in g if i not in meta]
    if missing:
        raise ps.PlanStatsError(f"cases missing from meta.csv: {sorted(set(missing))}")
    cases = {i: {"age": (int(meta[i]["age"]) if str(meta[i]["age"]).isdigit() else None),
                 "sex": (meta[i].get("gender") or "").lower() or "not recorded",
                 "study_type": meta[i].get("study_type"), "institute": meta[i].get("institute"),
                 "kvp": meta[i].get("kvp"), "dataset_split": meta[i].get("split"),
                 "group": ("held_out" if i in PATIENTS else "train" if i in TRAIN_PATIENTS else "cohort")}
             for g in ("train", "held_out", "cohort") for i in groups[g]}
    return {"source": ps.rel(META_CSV) + " (columns image_id, age, gender, study_type, institute, kvp, split); "
                      "cohort ids from " + ps.rel(COHORT_FROZEN) + " cases",
            "cases": cases, "summary": {g: sex_age_summary(meta, ids) for g, ids in groups.items()},
            "note": "gender is the dataset's own field; an empty field is reported as not recorded"}


def _arch_pp(**flt):
    rows = ARCHIVED_ROWS()
    rs = [r for r in rows if r["contact"] == "on" and r["eval_randomize"] == "off"
          and all(r[k] == v for k, v in flt.items())]
    return patient_means([float(r["final_pos"]) for r in rs], [r["patient"] for r in rs])


def c2_lopo(C, plug):
    """Leave-one-patient-out for each 5/6 contrast printed in the body (Sections 1 to 10) and for the Table 4 gaps."""
    R = read_json(os.path.join(V2, "RESULTS.json"))["table4"]
    static = _arch_pp(group="ladder", rung="static")
    cells = {}
    for st in STAGES:
        for arm, mk in (("ctmask", "ct"), ("matched", "learned")):
            cells[(mk, st, "final")] = _arch_pp(group=arm, arm=arm, stage=st, phase="final")
        cells[("learned", st, "base")] = _arch_pp(group="ladder", rung="ibvs_base", mask="learned", stage=st)
    items = {}

    def add(key, where, text, d, check=None, source=""):
        c = _contrast(d, C)
        if check is not None and abs(c["mean"] - check) > 1e-9 * max(1.0, abs(check)):
            raise ps.PlanStatsError(f"{key}: recomputed mean {c['mean']} differs from the printed source {check}")
        items[key] = {"where": where, "contrast": text, "source": source, **c}
    for st in STAGES:
        add(f"ctmask_final_minus_static_{st}", "Section 6.1",
            f"CT-mask final minus static start, {st}", vec(cells[("ct", st, "final")]) - vec(static),
            R["exploratory_final_minus_static"][f"ctmask/{st}"]["mean"],
            "outputs/v2/table4/episodes_long.csv; check RESULTS.json table4.exploratory_final_minus_static")
    add("learned_final_minus_static_sonospade", "Section 6.1", "learned-mask final minus static start, SonoSPADE",
        vec(cells[("learned", "sonospade", "final")]) - vec(static),
        R["exploratory_final_minus_static"]["matched/sonospade"]["mean"],
        "outputs/v2/table4/episodes_long.csv; check RESULTS.json table4.exploratory_final_minus_static")
    for st in STAGES:
        add(f"ct_to_learned_gap_{st}", "Section 6.2, Table 4, Abstract (physics), Sections 8.5 and 9",
            f"CT-to-learned-mask gap, {st} (learned-mask final minus CT-mask final)",
            vec(cells[("learned", st, "final")]) - vec(cells[("ct", st, "final")]),
            R["perception_gap"][st]["mean"],
            "outputs/v2/table4/episodes_long.csv; check RESULTS.json table4 gap of this stage")
    for st in ("physics", "cut", "cyclegan"):
        add(f"sonospade_minus_{st}_learned_final", "Section 6.3, Table 4",
            f"SonoSPADE minus {st}, learned-mask final",
            vec(cells[("learned", "sonospade", "final")]) - vec(cells[("learned", st, "final")]),
            R["learned_vs"][st]["mean"],
            "outputs/v2/table4/episodes_long.csv; check RESULTS.json table4.learned_vs")
    add("sonospade_minus_physics_learned_base", "Section 8.5 (original six)",
        "SonoSPADE minus physics, learned-mask servo base",
        vec(cells[("learned", "sonospade", "base")]) - vec(cells[("learned", "physics", "base")]),
        None, "outputs/v2/table4/episodes_long.csv (group ladder, rung ibvs_base, mask learned)")
    q8 = read_json(os.path.join(V21, "analysis_plan", "q8_gap_cause_v2.json"))
    m = q8["q9"]["family_Q9_A"]["members"]["R2_minus_base_ct"]
    add("hillclimb_minus_linear_servo_ct", "Section 6.5", "hill-climb minus linear servo, CT mask",
        vec(m["per_unit"]), m["mean"], "outputs/v2_1/analysis_plan/q8_gap_cause_v2.json "
                                       "q9.family_Q9_A.members.R2_minus_base_ct.per_unit")
    m = q8["q9"]["independent_target_speckle"]["tspk_base_learned"]["contrast"]
    add("independent_speckle_learned_base", "Section 8.2",
        "independent minus shared target speckle, learned-mask servo base (physics)", vec(m["per_unit"]), m["mean"],
        "outputs/v2_1/analysis_plan/q8_gap_cause_v2.json q9.independent_target_speckle.tspk_base_learned."
        "contrast.per_unit")
    # Table 5 rows and the Section 7.4 SSIM sentence: the plug contrasts computed above
    for k, rec in plug.items():
        c = rec.get("minus_ref_final")
        if c is None:
            continue
        items[f"table5_{k.replace('.', '_')}"] = {
            "printed": bool(rec.get("difference_printed_in_table5_build")),
            "where": "Table 5" + (", Sections 7.4 and 9.1" if k == "reward.ssim" else ""),
            "contrast": f"{k} minus its slot reference ({c['reference']}), final translation error, seeds {c['seeds']}",
            "source": "outputs/v2_1/plug and archived reference files (see plug_rows)",
            **{kk: vv for kk, vv in c.items() if kk not in ("reference", "seeds")}}
    for v in items.values():
        v.setdefault("printed", True)
    five = {k: v for k, v in items.items() if v["sign_k"] == 5 and v["n"] == 6 and v["printed"]}
    return {"items": items, "five_of_six": sorted(five),
            "five_of_six_not_printed": sorted(k for k, v in items.items()
                                              if v["sign_k"] == 5 and v["n"] == 6 and not v["printed"]),
            "five_of_six_sign_kept_in_all": sorted(k for k, v in five.items() if v["lopo"]["sign_kept_in_all"]),
            "five_of_six_sign_lost": sorted(k for k, v in five.items() if not v["lopo"]["sign_kept_in_all"]),
            "rule": "post hoc: each patient left out in turn; the contrast keeps its sign when all six leave-one-out "
                    "means share the sign of the full mean"}


def s0513_note(C, rungs):
    """The dissenting patient of several 5/6 contrasts: its CT-mask servo base and policy rows against its start."""
    rows = ARCHIVED_ROWS()
    st = _arch_pp(group="ladder", rung="static")
    base = _arch_pp(group="ladder", rung="ibvs_base", mask="ct", stage="physics")
    fin = {s: _arch_pp(group="ctmask", arm="ctmask", stage=s, phase="final") for s in STAGES}
    _ = rows
    return {"static_start_mm": st["s0513"], "ct_base_final_mm": base["s0513"],
            "ct_final_mm_by_stage": {s: fin[s]["s0513"] for s in STAGES},
            "ct_base_minus_static_by_patient": {p: base[p] - st[p] for p in PATIENTS},
            "ct_base_axes_on": {ax: rungs["ct_base"]["on"]["axes"][ax]["per_patient_abs"]["s0513"] for ax in AXES},
            "ct_base_axes_on_patient_mean": {ax: rungs["ct_base"]["on"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
            "source": "outputs/v2/table4/episodes_long.csv (archived) and the m6 rung block ct_base",
            "note": PI.text("phl s0513 note")}


# ============================================================================================ run
def run(args):
    C = ps.load_constants()
    asserted = ps.assert_constants(USED_CONSTANTS, C)
    get = _cache_blocks()
    # actuation constants from source text
    env_p = os.path.join(REPO, "src", "usbg", "mujoco_env.py")
    con_p = os.path.join(REPO, "src", "usbg", "_par", "contracts.py")
    goal_p = os.path.join(REPO, "src", "usbg", "goal_rl.py")
    pl_p = os.path.join(REPO, "src", "usbg", "placement.py")
    consts = code_constants(read_text(env_p), read_text(con_p), read_text(goal_p), read_text(pl_p))
    v21_env = os.path.join(V21_TREE, "src", "usbg", "mujoco_env.py")
    same_env = os.path.exists(v21_env) and ps.md5(_note(v21_env)) == ps.md5(env_p)
    sib = {}
    if os.path.exists(PAR_SIBLING):
        m = re.search(r"(?m)^DT_C\s*=\s*([0-9.]+)", read_text(PAR_SIBLING))
        sib = {"path": ps.rel(PAR_SIBLING), "DT_C": float(m.group(1)) if m else None}
    tv = os.path.join(V21_TREE, "scripts", "train_view_match_rl.py")
    mk = re.search(r"def make_env\(.*?\n(?:    .*\n)+", read_text(tv))
    make_env_settle = None if mk is None else ("settle_steps" in mk.group(0))
    consts.update({"sources": [ps.rel(p) for p in (env_p, con_p, goal_p, pl_p)],
                   "v21_mujoco_env_identical_md5": bool(same_env), "run_contracts": sib,
                   "make_env_overrides_settle_steps": make_env_settle,
                   "make_env_source": ps.rel(tv) + " make_env (USProbeEnv without settle_steps, so the default)"})
    if make_env_settle is not False:
        raise ps.PlanStatsError("make_env must not override settle_steps (else the default does not apply)")
    if sib and sib.get("DT_C") != consts["control_step_s"]:
        raise ps.PlanStatsError("DT_C differs between the vendored and the sibling contracts")
    dt = consts["control_step_s"]
    bound = np.asarray(consts["max_twist"], np.float64)

    rungs, stage_identity, oracle1 = m6_m7(get, C, dt, bound)
    st = starts(get)
    hz, hz_check = m8(get, C, bound, rungs)
    kb12 = keepbest_archived(C)
    d = read_json(os.path.join(V2, "table4", "ladder", "static_physics_ct.json"))
    static_fp = fingerprint([r for r in d["results"] if r.get("contact") == "on"][0]["episodes"])
    cells, plug_fps = plug_cells(static_fp)
    plug = plug_summary(cells, C)
    shared = shared_starts(get, plug_fps)
    meta = c2_meta()
    lo = c2_lopo(C, plug)
    s513 = s0513_note(C, rungs)
    # cross-checks against the archived values (rerun bit-identical; cell means as printed in Table 4)
    T4 = read_json(os.path.join(V2, "RESULTS.json"))["table4"]
    chk = {"ct_base_rerun_vs_archived_cell": abs(rungs["ct_base"]["on"]["final_pos_patient_mean"]
                                                 - T4["cell_means_mm"]["ctmask/physics/base"]),
           "scripted_rerun_vs_archived": abs(rungs["scripted"]["on"]["final_pos_patient_mean"]
                                             - T4["rungs_mm"]["scripted"]),
           "oracle_rerun_vs_archived": abs(rungs["pose_oracle"]["on"]["final_pos_patient_mean"]
                                           - T4["rungs_mm"]["pose_oracle"]),
           "ct_final_4stage_rerun_vs_archived": abs(rungs["ct_final_4stage"]["on"]["final_pos_patient_mean"]
                                                    - np.mean([T4["cell_means_mm"][f"ctmask/{s}/final"] for s in STAGES]))}
    if max(chk.values()) > 1e-4:
        raise ps.PlanStatsError(f"rerun does not reproduce the archived cells: {chk}")
    payload = {
        "label": LABEL, "status_note": STATUS_NOTE,
        "definitions": {k: v for k, v in DEFINITIONS.items()},
        "m7_actuation": {"constants": consts, "oracle_step1": oracle1,
                         "lag_by_rung": {k: {c: rungs[k][c]["cmd_lag_mm"] for c in ("on", "off")} for k in rungs},
                         "lag_by_rung_36": {k: _lag36(get, k) for k in h36_files()}},
        "m6_axes": {"rungs": rungs, "stage_identity": stage_identity, "rerun_vs_archived_abs_diff_mm": chk},
        "starts": st, "shared_starts": shared,
        "m8_horizon": {"arms36": hz, "checks": hz_check, "keepbest_archived_12": kb12, "keepbest_plug": plug},
        "c2": {"meta": meta, "lopo": lo, "s0513": s513},
    }
    out_path = os.path.abspath(args.out)
    if args.dry_run:
        print(json.dumps(ps._jsonable(_brief(payload)), indent=1)[:20000])
        print(f"[dry-run] {len(INPUTS)} inputs; nothing written")
        return 0
    w = ps.write_result(out_path, payload, analysis_status=STATUS, inputs=INPUTS, script=os.path.abspath(__file__),
                        constants_asserted=asserted, status_note=STATUS_NOTE)
    print(f"wrote {ps.rel(w)} ({len(INPUTS)} inputs)")
    return 0


def _lag36(get, key):
    sel = [pick(get(p)[1], "on") for p in h36_files()[key]]
    lag = np.concatenate([tr["cmd_lag"] for _, tr in sel]).astype(np.float64)
    return step_stats(lag)


DEFINITIONS = {
    "axes": "signed SE(3)-log error of the actual probe pose in the target probe frame (lateral x, elevational y, "
            "axial z in mm; out-of-plane tilt about x, in-plane tilt about y, roll about z in deg); per-axis error = "
            "|component|; patient-level mean = mean of the six per-patient means (episodes of every seed pooled)",
    "cmd_lag": "distance between the commanded (press-limited mocap) and the actual probe-face position after each "
               "step, steps 1..N pooled over episodes and seeds",
    "gap_min": "height of the lowest of the face centre and four face corners above the plane z = 0, steps 1..N",
    "contact_frac": "share of steps with a nonzero probe-skin normal force (episode field)",
    "twist_fraction": "executed twist over goal_rl.MAX_TWIST per channel (vx lateral, vy elevational, vz axial with "
                      "+ into tissue, wx out-of-plane tilt, wy in-plane tilt, wz roll), steps pooled",
    "keepbest": "translation error at the step of highest match score S over steps 0..N (pose-free selection)",
    "contrasts": "paired by patient (seeds averaged), two-sided paired t with df 5, 95% t CI, k of 6 sharing the sign "
                 "of the mean, exact sign-flip p and the patient bootstrap CI as sensitivity; post hoc, no "
                 "multiplicity adjustment, no decision",
    "tilt_split": "tangent windows (window_tilt_deg 0: the plane is the skin tangent) against tilted windows (the "
                  "beam tilts toward the organ, so the plane is tilted from the skin); episode-level means",
}


def _brief(p):
    r = p["m6_axes"]["rungs"]
    out = {"ct_base_on_axes": {ax: r["ct_base"]["on"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "ct_base_off_axes": {ax: r["ct_base"]["off"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "scripted_on": {ax: r["scripted"]["on"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "scripted_off": {ax: r["scripted"]["off"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "oracle_on": {ax: r["pose_oracle"]["on"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "oracle_off": {ax: r["pose_oracle"]["off"]["axes"][ax]["patient_mean_abs"] for ax in AXES},
           "lag": {k: {c: r[k][c]["cmd_lag_mm"] for c in ("on", "off")} for k in ("ct_base", "pose_oracle", "scripted")},
           "oracle_step1": p["m7_actuation"]["oracle_step1"], "constants": p["m7_actuation"]["constants"],
           "starts": p["starts"], "shared": {k: v for k, v in p["shared_starts"].items() if k != "differing"},
           "twist_ct_base_on": r["ct_base"]["on"]["twist_fraction"],
           "h36_ct_base": p["m8_horizon"]["arms36"]["ct_base"]["steps"],
           "h36_kb": {k: v["keepbest"]["t36_patient_mean"] for k, v in p["m8_horizon"]["arms36"].items()},
           "h36_checks": p["m8_horizon"]["checks"],
           "plug": {k: (v.get("status"), v.get("final_mean_mm"), v.get("keepbest_mean_mm"))
                    for k, v in p["m8_horizon"]["keepbest_plug"].items()},
           "lopo5": {k: (p["c2"]["lopo"]["items"][k]["mean"], p["c2"]["lopo"]["items"][k]["lopo"]["lopo_min"],
                         p["c2"]["lopo"]["items"][k]["lopo"]["lopo_max"],
                         p["c2"]["lopo"]["items"][k]["lopo"]["sign_kept_in_all"])
                     for k in p["c2"]["lopo"]["five_of_six"]},
           "meta": p["c2"]["meta"]["summary"], "s0513": p["c2"]["s0513"]}
    return out


# ============================================================================================ tables
def _f(x, d=2):
    return f"{x:.{d}f}"


def _sg(x, d=2):
    s = f"{x:+.{d}f}"
    return "$0.00$" if s in ("+0.00", "-0.00") and d == 2 else f"${s}$"


def _ci(c, d=2):
    lo, hi = c
    return f"$[{lo:+.{d}f}, {hi:+.{d}f}]$"


def _k(c):
    return f"{c['sign_k']}/{c['sign_n']}"


STYLE_BANNED = re.compile(r"\b(confirmed|significant\w*|comparable|deployable|superset|differentiable|"
                          r"observability gap|perception gap)\b", re.IGNORECASE)


def _dev_wording_check(text, name):
    """The development repository's own extra wording check of the printed tables (plan_private; skipped in public
    mode and in the public mirror)."""
    PP = SD.private_module()
    if PP is not None and PP.WORDING_RE.search(text):
        raise ps.PlanStatsError(f"{name}: {PP.WORDING_MSG}")


def _check_tex(text, name):
    printed = "\n".join(line.replace("\\%", "PCT").split("%")[0] for line in text.splitlines())
    if chr(0x2014) in text or "---" in text:
        raise ps.PlanStatsError(f"{name}: em-dash or three hyphens")
    if STYLE_BANNED.search(printed):
        raise ps.PlanStatsError(f"{name}: a never-print word: {STYLE_BANNED.search(printed).group(0)}")
    _dev_wording_check(text, name)
    for m in re.finditer(r"--", printed):
        a, b = printed[max(0, m.start() - 1)], printed[m.end():m.end() + 1]
        if not (a.isdigit() and b.isdigit()):
            raise ps.PlanStatsError(f"{name}: en-dash outside a numeric range")
    return text


HEAD = ("%% NEW BIB KEYS: none\n% generated by " + PI.text("posthoc_logs script") + " tables from " + JSON_REL
        + " (post hoc" + PI.text("phl head note") + ")\n% do not edit by hand: rerun the script\n")
# one comment line per table on the edits of three of these tables (no printed effect)
SCI_NOTE = "% " + PI.text("phl edit note") + PI.text("posthoc_logs script") + " tables: "
SCI_EDIT = {
    "posthoc_axes.tex": PI.text("phl edit axes"),
    "posthoc_keepbest.tex": PI.text("phl edit keepbest"),
    "posthoc_lopo.tex": PI.text("phl edit lopo"),
    "posthoc_starts.tex": PI.text("phl edit starts"),
}


def _head(name):
    return HEAD + SCI_NOTE + SCI_EDIT[name] + "\n" if name in SCI_EDIT else HEAD

AX_HEAD = r"Lat. & Elev. & Axial & Out & In & Roll"
RUNG_LABEL = {"scripted": "Scripted sweep", "pose_oracle": "Pose oracle$^{\\dagger}$",
              "ct_base": "CT-mask servo base$^{\\dagger}$", "ct_imit_4stage": "CT mask$^{\\dagger}$, after imitation",
              "ct_final_4stage": "CT mask$^{\\dagger}$, final, four stages",
              "learned_base_physics": "Learned-mask base, physics",
              "learned_base_cut": "Learned-mask base, CUT",
              "learned_base_cyclegan": "Learned-mask base, CycleGAN",
              "learned_base_sonospade": "Learned-mask base, SonoSPADE",
              "learned_final_4stage": "Learned mask, final, four stages",
              "static": "Static start"}


def tables(args):
    pub = SD.public()
    if not pub and args.out:
        raise ps.PlanStatsError("--out DIR is an option of public mode: add --public --results ROOT")
    if pub:
        if not args.out:
            raise ps.PlanStatsError("--public needs --out DIR (the tables are written to DIR/tables)")
        args.tables_dir = os.path.join(os.path.abspath(args.out), "tables")
    J = ps.read_json(args.json)
    P = JSON_REL
    T = {}
    r = J["m6_axes"]["rungs"]
    # ---------------------------------------------------------------- posthoc_axes.tex
    order = ["scripted", "pose_oracle", "ct_base", "ct_final_4stage", "learned_base_physics", "learned_base_cut",
             "learned_base_cyclegan", "learned_base_sonospade", "learned_final_4stage"]
    L = [_head("posthoc_axes.tex"), "\\begin{table*}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{Per-axis final error with contact on and off, 12 steps, the 60 held-out episodes "
         "(post hoc). Errors are patient-level means of the absolute SE(3)-log component in the target probe frame: "
         "lateral, elevational and axial translation, and out-of-plane tilt, in-plane tilt and roll. "
         "Base and policy rows pool seeds 0 to 2; four-stage rows also pool the four texture stages. "
         "In contact is the share of steps in contact. Gap is the mean height of the lowest face point above the "
         "plane, negative when pressed in. Lag is the mean distance between commanded and actual probe position after "
         "a step. "
         "The episodes are those of Table~\\ref{tab:ladder}. "
         "$^{\\dagger}$Privileged.} % src: " + P + " m6_axes.rungs.<rung>.<contact>.n_episodes (60 per seed), "
         "definitions.axes, definitions.contact_frac, definitions.gap_min, definitions.cmd_lag; 12 steps: n_steps of "
         "every rerun file",
         "\\label{tab:posthoc-axes}", "\\centering", "\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}l c rrr rrr r r r r@{}}", "\\toprule",
         " & & \\multicolumn{3}{c}{Translation (mm)} & \\multicolumn{3}{c}{Rotation ($^{\\circ}$)} & & & & \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         f"Rung & Contact & {AX_HEAD} & Final & In contact & Gap & Lag \\\\",
         " & & & & & & & & (mm) & (share) & (mm) & (mm) \\\\", "\\midrule"]
    s = J["starts"]["start_axes_patient_mean_abs"]
    L.append("Start (all rungs) & & " + " & ".join(_f(s[a]) for a in AXES)
             + f" & {_f(J['starts']['start_pos_patient_mean'])} & & & \\\\ % src: {P} starts.start_axes_patient_mean_abs.<axis>, "
               "starts.start_pos_patient_mean")
    for k in order:
        for c in ("on", "off"):
            b = r[k][c]
            L.append((RUNG_LABEL[k] if c == "on" else "") + f" & {c} & "
                     + " & ".join(_f(b["axes"][a]["patient_mean_abs"]) for a in AXES)
                     + f" & {_f(b['final_pos_patient_mean'])} & {_f(b['contact_frac_mean'])} & "
                       f"{_sg(b['gap_min_mm']['mean_steps'])} & {_f(b['cmd_lag_mm']['mean'])} \\\\ "
                       f"% src: {P} m6_axes.rungs.{k}.{c}.axes.<axis>.patient_mean_abs, final_pos_patient_mean, "
                       f"contact_frac_mean, gap_min_mm.mean_steps, cmd_lag_mm.mean")
        L.append("\\addlinespace[2pt]")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table*}", ""]
    T["posthoc_axes.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_contact.tex
    L = [HEAD, "\\begin{table*}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{Contact off minus contact on, per axis, paired by patient (post hoc, 12 steps). "
         "Columns give the patient-level mean absolute error, the paired difference with its 95\\% $t$ CI (df 5) and "
         "the patients of 6 sharing its sign. "
         "The last four columns split the episodes into the 15 windows where the plane is the skin tangent and the 45 "
         "where it is tilted from the skin; they are episode means. "
         "No multiplicity adjustment is made and no decision is drawn. $^{\\dagger}$Privileged.} % src: " + P
         + " definitions.contrasts; m6_axes.rungs.scripted.on.tilt_split.n_tangent_episodes (15), "
         "n_tilted_episodes (45)",
         "\\label{tab:posthoc-contact}", "\\centering", "\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}l l rr l rr rr@{}}", "\\toprule",
         " & & & & & \\multicolumn{2}{c}{On} & \\multicolumn{2}{c}{Off} \\\\",
         "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Rung & Axis & On & Off & Off minus on, 95\\% $t$ CI, $k$/6 & Tangent & Tilted & Tangent & Tilted \\\\",
         "\\midrule"]
    axn = {"lat_mm": "Lateral (mm)", "elev_mm": "Elevational (mm)", "ax_mm": "Axial (mm)",
           "tilt_out_deg": "Out-of-plane tilt ($^{\\circ}$)", "tilt_in_deg": "In-plane tilt ($^{\\circ}$)",
           "roll_deg": "Roll ($^{\\circ}$)"}
    for k in ("scripted", "pose_oracle", "ct_base", "ct_final_4stage"):
        for i, a in enumerate(AXES):
            on, off, c = r[k]["on"], r[k]["off"], r[k]["off_minus_on"][a]
            L.append((RUNG_LABEL[k] if i == 0 else "") + f" & {axn[a]} & {_f(on['axes'][a]['patient_mean_abs'])} & "
                     f"{_f(off['axes'][a]['patient_mean_abs'])} & {_sg(c['mean'])} {_ci(c['ci_t'])}, {_k(c)} & "
                     f"{_f(on['tilt_split'][a]['tangent_mean_abs'])} & {_f(on['tilt_split'][a]['tilted_mean_abs'])} & "
                     f"{_f(off['tilt_split'][a]['tangent_mean_abs'])} & {_f(off['tilt_split'][a]['tilted_mean_abs'])} \\\\ "
                     f"% src: {P} m6_axes.rungs.{k}.{{on,off}}.axes.{a}.patient_mean_abs; "
                     f"m6_axes.rungs.{k}.off_minus_on.{a}.{{mean,ci_t,sign_k}}; "
                     f"m6_axes.rungs.{k}.{{on,off}}.tilt_split.{a}")
        c = r[k]["off_minus_on"]["final_pos_mm"]
        L.append(f" & Final (mm) & {_f(r[k]['on']['final_pos_patient_mean'])} & {_f(r[k]['off']['final_pos_patient_mean'])} & "
                 f"{_sg(c['mean'])} {_ci(c['ci_t'])}, {_k(c)} & & & & \\\\ % src: {P} "
                 f"m6_axes.rungs.{k}.{{on,off}}.final_pos_patient_mean; m6_axes.rungs.{k}.off_minus_on.final_pos_mm")
        L.append("\\addlinespace[2pt]")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table*}", ""]
    T["posthoc_contact.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_starts.tex
    st = J["starts"]
    sh = J["shared_starts"]
    L = [_head("posthoc_starts.tex"), "\\begin{table}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\caption{Evaluation starts before and after the lift onto the contact plane (post hoc). "
         "Offsets are the start translation in the target probe frame. "
         "The pre-lift axial offset is the logged offset plus the lift. "
         "Positive axial offsets lie deeper than the target face.} % src: " + P + " starts.source",
         "\\label{tab:posthoc-starts}", "\\centering", "\\begin{tabular}{@{}l r@{}}", "\\toprule",
         "Quantity & Value \\\\", "\\midrule",
         f"Starts lifted & {st['n_lifted']} of {st['n_starts']} \\\\ % src: {P} starts.n_lifted, starts.n_starts",
         f"\\quad at tangent windows & {st['lifted_by_window']['tangent']} of {st['lifted_by_window']['n_tangent']} \\\\ "
         f"% src: {P} starts.lifted_by_window.tangent, n_tangent",
         f"\\quad at tilted windows & {st['lifted_by_window']['tilted']} of {st['lifted_by_window']['n_tilted']} \\\\ "
         f"% src: {P} starts.lifted_by_window.tilted, n_tilted",
         f"Lift of the lifted starts, mean (range), mm & {_f(st['lift_mm_lifted']['mean'], 1)} "
         f"({_f(st['lift_mm_lifted']['min'], 1)}--{_f(st['lift_mm_lifted']['max'], 1)}) \\\\ "
         f"% src: {P} starts.lift_mm_lifted.mean, min, max",
         f"Axial offset before the lift, mean (range), mm & {_sg(st['axial_offset_mm_before_lift']['mean'], 1)} "
         f"({_sg(st['axial_offset_mm_before_lift']['min'], 1)} to {_sg(st['axial_offset_mm_before_lift']['max'], 1)}) \\\\ "
         f"% src: {P} starts.axial_offset_mm_before_lift",
         f"Axial offset after the lift, mean (range), mm & {_sg(st['axial_offset_mm_after_lift']['mean'], 1)} "
         f"({_sg(st['axial_offset_mm_after_lift']['min'], 1)} to {_sg(st['axial_offset_mm_after_lift']['max'], 1)}) \\\\ "
         f"% src: {P} starts.axial_offset_mm_after_lift",
         f"Starts deeper than the target face, before and after (\\%) & "
         f"{_f(100 * st['axial_deeper_than_target_frac_before'], 0)} and {_f(100 * st['axial_deeper_than_target_frac_after'], 0)} \\\\ "
         f"% src: {P} starts.axial_deeper_than_target_frac_before, _after",
         f"Euclidean start distance before and after, mean, mm & {_f(st['euclid_start_mm_before_lift']['mean'])} and "
         f"{_f(st['euclid_start_mm_after_lift']['mean'])} \\\\ % src: {P} starts.euclid_start_mm_before_lift.mean, "
         "euclid_start_mm_after_lift.mean",
         "% " + PI.text("phl starts moved") + ": "
         f"Rows with the same 60 setups, of those read & {sh['n_identical']} of {sh['n_blocks_checked']} \\\\ "
         f"% src: {P} shared_starts.n_identical, n_blocks_checked",
         "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    T["posthoc_starts.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_lag.tex
    A = J["m7_actuation"]
    k_ = A["constants"]
    L = [HEAD, "\\begin{table}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\caption{Actuation as a lagged position servo (post hoc). "
         "Each control step integrates the commanded twist open loop into a target pose and then simulates a short "
         "physics interval of a probe welded to that pose. "
         "Lag is the distance between the commanded and the actual probe position after each step, pooled over steps, "
         "episodes and seeds. "
         "Step is the mean commanded translation per step. $^{\\dagger}$Privileged.} % src: " + P
         + " definitions.cmd_lag; m7_actuation.constants",
         "\\label{tab:posthoc-lag}", "\\centering", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}l c rrr r@{}}", "\\toprule",
         "Rung & Contact & Lag mean & p95 & Max & Step \\\\",
         " & & (mm) & (mm) & (mm) & (mm) \\\\", "\\midrule"]
    for k in ("scripted", "pose_oracle", "ct_base", "ct_final_4stage", "learned_base_sonospade",
              "learned_base_physics"):
        for c in ("on", "off"):
            g = r[k][c]["cmd_lag_mm"]
            L.append((RUNG_LABEL[k] if c == "on" else "") + f" & {c} & {_f(g['mean'])} & {_f(g['p95'])} & "
                     f"{_f(g['max'])} & {_f(r[k][c]['commanded_step_mm']['mean'])} \\\\ % src: {P} "
                     f"m6_axes.rungs.{k}.{c}.cmd_lag_mm.{{mean,p95,max}}, m6_axes.rungs.{k}.{c}.commanded_step_mm.mean")
    g = A["lag_by_rung_36"]["ct_base"]
    L.append(f"CT-mask servo base$^{{\\dagger}}$, 36 steps & on & {_f(g['mean'])} & {_f(g['p95'])} & {_f(g['max'])} & "
             f"\\\\ % src: {P} m7_actuation.lag_by_rung_36.ct_base")
    o = A["oracle_step1"]["on"]
    L += ["\\midrule",
          f"\\multicolumn{{6}}{{@{{}}l}}{{Control step {_f(k_['control_step_s'], 1)}~s; physics per step "
          f"{k_['settle_steps']} $\\times$ {_f(1000 * k_['timestep_s'], 0)}~ms = {_f(1000 * k_['physics_time_per_step_s'], 0)}~ms; "
          f"weld time constant {_f(k_['weld_solref_timeconst_s'], 2)}~s.}} \\\\ % src: {P} m7_actuation.constants."
          "{control_step_s, settle_steps, timestep_s, physics_time_per_step_s, weld_solref_timeconst_s}",
          f"\\multicolumn{{6}}{{@{{}}l}}{{Pose oracle$^{{\\dagger}}$, step 1, contact on: commands "
          f"{_f(o['commanded_step1_mm_mean'], 1)}~mm, closes {_f(o['closed_step1_mm_mean'], 1)}~mm, lags "
          f"{_f(o['cmd_lag_step1_mm_mean'], 1)}~mm.}} \\\\ % src: {P} m7_actuation.oracle_step1.on",
          "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    T["posthoc_lag.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_horizon.tex
    H = J["m8_horizon"]["arms36"]
    tw12 = r["ct_base"]["on"]["twist_fraction"]
    L = [HEAD, "\\begin{table*}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{The servo base at 12 and 36 steps, contact on (post hoc; the 36-step arms are exploratory). "
         "Errors are patient-level mean absolute SE(3)-log components; Out and In are the out-of-plane and in-plane "
         "tilts. The base rows pool seeds 0 to 2. "
         "The change row is the 36-step minus the 12-step value of the CT-mask base, paired by patient, with its "
         "95\\% $t$ CI and $k$/6. "
         "Bound rows give the executed twist as a fraction of its bound, averaged over the 12 steps, for the CT-mask "
         "base. $^{\\dagger}$Privileged.} % src: " + P + " m8_horizon.arms36.<arm>.files (36 steps; seeds 0 to 2 "
         "for the bases); definitions.twist_fraction",
         "\\label{tab:posthoc-horizon}", "\\centering", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l c rrr rrr r@{}}", "\\toprule",
         " & & \\multicolumn{3}{c}{Translation (mm)} & \\multicolumn{3}{c}{Rotation ($^{\\circ}$)} & \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         f"Arm & Step & {AX_HEAD} & Error (mm) \\\\", "\\midrule"]
    for k, lab in (("ct_base", "CT-mask servo base$^{\\dagger}$"), ("learned_base_physics", "Learned mask, physics"),
                   ("learned_base_sonospade", "Learned mask, SonoSPADE"), ("pose_oracle", "Pose oracle$^{\\dagger}$")):
        for i, t in enumerate(("t0", "t12", "t36")):
            v = H[k]["steps"][t]
            L.append((lab if i == 0 else "") + f" & {t[1:]} & " + " & ".join(_f(v[a]) for a in AXES)
                     + f" & {_f(v['pos_mm'])} \\\\ % src: {P} m8_horizon.arms36.{k}.steps.{t}")
        if k == "ct_base":
            ch = H[k]["t36_minus_t12"]
            L.append(" & 36 minus 12 & " + " & ".join(f"{_sg(ch[a]['mean'])}" for a in AXES)
                     + f" & {_sg(ch['pos_mm']['mean'])} \\\\ % src: {P} m8_horizon.arms36.ct_base.t36_minus_t12.<axis>.mean")
            L.append(" & 95\\% $t$ CI & " + " & ".join(
                f"$[{ch[a]['ci_t'][0]:+.1f}, {ch[a]['ci_t'][1]:+.1f}]$" for a in AXES + ("pos_mm",))
                     + f" \\\\ % src: {P} m8_horizon.arms36.ct_base.t36_minus_t12.<axis>.ci_t")
            L.append(" & $k$/6 & " + " & ".join(_k(ch[a]) for a in AXES + ("pos_mm",))
                     + f" \\\\ % src: {P} m8_horizon.arms36.ct_base.t36_minus_t12.<axis>.sign_k")
        L.append("\\addlinespace[2pt]")
    L.append("\\midrule")
    L.append("Bound, signed mean & 1--12 & " + " & ".join(_sg(tw12[ch]['signed_mean_frac']) for ch in TWIST_CH)
             + f" & \\\\ % src: {P} m6_axes.rungs.ct_base.on.twist_fraction.<channel>.signed_mean_frac")
    L.append("Bound, mean absolute & 1--12 & " + " & ".join(_f(tw12[ch]['mean_abs_frac']) for ch in TWIST_CH)
             + f" & \\\\ % src: {P} m6_axes.rungs.ct_base.on.twist_fraction.<channel>.mean_abs_frac")
    L.append("Bound, share of steps at it & 1--12 & " + " & ".join(_f(tw12[ch]['at_bound_frac']) for ch in TWIST_CH)
             + f" & \\\\ % src: {P} m6_axes.rungs.ct_base.on.twist_fraction.<channel>.at_bound_frac")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    T["posthoc_horizon.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_keepbest.tex
    K = J["m8_horizon"]["keepbest_archived_12"]
    PL = J["m8_horizon"]["keepbest_plug"]
    L = [_head("posthoc_keepbest.tex"), "\\begin{table}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         "\\caption{Keep-best error against final error, contact on (post hoc). "
         "Keep-best is the translation error at the step of highest match score $S$, a selection that reads no pose. "
         "The 12-step rows are the episodes of Table~\\ref{tab:ladder}, seeds pooled; four-stage rows average the "
         "four stages. "
         "The 36-step rows are exploratory. "
         "Plug rows use the episodes and seeds of Table~\\ref{tab:pluggability}. $^{\\dagger}$Privileged.} % src: "
         + P + " definitions.keepbest; m8_horizon.keepbest_archived_12.<row>.on.seeds; "
         "m8_horizon.keepbest_plug.<row>.seeds",
         "\\label{tab:posthoc-keepbest}", "\\centering", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l rr@{}}", "\\toprule", "Row & Final (mm) & Keep-best (mm) \\\\", "\\midrule",
         "\\multicolumn{3}{@{}l}{\\textit{Reference ladder, 12 steps}} \\\\"]

    def kb4(prefix):
        f = np.mean([K[f"{prefix}_{s}"]["on"]["final_patient_mean"] for s in STAGES])
        b = np.mean([K[f"{prefix}_{s}"]["on"]["keepbest_patient_mean"] for s in STAGES])
        return f, b
    for k, lab in (("static", "Static start"), ("scripted", "Scripted sweep"), ("pose_oracle", "Pose oracle$^{\\dagger}$")):
        L.append(f"{lab} & {_f(K[k]['on']['final_patient_mean'])} & {_f(K[k]['on']['keepbest_patient_mean'])} \\\\ "
                 f"% src: {P} m8_horizon.keepbest_archived_12.{k}.on.{{final_patient_mean,keepbest_patient_mean}}")
    for pre, lab in (("ct_base", "CT-mask servo base$^{\\dagger}$, four stages"),
                     ("ct_imit", "CT mask$^{\\dagger}$, after imitation, four stages"),
                     ("ct_final", "CT mask$^{\\dagger}$, final, four stages")):
        f, b = kb4(pre)
        L.append(f"{lab} & {_f(f)} & {_f(b)} \\\\ % src: {P} mean over stages of "
                 f"m8_horizon.keepbest_archived_12.{pre}_<stage>.on.{{final_patient_mean,keepbest_patient_mean}}")
    for st_ in STAGES:
        for pre, lab in (("learned_base", "servo base"), ("learned_final", "final")):
            k = f"{pre}_{st_}"
            L.append(f"Learned mask, {lab}, {V2_NAME[st_]} & {_f(K[k]['on']['final_patient_mean'])} & "
                     f"{_f(K[k]['on']['keepbest_patient_mean'])} \\\\ % src: {P} m8_horizon.keepbest_archived_12.{k}.on")
    L.append("\\multicolumn{3}{@{}l}{\\textit{36 steps (exploratory)}} \\\\")
    for k, lab in (("ct_base", "CT-mask servo base$^{\\dagger}$"), ("pose_oracle", "Pose oracle$^{\\dagger}$"),
                   ("scripted", "Scripted sweep")) + tuple((f"learned_base_{s}", f"Learned mask, servo base, {V2_NAME[s]}")
                                                           for s in STAGES):
        kb = H[k]["keepbest"]
        L.append(f"{lab} & {_f(kb['final36_patient_mean'])} & {_f(kb['t36_patient_mean'])} \\\\ % src: {P} "
                 f"m8_horizon.arms36.{k}.keepbest.{{final36_patient_mean,t36_patient_mean}}")
    L.append("\\multicolumn{3}{@{}l}{\\textit{Plug rows, 12 steps}} \\\\")
    for key, rec in PL.items():
        if rec.get("status") not in ("complete", "partial"):
            continue
        lab = PLUG_LABEL.get(key, key.replace("_", "\\_"))
        part = "" if rec["status"] == "complete" else f", seeds {','.join(str(s) for s in rec['seeds'])}"
        L.append(f"{lab}{part} & {_f(rec['final_mean_mm'])} & {_f(rec['keepbest_mean_mm'])} \\\\ % "
                 f"{PLUG_LABEL_NOTE.get(key, '')}src: {P} m8_horizon.keepbest_plug.{key}.{{final_mean_mm,keepbest_mean_mm}}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    T["posthoc_keepbest.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_cases.tex
    M = J["c2"]["meta"]
    L = [HEAD, "\\begin{table}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\small",
         "\\caption{Age and sex of the CT cases, from the TotalSegmentator metadata (post hoc). "
         "Sex is the dataset's own field, and an empty field is counted as not recorded.} % src: " + P
         + " c2.meta.source",
         "\\label{tab:posthoc-cases}", "\\centering", "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
         "Group or case & Cases & Age, median (range) & Female & Male & Not recorded \\\\", "\\midrule"]
    for g, lab in (("train", "Training"), ("held_out", "Held out"), ("cohort", "Replication cohort")):
        sm = M["summary"][g]
        med = f"{sm['age_median']:.1f}".rstrip("0").rstrip(".")
        L.append(f"{lab} & {sm['n']} & {med} ({sm['age_min']}--{sm['age_max']}) & {sm['female']} & "
                 f"{sm['male']} & {sm['sex_not_recorded']} \\\\ % src: {P} c2.meta.summary.{g}")
    L += ["\\midrule", "\\multicolumn{6}{@{}l}{\\textit{Held-out cases: age and sex}} \\\\"]
    for p in PATIENTS:
        cse = M["cases"][p]
        L.append(f"{p} & & {cse['age']} & \\multicolumn{{3}}{{l}}{{{SEX_TXT.get(cse['sex'], cse['sex'])}}} \\\\ "
                 f"% src: {P} c2.meta.cases.{p}.age, .sex")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    T["posthoc_cases.tex"] = "\n".join(L)
    # ---------------------------------------------------------------- posthoc_lopo.tex
    LO = J["c2"]["lopo"]
    L = [_head("posthoc_lopo.tex"), "\\begin{table*}[tbp]", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
         f"\\caption{{Leave-one-patient-out check of the {len(LO['five_of_six'])} contrasts listed, each printed with 5 "
         "of 6 patients sharing its sign (post hoc); other contrasts with 5 of 6 are not checked. Each patient is left "
         "out in turn and the mean of the other five is recomputed. Where: where the contrast prints. "
         "The sign column says whether all six such means keep the sign of the full mean. "
         "Contrasts of the primary results come from the reference evaluation; the others are exploratory.} % src: " + P
         + " c2.lopo.rule, c2.lopo.five_of_six",
         "\\label{tab:posthoc-lopo}", "\\centering", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X l r r l l@{}}", "\\toprule",
         "Contrast (mm) & Where & Mean & $k$/6 & Leave-one-out range & Sign kept; opposing \\\\", "\\midrule"]
    for key in LO["five_of_six"]:
        it = LO["items"][key]
        lp = it["lopo"]
        dd = 3 if abs(it["mean"]) < 0.2 else 2
        L.append(f"{_lopo_label(key)} & {_lopo_where(key, it['where'])} & {_sg(it['mean'], dd)} & "
                 f"{it['sign_k']}/6 & {_sg(lp['lopo_min'], dd)} to {_sg(lp['lopo_max'], dd)} & "
                 f"{'yes' if lp['sign_kept_in_all'] else 'no'}; {', '.join(lp['dissenting_patients'])} \\\\ "
                 f"% src: {P} c2.lopo.items.{key}.{{mean,sign_k,lopo.lopo_min,lopo.lopo_max,lopo.sign_kept_in_all,"
                 "lopo.dissenting_patients}")
    L += ["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""]
    T["posthoc_lopo.tex"] = "\n".join(L)
    os.makedirs(args.tables_dir, exist_ok=True)
    if pub:
        sys.path.insert(0, SD.paper_dir())
        import tex_comments
    for name, text in T.items():
        _check_tex(text, name)
        if pub:     # the printed text only: every LaTeX comment removed (tex_comments.py of the paper build)
            text = tex_comments.strip_comments(text)[0]
        with open(os.path.join(args.tables_dir, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("wrote", ps.rel(os.path.join(args.tables_dir, name)))
    return 0


V2_NAME = {"physics": "physics", "cut": "CUT", "cyclegan": "CycleGAN", "sonospade": "SonoSPADE"}
SEX_TXT = {"f": "female", "m": "male", "not recorded": "not recorded"}
PLUG_LABEL = {
    "texture.sonospade": "Texture: SonoSPADE (reference)", "texture.physics": "Texture: none, physics frame",
    "texture.cut": "Texture: CUT", "texture.cyclegan": "Texture: CycleGAN", "texture.fastcut": "Texture: FastCUT",
    "texture.dclgan": "Texture: DCLGAN", "texture.unsb": "Texture: UNSB", "texture.histmatch": "Texture: histogram matching",
    "mask.unet": "Mask: U-Net (reference)", "mask.ct": "Mask: CT label$^{\\dagger}$",
    "mask.lraspp_mbv3": "Mask: LR-ASPP, MobileNetV3-L", "mask.deeplabv3_mbv3": "Mask: DeepLabV3, MobileNetV3-L",
    "mask.deeplabv3_r50": "Mask: DeepLabV3, ResNet-50", "mask.fcn_r50": "Mask: FCN, ResNet-50",
    "mask.nnunet2d": "Mask: nnU-Net 2D, 2 epochs", "mask.sam_unet": "Mask: SAM, U-Net boxes",
    "mask.medsam_unet": "Mask: MedSAM, U-Net boxes", "mask.nnunet2d_e20": "Mask: nnU-Net 2D, 20 epochs",
    "mask.sam_priv": "Mask: SAM, CT boxes$^{\\dagger}$", "mask.medsam_priv": "Mask: MedSAM, CT boxes$^{\\dagger}$",
    "ctrl.reinforce": "Controller: REINFORCE residual (reference)", "ctrl.servo_base": "Controller: servo base alone",
    "ctrl.reinforce_bneval": "Controller: REINFORCE, BatchNorm eval", "ctrl.ppo_res": "Controller: PPO, residual",
    "ctrl.sac_res": "Controller: SAC, residual", "ctrl.td3_res": "Controller: TD3, residual",
    "ctrl.a2c_res": "Controller: A2C, residual", "ctrl.ppo_scratch": "Controller: PPO, full twist",
    "ctrl.dqn_scratch": "Controller: DQN, 13 moves", "ctrl.bc_res": "Controller: BC, residual",
    "ctrl.dagger_res": "Controller: DAgger, residual", "ctrl.bc_scratch": "Controller: BC, full twist",
    "ctrl.dagger_scratch": "Controller: DAgger, full twist",
    "ctrl.bc_res_bntrain": "Controller: BC, residual, BatchNorm train",
    "ctrl.dagger_res_bntrain": "Controller: DAgger, residual, BatchNorm train",
    "ctrl.cem": "Controller: CEM look-ahead$^{\\dagger}$",
    "reward.paper": "Reward: paper $S$ (reference)", "reward.seg_only": "Reward: organ overlap only",
    "reward.ncc_only": "Reward: appearance NCC only", "reward.ncc_sector": "Reward: NCC over the sector",
    "reward.ssim": "Reward: SSIM term", "reward.nmi": "Reward: NMI term",
    "reward.pose": "Reward: change in pose distance$^{\\dagger}$",
    "reward.pose_area": "Reward: pose distance plus area$^{\\dagger}$",
    "renderer.physics": "Renderer: physics (reference)", "renderer.lotus_orig": "Renderer: LOTUS, original code",
    "renderer.lotus_reimpl": "Renderer: LOTUS, authors' reimplementation", "renderer.simus": "Renderer: SIMUS",
}
# the src-comment note of the renamed row, in the form that posthoc_keepbest.tex holds (no printed effect)
PLUG_LABEL_NOTE = {"renderer.lotus_reimpl": PI.text("lotus label src")}


def _lopo_where(key, stored):
    """Label-based pointers to where each contrast prints now, in place of the
    hard-coded section numbers stored in posthoc_logs.json (c2.lopo.items.<key>.where), several of which went stale when
    sentences moved. The supplement resolves the article's labels through xr-hyper."""
    for pre, ref in LOPO_WHERE:
        if key.startswith(pre):
            return ref
    raise SystemExit(f"posthoc_lopo: no Where pointer for {key} (stored: {stored})")


LOPO_WHERE = [
    ("ctmask_final_minus_static_", "Section~\\ref{supp:static}"),
    ("learned_final_minus_static_", "Section~\\ref{sec:ladder-absolute}"),
    ("ct_to_learned_gap_", "Section~\\ref{sec:ladder-gap}"),
    ("sonospade_minus_physics_learned_base", "Section~\\ref{sec:checks-cohort}"),
    ("sonospade_minus_", "Section~\\ref{sec:ladder-gap}"),
    ("hillclimb_minus_linear_servo_ct", "Section~\\ref{sec:gap-cause}"),
    ("independent_speckle_learned_base", "Table~\\ref{tab:checks-summary}"),
    ("table5_", "Table~\\ref{tab:pluggability}"),
]


def _lopo_label(key):
    if key in LOPO_LABEL:
        return LOPO_LABEL[key]
    if key.startswith("table5_"):
        sl, rid = key[len("table5_"):].split("_", 1)
        return PLUG_LABEL.get(f"{sl}.{rid}", rid.replace("_", " ")).replace(" (reference)", "") + ", minus its reference"
    return key.replace("_", " ")


LOPO_LABEL = {
    "ctmask_final_minus_static_physics": "CT-mask final minus static, physics",
    "ctmask_final_minus_static_cut": "CT-mask final minus static, CUT",
    "ctmask_final_minus_static_cyclegan": "CT-mask final minus static, CycleGAN",
    "ctmask_final_minus_static_sonospade": "CT-mask final minus static, SonoSPADE",
    "learned_final_minus_static_sonospade": "Learned-mask final minus static, SonoSPADE",
    "ct_to_learned_gap_physics": "CT-to-learned-mask gap, physics",
    "ct_to_learned_gap_cyclegan": "CT-to-learned-mask gap, CycleGAN",
    "sonospade_minus_physics_learned_final": "SonoSPADE minus physics, learned-mask final",
    "sonospade_minus_cut_learned_final": "SonoSPADE minus CUT, learned-mask final",
    "sonospade_minus_cyclegan_learned_final": "SonoSPADE minus CycleGAN, learned-mask final",
    "sonospade_minus_physics_learned_base": "SonoSPADE minus physics, learned-mask servo base",
    "hillclimb_minus_linear_servo_ct": "Hill-climb minus servo base, CT mask",
    "independent_speckle_learned_base": "Independent minus shared target speckle, learned-mask base",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--out", default=DEFAULT_OUT)
    a.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("tables")
    b.add_argument("--json", default=DEFAULT_OUT)
    b.add_argument("--tables-dir", default=DEFAULT_TABLES)
    b.add_argument("--public", action="store_true", help="public mode: result files under --results ROOT")
    b.add_argument("--results", help="public mode: the result root (outputs/, data/, segmenter/outputs/)")
    b.add_argument("--out", help="public mode: the output root DIR (DIR/tables/posthoc_*.tex)")
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else tables(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PI.PlanInputsError as e:      # a missing or incomplete public_inputs/plan_constants.json: one line
        raise SystemExit(f"plan inputs: {e}")
