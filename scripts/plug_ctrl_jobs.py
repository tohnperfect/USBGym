#!/usr/bin/env python3
"""Write the v2.1 pluggability job spec of the "controller / policy learning" slot:
outputs/v2_1/jobs/plug_ctrl.json (EXPLORATORY, opt-in; new file; NOT launched; refuses to overwrite).

Design
  Conditions (4): {physics, SonoSPADE} x {CT mask, texture-matched learned mask}. Physics is the untextured
  reference arm and SonoSPADE the paper's texture (and its realism-best stage); the CT mask isolates control from
  segmentation and the matched learned mask is the non-privileged arm. CUT and CycleGAN lie between these two
  textures in the paper's per-texture ablation and would double the cost without a new contrast for a learner
  comparison, so they are left to the texture slot.
  Seeds 0, 1, 2. Residual learners use the paper's servo base of the same (texture, mask, seed), read from the
  v2 policy checkpoint (outputs/v2/table4/{ctmask,matched}/<texture>_deploy_s<seed>.pt, key 'ibvs'), with the
  paper's residual bound 0.5 MAX_TWIST; the learned mask is the stage- and seed-matched segmenter
  segtrain/outputs/v2/us_seg_<texture>_s<seed>.pt; render seed = seed (as the paper).
  Budgets, as in the paper: 57,600 env steps for RL (its REINFORCE budget); 40,000 env steps for BC and DAgger
  (its 40,000 imitation samples) with 8,000 gradient steps. Evaluation on the 6 held-out patients: eval seed 999,
  60 crossed episodes, contact on.
  Tier 1 (must): PPO and SAC residuals, DQN from scratch, BC and DAgger residuals on all 4 conditions; the
  eval-only rows random, servo base, the paper's v2 policies (final and imitation weights), static, pose
  oracle; the parity check and the summary table.
  Tier 2 (should): PPO from scratch and A2C residual on the 2 CT-mask conditions; the CEM look-ahead (privileged)
  on the 2 CT-mask conditions, seed 0.
  Tier 3 (optional): TD3 residual, BC and DAgger from scratch on the 2 CT-mask conditions.

Cost model (est_h): measured on this machine (Apple M5 Pro, MPS, one process, TRAIN-patient smoke):
  env step incl. resets: physics/CT 13.2 ms, physics/learned 17.9 ms, SonoSPADE/CT 27.0 ms, SonoSPADE/learned
  31.4 ms; learner cost per env step: PPO 14.0, A2C 10.0, SAC 26.6 (batch 128, one gradient step per 2 env
  steps), TD3 18.0, DQN 5.8 ms; BC 2.0 and DAgger 4.4 ms per env step plus 15 ms per gradient step. A 60-episode
  evaluation is 14 renders per episode plus 30 s of loading. CEM: the archived look-ahead runs (0.80 h
  SonoSPADE, about 0.35 h physics at 12 steps). Every estimate x 1.15 for MPS sharing among concurrent jobs.

  PYTHONPATH=src .venv/bin/python scripts/plug_ctrl_jobs.py --out outputs/v2_1/jobs/plug_ctrl.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = f"{ROOT}/.venv/bin/python"
TRAIN_SCRIPT = f"{ROOT}/scripts/plug_train_sb3.py"
COLLECT_SCRIPT = f"{ROOT}/scripts/plug_ctrl_collect.py"
SEGDIR = os.path.join(os.environ.get("USBG_SEG_REPO", os.path.join(ROOT, "segtrain")), "outputs", "v2")
TEXTURE = {"physics": None, "sonospade": f"{ROOT}/data/dataset/v2/sonospade_curvi.pt"}
WINDOWS = f"{ROOT}/data/cache/windows/usbg-windows-v1.json"
OUTDIR = f"{ROOT}/outputs/v2_1/plug/ctrl"
SEEDS = (0, 1, 2)
CONDS = ("physics_ct", "physics_matched", "sonospade_ct", "sonospade_matched")
CT_CONDS = ("physics_ct", "sonospade_ct")
ENV_MS = {"physics_ct": 13.2, "physics_matched": 17.9, "sonospade_ct": 27.0, "sonospade_matched": 31.4}
LEARNER_MS = {"ppo": 14.0, "a2c": 10.0, "sac": 26.6, "td3": 18.0, "dqn": 5.8}
IMIT_MS = {"bc": 2.0, "dagger": 4.4}
MARGIN = 1.15
RL_STEPS, IMIT_STEPS, FIT_STEPS = 57600, 40000, 8000
CEM_H = {"physics_ct": 0.40, "sonospade_ct": 0.85}


def md5(p):
    from usbg.provenance import md5 as _md5
    return _md5(p)


def cond_parts(cond, seed):
    tex, arm = cond.split("_")
    mask = "ct" if arm == "ct" else "learned"
    v2arm = "ctmask" if arm == "ct" else "matched"
    seg = f"{SEGDIR}/us_seg_{tex}_s{seed}.pt" if mask == "learned" else None
    servo = f"{ROOT}/outputs/v2/table4/{v2arm}/{tex}_deploy_s{seed}.pt"
    args = ["--texture", tex, "--texture-ckpt", TEXTURE[tex] or "none", "--mask", mask]
    if seg:
        args += ["--seg-ckpt", seg]
    return tex, mask, seg, servo, args


def eval_h(cond, blocks=1):
    return blocks * (60 * 14 * ENV_MS[cond] / 1000.0 + 30.0) / 3600.0


def job(name, label, algo, cond, seed, extra, est_h, tier, note, base_servo=True, policy_pt=None):
    from usbg.benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    tex, mask, seg, servo, cargs = cond_parts(cond, seed)
    out = f"{OUTDIR}/{label}/{cond}_s{seed}.json"
    cmd = [PY, TRAIN_SCRIPT, "--algo", algo, *cargs, "--seed", str(seed)]
    files = [WINDOWS]
    if base_servo:
        cmd += ["--servo-base", servo]
        files.append(servo)
    if policy_pt:
        cmd += ["--policy-pt", policy_pt]
        files.append(policy_pt)
    if seg:
        files.append(seg)
    if TEXTURE[tex]:
        files.append(TEXTURE[tex])
    cmd += [*extra, "--train-cases", *TRAIN_PATIENTS, "--eval-cases", *TEST_PATIENTS, "--targets", "liver", "kidney",
            "--min-windows", "2", "--eval-episodes", "60", "--eval-seed", "999", "--eval-contact", "on",
            "--torch-threads", "4", "--label", label, "--out", out]
    return {"name": f"v21:plug:ctrl:{label}:{cond}:s{seed}", "cmd": cmd, "out": out,
            "est_h": round(est_h * MARGIN, 3),
            "env": {"PYTHONPATH": f"{ROOT}/src", "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    "USBG_LEARNED_SEG": "0", "US_SEG_CKPT": "", "USBG_WINDOWS": ""},
            "deps": [], "files": sorted(set(files)),
            "expect": {"seat": "anatomical", "texture_md5": md5(TEXTURE[tex]) if TEXTURE[tex] else None,
                       "windows_md5": md5(WINDOWS), "seg_md5": md5(seg) if seg else None},
            "cwd": ROOT, "note": f"tier {tier}; {note}"}


def build():
    J = []
    rl = {"ppo_res": ("ppo", True, 1, CONDS), "sac_res": ("sac", True, 1, CONDS), "dqn_scratch": ("dqn", False, 1, CONDS),
          "ppo_scratch": ("ppo", False, 2, CT_CONDS), "a2c_res": ("a2c", True, 2, CT_CONDS),
          "td3_res": ("td3", True, 3, CT_CONDS)}
    for label, (algo, res, tier, conds) in rl.items():
        for cond in conds:
            for s in SEEDS:
                h = RL_STEPS * (ENV_MS[cond] + LEARNER_MS[algo]) / 3.6e6 + eval_h(cond)
                extra = ["--base", "servo" if res else "none", "--total-steps", str(RL_STEPS)]
                J.append(job(None, label, algo, cond, s, extra, h, tier,
                             f"SB3 {algo.upper()} {'residual over the paper servo base' if res else 'from scratch'}; "
                             f"{RL_STEPS} env steps; trains then evaluates", base_servo=res))
    imit = {"bc_res": ("bc", True, 1, CONDS), "dagger_res": ("dagger", True, 1, CONDS),
            "bc_scratch": ("bc", False, 3, CT_CONDS), "dagger_scratch": ("dagger", False, 3, CT_CONDS)}
    for label, (algo, res, tier, conds) in imit.items():
        for cond in conds:
            for s in SEEDS:
                h = IMIT_STEPS * (ENV_MS[cond] + IMIT_MS[algo]) / 3.6e6 + FIT_STEPS * 0.015 / 3600 + eval_h(cond, 2)
                extra = ["--base", "servo" if res else "none"]
                J.append(job(None, label, algo, cond, s, extra, h, tier,
                             f"{algo} from the privileged pose-oracle teacher, GoalPolicy net, {IMIT_STEPS} env steps, "
                             f"{FIT_STEPS} gradient steps; {'residual' if res else 'from scratch'}", base_servo=res))
    for cond in CONDS:
        for s in SEEDS:
            J.append(job(None, "random", "random", cond, s, ["--base", "none"], eval_h(cond) + 0.005, 1,
                         "eval only: uniform random actions (floor row)", base_servo=False))
            J.append(job(None, "servo_base", "servo_base", cond, s, ["--base", "servo"], eval_h(cond) + 0.005, 1,
                         "eval only: the paper's moment servo base alone; parity with outputs/v2/table4/ladder/ibvs_base_*"))
            for w, suf in (("v2_final", ""), ("v2_imit", "_imit")):
                tex, mask, seg, servo, _ = cond_parts(cond, s)
                pt = servo[:-3] + f"{suf}.pt"
                J.append(job(None, w, "v2_policy", cond, s, ["--base", "servo"], eval_h(cond, 2) + 0.005, 1,
                             f"eval only: the paper's v2 policy ({'REINFORCE final' if not suf else 'imitation only'}) "
                             "through the gym; parity with the archived oracle-ladder episodes", policy_pt=pt))
        J.append(job(None, "static", "static", cond, 0, ["--base", "none"], eval_h(cond) + 0.005, 1,
                     "eval only: no commanded motion; parity with ladder/static_*_ct for CT", base_servo=False))
        J.append(job(None, "pose_oracle", "pose_oracle", cond, 0, ["--base", "pose_oracle"], eval_h(cond) + 0.005, 1,
                     "eval only: PRIVILEGED analytic base (ceiling); parity with ladder/pose_oracle_*_ct for CT",
                     base_servo=False))
    for cond in CT_CONDS:
        J.append(job(None, "cem", "cem", cond, 0, ["--base", "none", "--cem-k", "64", "--cem-rounds", "2",
                                                    "--cem-elite", "8", "--cem-seed-offset", "7907"],
                     CEM_H[cond] / MARGIN, 2, "eval only: PRIVILEGED CEM look-ahead bound (controllers_v21) through the gym; "
                                            "parity with outputs/v2_1/s22_lookahead/lookahead_*_ct_s0_n12", base_servo=False))
    names = [j["name"] for j in J]
    assert len(names) == len(set(names))
    parity_deps = [j["name"] for j in J if any(f":{k}:" in j["name"] for k in
                                                ("v2_final", "v2_imit", "servo_base", "static", "pose_oracle", "cem"))]
    for what, deps, out in (("parity", parity_deps, f"{OUTDIR}/parity.json"), ("table", names, f"{OUTDIR}/summary.json")):
        J.append({"name": f"v21:plug:ctrl:collect:{what}", "cmd": [PY, COLLECT_SCRIPT, what, "--root", OUTDIR, "--out", out],
                  "out": out, "est_h": 0.01,
                  "env": {"PYTHONPATH": f"{ROOT}/src", "USBG_SEAT": "anatomical", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                          "USBG_LEARNED_SEG": "0", "US_SEG_CKPT": "", "USBG_WINDOWS": ""},
                  "deps": deps, "files": [WINDOWS], "expect": {"seat": "anatomical", "windows_md5": md5(WINDOWS)},
                  "cwd": ROOT, "note": f"tier 1; {what} over the slot's outputs (runs after its deps; a failed dep blocks it)"})
    return J


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{ROOT}/outputs/v2_1/jobs/plug_ctrl.json")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    J = build()
    missing = sorted({f for j in J for f in j["files"] if not os.path.exists(f)})
    if missing:
        raise SystemExit(f"missing input files: {missing}")
    clash = [j["out"] for j in J if os.path.exists(j["out"])]
    if clash:
        raise SystemExit(f"outputs already exist: {clash[:5]}")
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(J, fh, indent=1)
    os.replace(tmp, a.out)
    by = {}
    for j in J:
        t = j["note"].split(";")[0]
        by.setdefault(t, [0, 0.0])
        by[t][0] += 1
        by[t][1] += j["est_h"]
    print(f"wrote {a.out}: {len(J)} jobs, {sum(j['est_h'] for j in J):.1f} job-h; " +
          ", ".join(f"{k}: {n} jobs {h:.1f} h" for k, (n, h) in sorted(by.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
