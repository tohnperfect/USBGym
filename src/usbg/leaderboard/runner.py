"""usbg.leaderboard.runner: run a slot condition with a plug-in and write submission.json.

For each seed the runner executes the steps of the slot condition, each in its own process through
usbg.leaderboard._exec (which installs the plug-in), with the arguments of the reported runs: the same scripts,
patients, targets, windows, evaluation seed 999, 60 crossed episodes, 12 steps and contact on. It then reads the
per-episode final translation errors of the contact-on block, checks that every episode has the setup of the
fixed design (the identical-episode audit), and writes the submission with its summary.

Run it from the root of a clone with an editable install (the scripts/ and segtrain/ folders of the clone are
used; a wheel install has neither). The submission records its commands with the interpreter as "python" and
every path relative to the repository root (or under "~" for the home folder), so the file names no local folder
when it is pasted into a public issue. --device sets the device of the plug-in and of the evaluation (through
USBG_DEVICE, which usbg.render_lotus.get_device reads), and the provenance records that device.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from . import protocol as PRO
from . import submission as SUB

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS_PKG = os.path.join(os.path.dirname(HERE), "data", PRO.WINDOWS_FILE)
TRAIN = list(PRO.TRAIN_PATIENTS)
TEST = list(PRO.HELD_OUT_PATIENTS)
SKIP_PAIRS = "s0058:kidney,s0522:kidney,s0727:kidney"     # training pairs without a screened kidney window
SEG_ITERS, SEG_POSES = 4000, 27
SONOSPADE_MD5 = "70440a5c6167e69489835ef553bfe37c"
COMMON_EVAL = ["--train-cases", *TRAIN, "--eval-cases", *TEST, "--targets", *PRO.TARGETS,
               "--min-windows", str(PRO.MIN_WINDOWS), "--eval-episodes", str(PRO.N_EPISODES),
               "--eval-seed", str(PRO.EVAL_SEED), "--eval-design", PRO.EVAL_DESIGN]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def public_path(p, root=None):
    """A path as the submission records it: relative to the repository root when inside it, '~/...' under the
    home folder, else '<abs>/<file name>'."""
    root = os.path.abspath(root or os.getcwd())
    ap = os.path.abspath(p)
    if ap == root or ap.startswith(root + os.sep):
        return os.path.relpath(ap, root).replace(os.sep, "/")
    home = os.path.expanduser("~")
    if home and home != os.sep and ap.startswith(home + os.sep):
        return "~/" + os.path.relpath(ap, home).replace(os.sep, "/")
    return "<abs>/" + os.path.basename(ap)


def public_command(cmd, root=None):
    """A command for the submission's provenance: the interpreter as 'python' and every absolute path argument
    (also after '=') through public_path."""
    out = []
    for i, x in enumerate(cmd):
        if i == 0 and x == sys.executable:
            out.append("python")
            continue
        k, eq, v = x.partition("=") if x.startswith("--") else ("", "", x)
        if os.path.isabs(v):
            out.append(f"{k}{eq}{public_path(v, root)}")
        else:
            out.append(x)
    return " ".join(out)


def plugin_sha256(entry):
    """sha256 over the plug-in's source: its module file, or every .py file of its package (sorted, relative
    paths with '/' so the hash is the same on every operating system)."""
    import importlib.util
    if entry == "reference":
        return hashlib.sha256(b"reference").hexdigest()
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    spec = importlib.util.find_spec(entry.partition(":")[0])
    if spec is None or not spec.origin:
        raise SystemExit(f"cannot find the source of {entry}")
    files = [spec.origin]
    if os.path.basename(spec.origin) == "__init__.py":
        files = sorted(glob.glob(os.path.join(os.path.dirname(spec.origin), "**", "*.py"), recursive=True))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, os.path.dirname(spec.origin)).replace(os.sep, "/").encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def need(path, what, md5_want=None):
    if not path or not os.path.exists(path):
        raise SystemExit(f"{what}: {path!r} not found")
    if md5_want and md5(path) != md5_want:
        raise SystemExit(f"{what}: {path} has md5 {md5(path)}, the reported runs used {md5_want}")
    return os.path.abspath(path)


def slot_steps(slot, seed, work, a):
    """[(name, env, argv after `--`, output JSON or None)] of one seed of a slot condition."""
    out = os.path.join(work, f"{slot}_s{seed}.json")
    base_env = {"USBG_SEAT": PRO.SEAT, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "USBG_WINDOWS": a.windows,
                "USBG_LB_WORK": work}
    if slot == "renderer":
        from .assets import verify
        pj = need(os.path.join(a.assets_dir, f"usbg_physics_policy_s{seed}.json"),
                  f"the physics-trained policy of seed {seed} (python -m usbg.leaderboard fetch)")
        need(pj[:-5] + ".pt", f"the physics-trained policy weights of seed {seed}")
        verify(a.assets_dir, [f"usbg_physics_policy_s{seed}.json", f"usbg_physics_policy_s{seed}.pt"])
        return [("evaluate", dict(base_env, USBG_LEARNED_SEG="0"),
                 ["eval_view_ladder", "--rung", "policy", "--policy-json", pj, "--weights", "final", "--contact",
                  PRO.CONTACT, "--ext-log", "--out", out], out)]
    if slot == "texture":
        split = os.path.join(_repo(), "segtrain", "data", "real", "kidneyus_split.npz")
        if not os.path.exists(split):
            raise SystemExit("the texture slot trains the matched U-Net with the KidneyUS training split, which needs "
                             "segtrain/data/real/kidneyus_split.npz (the fixed test assignment of the reported runs). "
                             "That file is not public yet: request a maintainer run with your gate record "
                             "(SUBMITTING.md)")
        marker = os.path.join(work, "lb_plugin_stage.pt")
        nn = os.path.join(work, "nnunet")
        ds = os.path.join(nn, f"Dataset{900 + seed}_USOrgansLB_s{seed}")
        ck = os.path.join(work, f"us_seg_lb_plugin_s{seed}.pt")
        env0 = dict(base_env, USBG_LEARNED_SEG="0", US_SEG_CKPT="")
        return [("marker", None, marker, None),
                ("segmenter data", env0,
                 ["segtrain:scripts/build_dataset.py", "--out", nn, "--dataset-id", str(900 + seed), "--name",
                  f"USOrgansLB_s{seed}", "--translator", marker, "--cases", "train", "--role", "train",
                  "--seat-organs", "liver,kidney", "--strict-seat", "--skip-pairs", SKIP_PAIRS, "--require-seat",
                  PRO.SEAT, "--n-poses", str(SEG_POSES), "--seed", str(seed)], None),
                ("segmenter training", env0,
                 ["segtrain:scripts/train.py", "--dataset", ds, "--iters", str(SEG_ITERS), "--seed", str(seed),
                  "--kidneyus", "--vitale-split", a.vitale_split, "--forbid-test-patients", "--out", ck], None),
                ("evaluate", dict(base_env, USBG_LEARNED_SEG="1", US_SEG_CKPT=ck),
                 ["eval_view_ladder", "--rung", "ibvs_base", "--translator", "lb_plugin", "--texture-ckpt", marker,
                  *COMMON_EVAL, "--calib-seed", str(seed), "--contact", PRO.CONTACT, "--ext-log", "--out", out], out)]
    sono = need(a.sonospade_ckpt, "the SonoSPADE stage checkpoint (maintainers)", SONOSPADE_MD5)
    if slot == "mask":
        return [("evaluate", dict(base_env, USBG_LEARNED_SEG="1"),
                 ["eval_view_ladder", "--rung", "ibvs_base", "--translator", "sonospade", "--texture-ckpt", sono,
                  *COMMON_EVAL, "--calib-seed", str(seed), "--contact", PRO.CONTACT, "--ext-log", "--out", out], out)]
    if slot == "ctrl":
        sb = need(os.path.join(a.reference_dir, f"sonospade_deploy_s{seed}.pt"), "the reference policy (servo base)")
        return [("train and evaluate", dict(base_env, USBG_LEARNED_SEG="0"),
                 ["ctrl_eval", "--texture-ckpt", sono, "--servo-base", sb, "--seed", str(seed), "--out", out], out)]
    if slot == "reward":
        im = need(os.path.join(a.reference_dir, f"sonospade_deploy_s{seed}_imit.pt"), "the imitation checkpoint")
        return [("train and evaluate", dict(base_env, USBG_LEARNED_SEG="0"),
                 ["plug_reward_train", "--reward", "lb_plugin", "--", "--translator", "sonospade", "--texture-ckpt",
                  sono, "--fix-residual", "--residual-image", "--ibvs-features", "rich",
                  "--w-seg", str(PRO.REWARD_WEIGHTS["w_seg"]), "--w-app", str(PRO.REWARD_WEIGHTS["w_app"]),
                  "--imitation", "0", "--iters", "300", "--episodes-per-iter", "16", "--n-steps", str(PRO.N_STEPS),
                  *COMMON_EVAL, "--contact", PRO.CONTACT, "--seed", str(seed), "--ablation", "plug:reward:lb_plugin",
                  "--init-from", im, "--curve-every", "50", "--out", out], out)]
    raise SystemExit(f"slot {slot!r} has no runner (the CT-anatomy slot is run by the maintainers)")


def _repo():
    return os.path.abspath(os.environ.get("USBG_REPO", os.getcwd()))


def contact_on_block(d):
    """The per-episode records of the contact-on, DR-off evaluation block of a result file."""
    if isinstance(d.get("final"), dict) and d["final"].get("episodes"):
        return d["final"]["episodes"]
    bs = [r for r in d.get("results", []) if r.get("contact", "on") == "on" and r.get("eval_randomize", "off") == "off"
          and r.get("weights") in ("final", "zero_residual", "plugin")]
    if len(bs) != 1:
        raise SystemExit(f"expected one contact-on, DR-off block, found {len(bs)}")
    return bs[0]["episodes"]


def episode_audit(eps, board, patients=None):
    """Every evaluated episode has the setup of the fixed design (ep, patient, target, start errors to 1e-6)."""
    ref = {e["ep"]: e for e in board["episodes"]}
    bad = []
    for e in eps:
        r = ref.get(int(e["ep"]))
        if r is None or (e["patient"], e.get("target_requested") or e["target"]) != (r["patient"], r["target"]) or \
                round(float(e["start_pos"]), 6) != r["start_pos"] or round(float(e["start_ang"]), 6) != r["start_ang"]:
            bad.append(int(e["ep"]))
    want = [e["ep"] for e in board["episodes"] if patients is None or e["patient"] in patients]
    got = sorted(int(e["ep"]) for e in eps)
    if got != want:
        bad.append(f"episodes {got} != {want}")
    return bad


def run(a):
    from . import recompute as RC
    board = RC.load_board()
    slot = a.slot
    if slot not in SUB.RUNNABLE_SLOTS:
        raise SystemExit(f"--slot {slot}: the CT-anatomy slot is run by the maintainers (SUBMITTING.md)")
    seeds = [int(s) for s in (a.seeds or (PRO.SEEDS if a.learned else PRO.SEED0))]
    if not os.path.isfile(os.path.join(_repo(), "scripts", "eval_view_ladder.py")):
        raise SystemExit("run the leaderboard from the root of a USB Gym clone with an editable install (pip install "
                         "-e .): the runner needs its scripts/ and segtrain/ folders (or set USBG_REPO to the clone)")
    if a.device and a.device.split(":")[0] not in ("cpu", "mps", "cuda"):
        raise SystemExit(f"--device {a.device!r}: cpu, mps or cuda")
    work = os.path.abspath(a.work_dir or os.path.join("outputs", "leaderboard", a.slot, a.run_name))
    if not a.dry_run:
        os.makedirs(work, exist_ok=True)
    plan, t0 = [], time.time()
    for s in seeds:
        plan.append((s, slot_steps(slot, s, work, a)))
    cmds = []
    for s, steps in plan:
        for name, env, argv, out in steps:
            if name == "marker":
                continue
            cmd = [sys.executable, "-m", "usbg.leaderboard._exec", "--slot", slot, "--plugin", a.plugin]
            if a.patients:
                cmd += ["--patients", *a.patients]
            if a.device:
                cmd += ["--device", a.device]
                env = dict(env, USBG_DEVICE=a.device)
            if a.privileged:
                cmd += ["--privileged"]
            cmds.append((s, name, env, cmd + ["--", *argv], out))
    if a.dry_run:
        for s, name, env, cmd, out in cmds:
            print(f"# seed {s}: {name}")
            print(" ".join(f"{k}={public_path(v) if os.path.isabs(str(v)) else v}" for k, v in sorted(env.items())),
                  public_command(cmd))
        return 0
    per_seed, wall, logs = {}, {}, []
    for s, steps in plan:
        for name, env, argv, out in steps:
            if name == "marker":
                from ._exec import write_texture_marker
                write_texture_marker(argv, a.plugin)
    for s, name, env, cmd, out in cmds:
        if out and os.path.exists(out):
            raise SystemExit(f"{out} exists: use a new --run-name or --work-dir (results are never overwritten)")
        e = dict(os.environ, **env)
        if env.get("US_SEG_CKPT") == "":
            e.pop("US_SEG_CKPT", None)
        e["PYTHONPATH"] = os.pathsep.join(x for x in (os.path.dirname(os.path.dirname(HERE)),
                                                       os.environ.get("PYTHONPATH")) if x)
        ts = time.time()
        print(f"[run] seed {s}: {name}", flush=True)
        log_path = os.path.join(work, f"{slot}_s{s}_{name.replace(' ', '_')}.log")
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, env=e, stdout=lf, stderr=subprocess.STDOUT).returncode
        wall[f"s{s}:{name}"] = round(time.time() - ts, 1)
        logs.append(log_path)
        if rc != 0:
            raise SystemExit(f"seed {s} step {name!r} failed (exit {rc}); see {log_path}")
        if out:
            eps = contact_on_block(json.load(open(out)))
            bad = episode_audit(eps, board, a.patients)
            if bad:
                raise SystemExit(f"seed {s}: episodes differ from the fixed design: {bad[:10]}")
            fp = {int(x["ep"]): float(x["final_pos"]) for x in eps}
            fa = {int(x["ep"]): float(x["final_ang"]) for x in eps}
            per_seed[str(s)] = {"final_pos": [fp.get(e["ep"]) for e in board["episodes"]],
                                "final_ang": [fa.get(e["ep"]) for e in board["episodes"]]}
    from . import stats as S
    for s, v in per_seed.items():
        v["per_patient"] = S.patient_means(v["final_pos"], board["episodes"])
    method = {"name": a.name, "slot": slot, "doi": a.doi, "code_url": a.code_url, "license": a.license,
              "description": a.description, "learned": bool(a.learned), "privileged": bool(a.privileged),
              "label_access": bool(a.label_access)}
    complete = not a.patients and all(None not in v["final_pos"] for v in per_seed.values())
    summ = SUB.summarize(per_seed, slot, method, board) if complete else None
    import numpy
    try:
        import torch
        tv = torch.__version__
    except Exception:                                     # pragma: no cover
        tv = None
    from usbg import __version__ as usbg_version
    from usbg import render_lotus as R
    sha = plugin_sha256(a.plugin)
    sub = {"schema": PRO.SUBMISSION_SCHEMA, "protocol": PRO.PROTOCOL_VERSION, "method": method,
           "plugin": {"entry": a.plugin, "source_sha256": sha},
           "protocol_settings": SUB.protocol_settings(seeds),
           "results": {"episodes": [{"ep": e["ep"], "patient": e["patient"], "target": e["target"]}
                                    for e in board["episodes"]], "per_seed": per_seed},
           "summary": None if summ is None else {k: summ[k] for k in ("err_mm", "diff", "decision", "label",
                                                                      "reference")},
           "provenance": {"usbg_version": usbg_version, "plugin_source_sha256": sha,
                          "python": platform.python_version(), "torch": tv, "numpy": numpy.__version__,
                          "device": a.device or R.get_device(), "platform": f"{platform.system()} {platform.machine()}",
                          "wall_time_s": round(time.time() - t0, 1), "wall_time_by_step_s": wall,
                          "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "commands": [public_command(c[3]) for c in cmds],
                          "partial": None if complete else {"patients": a.patients}}}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(sub, fh, indent=1)
    print(f"[run] wrote {a.out} ({'complete' if complete else 'PARTIAL: not a valid submission'}) in "
          f"{time.time() - t0:.0f} s; the file holds no local path, and pasting it into the issue form makes it public")
    if summ is not None:
        from .submission import summary_row
        print(json.dumps(summary_row(sub, summ), indent=1, ensure_ascii=True))
    return 0


__all__ = ["run", "slot_steps", "contact_on_block", "episode_audit", "plugin_sha256", "public_command", "public_path"]
