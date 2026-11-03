#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): the admission gate of the LPIPS reward term (usbg.plug.rewards, reward=lpips).

  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \\
  TORCH_HOME=<plugin weights>/torch_home \\
      <plugin venv>/bin/python scripts/plug/admission_gate_lpips.py \\
      --out outputs/v2_1/checks/gate/reward_lpips.json

Why a separate gate: the reward:terms gate (scripts/plug/admission_gate.py --adapter reward:terms) excluded LPIPS
because the AlexNet weights were not on disk when it ran. This gate runs the same protocol (usbg.plug.admission;
the pre-specified protocol) on reward=lpips, in the plugin venv (the lpips package), with
the potential built exactly as the LPIPS jobs build it (RewardPlug('lpips', lpips_kw={alexnet_path, device,
random_trunk=False})). It reuses admission_gate.py's helpers; that file is not changed. TRAIN patients only.

  (i)   privileged channel: AD.check_reward on the LPIPS potential (non-privileged: identical values under poisoned and
        absent poses); the pose reward as the detector-power control (must be DETECTED); PluggedRewardWrapper
        accepts lpips on a privileged_info=False environment.
  (ii)  pretraining corpus (PRETRAINING below: AlexNet trunk on ImageNet-1k, LPIPS v0.1 linear heads on BAPPS) and
        the weights checks: the trunk is the local file (sha256 tag 7be5be79; any download attempt fails the gate),
        the heads are the lpips wheel's v0.1 file, and the plug's LPIPS equals the stock lpips package built through
        torchvision's own ImageNet loader on the same local file.
  (iii) CUDA-free and CPU / MPS parity: distance maps on identical TRAIN frames (TOL frame_cpu_mps_*); one TRAIN
        episode per device through PluggedRewardWrapper with the servo base (environment and LPIPS on that device;
        TOL episode_*, applied to the plugged S); and the JOB CODE PATH: scripts/plug_reward_train.py run in-process
        with the arguments of the queue job --job-name (read from --job-source) on a smoke budget (TRAIN patients,
        1 iteration of 1 episode, output in a temporary plug_smoke folder). Every LPIPS potential that run formed is
        recomputed by the gate's certified CPU plug (must be bit-identical) and by an MPS plug (TOL
        episode_match_max_abs), and the run's own training log must report the same S.

Output: --out (exists only if admitted; else <out minus .json>.failed.json and exit 1). Never overwrites.
analysis_status "exploratory, v2.1 pluggability".
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ADAPTER = "reward:lpips"
TORCH_HOME = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "torch_home")
JOB_SOURCE = os.path.join(REPO, "outputs", "v2_1", "jobs_excluded_pending_downloads.json")
JOB_NAME = "v21:plug:reward:train:lpips:physics:s0"
REF_TOL = 1e-6          # plug vs stock lpips, same weights and device (expected bit-identical)
# Smoke budget of the job-path run (every other job argument is kept as the job passes it)
SMOKE_SET = {"--iters": ["1"], "--episodes-per-iter": ["1"], "--eval-episodes": ["2"]}
# env keys whose value must equal the LPIPS job's (they change what the trainer renders, masks or loads)
JOB_ENV_KEYS = ("USBG_SEAT", "USBG_LEARNED_SEG", "US_SEG_CKPT", "USBG_WINDOWS", "TORCH_HOME", "USBG_PLUG_ALEXNET")

# (ii) Declared pretraining corpus of the two weight files LPIPS loads. Citations checked against Crossref
# (doi:10.1109/CVPR.2018.00068, doi:10.1007/s11263-015-0816-y) and arXiv (1801.03924, 1404.5997); the head-training
# details (fixed trunk, BAPPS 2AFC train, 64x64 patches from MIT-Adobe 5k, about 151.4k triplets) from arXiv:1801.03924.
PRETRAINING = {
    "corpus": "AlexNet trunk (conv1-conv5): ImageNet-1k, the ILSVRC-2012 classification training set (about 1.28M "
              "labeled natural photographs, 1000 classes; Russakovsky et al., IJCV 2015), torchvision weights "
              "IMAGENET1K_V1 (alexnet-owt-7be5be79, the AlexNet variant of Krizhevsky 2014, the trunk LPIPS uses). "
              "LPIPS v0.1 linear heads: non-negative per-channel weights learned on the BAPPS 2AFC training set "
              "(about 151.4k triplets of 64x64 patches derived from MIT-Adobe 5k, traditional and CNN-based "
              "distortions, 2 human judgments per triplet) with the trunk kept fixed (Zhang et al., CVPR 2018)",
    "trunk": {"dataset": "ImageNet-1k (ILSVRC-2012 classification train)",
              "weights": "torchvision AlexNet_Weights.IMAGENET1K_V1 (alexnet-owt-7be5be79.pth)",
              "citation": "Krizhevsky, One weird trick for parallelizing convolutional neural networks, "
                          "arXiv:1404.5997 (2014); architecture: Krizhevsky, Sutskever, Hinton, NeurIPS 2012; data: "
                          "Russakovsky et al., ImageNet Large Scale Visual Recognition Challenge, Int. J. Comput. Vis. "
                          "115 (2015) 211-252, doi:10.1007/s11263-015-0816-y"},
    "heads": {"dataset": "BAPPS 2AFC train (traditional, CNN-based and mixed distortions of 64x64 patches derived "
                         "from MIT-Adobe 5k; 2 judgments per triplet)",
              "variant": "LPIPS 'lin' (linear calibration on a fixed pretrained trunk), version 0.1, net alex",
              "trunk_fixed_during_head_training": True,
              "citation": "Zhang, Isola, Efros, Shechtman, Wang, The Unreasonable Effectiveness of Deep Features as "
                          "a Perceptual Metric, CVPR 2018, 586-595, doi:10.1109/CVPR.2018.00068; lpips 0.1.4 "
                          "(BSD-2-Clause)"},
    "overlap": {"vitale_aul_kidneyus": "not expected: natural photographs (ImageNet) and distorted natural-image "
                                       "patches (BAPPS); neither dataset description includes ultrasound",
                "heldout_and_cohort_ct": "not expected: no CT in either corpus"},
    "status": "documented from the dataset papers (citations verified)",
    "real_frame_dice_reportable": None,
    "role": "reward term only: it scores simulated frames during RL training and in the reward landscape; no "
            "real-frame number is computed from it",
}


# ============================================================================================ helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def no_network():
    """Every download / socket attempt raises and is counted (torch hub, urllib, raw sockets)."""
    import socket
    import urllib.request
    import torch.hub
    seen = {"n": 0, "where": []}

    def deny(where):
        def f(*a, **kw):
            seen["n"] += 1
            seen["where"].append(where)
            raise RuntimeError(f"network access refused by the LPIPS gate ({where})")
        return f

    saved = [(torch.hub, "download_url_to_file"), (urllib.request, "urlopen"), (socket, "create_connection")]
    old = [getattr(o, a) for o, a in saved]
    for (o, a) in saved:
        setattr(o, a, deny(f"{o.__name__}.{a}"))
    try:
        yield seen
    finally:
        for (o, a), v in zip(saved, old):
            setattr(o, a, v)


def make_plug(A, device):
    """The LPIPS potential exactly as the LPIPS jobs build it (plug_reward_train.py / plug_reward_landscape.py)."""
    from usbg.plug import rewards as RW
    return RW.RewardPlug("lpips", ang_w=RW.ANG_W, pose_scale=RW.POSE_SCALE, area_coef=RW.AREA_COEF,
                         lpips_kw={"alexnet_path": A.alexnet, "device": device, "random_trunk": False})


def view_pairs(views):
    return [(views[i], views[i + 1]) for i in range(0, len(views) - 1, 2)]


def _set_opt(argv, flag, values):
    """Replace the value run after flag (up to the next --option) by values; append if absent."""
    out = list(argv)
    if flag not in out:
        return out + [flag] + list(values)
    i = out.index(flag)
    j = i + 1
    while j < len(out) and not out[j].startswith("--"):
        j += 1
    return out[:i + 1] + list(values) + out[j:]


def _opt(argv, flag):
    if flag not in argv:
        return None
    i = argv.index(flag)
    j = i + 1
    while j < len(argv) and not argv[j].startswith("--"):
        j += 1
    return argv[i + 1:j]


# ============================================================================================ (ii) weights
def check_weights(A, plug_cpu, views):
    """Trunk = the local file, heads = the wheel's v0.1 file, and the plug equals the stock lpips package."""
    import torch
    import lpips
    app = plug_cpu._app
    comps = []
    sd = torch.load(A.alexnet, map_location="cpu", weights_only=True)
    slices = (app.net.net.slice1, app.net.net.slice2, app.net.net.slice3, app.net.net.slice4, app.net.net.slice5)
    d_trunk, n_trunk = 0.0, 0
    for sl in slices:
        for name, p in sl.named_parameters():
            d_trunk = max(d_trunk, float((p.detach().cpu() - sd[f"features.{name}"]).abs().max()))
            n_trunk += 1
    heads_path = os.path.join(os.path.dirname(lpips.__file__), "weights", "v0.1", "alex.pth")
    hd = torch.load(heads_path, map_location="cpu", weights_only=True)
    own = {k: v for k, v in app.net.state_dict().items() if k.startswith("lin")}
    d_heads = max(float((own[k].cpu() - hd[k]).abs().max()) for k in hd) if set(hd) <= set(own) else float("inf")
    sha = sha256_file(A.alexnet)
    info = app.info()
    ok = (d_trunk == 0.0 and n_trunk == 10 and d_heads == 0.0 and sha.startswith("7be5be79")
          and info["alexnet_sha256"] == sha and not info["random_trunk_smoke_only"])
    comps.append({"component": "lpips weights: trunk = the local ImageNet AlexNet file, heads = the lpips v0.1 file",
                  "alexnet_path": os.path.abspath(A.alexnet), "alexnet_sha256": sha,
                  "alexnet_bytes": os.path.getsize(A.alexnet), "trunk_tensors": n_trunk,
                  "trunk_max_abs_diff_vs_file": d_trunk, "heads_path": heads_path, "heads_sha256": sha256_file(heads_path),
                  "heads_tensors": sorted(hd), "heads_max_abs_diff_vs_file": d_heads,
                  "random_trunk_smoke_only": info["random_trunk_smoke_only"], "passed": bool(ok)})
    # the stock lpips package: torchvision's own ImageNet loader (load_state_dict_from_url) on the pinned hub dir
    hub_dir = os.path.join(A.torch_home, "hub")
    hub_file = os.path.join(hub_dir, "checkpoints", os.path.basename(A.alexnet))
    same = os.path.isfile(hub_file) and (os.path.realpath(hub_file) == os.path.realpath(A.alexnet)
                                         or sha256_file(hub_file) == sha)
    rec = {"component": "lpips equals the stock lpips package (torchvision ImageNet loader, same local file)",
           "hub_dir": hub_dir, "hub_file": hub_file, "hub_file_is_the_alexnet_file": bool(same)}
    if not rec["hub_file_is_the_alexnet_file"]:
        comps.append(dict(rec, passed=False, verdict=f"{hub_file} is not the --alexnet file; not compared"))
        return comps
    old_dir = torch.hub.get_dir()
    torch.hub.set_dir(hub_dir)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = lpips.LPIPS(net="alex", pretrained=True, pnet_rand=False, spatial=True, verbose=False).eval()
    finally:
        torch.hub.set_dir(old_dir)
    rs, ps = ref.state_dict(), app.net.state_dict()
    d_par = (max(float((rs[k].detach() - ps[k].detach().cpu()).abs().max()) for k in rs)
             if set(rs) == set(ps) else float("inf"))
    dm, ds = 0.0, 0.0
    for cur, tgt in view_pairs(views):
        a = np.asarray(cur["img"], np.float32)
        b = np.asarray(tgt["img"], np.float32)

        def t(x):
            return torch.from_numpy(np.repeat((x * 2.0 - 1.0)[None, None], 3, axis=1))
        with torch.no_grad():
            r = ref(t(a), t(b))[0, 0].numpy()
        p = app.distance_map(a, b)
        dm = max(dm, float(np.abs(r - p).max()))
        from usbg.plug import rewards as RW
        m = RW.lit_mask(a, b)
        if int(m.sum()) >= RW.MIN_PIX:
            ds = max(ds, abs(float(np.clip(1.0 - r[m].mean(), 0, 1)) - app(a, b)))
    ok = d_par == 0.0 and dm <= REF_TOL and ds <= REF_TOL
    comps.append(dict(rec, state_dict_max_abs_diff=d_par, distance_map_max_abs_diff=dm, appearance_max_abs_diff=ds,
                      pairs=len(view_pairs(views)), tolerance=REF_TOL, passed=bool(ok)))
    return comps


# ============================================================================================ (iii) parity
def map_parity(A, plugs, views):
    """LPIPS distance maps on identical TRAIN frames, CPU vs MPS (AD.frame_parity, TOL frame_cpu_mps_*)."""
    from usbg.plug import admission as AD
    pairs = [(np.asarray(c["img"], np.float32), np.asarray(t["img"], np.float32)) for c, t in view_pairs(views)]
    r = AD.frame_parity(lambda dev: (lambda x, dev=dev: plugs[dev]._app.distance_map(*x)), pairs,
                        label="lpips distance map (identical TRAIN frames)")
    if "mps" in plugs:
        pot = [abs(plugs["cpu"].potential(c["img"], c["lab"], t["img"], t["lab"])
                   - plugs["mps"].potential(c["img"], c["lab"], t["img"], t["lab"])) for c, t in view_pairs(views)]
        r["potential_max_abs_diff"] = float(max(pot))
    return r


def _plug_episode(A, dev, plug):
    """One TRAIN episode (eval_episode 0) through PluggedRewardWrapper on dev: the servo base alone, environment and
    LPIPS on dev. Records the plugged S, the environment's own S, the step rewards and the final pose error."""
    import admission_gate as AG
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    from usbg.plug import rewards as RW
    with AD.cuda_watch() as cw:
        t0 = time.perf_counter()
        env = RW.PluggedRewardWrapper(AG._env(A, A.cases, device=dev, base="servo", servo_base=A.servo_base),
                                      reward=plug)
        ctrl = C.ZeroController("servo_base")
        obs, info = env.reset(options={"eval_episode": 0})
        ctrl.reset(0, env)
        S, M, R, steps = [float(info["plug_reward"]["S"])], [float(info["match"])], [], []
        for _ in range(env.unwrapped.n_steps):
            ts = time.perf_counter()
            obs, r, _te, _tr, info = env.step(np.asarray(ctrl.act(obs, info), np.float32).reshape(6))
            AD.sync(dev)
            steps.append((time.perf_counter() - ts) * 1e3)
            S.append(float(info["plug_reward"]["S"]))
            M.append(float(info["match"]))
            R.append(float(r))
        P = info.get("privileged") or {}
        wall = time.perf_counter() - t0
        env.close()
    return {"device": dev, "episode": 0, "final_pos_mm": float(P.get("pose_error_mm", np.nan)),
            "final_ang_deg": float(P.get("rotation_error_deg", np.nan)), "S_plug": [round(s, 6) for s in S],
            "S_env": [round(s, 6) for s in M], "rewards": [round(x, 6) for x in R], "wall_s": round(wall, 2),
            "step_ms": {"median": round(float(np.median(steps)), 2), "max": round(float(np.max(steps)), 2)},
            "peak_rss_mb_cumulative": AD.peak_rss_mb(), "cuda_moves": int(cw["n"])}


def episode_parity(A, plugs):
    from usbg.plug import admission as AD
    out = {"component": "lpips reward episode (servo base, PluggedRewardWrapper; environment and LPIPS on the device)"}
    eps = {d: _plug_episode(A, d, plugs[d]) for d in plugs}
    out.update(episodes=eps, cuda_moves=int(sum(e["cuda_moves"] for e in eps.values())))
    moved = max(abs(s - eps["cpu"]["S_plug"][0]) for s in eps["cpu"]["S_plug"])
    out["plug_S_range_cpu"] = round(moved, 6)            # > 0: the episode exercises LPIPS on changing frames
    if len(eps) < 2:
        return dict(out, cpu_only=True, passed=out["cuda_moves"] == 0 and moved > 0)
    a, b = eps["cpu"], eps["mps"]
    dp = abs(a["final_pos_mm"] - b["final_pos_mm"])
    ds = AD._maxdiff(a["S_plug"], b["S_plug"])
    de = AD._maxdiff(a["S_env"], b["S_env"])
    dr = AD._maxdiff(a["rewards"], b["rewards"])
    ok = (dp <= AD.TOL["episode_final_pos_mm"] and ds <= AD.TOL["episode_match_max_abs"]
          and de <= AD.TOL["episode_match_max_abs"] and out["cuda_moves"] == 0 and moved > 0)
    return dict(out, final_pos_diff_mm=dp, plug_S_max_abs_diff=ds, env_S_max_abs_diff=de, reward_max_abs_diff=dr,
                tolerance={"final_pos_mm": AD.TOL["episode_final_pos_mm"], "match": AD.TOL["episode_match_max_abs"]},
                passed=bool(ok))


def job_path(A, plugs):
    with tempfile.TemporaryDirectory(prefix="usbg_gate_lpips_") as td:
        return _job_path(A, plugs, td)


def _job_path(A, plugs, td):
    """scripts/plug_reward_train.py in-process with the queue job's own arguments on a smoke budget; every LPIPS
    potential it forms is recorded and recomputed by the certified CPU plug (bit-identical) and the MPS plug."""
    import plug_reward_train as PRT
    from usbg.plug import admission as AD
    from usbg.plug import rewards as RW
    from usbg.benchmark import TRAIN_PATIENTS
    src = json.load(open(A.job_source))
    rows = [r for r in src["jobs"] if r["name"] == A.job_name]
    out = {"component": f"lpips job code path ({os.path.basename(PRT.__file__)} as {A.job_name})",
           "job_source": os.path.abspath(A.job_source), "job_name": A.job_name,
           "job_source_md5": hashlib.md5(open(A.job_source, "rb").read()).hexdigest()}
    if len(rows) != 1:
        return dict(out, passed=False, verdict=f"{A.job_name} not found once in {A.job_source}")
    job = rows[0]["job"]
    cmd = list(job["cmd"])
    if os.path.basename(cmd[1]) != "plug_reward_train.py" or "--" not in cmd:
        return dict(out, passed=False, verdict=f"{A.job_name} is not a plug_reward_train.py job")
    k = cmd.index("--")
    wrap, tr = cmd[2:k], cmd[k + 1:]
    if _opt(wrap, "--reward") != ["lpips"] or _opt(wrap, "--lpips-alexnet") is None:
        return dict(out, passed=False, verdict="the job is not an LPIPS reward job with a local AlexNet file")
    # env: the keys that change what the trainer renders, masks or loads must match the job's
    jenv = job.get("env") or {}
    env_diff = {}
    for key in JOB_ENV_KEYS:
        if key in jenv:
            want = jenv[key] or None
            got = os.environ.get(key) or None
            if want != got:
                env_diff[key] = {"job": want, "gate": got}
    smoke_out = os.path.join(td, "plug_smoke", "lpips", os.path.basename(_opt(tr, "--out")[0]))
    train = [c for c in A.cases if c in TRAIN_PATIENTS and c in (_opt(tr, "--train-cases") or [])]
    if not train:
        return dict(out, passed=False, verdict=f"none of --cases {A.cases} is a TRAIN case of the job")
    tr2 = _set_opt(tr, "--train-cases", train)
    tr2 = _set_opt(tr2, "--eval-cases", train)                 # TRAIN patients only (gate rule)
    for f, v in SMOKE_SET.items():
        tr2 = _set_opt(tr2, f, v)
    tr2 = _set_opt(tr2, "--out", [smoke_out])
    wrap2 = list(wrap) + ["--smoke", "--log-every", "1"]
    argv = wrap2 + ["--"] + tr2
    changed = {f: {"job": _opt(tr, f), "gate": _opt(tr2, f)} for f in
               ["--train-cases", "--eval-cases", "--out"] + list(SMOKE_SET) if _opt(tr, f) != _opt(tr2, f)}
    rec, built = [], []

    class Recording(RW.RewardPlug):                      # the job's own plug, recording what it scores
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append({"name": self.name, "lpips_kw": copy.deepcopy(kw.get("lpips_kw")),
                          "info": self._app.info() if self.app_name == "lpips" else None})

        def potential(self, cur_img, cur_lab, tgt_img, tgt_lab, **kw):
            s = super().potential(cur_img, cur_lab, tgt_img, tgt_lab, **kw)
            rec.append({"args": tuple(np.array(x, copy=True) for x in (cur_img, cur_lab, tgt_img, tgt_lab)),
                        "kw": copy.deepcopy(kw),
                        "S": float(s)})
            return s

    t0 = time.time()
    err = None
    try:
        with RW.patched(RW, "RewardPlug", Recording):
            rc = PRT.main(argv)
    except BaseException as e:                           # noqa: BLE001  (SystemExit included: recorded, gate fails)
        rc, err = None, f"{type(e).__name__}: {e}"
    wall = round(time.time() - t0, 1)
    res = {"argv": argv, "changed_from_job": changed, "env_vs_job": env_diff or "identical", "wall_s": wall,
           "potentials_recorded": len(rec), "plugs_built": built}
    if err is not None or rc not in (0, None):
        return dict(out, **res, passed=False, verdict=f"the job script failed: {err or rc}")
    d = json.load(open(smoke_out))
    plog = d.get("plug") or {}
    blk = (plog.get("train_log") or [{}])[0]
    # recompute every recorded potential with the certified plugs
    diff = {}
    for dev, p in plugs.items():
        diff[dev] = max(abs(p.potential(*r["args"], **r["kw"]) - r["S"]) for r in rec) if rec else None
    s_traj = [round(r["S"], 6) for r in rec]
    log_ok = (bool(rec) and abs(blk.get("plug_S_start", np.nan) - rec[0]["S"]) <= 1e-9
              and abs(blk.get("plug_S_end", np.nan) - rec[-1]["S"]) <= 1e-9)
    kw0 = (built[0]["lpips_kw"] or {}) if built else {}
    built_ok = (len(built) == 1 and built[0]["name"] == "lpips" and kw0.get("random_trunk") is False
                and os.path.realpath(kw0.get("alexnet_path") or "") == os.path.realpath(A.alexnet)
                and kw0.get("device") == (_opt(wrap, "--lpips-device") or ["cpu"])[0])
    info = plog.get("lpips") or {}
    meta_ok = (plog.get("reward") == "lpips" and str(info.get("alexnet_sha256", "")).startswith("7be5be79")
               and info.get("random_trunk_smoke_only") is False
               and d.get("analysis_status") == RW.ANALYSIS_STATUS)
    tol = AD.TOL["episode_match_max_abs"]
    ok = (built_ok and meta_ok and log_ok and not env_diff and len(rec) == int(_opt(tr2, "--n-steps")[0]) + 1
          and diff.get("cpu") == 0.0 and (diff.get("mps") is None or diff["mps"] <= tol))
    from usbg import render_lotus as R
    render_dev = (_opt(tr, "--device") or [None])[0] or R.get_device()      # = the trainer's args.device or R.get_device()
    return dict(out, **res, job_lpips_device=kw0.get("device"), job_render_device=render_dev,
                S_plug_first_rollout=s_traj, train_log_first_block=blk, lpips_info_in_output=info,
                recomputed_max_abs_diff={"cpu_certified_plug": diff.get("cpu"), "mps_plug": diff.get("mps")},
                tolerance={"cpu_certified_plug": 0.0, "mps_plug": tol},
                checks={"one_lpips_plug_built_with_the_job_kw": built_ok, "output_records_lpips_and_status": meta_ok,
                        "train_log_matches_recorded_S": log_ok, "env_matches_job": not env_diff},
                smoke_eval={k: {m: (d.get(k) or {}).get(m) for m in ("start_pos", "pos", "ang", "match")}
                            for k in ("after_imitation", "final")}, passed=bool(ok))


# ============================================================================================ gate
def gate_lpips(A):
    import admission_gate as AG
    from usbg.plug import admission as AD
    from usbg.plug import rewards as RW
    views = AD.train_views(A.case, n=A.n_views)
    plugs = {"cpu": make_plug(A, "cpu")}
    if AD.mps_ok():
        plugs["mps"] = make_plug(A, "mps")
    comps = [AD.check_reward(plugs["cpu"], views, label="lpips (AlexNet ImageNet trunk, LPIPS v0.1 heads)"),
             AD.check_reward(RW.RewardPlug("pose"), views, label="pose (privileged; detector power)")]
    acc = {"component": "PluggedRewardWrapper accepts lpips without privileged_info"}
    e = AG._env(A, A.cases, privileged_info=False)
    try:
        RW.PluggedRewardWrapper(e, reward=plugs["cpu"])
        acc.update(passed=True, verdict="accepted on a privileged_info=False environment (non-privileged)")
    except ValueError as ex:
        acc.update(passed=False, verdict=f"refused: {ex}")
    e.close()
    comps.append(acc)
    weights = check_weights(A, plugs["cpu"], views)
    par = [map_parity(A, plugs, views), episode_parity(A, plugs)]
    par.append(job_path(A, plugs))
    return comps, par, weights, {"lpips": plugs["cpu"]._app.info(), "versions": RW.package_versions(),
                                 "citation": RW.CITATIONS["lpips"]}


def main(argv=None):
    from usbg.benchmark import TRAIN_PATIENTS
    from usbg.plug import rewards as RW
    import admission_gate as AG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="s0011", help="TRAIN patient of the frame checks")
    ap.add_argument("--cases", nargs="+", default=["s0011", "s0344"], help="TRAIN patients of the episode checks")
    ap.add_argument("--n-views", type=int, default=6)
    ap.add_argument("--alexnet", default=os.environ.get("USBG_PLUG_ALEXNET") or RW.DEFAULT_ALEXNET)
    ap.add_argument("--torch-home", default=TORCH_HOME, help="hub dir root of the stock-lpips reference")
    ap.add_argument("--servo-base", default=AG.DEFAULTS["servo_base"])
    ap.add_argument("--job-source", default=JOB_SOURCE, help="JSON with the LPIPS job (round-1 exclusions file)")
    ap.add_argument("--job-name", default=JOB_NAME)
    A = ap.parse_args(argv)
    if A.case not in TRAIN_PATIENTS or not set(A.cases) <= set(TRAIN_PATIENTS):
        ap.error("the admission gate runs on TRAIN patients only")
    if not os.path.isfile(A.alexnet):
        ap.error(f"--alexnet {A.alexnet} missing (never downloaded here)")
    os.environ.setdefault("USBG_SEAT", "anatomical")
    A.out = os.path.abspath(A.out)
    if os.path.exists(A.out):
        raise SystemExit(f"refusing to overwrite {A.out}")
    failed = (A.out[:-5] if A.out.endswith(".json") else A.out) + ".failed.json"
    if os.path.exists(failed):                           # an earlier failed attempt: keep it, renamed
        os.replace(failed, failed[:-5] + f".{int(time.time())}.json")
    from usbg.plug import admission as AD
    t0 = time.time()
    with no_network() as net, AD.cuda_watch() as cw:
        comps, par, weights, extra = gate_lpips(A)
    weights.insert(0, {"component": "no download or network access during the gate", "attempts": int(net["n"]),
                       "where": net["where"][:5], "passed": net["n"] == 0})
    extra = dict(extra or {}, cuda_moves_total=int(cw["n"]), wall_s=round(time.time() - t0, 1),
                 peak_rss_mb=AD.peak_rss_mb(), argv=list(sys.argv), cases=list(A.cases), frame_case=A.case,
                 covers="the LPIPS reward jobs (reward_terms.json excluded LPIPS because the AlexNet weights were not "
                        "on disk when that gate ran)")
    body = AD.summarize(ADAPTER, "reward", comps, par, extra)
    body["checks"]["pretraining_corpus"] = dict(
        PRETRAINING, key="lpips",
        files={"trunk": weights[1].get("alexnet_path"), "trunk_sha256": weights[1].get("alexnet_sha256"),
               "heads": weights[1].get("heads_path"), "heads_sha256": weights[1].get("heads_sha256")})
    w_ok = all(c.get("passed") for c in weights)
    body["checks"]["lpips_weights"] = {"passed": w_ok, "components": weights}
    body["admitted"] = bool(body["admitted"] and w_ok)
    if cw["n"]:
        body["admitted"] = False
        body["checks"]["cuda_free"]["passed"] = False
    path = AD.write_result(A.out, body)
    bad = [c.get("component") for c in comps + par + weights if not c.get("passed")]
    print(f"[gate] {ADAPTER}: {'ADMITTED' if body['admitted'] else 'NOT ADMITTED'} in {extra['wall_s']:.0f}s "
          f"(failed: {bad or 'none'}; cuda moves {cw['n']}; network attempts {net['n']}) -> {path}")
    return 0 if body["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
