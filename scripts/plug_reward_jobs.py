#!/usr/bin/env python3
"""plug_reward_jobs.py: write the job spec of the reward-slot pluggability runs (v2.1 pluggability,
exploratory). Writes outputs/v2_1/jobs/plug_reward.json in the schema of the other outputs/v2_1/jobs/*.json
(read by scripts/run_table4_v2.py); refuses to overwrite it. Launches nothing.

Jobs
  landscape  the paper's held-out appearance-term learnability (6 held-out patients, 5 liver and 5 kidney
             windows each, 40 samples per view, the four v2 texture stages, DR off) with the plugged terms;
             the LPIPS variant runs in the plugin venv and waits for the AlexNet weights file.
  train      the paper's learner (servo base plus residual, REINFORCE with the BC anchor, 57,600 RL steps:
             300 iterations of 16 episodes, 12 steps each) from the paper's post-imitation checkpoint of the same
             stage and seed, CT mask, physics and SonoSPADE, seeds 0-2, one job per reward arm. Held-out
             evaluation: the 60 crossed episodes, eval seed 999, learning curve every 50 iterations.
  analysis   scripts/plug_reward_agg.py once the runs exist (one summary without and one with LPIPS).
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")
PLUG_PY = os.environ.get("USBG_PLUGIN_PYTHON", PY)
WINDOWS = os.path.join(REPO, "data", "cache", "windows", "usbg-windows-v1.json")
WINDOWS_MD5 = "af35f0bddeb57c079981eb0ba8aca5b3"
SONO = os.path.join(REPO, "data", "dataset", "v2", "sonospade_curvi.pt")
SONO_MD5 = "70440a5c6167e69489835ef553bfe37c"
TEX_MANIFEST = os.path.join(REPO, "outputs", "v2", "textures", "manifest.json")
TEX_CKPTS = [os.path.join(REPO, "data", "dataset", "v2", f"{s}_curvi.pt") for s in ("cut", "cyclegan", "sonospade")]
PAPER_LANDSCAPE = os.path.join(REPO, "outputs", "v2", "view_match", "landscape_heldout.json")
TORCH_HOME = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "torch_home")
ALEXNET = os.path.join(TORCH_HOME, "hub", "checkpoints", "alexnet-owt-7be5be79.pth")
OUT = os.path.join(REPO, "outputs", "v2_1", "plug", "reward")
TRAIN_CASES = ["s0011", "s0058", "s0344", "s0461", "s0477", "s0511", "s0522", "s0703", "s0720", "s0727",
               "s0777", "s1086", "s1373"]
EVAL_CASES = ["s0358", "s0513", "s0519", "s0591", "s0649", "s0762"]
SEEDS = (0, 1, 2)
STAGES = ("physics", "sonospade")
# arm -> (--reward, w_seg, w_app, tier). core = the methods the reward slot was asked for (and the reference)
ARMS = {
    "paper": ("paper", 0.6, 0.4, "core: reference (the paper's S)"),
    "ncc_sector": ("ncc_sector", 0.6, 0.4, "core"),
    "ssim": ("ssim", 0.6, 0.4, "core"),
    "nmi": ("nmi", 0.6, 0.4, "core"),
    "lpips": ("lpips", 0.6, 0.4, "core (needs the AlexNet weights file)"),
    "pose": ("pose", 0.6, 0.4, "core (privileged training reward)"),
    "seg_only": ("paper", 1.0, 0.0, "secondary: term ablation"),
    "ncc_only": ("paper", 0.0, 1.0, "secondary: term ablation"),
    "pose_area": ("pose_area", 0.6, 0.4, "secondary (privileged; VesNet-RL-style adaptation)"),
}
EST_H = {"physics": 0.45, "sonospade": 0.8}       # measured RL phase: 0.36-0.37 h physics (v2.1 rlpos), 0.63 h sonospade (v2 ctmask)


def base_env():
    return {"PYTHONPATH": os.path.join(REPO, "src"), "USBG_SEAT": "anatomical", "USBG_LEARNED_SEG": "0",
            "US_SEG_CKPT": "", "USBG_WINDOWS": "", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}


def lpips_env():
    return {**base_env(), "TORCH_HOME": TORCH_HOME, "USBG_PLUG_ALEXNET": ALEXNET}


def landscape_jobs():
    common = ["--cases", *EVAL_CASES, "--organs", "liver", "kidney", "--targets-per-case", "5",
              "--samples-per-target", "40", "--tex-manifest", TEX_MANIFEST, "--dr", "off", "--device", "cpu"]
    expect = {"seat": "anatomical", "seg_md5": None, "windows_md5": WINDOWS_MD5}
    o1 = os.path.join(OUT, "landscape", "heldout.json")
    o2 = os.path.join(OUT, "landscape", "heldout_lpips.json")
    return [
        {"name": "v21:plug:reward:landscape:heldout",
         "cmd": [PY, os.path.join(REPO, "scripts", "plug_reward_landscape.py"),
                 "--terms", "ncc_sector", "ncc_frame", "ssim", "nmi", "mi", "--potentials", "pose", "pose_area",
                 "--parity-ref", PAPER_LANDSCAPE, "--out", o1, "--", *common],
         "out": o1, "est_h": 0.35, "env": base_env(), "deps": [],
         "files": [TEX_MANIFEST, *TEX_CKPTS, PAPER_LANDSCAPE, WINDOWS], "expect": expect,
         "cwd": REPO, "tier": "core",
         "note": "the paper's held-out landscape protocol; ncc_lit must reproduce landscape_heldout.json "
                 "(parity_with_paper_landscape in the output)"},
        {"name": "v21:plug:reward:landscape:heldout_lpips",
         "cmd": [PLUG_PY, os.path.join(REPO, "scripts", "plug_reward_landscape.py"),
                 "--terms", "lpips", "--potentials", "--lpips-alexnet", ALEXNET, "--lpips-device", "cpu",
                 "--parity-ref", PAPER_LANDSCAPE, "--out", o2, "--", *common],
         "out": o2, "est_h": 0.3, "env": lpips_env(), "deps": [],
         "files": [TEX_MANIFEST, *TEX_CKPTS, PAPER_LANDSCAPE, WINDOWS, ALEXNET], "expect": expect,
         "cwd": REPO, "tier": "core (blocked until the AlexNet weights are placed)",
         "note": "plugin venv (lpips); the AlexNet file is not downloaded by any job"},
    ]


def train_job(arm, stage, seed):
    reward, ws, wa, tier = ARMS[arm]
    out = os.path.join(OUT, "train", arm, f"{stage}_s{seed}.json")
    imit = os.path.join(REPO, "outputs", "v2", "table4", "ctmask", f"{stage}_deploy_s{seed}_imit.pt")
    tex = ["--texture-ckpt", "none"] if stage == "physics" else ["--texture-ckpt", SONO]
    is_lp = reward == "lpips"
    pre = [PLUG_PY if is_lp else PY, os.path.join(REPO, "scripts", "plug_reward_train.py"), "--reward", reward]
    if is_lp:
        pre += ["--lpips-alexnet", ALEXNET, "--lpips-device", "cpu"]
    trainer = ["--translator", stage, *tex, "--fix-residual", "--residual-image", "--ibvs-features", "rich",
               "--w-seg", f"{ws:g}", "--w-app", f"{wa:g}", "--imitation", "0", "--iters", "300",
               "--episodes-per-iter", "16", "--n-steps", "12", "--eval-episodes", "60", "--eval-seed", "999",
               "--eval-design", "crossed", "--train-cases", *TRAIN_CASES, "--eval-cases", *EVAL_CASES,
               "--targets", "liver", "kidney", "--min-windows", "2", "--contact", "on", "--seed", str(seed),
               "--ablation", f"plug:reward:{arm}", "--init-from", imit, "--curve-every", "50", "--out", out]
    files = [imit, WINDOWS] + ([SONO] if stage == "sonospade" else []) + ([ALEXNET] if is_lp else [])
    return {"name": f"v21:plug:reward:train:{arm}:{stage}:s{seed}", "cmd": [*pre, "--", *trainer], "out": out,
            "est_h": EST_H[stage], "env": lpips_env() if is_lp else base_env(), "deps": [], "files": files,
            "expect": {"seat": "anatomical", "texture_md5": None if stage == "physics" else SONO_MD5,
                       "seg_md5": None, "windows_md5": WINDOWS_MD5},
            "cwd": REPO, "tier": tier}


def analysis_jobs(train, land):
    agg = os.path.join(REPO, "scripts", "plug_reward_agg.py")
    no_lp = [j["name"] for j in train + land if ":lpips:" not in j["name"] and not j["name"].endswith("_lpips")]
    allj = [j["name"] for j in train + land]
    lands = [j["out"] for j in land]
    o1, o2 = os.path.join(OUT, "summary.json"), os.path.join(OUT, "summary_with_lpips.json")
    env = {"PYTHONPATH": os.path.join(REPO, "src"), "USBG_SEAT": "anatomical", "USBG_LEARNED_SEG": "0",
           "US_SEG_CKPT": "", "USBG_WINDOWS": "", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    return [
        {"name": "v21:plug:reward:analysis", "cmd": [PY, agg, "--train-root", os.path.join(OUT, "train"),
                                                     "--landscape", lands[0], "--out", o1],
         "out": o1, "est_h": 0.02, "env": env, "deps": no_lp, "files": [], "expect": {}, "cwd": REPO,
         "tier": "core", "note": "summary without the LPIPS arms (they wait for the AlexNet weights)"},
        {"name": "v21:plug:reward:analysis_with_lpips", "cmd": [PY, agg, "--train-root", os.path.join(OUT, "train"),
                                                                "--landscape", *lands, "--out", o2],
         "out": o2, "est_h": 0.02, "env": env, "deps": allj, "files": [], "expect": {}, "cwd": REPO,
         "tier": "core"},
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "v2_1", "jobs", "plug_reward.json"))
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    land = landscape_jobs()
    train = [train_job(a, st, s) for a in ARMS for st in STAGES for s in SEEDS]
    jobs = land + train + analysis_jobs(train, land)
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "duplicate job names"
    outs = [j["out"] for j in jobs]
    assert len(outs) == len(set(outs)), "duplicate outputs"
    for j in jobs:
        assert os.path.exists(j["cmd"][0]), j["cmd"][0]
        assert os.path.exists(j["cmd"][1]), j["cmd"][1]
        assert not os.path.exists(j["out"]), f"output exists: {j['out']}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(jobs, fh, indent=1)
    tot = sum(j["est_h"] for j in jobs)
    missing = sorted({f for j in jobs for f in j["files"] if not os.path.exists(f)})
    print(f"wrote {args.out}: {len(jobs)} jobs, {tot:.2f} est job-h; missing input files now: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
