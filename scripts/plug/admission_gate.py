#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): the admission gate of ONE adapter (usbg.plug.admission; protocol in
the pre-specified protocol). Minutes per adapter, TRAIN patients only.

  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 CUDA_VISIBLE_DEVICES= PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \\
      .venv/bin/python scripts/plug/admission_gate.py --adapter ctrl:sb3 --out outputs/v2_1/checks/gate/ctrl_sb3.json

Adapters (--adapter)
  ctrl:sb3         SB3Controller (PPO, SAC, TD3, A2C residual; PPO and DQN from scratch) on random-init models, the
                   SB3 training environment (privileged_info=False), detector-power leaky probes
  ctrl:goalnet     GoalNetController (BC / DAgger nets, random init, BatchNorm train and eval) and the paper's v2
                   policy; the privileged pose-oracle teacher of BC and DAgger must be detected
  ctrl:reference   static, servo base, random, the pose-oracle base condition, CEM look-ahead (privileged, detected)
  mask:tv          TorchvisionMaskSource, 4 architectures (random init; the code path, not the trained weights)
  mask:nnunet      NnUNetMaskSource on a trained model (--nnunet-model; plugin venv)
  mask:sam         SamBoxMaskSource (SAM ViT-B): U-Net-prompted (non-privileged) and CT-prompted (privileged, detected)
  mask:medsam      the same with MedSAM ViT-B
  reward:terms     RewardPlug paper / ncc_sector / ncc_frame / ssim / nmi / mi (non-privileged), pose and pose_area
                   (privileged, detected), PluggedRewardWrapper guard
  texture:<stage>  fastcut, dclgan, unsb (random-init generators) or histmatch (fitted on TRAIN renders): label-free
                   check, CPU / MPS frame parity, one episode per device; SonoSPADE (wants_label) must be detected
  renderer:<kind>  lotus_reimpl, lotus_orig, simus: determinism, CPU / MPS parity (SIMUS: CPU only), one episode

Output: --out (exists only if the adapter is admitted; else <out minus .json>.failed.json and exit 1, so a failed
gate BLOCKS its dependents and a runner restart never SKIPs it). Never overwrites an admitted result.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEG_REPO = os.environ.get("USBG_SEG_REPO", os.path.join(REPO, "segtrain"))
WEIGHTS = os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights"))
DEFAULTS = {
    "servo_base": os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s0.pt"),
    "v2_policy": os.path.join(REPO, "outputs", "v2", "table4", "ctmask", "physics_deploy_s0.pt"),
    "sonospade": os.path.join(REPO, "data", "dataset", "v2", "sonospade_curvi.pt"),
    "prompt_ckpt": os.path.join(SEG_REPO, "outputs", "v2", "us_seg_sonospade_s0.pt"),
    "sam": os.path.join(WEIGHTS, "sam", "sam_vit_b_01ec64.pth"),
    "medsam": os.path.join(WEIGHTS, "medsam", "medsam_vit_b.pth"),
}
ADAPTERS = ("ctrl:sb3", "ctrl:goalnet", "ctrl:reference", "mask:tv", "mask:nnunet", "mask:sam", "mask:medsam",
            "reward:terms", "texture:fastcut", "texture:dclgan", "texture:unsb", "texture:histmatch",
            "renderer:lotus_reimpl", "renderer:lotus_orig", "renderer:simus")


def _env(A, patients, device=None, **kw):
    from usbg.plug import controllers as C
    base = dict(patients=list(patients), texture="physics", mask="ct", base=None, render_seed=0, contact=True,
                device=device, frame_dtype="float32", privileged_info=True)
    base.update(kw)
    return C.make_env(**base)


def _obs_samples(env, n_eps=1):
    obs = []
    for ep in range(n_eps):
        o, _ = env.reset(options={"eval_episode": ep})
        obs.append(copy.deepcopy(o))
        for _ in range(env.unwrapped.n_steps):
            o, *_ = env.step(np.zeros(6, np.float32))
            obs.append(copy.deepcopy(o))
    return obs


# ============================================================================================ controllers
def gate_ctrl_sb3(A):
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    pats = A.cases
    comps, par = [], []
    specs = [("ppo_res", PPO, "servo"), ("sac_res", SAC, "servo"), ("td3_res", TD3, "servo"),
             ("a2c_res", A2C, "servo"), ("ppo_scratch", PPO, None), ("dqn_scratch", DQN, None)]

    def wrapped(base, discrete, priv, dtype="uint8", device=None):
        e = _env(A, pats, device=device, base=base, servo_base=A.servo_base if base == "servo" else None,
                 privileged_info=priv, frame_dtype=dtype)
        if discrete:
            e = C.DiscreteTwist(e, step_frac=1.0)
        return C.FiniteHorizonTerminal(C.StackedGoalObs(e))

    def build(cls, base, discrete, device):
        venv = DummyVecEnv([lambda: wrapped(base, discrete, False)])
        kw = dict(policy="CnnPolicy", env=venv, gamma=0.9, seed=0, device=device, verbose=0)
        if cls in (SAC, TD3, DQN):
            kw.update(buffer_size=16, learning_starts=0, optimize_memory_usage=False,
                      replay_buffer_kwargs={"handle_timeout_termination": False})
        return cls(**kw), venv

    models = {}
    for name, cls, base in specs:
        discrete = cls is DQN
        m, venv = build(cls, base, discrete, "cpu")
        models[name] = (m, cls, base, discrete)
        table = C.discrete_table(1.0) if discrete else None
        env = _env(A, pats, base=base, servo_base=A.servo_base if base == "servo" else None)
        comps.append(AD.check_controller(env, lambda m=m, t=table, n=name: C.SB3Controller(m, n, t), label=name,
                                         declared_privileged=False, episodes=A.episodes))
        tr = AD.check_training_env(lambda b=base, d=discrete: wrapped(b, d, False))
        tr["component"] = f"{name}: training environment"
        comps.append(tr)
        env.close()
        venv.close()
    env = _env(A, pats, base="servo", servo_base=A.servo_base)
    for probe in (AD.LeakyInfoProbe, AD.LeakyEnvProbe):
        comps.append(AD.check_controller(env, probe, label=probe.name, declared_privileged=True, require_detect=True,
                                         episodes=A.episodes[:1]))
    env.close()
    # (iii) identical observations on CPU and MPS (same weights)
    obs_env = _env(A, pats, base="servo", servo_base=A.servo_base)
    obs = _obs_samples(obs_env, 1)
    obs_env.close()
    for name, (m, cls, base, discrete) in models.items():
        sd = copy.deepcopy(m.policy.state_dict())
        table = C.discrete_table(1.0) if discrete else None

        def make(dev, cls=cls, base=base, discrete=discrete, sd=sd, table=table, name=name):
            mm, _ = build(cls, base, discrete, dev)
            mm.policy.load_state_dict({k: v.to(mm.device) for k, v in sd.items()})
            return C.SB3Controller(mm, name, table)
        par.append(AD.action_parity(make, obs, label=name, discrete=discrete))
    m0 = models["ppo_res"][0]
    sd0 = copy.deepcopy(m0.policy.state_dict())

    def make_ppo(dev):
        mm, _ = build(PPO, "servo", False, dev)
        mm.policy.load_state_dict({k: v.to(mm.device) for k, v in sd0.items()})
        return C.SB3Controller(mm, "ppo_res", None)
    par.append(AD.episode_parity(lambda dev: _env(A, pats, device=dev, base="servo", servo_base=A.servo_base),
                                 make_ppo, label="ppo_res episode"))
    return comps, par, {"conditions": _conditions()}


class _Teacher:
    """The BC / DAgger teacher as a controller (privileged by design)."""
    privileged = True
    name = "pose_oracle_teacher"

    def reset(self, ep, env):
        self.env = env

    def act(self, obs, info):
        from usbg.plug import controllers as C
        return C.teacher_action(self.env)


def gate_ctrl_goalnet(A):
    import torch
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    from usbg.policy import GoalPolicy
    pats = A.cases
    torch.manual_seed(0)
    rnd = GoalPolicy.build(history=4).state_dict()
    ck = torch.load(A.v2_policy, map_location="cpu", weights_only=False)
    v2 = ck["state_dict"]
    nets = {"goalnet_random_init": rnd, "v2_policy": v2}
    comps, par = [], []
    env = _env(A, pats, base="servo", servo_base=A.servo_base)
    for nm, sd in nets.items():
        for mode in ("train", "eval"):
            def fac(sd=sd, mode=mode, nm=nm):
                n = GoalPolicy.build(history=4)
                n.load_state_dict(copy.deepcopy(sd))
                return C.GoalNetController(n, "cpu", f"{nm}_bn{mode}", bn_mode=mode)
            comps.append(AD.check_controller(env, fac, label=f"{nm} (BatchNorm {mode})", declared_privileged=False,
                                             episodes=A.episodes))
    comps.append(AD.check_controller(env, _Teacher, label="BC/DAgger teacher (pose oracle)", declared_privileged=True,
                                     require_detect=True, episodes=A.episodes[:1]))
    env.close()
    obs_env = _env(A, pats, base="servo", servo_base=A.servo_base)
    obs = _obs_samples(obs_env, 1)
    obs_env.close()
    for nm, sd in nets.items():
        for mode in ("train", "eval"):
            def make(dev, sd=sd, mode=mode, nm=nm):
                n = GoalPolicy.build(history=4)
                n.load_state_dict(copy.deepcopy(sd))
                return C.GoalNetController(n.to(dev), dev, nm, bn_mode=mode)
            par.append(AD.action_parity(make, obs, label=f"{nm} (BatchNorm {mode})"))

    def make_v2(dev):
        n = GoalPolicy.build(history=4)
        n.load_state_dict(copy.deepcopy(v2))
        return C.GoalNetController(n.to(dev), dev, "v2_policy", bn_mode="eval")
    par.append(AD.episode_parity(lambda dev: _env(A, pats, device=dev, base="servo", servo_base=A.servo_base),
                                 make_v2, label="v2_policy episode (BatchNorm eval)"))
    return comps, par, {"conditions": _conditions(),
                        "note": "BC and DAgger TRAIN from the privileged teacher (detected above); the learned "
                                "GoalPolicy controllers they produce are pose-free at run time"}


def gate_ctrl_reference(A):
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    pats = A.cases
    comps, par = [], []
    for nm, base, fac in (("static", None, lambda: C.ZeroController("static")),
                          ("servo_base", "servo", lambda: C.ZeroController("servo_base")),
                          ("random", None, lambda: C.RandomController(0)),
                          ("pose_oracle (the analytic base, residual zero)", "pose_oracle",
                           lambda: C.ZeroController("pose_oracle"))):
        env = _env(A, pats, base=base, servo_base=A.servo_base if base == "servo" else None)
        r = AD.check_controller(env, fac, label=nm, declared_privileged=False, episodes=A.episodes)
        if base == "pose_oracle":
            r["condition_privileged"] = "the environment's base is the analytic pose oracle (privileged condition)"
        comps.append(r)
        env.close()
    env = _env(A, pats, base=None)
    comps.append(AD.check_controller(env, lambda: C.CEMController(0, k=8, rounds=2, n_elite=4),
                                     label="cem look-ahead (k 8 for the gate)", declared_privileged=True,
                                     require_detect=True, episodes=A.episodes[:1]))
    env.close()
    par.append(AD.episode_parity(lambda dev: _env(A, pats, device=dev, base="servo", servo_base=A.servo_base),
                                 lambda dev: C.ZeroController("servo_base"), label="servo_base episode"))
    par.append(AD.episode_parity(lambda dev: _env(A, pats, device=dev, base=None),
                                 lambda dev: C.RandomController(0), label="random episode"))
    return comps, par, {"conditions": _conditions()}


def _conditions():
    return {"mask=ct": "privileged mask source (CT label slices; no rig provides them)",
            "mask=learned": "non-privileged mask source (texture-matched segmenter)",
            "base=pose_oracle": "privileged condition (analytic twist from the true pose)",
            "base=servo": "non-privileged base (pose-free at run time; calibrated on privileged analytic twists)",
            "reward=pose, pose_area": "privileged reward (training signal only; the policy observes images)"}


# ============================================================================================ masks
def _mask_episode(A, src_by_dev, label):
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    return AD.episode_parity(lambda dev: _env(A, A.cases, device=dev, mask=src_by_dev(dev)),
                             lambda dev: C.ZeroController("static"), label=label)


def gate_mask_tv(A):
    from usbg.plug import admission as AD
    from usbg.plug import masks as PM
    views = AD.train_views(A.case, n=A.n_views)
    comps, par = [], []
    with tempfile.TemporaryDirectory(prefix="usbg_gate_tv_") as td:
        for arch in PM.TV_ARCHS:
            p = os.path.join(td, f"{arch}.pt")
            PM.TVSegmenter(device="cpu", arch=arch, seed=0).save(p)
            src = PM.TorchvisionMaskSource(p, device="cpu")
            comps.append(AD.check_mask_source(src, views, label=f"tv:{arch} (random init)"))
            par.append(AD.mask_parity(lambda dev, p=p: PM.TorchvisionMaskSource(p, device=dev), views,
                                      label=f"tv:{arch}"))
            par.append(_mask_episode(A, lambda dev, p=p: PM.TorchvisionMaskSource(p, device=dev),
                                     f"tv:{arch} episode"))
    return comps, par, {"weights": "random initialization (seed 0); the gate certifies the adapter code path"}


def gate_mask_nnunet(A):
    from usbg.plug import admission as AD
    from usbg.plug import masks as PM
    if not A.nnunet_model or not os.path.isdir(A.nnunet_model):
        raise SystemExit(f"--nnunet-model {A.nnunet_model!r} is not a trained nnU-Net model folder")
    views = AD.train_views(A.case, n=A.n_views)
    mk = lambda dev: PM.NnUNetMaskSource(A.nnunet_model, fold=0, device=dev)       # noqa: E731
    comps = [AD.check_mask_source(mk("cpu"), views, label="nnunet2d")]
    par = [AD.mask_parity(mk, views, label="nnunet2d"), _mask_episode(A, mk, "nnunet2d episode")]
    return comps, par, {"weights": {"model": A.nnunet_model,
                                    "md5": PM._md5(os.path.join(A.nnunet_model, "fold_0", "checkpoint_final.pth"))}}


def gate_mask_sam(A, variant):
    from usbg.plug import admission as AD
    from usbg.plug import masks as PM
    w = A.sam_weights if variant == "sam" else A.medsam_weights
    views = AD.train_views(A.case, n=A.n_views)
    mk_seg = lambda dev: PM.load_mask_source(f"kind={variant},weights={w},prompt=seg,prompt_ckpt={A.prompt_ckpt}",  # noqa: E731
                                             device=dev)
    comps = [AD.check_mask_source(mk_seg("cpu"), views, label=f"{variant}_unet (U-Net box prompts)")]
    comps.append(AD.check_mask_source(PM.load_mask_source(f"kind={variant},weights={w},prompt=ct", device="cpu"),
                                      views, label=f"{variant}_priv (CT-label box prompts)"))
    par = [AD.mask_parity(mk_seg, views, label=f"{variant}_unet"), _mask_episode(A, mk_seg, f"{variant}_unet episode")]
    return comps, par, {"weights": {"path": w, "manifest": PM._manifest_entry(w)},
                        "note": "the CT-prompted variant shares the network; its parity is covered by the U-Net-"
                                "prompted run"}


# ============================================================================================ rewards
def gate_reward(A):
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    from usbg.plug import rewards as RW
    views = AD.train_views(A.case, n=A.n_views)
    comps = []
    for name in ("paper", "ncc_sector", "ncc_frame", "ssim", "nmi", "mi", "pose", "pose_area"):
        comps.append(AD.check_reward(RW.RewardPlug(name), views, label=name))
    guard = {"component": "PluggedRewardWrapper privilege guard"}
    e = _env(A, A.cases, privileged_info=False)
    try:
        RW.PluggedRewardWrapper(e, reward="pose")
        guard.update(passed=False, verdict="FAIL: a privileged reward was accepted on a privileged_info=False env")
    except ValueError as ex:
        guard.update(passed=True, verdict=f"refuses a privileged reward without privileged_info ({ex})")
    try:
        RW.PluggedRewardWrapper(e, reward="ssim")
        guard["deployable_reward_on_unprivileged_env"] = "accepted"
    except ValueError as ex:
        guard.update(passed=False, deployable_reward_on_unprivileged_env=f"refused: {ex}")
    e.close()
    comps.append(guard)
    par = [AD.episode_parity(lambda dev: RW.PluggedRewardWrapper(_env(A, A.cases, device=dev), reward="ssim"),
                             lambda dev: C.ZeroController("static"), label="ssim reward episode (static)")]
    return comps, par, {"excluded": {"lpips": "AlexNet weights not downloaded; LPIPS jobs are not launched"},
                        "device_parity_note": "the terms are NumPy on the CPU; the episode checks the rendered "
                                              "frames they read"}


# ============================================================================================ textures
def _texture_ckpt(stage, td, views):
    import torch
    from usbg import plug_texture as PT
    p = os.path.join(td, f"{stage}.pt")
    torch.manual_seed(0)
    if stage == "fastcut":
        PT.FastCUT("cpu").save(p)
    elif stage == "dclgan":
        PT.DCLGAN("cpu").save(p)
    elif stage == "unsb":
        PT.UNSB("cpu").save(p)
    elif stage == "histmatch":
        sim = np.stack([v["img"] for v in views])
        lut, stats = PT.fit_histmatch(sim, np.clip(sim * 0.8 + 0.1, 0, 1))
        PT.save_histmatch(p, lut, stats)
    else:
        raise SystemExit(f"unknown texture stage {stage}")
    return p


def gate_texture(A, stage):
    from usbg import texture_gan as TG
    from usbg import plug_texture as PT                    # noqa: F401  (registers the unsb / histmatch loaders)
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    views = AD.train_views(A.case, n=A.n_views)
    comps, par = [], []
    with tempfile.TemporaryDirectory(prefix="usbg_gate_tex_") as td:
        p = _texture_ckpt(stage, td, views)
        tr = TG.load_translator(p, "cpu")
        comps.append(AD.check_texture(tr, views, label=f"{stage} ({'fitted' if stage == 'histmatch' else 'random init'})"))
        if os.path.exists(A.sonospade):
            comps.append(AD.check_texture(TG.load_translator(A.sonospade, "cpu"), views[:2],
                                          label="sonospade (wants_label; detector power)",
                                          declared_label_conditioned=True))
        inputs = [np.asarray(v["img"], np.float32) for v in views]
        par.append(AD.frame_parity(lambda dev: TG.load_translator(p, dev), inputs, label=stage))
        par.append(AD.episode_parity(lambda dev: _env(A, A.cases, device=dev, texture=TG.load_translator(p, dev)),
                                     lambda dev: C.ZeroController("static"), label=f"{stage} episode (CT mask)"))
    return comps, par, {"weights": "random initialization (seed 0) or a LUT fitted on TRAIN renders; the gate "
                                   "certifies the adapter code path"}


# ============================================================================================ renderers
def gate_renderer(A, kind):
    from usbg import render_lotus as R
    from usbg import slicer as S
    from usbg.plug import admission as AD
    from usbg.plug import controllers as C
    from usbg.plug import renderers as PR
    views = AD.train_views(A.case, n=min(A.n_views, 2))
    labs = [S.slice_at_pose(v["lv"], np.asarray(v["T"], np.float64), v["Wfv"]) for v in views]
    PR.register_all()
    cfg = R.RenderConfig().with_seed(0)
    comps = [{"component": kind, "passed": True,
              "verdict": "n/a: the renderer is the simulator; it reads the label slice by definition and never the "
                         "target pose or the controller's state"}]
    det = []
    for lab in labs:
        a = R.render(lab, kind, device="cpu", cfg=cfg)
        b = R.render(lab, kind, device="cpu", cfg=cfg)
        det.append(float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))))
    comps.append({"component": f"{kind} determinism (same seed twice, CPU)", "max_abs_diff": max(det),
                  "passed": max(det) == 0.0})
    par = []
    if kind == "simus":
        par.append({"component": "simus", "cpu_only": True, "passed": True,
                    "note": "PyMUST runs on the CPU only (no MPS path); CUDA is never used"})
    else:
        par.append(AD.frame_parity(lambda dev: (lambda lab, dev=dev: R.render(lab, kind, device=dev, cfg=cfg)), labs,
                                   label=kind, device_rng=(kind == "lotus_reimpl")))
    PR.install(kind)
    try:
        devs = ("cpu",) if kind == "simus" else ("cpu", "mps")
        par.append(AD.episode_parity(lambda dev: _env(A, A.cases, device=dev), lambda dev: C.ZeroController("static"),
                                     label=f"{kind} episode (CT mask, static)", devices=devs))
    finally:
        PR.uninstall()
    return comps, par, {"renderer": PR.plug_info(kind)}


# ============================================================================================ main
def main(argv=None):
    from usbg.benchmark import TRAIN_PATIENTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, choices=ADAPTERS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="s0011", help="TRAIN patient of the frame checks")
    ap.add_argument("--cases", nargs="+", default=["s0011", "s0344"], help="TRAIN patients of the episode checks")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--servo-base", default=DEFAULTS["servo_base"])
    ap.add_argument("--v2-policy", default=DEFAULTS["v2_policy"])
    ap.add_argument("--sonospade", default=DEFAULTS["sonospade"])
    ap.add_argument("--prompt-ckpt", default=DEFAULTS["prompt_ckpt"])
    ap.add_argument("--sam-weights", default=DEFAULTS["sam"])
    ap.add_argument("--medsam-weights", default=DEFAULTS["medsam"])
    ap.add_argument("--nnunet-model", default=None)
    A = ap.parse_args(argv)
    if A.case not in TRAIN_PATIENTS or not set(A.cases) <= set(TRAIN_PATIENTS):
        ap.error("the admission gate runs on TRAIN patients only")
    os.environ.setdefault("USBG_SEAT", "anatomical")
    A.out = os.path.abspath(A.out)
    if os.path.exists(A.out):
        raise SystemExit(f"refusing to overwrite {A.out}")
    failed = (A.out[:-5] if A.out.endswith(".json") else A.out) + ".failed.json"
    if os.path.exists(failed):                           # an earlier failed attempt: keep it, renamed
        os.replace(failed, failed[:-5] + f".{int(time.time())}.json")
    from usbg.plug import admission as AD
    slot, name = A.adapter.split(":", 1)
    t0 = time.time()
    with AD.cuda_watch() as cw:
        if A.adapter == "ctrl:sb3":
            comps, par, extra = gate_ctrl_sb3(A)
        elif A.adapter == "ctrl:goalnet":
            comps, par, extra = gate_ctrl_goalnet(A)
        elif A.adapter == "ctrl:reference":
            comps, par, extra = gate_ctrl_reference(A)
        elif A.adapter == "mask:tv":
            comps, par, extra = gate_mask_tv(A)
        elif A.adapter == "mask:nnunet":
            comps, par, extra = gate_mask_nnunet(A)
        elif A.adapter in ("mask:sam", "mask:medsam"):
            comps, par, extra = gate_mask_sam(A, name)
        elif A.adapter == "reward:terms":
            comps, par, extra = gate_reward(A)
        elif slot == "texture":
            comps, par, extra = gate_texture(A, name)
        else:
            comps, par, extra = gate_renderer(A, name)
    extra = dict(extra or {}, cuda_moves_total=int(cw["n"]), wall_s=round(time.time() - t0, 1),
                 peak_rss_mb=AD.peak_rss_mb(), argv=list(sys.argv), cases=list(A.cases), frame_case=A.case)
    body = AD.summarize(A.adapter, slot, comps, par, extra)
    if cw["n"]:
        body["admitted"] = False
        body["checks"]["cuda_free"]["passed"] = False
    path = AD.write_result(A.out, body)
    bad = [c.get("component") for c in comps + par if not c.get("passed")]
    print(f"[gate] {A.adapter}: {'ADMITTED' if body['admitted'] else 'NOT ADMITTED'} in {extra['wall_s']:.0f}s "
          f"(failed: {bad or 'none'}; cuda moves {cw['n']}) -> {path}")
    return 0 if body["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
