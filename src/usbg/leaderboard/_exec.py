"""usbg.leaderboard._exec: run one step of a slot condition with a plug-in installed (one process per step).

    python -m usbg.leaderboard._exec --slot SLOT --plugin package.module:factory [--patients ID ...] \\
        -- TARGET [target arguments]

TARGET is a script of the repository's scripts/ folder that the reported runs used (eval_view_ladder,
plug_reward_train), a segmenter-trainer script (segtrain:<path relative to segtrain/>), or ctrl_eval (the
controller condition, defined here). Before the target's main() starts, the plug-in is installed for this process
only, through the same hooks the published adapters use:

  renderer  every render_lotus.render(..., "physics", cfg) call goes to the plug-in, seeded by cfg.seed
            (as usbg.plug.renderers.install routes the published renderers)
  texture   a checkpoint whose state["arch"] is "lb_plugin" loads the plug-in (usbg.texture_gan.register_translator),
            and the --translator choice lists accept "lb_plugin"
  mask      view_match.set_mask_source(plug-in) and US_SEG_CKPT = the plug-in's source file (the run guards and
            the provenance record it)
  reward    the reward name "lb_plugin" of scripts/plug_reward_train.py builds the plug-in
  ctrl      ctrl_eval trains the plug-in on the training patients (its train(make_env, budget, seed)) and evaluates
            it on the 60 held-out episodes; a plug-in not declared privileged (--privileged) never sees the
            privileged keys of info (the pose errors), which only the scoring code reads

--patients restricts a run to the given held-out patients' episodes (a smoke or identity run; the other episodes
are skipped, and a submission must have all 60).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
import time

import numpy as np

PLUGIN_NAME = "lb_plugin"


def load_entry(entry):
    """package.module:factory -> the factory callable."""
    mod, _, fn = entry.partition(":")
    if not mod or not fn:
        raise SystemExit(f"--plugin {entry!r}: expected package.module:factory")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


def entry_source(entry):
    """The source file of the plug-in's module."""
    mod = entry.partition(":")[0]
    spec = importlib.util.find_spec(mod)
    return spec.origin if spec is not None else None


def device_of(dev=None):
    from usbg import render_lotus as R
    return dev or R.get_device()


# ============================================================================================ installers
def install_renderer(factory, device=None):
    """Route the loop's physics renders to the plug-in (the routing of usbg.plug.renderers.install)."""
    from usbg import render_lotus as R
    fn = factory(device_of(device))
    R.register_renderer(PLUGIN_NAME)(fn)
    orig = R.render

    def render(label_slice, renderer="physics", device=None, texture=None, cfg=None, **kw):
        if renderer != "physics":
            return orig(label_slice, renderer, device=device, texture=texture, cfg=cfg, **kw)
        c = cfg if cfg is not None else R.RenderConfig()
        seed = kw.pop("seed", c.seed)
        return orig(label_slice, PLUGIN_NAME, device=device, texture=texture, cfg=c, seed=seed, **kw)

    render.__usbg_plug__ = True
    R.render = render
    return fn


def install_physics_reference():
    """The reference renderer through the same routing (identity runs): nothing is replaced."""
    return None


def install_texture(factory, device=None):
    from usbg import plug_texture as PT
    from usbg import texture_gan as TG

    @TG.register_translator(PLUGIN_NAME)
    def _load(ckpt_path, device=None):
        return factory(device or device_of())

    PT.widen_choices(extra=tuple(PT.PLUG_STAGES) + (PLUGIN_NAME,))


def write_texture_marker(path, entry):
    """The checkpoint file the texture-slot scripts receive: it only names the plug-in."""
    import torch
    torch.save({"arch": PLUGIN_NAME, "entry": entry}, path)


def install_mask(factory, entry, device=None):
    from usbg import view_match as VM
    src = factory(device_of(device))
    os.environ["US_SEG_CKPT"] = entry_source(entry) or entry
    os.environ["USBG_LEARNED_SEG"] = "1"
    VM.set_mask_source(src)
    return src


def install_reward(factory):
    from usbg.plug import rewards as RW
    orig = RW.RewardPlug
    plug = factory()
    if PLUGIN_NAME not in RW.REWARDS:
        RW.REWARDS = tuple(RW.REWARDS) + (PLUGIN_NAME,)
    cit = getattr(RW, "CITATIONS", None)
    if isinstance(cit, dict) and PLUGIN_NAME not in cit:
        cit[PLUGIN_NAME] = getattr(plug, "citation", "submitted method")

    def make(name="paper", **kw):
        return plug if name == PLUGIN_NAME else orig(name, **kw)

    RW.RewardPlug = make
    return plug


def restrict_patients(patients):
    """Skip the held-out episodes of every patient not in `patients` (smoke and identity runs only)."""
    import train_view_match_rl as T
    from usbg.placement import NoAcceptedWindow
    keep, ids = set(patients), {}
    load0, setup0 = T.load_volumes, T.make_setup

    def load_volumes(cases):
        lvs = load0(cases)
        for c, lv in zip(cases, lvs):
            ids[id(lv)] = c
        return lvs

    def make_setup(lv, rng, organ, setup_kw, window=None, count=True):
        c = ids.get(id(lv))
        from usbg.leaderboard.protocol import HELD_OUT_PATIENTS
        if c in HELD_OUT_PATIENTS and c not in keep:
            raise NoAcceptedWindow(f"{c}: skipped (leaderboard run restricted to {sorted(keep)})")
        return setup0(lv, rng, organ, setup_kw, window=window, count=count)

    T.load_volumes, T.make_setup = load_volumes, make_setup


# ============================================================================================ ctrl_eval
PRIVILEGED_INFO_KEYS = ("privileged",)


class Unprivileged:
    """A controller that is not declared privileged, as the evaluation drives it: act() receives a copy of info
    without its privileged keys (the scoring code keeps the full info)."""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def reset(self, *a, **kw):
        return self.ctrl.reset(*a, **kw)

    def act(self, obs, info):
        return self.ctrl.act(obs, {k: v for k, v in info.items() if k not in PRIVILEGED_INFO_KEYS})

    def __getattr__(self, name):
        return getattr(self.ctrl, name)


def ctrl_eval(argv, factory, entry, privileged=False):
    """The controller condition: SonoSPADE frames, CT mask, the servo base of the reference policy, the plug-in
    trained on the 13 training patients within the protocol budget, then the 60 held-out episodes."""
    from usbg.leaderboard import protocol as PRO
    from usbg.plug import controllers as C
    ap = argparse.ArgumentParser(prog="ctrl_eval")
    ap.add_argument("--texture-ckpt", required=True)
    ap.add_argument("--servo-base", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default="servo", choices=["servo", "none"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = device_of(a.device)

    def make_env(patients=PRO.TRAIN_PATIENTS, privileged_info=False, **kw):
        if set(patients) & set(PRO.HELD_OUT_PATIENTS):
            raise SystemExit("ctrl_eval: training environments take training patients only")
        return C.make_env(patients=list(patients), texture="sonospade", texture_ckpt=a.texture_ckpt, mask="ct",
                          base=None if a.base == "none" else "servo",
                          servo_base=None if a.base == "none" else a.servo_base, device=dev,
                          privileged_info=privileged_info, **kw)

    ctrl = factory(a.seed, dev)
    t0 = time.time()
    budget = {"rl_env_steps": PRO.RL_ENV_STEPS, "imitation_samples": PRO.IMITATION_SAMPLES}
    if hasattr(ctrl, "train"):
        ctrl.train(make_env, budget, a.seed)
    train_s = time.time() - t0
    test = C.make_env(patients=list(PRO.HELD_OUT_PATIENTS), texture="sonospade", texture_ckpt=a.texture_ckpt,
                      mask="ct", base=None if a.base == "none" else "servo",
                      servo_base=None if a.base == "none" else a.servo_base, device=dev, privileged_info=True)
    ev = C.evaluate(test, ctrl if privileged else Unprivileged(ctrl), n_episodes=PRO.N_EPISODES)
    out = {"slot": "ctrl", "plugin": entry, "seed": a.seed, "train_wall_s": round(train_s, 1), "device": dev,
           "privileged_info_to_plugin": bool(privileged),
           "results": [{"contact": "on", "eval_randomize": "off", "weights": "plugin", **ev}],
           "wall_sec": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


# ============================================================================================ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m usbg.leaderboard._exec --slot S --plugin mod:factory -- TARGET [args]")
    k = argv.index("--")
    ap = argparse.ArgumentParser(prog="usbg.leaderboard._exec")
    ap.add_argument("--slot", required=True, choices=["texture", "mask", "ctrl", "reward", "renderer"])
    ap.add_argument("--plugin", required=True, help="package.module:factory, or 'reference' for the slot's own "
                                                    "reference component")
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--privileged", action="store_true", help="the plug-in is declared privileged (controller slot: "
                                                              "it receives info['privileged'])")
    a = ap.parse_args(argv[:k])
    if a.device:
        os.environ["USBG_DEVICE"] = a.device         # usbg.render_lotus.get_device: the evaluation's device too
    rest = argv[k + 1:]
    if not rest:
        raise SystemExit("no target after --")
    repo = os.environ.get("USBG_REPO", os.getcwd())
    scripts = os.path.join(repo, "scripts")
    for p in (scripts, os.path.join(scripts, "plug")):
        if p not in sys.path:
            sys.path.insert(0, p)
    target, targv = rest[0], rest[1:]
    ref = a.plugin == "reference"
    factory = None if ref else load_entry(a.plugin)
    if not ref:
        if a.slot == "renderer":
            install_renderer(factory, a.device)
        elif a.slot == "texture":
            install_texture(factory, a.device)
        elif a.slot == "mask":
            install_mask(factory, a.plugin, a.device)
        elif a.slot == "reward":
            install_reward(factory)
    if a.patients:
        restrict_patients(a.patients)
    if target == "ctrl_eval":
        if ref:
            raise SystemExit("ctrl_eval runs a plug-in; the reference controller is the archived REINFORCE policy")
        return ctrl_eval(targv, factory, a.plugin, privileged=a.privileged)
    if target.startswith("segtrain:"):
        script = os.path.join(repo, "segtrain", target.split(":", 1)[1])
        seg_src = os.path.join(repo, "segtrain", "src")
        if seg_src not in sys.path:
            sys.path.insert(0, seg_src)
        sys.argv = [script] + targv
        sys.path.insert(0, os.path.dirname(script))
        cwd = os.getcwd()
        os.chdir(os.path.join(repo, "segtrain"))
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            os.chdir(cwd)
        return 0
    mod = importlib.import_module(target)
    if target == "plug_reward_train" and os.environ.get("USBG_LB_WORK"):
        mod.OUT_ROOTS = tuple(mod.OUT_ROOTS) + (os.path.abspath(os.environ["USBG_LB_WORK"]),)
    return int(mod.main(targv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
