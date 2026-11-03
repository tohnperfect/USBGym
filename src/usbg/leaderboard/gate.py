"""usbg.leaderboard.gate: the admission gate for a submitted plug-in (python -m usbg.leaderboard check).

The checks of usbg.plug.admission, on two training patients only and with CUDA watched:

  (i)   privileged channel: a plug-in declared non-privileged must give identical outputs when the privileged
        inputs it could reach are removed or poisoned, and must read none of them; a plug-in declared privileged
        must be caught reading them;
  (iii) CUDA-free and device parity: no tensor may move to a CUDA device; where Apple MPS is available the
        outputs on the CPU and on MPS must agree within the gate's fixed tolerances, and one training episode
        runs on each device. Without MPS the parity checks are recorded as CPU only.

A renderer reads the label slice by design, so its checks are a renderer's own ("renderer_checks": the same seed
gives the same frame, and every frame lies in [0, 1]). The record states no pretraining corpus: the submitter
states the method's training data in the method description.

The gate never touches a held-out patient. A plug-in is admitted only if every check passes.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ._exec import install_renderer, load_entry

GATE_CASES = ("s0011", "s0344")


def _env(cases, device=None, **kw):
    from usbg.plug import controllers as C
    return C.make_env(patients=list(cases), mask=kw.pop("mask", "ct"), device=device, **kw)


def _static():
    from usbg.plug import controllers as C
    return C.ZeroController("static")


def check(slot, entry, *, privileged=False, cases=GATE_CASES, n_views=4, out=None):
    from usbg.plug import admission as AD
    factory = load_entry(entry)
    label = f"{slot}:{entry}"
    views = AD.train_views(cases[0], n=n_views)
    comps, par, rchecks = [], [], []
    t0 = time.time()
    with AD.cuda_watch() as cw:
        if slot == "texture":
            tr = factory("cpu")
            comps.append(AD.check_texture(tr, views, label=label,
                                          declared_label_conditioned=bool(getattr(tr, "wants_label", False))))
            inputs = [np.asarray(v["img"], np.float32) for v in views]
            par.append(AD.frame_parity(lambda dev: factory(dev), inputs, label=label))
            par.append(AD.episode_parity(lambda dev: _env(cases, dev, texture=factory(dev)), lambda dev: _static(),
                                         label=f"{label} episode (CT mask)"))
        elif slot == "mask":
            comps.append(AD.check_mask_source(factory("cpu"), views, label=label))
            par.append(AD.mask_parity(lambda dev: factory(dev), views, label=label))
            par.append(AD.episode_parity(lambda dev: _env(cases, dev, mask=factory(dev)), lambda dev: _static(),
                                         label=f"{label} episode"))
        elif slot == "renderer":
            from usbg import render_lotus as R
            from usbg import slicer as SL
            labs = [SL.slice_at_pose(v["lv"], np.asarray(v["T"], np.float64), v["Wfv"]) for v in views[:2]]
            cfg = R.RenderConfig().with_seed(0)
            fn = factory("cpu")
            det = []
            for lab in labs:
                a = np.asarray(fn(lab, device="cpu", cfg=cfg, seed=0), np.float64)
                b = np.asarray(fn(lab, device="cpu", cfg=cfg, seed=0), np.float64)
                det.append(float(np.max(np.abs(a - b))))
            rchecks.append({"component": f"{label} determinism (same seed twice, CPU)", "max_abs_diff": max(det),
                            "passed": max(det) == 0.0})
            rng = [(float(np.min(fn(lab, device="cpu", cfg=cfg, seed=0))), float(np.max(fn(lab, device="cpu", cfg=cfg,
                                                                                         seed=0)))) for lab in labs]
            rchecks.append({"component": f"{label} output range [0, 1]", "range": rng,
                            "passed": all(lo >= 0.0 and hi <= 1.0 for lo, hi in rng)})
            par.append(AD.frame_parity(lambda dev: (lambda lab, dev=dev: factory(dev)(lab, device=dev, cfg=cfg,
                                                                                      seed=0)),
                                       labs, label=label))
            install_renderer(factory, "cpu")
            par.append(AD.episode_parity(lambda dev: _env(cases, dev), lambda dev: _static(),
                                         label=f"{label} episode (CT mask, static)"))
        elif slot == "reward":
            from usbg.plug import rewards as RW
            plug = factory()
            comps.append(AD.check_reward(plug, views, label=label))
            par.append(AD.episode_parity(lambda dev: RW.PluggedRewardWrapper(_env(cases, dev), reward=factory()),
                                         lambda dev: _static(), label=f"{label} episode (static)"))
        elif slot == "ctrl":
            env = _env(cases, "cpu", privileged_info=True)
            comps.append(AD.check_controller(env, lambda: factory(0, "cpu"), label=label,
                                             declared_privileged=bool(privileged)))
            env.close()
            par.append(AD.episode_parity(lambda dev: _env(cases, dev), lambda dev: factory(0, dev),
                                         label=f"{label} episode (untrained)"))
        else:
            raise SystemExit(f"--slot {slot}: the gate covers texture, mask, ctrl, reward and renderer")
    from .runner import plugin_sha256
    extra = {"plugin": entry, "plugin_source_sha256": plugin_sha256(entry), "cases": list(cases),
             "cuda_moves_total": int(cw["n"]), "wall_s": round(time.time() - t0, 1), "declared_privileged": privileged}
    body = AD.summarize(label, slot, comps, par, extra)
    body["checks"]["pretraining_corpus"] = {"corpus": "not recorded by the leaderboard gate: the submitter states the "
                                                      "method's training data in the method description", "key": None}
    if slot == "renderer":
        ok = all(c["passed"] for c in rchecks)
        body["checks"]["renderer_checks"] = {"passed": ok, "components": rchecks}
        body["admitted"] = bool(body["admitted"] and ok)
    if cw["n"]:
        body["admitted"] = False
        body["checks"]["cuda_free"]["passed"] = False
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(body, fh, indent=1, default=float)
    return body


__all__ = ["check", "GATE_CASES"]
