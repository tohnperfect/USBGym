#!/usr/bin/env python3
"""Time one full closed-loop step of the v2 view-matching loop, per texture stage and device.

One step is exactly the body of the evaluation loop in scripts/train_view_match_rl.py:evaluate (the
loop every oracle-ladder policy number came from), run under the learned-mask setup of the v2 matched
policies (USBG_LEARNED_SEG=1, the stage- and seed-matched segmenter from the segmenter manifest, the
saved policy and IBVS base from outputs/v2/table4/matched/<stage>_deploy_s<seed>.pt):

  policy     history stack -> device, GoalPolicy.act (deterministic), action back to the CPU
  ibvs_base  the calibrated segmentation-moment base twist (NumPy)
  mujoco     USProbeEnv.step (10 mj_step settle iterations, patient-surface contact on) + contact_force
  slice      slicer.slice_at_pose (CT label slice at the probe pose)
  render     render_lotus.render(.., "physics") (the analytic forward model)
  warp       resize to 128 x 128, label alignment, linear -> curvilinear sector warp
  texture    the texture stage (none for physics; CUT / CycleGAN single input; SonoSPADE image + label)
  segment    the learned cross-domain segmenter on the textured frame (the mask the reward and base use)
  other      history update, pose error, organ IoU, combined match score and loop overhead

The step is timed as ONE block (perf_counter around the whole body); the components are accumulated by
thin wrappers on the same calls, so the code path is the evaluation code path, unmodified. Every
component ends in a device-to-host copy (.cpu()), so MPS work is complete when its timer stops.

Batch 1, one process per device, stages in sequence. Each device runs in its own child process; the
CPU child hides MPS (torch.backends.mps.is_available -> False before usbg is imported), so every
usbg get_device() call resolves to the CPU, as on a machine with no GPU at all; the child also checks
that no tensor was allocated on MPS. CUDA is never requested (torch.cuda.is_available() is recorded).

Episodes follow the evaluation design (eval seed 999, crossed patient x target x window cells on the six
held-out patients, 12 steps per episode). The first --warmup-episodes are discarded; the next
--episodes are timed (default 2 + 20 episodes, 240 timed steps per stage and device).

  USBG_SEAT=anatomical PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python \\
      scripts/bench_closed_loop.py --out outputs/v2/bench_closed_loop.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ["physics", "cut", "cyclegan", "sonospade"]
SEG_MANIFEST = os.path.join(os.environ.get("USBG_SEG_REPO", os.path.join(REPO, "segtrain")), "outputs", "v2",
                            "seg_manifest.json")
COMPONENTS = ("policy", "ibvs_base", "mujoco", "slice", "render", "warp", "texture", "segment")


def _stats(ms):
    a = np.asarray(ms, float)
    med = float(np.median(a))
    return {"n": int(a.size), "median_ms": med, "p25_ms": float(np.percentile(a, 25)),
            "p75_ms": float(np.percentile(a, 75)), "mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "steps_per_s_from_median": 1000.0 / med, "steps_per_s_from_mean": 1000.0 / float(a.mean())}


def _loadavg():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


# ------------------------------------------------------------------------------------------ child
def child(dev, a):
    import torch
    if dev == "cpu":                                  # emulate a machine with no GPU before usbg loads
        torch.backends.mps.is_available = lambda: False
    elif not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    sys.path.insert(0, os.path.join(REPO, "src"))
    from collections import deque
    import train_view_match_rl as T
    import eval_view_ladder as EVL
    from usbg import goal_rl as G, render_lotus as R, slicer as S, view_match as VM
    from usbg import provenance as PROV, us_segmenter as USS
    from usbg.similarity import load_similarity
    from usbg.texture_gan import load_translator

    if R.get_device() != dev:
        raise SystemExit(f"get_device() resolves to {R.get_device()!r}, expected {dev!r}")

    acc = {k: 0.0 for k in COMPONENTS}
    active = [False]

    def timed(name, fn):
        def w(*args, **kw):
            if not active[0]:
                return fn(*args, **kw)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                acc[name] += (time.perf_counter() - t0) * 1e3
        return w

    # wrap the calls VM.curvi_render makes (module attributes are looked up at call time)
    S.slice_at_pose = timed("slice", S.slice_at_pose)
    R.render = timed("render", R.render)
    VM._resize01 = timed("warp", VM._resize01)
    VM.align_label_to_image = timed("warp", VM.align_label_to_image)
    VM.warp_to_sector = timed("warp", VM.warp_to_sector)

    class TimedTex:
        def __init__(self, tr):
            self.tr, self.wants_label = tr, getattr(tr, "wants_label", False)

        def translate_aligned(self, img, lab):
            return timed("texture", self.tr.translate_aligned)(img, lab)

        def __call__(self, img):
            return timed("texture", self.tr)(img)

    class TimedSeg:
        def __init__(self, seg):
            self.seg = seg

        def segment(self, img):
            return timed("segment", self.seg.segment)(img)

    seg_manifest = json.load(open(a.seg_manifest))
    out = {"device": dev, "torch_threads": torch.get_num_threads(),
           "cuda_available": bool(torch.cuda.is_available()), "stages": {}}
    T._CONTACT, T._RANDOMIZE = True, "off"
    lv_cache = {}
    for stage in a.stages:
        pj = os.path.join(REPO, a.policy_dir, f"{stage}_deploy_s{a.seed}.json")
        J = json.load(open(pj))
        c = J["config"]
        if J.get("residual_mode") != "image" or not c.get("fix_residual"):
            raise SystemExit(f"{pj}: expected the image-servo residual setup of the v2 matched policies")
        T.SEAT_ORGANS = list(c["targets"])
        T._MIN_WINDOWS = int(c.get("min_windows") or 1)
        eval_cases = list(c["eval_cases"])
        key = tuple(eval_cases)
        if key not in lv_cache:
            lv_cache[key] = T.load_volumes(eval_cases)
        lv_eval = lv_cache[key]
        cells, _skipped = T.eval_cells(lv_eval, eval_cases, c.get("eval_design", "crossed"))
        tex = T.resolve_texture(stage, c.get("texture_ckpt") or "none")
        tr = None if tex is None else TimedTex(load_translator(tex, dev))
        seg_ckpt = seg_manifest[f"{stage}_s{a.seed}"]
        os.environ["US_SEG_CKPT"] = seg_ckpt                     # so provenance.mask_info records it
        seg = USS.load_segmenter(seg_ckpt, device=dev)
        VM.set_mask_source(TimedSeg(seg))
        pt = pj.replace(".json", ".pt")
        ck0 = torch.load(pt, map_location=dev, weights_only=False)
        ib = ck0["ibvs"]
        T._RESIDUAL_MODE = "image"
        T._IBVS = {"kind": ib["kind"], "model": np.asarray(ib["model"]), "rich": ib["rich"]}
        pol, _ = EVL.load_policy(pt, c.get("residual_frac", 0.5), True, dev)
        cfg = R.RenderConfig(seed=int(c["seed"]))
        sim = load_similarity("ssim", device=dev)
        history, size, n_steps = int(c["history"]), int(c["image_size"]), int(c["n_steps"])
        setup_kw = dict(lateral_m=c["lateral_m"], axial_m=c["axial_m"], fan_deg=c["fan_deg"])
        steps, comps, setups, eps_meta = [], {k: [] for k in COMPONENTS}, [], []
        n_eps = a.warmup_episodes + a.episodes
        with torch.no_grad():
            for ep in range(n_eps):
                ts0 = time.perf_counter()
                rng = np.random.default_rng(a.eval_seed + ep)            # = evaluate()'s per-episode setup
                vi, organ = cells[ep % len(cells)]
                window = ep // len(cells)
                lv = lv_eval[vi]
                Wfv, T_tgt, T_start, meta = T.make_setup(lv, rng, organ, setup_kw, window=window, count=False)
                env = T.make_env(lv, Wfv, T_start, dev, np.random.default_rng(a.eval_seed + ep))
                tgt_img, tgt_lab = VM.curvi_render(lv, T_tgt, Wfv, tr, cfg, size)
                tgt_prep = T._prep(tgt_img, size)
                tgt_t = torch.from_numpy(tgt_prep[None, None]).to(dev)
                env.reset()
                s0_img, s0_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, tr, cfg, size)
                hist = deque([T._prep(s0_img, size)] * history, maxlen=history)
                cur_lab = s0_lab
                timed_ep = ep >= a.warmup_episodes
                if timed_ep:
                    setups.append((time.perf_counter() - ts0) * 1e3)
                    eps_meta.append(f"{eval_cases[vi]}:{organ}:w{window}")
                for _ in range(n_steps):
                    for k in acc:
                        acc[k] = 0.0
                    active[0] = True
                    t0 = time.perf_counter()
                    # ---- the evaluate() loop body (scripts/train_view_match_rl.py), unchanged ----
                    tp = time.perf_counter()
                    cur_np = np.stack(hist, 0)
                    cur = torch.from_numpy(cur_np[None]).to(dev)
                    act, _ = pol.act(cur, tgt_t, deterministic=True)
                    a_pol = act[0].detach().cpu().numpy()
                    acc["policy"] += (time.perf_counter() - tp) * 1e3
                    tb = time.perf_counter()
                    base = T.base_twist(env.world_from_probe, T_tgt, cur_lab, tgt_lab, cur_np, tgt_prep, True)
                    acc["ibvs_base"] += (time.perf_counter() - tb) * 1e3
                    a_exec = np.clip(a_pol + base, -G.MAX_TWIST, G.MAX_TWIST)
                    tm = time.perf_counter()
                    env.step(a_exec)
                    _force = env.contact_force()
                    acc["mujoco"] += (time.perf_counter() - tm) * 1e3
                    oi, cur_lab = VM.curvi_render(lv, env.world_from_probe, Wfv, tr, cfg, size)
                    hist.append(T._prep(oi, size))
                    _pe, _ae = VM.pose_error(env.world_from_probe, T_tgt)
                    _seg = VM.seg_organ_iou(cur_lab, tgt_lab)
                    _m = VM.combined_score(oi, cur_lab, tgt_img, tgt_lab, sim, c["w_seg"], c["w_app"])
                    # ---- end of the loop body ----
                    dt = (time.perf_counter() - t0) * 1e3
                    active[0] = False
                    if timed_ep:
                        steps.append(dt)
                        for k in COMPONENTS:
                            comps[k].append(acc[k])
                env.close()
        other = [s - sum(comps[k][i] for k in COMPONENTS) for i, s in enumerate(steps)]
        res = {"step": _stats(steps),
               "components_median_ms": {k: float(np.median(v)) for k, v in comps.items()},
               "components_mean_ms": {k: float(np.mean(v)) for k, v in comps.items()},
               "other_median_ms": float(np.median(other)), "other_mean_ms": float(np.mean(other)),
               "episode_setup_median_ms": float(np.median(setups)),
               "n_timed_episodes": a.episodes, "n_warmup_episodes": a.warmup_episodes,
               "steps_per_episode": n_steps, "episodes": eps_meta,
               "policy_json": os.path.relpath(pj, REPO), "policy_pt_md5": PROV.md5(pt),
               "texture_ckpt": tex and os.path.relpath(tex, REPO), "texture_md5": tex and PROV.md5(tex),
               "seg_ckpt": seg_ckpt, "seg_md5": PROV.md5(seg_ckpt)}
        out["stages"][stage] = res
        s = res["step"]
        print(f"[{dev}] {stage:9s} step median {s['median_ms']:7.2f} ms (IQR {s['p25_ms']:.2f}-{s['p75_ms']:.2f}, "
              f"n={s['n']}) -> {s['steps_per_s_from_median']:6.1f} steps/s | " +
              " ".join(f"{k} {v:.1f}" for k, v in res["components_median_ms"].items()), flush=True)
        VM.set_mask_source(None)
    if dev == "cpu":
        try:
            out["mps_allocated_bytes_at_end"] = int(torch.mps.current_allocated_memory())
        except Exception as e:                       # no MPS backend at all: nothing could be allocated
            out["mps_allocated_bytes_at_end"] = 0
            out["mps_check_note"] = repr(e)
        if out["mps_allocated_bytes_at_end"] != 0:
            raise SystemExit(f"CPU run allocated {out['mps_allocated_bytes_at_end']} bytes on MPS")
    out["get_device"] = R.get_device()
    return out


# ------------------------------------------------------------------------------------------ parent
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None, choices=["cpu", "mps"],
                    help="default: cpu, plus mps when available")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--seed", type=int, default=0, help="policy / segmenter seed (matched arm)")
    ap.add_argument("--policy-dir", default="outputs/v2/table4/matched")
    ap.add_argument("--seg-manifest", default=SEG_MANIFEST)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--episodes", type=int, default=20, help="timed episodes (12 steps each)")
    ap.add_argument("--warmup-episodes", type=int, default=2)
    ap.add_argument("--out", default="outputs/v2/bench_closed_loop.json")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_child_out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a._child:
        res = child(a._child, a)
        with open(a._child_out, "w") as fh:
            json.dump(res, fh, indent=1)
        return 0

    import torch
    from usbg.seatmode import seat_mode
    if seat_mode() != "anatomical":
        raise SystemExit("bench_closed_loop times the v2 (anatomical-seat) policies; set USBG_SEAT=anatomical")
    devices = a.devices or (["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
    env = dict(os.environ, USBG_LEARNED_SEG="1", PYTHONPATH=os.path.join(REPO, "src"))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    load_before = _loadavg()
    t0 = time.time()
    per_device = {}
    for dev in devices:
        fd, tmp = tempfile.mkstemp(suffix=f"_{dev}.json")
        os.close(fd)
        cmd = [sys.executable, os.path.abspath(__file__), "--_child", dev, "--_child_out", tmp,
               "--stages", *a.stages, "--seed", str(a.seed), "--policy-dir", a.policy_dir,
               "--seg-manifest", a.seg_manifest, "--eval-seed", str(a.eval_seed),
               "--episodes", str(a.episodes), "--warmup-episodes", str(a.warmup_episodes)]
        print(f"== device {dev}: one child process", flush=True)
        r = subprocess.run(cmd, cwd=REPO, env=env)
        if r.returncode != 0:
            raise SystemExit(f"{dev} child failed (rc={r.returncode})")
        per_device[dev] = json.load(open(tmp))
        os.remove(tmp)
    os.environ["USBG_LEARNED_SEG"] = "1"
    from usbg import provenance as P
    def _sysctl(k):
        try:
            return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
        except Exception:
            return None
    mem = _sysctl("hw.memsize")
    payload = {
        "what": "one closed-loop step of the v2 evaluation loop (train_view_match_rl.evaluate body), "
                "matched learned-mask policies, batch 1, one process per device",
        "per_device": per_device,
        "stages": a.stages, "seed": a.seed, "eval_seed": a.eval_seed,
        "machine": {"chip": _sysctl("machdep.cpu.brand_string"),
                    "memory_gb": round(int(mem) / 2 ** 30, 1) if mem else None,
                    "ncpu": _sysctl("hw.ncpu"), "system": f"{platform.system()} {platform.machine()}",
                    "cuda_available": bool(torch.cuda.is_available()),
                    "mps_available": bool(torch.backends.mps.is_available())},
        "loadavg_before": load_before, "loadavg_after": _loadavg(),
        "wall_sec": round(time.time() - t0, 1),
        "provenance": P.collect(stages=a.stages, policy_dir=a.policy_dir, seg_manifest=a.seg_manifest,
                                seg_manifest_md5=P.md5(a.seg_manifest)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[report] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
