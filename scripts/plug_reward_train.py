#!/usr/bin/env python3
"""plug_reward_train.py: closed-loop training of the paper's learner with a plugged reward (v2.1
pluggability, exploratory, opt-in). Runs scripts/train_view_match_rl.py UNCHANGED.

The learner is fixed: the paper's servo base plus residual policy, imitation warm start and REINFORCE with
the BC anchor (train_view_match_rl.main with the arguments after "--"). Only the training reward changes:
inside every RL rollout (train_view_match_rl.rollout, the only place the reward is formed) the potential S
of r_t = S_t - S_{t-1} - effort * |a_t|^2 is the plugged one (usbg.plug.rewards.RewardPlug). Evaluation
is untouched: the held-out pose error, the paper's match score S, the keep-best rule and the learning curve
use the trainer's own code, so every plugged reward is scored on the same 60 crossed held-out episodes
(eval seed 999) as the existing rows.

How the swap is done (no trainer edit): the trainer is imported from its file, its module-level rollout
is replaced by a wrapper that binds view_match.combined_score to the plugged potential for the duration of
one rollout and restores it afterwards, ANALYSIS_STATUS_V21 is set to "exploratory, v2.1 pluggability",
and usbg.provenance.collect adds that status and a "plug" record to every provenance block (the output
JSON and the checkpoint sidecars). With --reward paper the wrapper is a no-op on the numbers: the
plugged potential is view_match.combined_score's arithmetic on the same inputs.

The output JSON gains "plug": the reward record and a training log (per block of --log-every rollouts:
the mean plugged S and the mean paper S (weights 0.6 and 0.4) at the first and the last step, and the mean step reward), so a
run shows both what it optimized and how the paper's score moved.

Recommended protocol (the job spec): start the RL phase from the paper's post-imitation checkpoint of the
same stage and seed (--init-from outputs/v2/table4/ctmask/<stage>_deploy_s<seed>_imit.pt --imitation 0),
so every reward arm shares the identical imitation policy and image base and the arms differ only in the
reward of the 57,600 RL steps.

The opt-in combination arm:
  (a) outputs/v2_1/combination/ joins outputs/v2_1/plug/reward/ as an allowed output root; any other root is refused.
  (b) --render-seed N (a wrapper option, before "--"): it needs the trainer's --init-from and DR off, and it is
      refused together with --real-eval. It wraps three module-level functions of the trainer, rollout,
      imitation_batch and evaluate (the reward swap wraps rollout too); each of them is called with
      cfg.with_seed(N) where the run passes RenderConfig(seed=--seed). The learner seed (--seed) still drives the RL
      setup stream, the episode streams, the action noise and the samples of the behavior-cloning anchor. Each wrapped call must arrive with RenderConfig(seed=--seed)
      and, for evaluate, eval_randomize "off", else the run stops. The calls are counted, the seeds they saw are
      recorded, and the output JSON gains "condition" {render_seed, learner_seed} (the first path that
      scripts/plug/episode_audit.render_seed reads) and plug.render_seed_record (call counts, seeds seen, and over
      every scored step of every rollout the largest |plugged potential - paper S| and the number of unequal steps).
      The provenance blocks gain plug.render_seed. Without the flag no wrapper is installed and no key is added.

Usage:
  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 PYTHONPATH=src python scripts/plug_reward_train.py --reward ssim \
      -- --translator physics --texture-ckpt none --fix-residual --residual-image --ibvs-features rich \
      --w-seg 0.6 --w-app 0.4 --imitation 0 --init-from outputs/v2/table4/ctmask/physics_deploy_s0_imit.pt \
      --iters 300 --episodes-per-iter 16 --eval-episodes 60 --seed 0 \
      --out outputs/v2_1/plug/reward/train/ssim/physics_s0.json
  the pre-specified protocol (learner seed s + 6, render seed s, learned masks):
  USBG_SEAT=anatomical USBG_LEARNED_SEG=1 US_SEG_CKPT=<seg_root>/us_seg_physics_s0.pt PYTHONPATH=src \
      python scripts/plug_reward_train.py --reward ssim --render-seed 0 -- ... \
      --init-from outputs/v2/table4/matched/physics_deploy_s0_imit.pt --seed 6 \
      --out outputs/v2_1/combination/train/physics_s0_l6.json
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_W = (0.6, 0.4)          # the paper's weights: the logged paper S is always the paper's score
OUT_ROOT = os.path.join(REPO, "outputs", "v2_1", "plug", "reward")
COMBO_ROOT = os.path.join(REPO, "outputs", "v2_1", "combination")      # the pre-specified protocol
OUT_ROOTS = (OUT_ROOT, COMBO_ROOT)
RENDER_SEED_CALLS = ("rollout", "imitation_batch", "evaluate")   # the pre-specified protocol: the wrapped renders


def load_trainer():
    path = os.path.join(REPO, "scripts", "train_view_match_rl.py")
    spec = importlib.util.spec_from_file_location("train_view_match_rl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_view_match_rl"] = mod
    spec.loader.exec_module(mod)
    return mod


def _arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _render_seed_refusals(ap, rest):
    """--render-seed (the pre-specified protocol): needs --init-from and DR off; refused with --real-eval.
    The literal flags are checked here; abbreviations the trainer's argparse would accept are caught at run time
    (RenderSeed.capture_init and the wrapped calls)."""
    if not _arg_value(rest, "--init-from"):
        ap.error("--render-seed is allowed only with the trainer's --init-from (the pre-specified protocol)")
    for flag in ("--randomize", "--eval-randomize"):
        v = _arg_value(rest, flag)
        if v is not None and v != "off":
            ap.error(f"--render-seed is allowed only with DR off; got {flag} {v}")
    if any(a == "--real-eval" or a.startswith("--real-eval=") for a in rest):
        ap.error("--render-seed is refused with --real-eval")


class RenderSeed:
    """The opt-in render seed of the pre-specified protocol: wraps the trainer's module-level rollout,
    imitation_batch and evaluate; each is called with cfg.with_seed(N) where the run passes RenderConfig(seed=--seed).

    The trainer's own parsed arguments are captured when it calls load_init_policy (so the learner seed and the DR
    and --real-eval settings are read as the trainer resolved them, abbreviations included); a wrapped call that
    arrives before that, with another render config, or (evaluate) with eval_randomize other than "off" stops the
    run. Calls are counted and the seeds they passed on are recorded."""

    def __init__(self, TVM, R, seed):
        self.TVM, self.R, self.seed = TVM, R, int(seed)
        self.args = None
        self.calls = {n: 0 for n in RENDER_SEED_CALLS}
        self.seen = {n: set() for n in RENDER_SEED_CALLS}
        self.sigs = {n: inspect.signature(getattr(TVM, n)) for n in RENDER_SEED_CALLS}
        self.n_scored = 0
        self.n_unequal = 0
        self.max_abs = 0.0

    def capture_init(self, fn):
        def load_init_policy(path, args, prov):
            if getattr(args, "randomize", "off") != "off" or getattr(args, "eval_randomize", "off") != "off":
                raise SystemExit("--render-seed: DR is on in the trainer (randomize "
                                 f"{getattr(args, 'randomize', None)!r}, eval_randomize "
                                 f"{getattr(args, 'eval_randomize', None)!r}); it is allowed only with DR off")
            if getattr(args, "real_eval", None):
                raise SystemExit("--render-seed is refused with --real-eval")
            self.args = args
            return fn(path, args, prov)
        return load_init_policy

    def wrap(self, name, fn):
        sig = self.sigs[name]

        def seeded(*a, **kw):
            if self.args is None:
                raise SystemExit(f"--render-seed: the trainer called {name} before loading --init-from")
            ba = sig.bind(*a, **kw)
            ba.apply_defaults()
            want = self.R.RenderConfig(seed=int(self.args.seed))
            got = ba.arguments["cfg"]
            if got != want:
                raise SystemExit(f"--render-seed: {name} received a render config other than "
                                 f"RenderConfig(seed={int(self.args.seed)}) (a DR draw or another seed); stopping")
            if name == "evaluate" and ba.arguments.get("eval_randomize") != "off":
                raise SystemExit(f"--render-seed: evaluate called with eval_randomize "
                                 f"{ba.arguments.get('eval_randomize')!r}; DR must be off")
            if getattr(self.TVM, "_RANDOMIZE", "off") != "off":
                raise SystemExit("--render-seed: the trainer's DR switch is on; DR must be off")
            ba.arguments["cfg"] = got.with_seed(self.seed)
            self.calls[name] += 1
            self.seen[name].add(int(ba.arguments["cfg"].seed))
            return fn(*ba.args, **ba.kwargs)
        return seeded

    def note_steps(self, trace):
        for s, ref in trace:
            self.n_scored += 1
            if s != ref:
                self.n_unequal += 1
            self.max_abs = max(self.max_abs, abs(float(s) - float(ref)))

    def record(self):
        called = [n for n in RENDER_SEED_CALLS if self.calls[n]]
        if not called:
            raise SystemExit("--render-seed: no render call was wrapped")
        ok = all(self.seen[n] == {self.seed} for n in called)
        if not ok:
            raise SystemExit(f"--render-seed: a wrapped call passed on a seed other than {self.seed}: {self.seen}")
        return {"value": self.seed, "learner_seed": int(self.args.seed), "calls": dict(self.calls),
                "seeds_seen": {n: sorted(self.seen[n]) for n in RENDER_SEED_CALLS},
                "all_calls_saw_render_seed": True,
                "scored_steps": {"n": int(self.n_scored), "n_plug_unequal_paper": int(self.n_unequal),
                                 "max_abs_plug_minus_paper": float(self.max_abs),
                                 "note": "every scored step of every rollout: the plugged potential against the "
                                         "paper S (weights 0.6 and 0.4); the two are equal bit for bit when "
                                         "--reward paper runs with the paper weights"},
                "wrapped": list(RENDER_SEED_CALLS),
                "rule": "every wrapped call is made with cfg.with_seed(render seed) where the run passes "
                        "RenderConfig(seed=--seed); all other random streams follow the learner seed "
                        "(the pre-specified protocol)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from usbg.plug import rewards as RW
    ap.add_argument("--reward", required=True, choices=list(RW.REWARDS))
    ap.add_argument("--pose-scale", type=float, default=RW.POSE_SCALE)
    ap.add_argument("--ang-w", type=float, default=RW.ANG_W)
    ap.add_argument("--area-coef", type=float, default=RW.AREA_COEF)
    ap.add_argument("--lpips-alexnet", default=None, help="local torchvision AlexNet weights for lpips")
    ap.add_argument("--lpips-device", default="cpu")
    ap.add_argument("--lpips-random-trunk-smoke", action="store_true",
                    help="SMOKE ONLY: random AlexNet trunk (no weights); requires --smoke")
    ap.add_argument("--log-every", type=int, default=16, help="rollouts per training-log block")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: outputs must go under a plug_smoke directory")
    ap.add_argument("--render-seed", type=int, default=None,
                    help="The opt-in combination arm: render seed N for every wrapped render call (rollout, imitation_batch, "
                         "evaluate) in place of the trainer's --seed; needs --init-from and DR off")
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.render_seed is not None:
        _render_seed_refusals(ap, rest)
    if args.lpips_random_trunk_smoke and not args.smoke:
        ap.error("--lpips-random-trunk-smoke is a code-path smoke test only; it needs --smoke")
    out = _arg_value(rest, "--out")
    if not out:
        ap.error("pass the trainer's --out after --")
    out = os.path.abspath(out)
    if args.smoke:
        if "/plug_smoke/" not in out:
            ap.error(f"--smoke outputs must go under a plug_smoke directory, got {out}")
    elif not any(out.startswith(r + os.sep) for r in OUT_ROOTS):
        ap.error(f"outputs must go under {OUT_ROOT}/ or {COMBO_ROOT}/, got {out}")
    for p in (out, out.replace(".json", ".pt"), out.replace(".json", "_imit.pt")):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    if "--fix-reward" in rest and args.reward != "paper":
        ap.error("--fix-reward adds the trainer's pose shaping on top of the reward; it is not combined with a "
                 "plugged reward (keep the arms clean)")

    from usbg import provenance as P
    from usbg import view_match as VM
    plug = RW.RewardPlug(args.reward, ang_w=args.ang_w, pose_scale=args.pose_scale, area_coef=args.area_coef,
                         lpips_kw={"alexnet_path": args.lpips_alexnet, "device": args.lpips_device,
                                   "random_trunk": args.lpips_random_trunk_smoke})
    w_seg_cli, w_app_cli = _arg_value(rest, "--w-seg"), _arg_value(rest, "--w-app")
    plug_info = plug.info()
    plug_info.update(versions=RW.package_versions(), wrapper=os.path.abspath(__file__), smoke=bool(args.smoke),
                     trainer_weights={"w_seg": w_seg_cli, "w_app": w_app_cli},
                     weights_note=("the combined potentials use the trainer's --w-seg / --w-app"
                                   if not plug.privileged else "the privileged potentials ignore w_seg / w_app"),
                     evaluation="unchanged trainer evaluate(): pose error, paper S, keep-best on the paper S")
    if args.render_seed is not None:                    # the pre-specified protocol: provenance blocks and sidecars carry it
        plug_info["render_seed"] = int(args.render_seed)

    TVM = load_trainer()
    TVM.ANALYSIS_STATUS_V21 = RW.ANALYSIS_STATUS        # this process only
    orig_rollout = TVM.rollout
    sig = inspect.signature(orig_rollout)
    ref_combined = VM.combined_score
    log = []                                            # per rollout: s0_plug, sT_plug, s0_ref, sT_ref, r_mean

    def plug_rollout(env, lv, Wfv, T_tgt, *a, **kw):
        ba = sig.bind(env, lv, Wfv, T_tgt, *a, **kw)
        ba.apply_defaults()
        trace = []

        def score(cur_img, cur_lab, tgt_img, tgt_lab, sim=None, w_seg=0.5, w_app=0.5):
            s = plug.potential(cur_img, cur_lab, tgt_img, tgt_lab, pose=env.world_from_probe, tgt_pose=T_tgt,
                               w_seg=w_seg, w_app=w_app)
            trace.append((s, ref_combined(cur_img, cur_lab, tgt_img, tgt_lab, sim, *PAPER_W)))
            return s

        with RW.patched(VM, "combined_score", score):
            res = orig_rollout(*ba.args, **ba.kwargs)
        if rs is not None:
            rs.note_steps(trace)
        if trace:
            r = np.asarray(res["rewards"], float)
            log.append((trace[0][0], trace[-1][0], trace[0][1], trace[-1][1], float(r.mean()) if r.size else 0.0))
        return res

    orig_collect = P.collect

    def collect_plug(**kw):
        prov = orig_collect(**kw)
        prov["analysis_status"] = RW.ANALYSIS_STATUS
        prov["plug"] = plug_info
        return prov

    rs = None
    if args.render_seed is not None:                    # the pre-specified protocol: opt-in only
        from usbg import render_lotus as R
        rs = RenderSeed(TVM, R, args.render_seed)
        TVM.load_init_policy = rs.capture_init(TVM.load_init_policy)
        TVM.imitation_batch = rs.wrap("imitation_batch", TVM.imitation_batch)
        TVM.evaluate = rs.wrap("evaluate", TVM.evaluate)
        TVM.rollout = rs.wrap("rollout", plug_rollout)
    else:
        TVM.rollout = plug_rollout
    t0 = time.time()
    with RW.patched(P, "collect", collect_plug):
        rc = TVM.main(list(rest))
    if rc not in (0, None):
        raise SystemExit(f"train_view_match_rl.main returned {rc}")

    # ---------------------------------------------------------------- the plug record in the output JSON
    L = np.asarray(log, float).reshape(-1, 5)
    k = max(1, int(args.log_every))
    blocks = [{"rollouts": [int(i), int(min(i + k, len(L)))],
               "plug_S_start": float(L[i:i + k, 0].mean()), "plug_S_end": float(L[i:i + k, 1].mean()),
               "paper_S_start": float(L[i:i + k, 2].mean()), "paper_S_end": float(L[i:i + k, 3].mean()),
               "reward_mean_step": float(L[i:i + k, 4].mean())} for i in range(0, len(L), k)]
    d = json.load(open(out))
    d["analysis_status"] = RW.ANALYSIS_STATUS
    d.setdefault("provenance", {})["analysis_status"] = RW.ANALYSIS_STATUS
    d["plug"] = {**plug_info, "n_rollouts": int(len(L)), "log_every": k, "train_log": blocks,
                 "train_summary": {"plug_gain_mean": float((L[:, 1] - L[:, 0]).mean()) if len(L) else None,
                                   "paper_gain_mean": float((L[:, 3] - L[:, 2]).mean()) if len(L) else None,
                                   "first_block": blocks[0] if blocks else None,
                                   "last_block": blocks[-1] if blocks else None},
                 "wrapper_wall_s": round(time.time() - t0, 1)}
    if rs is not None:                                  # the pre-specified protocol: the record of the render seed
        d["plug"]["render_seed_record"] = rs.record()
        d["condition"] = {"render_seed": rs.seed, "learner_seed": int(rs.args.seed),
                          "source": "scripts/plug_reward_train.py --render-seed (the pre-specified protocol)"}
    tmp = out + ".plugtmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2, default=float)
    os.replace(tmp, out)                                 # the file this run just wrote, now with the plug block
    print(f"  [plug] reward={args.reward} rollouts={len(L)} plug gain {d['plug']['train_summary']['plug_gain_mean']} "
          f"paper gain {d['plug']['train_summary']['paper_gain_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
