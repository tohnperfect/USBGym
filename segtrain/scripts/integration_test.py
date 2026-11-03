"""Integration acceptance: is segment() a drop-in for curvi_render's mask in the controller?

The controller (usbg view_match RL) uses curvi_render's 2nd return (a CT-derived organ label) as the
mask source for BOTH the seg-anchored reward (view_match.seg_organ_iou / combined_score) and the
pose-free image-servoing base (ibvs_features in scripts/train_view_match_rl.py). This test feeds those
same consumers with segment(render) instead of the CT label and checks they run and behave sanely:

  * self-consistency: seg_organ_iou(segment(v), segment(v)) == 1.0 for a view v
  * validity: masks are int in the repo ontology {0,1,2,3,4,7}, same shape as the render
  * responsiveness: reward is high when current≈target and drops as the view is perturbed
  * (if importable) the real ibvs_features + image_base_twist run on segment() masks

Passing this = the segmenter can replace the CT-oracle mask with no controller code change
(cleanest swap: view_match.py:85 -> return tex, SEG.segment(tex)).

  ./run.sh scripts/integration_test.py --ckpt outputs/us_seg_6cls.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import gym_bridge  # noqa: E402
from us_seg.infer import load_segmenter  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="outputs/us_seg_6cls.pt")
    ap.add_argument("--translator", default="sonospade")
    args = ap.parse_args(argv)

    gym_bridge.ensure_usbg_importable()
    from usbg import volume as V
    from usbg import view_match as VM
    from usbg import goal_rl as G
    from usbg import render_lotus as R
    from usbg import texture_gan as TG

    seg = load_segmenter(args.ckpt)
    dev = R.get_device()
    tr = TG.load_translator(gym_bridge.translator_ckpt(args.translator), device=dev)
    lv = V.load_cache(gym_bridge.cache_cases()[0])
    rng = np.random.default_rng(0)

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    # render a target view and a MILD-perturbation current view (liver seat + small perturbation,
    # so the two views genuinely share organ overlap to measure)
    seat = V.LIVER if lv.has(V.LIVER) else V.KIDNEY
    Wfv, T_tgt, T_cur = G.sample_setup(lv, seat, rng, lateral_m=0.006, axial_m=0.004,
                                       fan_deg=5.0, roll_deg=2.0)
    cfg = R.domain_randomize(R.RenderConfig(), rng)
    img_tgt, ct_tgt = VM.curvi_render(lv, T_tgt, Wfv, tr, cfg=cfg, size=128)
    img_cur, ct_cur = VM.curvi_render(lv, T_cur, Wfv, tr, cfg=cfg, size=128)

    seg_tgt = seg.segment(img_tgt)
    seg_cur = seg.segment(img_cur)

    print("\n== drop-in mask validity ==")
    check("shape matches render", seg_tgt.shape == img_tgt.shape == (128, 128),
          f"{seg_tgt.shape}")
    check("dtype int64", seg_tgt.dtype == np.int64, str(seg_tgt.dtype))
    ok_ont = set(np.unique(seg_tgt)).issubset({0, 1, 2, 3, 4, 7}) and \
             set(np.unique(seg_cur)).issubset({0, 1, 2, 3, 4, 7})
    check("repo ontology {0,1,2,3,4,7}", ok_ont,
          f"tgt={np.unique(seg_tgt).tolist()} cur={np.unique(seg_cur).tolist()}")

    print("\n== seg-anchored reward (view_match.seg_organ_iou / combined_score) ==")
    iou_self = VM.seg_organ_iou(seg_tgt, seg_tgt)
    check("self-consistency IoU==1", abs(iou_self - 1.0) < 1e-6, f"{iou_self:.4f}")
    iou_seg = VM.seg_organ_iou(seg_cur, seg_tgt)
    iou_ct = VM.seg_organ_iou(ct_cur, ct_tgt)
    # the meaningful property: segment() masks reproduce the CT-oracle reward signal
    check("segmenter reward tracks CT-oracle (|Δ|<0.35)", abs(iou_seg - iou_ct) < 0.35,
          f"seg={iou_seg:.3f} vs ct-oracle={iou_ct:.3f}")
    cs = VM.combined_score(img_cur, seg_cur, img_tgt, seg_tgt)
    check("combined_score runs, in [0,1]", 0.0 <= cs <= 1.0, f"{cs:.3f}")

    # responsiveness: a MILD-perturbation view (near) should score higher than LARGE-perturbation
    # views (far). 'near' = the mild-perturbation current view above (seg_cur vs seg_tgt), NOT a
    # self-comparison. Average a few far draws to reduce render noise.
    iou_near = iou_seg  # seg_organ_iou(seg_cur, seg_tgt), mild perturbation
    fars = []
    for _ in range(3):
        _, _, T_far = G.sample_setup(lv, seat, rng, lateral_m=0.05, axial_m=0.02,
                                     fan_deg=45.0, roll_deg=15.0)
        img_far, _ = VM.curvi_render(lv, T_far, Wfv, tr, cfg=cfg, size=128)
        fars.append(VM.seg_organ_iou(seg.segment(img_far), seg_tgt))
    iou_farv = float(np.mean(fars))
    check("reward responsive (near > far)", iou_near > iou_farv,
          f"near={iou_near:.3f} far(mean of 3)={iou_farv:.3f}")

    print("\n== pose-free IBVS base (scripts/train_view_match_rl.py) ==")
    try:
        import importlib.util
        spath = gym_bridge.gym_path("scripts", "train_view_match_rl.py")
        spec = importlib.util.spec_from_file_location("tvmrl", spath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        feats = mod.ibvs_features(seg_cur, seg_tgt, getattr(mod, "_IBVS", {}).get("rich", False)
                                  if isinstance(getattr(mod, "_IBVS", None), dict) else False)
        check("ibvs_features runs on segment() masks", feats is not None and np.isfinite(feats).all(),
              f"len={len(feats)}")
        tw = mod.image_base_twist(seg_cur, seg_tgt)
        check("image_base_twist runs, finite (6,)", tw.shape == (6,) and np.isfinite(tw).all(),
              f"{np.round(tw, 3).tolist()}")
    except Exception as e:  # noqa: BLE001
        check("controller module import", False, f"could not import script module: {e}")

    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n=== {npass}/{len(checks)} checks passed ===")
    print("segment() is a drop-in mask source for the controller." if npass == len(checks)
          else "SOME CHECKS FAILED, see above.")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
