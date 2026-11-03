#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): the admission gate of the SAMUS mask adapter (usbg.plug.samus_mask).

  USBG_SEAT=anatomical USBG_LEARNED_SEG=0 CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \\
      .venv/bin/python scripts/plug/admission_gate_samus.py --out outputs/v2_1/checks/gate/mask_samus.json

The same protocol as scripts/plug/admission_gate.py --adapter mask:sam (usbg.plug.admission; the pre-specified protocol), with kind=samus: (i) the U-Net-prompted SAMUS must be non-privileged (no
CT-label read, identical masks under a poisoned CT label) and the CT-prompted SAMUS must be DETECTED as
privileged; (ii) the pretraining corpus record (usbg.plug.samus_mask.PRETRAINING); (iii) CUDA-free, CPU / MPS
mask agreement on identical TRAIN frames and one TRAIN episode on each device. It reuses admission_gate.py's
own helpers; that file is not changed. TRAIN patients only (default s0011 frames, s0011 and s0344 episodes).

Output: --out (exists only if admitted; else <out minus .json>.failed.json and exit 1). Never overwrites.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ADAPTER = "mask:samus"


def gate_mask_samus(A):
    import admission_gate as AG
    from usbg.plug import admission as AD
    from usbg.plug import masks as PM
    w = A.samus_weights
    views = AD.train_views(A.case, n=A.n_views)
    mk_seg = lambda dev: PM.load_mask_source(f"kind=samus,weights={w},prompt=seg,prompt_ckpt={A.prompt_ckpt}",  # noqa: E731
                                             device=dev)
    comps = [AD.check_mask_source(mk_seg("cpu"), views, label="samus_unet (U-Net box prompts)")]
    comps.append(AD.check_mask_source(PM.load_mask_source(f"kind=samus,weights={w},prompt=ct", device="cpu"),
                                      views, label="samus_priv (CT-label box prompts)"))
    par = [AD.mask_parity(mk_seg, views, label="samus_unet"), AG._mask_episode(A, mk_seg, "samus_unet episode")]
    src = mk_seg("cpu")
    return comps, par, {"weights": {"path": w, "manifest": PM._manifest_entry(w)}, "adapter_describe": src.describe(),
                        "note": "the CT-prompted variant shares the network; its parity is covered by the U-Net-"
                                "prompted run"}


def main(argv=None):
    from usbg.benchmark import TRAIN_PATIENTS
    from usbg.plug import samus_mask as SM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="s0011", help="TRAIN patient of the frame checks")
    ap.add_argument("--cases", nargs="+", default=["s0011", "s0344"], help="TRAIN patients of the episode checks")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--prompt-ckpt", default=os.path.join(os.environ.get("USBG_SEG_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "segtrain")),
                                                          "outputs", "v2", "us_seg_sonospade_s0.pt"))
    ap.add_argument("--samus-weights", default=SM.DEFAULT_WEIGHTS)
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
    t0 = time.time()
    with AD.cuda_watch() as cw:
        comps, par, extra = gate_mask_samus(A)
    extra = dict(extra or {}, cuda_moves_total=int(cw["n"]), wall_s=round(time.time() - t0, 1),
                 peak_rss_mb=AD.peak_rss_mb(), argv=list(sys.argv), cases=list(A.cases), frame_case=A.case)
    body = AD.summarize(ADAPTER, "mask", comps, par, extra)
    body["checks"]["pretraining_corpus"] = dict(SM.PRETRAINING, key="samus")
    if cw["n"]:
        body["admitted"] = False
        body["checks"]["cuda_free"]["passed"] = False
    path = AD.write_result(A.out, body)
    bad = [c.get("component") for c in comps + par if not c.get("passed")]
    print(f"[gate] {ADAPTER}: {'ADMITTED' if body['admitted'] else 'NOT ADMITTED'} in {extra['wall_s']:.0f}s "
          f"(failed: {bad or 'none'}; cuda moves {cw['n']}) -> {path}")
    return 0 if body["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
