#!/usr/bin/env python3
"""v2.1 pluggability (EXPLORATORY): run scripts/eval_view_ladder.py with a PLUGGED mask source.

  USBG_SEAT=anatomical USBG_LEARNED_SEG=1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 \\
  .venv/bin/python scripts/plug/eval_plug_mask.py --mask-plug kind=tv,ckpt=<...>.pt -- \\
      --rung ibvs_base --translator sonospade --texture-ckpt <...> --calib-seed 0 --contact on off \\
      --out outputs/v2_1/plug/seg/ladder/ibvs_base_sonospade_lraspp_mbv3_s0.json

What it does, in order:
  1. usbg.plug.masks.bind_mask_source(SPEC): builds the adapter, exports US_SEG_CKPT=<its weights file>
     (the run guards and usbg.provenance.mask_info read it), installs the process-local CT-label tap for a
     CT-prompted (privileged) SAM / MedSAM, and calls view_match.set_mask_source(adapter);
  2. calls eval_view_ladder.main(argv) UNCHANGED (same rungs, same crossed held-out design: 6 held-out
     patients, eval seed 999, 60 crossed episodes, contact on/off), adding --ext-log and
     --analysis-status "exploratory, v2.1 pluggability" when absent;
  3. adds a "plug" block (the adapter's description and its measured per-frame cost) to the output JSON
     and to its provenance; with a CT-prompted (privileged) mask it also sets "privileged": true (the
     rung's own flag is kept as "privileged_rung") and marks rung_class as privileged.
Only this process is affected; the v2 / v2.1 queues never run this script. A policy rung evaluates a
policy trained with the stage-matched U-Net under the plugged mask, so pass --allow-mismatch (the
mismatch is recorded in provenance.mismatch_vs_run); the policy keeps its saved servo base.

Every output is refused if it already exists (never overwrite).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"


def _arg(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        return argv[i + 1] if i + 1 < len(argv) else None
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__ or "usage: eval_plug_mask.py --mask-plug SPEC [--method-key K] -- <eval_view_ladder args>")
        return 0
    if "--" not in argv:
        raise SystemExit("usage: eval_plug_mask.py --mask-plug SPEC [--method-key K] -- <eval_view_ladder args>")
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-plug", required=True)
    ap.add_argument("--method-key", default=None)
    args = ap.parse_args(argv[:k])
    ev = argv[k + 1:]

    if os.environ.get("USBG_LEARNED_SEG", "").strip() != "1":
        raise SystemExit("export USBG_LEARNED_SEG=1: a plugged mask source is a learned-mask run")
    out = _arg(ev, "--out")
    if not out:
        raise SystemExit("eval_view_ladder needs --out")
    out = os.path.abspath(out)
    side = [out, out + ".tmp", (out[:-5] if out.endswith(".json") else out) + ".traj.npz",
            (out[:-5] if out.endswith(".json") else out) + ".cnn_base.pt"]
    have = [p for p in side if os.path.exists(p)]
    if have:
        raise SystemExit(f"refusing to overwrite existing output(s): {have}")
    if "--ext-log" not in ev:
        ev.append("--ext-log")
    if "--analysis-status" not in ev:
        ev += ["--analysis-status", ANALYSIS_STATUS]
    elif _arg(ev, "--analysis-status") != ANALYSIS_STATUS:
        raise SystemExit(f"--analysis-status must be {ANALYSIS_STATUS!r}")

    from usbg.plug import masks as PM
    t0 = time.time()
    src = PM.bind_mask_source(args.mask_plug)
    load_sec = time.time() - t0
    print(f"[plug] mask source {src.name} ({'PRIVILEGED' if src.privileged else 'non-privileged'}) "
          f"weights {src.ckpt_path} loaded in {load_sec:.1f}s; US_SEG_CKPT={os.environ['US_SEG_CKPT']}")

    import eval_view_ladder as EVL
    rc = EVL.main(ev)

    plug = dict(src.describe(), spec=args.mask_plug, method_key=args.method_key, load_sec=round(load_sec, 2),
                timing=src.timing())
    with open(out) as fh:
        res = json.load(fh)
    res["plug"] = plug
    res["analysis_status"] = ANALYSIS_STATUS
    res["mask_privileged"] = bool(src.privileged)
    if src.privileged:                    # a CT-prompted mask makes any rung privileged; keep the rung's own flag
        res["privileged_rung"] = bool(res.get("privileged"))
        res["privileged"] = True
        res["rung_class"] = f"{res.get('rung_class') or res.get('rung')} + PRIVILEGED mask ({plug.get('privileged_reason')})"
    res.setdefault("provenance", {})["plug"] = plug
    res["provenance"]["analysis_status"] = ANALYSIS_STATUS
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    os.replace(tmp, out)
    print(f"[plug] {src.name}: {json.dumps(plug['timing'])} -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
