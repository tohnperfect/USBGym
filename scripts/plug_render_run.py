#!/usr/bin/env python3
"""plug_render_run.py: run an existing v2 / v2.1 script with a published renderer plugged into the loop.

v2.1 pluggability study, renderer slot (exploratory, opt-in). The target script runs UNCHANGED; before its
main() starts, usbg.plug.renderers.install(<renderer>) routes every render_lotus.render(..., "physics", cfg)
call (the one view_match.curvi_render and the landscape make) to the plugged renderer, seeded by cfg.seed,
so the sector geometry, texture stage, mask source, servo base, reward, crossed held-out design (6 held-out
patients, eval seed 999, 60 episodes) and every logged number are produced by the same code as the paper's
rows. Without this wrapper nothing changes anywhere.

    plug_render_run.py --renderer lotus_orig -- eval_view_ladder --rung ibvs_base --translator physics ...
    plug_render_run.py --renderer lotus_orig -- train_view_match_rl --translator physics ... --out X.json
    plug_render_run.py --renderer simus --calib-renderer physics -- eval_view_ladder --rung ibvs_base ...
    plug_render_run.py --renderer lotus_orig -- view_match_landscape --translators physics= ... --out X.json

Targets: eval_view_ladder, train_view_match_rl, view_match_landscape (scripts/<target>.py, main(argv)).

--calib-renderer R  render the servo-base calibration (train_view_match_rl.calibrate_ibvs / calibrate_cnn_base)
                    with renderer R instead. Allowed only with CT masks (USBG_LEARNED_SEG=0), where the linear
                    servo base reads only the CT label slices, so the calibrated base is identical whatever
                    renders the (discarded) frames; used for SIMUS, whose 1000 calibration frames would
                    otherwise cost hours. Recorded in the output's plug block.

Output: the target's --out JSON gains a "plug" block (renderer, citation, source hash, label map, deviations,
render-call timing) at the top level and inside "provenance", and provenance.analysis_status (and the
top-level analysis_status) become "exploratory, v2.1 pluggability". The wrapper refuses to start when --out
already exists (nothing is ever overwritten).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ("eval_view_ladder", "train_view_match_rl", "view_match_landscape")


def _out_of(argv):
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if "--" not in argv:
        raise SystemExit("usage: plug_render_run.py --renderer NAME [--calib-renderer NAME] -- TARGET [args...]")
    k = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renderer", required=True)
    ap.add_argument("--calib-renderer", default=None)
    ap.add_argument("--allow-existing-out", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv[:k])
    rest = argv[k + 1:]
    if not rest:
        raise SystemExit("no target script after --")
    target = os.path.splitext(os.path.basename(rest[0]))[0]
    if target not in TARGETS:
        raise SystemExit(f"target {rest[0]!r} not supported; expected one of {TARGETS}")
    targv = rest[1:]
    out = _out_of(targv)
    if out is None:
        raise SystemExit("the target needs --out (the wrapper post-processes it)")
    if os.path.exists(out) and not args.allow_existing_out:
        raise SystemExit(f"refusing to run: {out} exists (never overwrite)")

    from usbg import provenance as PROV
    from usbg.plug import renderers as PR
    kind = PR.install(args.renderer)
    calib = PR.canonical_name(args.calib_renderer) if args.calib_renderer else None
    learned = os.environ.get("USBG_LEARNED_SEG", "1").strip().lower() not in ("0", "false", "no", "")
    if calib is not None and learned:
        raise SystemExit("--calib-renderer needs CT masks (USBG_LEARNED_SEG=0): with a learned mask source the "
                         "servo base depends on the rendered frames")
    if target == "eval_view_ladder" and "--analysis-status" not in targv:
        targv += ["--analysis-status", PR.PLUG_STATUS]

    extra = {"target_script": os.path.join(HERE, target + ".py"), "calibration_renderer": calib or kind,
             "wrapper": os.path.abspath(__file__)}
    orig_collect = PROV.collect

    def collect(**kw):
        prov = orig_collect(**kw)
        prov["plug"] = PR.plug_info(kind, **extra)
        prov["analysis_status"] = PR.PLUG_STATUS
        return prov

    orig_write = PROV.write_provenance

    def write_provenance(*a, **kw):
        # checkpoint sidecars of a plugged run (train_view_match_rl) carry the same status and plug record
        kw["extra"] = dict(kw.get("extra") or {}, analysis_status=PR.PLUG_STATUS,
                           plug=PR.plug_info(kind, **extra))
        return orig_write(*a, **kw)

    PROV.collect = collect
    PROV.write_provenance = write_provenance
    sys.path.insert(0, HERE)
    import importlib
    mod = importlib.import_module(target)
    if target == "train_view_match_rl":
        mod.ANALYSIS_STATUS_V21 = PR.PLUG_STATUS            # this process only: a status label, no behavior
    if calib is not None:
        T = importlib.import_module("train_view_match_rl")
        for fname in ("calibrate_ibvs", "calibrate_cnn_base"):
            f = getattr(T, fname)

            def wrapped(*a, _f=f, **kw):
                with PR.plugged(calib):
                    return _f(*a, **kw)
            setattr(T, fname, wrapped)
    t0 = time.time()
    rc = mod.main(targv)
    wall = time.time() - t0
    if os.path.exists(out) and out.endswith(".json"):
        with open(out) as fh:
            d = json.load(fh)
        blk = PR.plug_info(kind, **extra, wrapper_wall_sec=round(wall, 1))
        d["plug"] = blk
        d["analysis_status"] = PR.PLUG_STATUS
        p = d.get("provenance")
        if isinstance(p, dict):
            p["plug"] = blk
            p["analysis_status"] = PR.PLUG_STATUS
        tmp = out + ".plugtmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1, default=float)
        os.replace(tmp, out)
        print(f"[plug] {kind}: render stats {PR.stats()} -> {out}")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
