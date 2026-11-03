#!/usr/bin/env python3
"""Organ-identity table (v2): per-organ same-AND-correct cross-domain class, 4 stages x 3 seeds, v2 segmenters.

Real side per organ (each a prediction from a model that never trained on that frame):
  liver / GB / vessel / spleen <- Vitale, 60 frames = OOF over the train frames + the deployment
        model on the fixed holdout (consistency_<t>_s<n>_vitale.json; --real-side vitale)
        or holdout-only (consistency_<t>_s<n>_vitaleho.json; --real-side vitaleho)
  kidney <- KidneyUS patient-disjoint holdout, 107 frames (consistency_<t>_s<n>_kidneyus.json)
  liver (independent check) <- AUL 732 frames (consistency_<t>_s<n>_aul.json), reported separately
Sim side: the deployment model on the TEST-patient sim-eval build (consistency.py --sim-all).

Organs outside the v2 sim target set (spleen; kidney if it is dropped from that set) are still scored but flagged
'rarely in view'. Population std over seeds. Missing files are reported, never guessed.

  ./run.sh scripts/agg_tablev_v2.py --dir outputs/v2 [--real-side vitale|vitaleho] [--organs ...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

STAGES = ["sonospade", "physics", "cut", "cyclegan"]
SEEDS = [0, 1, 2]
PRETTY = {"sonospade": "SonoSPADE", "physics": "physics", "cut": "CUT", "cyclegan": "CycleGAN"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ok(rep, organ):
    if rep is None:
        return None
    m = rep.get("per_class", {}).get(organ)
    return None if m is None else bool(m["modal_is_correct"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/v2")
    ap.add_argument("--real-side", choices=["vitale", "vitaleho"], default="vitale")
    ap.add_argument("--organs", default="liver,gallbladder,vessel,kidney,spleen")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seeds", default="0,1,2", help="comma list (smoke: 0)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--require-seat", default="anatomical",
                    help="every consistency JSON must carry this provenance.seat_mode ('' to skip)")
    args = ap.parse_args(argv)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stages = [t for t in args.stages.split(",") if t.strip()]
    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    vit_organs = [o for o in organs if o != "kidney"]

    grid, missing, aul = {}, [], {}
    for t in stages:
        grid[t], aul[t] = {}, []
        for s in seeds:
            vf = os.path.join(args.dir, f"consistency_{t}_s{s}_{args.real_side}.json")
            kf = os.path.join(args.dir, f"consistency_{t}_s{s}_kidneyus.json")
            af = os.path.join(args.dir, f"consistency_{t}_s{s}_aul.json")
            vit, kid, au = load(vf), load(kf), load(af)
            for f, r in ((vf, vit), (kf, kid)):
                if r is None:
                    missing.append(os.path.basename(f))
            cells = {o: ok(vit, o) for o in vit_organs}
            if "kidney" in organs:
                cells["kidney"] = ok(kid, "kidney")
            grid[t][s] = cells
            aul[t].append(ok(au, "liver"))

    summary = {}
    print(f"same-AND-correct class; real side = {args.real_side}; kidney = KidneyUS holdout\n")
    for t in stages:
        counts = []
        for s in seeds:
            c = grid[t][s]
            if any(v is None for v in c.values()):
                print(f"{PRETTY[t]:<10} s{s}  pending")
                continue
            counts.append(sum(bool(v) for v in c.values()))
            print(f"{PRETTY[t]:<10} s{s}  " + " ".join(f"{o[:6]}={'Y' if c[o] else '.'}" for o in organs)
                  + f"  -> {counts[-1]}/{len(organs)}")
        if len(counts) == len(seeds):
            per_organ = {o: sum(1 for s in seeds if grid[t][s][o]) for o in organs}
            summary[t] = {"per_organ_of_3": per_organ, "per_seed_counts": counts,
                          "count_mean": round(statistics.mean(counts), 3),
                          "count_pstd": round(statistics.pstdev(counts), 3),
                          "aul_liver_of_3": sum(1 for v in aul[t] if v), "aul_pending": aul[t].count(None)}
    seats, used = {}, []
    for t in stages:
        for s in seeds:
            for src in (args.real_side, "kidneyus", "aul"):
                r = load(os.path.join(args.dir, f"consistency_{t}_s{s}_{src}.json"))
                if r is not None:
                    seats.setdefault((r.get("provenance") or {}).get("seat_mode"), []).append(f"{t}_s{s}_{src}")
                    used.append(f"consistency_{t}_s{s}_{src}.json")
    if args.require_seat and seats and set(seats) != {args.require_seat}:
        raise SystemExit(f"[guard] mixed or wrong seat modes in the inputs: "
                         f"{ {k: len(v) for k, v in seats.items()} } (required {args.require_seat!r})")
    if missing:
        print(f"\nMISSING {len(set(missing))}: " + ", ".join(sorted(set(missing))[:6]) + (" ..." if len(set(missing)) > 6 else ""))
    if summary:
        print("\n% ---- organ-identity table rows (v2) ----")
        for t, r in summary.items():
            cells = " & ".join(f"${r['per_organ_of_3'][o]}/{len(seeds)}$" for o in organs)
            print(f"{PRETTY[t]} & {cells} & ${r['count_mean']:.1f} \\pm {r['count_pstd']:.1f}$ \\\\")
        out = args.json_out or os.path.join(args.dir, f"tablev_v2_{args.real_side}.json")
        with open(out, "w") as f:
            json.dump({"real_side": args.real_side, "organs": organs, "std": "population",
                       "seeds": seeds, "seat_mode": next(iter(seats)) if len(seats) == 1 else None,
                       "inputs": sorted(used), "missing": sorted(set(missing)),
                       "metric": "per organ: modal predicted class == true class in BOTH sim (TEST-patient "
                                 "sim-eval build) and real; kidney real side = KidneyUS holdout",
                       "stages": summary}, f, indent=2)
        print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
