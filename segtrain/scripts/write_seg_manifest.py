"""v2: write the segmenter manifest the gym reads, {<stage>_s<seed>: absolute checkpoint path}.

Scans --dir for us_seg_<stage>_s<seed>.pt (stage in sonospade|physics|cut|cyclegan) that have a
_history.json and a .provenance.json sidecar whose md5 matches (a checkpoint still being written, or
one changed after its provenance, is left out and printed). The write is atomic (tmp + rename) and
the content is a pure function of the directory, so parallel per-stage runners can each call it.

  ./run.sh scripts/write_seg_manifest.py --dir outputs/v2            # -> outputs/v2/seg_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from us_seg import gym_bridge  # noqa: E402

PAT = re.compile(r"^us_seg_(sonospade|physics|cut|cyclegan)_s(\d+)\.pt$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/v2")
    ap.add_argument("--out", default=None, help="default <dir>/seg_manifest.json")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    out = args.out or os.path.join(d, "seg_manifest.json")
    man, left_out = {}, []
    for f in sorted(os.listdir(d)):
        m = PAT.match(f)
        if not m:
            continue
        ck = os.path.join(d, f)
        hist = os.path.splitext(ck)[0] + "_history.json"
        sc = ck + ".provenance.json"
        if not os.path.exists(hist) or not os.path.exists(sc):
            left_out.append(f"{f}: no history or provenance sidecar yet")
            continue
        if json.load(open(sc)).get("artifact", {}).get("md5") != gym_bridge.file_md5(ck):
            left_out.append(f"{f}: sidecar md5 != checkpoint md5")
            continue
        man[f"{m.group(1)}_s{int(m.group(2))}"] = ck
    tmp = out + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(dict(sorted(man.items())), fh, indent=2)
    os.replace(tmp, out)
    print(f"[seg-manifest] {out}: {len(man)} entries {sorted(man)}")
    for x in left_out:
        print(f"  left out: {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
