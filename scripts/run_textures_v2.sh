#!/usr/bin/env bash
# run_textures_v2.sh: leakage-free texture stages in the anatomical seat.
#
# Writes (never overwrites an old artifact; refuses to overwrite these v2 ones unless V2_OVERWRITE=1):
#   data/dataset/v2/pairs_med.npz             linear aligned (physics image, label) pairs, TRAIN-only cases,
#                                             accepted LIVER and KIDNEY windows (110 frames per case per organ)
#   data/dataset/v2/pairs_med_curvi.npz       the same frames warped to the measured Vitale sector
#   data/dataset/v2/pairs_med_curvi_aug.npz   + 2 view-augmented copies per frame (seed 0, rot 15, trans 0.10)
#   data/dataset/v2/s_us_grouped_curvi.pt     S_US (grouped 9 classes, 2000 iters) on the non-augmented curvi pairs
#   data/dataset/v2/sonospade_curvi.pt        SonoSPADE (1500 iters) on the 2x view-augmented curvi pairs
#   data/dataset/v2/cut_curvi.pt              CUT (800 iters) on the non-augmented curvi frames
#   data/dataset/v2/cyclegan_curvi.pt         CycleGAN (800 iters) on the non-augmented curvi frames
#   outputs/v2/textures/manifest.json         exactly {physics: null, cut, cyclegan, sonospade}
#   outputs/v2/textures/manifest_meta.json    md5s, cases, windows file, S_US, transitive lineage, git
#   outputs/v2/textures/*.log, *preview*.png  logs and previews
# Every artifact gets <artifact>.provenance.json (usbg.provenance; train cases, parents, real pool
# fingerprint, seat). The recipe is the v1 one except seat and patients: the same real pool (404 public
# Vitale TRAIN-split frames), the same iterations, seeds and configs, SonoSPADE on the view-augmented
# pairs and CUT/CycleGAN on the non-augmented frames. CUT/CycleGAN read the curvi pairs npz directly
# (train_texture_gan.load_sim_frames gives the identical frames the v1 1-clip wrapper gave).
#
# Usage: USBG_SEAT=anatomical bash scripts/run_textures_v2.sh      (about 25 min on MPS, idle machine)
# SMOKE=1: the same pipeline with 3 frames per (case, organ) and a few iterations, written under
#   outputs/v2/_smoke/C/runbook/ (never data/dataset/v2/); about 1 min.
# Knobs (env): PAIRS_MIN_WIN (default 2: a (case, organ) with fewer screened windows is skipped and logged;
#   with the current windows file this skips kidney for s0058 and s0522, which have 1 kidney window each),
#   V2_OVERWRITE=1 to replace existing v2 outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${USBG_SEAT:-}" != "anatomical" ]; then
  echo "refusing to run: export USBG_SEAT=anatomical explicitly (got '${USBG_SEAT:-unset}')" >&2
  exit 2
fi
export PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 USBG_SEAT=anatomical
PY=.venv/bin/python
POOL=data/real/kaggle_vitale/pool
TRAIN8="s0011 s0058 s0344 s0461 s0477 s0511 s0522 s0703"
PAIRS_MIN_WIN="${PAIRS_MIN_WIN:-2}"
D=data/dataset/v2
O=outputs/v2/textures
PAIRS_N=110; S_US_IT=2000; SPADE_IT=1500; GAN_IT=800
if [ "${SMOKE:-0}" = "1" ]; then
  D=outputs/v2/_smoke/C/runbook/data; O=outputs/v2/_smoke/C/runbook
  PAIRS_N=3; S_US_IT=5; SPADE_IT=3; GAN_IT=3; V2_OVERWRITE=1
  echo "SMOKE run -> $D, $O"
fi
mkdir -p "$D" "$O"

OUTS="$D/pairs_med.npz $D/pairs_med_curvi.npz $D/pairs_med_curvi_aug.npz $D/s_us_grouped_curvi.pt \
$D/sonospade_curvi.pt $D/cut_curvi.pt $D/cyclegan_curvi.pt $O/manifest.json $O/manifest_meta.json"
if [ "${V2_OVERWRITE:-0}" != "1" ]; then
  for f in $OUTS; do
    if [ -e "$f" ]; then echo "refusing to overwrite $f (V2_OVERWRITE=1 to replace v2 outputs)" >&2; exit 3; fi
  done
fi

# 0. guards: TRAIN-only cases, seat, windows file, bladder-free caches, public pool
$PY - <<PYEOF
import os, sys
from usbg.benchmark import TEST_PATIENTS, TRAIN_PATIENTS
from usbg.seatmode import seat_mode
from usbg import windows as W, provenance as P
cases = "$TRAIN8".split()
assert seat_mode() == "anatomical", seat_mode()
bad = sorted(set(cases) & set(TEST_PATIENTS))
if bad: sys.exit(f"LEAK: held-out {bad} in the texture cases")
extra = sorted(set(cases) - set(TRAIN_PATIENTS))
if extra: sys.exit(f"not TRAIN patients: {extra}")
assert os.path.exists(W.WINDOWS_PATH), W.WINDOWS_PATH
miss = [c for c in cases if not os.path.exists(f"data/cache_anat/{c}.npz")]
if miss: sys.exit(f"missing bladder-free caches {miss} (scripts/build_anat_cache.py)")
assert not P.is_private_path("$POOL"), "private pool"
n, fp = P.pool_fingerprint("$POOL")
print(f"guards OK: cases {cases}; windows {W.WINDOWS_PATH} md5 {W.file_md5()}; pool {n} frames {fp}")
PYEOF

# 1. linear aligned pairs at the accepted liver + kidney windows (110 per case per organ)
$PY scripts/make_dataset.py --cases $TRAIN8 --target liver --pairs-organs liver kidney \
    --pairs-min-windows "$PAIRS_MIN_WIN" --pairs-press-mm 0 --image-size 128 --renderer physics \
    --pairs-out $D/pairs_med.npz --pairs-n $PAIRS_N --pairs-dr full --seed 0 --pairs-only \
    2>&1 | tee $O/pairs_med.log

# 2. curvilinear warp (canonical and 2x view-augmented)
$PY scripts/curvi_warp.py --in-npz $D/pairs_med.npz --out-npz $D/pairs_med_curvi.npz --n-aug 0 \
    --preview $O/pairs_med_curvi_preview.png 2>&1 | tee $O/curvi_warp.log
$PY scripts/curvi_warp.py --in-npz $D/pairs_med.npz --out-npz $D/pairs_med_curvi_aug.npz --n-aug 2 --seed 0 \
    2>&1 | tee -a $O/curvi_warp.log

# 3. S_US (grouped) on the non-augmented curvi pairs (session split, seed 0)
$PY scripts/train_segmenter.py --pairs $D/pairs_med_curvi.npz --group --iters $S_US_IT --seed 0 \
    --out $D/s_us_grouped_curvi.pt --preview $O/s_us_curvi_preview.png 2>&1 | tee $O/s_us_grouped_curvi.log

# 4. SonoSPADE on the 2x view-augmented curvi pairs with the frozen S_US
$PY scripts/train_texture_gan.py --variant sonospade --pairs $D/pairs_med_curvi_aug.npz --real-dir $POOL \
    --s-us $D/s_us_grouped_curvi.pt --config configs/sonospade_curvi.yaml --seed 0 --iters $SPADE_IT \
    --out $D/sonospade_curvi.pt --preview $O/sonospade_curvi_preview.png 2>&1 | tee $O/sonospade_curvi.log

# 5. CUT and CycleGAN on the non-augmented curvi frames (same real pool)
for V in cut cyclegan; do
  $PY scripts/train_texture_gan.py --variant $V --sim-npz $D/pairs_med_curvi.npz --real-dir $POOL \
      --iters $GAN_IT --seed 0 --out $D/${V}_curvi.pt --preview $O/${V}_curvi_preview.png 2>&1 | tee $O/${V}_curvi.log
done

# 6. manifest + lineage audit (no held-out case anywhere in the transitive training sets)
$PY - <<PYEOF
import json, os, sys
from usbg import provenance as P, windows as W
D, O = "$D", "$O"
man = {"physics": None, "cut": f"{D}/cut_curvi.pt", "cyclegan": f"{D}/cyclegan_curvi.pt",
       "sonospade": f"{D}/sonospade_curvi.pt"}
for k, p in man.items():
    if p and not os.path.exists(P.sidecar_path(p)):
        sys.exit(f"{p}: no provenance sidecar")
rep = P.scan([D])
if rep["findings"]:
    sys.exit("provenance findings:\n  " + "\n  ".join(rep["findings"]))
meta = {"manifest": f"{O}/manifest.json", "seat_mode": "anatomical", "cases": "$TRAIN8".split(), "pairs_min_windows": int("$PAIRS_MIN_WIN"),
                "real_pool": "$POOL", "windows": P.windows_ref(),
                "md5": {k: P.md5(v) for k, v in man.items() if isinstance(v, str)},
                "s_us": {"path": f"{D}/s_us_grouped_curvi.pt", "md5": P.md5(f"{D}/s_us_grouped_curvi.pt")},
                "transitive_cases": {os.path.relpath(k): v for k, v in rep["transitive"].items()},
                "git": P.git_info()}
with open(f"{O}/manifest.json", "w") as fh:                  # exactly {physics, cut, cyclegan, sonospade}
    json.dump(man, fh, indent=2)
with open(f"{O}/manifest_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"wrote {O}/manifest.json (+ manifest_meta.json); lineage clean for {len(rep['records'])} artifacts")
PYEOF
echo "textures v2 DONE"
