#!/usr/bin/env bash
# v2 segmenter cascade: 4 texture stages x 3 seeds, leakage-free.
#
#   sim   : 13 TRAIN patients only (usbg.benchmark), anatomical seat (USBG_SEAT=anatomical: screened
#           acoustic windows, gel coupling, and the bladder-free KIDNEY cache that the gym's
#           volume.load_cache reads from data/cache_anat automatically), per-frame DR, CT-oracle labels
#           (USBG_LEARNED_SEG=0). Seat organs liver,kidney (spleen is not a target; it stays a class).
#           Kidney is skipped (logged) for s0058 s0522 s0727 (SKIP_PAIRS); never replaced by liver.
#   real  : Vitale non-holdout frames (fixed data/real/vitale_split_v2.npz, group g,b: 42/18) +
#           KidneyUS partial-label train frames (427)
#   held  : Vitale holdout (never trained on by ANY v2 model), KidneyUS 107 holdout, AUL 732
#   eval  : per stage+seed a sim-eval build on the 6 TEST patients (same translator), the sim side of
#           the organ-identity table (scripts/agg_tablev_v2.py)
#
# Per stage+seed: build(train) -> build(test-eval) -> manifest check -> train deploy -> seg manifest +
# check -> K-fold OOF over the Vitale train frames (+deploy fills the holdout) -> consistency x4 ->
# eval_seg. Resumable; every skip is printed. One MPS job at a time per invocation; run at most 2-3
# invocations in parallel (one per stage).
#
# Translators: the gym's v2 texture manifest (TEX_MANIFEST, default
# $GYM/outputs/v2/textures/manifest.json: {physics: null, cut, cyclegan, sonospade}). TX_SONOSPADE /
# TX_CUT / TX_CYCLEGAN override per stage. The D3-leaky exp_real_*_curvi.pt are refused (here and in
# build_dataset.py).
#
# Optional env: SEEDS="0 1 2" SEAT_ORGANS=liver,kidney NPOSES=27 NPOSES_EVAL=12 FOLDS=6 CV_ITERS=2500
#   TRAIN_ITERS=4000 SPLIT=data/real/vitale_split_v2.npz NN=data/nnunet_v2 OUT=outputs/v2
#   ID_TRAIN=100 ID_EVAL=200 SKIP_PAIRS=s0058:kidney,s0522:kidney,s0727:kidney MAX_CASES=0 DRY=1
# Usage: scripts/run_seg_v2.sh physics          # one stage, all seeds
#        scripts/run_seg_v2.sh sonospade cut    # several, serially
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GYM="${USBG_GYM_ROOT:-$(cd .. && pwd)}"
SEEDS=${SEEDS:-"0 1 2"}; SEAT_ORGANS=${SEAT_ORGANS:-liver,kidney}
NPOSES=${NPOSES:-27}; NPOSES_EVAL=${NPOSES_EVAL:-12}
FOLDS=${FOLDS:-6}; CV_ITERS=${CV_ITERS:-2500}; TRAIN_ITERS=${TRAIN_ITERS:-4000}
SPLIT=${SPLIT:-data/real/vitale_split_v2.npz}
NN=${NN:-data/nnunet_v2}; OUT=${OUT:-outputs/v2}
ID_TRAIN=${ID_TRAIN:-100}; ID_EVAL=${ID_EVAL:-200}
SKIP_PAIRS=${SKIP_PAIRS-s0058:kidney,s0522:kidney,s0727:kidney}
MAX_CASES=${MAX_CASES:-0}
TEX_MANIFEST=${TEX_MANIFEST:-$GYM/outputs/v2/textures/manifest.json}
SEL="${*:-sonospade physics cut cyclegan}"
export USBG_LEARNED_SEG=0
unset US_SEG_CKPT || true
mkdir -p "$NN" "$OUT"

[ "${USBG_SEAT:-}" = anatomical ] || { echo "refusing: export USBG_SEAT=anatomical (got '${USBG_SEAT:-}')"; exit 2; }
[ -z "${USBG_CACHE_DIR:-}" ] || { echo "refusing: USBG_CACHE_DIR is obsolete (the gym redirects data/cache -> data/cache_anat); unset it"; exit 2; }
[ -f "$SPLIT" ] || { echo "refusing: $SPLIT missing (run scripts/make_vitale_split.py --mode group --holdout-groups g,b once)"; exit 2; }
[ -d "$GYM/data/cache_anat" ] || { echo "refusing: $GYM/data/cache_anat missing (gym scripts/build_anat_cache.py)"; exit 2; }

run() { echo "+ $*"; if [ "${DRY:-0}" = 0 ]; then "$@"; fi; }
skip() { echo "  !! SKIP $1 (exists: $2)"; }
base() { case "$1" in sonospade) echo 1 ;; physics) echo 2 ;; cut) echo 3 ;; cyclegan) echo 4 ;; *) echo "bad stage $1" >&2; exit 2 ;; esac; }
tx() {   # stage -> translator arg for build_dataset (abs path, or none for physics)
  local t=$1 ov=""
  case "$t" in
    physics) echo none; return ;;
    sonospade) ov=${TX_SONOSPADE:-} ;; cut) ov=${TX_CUT:-} ;; cyclegan) ov=${TX_CYCLEGAN:-} ;;
  esac
  if [ -z "$ov" ]; then
    [ -f "$TEX_MANIFEST" ] || { echo "no TX override for $t and no texture manifest $TEX_MANIFEST" >&2; exit 2; }
    ov=$(./run.sh -c "import sys; from us_seg import gym_bridge as g; v=g.v2_texture_manifest(sys.argv[1]).get(sys.argv[2]); print(v or '')" "$TEX_MANIFEST" "$t")
    [ -n "$ov" ] || { echo "texture manifest $TEX_MANIFEST gives no usable checkpoint for $t (null entry, or see the error above)" >&2; exit 2; }
  fi
  case "$(basename "$ov")" in exp_real_*_curvi.pt) echo "refusing legacy translator $ov for $t" >&2; exit 2 ;; esac
  case "$ov" in /*) ;; *) echo "translator path for $t must be absolute: $ov" >&2; exit 2 ;; esac
  [ -f "$ov" ] || { echo "translator for $t missing: $ov" >&2; exit 2; }
  echo "$ov"
}
SKIPARG=(); [ -n "$SKIP_PAIRS" ] && SKIPARG=(--skip-pairs "$SKIP_PAIRS")
MAXARG=(); [ "$MAX_CASES" != 0 ] && MAXARG=(--max-cases "$MAX_CASES")

for t in $SEL; do
  TX=$(tx "$t"); b=$(base "$t")
  for s in $SEEDS; do
    id=$((ID_TRAIN + b * 10 + s)); eid=$((ID_EVAL + b * 10 + s))
    ds="$NN/Dataset${id}_USOrgansV2_${t}_s${s}"; eds="$NN/Dataset${eid}_USOrgansV2eval_${t}_s${s}"
    ck="$OUT/us_seg_${t}_s${s}.pt"; cv="$OUT/crossval_${t}_s${s}.json"; oof="$OUT/real_oof_${t}_s${s}.npz"
    echo ""; echo "===== ${t} seed ${s}  (translator=${TX}) $(date '+%F %T') ====="

    if [ -f "$ds/dataset.json" ]; then skip build "$ds"; else
      run ./run.sh scripts/build_dataset.py --out "$NN" --dataset-id "$id" --name "USOrgansV2_${t}_s${s}" \
        --translator "$TX" --cases train --role train --seat-organs "$SEAT_ORGANS" --strict-seat \
        ${SKIPARG[@]+"${SKIPARG[@]}"} ${MAXARG[@]+"${MAXARG[@]}"} --require-seat anatomical --n-poses "$NPOSES" --seed "$s"; fi
    if [ -f "$eds/dataset.json" ]; then skip build-eval "$eds"; else
      run ./run.sh scripts/build_dataset.py --out "$NN" --dataset-id "$eid" --name "USOrgansV2eval_${t}_s${s}" \
        --translator "$TX" --cases test --role eval --seat-organs "$SEAT_ORGANS" --strict-seat \
        ${SKIPARG[@]+"${SKIPARG[@]}"} ${MAXARG[@]+"${MAXARG[@]}"} --require-seat anatomical --n-poses "$NPOSES_EVAL" \
        --seed "$((1000 + s))" --no-real-test; fi
    run ./run.sh scripts/check_manifest.py --datasets "$ds" --eval-datasets "$eds" --require-seat anatomical

    if [ -f "$ck" ]; then skip train "$ck"; else
      run ./run.sh scripts/train.py --dataset "$ds" --iters "$TRAIN_ITERS" --seed "$s" --kidneyus \
        --vitale-split "$SPLIT" --forbid-test-patients --out "$ck"; fi
    run ./run.sh scripts/write_seg_manifest.py --dir "$OUT"
    run ./run.sh scripts/check_manifest.py --histories "${ck%.pt}_history.json" --vitale-split "$SPLIT" \
      --seg-manifest "$OUT/seg_manifest.json" --require-seat anatomical

    if [ -f "$oof" ]; then skip crossval "$oof"; else
      run ./run.sh scripts/crossval_real.py --dataset "$ds" --folds "$FOLDS" --iters "$CV_ITERS" --seed "$s" \
        --kidneyus --vitale-split "$SPLIT" --forbid-test-patients --deploy-ckpt "$ck" \
        --out "$cv" --oof-out "$oof"; fi

    for src in vitale vitaleho kidneyus aul; do
      c="$OUT/consistency_${t}_s${s}_${src}.json"
      if [ -f "$c" ]; then skip "consistency-$src" "$c"; continue; fi
      case "$src" in
        vitale)   real=(--oof "$oof") ;;
        vitaleho) real=(--real-npz data/real/vitale_holdout_v2.npz) ;;
        kidneyus) real=(--real-npz data/real/kidneyus_holdout_test.npz) ;;
        aul)      real=(--real-npz data/real/aul_liver_test.npz) ;;
      esac
      run ./run.sh scripts/consistency.py --ckpt "$ck" --dataset "$eds" --sim-all --seed "$s" "${real[@]}" --out "$c"
    done

    ev="$OUT/eval_${t}_s${s}.json"
    if [ -f "$ev" ]; then skip eval "$ev"; else
      run ./run.sh scripts/eval_seg.py --ckpt "$ck" --dataset "$eds" --sim-all \
        --real-npz data/real/vitale_holdout_v2.npz --real-npz data/real/aul_liver_test.npz \
        --real-npz data/real/kidneyus_holdout_test.npz --out "$ev"; fi
  done
done
echo ""; echo "done. aggregate: ./run.sh scripts/agg_tablev_v2.py --dir $OUT --organs liver,gallbladder,vessel,kidney,spleen"
