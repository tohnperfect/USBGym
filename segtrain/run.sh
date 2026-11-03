#!/usr/bin/env bash
# Run a us_seg script with the USB Gym environment, with both src trees (the gym's src/ and this folder's
# src/) on PYTHONPATH and MPS CPU-fallback enabled. Usage, from this folder:  ./run.sh scripts/build_dataset.py --smoke
# The gym root is the folder above this one (segtrain/ is vendored in the USB Gym repository); USBG_GYM_ROOT
# overrides it. The interpreter is USBG_PYTHON, else <gym>/.venv/bin/python, else python3 on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM="${USBG_GYM_ROOT:-$(cd "$HERE/.." && pwd)}"
PY="${USBG_PYTHON:-$GYM/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "error: no Python found (set USBG_PYTHON, or create $GYM/.venv)" >&2
  exit 1
fi
if [[ ! -d "$GYM/src/usbg" ]]; then
  echo "error: no src/usbg under $GYM (set USBG_GYM_ROOT to the USB Gym repository root)" >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$GYM/src:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
