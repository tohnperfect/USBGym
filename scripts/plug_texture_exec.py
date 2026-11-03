#!/usr/bin/env python3
"""Run an EXISTING script with the v2.1 texture plug-ins registered (exploratory, opt-in).

    python scripts/plug_texture_exec.py <script.py> [args ...]

The wrapper imports usbg.plug_texture (which registers the fastcut, dclgan and unsb trainers and the unsb
and histmatch translator loaders), widens the --variant / --translator choice lists of the target script
in this process only (usbg.plug_texture.widen_choices), then runs the target as __main__ with its own
directory first on sys.path, exactly like `python <script.py> ...`. No file of the target is edited, so
every existing job keeps its behavior. Works for gym scripts (train_texture_gan.py, eval_view_ladder.py,
view_match_landscape.py, eval_realism.py) and for segtrain's scripts (build_dataset.py,
run through its run.sh so both src trees are on PYTHONPATH).
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    target = os.path.abspath(sys.argv[1])
    if not os.path.isfile(target):
        print(f"plug_texture_exec: no such script {target}", file=sys.stderr)
        return 2
    from usbg import plug_texture as PT
    PT.widen_choices()
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(target))
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
