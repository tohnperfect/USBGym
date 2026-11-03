"""usbg._par: code vendored from the authors' probe-action recognizer code (package `par`).

Source: the authors' probe-action recognizer repository at commit 6d5fe23 (2026-10-14 HEAD; the
two package files were last changed in 73d3893). The code is the usbg authors' own and is distributed
here under the usbg license (LICENSE). Until v2.0.0 usbg imported these modules from the installed
`par` package and executed scripts/us_render.py from the sibling working tree (usbg._reuse); vendoring
them makes usbg installable and runnable from a single clone.

  se3.py       src/par/se3.py, byte-identical. Its sha256 is contracts.GEOMETRY_SHA256 (the cross-repo
               geometry provenance hash), so the file must never be edited; assert_geometry_parity()
               checks it.
  contracts.py src/par/contracts.py, reduced to what usbg uses (channel, class and timing constants,
               GEOMETRY_SHA256, the se3 parity check, NormStats, class_names). The kept code is verbatim;
               see the header of that file for what was dropped.
  us_render.py scripts/us_render.py (untracked in the source repo; sha256 6dbe4cb3...41bf84), reduced
               to the probe scan geometry, the analytic phantom and the NumPy appearance model; the
               plotting demo main() was dropped and the kept code is verbatim.

tests/test_vendor_regression.py pins the numbers the v2 pipeline produced before the switch (stored
reference, bit for bit on the capture platform) and, where `par` is still installed, compares every
vendored function with its original. Import through usbg.contracts_bridge and usbg._reuse, not from here.
"""
