"""usbg: USB Gym (UltraSound B-mode Gym), version 2.

A CT-based ultrasound probe view-matching testbed. For any probe pose over a patient's labeled
CT it renders a B-mode image (a physics renderer in device-agnostic PyTorch that shares its
acoustic tables with a reimplementation of the LOTUS forward model, plus an optional texture
stage) and resolves probe-skin contact in MuJoCo. The probe is seated at screened acoustic windows on the patient's own CT skin
(usbg.seatmode, usbg.windows). The v2 evaluation is the view-matching oracle ladder
(scripts/train_view_match_rl.py, scripts/eval_view_ladder.py, scripts/run_table4_v2.py);
CHANGELOG.md lists what changed from 0.1.0.

Modules written for the retired v1 tasks (the RL demonstrator, MS-TCN auto-labeling, the
(image, action) dataset generator, behavior cloning and visibility RL) remain in the package for
the record; no v2 result reports them.

The cross-repo contract (channels, classes, timing, SE(3) geometry) and the probe scan geometry
come from the authors' probe-action recognizer code (package `par`). They are vendored verbatim in
usbg._par and re-exported through usbg.contracts_bridge and usbg._reuse, so the v2 code paths need
no sibling repo. Nothing in usbg redefines those values. Only the retired v1 modules above still
import the sibling packages (par's recognizer, us_rl_sim, stable_baselines3), lazily.
"""
from __future__ import annotations

import warnings

# Known false positive: numpy 2.1.x built against Apple's Accelerate BLAS raises spurious
# "overflow / divide by zero / invalid value encountered in matmul" RuntimeWarnings on
# finite, correct inputs (the SIMD remainder path sets FP status flags that numpy surfaces).
# The reported v2 numbers were produced with numpy 2.1.3 (once pinned < 2.2 by the sibling us_rl_sim
# repo, which the v2 paths no longer need); suppress only this specific message and only for matmul,
# leaving all other warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

__version__ = "2.2.0"
