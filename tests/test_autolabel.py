"""Phase 5 acceptance: the recognizer runs over trajectory kinematics and produces a
per-maneuver agreement report. The decisive angular maneuvers (rock, tilt) should be
recognized with high agreement, validating the predict_record integration end to end.

Marked slow: loads the MS-TCN teacher and builds the Repo B env (clean demonstrator, no PPO)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from usbg import autolabel as AL
from usbg.contracts_bridge import CLASSES_5


@pytest.mark.slow
def test_agreement_report_structure_and_strong_maneuvers():
    if not (AL.DEFAULT_CKPT and os.path.exists(AL.DEFAULT_CKPT)):
        pytest.skip("v1 only: needs the probe-action recognizer checkpoint (USBG_RECOGNIZER_CKPT) and its package")
    from usbg.demonstrator import Demonstrator
    model, norm, device = AL.load_recognizer()

    demo = Demonstrator()
    clips = []
    try:
        rng = np.random.default_rng(0)
        tid = 0
        for action in ("rock", "tilt", "slide"):
            for _ in range(2):
                tr = demo.clean(action, rng)
                clips.append({"twist": tr.twist, "rel6": tr.rel6, "valid": tr.valid,
                              "label": tr.label, "source": "clean", "case": "test", "traj_id": tid})
                tid += 1
    finally:
        demo.close()

    report = AL.agreement_report(clips, model, norm, device)
    # structure
    assert set(report["per_maneuver"]) == set(CLASSES_5)
    assert np.shape(report["confusion_commanded_x_predicted"]) == (6, 6)
    assert 0.0 <= report["overall_frame_agreement"] <= 1.0
    assert isinstance(report["flagged_low_agreement"], list)
    # rock and tilt are decisive angular maneuvers -> recognizer should agree strongly
    assert report["per_maneuver"]["rock"]["frame_agreement"] > 0.8
    assert report["per_maneuver"]["tilt"]["frame_agreement"] > 0.8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "slow"]))
