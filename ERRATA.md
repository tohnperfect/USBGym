# Errata: v1 defects fixed in v2 (package version 0.1.0)

This file lists the defects found in the code and documentation of the earlier public release of USB
Gym (v1, package version 0.1.0). For each defect it states what v1 did, what changed in v2 (package
version 2.0.0 and later, including this release), and where the change can be checked. Every v2 result
comes from runs made after these changes.

TRAIN denotes the 13 training patients and held-out the 6 evaluation patients (`usbg.benchmark`).
Scripts are in `scripts/`, tests in `tests/`, modules in `src/usbg/`, and results are written under
`outputs/`.

| ID | What v1 did | What changed in v2 | Where it is checked |
| :-- | :-- | :-- | :-- |
| D1 | The v1 rung named "oracle (CT mask)" ran a privileged pose-oracle controller, not a servo on the CT label mask. | The oracle ladder reports the analytic pose oracle, labeled privileged, and a separate CT-mask image-servo rung on held-out patients with 3 seeds. | `eval_view_ladder.py`; `test_table4_v2.py`; the ladder results (`outputs/v2/table4/`) |
| D2 | The four learned-mask configurations used one shared segmenter, trained on SonoSPADE renders that included held-out patients. | Twelve texture-matched segmenters (4 stages, 3 seeds) are trained on TRAIN patients only; the shared checkpoint is refused by its md5. | `provenance.py`; `test_table4_v2.py`; `check_leakage.py` |
| D3 | The SonoSPADE training pairs included held-out patients, and the CUT and CycleGAN pools were unrecorded. | CUT, CycleGAN and SonoSPADE are trained with their earlier recipes on pairs from the same TRAIN-only patients and on the same public real pool, each with a provenance sidecar. | `outputs/v2/textures/manifest.json`; `test_heldout_leakage.py`; `check_leakage.py` |
| D4 | The v1 seat kept the identity rotation: the image plane was coronal, the beam ran cranio-caudally, the probe face sat inside the trunk, and the flat contact box ignored the skin. | At a screened acoustic window, the probe face rests on the skin surface of the CT volume and the beam points inward; contact uses a plane through the window face, perpendicular to the beam. The old seat is kept only to reproduce v1. | `seatmode.py`, `placement.py`, `windows.py`; `test_placement_anatomical.py`; the seat check of the leakage gate |
| D5 | A CUT stage trained on a private clinical pool was used by the supporting tasks and by the benchmark default. | Nothing uses the private pool or any model derived from it; the default pipeline renders physics only, and the supporting tasks are retired. | `conftest.py` (audit hook); `test_private_data_guard.py`; `test_benchmark.py` |
| D6 | The goal-reaching benchmark had no random-policy baseline. | The goal task and its benchmark are retired, and none of their results is reported (see below). | `CHANGELOG.md` |
| D7 | The v1 documentation described a gradient path through this renderer; the NumPy slicer has no gradient. | The README, the datasheet and the renderer docstrings no longer make that claim for this renderer. | Text search of these files |
| D8 | The v1 documentation attributed the policy's reliance on geometry to domain randomization (DR), which was off in the runs behind that statement. | The claim is removed. Every result records its DR setting, and a DR parameter table, a DR landscape study and a DR policy ablation are added. | `dr_probe.py`; `outputs/v2/view_match/landscape_dr.json`; `outputs/v2/table4/dr_on`, `dr_eval` |
| D9 | The v1 realism evaluation set linear renders against real frames of convex-sector geometry, on the split that the translators had been trained on. | Distance to real frames is measured in matched sector geometry: the sim views come from held-out patients and the real reference is the set of 60 annotated frames; the rule to adopt CUT is dropped. | `eval_realism.py`; `fig_matched_layout.py`; `outputs/v2/realism/realism_sector.json` |
| D10 | The v1 results reported the controller after RL fine-tuning without its error after imitation, and its step count and compute time were not taken from the run commands. | After-imitation and final errors are reported for every cell, REINFORCE is presented as an optional stage, and the step count is taken from the run commands. | `agg_table4_v2.py`; the command line in each provenance block |
| D11 | The v1 learnability evaluation measured only the appearance term of the reward, and the image-servo base was not reported on its own. | The table is labeled as the appearance term, and the servo base is described and reported alone as a ladder rung. | `view_match_landscape.py`, `agg_table3.py`; `train_view_match_rl.py`; `eval_view_ladder.py` (rung `ibvs_base`) |
| D12 | An earlier registration result on real targets used a segmenter that had seen the target frames. | Registration is reported per organ, with its n, on frame-disjoint real targets that no segmenter was trained on. | `eval_real_registration.py`; `outputs/v2/table4/registration_long.csv` |

## Retired parts of v1

Three parts of v1 are retired rather than repaired.

- The benchmark usbgym-bench-v1 and its goal-reaching task are withdrawn. Their episodes were rendered
  through the texture stage trained on the private clinical pool, so their results cannot be
  reproduced from public data and none is reported.
- The supporting tasks (behavior cloning, visibility RL and contact rate) are withdrawn, because they
  were built on that same privately trained CUT stage.
- The avatar body surface, a template body registered to the CT, is retired; no v2 result uses it.

## Further v1 details

- In v1, the urinary-bladder mask was merged into the kidney label, which drew kidney views toward the
  midline bladder. The bladder is now left out of that label, and it was verified on all 19 cases that
  only voxels formerly labeled kidney changed (`scripts/build_anat_cache.py`,
  `data/cache_anat/build_report.json`).
- The v1 seat, with the probe face inside the trunk, survives only as a switch for reproducing its
  numbers (`USBG_SEAT=canonical`).
- The library refuses the withdrawn shared segmenter and any checkpoint descended from the private
  clinical pool.
- The v1 default texture stage, trained on that private pool, is retired.
