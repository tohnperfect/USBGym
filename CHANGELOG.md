# Changelog

Changes to USB Gym (package `usbg`). Defect IDs (D1 to D12) refer to [ERRATA.md](ERRATA.md), which
states for each defect what 0.1.0 did, what changed and where the change is checked.

## 2.2.1 (2026-11-04)

### Table build
- Table 5 of the article prints the three slots of the guide (mask source, controller and reward); the
  three slots of the standard simulation engine (texture stage, renderer and CT anatomy) print in a new
  table, `tables/pluggability_engine.tex` (Table B.1 of the article). `scripts/paper/make_tables.py`
  writes both from the same rows, and the expected printed text of the public build now holds 61 tables
  (was 60).
- Table 4 (`tables/ladder.tex`) shades the SonoSPADE column, the texture stage of the standard engine,
  and its caption says so; the article loads `colortbl` for it.
- The caption of Table C.1 (`tables/leaderboard.tex`) names both tables and cites the software record.
- No number changes. The results archive (DOI 10.5281/zenodo.23913142) is unchanged, and the release
  assets stay attached to v2.2.0.

## 2.2.0 (2026-11-03)

First release in this repository, `tohnperfect/USBGym`.

### License
- USB Gym is licensed under the Apache License 2.0 from this release on (`LICENSE`, `NOTICE`). Earlier
  versions were released under the MIT license.

### Plug layer
- The plug layer of the 2.1 line is merged into the main code: the slot adapters in `usbg.plug`
  (renderers, mask sources, controllers, rewards and the environment copy they were built on), the
  admission gate (`usbg.plug.admission`, `scripts/plug/admission_gate*.py`), the plugged texture stages
  (`usbg.plug_texture`), the extended ladder evaluator (`usbg.controllers_v21`), the label controls
  (`usbg.label_controls`), and the plug scripts and job builders (`scripts/plug_*.py`, `scripts/plug/`).
  Every plug path is opt-in: without a plug flag the reference loop runs unchanged.
- The anatomy slot rebuilds the label caches from TotalSegmentator v2 automatic labels
  (`scripts/plug/totalseg_labels.py`).
- An opt-in skin-tangent contact mode (`usbg.placement`, `usbg.mujoco_env`, `usbg.body_surface`) and an
  opt-in render seed for the reward-slot trainer (`scripts/plug_reward_train.py`) are added; the defaults
  are unchanged.

### Leaderboard
- `usbg.leaderboard` and [LEADERBOARD.md](LEADERBOARD.md): every published method plugged into a slot (the
  rows of Table 5 of the article, both parts) with the per-episode values behind each row, the benchmark
  baseline, and a methods list with each method's full reference, its DOI (checked against Crossref or
  DataCite) or stable identifier, its code and weights license and its adapter. `python -m
  usbg.leaderboard recompute` rebuilds every printed cell from the shipped values.
- The request path for a new method ([SUBMITTING.md](SUBMITTING.md)): a template plug-in per slot, the
  admission gate (`check`), the protocol runner (`run`, which writes `submission.json`), the validator
  (`validate`), a GitHub issue form and a workflow that validates the pasted file without running
  submitted code, and the maintainers' `add` and `render`.
- Release assets: the physics-trained reference policies of seeds 0 to 2 for the renderer slot, fetched
  with `python -m usbg.leaderboard fetch` and checked against their sha256 (by `fetch` and again by the
  runner). License: Apache-2.0 for the authors' weights, with the TotalSegmentator CC BY 4.0 attribution
  (`NOTICE`).
- A submitted row is post hoc and outside the pre-specified families: it shows its unadjusted paired t p,
  its patient-by-seed interval (three seeds) and no pre-specified decision word, keeps its privileged and
  label-access marks, and is recomputed by `recompute`. The maintainers' `add` resolves the method's DOI at
  Crossref or DataCite and builds its reference in the methods list from that record. A slot whose
  condition needs an unreleased artifact takes a gate record in the issue form for a maintainer run.

- LEADERBOARD.md lists the published rows of each slot in the order of Table C.1 of the article: the
  reference, then the rows whose difference is taken from it by unrounded mean error with the privileged
  rows last, then the reference's seed-0 line with the rows run at seed 0 only; the CT-anatomy rows keep the
  order of Table 5, and the four pretrained-backbone differences are listed apart, outside the order. A
  Role column gives each row's group. The order is not a ranking (`tests/test_leaderboard_order.py` checks
  it row by row against a fixture of Table C.1).

### Tables and figures of the article
- `scripts/paper/` (`make_tables.py`, `make_figures.py`, `reproduce.py` and their tests) and the analysis
  scripts they import (`scripts/analysis/`, with helpers in `scripts/`) rebuild every table and figure
  data file of the article and its supplement from the results archive, in public mode only: the build
  reads the code and the archive, checks every result file against the archive's `MANIFEST.tsv`, takes the
  pre-specified constants from `scripts/paper/public_inputs/plan_constants.json`, and compares every output
  with `scripts/paper/public_inputs/expected/` (`scripts/paper/README.md`).
- The results archive is a separate Zenodo record, doi:10.5281/zenodo.23913142
  (https://doi.org/10.5281/zenodo.23913142): the result files of the reported runs, with `MANIFEST.tsv`,
  `EXCLUDED.tsv` and its own README. Items whose inputs the archive leaves out for third-party terms
  (Fig. A.1, Fig. A.3, Fig. S3.1) are reported as NEEDS.

### Requirements
- Python 3.12 or later (the version CI tests). The extra `plug` installs scikit-image for the SSIM
  reward term.

### Notices
- `NOTICE` carries the MIT notice of the LOTUS code that `usbg.render_lotus` and the acoustic table of
  `usbg.volume` are adapted from, and the CC BY 4.0 attribution of the TotalSegmentator dataset for the
  derived windows file and episode setups.
- Code comments that name versions 2.1.1 to 2.1.5 refer to internal versions of the 2.1 line, which were
  not released.

### Segmenter trainer
- The organ-segmenter trainer (package `us_seg`, the authors' own code) is vendored in `segtrain/` under
  the same license, with its runner, the v2 segmenter cascade (`scripts/run_seg_v2.sh`), the real-data
  ingest and split scripts, and the mask-source plug slot (torchvision networks, nnU-Net v2, adapter
  evaluation). It finds the gym as its parent folder, and the gym scripts find its outputs through
  `USBG_SEG_REPO` (default `segtrain/`).

### Self-contained install
- The SE(3) math, the contract and the B-mode scan geometry that earlier versions imported from a
  sibling package are vendored in `usbg._par` and re-exported through `usbg.contracts_bridge` and
  `usbg._reuse`; no default code path needs a sibling repository, and the numpy < 2.2 pin is lifted.
  `tests/test_vendor_regression.py` checks the data-free pipeline against a stored reference within a
  stated float tolerance.
- The Gymnasium environment `USBGym/ViewMatch-v0` (`usbg.gym_env`) exposes the view-matching task,
  including a callable texture stage (`texture=`) and any mask source with a `segment` method (`mask=`).
- The windows file of the reported runs ships as package data (`src/usbg/data/usbg-windows-v1.json`).

### Tests and CI
- CI (`.github/workflows/tests.yml`) runs the fast suite with CPU-only PyTorch on Ubuntu, Windows and
  macOS for every push, pull request and version tag, against a pinned package stack
  (`.github/ci-constraints.txt`). The fast suite includes the tests of the table and figure build
  (`scripts/paper`, `scripts/analysis`), which run without the results archive.
- The workflows run only in the repository `tohnperfect/USBGym` (a job condition on the repository name).
- The private-data guard (`tests/conftest.py`, `tests/test_private_data_guard.py`) and the leakage gate
  (`scripts/check_leakage.py`, `tests/test_heldout_leakage.py`) run in the fast suite.

## 2.0.0 (2026-10-13)

Every v2 result comes from runs made after these changes.

### Seat and windows
- The anatomical seat is the default (`USBG_SEAT=anatomical`): at a screened acoustic window, the probe face rests on the skin of the patient's CT and the beam points into the body; see `src/usbg/seatmode.py` and `placement.py` (D4).
- Screened acoustic windows for the liver and each kidney flank, eight screens per candidate, are stored in the versioned file `data/cache/windows/usbg-windows-v1.json`, built by `scripts/build_windows.py` (`src/usbg/windows.py`; D4).
- The v1 seat (identity rotation, coronal image plane, probe face inside the trunk) remains available only as `USBG_SEAT=canonical`, to reproduce v1 numbers (D4).
- The kidney label no longer includes the urinary bladder: bladder-free caches in `data/cache_anat/` come from `scripts/build_anat_cache.py`, which checks that only former kidney voxels change.
- Targets are liver windows and right or left flank kidney windows; the spleen is no longer a target.

### Rendering
- Under the anatomical seat, anechoic gel fills each scanline's air gap before it first enters the body, so the fan edges are not blacked out (`render_lotus._leading_gap`, `seatmode.gel_coupling`).
- The README, `DATASHEET.md` and the renderer docstrings no longer describe a gradient path through the renderer, which has none: the NumPy slicer has no gradient, and no result takes a gradient through the renderer (D7).
- The domain-randomization ranges live in one table, `render_lotus.DR_RANGES`, which `scripts/dr_probe.py` reads for the DR parameter table (D8).

### Contact
- The contact plane passes through the window face, perpendicular to the beam, and replaces the flat contact box that did not follow the skin (`placement.seat_from_window`; D4).
- A perturbed start below the contact plane is lifted along the plane's normal until the probe face clears it, and the lift is logged (`placement.clamp_start_outside_skin`).
- The press is capped: the commanded face sits at most 5 mm below the contact plane (`mujoco_env._press_limited`, `USBG_MAX_PRESS_MM`), a model of a robot with a force limit; the tissue does not deform.

### Leakage and provenance
- CUT, CycleGAN and SonoSPADE are retrained with their earlier recipes on pairs from TRAIN patients only and on the public Vitale pool (`scripts/run_textures_v2.sh`, `data/dataset/v2/`, `outputs/v2/textures/manifest.json`; D3).
- Twelve texture-matched segmenters (four stages, three seeds), trained on TRAIN patients only, replace the one shared segmenter, which is now refused by its md5 (`provenance.WITHDRAWN_SEG_MD5`; D2).
- Every trained artifact carries a provenance sidecar (`<artifact>.provenance.json`, schema `usbg-provenance-1.0`), and every result JSON carries a provenance block (`src/usbg/provenance.py`).
- The leakage gate `scripts/check_leakage.py` fails on a held-out patient in any training lineage, a private-data ancestor, a stale md5 or mixed seat modes (`tests/test_heldout_leakage.py`, `tests/test_table4_v2.py`).

### Private-data guard
- Nothing uses the private clinical pool or any model trained on it, and the default render pipeline is physics-only (`benchmark.DEFAULT_TRANSLATOR = None`, `tests/test_benchmark.py`; D5).
- An audit hook, armed for the whole test suite, raises an error on any Python-level read of a private path (`tests/conftest.py`; D5).
- A static scan fails if code outside its allowlists names a private path or a retired checkpoint (`tests/test_private_data_guard.py`; D5).

### Evaluation and statistics
- The statistics follow a pre-specified protocol fixed before any v2 result existed: the patient is the unit, with a patient-cluster bootstrap and a sign-consistency rule (no registry was used).
- The oracle ladder replaces the v1 "oracle (CT mask)" rung: a privileged analytic pose oracle, a CT-mask servo, a learned-mask servo (each stage with its own segmenter), the servo base on its own, a static rung and a scripted sweep, on identical held-out episodes with three seeds (`scripts/eval_view_ladder.py`, `scripts/run_table4_v2.py`; D1).
- After-imitation and final errors are reported for every cell, REINFORCE is an optional stage, and step counts come from the run commands (D10).
- The learnability evaluation is labeled as the appearance term, and the image-servo base is reported as its own rung (D11).
- Distance to real frames is measured in matched sector geometry on the held-out patients (`scripts/eval_realism.py`; D9).
- Registration is evaluated offline on frame-disjoint real targets that no segmenter was trained on (Vitale holdout, AUL, KidneyUS holdout) (`scripts/eval_real_registration.py`; D12).
- A DR parameter table, a DR landscape study and a DR policy ablation are added, every result records its DR setting, and the statement that DR makes the policy rely on geometry is removed (D8).

### Retired v1 items
- The benchmark usbgym-bench-v1 and its goal-reaching (GOAL) task are withdrawn, because the task did not separate learned policies from a random policy (D6).
- The supporting tasks built on the private-pool CUT stage (behavior cloning, visibility RL and contact rate) are withdrawn, and no v2 result uses them (D5).
- The avatar body surface (a template body registered to the CT) is retired; no v2 result uses it.
- The v1 realism evaluation (linear renders against a convex-sector real pool) and its rule to adopt CUT are dropped (D9).

## 0.1.0

- Earlier public release (v1); superseded by 2.0.0. [ERRATA.md](ERRATA.md) lists its defects.
