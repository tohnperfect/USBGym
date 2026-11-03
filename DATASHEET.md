# Datasheet for USB Gym (UltraSound B-mode Gym), version 2.2.0

This datasheet follows the "Datasheets for Datasets" template (Gebru et al., 2021), adapted for a
simulation testbed. USB Gym is software with a fixed evaluation protocol, not a static dataset. For
any probe pose over a patient's labeled CT, it renders a B-mode image and computes a probe-skin
contact force. This datasheet covers code version 2.2.0 and the data behind the v2 results.
`CHANGELOG.md` lists the changes from version 0.1.0 (v1), and `ERRATA.md` (D1 to D12) lists the v1
defects and how each one was fixed.

## Motivation

- **Purpose.** A laptop-native testbed, with no CUDA requirement, for studying closed-loop probe
  view matching in simulation. The agent moves a virtual probe to reproduce a target B-mode view of
  one patient. An oracle ladder of privileged and image-only controllers shows where the remaining
  pose error comes from. The testbed needs no robot, instrumented probe or GPU cluster.
- **Who created it.** Thanapong Intharah (corresponding author; Visual Intelligence Laboratory,
  Department of Statistics, Faculty of Science, Khon Kaen University, Khon Kaen 40002, Thailand),
  Zhichao Gao, Yiliang Liu and Hao Dong (School of Computer Science, Peking University, Beijing 100871,
  China).

## Composition

- **What the software contains.**
  1. A physics B-mode renderer written in PyTorch, deterministic, which runs on MPS or on the CPU
     (`src/usbg/render_lotus.py`). It renders acoustic shadows, a point spread function that changes
     with depth, and correlated speckle. No gradient flows through it: the slicer looks up the nearest
     CT label in NumPy, and no result uses a gradient of the renderer. The baselines are a PyTorch version
     of the LOTUS forward model and a NumPy renderer.
  2. The anatomical seat: at a screened acoustic window, the probe face rests on the skin surface of the
     CT volume and the beam points inward (`src/usbg/seatmode.py`, `placement.py`,
     `windows.py`).
  3. A MuJoCo probe-skin contact environment with a force-capped press and rigid tissue
     (`src/usbg/mujoco_env.py`).
  4. Optional texture stages (CUT, CycleGAN and SonoSPADE) applied after the physics render
     (`src/usbg/texture_gan.py`, `view_match.py`).
  5. The view-matching task, its segmentation-anchored reward and the oracle-ladder controllers
     (`src/usbg/view_match.py`, `goal_rl.py`, `policy.py`; `scripts/train_view_match_rl.py`,
     `scripts/eval_view_ladder.py`, `scripts/run_table4_v2.py`).
  6. Provenance records, a held-out leakage gate and a private-data guard
     (`src/usbg/provenance.py`, `scripts/check_leakage.py`, `tests/conftest.py`,
     `tests/test_private_data_guard.py`).
  7. The plug layer: adapters that place published methods in the renderer, texture, mask-source,
     controller, reward and anatomy slots, and an admission gate (`src/usbg/plug/`,
     `src/usbg/plug_texture.py`, `scripts/plug_*.py`, `scripts/plug/`).
  8. The organ-segmenter trainer (`segtrain/`, package `us_seg`).
- **Anatomy instances.** 19 CT cases from the small subset of the public TotalSegmentator dataset,
  each with non-empty liver and gallbladder masks. Each case becomes one volume with 13 labels.
- **Split.** A frozen patient-disjoint split in `src/usbg/benchmark.py`: 13 training patients
  (`TRAIN_PATIENTS`) and 6 held-out patients (`TEST_PATIENTS`: s0358, s0513, s0519, s0591, s0649,
  s0762). Held-out patients enter no texture stage, segmenter or policy training. The leakage gate
  checks this through each artifact's lineage. Do not reshuffle the split, and report results on the
  held-out patients.
- **Label caches.** `data/cache/<case>.npz` holds the labeled volume built by
  `scripts/build_ct_volume.py`. In v2 the urinary bladder is no longer merged into the kidney label.
  The bladder-free twin `data/cache_anat/<case>.npz` is built by `scripts/build_anat_cache.py`, and
  `volume.load_cache` reads it under the anatomical seat. Its build report,
  `data/cache_anat/build_report.json`, records that only former kidney voxels changed.
- **Acoustic windows.** `data/cache/windows/usbg-windows-v1.json` (version `usbg-windows-v1`) holds
  the screened liver and per-side kidney windows of every case. It is built by
  `scripts/build_windows.py`. The file of the reported runs has md5
  `af35f0bddeb57c079981eb0ba8aca5b3` and ships as package data (`src/usbg/data/usbg-windows-v1.json`;
  select it with `USBG_WINDOWS`). Every v2 result records the md5 it read. Seat lookups refuse a stale
  or missing file. The file is derived from the public TotalSegmentator CTs only.
- **Targets.** Liver windows and right or left flank kidney windows. The spleen is not a target.
  Three training cases (s0058, s0522 and s0727) have fewer than two kidney windows, so they give
  liver targets only.
- **Labels per frame.** Each simulated frame comes with its CT label slice and the probe pose. The
  reward uses organ masks from the CT label slice or from a texture-matched segmenter.
- **Real ultrasound.** Only public sets are used (see the table under Distribution).
  - Vitale et al. abdominal set: its 404 training-split frames are the real pool of the three texture
    stages (`data/real/kaggle_vitale/pool`). Of its 60 annotated test frames, 42 train the
    segmenters and 18 form a holdout used only as real targets. The 60 annotated frames are also the
    real side of the matched-geometry distance comparison.
  - KidneyUS: the training split trains the segmenters, and the holdout split serves only as real
    targets.
  - AUL: liver frames serve only as real targets.
- **Known limits.** TotalSegmentator masks are automatic and can be wrong. Tissue is rigid. The
  contact plane passes through the window face, perpendicular to the beam, so it is tangent to the
  skin only when the beam follows the skin normal. The renderer is a simplified ray model, and a
  simulated frame is not a real ultrasound frame. No realism claim is made for any texture stage.

## Collection and preprocessing

- **CT anatomy.** Hounsfield thresholds label each voxel air, fat or soft tissue. Organ masks are
  overlaid on top, and bone masks and body voxels of at least 300 HU are overlaid last. Volumes are
  reoriented to a canonical frame, with the voxel-to-world affine in meters. No new CT data were
  collected.
- **Real ultrasound.** Each public set is obtained from its source. `scripts/ingest_kaggle_us.py`
  builds the Vitale pool and the annotated test arrays. No new human or animal data were collected,
  and no experiment involved human participants.
- **Rendering.** A render is fixed by its label slice, configuration and seed. A test checks that
  it is bit-reproducible on the CPU (`tests/test_renderer_upgrade.py`); no test covers the MPS
  backend.
- **Texture stages.** CUT, CycleGAN and SonoSPADE were retrained once, with seed 0, on physics
  renders from eight training patients and the 404 Vitale frames (`scripts/run_textures_v2.sh`).
  The checkpoints are `data/dataset/v2/{cut,cyclegan,sonospade}_curvi.pt`, listed in
  `outputs/v2/textures/manifest.json`.
- **Segmenters.** Twelve texture-matched segmenters (four stages, seeds 0, 1 and 2) were trained with
  the segmenter trainer in `segtrain/`, on renders of ten training patients with three more for
  validation, plus the 42 Vitale frames and the KidneyUS training split. Their sidecars are
  `us_seg_<stage>_s<seed>.pt.provenance.json`, listed in the trainer's `outputs/v2/seg_manifest.json`.

## Uses

- **Intended uses.** Research on probe view-matching controllers, image-based rewards and texture
  stages, in simulation, on a laptop.
- **Out-of-scope uses.** USB Gym must not be used for clinical decisions or as evidence of
  real-world scanning performance. No physical probe, phantom or robot was used. No clinical
  validation is claimed, and no transfer of a policy from simulation to real scanning is claimed.

## Distribution and access

- **Code.** Apache License 2.0 (`LICENSE`, `NOTICE`) from version 2.2.0; earlier versions were
  released under the MIT license. The git repository holds the code, configurations, tests, scripts,
  the vendored segmenter trainer and the windows file of the reported runs.
- **Not in the git repository.** CT volumes, label caches, texture and segmenter checkpoints, and
  `outputs/` (result JSONs, provenance blocks). Rebuild them from the public sources with the scripts
  named above.
- **Third-party data.** This repository does not redistribute these datasets; obtain each one from
  its source. The repository ships no images or videos.

  | Source | Use in v2 | License and access |
  | :-- | :-- | :-- |
  | TotalSegmentator CT (Wasserthal et al., 2023), small subset | Anatomy, 19 cases | CC BY 4.0 |
  | Abdominal ultrasound set of Vitale et al. (2020), Kaggle | Texture real pool; segmenter training (42 annotated frames); real targets (18-frame holdout); distance comparison | Used under its original license, as stated on its Kaggle page |
  | Open Kidney Ultrasound Data Set, KidneyUS (Singla et al., 2023) | Segmenter training (training split); real targets (holdout split) | CC BY-NC-SA; access after registration and approval by its maintainers |
  | Annotated Ultrasound Liver images, AUL (Xu et al., 2022), Zenodo 10.5281/zenodo.7272660 | Real liver targets only | CC BY 4.0 |

- **Trained weights.** The segmenters were trained on KidneyUS frames, so they inherit its terms. They
  are not in this git repository.
- **Release assets.** Six files are attached to the GitHub release v2.2.0 for the renderer slot of the
  leaderboard: the physics-trained reference policies of seeds 0 to 2, `usbg_physics_policy_s{0,1,2}.pt`
  (weights and servo base), and their `.json` configurations. They were trained only on physics renders
  of the 13 training TotalSegmentator CTs, with no real ultrasound and no held-out patient. License:
  Apache-2.0 for the authors' weights; the TotalSegmentator dataset's CC BY 4.0 attribution applies.
  `src/usbg/leaderboard/data/assets.json` lists their sha256 and size, which `python -m usbg.leaderboard
  fetch` and the runner check.
- **Derived protocol files.** `src/usbg/data/usbg-windows-v1.json` (the screened windows of the reported
  runs) and the episode setups in `src/usbg/leaderboard/data/` are derived from the TotalSegmentator
  label maps (CC BY 4.0; attribution and changes in `NOTICE`). The fixed KidneyUS test assignment of the
  reported runs (`segtrain/data/real/kidneyus_split.npz`) is not yet published.
- **Private data: not used and not distributed.** The v1 benchmark default and the v1 supporting
  tasks used a CUT stage trained on a private clinical pool (ledger D5). In v2 nothing uses that
  pool or any model trained on it, and none of it is in the repository or distributed. The
  default render pipeline is physics-only (`benchmark.DEFAULT_TRANSLATOR = None`, checked by
  `tests/test_benchmark.py`). An audit hook, armed for the whole test suite, raises an error on any
  Python-level read of a private path (`tests/conftest.py`). A static scan fails if code outside its
  allowlists names a private path or a retired checkpoint (`tests/test_private_data_guard.py`). The
  earlier research scripts that read the private pool are not part of this repository.
- **Provenance files.**
  - `<artifact>.provenance.json` next to every trained artifact (schema `usbg-provenance-1.0`,
    `src/usbg/provenance.py`): training and validation cases, real pool with a frame fingerprint,
    seed, seat mode, randomization mode, parents with md5, git commit and dirty state, command and
    creation time. Their license fields are free-text notes written at training time; the table
    above is the authoritative statement.
  - A `provenance` block inside every v2 result JSON: seat mode, windows-file md5, texture and
    segmenter md5, mask source, patients, command line, git commit and the md5 of any uncommitted
    diff.
  - `outputs/v2/textures/manifest.json` and `manifest_meta.json`: the texture stages, their md5,
    the windows file and the transitive training cases.
  - `outputs/v2/leakage_gate.json`, written by `scripts/check_leakage.py`.
  - `data/cache_anat/build_report.json`: the bladder-removal check.

## Maintenance

- **Maintainers.** The authors listed above; contact the corresponding author through the issue
  tracker of the repository.
- **Versioning.** The package follows semantic versions, recorded in `CHANGELOG.md`. The windows
  file carries its own version string, and every provenance record carries the git commit. The
  frozen split in `src/usbg/benchmark.py` is still labeled `usbgym-bench-v1`, and provenance records
  copy that label with the held-out list. The goal-reaching benchmark task that carried this label is
  retired (ledger D6).
- **Contributions.** New CT cases enter through the CT pipeline, after which the windows file is
  rebuilt. New texture stages, controllers and rewards plug into the existing interfaces. The tests,
  the leakage gate and the private-data guard gate correctness.
