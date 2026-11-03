# segtrain: the organ-segmenter trainer (`us_seg`)

`segtrain/` trains the organ segmenters that USB Gym uses as its learned mask source, and the
segmenters of the mask-source plug slot. It is the authors' own code, released with USB Gym under the
same Apache License 2.0 (see the repository's [LICENSE](../LICENSE) and [NOTICE](../NOTICE)).

The gym loads a trained checkpoint through `usbg.us_segmenter` (`US_SEG_CKPT=<checkpoint>` with
`USBG_LEARNED_SEG=1`) and does not import this package at run time.

## What it does

- Renders simulated training frames with `usbg` (13 training patients only, anatomical seat, per-frame
  domain randomization, CT-derived labels) through a chosen texture stage, and writes them with the
  public real frames into nnU-Net v2 raw layout (`scripts/build_dataset.py`).
- Trains the compact six-class U-Net of `usbg.segmenter` on that build plus real labeled frames
  (`scripts/train.py`), and evaluates it per domain (`scripts/eval_seg.py`, `scripts/crossval_real.py`,
  `scripts/consistency.py`).
- Runs the whole v2 cascade (four texture stages, seeds 0, 1 and 2) with `scripts/run_seg_v2.sh`, and
  writes `outputs/v2/seg_manifest.json`, which the gym's runners read.
- Mask-source plug slot: torchvision networks with the same recipe (`scripts/plug/train_plug_seg.py`),
  nnU-Net v2 2D (`scripts/plug/build_nnunet_plug_dataset.py`, `scripts/plug/train_nnunet_plug.py`) and
  the evaluation of any `usbg.plug.masks` adapter by the same scripts (`scripts/plug/eval_plug_seg.py`).

## Layout

```
segtrain/
  run.sh                  runs a script with the gym's src/ and segtrain/src/ on PYTHONPATH
  src/us_seg/             the package: rendering pairs, nnU-Net raw writer, training, inference, ontology
  scripts/                build, train, evaluate, ingest, split, manifest, the v2 cascade
  scripts/plug/           the mask-source plug slot (torchvision, nnU-Net, adapter evaluation)
  data/, outputs/         created locally and git-ignored
```

## Running

Run every command from this folder. `run.sh` finds the gym as the folder above (`USBG_GYM_ROOT`
overrides it) and uses `USBG_PYTHON`, else `<gym>/.venv/bin/python`, else `python3` on the `PATH`.

```bash
cd segtrain

# real data (download each dataset yourself under its terms; see "Data and licenses" below)
./run.sh scripts/make_vitale_split.py --mode group --holdout-groups g,b       # fixed Vitale split
./run.sh scripts/ingest_kidneyus.py --images-dir data/real/kidneyus/images \
    --masks-dir data/real/kidneyus_repo/labels/reviewed_masks_1 --out data/real/kidneyus_kidney_test.npz
./run.sh scripts/ingest_aul.py --src data/real/aul --out data/real/aul_liver_test.npz

# the v2 cascade: per texture stage, seeds 0, 1 and 2 (reads the gym's outputs/v2/textures/manifest.json)
USBG_SEAT=anatomical scripts/run_seg_v2.sh sonospade physics cut cyclegan
./run.sh scripts/agg_tablev_v2.py --dir outputs/v2       # organ-identity table over stages and seeds
```

`run_seg_v2.sh` prints every command before it runs it, skips finished steps, and with `DRY=1` only
prints. The gym's annotated Vitale test set (`data/eval/kaggle_vitale_test.npz` in the gym, from
`scripts/ingest_kaggle_us.py`) is read through `us_seg.gym_bridge`.

Mask-source plug slot (one architecture, one stage and seed):

```bash
USBG_SEAT=anatomical USBG_LEARNED_SEG=0 ./run.sh scripts/plug/train_plug_seg.py train --arch lraspp_mbv3 \
    --dataset data/nnunet_v2/Dataset110_USOrgansV2_sonospade_s0 --seed 0 --kidneyus \
    --vitale-split data/real/vitale_split_v2.npz --forbid-test-patients --iters 4000 \
    --out outputs/v2_1/plug/seg/lraspp_mbv3/us_seg_lraspp_mbv3_sonospade_s0.pt
```

The gym's job builders for this slot (`scripts/plug/make_plug_seg_jobs.py` and its siblings in the gym)
write the exact command of every plug run. `USBG_SEG_REPO` points the gym at this folder; its default
is this folder.

## Rules the code enforces

- Simulated training frames come from the 13 training patients only (`--forbid-test-patients`); the six
  held-out patients appear only in the separate sim-evaluation builds.
- Labels of simulated frames are CT-derived (`USBG_LEARNED_SEG=0` is forced while building).
- The texture stages trained in v1 on pairs that included held-out patients are refused by name and
  md5 (`us_seg.gym_bridge.is_legacy_translator`).
- Every checkpoint gets a `<checkpoint>.provenance.json` sidecar (training cases, real data, seed, seat,
  parents), which the gym's leakage gate (`scripts/check_leakage.py`) reads.

## Data and licenses

- Vitale et al. abdominal ultrasound (Kaggle): used under its original license.
- Open Kidney Ultrasound Data Set (KidneyUS; Singla et al.): CC BY-NC-SA; the images need registration
  with its maintainers. Segmenters trained on KidneyUS frames inherit its terms.
- Annotated Ultrasound Liver (AUL; Xu et al.): Zenodo record 7272660, CC BY 4.0.

None of these datasets ship with this repository.
