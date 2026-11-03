"""Read/write the nnU-Net v2 **raw** dataset layout for 2D PNG data.

    nnUNet_raw/DatasetXXX_Name/
      imagesTr/{case}_0000.png     grayscale B-mode (uint8 0..255)
      labelsTr/{case}.png          class ids 0..5 (uint8)
      imagesTs/{case}_0000.png     optional held-out images (the real test frames)
      dataset.json                 channel_names / labels / numTraining / file_ending
      domain.json                  sidecar: per-case provenance (domain sim/real, source case,
                                    seat organ, pose, organ pixel counts) for balanced sampling
                                    and per-domain evaluation

This layout is exactly what `nnUNetv2_plan_and_preprocess` + `nnUNetv2_train ... 2d` consume on a
CUDA box, and what this project's own MPS trainer reads via `RawDataset`. Writing it needs no
`nnunetv2` install: it is pure file emission.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from . import ontology as ONT

CHANNEL_SUFFIX = "_0000"      # nnU-Net channel 0
FILE_ENDING = ".png"


class RawWriter:
    """Incrementally write cases into an nnU-Net v2 raw dataset directory."""

    def __init__(self, root: str, dataset_id: int = 11, name: str = "USOrgansSimReal",
                 case_prefix: str = "USOrgans"):
        self.dir = os.path.join(root, f"Dataset{dataset_id:03d}_{name}")
        self.images_tr = os.path.join(self.dir, "imagesTr")
        self.labels_tr = os.path.join(self.dir, "labelsTr")
        self.images_ts = os.path.join(self.dir, "imagesTs")
        for d in (self.images_tr, self.labels_tr, self.images_ts):
            os.makedirs(d, exist_ok=True)
        self.case_prefix = case_prefix
        self._n_tr = 0
        self._n_ts = 0
        self._domain: dict[str, dict] = {}

    # -- training cases (image + label) -----------------------------------------------------------
    def add_training(self, image01: np.ndarray, label6: np.ndarray, provenance: dict) -> str:
        case = f"{self.case_prefix}_{self._n_tr:06d}"
        _save_image(os.path.join(self.images_tr, f"{case}{CHANNEL_SUFFIX}{FILE_ENDING}"), image01)
        _save_label(os.path.join(self.labels_tr, f"{case}{FILE_ENDING}"), label6)
        self._domain[case] = {"split": "train", **provenance}
        self._n_tr += 1
        return case

    # -- test images (image only; labels live in a separate eval npz) -----------------------------
    def add_test_image(self, image01: np.ndarray, provenance: dict) -> str:
        case = f"{self.case_prefix}_ts_{self._n_ts:06d}"
        _save_image(os.path.join(self.images_ts, f"{case}{CHANNEL_SUFFIX}{FILE_ENDING}"), image01)
        self._domain[case] = {"split": "test", **provenance}
        self._n_ts += 1
        return case

    def finalize(self) -> str:
        dataset_json = {
            "channel_names": {"0": "US"},
            "labels": ONT.NNUNET_LABELS,
            "numTraining": self._n_tr,
            "file_ending": FILE_ENDING,
            "description": "Cross-domain abdominal US organ segmentation (sim train, real test). "
                           "6-class contiguous scheme; export inverts spleen 5->7 to repo ontology.",
            "overwrite_image_reader_writer": "NaturalImage2DIO",
        }
        with open(os.path.join(self.dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)
        with open(os.path.join(self.dir, "domain.json"), "w") as f:
            json.dump(self._domain, f, indent=2, default=_json_default)
        return self.dir


def _json_default(o):
    """numpy values in the per-case provenance as lists and numbers. The gym's seat meta holds a 4x4
    `start_prelift` pose (an ndarray) since the opt-in contact surface was added; json uses this hook only for values it
    has no encoding for, so each sidecar written earlier keeps its bytes."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _save_image(path: str, image01: np.ndarray) -> None:
    arr = np.clip(np.asarray(image01, np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_label(path: str, label6: np.ndarray) -> None:
    lab = np.asarray(label6)
    if lab.min() < 0 or lab.max() > ONT.N_CLASSES - 1:
        raise ValueError(f"label out of 6-class range: [{lab.min()},{lab.max()}]")
    Image.fromarray(lab.astype(np.uint8), mode="L").save(path)


# =================================================================================================
# Reader (feeds the MPS trainer)
# =================================================================================================

class RawDataset:
    """Read an nnU-Net v2 raw dataset back into arrays, with the domain sidecar.

    Provides case-level (session) splitting so held-out validation never shares a source CT case
    with training (no pose leakage).
    """

    def __init__(self, dataset_dir: str):
        self.dir = dataset_dir
        with open(os.path.join(dataset_dir, "dataset.json")) as f:
            self.meta = json.load(f)
        dom_path = os.path.join(dataset_dir, "domain.json")
        self.domain = json.load(open(dom_path)) if os.path.exists(dom_path) else {}
        self.images_tr = os.path.join(dataset_dir, "imagesTr")
        self.labels_tr = os.path.join(dataset_dir, "labelsTr")
        self.train_cases = sorted(
            c for c, d in self.domain.items() if d.get("split") == "train"
        ) or self._scan_train_cases()

    def _scan_train_cases(self) -> list[str]:
        cases = []
        for f in os.listdir(self.labels_tr):
            if f.endswith(FILE_ENDING):
                cases.append(f[: -len(FILE_ENDING)])
        return sorted(cases)

    def __len__(self) -> int:
        return len(self.train_cases)

    def load_case(self, case: str) -> tuple[np.ndarray, np.ndarray]:
        img = np.asarray(
            Image.open(os.path.join(self.images_tr, f"{case}{CHANNEL_SUFFIX}{FILE_ENDING}")),
            dtype=np.float32,
        ) / 255.0
        lab = np.asarray(
            Image.open(os.path.join(self.labels_tr, f"{case}{FILE_ENDING}")), dtype=np.int64
        )
        return img, lab

    def load_all(self, cases: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        cases = cases or self.train_cases
        imgs, labs = [], []
        for c in cases:
            im, la = self.load_case(c)
            imgs.append(im)
            labs.append(la)
        return np.stack(imgs).astype(np.float32), np.stack(labs).astype(np.int64)

    def source_case(self, case: str) -> str:
        return self.domain.get(case, {}).get("case", case)

    def split_by_source_case(self, val_frac: float = 0.2, seed: int = 0
                             ) -> tuple[list[str], list[str]]:
        """Hold out whole source CT cases (no pose leakage between train and val)."""
        by_src: dict[str, list[str]] = {}
        for c in self.train_cases:
            by_src.setdefault(self.source_case(c), []).append(c)
        srcs = sorted(by_src)
        if len(srcs) < 2:
            raise ValueError(
                f"need >=2 source CT cases for a leakage-free case split (got {len(srcs)}: {srcs}). "
                "Build the dataset with more cases (drop --smoke / raise --max-cases)."
            )
        rng = np.random.default_rng(seed)
        rng.shuffle(srcs)
        n_val = max(1, int(round(len(srcs) * val_frac)))
        val_srcs = set(srcs[:n_val])
        train, val = [], []
        for s, cs in by_src.items():
            (val if s in val_srcs else train).extend(cs)
        return sorted(train), sorted(val)
