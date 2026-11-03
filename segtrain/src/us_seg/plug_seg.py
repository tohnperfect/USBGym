"""v2.1 pluggability study (EXPLORATORY): published segmenters run through the UNCHANGED v2 segmenter
recipe and the UNCHANGED v2 segmenter evaluation.

How "unchanged" is guaranteed: nothing here re-implements the v2 code. The v2 functions are called as
they are, with exactly one thing swapped:

  * training (scripts/plug/train_plug_seg.py): us_seg.training.train builds its network with
    ``usbg.segmenter.Segmenter(device=..., n_classes=6, base=..., depth=..., group_map=None)``.
    tv_segmenter_patch() points that name at usbg.plug.masks.TVSegmenter (a torchvision network with
    the same .net / .predict / .save surface) for the duration of the call. Data, the 13-patient sim
    build, the Vitale split, KidneyUS partial labels, the 0.4 real fraction, the rare-class boost, the
    class-weighted CE + Dice loss, Adam 5e-4, batch 8, 4000 iterations and the final-iterate checkpoint
    are therefore the v2 ones by construction. The batch-index RNG (default_rng(seed)) is the same
    stream, so for a given seed every network sees the same batches as the v2 U-Net.
  * evaluation (scripts/plug/eval_plug_seg.py): scripts/eval_seg.py, scripts/consistency.py and
    scripts/crossval_real.py are loaded as modules and their ``load_segmenter`` name is pointed at a
    plugged adapter (usbg.plug.masks) for the call.

Nothing in the v2 or v2.1 queues imports this module.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

from . import gym_bridge

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
SEG_ROOT = gym_bridge.seg_root()
SCRIPTS_DIR = os.path.join(SEG_ROOT, "scripts")
PLUG_OUT = os.path.join(SEG_ROOT, "outputs", "v2_1", "plug")


@contextlib.contextmanager
def tv_segmenter_patch(arch: str, backbone_weights: str = "none", seed: int | None = None):
    """Within the block, usbg.segmenter.Segmenter(...) builds a TVSegmenter(arch) instead of the U-Net."""
    gym_bridge.ensure_usbg_importable()
    from usbg import segmenter as SEG
    from usbg.plug.masks import TVSegmenter
    orig = SEG.Segmenter
    SEG.Segmenter = functools.partial(TVSegmenter, arch=arch, backbone_weights=backbone_weights, seed=seed)
    try:
        yield
    finally:
        SEG.Segmenter = orig


def load_v2_script(name: str):
    """Import scripts/<name>.py as a fresh module object (so its globals can be patched per call)."""
    path = os.path.join(SCRIPTS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_usbg_plug_v2_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EvalSegmenter:
    """The us_seg.infer.Segmenter6 surface (device, segment, segment6, segment_batch) over a plugged
    usbg.plug.masks adapter."""

    def __init__(self, adapter):
        self._a = adapter

    @property
    def device(self):
        return getattr(self._a, "device", None)

    def segment6(self, img01):
        return np.asarray(self._a.segment6(np.asarray(img01, np.float32)), np.int64)

    def segment(self, img01):
        return np.asarray(self._a.segment(np.asarray(img01, np.float32)), np.int64)

    def segment_batch(self, imgs):
        return np.stack([self.segment(im) for im in np.asarray(imgs, np.float32)])


@contextlib.contextmanager
def patched_loader(module, adapter_factory):
    """module.load_segmenter(path, device=None) -> EvalSegmenter(adapter_factory(path)) inside the block."""
    orig = module.load_segmenter
    module.load_segmenter = lambda path, device=None: EvalSegmenter(adapter_factory(path))
    try:
        yield
    finally:
        module.load_segmenter = orig


def tv_adapter_factory(cache_size=4096):
    gym_bridge.ensure_usbg_importable()
    from usbg.plug.masks import TorchvisionMaskSource
    return lambda path: TorchvisionMaskSource(path, cache_size=cache_size)


def refuse_existing(*paths):
    """Never overwrite: raise if any output already exists."""
    have = [p for p in paths if p and os.path.exists(p)]
    if have:
        raise SystemExit(f"refusing to overwrite existing output(s): {have}")


def file_sha1_of_array(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()


def augment_json(path: str, plug: dict, *, texture=None, segmenter=None, mask_source="learned",
                 train_cases=(), eval_cases=(), extra=None):
    """Add the v2.1 pluggability blocks to a JSON the v2 script just wrote (this run's own output):
    analysis_status, the plug description, and a usbg.provenance.result_block."""
    from usbg import provenance as GP
    with open(path) as fh:
        rep = json.load(fh)
    blk = GP.result_block(texture=texture, segmenter=segmenter, mask_source=mask_source,
                          train_cases=train_cases, eval_cases=eval_cases,
                          extra=dict(extra or {}, plug=plug), repo=gym_bridge.gym_root())
    blk["analysis_status"] = ANALYSIS_STATUS
    rep["analysis_status"] = ANALYSIS_STATUS
    rep["plug"] = plug
    rep["provenance_usbg"] = blk
    if isinstance(rep.get("provenance"), dict):
        rep["provenance"]["analysis_status"] = ANALYSIS_STATUS
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    os.replace(tmp, path)
    return rep


def argv_str():
    return " ".join([os.path.basename(sys.executable)] + sys.argv)
