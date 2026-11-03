"""Typed interfaces for the pluggable components, so the prose conventions in the README become
checkable contracts. These are `typing.Protocol`s: declaration-only, zero runtime behavior. The
shipped classes already satisfy them structurally; you do not have to subclass anything.

    Renderer          a label slice -> B-mode image, float32 [0,1]
    TextureTranslator a B-mode image -> a more-real B-mode image (single- or dual-input)
    Reward            (obs) -> scalar; the RL loops maximize it
    Similarity        (cur, target) -> [0,1] closeness  (see usbg.similarity.Similarity, an ABC)

Register a NEW implementation of a family with the family's `register_*` decorator (see
usbg.render_lotus.register_renderer, usbg.texture_gan.register_trainer,
usbg.similarity.register_similarity) so a name -> constructor lookup, not a library edit, adds it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Renderer(Protocol):
    """label slice (int64 H*W) -> B-mode float32 in [0,1]. Deterministic given (pose, cfg, seed)."""
    def __call__(self, label_slice: np.ndarray, *, device: str | None = None, **kw) -> np.ndarray: ...


@runtime_checkable
class TextureTranslator(Protocol):
    """A sim-to-real appearance stage: img01 (H*W float [0,1]) -> img01 of the same shape.

    A per-tissue (dual-input) translator additionally sets ``wants_label = True`` and implements
    ``translate_aligned(img01, aligned_label) -> img01`` so the render loop passes the label slice.
    """
    def __call__(self, img01: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Reward(Protocol):
    """A step reward the RL loops maximize. Signatures vary by task (visibility reads the label
    slice; goal/view-match compare current vs target); this marks the role, not one fixed shape."""
    def __call__(self, *args, **kw) -> float: ...


@runtime_checkable
class MaskSource(Protocol):
    """A cross-domain organ segmenter: img01 (H*W float [0,1]) -> int64 mask in the label ontology.
    Bind one with usbg.view_match.set_mask_source(). The default is usbg.us_segmenter."""
    def segment(self, img01: np.ndarray) -> np.ndarray: ...


__all__ = ["Renderer", "TextureTranslator", "Reward", "MaskSource"]
