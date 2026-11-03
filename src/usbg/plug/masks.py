"""usbg.plug.masks: published segmenters plugged in as the gym's mask source (v2.1 pluggability study,
exploratory). OPT-IN only: nothing in the v2 or v2.1 job queues imports this module. It is reached
through scripts/plug/eval_plug_mask.py (the gym rungs) and through segtrain's
scripts/plug/*.py (training, and the segmenter's own sim and real Dice).

Every adapter implements ``usbg.interfaces.MaskSource``: ``segment(img01)`` maps a (H, W) float frame in
[0, 1] to an int64 mask in the repo ontology {0, 1, 2, 3, 4, 7}, which is exactly what
``view_match.curvi_render`` returns from the default segmenter. The reward, the servo base and every
rung therefore consume a plugged mask unchanged. Each adapter also exposes ``segment6`` (the contiguous
6-class scheme {0..5} segtrain's evaluation scores), ``ckpt_path`` (the primary weights file, which
``usbg.provenance.mask_info`` hashes into every run's provenance), ``privileged``, ``describe()`` and
``timing()`` (the measured per-frame cost).

Methods (kind=...)
  tv       torchvision semantic segmenters, trained by segtrain on the v2 segmenter data with
           the v2 recipe (scripts/plug/train_plug_seg.py swaps only the network):
             lraspp_mbv3     LR-ASPP, MobileNetV3-Large backbone (Howard et al., ICCV 2019)
             deeplabv3_r50   DeepLabV3, ResNet-50 backbone (Chen et al., arXiv:1706.05587)
             deeplabv3_mbv3  DeepLabV3, MobileNetV3-Large backbone
             fcn_r50         FCN, ResNet-50 backbone (Long, Shelhamer, Darrell, CVPR 2015)
  nnunet   nnU-Net v2, 2D configuration, through nnunetv2's own predictor (Isensee et al., Nat. Methods
           2021). Needs the plugin venv (<plugin venv>).
  sam      Segment Anything ViT-B, box prompts (Kirillov et al., ICCV 2023), official SamPredictor.
  medsam   MedSAM ViT-B, box prompts (Ma et al., Nat. Commun. 2024), MedSAM's own preprocessing
           (bicubic resize to 1024 x 1024, min-max to [0, 1], image_encoder called directly).

Box prompts for sam / medsam (prompt=...)
  ct       the CT label slice of the SAME frame. PRIVILEGED: the mask source reads the CT. Needs
           install_ct_label_tap() (process-local; see there), which bind_mask_source() installs.
  gt       a ground-truth label registered beforehand with register_prompts(). PRIVILEGED; offline
           evaluation only (segtrain's sim and real Dice).
  seg      the mask of a learned mask source (prompt_ckpt=<a compact U-Net .pt>): non-privileged, SAM as a
           refiner of the gym's own segmenter. No CT at run time.
  One box per connected component (8-connected, >= min_px pixels, at most max_boxes per organ) of each
  organ in view_match.MATCH_ORGANS; each box's mask is labeled with its prompt's organ; overlaps go to
  the highest probability; probability <= 0.5 is background.

Spec strings (load_mask_source / bind_mask_source): comma-separated key=value, for example
  kind=tv,ckpt=/abs/us_seg_lraspp_mbv3_sonospade_s0.pt
  kind=nnunet,model=/abs/.../nnUNetTrainer_100epochs__nnUNetPlans__2d,fold=0
  kind=sam,weights=/abs/sam_vit_b_01ec64.pth,prompt=ct
  kind=medsam,weights=/abs/medsam_vit_b.pth,prompt=seg,prompt_ckpt=/abs/us_seg_sonospade_s0.pt
  kind=samus,weights=/abs/SAMUS.pth,prompt=ct      (SAMUS: see usbg.plug.samus_mask, extra key ptype)
Optional keys: device (cpu | mps), mirror (nnunet: 0 | 1, default 0), min_px (20), max_boxes (5),
box_mode (component | organ).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict

import numpy as np

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"

# 6-class training id (index) -> repo id (spleen 5 -> 7), and canonical 0..12 -> 6-class (non-target -> 0).
SIX_TO_REPO = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
_CANON_TO_SIX = np.zeros(13, dtype=np.int64)
for _six, _canon in enumerate((1, 2, 3, 4, 7), start=1):
    _CANON_TO_SIX[_canon] = _six
MATCH_ORGANS_REPO = (1, 2, 3, 4, 7)          # == usbg.view_match.MATCH_ORGANS (liver, GB, vessel, kidney, spleen)

CITATIONS = {
    "lraspp_mbv3": "Howard et al., Searching for MobileNetV3, ICCV 2019, 1314-1324, doi:10.1109/ICCV.2019.00140",
    "deeplabv3_r50": "Chen et al., Rethinking Atrous Convolution for Semantic Image Segmentation, arXiv:1706.05587",
    "deeplabv3_mbv3": "Chen et al., arXiv:1706.05587 (DeepLabV3 head); Howard et al., ICCV 2019 (MobileNetV3 backbone)",
    "fcn_r50": "Long, Shelhamer, Darrell, Fully Convolutional Networks, CVPR 2015, 3431-3440, doi:10.1109/CVPR.2015.7298965",
    "nnunet": "Isensee et al., nnU-Net, Nat. Methods 18 (2021) 203-211, doi:10.1038/s41592-020-01008-z",
    "sam": "Kirillov et al., Segment Anything, ICCV 2023, 3992-4003, doi:10.1109/ICCV51070.2023.00371",
    "medsam": "Ma et al., Segment anything in medical images, Nat. Commun. 15 (2024) 654, doi:10.1038/s41467-024-44824-z",
}
TV_ARCHS = ("lraspp_mbv3", "deeplabv3_r50", "deeplabv3_mbv3", "fcn_r50")
TV_FORMAT = "usbg-plug-tvseg-1"
_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


def to_repo(lab6: np.ndarray) -> np.ndarray:
    return SIX_TO_REPO[np.asarray(lab6, np.int64)]


def canon_to_six(lab: np.ndarray) -> np.ndarray:
    """Canonical (0..12) or repo-ontology label -> contiguous 6-class (non-target organs -> background)."""
    return _CANON_TO_SIX[np.asarray(lab, np.int64)]


def _md5(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _manifest_entry(path):
    """The install role's MANIFEST.json record (sha256, url, license) for a weights file, if any."""
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    for _ in range(3):
        mf = os.path.join(d, "MANIFEST.json")
        if os.path.isfile(mf):
            try:
                M = json.load(open(mf))
                rel = os.path.relpath(p, d)
                for e in M.get("files", []):
                    if e.get("path") == rel:
                        return {k: e.get(k) for k in ("path", "url", "license", "sha256", "md5", "size_bytes")}
            except Exception:
                return None
        d = os.path.dirname(d)
    return None


def _pkg_versions(*names):
    out = {}
    for n in names:
        try:
            from importlib.metadata import version
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def _device(dev=None):
    import torch
    if dev:
        return torch.device(dev)
    from ..render_lotus import get_device
    return torch.device(get_device())


# =========================================================================== base class + timing
class _PlugMask:
    """Shared plumbing: repo-ontology output, an optional result cache keyed by the frame bytes (a
    deterministic model gives the same answer for the same frame), and per-call timing."""

    kind = "?"
    privileged = False

    def __init__(self, cache_size=0):
        self._cache = OrderedDict()
        self._cache_size = int(cache_size)
        self._times = []
        self._n_cached = 0

    # subclasses implement _segment6(img01) -> (H, W) int64 in {0..5}
    def segment6(self, img01: np.ndarray) -> np.ndarray:
        img = np.asarray(img01, np.float32)
        key = None
        if self._cache_size:
            key = hashlib.sha1(img.tobytes()).hexdigest() + str(img.shape)
            hit = self._cache.get(key)
            if hit is not None:
                self._n_cached += 1
                self._cache.move_to_end(key)
                return hit.astype(np.int64)
        t0 = time.perf_counter()
        out = np.asarray(self._segment6(img), np.int64)
        self._times.append((time.perf_counter() - t0) * 1000.0)
        if key is not None:
            self._cache[key] = out.astype(np.uint8)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return out

    def segment(self, img01: np.ndarray) -> np.ndarray:
        """(H, W) float [0, 1] -> (H, W) int64 mask in the repo ontology {0, 1, 2, 3, 4, 7}."""
        return to_repo(self.segment6(img01)).astype(np.int64)

    def segment_batch(self, imgs):
        return np.stack([self.segment(im) for im in np.asarray(imgs, np.float32)])

    def timing(self) -> dict:
        t = np.asarray(self._times, np.float64)
        if not len(t):
            return {"n_calls": 0, "n_cache_hits": self._n_cached}
        return {"n_calls": int(len(t)), "n_cache_hits": self._n_cached, "mean_ms": round(float(t.mean()), 3),
                "median_ms": round(float(np.median(t)), 3), "p90_ms": round(float(np.percentile(t, 90)), 3),
                "first_call_ms": round(float(t[0]), 3)}

    def describe(self) -> dict:
        raise NotImplementedError


# =========================================================================== torchvision segmenters
def _tv_builder(arch, n_classes, backbone_weights):
    """torchvision.models.segmentation builder. backbone_weights 'none' trains from scratch (no
    download); 'imagenet' uses the torchvision ImageNet backbone weights, which must already be in
    $TORCH_HOME (this module never downloads)."""
    import torchvision.models.segmentation as TS
    import torchvision.models as TM
    if arch not in TV_ARCHS:
        raise ValueError(f"unknown torchvision arch {arch!r}; one of {TV_ARCHS}")
    if backbone_weights not in ("none", "imagenet"):
        raise ValueError("backbone_weights must be 'none' or 'imagenet'")
    mb = TM.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    rb = TM.ResNet50_Weights.IMAGENET1K_V1
    wb = None
    if backbone_weights == "imagenet":
        wb = mb if arch.endswith("mbv3") else rb
        import torch
        fn = os.path.basename(wb.url)
        hub = os.path.join(torch.hub.get_dir(), "checkpoints", fn)
        if not os.path.isfile(hub):
            raise FileNotFoundError(f"ImageNet backbone {fn} is not in {hub}; download it into the torch hub "
                                    "cache first, or use backbone_weights='none'.")
    if arch == "lraspp_mbv3":
        return TS.lraspp_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=n_classes)
    if arch == "deeplabv3_r50":
        return TS.deeplabv3_resnet50(weights=None, weights_backbone=wb, num_classes=n_classes, aux_loss=False)
    if arch == "deeplabv3_mbv3":
        return TS.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=wb, num_classes=n_classes,
                                               aux_loss=False)
    return TS.fcn_resnet50(weights=None, weights_backbone=wb, num_classes=n_classes, aux_loss=False)


def build_tv_net(arch, n_classes=6, backbone_weights="none"):
    """A torch.nn.Module with the usbg U-Net's calling convention: (B, 1, H, W) in [0, 1] -> logits
    (B, n_classes, H, W). The gray frame is replicated to 3 channels and ImageNet-normalized (the
    torchvision convention, so the optional ImageNet backbone sees its training statistics)."""
    import torch
    import torch.nn as nn

    class TVSegNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _tv_builder(arch, n_classes, backbone_weights)
            self.register_buffer("mean", torch.tensor(_IMNET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_IMNET_STD, dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, x):
            x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
            return self.model((x - self.mean) / self.std)["out"]

    return TVSegNet()


class TVSegmenter:
    """Drop-in for usbg.segmenter.Segmenter (same constructor keywords, .net, .predict, .save, .load)
    around a torchvision network, so segtrain's unchanged training loop
    (us_seg.training.train) trains it with the v2 data, sampling, loss, optimizer and budget. ``base``
    and ``depth`` are accepted and ignored (they size the U-Net)."""

    def __init__(self, device=None, n_classes=6, base=32, depth=4, group_map=None, *, arch="lraspp_mbv3",
                 backbone_weights="none", seed=None):
        import torch
        from ..render_lotus import get_device
        if group_map is not None:
            raise ValueError("TVSegmenter: group_map is not supported (6-class only)")
        self.device = device or get_device()
        self.n_classes = int(n_classes)
        self.group_map = None
        if seed is not None:
            torch.manual_seed(int(seed))
        self.net = build_tv_net(arch, self.n_classes, backbone_weights).to(self.device)
        self._cfg = {"format": TV_FORMAT, "arch": arch, "n_classes": self.n_classes,
                     "backbone_weights": backbone_weights, "group_map": None, "init_seed": seed,
                     "input": "gray [0,1] replicated to 3 channels, ImageNet mean/std normalization",
                     "inference": "net.eval() (BatchNorm running statistics), argmax"}

    def remap(self, canonical_label):
        return np.asarray(canonical_label)

    def predict_logits(self, img01):
        import torch
        a = torch.from_numpy(np.asarray(img01, np.float32))[None, None].to(self.device)
        self.net.eval()
        with torch.no_grad():
            return self.net(a)

    def predict(self, img01) -> np.ndarray:
        return self.predict_logits(img01)[0].argmax(0).cpu().numpy().astype(np.int64)

    def save(self, path):
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state": self.net.state_dict(), **self._cfg}, path)

    @classmethod
    def load(cls, path, device=None):
        import torch
        dev = device or _device()
        ck = torch.load(path, map_location=dev, weights_only=False)
        if ck.get("format") != TV_FORMAT:
            raise ValueError(f"{path}: not a {TV_FORMAT} checkpoint (format={ck.get('format')!r})")
        seg = cls(device=dev, n_classes=ck["n_classes"], arch=ck["arch"], backbone_weights="none")
        seg._cfg.update(backbone_weights=ck.get("backbone_weights"), init_seed=ck.get("init_seed"))
        seg.net.load_state_dict(ck["state"], strict=True)
        seg.net.eval()
        return seg


class TorchvisionMaskSource(_PlugMask):
    kind = "tv"

    def __init__(self, ckpt, device=None, cache_size=0):
        super().__init__(cache_size)
        self.ckpt_path = os.path.abspath(ckpt)
        self._seg = TVSegmenter.load(self.ckpt_path, device=_device(device))
        self.arch = self._seg._cfg["arch"]
        self.device = self._seg.device
        self.name = f"plug:tv:{self.arch}"

    def _segment6(self, img01):
        return self._seg.predict(img01)

    def describe(self):
        import torch
        import torchvision
        return {"slot": "mask_source", "kind": "tv", "method": self.arch, "citation": CITATIONS[self.arch],
                "privileged": False, "weights": {"path": self.ckpt_path, "md5": _md5(self.ckpt_path)},
                "backbone_weights": self._seg._cfg.get("backbone_weights"), "device": str(self.device),
                "packages": {"torch": torch.__version__, "torchvision": torchvision.__version__},
                "preprocessing": self._seg._cfg["input"]}


# =========================================================================== nnU-Net v2
class NnUNetMaskSource(_PlugMask):
    """nnU-Net v2 2D through nnunetv2.inference.predict_from_raw_data.nnUNetPredictor (its own
    preprocessing: crop to nonzero, z-score, sliding window). The frame is passed as nnU-Net's
    NaturalImage2DIO would read the training PNG: (c=1, 1, H, W) float on the 0..255 scale, spacing
    (999, 1, 1). Mirroring test-time augmentation is OFF by default (mirror=1 turns it on)."""

    kind = "nnunet"

    def __init__(self, model_dir, fold=0, checkpoint="checkpoint_final.pth", device=None, use_mirroring=False,
                 cache_size=0):
        super().__init__(cache_size)
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.model_dir = os.path.abspath(model_dir)
        self.fold = int(fold)
        self.ckpt_path = os.path.join(self.model_dir, f"fold_{self.fold}", checkpoint)
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(self.ckpt_path)
        self.device = _device(device)
        self.use_mirroring = bool(use_mirroring)
        self._pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=self.use_mirroring,
                                     perform_everything_on_device=False, device=self.device, verbose=False,
                                     verbose_preprocessing=False, allow_tqdm=False)
        self._pred.initialize_from_trained_model_folder(self.model_dir, use_folds=(self.fold,),
                                                        checkpoint_name=checkpoint)
        labels = self._pred.dataset_json.get("labels", {})
        want = {"background": 0, "liver": 1, "gallbladder": 2, "vessel": 3, "kidney": 4, "spleen": 5}
        if any(labels.get(k) != v for k, v in want.items()):
            raise ValueError(f"{self.model_dir}: dataset.json labels {labels} != the 6-class scheme {want}")
        self.trainer_name = self._pred.trainer_name
        self.name = f"plug:nnunet:{self.trainer_name}"

    def _segment6(self, img01):
        x = np.clip(np.asarray(img01, np.float32), 0.0, 1.0) * 255.0
        x = np.round(x).astype(np.float32)[None, None]          # (c=1, 1, H, W), 0..255 like the PNG
        seg = self._pred.predict_single_npy_array(x, {"spacing": [999.0, 1.0, 1.0]}, None, None, False)
        seg = np.asarray(seg)
        seg = seg[0] if seg.ndim == 3 else seg
        if seg.shape != np.asarray(img01).shape:
            raise RuntimeError(f"nnU-Net output {seg.shape} != input {np.asarray(img01).shape}")
        return seg.astype(np.int64)

    def describe(self):
        plans = os.path.join(self.model_dir, "plans.json")
        return {"slot": "mask_source", "kind": "nnunet", "method": "nnU-Net v2 2d", "citation": CITATIONS["nnunet"],
                "privileged": False, "trainer": self.trainer_name, "fold": self.fold,
                "weights": {"path": self.ckpt_path, "md5": _md5(self.ckpt_path)},
                "plans": {"path": plans, "md5": _md5(plans)}, "use_mirroring": self.use_mirroring,
                "device": str(self.device), "packages": _pkg_versions("nnunetv2", "torch"),
                "preprocessing": "nnunetv2 predictor (crop to nonzero, z-score, sliding window); input 0..255"}


# =========================================================================== CT-label tap (privileged)
_TAP = threading.local()


def install_ct_label_tap():
    """Process-local, opt-in instrumentation so a PRIVILEGED box-prompted mask source can read the CT
    label of the frame it segments, without editing view_match.py. It wraps view_match.curvi_render and
    view_match.warp_to_sector (module attributes; every caller looks them up at call time) so that:
      * inside a curvi_render call, the FIRST warp_to_sector result is the true-pose sector label
        (curvi_render warps the true slice before any label-control slice), and _US.segment(tex) in that
        same call sees it;
      * after a curvi_render call, the returned frame object is remembered with that label, so a caller
        that segments the returned frame itself (usbg.plug.viewmatch_env) also gets it.
    Images and labels are unchanged; the wrapped functions return exactly what the originals return.
    Idempotent. Never installed by the v2 or v2.1 queues."""
    from .. import view_match as VM
    if getattr(VM.curvi_render, "_usbg_plug_tap", False):
        return
    orig_render, orig_warp = VM.curvi_render, VM.warp_to_sector

    def warp_tap(img, label, p=VM.SECTOR):
        out = orig_warp(img, label, p)
        st = getattr(_TAP, "stack", None)
        if st and st[-1] is None:
            st[-1] = np.asarray(out[1], np.int64)
        return out

    def render_tap(*a, **kw):
        st = _TAP.__dict__.setdefault("stack", [])
        st.append(None)
        try:
            img, lab = orig_render(*a, **kw)
            _TAP.last = (img, st[-1])
            return img, lab
        finally:
            st.pop()

    warp_tap._usbg_plug_tap = True
    render_tap._usbg_plug_tap = True
    render_tap.__wrapped__ = orig_render
    VM.warp_to_sector, VM.curvi_render = warp_tap, render_tap


def current_ct_label(img01=None):
    """The CT sector label (canonical ids) of the frame being segmented: the one inside the running
    curvi_render call, else the label of the frame object the last curvi_render returned (identity
    check). Raises if neither applies (a privileged source must never guess)."""
    st = getattr(_TAP, "stack", None)
    if st and st[-1] is not None:
        return st[-1]
    last = getattr(_TAP, "last", None)
    if img01 is not None and last is not None and last[0] is img01 and last[1] is not None:
        return last[1]
    raise RuntimeError("privileged (prompt=ct) mask source called without a CT label: install_ct_label_tap() "
                       "and segment only frames rendered by view_match.curvi_render")


# =========================================================================== SAM / MedSAM box prompts
def boxes_from_mask(mask_repo, organs=MATCH_ORGANS_REPO, min_px=20, mode="component", max_boxes=5):
    """[(organ_id, [x0, y0, x1, y1])] in pixel coordinates (x1, y1 inclusive, MedSAM's convention), one
    per connected component (8-connected, >= min_px) of each organ, largest first, at most max_boxes per
    organ; mode='organ' gives one box around all pixels of the organ."""
    from scipy import ndimage
    m_all = np.asarray(mask_repo)
    out = []
    for o in organs:
        m = m_all == o
        if int(m.sum()) < min_px:
            continue
        if mode == "organ":
            comps = [m]
        else:
            lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            order = [int(i) for i in np.argsort(-sizes, kind="stable") if sizes[i] >= min_px][:max_boxes]
            comps = [lab == (i + 1) for i in order]
        for c in comps:
            ys, xs = np.nonzero(c)
            out.append((int(o), [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
    return out


def _sam_model(weights, device):
    import torch
    from segment_anything import sam_model_registry
    m = sam_model_registry["vit_b"](checkpoint=None)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


class SamBoxMaskSource(_PlugMask):
    """SAM ViT-B (variant='sam', official SamPredictor) or MedSAM ViT-B (variant='medsam', MedSAM's
    own inference recipe) with box prompts (see the module docstring)."""

    kind = "sam"

    def __init__(self, variant, weights, prompt="ct", prompt_source=None, device=None, min_px=20, max_boxes=5,
                 box_mode="component", cache_size=0):
        super().__init__(cache_size)
        if variant not in ("sam", "medsam"):
            raise ValueError("variant must be 'sam' or 'medsam'")
        if prompt not in ("ct", "gt", "seg"):
            raise ValueError("prompt must be 'ct', 'gt' or 'seg'")
        if prompt == "seg" and prompt_source is None:
            raise ValueError("prompt='seg' needs prompt_source (a MaskSource)")
        self.variant, self.prompt, self.prompt_source = variant, prompt, prompt_source
        self.kind = variant
        self.privileged = prompt in ("ct", "gt")
        self.ckpt_path = os.path.abspath(weights)
        self.device = _device(device)
        self.min_px, self.max_boxes, self.box_mode = int(min_px), int(max_boxes), box_mode
        self._model = _sam_model(self.ckpt_path, self.device)
        if variant == "sam":
            from segment_anything import SamPredictor
            self._pred = SamPredictor(self._model)
        self._gt = {}
        self._n_boxes = []
        self.name = f"plug:{variant}:{prompt}"

    # ---- prompts
    def register_prompts(self, images, labels6):
        """prompt='gt': register ground-truth 6-class labels for frames (keyed by the frame bytes)."""
        for im, lb in zip(np.asarray(images, np.float32), np.asarray(labels6)):
            self._gt[hashlib.sha1(np.ascontiguousarray(im).tobytes()).hexdigest()] = to_repo(lb.astype(np.int64))

    def _prompt_mask(self, img01):
        if self.prompt == "ct":
            return canon_to_six_repo(current_ct_label(img01))
        if self.prompt == "gt":
            k = hashlib.sha1(np.ascontiguousarray(np.asarray(img01, np.float32)).tobytes()).hexdigest()
            if k not in self._gt:
                raise KeyError("prompt='gt': frame not registered (register_prompts)")
            return self._gt[k]
        return np.asarray(self.prompt_source.segment(img01), np.int64)

    def segment(self, img01):
        # prompt=ct must see the caller's frame object (identity check in current_ct_label)
        if self.prompt == "ct":
            t0 = time.perf_counter()
            out = self._segment_repo(img01)
            self._times.append((time.perf_counter() - t0) * 1000.0)
            return out
        return super().segment(img01)

    def segment6(self, img01):
        if self.prompt == "ct":
            return canon_to_six(self.segment(img01))
        return super().segment6(img01)

    def _segment6(self, img01):
        return canon_to_six(self._segment_repo(img01))

    def _segment_repo(self, img01):
        img = np.asarray(img01, np.float32)
        H, W = img.shape
        boxes = boxes_from_mask(self._prompt_mask(img01), min_px=self.min_px, mode=self.box_mode,
                                max_boxes=self.max_boxes)
        self._n_boxes.append(len(boxes))
        out = np.zeros((H, W), np.int64)
        if not boxes:
            return out
        organs = np.asarray([o for o, _ in boxes], np.int64)
        xyxy = np.asarray([b for _, b in boxes], np.float32)
        probs = self._probs_sam(img, xyxy) if self.variant == "sam" else self._probs_medsam(img, xyxy)
        best = probs.argmax(0)
        pbest = np.take_along_axis(probs, best[None], 0)[0]
        fg = pbest > 0.5
        out[fg] = organs[best[fg]]
        return out

    def _u8_rgb(self, img):
        u8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.repeat(u8[..., None], 3, axis=-1)

    def _probs_sam(self, img, xyxy):
        import torch
        rgb = self._u8_rgb(img)
        with torch.no_grad():
            self._pred.set_image(rgb, image_format="RGB")
            tb = torch.as_tensor(xyxy, dtype=torch.float32, device=self._pred.device)
            tb = self._pred.transform.apply_boxes_torch(tb, rgb.shape[:2])
            logits, _, _ = self._pred.predict_torch(None, None, boxes=tb, multimask_output=False, return_logits=True)
            return torch.sigmoid(logits[:, 0]).float().cpu().numpy()

    def _probs_medsam(self, img, xyxy):
        """MedSAM_Inference.py (bowang-lab/MedSAM): bicubic resize to 1024 x 1024 (skimage, order 3,
        preserve_range, anti_aliasing), min-max to [0, 1], image_encoder, prompt_encoder(boxes in 1024
        coordinates), mask_decoder(multimask_output=False), sigmoid, bilinear resize to (H, W)."""
        import torch
        import torch.nn.functional as F
        from skimage import transform
        H, W = img.shape
        # the frame is gray (3 identical channels), so resizing the single channel and replicating it is
        # bit-identical to MedSAM's 3-channel resize (checked: max abs difference 0) and ~30x faster
        g = self._u8_rgb(img)[..., 0]
        g = transform.resize(g, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        x = np.repeat(g[..., None], 3, axis=-1)
        x = (x - x.min()) / np.clip(x.max() - x.min(), a_min=1e-8, a_max=None)
        m = self._model
        with torch.no_grad():
            t = torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            emb = m.image_encoder(t)
            b1024 = xyxy / np.array([W, H, W, H], np.float32) * 1024.0
            bt = torch.as_tensor(b1024, dtype=torch.float32, device=self.device)[:, None, :]
            sparse, dense = m.prompt_encoder(points=None, boxes=bt, masks=None)
            low, _ = m.mask_decoder(image_embeddings=emb, image_pe=m.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                    multimask_output=False)
            p = F.interpolate(torch.sigmoid(low), size=(H, W), mode="bilinear", align_corners=False)
            return p[:, 0].float().cpu().numpy()

    def timing(self):
        t = super().timing()
        if self._n_boxes:
            t["boxes_per_frame_mean"] = round(float(np.mean(self._n_boxes)), 3)
            t["frames_without_box"] = int(np.sum(np.asarray(self._n_boxes) == 0))
        return t

    def describe(self):
        import torch
        d = {"slot": "mask_source", "kind": self.variant, "method": f"{'SAM' if self.variant == 'sam' else 'MedSAM'} "
             f"ViT-B, box prompts from {self.prompt}", "citation": CITATIONS[self.variant],
             "privileged": self.privileged,
             "privileged_reason": ("box prompts are computed from the CT label of the frame" if self.prompt == "ct" else
                                   "box prompts are computed from the ground-truth label" if self.prompt == "gt" else None),
             "prompt": self.prompt, "box_mode": self.box_mode, "min_px": self.min_px, "max_boxes": self.max_boxes,
             "weights": {"path": self.ckpt_path, "md5": _md5(self.ckpt_path), "manifest": _manifest_entry(self.ckpt_path)},
             "device": str(self.device), "packages": dict(_pkg_versions("segment_anything", "scikit-image"),
                                                          torch=torch.__version__),
             "preprocessing": ("SamPredictor (ResizeLongestSide 1024, pixel mean/std, pad)" if self.variant == "sam" else
                               "MedSAM_Inference: skimage bicubic 1024x1024, min-max [0,1], image_encoder")}
        if self.prompt_source is not None:
            ps = self.prompt_source
            d["prompt_source"] = ps.describe() if hasattr(ps, "describe") else \
                {"ckpt_path": getattr(ps, "ckpt_path", None), "md5": _md5(getattr(ps, "ckpt_path", None))}
        return d


def canon_to_six_repo(lab):
    """Canonical CT label (0..12) -> repo ontology {0,1,2,3,4,7} (non-target organs -> background)."""
    return to_repo(canon_to_six(lab))


class _UNetPrompt:
    """The gym's own compact U-Net (usbg.us_segmenter) as a prompt source for the SAM refiner."""

    def __init__(self, ckpt, device=None):
        from ..us_segmenter import load_segmenter
        self._s = load_segmenter(ckpt, device=device)
        self.ckpt_path = self._s.ckpt_path

    def segment(self, img01):
        return self._s.segment(img01)

    def describe(self):
        return {"kind": "unet", "method": "usbg compact U-Net (stage-matched v2 segmenter)",
                "weights": {"path": self.ckpt_path, "md5": _md5(self.ckpt_path)}}


# =========================================================================== factory
def parse_spec(spec: str) -> dict:
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"mask plug spec part {part!r} is not key=value (spec {spec!r})")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    if "kind" not in out:
        raise ValueError(f"mask plug spec {spec!r} has no kind=")
    return out


def load_mask_source(spec, device=None, cache_size=0):
    """Build an adapter from a spec string (module docstring) or dict."""
    s = parse_spec(spec) if isinstance(spec, str) else dict(spec)
    dev = s.get("device", device)
    kind = s["kind"]
    if kind == "tv":
        return TorchvisionMaskSource(s["ckpt"], device=dev, cache_size=cache_size)
    if kind == "nnunet":
        return NnUNetMaskSource(s["model"], fold=int(s.get("fold", 0)),
                                checkpoint=s.get("checkpoint", "checkpoint_final.pth"), device=dev,
                                use_mirroring=s.get("mirror", "0") in ("1", "true", "yes"), cache_size=cache_size)
    if kind in ("sam", "medsam"):
        prompt = s.get("prompt", "ct")
        ps = _UNetPrompt(s["prompt_ckpt"], device=_device(dev)) if prompt == "seg" else None
        return SamBoxMaskSource(kind, s["weights"], prompt=prompt, prompt_source=ps, device=dev,
                                min_px=int(s.get("min_px", 20)), max_boxes=int(s.get("max_boxes", 5)),
                                box_mode=s.get("box_mode", "component"), cache_size=cache_size)
    if kind == "samus":                   # SAMUS: usbg.plug.samus_mask, imported only for this kind
        from .samus_mask import load_samus_source
        return load_samus_source(s, device=dev, cache_size=cache_size)
    raise ValueError(f"unknown mask plug kind {kind!r} (tv, nnunet, sam, medsam, samus)")


def bind_mask_source(spec, device=None):
    """Build the adapter, install the CT-label tap if it is CT-prompted, export US_SEG_CKPT (the run
    guards require it for a learned mask) and bind it with view_match.set_mask_source. -> adapter."""
    src = load_mask_source(spec, device=device)
    if getattr(src, "prompt", None) == "ct":
        install_ct_label_tap()
    os.environ["US_SEG_CKPT"] = src.ckpt_path
    from .. import view_match as VM
    VM.set_mask_source(src)
    return src


__all__ = ["ANALYSIS_STATUS", "TV_ARCHS", "CITATIONS", "TVSegmenter", "build_tv_net", "TorchvisionMaskSource",
           "NnUNetMaskSource", "SamBoxMaskSource", "boxes_from_mask", "install_ct_label_tap", "current_ct_label",
           "load_mask_source", "bind_mask_source", "parse_spec", "to_repo", "canon_to_six"]
