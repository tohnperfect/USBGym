"""usbg.plug.samus_mask: SAMUS (an ultrasound-adapted SAM ViT-B) plugged in as the gym's mask source
(v2.1 pluggability study, exploratory). OPT-IN only: no v2 job and no round-1 v2.1 job imports this module;
only the round-2 SAMUS jobs (outputs/v2_1/jobs/plug_samus.json) do. It is reached through usbg.plug.masks.load_mask_source / bind_mask_source with ``kind=samus`` (the
gym rungs, scripts/plug/eval_plug_mask.py, and segtrain's scripts/plug/eval_plug_seg.py).

Model
  SAMUS ViT-B (Lin et al., MICCAI 2024, doi:10.1007/978-3-031-72111-3_3; preprint arXiv:2309.06824):
  SAM ViT-B with a parallel CNN branch, adapters and a 256 x 256 encoder input, trained on US30K.
  Official code (github.com/xianlin7/SAMUS, MIT) at commit 030ed07, used from a git clone by path;
  nothing is pip installed. Weights: SAMUS.pth from the README link (see the weights MANIFEST.json).

Preprocessing (SAMUS's own test-time recipe, utils/data_us.py + models/.../samus.py)
  The frame is quantized to uint8, resized with PIL bilinear to 256 x 256 (torchvision 0.9's F.resize
  on a PIL image, as in JointTransform2D), scaled to [0, 1] (F.to_tensor) and given as ONE channel to
  the image encoder, which repeats it to 3 channels. No mean/std normalization: Samus.forward calls
  image_encoder directly. The mask decoder's 128 x 128 logits are resized bilinearly to 256 x 256
  (Samus.forward), passed through a sigmoid, and resized bilinearly to the frame size.

Prompts (prompt=..., the same sources as the SAM / MedSAM adapters in usbg.plug.masks)
  ct    the CT label slice of the SAME frame. PRIVILEGED (needs install_ct_label_tap, which
        bind_mask_source installs).
  gt    a ground-truth label registered with register_prompts(). PRIVILEGED; offline evaluation only.
  seg   the mask of the stage-and-seed v2 compact U-Net (prompt_ckpt=<us_seg_<stage>_s<seed>.pt>):
        non-privileged, SAMUS as a refiner of the gym's own segmenter. No CT at run time.
  Components are those of usbg.plug.masks.boxes_from_mask (8-connected, >= min_px pixels, largest
  first, at most max_boxes per organ, organs in view_match.MATCH_ORGANS). Each component mask is resized
  NEAREST to 256 x 256 (SAMUS resizes its labels that way) and turned into a prompt by SAMUS's own
  test-time prompt functions (utils/data_us.py, re-implemented here because that module imports cv2,
  batchgenerators and pandas):
    ptype=box    fixed_bbox: [min x, min y, max x, max y] of the component (default; the SAM / MedSAM
                 rungs' prompt geometry, so the three adapters differ only in the model).
    ptype=point  fixed_click: one positive point, the middle pixel of the component in raster order.
                 This is SAMUS's NATIVE protocol: it was trained and evaluated with point prompts only
                 (Samus.forward ignores its bbox argument).
  Box prompts are valid input: SAMUS keeps SAM ViT-B's prompt encoder and mask decoder frozen during its
  training (Samus.__init__ sets requires_grad False), so its box pathway is SAM's own. The fusion is the
  SAM / MedSAM adapters' rule: each component's probability map is labeled with its organ, overlaps go
  to the highest probability, probability <= 0.5 is background.

torch 2.x shim (this file only; the clone is not modified)
  The SAMUS requirements pin torch 1.8 and Python 3.8. Under torch 2.12.1 / Python 3.13 the model code
  runs unchanged; the only obstacles are three imports whose names the loaded modules never use
  (``from tkinter import X``, ``from turtle import shape``, ``from einops import rearrange``; einops is
  not in the shared .venv). During the one-time import, any of these modules that is not already
  imported is replaced by an inert stub and removed again afterwards; an AST check refuses the shim if a
  stubbed name is ever read in the loaded files. The package is imported under the private name
  ``_usbg_samus_vendor`` (no sys.path change), so its generic top-level name ``models`` cannot clash
  with other code. Only build_sam_us and the modeling subpackage are loaded (not the automatic mask
  generator). The checkpoint is loaded with torch.load(weights_only=True) and a strict state-dict load.

Spec strings (comma-separated key=value)
  kind=samus,weights=/abs/samus/SAMUS.pth,prompt=ct
  kind=samus,weights=/abs/samus/SAMUS.pth,prompt=seg,prompt_ckpt=/abs/us_seg_sonospade_s0.pt
  kind=samus,weights=/abs/samus/SAMUS.pth,prompt=gt,ptype=point
Optional keys: device (cpu | mps), ptype (box | point, default box), min_px (20), max_boxes (5),
box_mode (component | organ), code (the SAMUS clone; default $USBG_SAMUS_CODE or the plugin weights dir).
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.machinery
import os
import sys
import time
import types

import numpy as np

from . import masks as PM
from .masks import ANALYSIS_STATUS, MATCH_ORGANS_REPO, SamBoxMaskSource, _PlugMask

WEIGHTS_DIR = os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights"))
DEFAULT_CODE = os.environ.get("USBG_SAMUS_CODE") or os.path.join(WEIGHTS_DIR, "samus_code")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "samus", "SAMUS.pth")
SAMUS_COMMIT = "030ed07cebe44bd40a00e97c9044660b7dc12523"
ENCODER_SIZE = 256
CITATION = ("Lin, Xiang, Yu, Yan, Beyond Adapting SAM: Towards End-to-End Ultrasound Image Segmentation via Auto "
            "Prompting, MICCAI 2024, LNCS, pp. 24-34, doi:10.1007/978-3-031-72111-3_3 (SAMUS and AutoSAMUS); "
            "preprint arXiv:2309.06824, whose v1 (2023) is titled 'SAMUS: Adapting Segment Anything Model for "
            "Clinically-Friendly and Generalizable Ultrasound Image Segmentation'")
PROTOCOL_NOTE = ("SAMUS was trained and evaluated with one positive point per object (fixed_click at test "
                 "time); its prompt encoder and mask decoder are SAM ViT-B's, frozen during SAMUS training, "
                 "so box prompts use SAM's own box pathway. ptype=point is SAMUS's native protocol.")

# (ii) of the admission gate (usbg.plug.admission.PRETRAINING format), used by scripts/plug/admission_gate_samus.py
PRETRAINING = {
    "corpus": "SAM ViT-B (SA-1B natural photographs) adapted on US30K: about 30K ultrasound images and 69K masks from "
              "7 public datasets named in the SAMUS README: TN3K, DDTI, TG3K (thyroid), BUSI, UDIAT (breast), CAMUS, "
              "HMC-QU (echocardiography)",
    "overlap": {"vitale_aul_kidneyus": "not expected: no abdominal, liver or kidney dataset in the README list",
                "heldout_and_cohort_ct": "not expected (no CT)"},
    "status": "documented from the SAMUS README dataset list (the MICCAI 2024 paper's dataset table not re-read)",
    "real_frame_dice_reportable": True,
}

_PKG = "_usbg_samus_vendor"
# module -> names imported from it by the loaded SAMUS files (none of them is used)
_STUBS = {"tkinter": ("X",), "turtle": ("shape",), "einops": ("rearrange",)}
_SHIMS_USED = []


# =========================================================================== import shim
def _git_commit(code_dir):
    """HEAD commit of the clone, read from .git without running git (None if unreadable)."""
    g = os.path.join(code_dir, ".git")
    try:
        head = open(os.path.join(g, "HEAD")).read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        p = os.path.join(g, ref)
        if os.path.isfile(p):
            return open(p).read().strip()
        pr = os.path.join(g, "packed-refs")
        if os.path.isfile(pr):
            for line in open(pr):
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def _check_stub_names_unused(pkg_dir):
    """Refuse the shim if any loaded SAMUS file READS a name it imports from a stubbed module."""
    mod_dir = os.path.join(pkg_dir, "modeling")
    files = [os.path.join(pkg_dir, "build_sam_us.py")] + \
        [os.path.join(mod_dir, f) for f in sorted(os.listdir(mod_dir)) if f.endswith(".py")]
    for fn in files:
        tree = ast.parse(open(fn).read(), filename=fn)
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _STUBS:
                bound |= {a.asname or a.name for a in node.names}
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in _STUBS:
                        bound.add(a.asname or a.name.split(".")[0])
        if not bound:
            continue
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        used |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        bad = bound & used
        if bad:
            raise RuntimeError(f"SAMUS shim: {fn} uses {sorted(bad)} from a stubbed module; install the real "
                               "package instead of the stub")


class _Inert:
    def __init__(self, qual):
        self._qual = qual

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._qual} is an import-time stub of usbg.plug.samus_mask (never used by SAMUS)")


@contextlib.contextmanager
def _import_stubs():
    added = []
    for mod, names in _STUBS.items():
        if mod in sys.modules:
            continue
        stub = types.ModuleType(mod)
        stub.__doc__ = "usbg.plug.samus_mask import-time stub"
        for n in names:
            setattr(stub, n, _Inert(f"{mod}.{n}"))
        sys.modules[mod] = stub
        added.append(mod)
    try:
        yield added
    finally:
        for mod in added:
            if sys.modules.get(mod) is not None and getattr(sys.modules[mod], "__doc__", "") == \
                    "usbg.plug.samus_mask import-time stub":
                del sys.modules[mod]


def import_samus(code_dir=None):
    """-> the SAMUS build_sam_us module, imported once under the private package name _usbg_samus_vendor."""
    key = f"{_PKG}.build_sam_us"
    if key in sys.modules:
        return sys.modules[key]
    code_dir = os.path.abspath(code_dir or DEFAULT_CODE)
    pkg_dir = os.path.join(code_dir, "models", "segment_anything_samus")
    if not os.path.isfile(os.path.join(pkg_dir, "build_sam_us.py")):
        raise FileNotFoundError(f"SAMUS code not found at {pkg_dir} (clone github.com/xianlin7/SAMUS)")
    _check_stub_names_unused(pkg_dir)
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [pkg_dir]
    pkg.__file__ = os.path.join(pkg_dir, "__init__.py")
    pkg.__package__ = _PKG
    pkg.__spec__ = importlib.machinery.ModuleSpec(_PKG, None, is_package=True)
    pkg.__spec__.submodule_search_locations = [pkg_dir]
    pkg.__usbg_code_dir__ = code_dir
    sys.modules[_PKG] = pkg                        # the package __init__ (automatic mask generator) is NOT run
    try:
        with _import_stubs() as added:
            mod = importlib.import_module(key)
    except Exception:
        for k in [k for k in sys.modules if k == _PKG or k.startswith(_PKG + ".")]:
            del sys.modules[k]
        raise
    _SHIMS_USED[:] = [f"{m}: inert import-time stub of {', '.join(_STUBS[m])} (imported, never used)" for m in added]
    return mod


def build_samus(weights, device="cpu", code_dir=None):
    """SAMUS ViT-B, encoder input 256, strict load of ``weights`` (torch.load weights_only=True)."""
    import torch
    B = import_samus(code_dir)
    m = B.samus_model_registry["vit_b"](args=types.SimpleNamespace(encoder_input_size=ENCODER_SIZE),
                                        checkpoint=None)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}   # test.py's DataParallel rule
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


# =========================================================================== prompts
def components_from_mask(mask_repo, organs=MATCH_ORGANS_REPO, min_px=20, mode="component", max_boxes=5):
    """[(organ_id, bool component mask)] with exactly the selection of masks.boxes_from_mask."""
    from scipy import ndimage
    m_all = np.asarray(mask_repo)
    out = []
    for o in organs:
        m = m_all == o
        if int(m.sum()) < min_px:
            continue
        if mode == "organ":
            out.append((int(o), m))
            continue
        lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        order = [int(i) for i in np.argsort(-sizes, kind="stable") if sizes[i] >= min_px][:max_boxes]
        out.extend((int(o), lab == (i + 1)) for i in order)
    return out


def _resize_nearest(mask_bool, size=ENCODER_SIZE):
    from PIL import Image
    im = Image.fromarray(np.asarray(mask_bool, np.uint8))
    return np.asarray(im.resize((size, size), Image.NEAREST)) > 0


def fixed_bbox(mask_bool):
    """SAMUS utils/data_us.fixed_bbox: [min x, min y, max x, max y] of the mask (None if empty)."""
    ys, xs = np.nonzero(mask_bool)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def fixed_click(mask_bool):
    """SAMUS utils/data_us.fixed_click: the middle element of np.argwhere(mask) (raster order) as (x, y)."""
    idx = np.argwhere(mask_bool)
    if not len(idx):
        return None
    y, x = idx[len(idx) // 2]
    return [int(x), int(y)]


def _fallback_prompt(comp, H, W, ptype, S=ENCODER_SIZE):
    """A component lost by the NEAREST resize (frames larger than 256 only): its native prompt, scaled."""
    sx, sy = S / float(W), S / float(H)
    if ptype == "box":
        x0, y0, x1, y1 = fixed_bbox(comp)
        return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
    x, y = fixed_click(comp)
    return [x * sx, y * sy]


# =========================================================================== adapter
class SamusMaskSource(SamBoxMaskSource):
    """SAMUS ViT-B as a MaskSource (see the module docstring). Shares the prompt plumbing of
    masks.SamBoxMaskSource (register_prompts, the CT-label prompt, the U-Net prompt, segment / segment6,
    caching, timing); only the model, its preprocessing and the prompt geometry differ."""

    kind = "samus"

    def __init__(self, weights=DEFAULT_WEIGHTS, prompt="ct", prompt_source=None, device=None, ptype="box",
                 min_px=20, max_boxes=5, box_mode="component", cache_size=0, code_dir=None):
        _PlugMask.__init__(self, cache_size)
        if prompt not in ("ct", "gt", "seg"):
            raise ValueError("prompt must be 'ct', 'gt' or 'seg'")
        if prompt == "seg" and prompt_source is None:
            raise ValueError("prompt='seg' needs prompt_source (a MaskSource)")
        if ptype not in ("box", "point"):
            raise ValueError("ptype must be 'box' or 'point'")
        self.variant = "samus"
        self.kind = "samus"
        self.prompt, self.prompt_source, self.ptype = prompt, prompt_source, ptype
        self.privileged = prompt in ("ct", "gt")
        self.ckpt_path = os.path.abspath(weights)
        self.code_dir = os.path.abspath(code_dir or DEFAULT_CODE)
        self.device = PM._device(device)
        self.min_px, self.max_boxes, self.box_mode = int(min_px), int(max_boxes), box_mode
        t0 = time.perf_counter()
        self._model = build_samus(self.ckpt_path, self.device, self.code_dir)
        self.build_sec = round(time.perf_counter() - t0, 2)
        self.code_commit = _git_commit(self.code_dir)
        self._gt = {}
        self._n_boxes = []
        self.name = f"plug:samus:{prompt}" + ("" if ptype == "box" else ":point")

    # ---- model input
    def _encoder_input(self, img):
        from PIL import Image
        u8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        S = ENCODER_SIZE
        if u8.shape != (S, S):
            u8 = np.asarray(Image.fromarray(u8).resize((S, S), Image.BILINEAR))
        return u8.astype(np.float32) / 255.0

    def _prompts(self, comps, H, W):
        out = []
        for _, c in comps:
            c256 = _resize_nearest(c) if c.shape != (ENCODER_SIZE, ENCODER_SIZE) else c
            p = fixed_bbox(c256) if self.ptype == "box" else fixed_click(c256)
            out.append(p if p is not None else _fallback_prompt(c, H, W, self.ptype))
        return np.asarray(out, np.float32)

    def probs(self, img01, comps):
        """(n, H, W) foreground probabilities, one per component prompt."""
        import torch
        import torch.nn.functional as F
        img = np.asarray(img01, np.float32)
        H, W = img.shape
        S = ENCODER_SIZE
        x = self._encoder_input(img)
        pr = self._prompts(comps, H, W)
        m = self._model
        with torch.no_grad():
            t = torch.from_numpy(x)[None, None].to(self.device)
            emb = m.image_encoder(t)                                   # (1, 256, 32, 32)
            if self.ptype == "box":
                bt = torch.as_tensor(pr, dtype=torch.float32, device=self.device)          # (n, 4)
                sparse, dense = m.prompt_encoder(points=None, boxes=bt, masks=None)
            else:
                pc = torch.as_tensor(pr, dtype=torch.float32, device=self.device)[:, None, :]   # (n, 1, 2)
                pl = torch.ones((len(pr), 1), dtype=torch.int, device=self.device)
                sparse, dense = m.prompt_encoder(points=(pc, pl), boxes=None, masks=None)
            low, _ = m.mask_decoder(image_embeddings=emb[0], image_pe=m.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                    multimask_output=False)            # (n, 1, 128, 128)
            p = torch.sigmoid(F.interpolate(low, (S, S), mode="bilinear", align_corners=False))
            if (H, W) != (S, S):
                p = F.interpolate(p, (H, W), mode="bilinear", align_corners=False)
            return p[:, 0].float().cpu().numpy()

    def _segment_repo(self, img01):
        img = np.asarray(img01, np.float32)
        H, W = img.shape
        comps = components_from_mask(self._prompt_mask(img01), min_px=self.min_px, mode=self.box_mode,
                                     max_boxes=self.max_boxes)
        self._n_boxes.append(len(comps))
        out = np.zeros((H, W), np.int64)
        if not comps:
            return out
        organs = np.asarray([o for o, _ in comps], np.int64)
        probs = self.probs(img, comps)
        best = probs.argmax(0)
        pbest = np.take_along_axis(probs, best[None], 0)[0]
        fg = pbest > 0.5
        out[fg] = organs[best[fg]]
        return out

    def describe(self):
        import torch
        d = {"slot": "mask_source", "kind": "samus",
             "method": f"SAMUS ViT-B (encoder {ENCODER_SIZE}), {self.ptype} prompts from {self.prompt}",
             "citation": CITATION, "privileged": self.privileged,
             "privileged_reason": ("prompts are computed from the CT label of the frame" if self.prompt == "ct" else
                                   "prompts are computed from the ground-truth label" if self.prompt == "gt" else None),
             "prompt": self.prompt, "prompt_type": self.ptype, "box_mode": self.box_mode, "min_px": self.min_px,
             "max_boxes": self.max_boxes,
             "weights": {"path": self.ckpt_path, "md5": PM._md5(self.ckpt_path),
                         "manifest": PM._manifest_entry(self.ckpt_path)},
             "code": {"path": self.code_dir, "commit": self.code_commit, "expected_commit": SAMUS_COMMIT,
                      "commit_ok": self.code_commit == SAMUS_COMMIT, "license": "MIT",
                      "url": "https://github.com/xianlin7/SAMUS", "import": f"private package {_PKG}"},
             "shims": list(_SHIMS_USED), "build_sec": self.build_sec, "device": str(self.device),
             "packages": dict(PM._pkg_versions("torchvision", "Pillow", "scipy"), torch=torch.__version__),
             "preprocessing": ("SAMUS data_us.py: uint8 gray, PIL bilinear resize to 256x256, [0,1], one channel "
                               "(no mean/std); low-res logits bilinear to 256 (Samus.forward), sigmoid, bilinear "
                               "to the frame size"),
             "prompt_geometry": (f"SAMUS fixed_{'bbox' if self.ptype == 'box' else 'click'} on each component "
                                 "mask resized NEAREST to 256x256"),
             "protocol_note": PROTOCOL_NOTE, "analysis_status": ANALYSIS_STATUS}
        if self.prompt_source is not None:
            ps = self.prompt_source
            d["prompt_source"] = ps.describe() if hasattr(ps, "describe") else \
                {"ckpt_path": getattr(ps, "ckpt_path", None), "md5": PM._md5(getattr(ps, "ckpt_path", None))}
        return d


def load_samus_source(s, device=None, cache_size=0):
    """Build a SamusMaskSource from a parsed spec dict (usbg.plug.masks.parse_spec)."""
    s = dict(s)
    if s.get("kind", "samus") != "samus":
        raise ValueError(f"load_samus_source: kind {s.get('kind')!r} is not 'samus'")
    known = {"kind", "weights", "prompt", "prompt_ckpt", "ptype", "device", "min_px", "max_boxes", "box_mode", "code"}
    extra = set(s) - known
    if extra:
        raise ValueError(f"samus spec: unknown keys {sorted(extra)} (known: {sorted(known)})")
    dev = s.get("device", device)
    prompt = s.get("prompt", "ct")
    ps = PM._UNetPrompt(s["prompt_ckpt"], device=PM._device(dev)) if prompt == "seg" else None
    return SamusMaskSource(s.get("weights", DEFAULT_WEIGHTS), prompt=prompt, prompt_source=ps, device=dev,
                           ptype=s.get("ptype", "box"), min_px=int(s.get("min_px", 20)),
                           max_boxes=int(s.get("max_boxes", 5)), box_mode=s.get("box_mode", "component"),
                           cache_size=cache_size, code_dir=s.get("code"))


__all__ = ["SamusMaskSource", "load_samus_source", "build_samus", "import_samus", "components_from_mask",
           "fixed_bbox", "fixed_click", "CITATION", "SAMUS_COMMIT", "ENCODER_SIZE", "DEFAULT_WEIGHTS"]
