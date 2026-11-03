"""usbg.plug.renderers: published ultrasound forward models plugged into the renderer slot.

v2.1 pluggability study (exploratory, opt-in). Nothing in the v2 or v2.1 queues imports this module,
and importing it changes nothing: a renderer becomes active only through ``install(kind)`` (or the
``plugged(kind)`` context manager), which the new scripts under ``scripts/plug/`` call.

Every renderer here maps a canonical label slice (N_LINES, N_SAMPLES) from ``usbg.slicer`` (rows =
scan lines over the 60 deg convex field of view, columns = depth samples) to a float32 B-mode in
[0, 1] of shape (N_SAMPLES, N_LINES), the contract of ``usbg.render_lotus.render``. The view-match
loop then resizes it to 128 x 128 and warps it into the measured Vitale sector
(``view_match.curvi_render``), so every plugged renderer is scored in the same sector geometry as the
in-house physics renderer, and the rest of the loop (texture stage, mask source, servo base, reward)
is unchanged.

Renderers (names accepted by ``install``)
  physics       the in-house physics renderer (reference; ``install("physics")`` is a no-op patch).
  lotus_reimpl  the in-repo LOTUS reimplementation (``render_lotus.render_bmode_lotus``), seeded by
                ``cfg.seed`` and with the global torch RNG state saved and restored around each call
                (the built-in calls ``torch.manual_seed``, which would otherwise reset the policy's
                sampling noise at every render during training). Its frames equal the built-in "lotus"
                renderer's frames for the same seed (the "prior LOTUS" row of the realism table).
  lotus_orig    the ORIGINAL LOTUS forward model, ``models/us_rendering_model.py`` of
                Velikova et al., MICCAI 2023 (third_party/lotus, MIT, commit 3d20ebb), loaded from its
                own file with no edit to it. As shipped it moves its tensors to CUDA at import and in
                every call (``.to(device='cuda')``); the shim redirects those moves to the requested
                device ("cpu" or "mps") through a scoped patch of ``torch.Tensor.to`` that is active only
                while the LOTUS module is imported and while its ``forward`` runs. On MPS, a float64
                tensor (its time-gain ramp, built from NumPy) is cast to float32 first because MPS has no
                float64; on CPU the arithmetic is exactly the original. The published default
                parameters are used (``default_param=True``; no LOTUS training on our data). LOTUS's own
                display warp (a 256 x 256 sector of its own geometry) is bypassed so the intensity map
                enters the usbg sector warp like every other renderer. The model implements the
                convolutional ray-tracing family of Salehi et al., MICCAI 2015, which has no separate
                author code, so this row covers both.
  simus         SIMUS (Garcia 2022) through PyMUST 0.1.9, CPU only, evaluation only (seconds per
                frame). A convex array with the testbed's footprint (radius 40 mm, 60 deg arc, 128
                elements; the C5-2v center frequency, bandwidth, element height and elevation focus)
                transmits one diverging wave (``n_tx`` > 1 compounds tilted waves); scatterers are drawn
                uniformly over the sector at ``density_per_mm2``, with Gaussian reflectivity scaled by the
                label's mean backscatter (the testbed's acoustic table, ``volume.ACOUSTIC_LOTUS``), and
                interface reflectors scaled by the Fresnel intensity coefficient; RF -> I/Q (rf2iq) ->
                delay-and-sum on the canonical scan-line grid (dasmtx) -> envelope -> time-gain
                compensation calibrated once on a homogeneous soft-tissue phantom -> log compression.
                SIMUS is a linear single-scattering simulator of a homogeneous medium (attenuation
                ``SimusConfig.attenuation`` = 0.54 dB/cm/MHz, the table's soft tissue); the tissue-dependent
                transmission loss (Fresnel transmission at interfaces, and the table's attenuation in excess
                of SIMUS's, in dB/cm/MHz at the center frequency) is folded into the reflectivity along each
                scan line (``shadow=True``), as point-scatterer simulators are commonly driven. At the
                default 2 scatterers/mm^2 the speckle is not fully developed (fewer than one scatterer per
                resolution cell); fully developed speckle (10-40 per mm^2) costs 5-20x more per frame. RF
                superposition is linear in the reflectivity, so the scatterers are split into a FIXED set of
                chunks (frames do not depend on the worker count) run on a persistent process pool, and the
                RF is summed.

Not run here (reported, not implemented): Field II (Jensen and Svendsen 1992) needs MATLAB; k-Wave
(Treeby and Cox 2010) through k-wave-python conflicts with the shared environment's pins and its fast
binary needs CUDA, while its NumPy backend takes about 81 s for a 64^3 grid on an Apple M1 (its README),
i.e. hours per B-mode frame, so it is not usable per step.

Determinism: every renderer here is a deterministic function of (label slice, seed). The seed comes
from ``cfg.seed`` (``render_lotus.RenderConfig``) unless a ``seed=`` keyword is passed, so the
view-match reward's same-pipeline rule (target and observation share one seed) holds. The other
RenderConfig knobs (per-label scales, PSF, gain, domain randomization) belong to the physics renderer
and are ignored by the plugged renderers (recorded as ``dr_knobs_ignored`` in ``plug_info``).
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np

PLUG_STATUS = "exploratory, v2.1 pluggability"

# ===================================================================================== metadata
CITATIONS = {
    "physics": "in-house physics renderer (usbg.render_lotus.render_bmode_physics)",
    "lotus_reimpl": ("Velikova et al., LOTUS, MICCAI 2023, LNCS 14220, 435-445, doi:10.1007/978-3-031-43907-0_42 "
                     "(in-repo reimplementation)"),
    "lotus_orig": ("Velikova et al., LOTUS, MICCAI 2023, LNCS 14220, 435-445, doi:10.1007/978-3-031-43907-0_42; "
                   "original code github.com/danivelikova/lotus (MIT), commit 3d20ebb; convolutional ray "
                   "tracing after Salehi et al., MICCAI 2015, doi:10.1007/978-3-319-24571-3_61"),
    "simus": ("Garcia, SIMUS: an open-source simulator for medical ultrasound imaging. Part I: theory & "
              "examples, Comput. Methods Programs Biomed. 218 (2022) 106726, doi:10.1016/j.cmpb.2022.106726; "
              "Cigier, Varray, Garcia, Part II, Comput. Methods Programs Biomed. 220 (2022) 106774, doi:10.1016/j.cmpb.2022.106774; "
              "PyMUST 0.1.9 (LGPL-3.0)"),
}
INFEASIBLE = {
    "field_ii": {"citation": "Jensen and Svendsen, IEEE TUFFC 39 (1992) 262-267, doi:10.1109/58.139123",
                 "reason": "needs MATLAB (no MATLAB or Octave on this machine); SIMUS stands in "
                           "(Cigier et al. 2022 compare SIMUS with Field II and k-Wave)"},
    "k_wave": {"citation": "Treeby and Cox, J. Biomed. Opt. 15 (2010) 021314, doi:10.1117/1.3360308",
               "reason": "k-wave-python 0.6.2 pins conflict with the shared environment; its fast binary "
                         "needs CUDA; the NumPy backend takes ~81 s for a 64^3 grid on an Apple M1 (README), "
                         "i.e. one simulation per scan line and hours per B-mode frame on CPU"},
}
ALIASES = {"lotus": "lotus_reimpl", "lotus_original": "lotus_orig", "pymust": "simus"}
PLUG_NAMES = ("lotus_reimpl", "lotus_orig", "simus")


def canonical_name(kind: str) -> str:
    k = ALIASES.get(str(kind), str(kind))
    if k not in ("physics", *PLUG_NAMES):
        raise ValueError(f"unknown plugged renderer {kind!r}; expected physics, {', '.join(PLUG_NAMES)} "
                         f"(aliases {ALIASES})")
    return k


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def plug_device(device=None):
    """The torch device for a plugged torch renderer: the explicit argument, else $USBG_PLUG_RENDER_DEVICE,
    else ``render_lotus.get_device()`` read at call time (so a scoped device pin applies)."""
    if device:
        return str(device)
    env = os.environ.get("USBG_PLUG_RENDER_DEVICE", "").strip()
    if env:
        return env
    from .. import render_lotus as R
    return R.get_device()


@contextlib.contextmanager
def _rng_isolated(seed):
    """Seed torch's global generators with ``seed`` for the body, then restore the CPU (and MPS) state, so a
    plugged renderer never moves the RNG stream a trainer's policy sampling draws from."""
    import torch
    cpu = torch.random.get_rng_state()
    mps = None
    try:
        if torch.backends.mps.is_available():
            mps = torch.mps.get_rng_state()
    except Exception:                                   # pragma: no cover
        mps = None
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def _leading_gap_np(lab):
    """(L,S) bool: BACKGROUND samples before a scan line first enters the body (render_lotus._leading_gap)."""
    inside = (lab != 0).astype(np.int32)
    return (np.cumsum(inside, axis=1) == 0) & (lab == 0)


# ===================================================================================== in-repo LOTUS
def render_lotus_reimpl(label_slice, *, device=None, cfg=None, seed=None, warp=False, **kw):
    """The in-repo LOTUS reimplementation with the global torch RNG isolated. float32 (S, L) in [0,1]."""
    from .. import render_lotus as R
    s = int(seed if seed is not None else (cfg.seed if cfg is not None else 0))
    with _rng_isolated(s):
        return R.render_bmode_lotus(label_slice, device=plug_device(device), warp=warp, seed=s)


# ===================================================================================== original LOTUS
# usbg canonical label -> LOTUS id. LOTUS ids (models/us_rendering_model.py): 2 lung, 3 fat, 4 vessel,
# 6 kidney, 8 muscle, 9 background, 11 liver, 12 soft tissue, 13 bone. Rule: a usbg label with its own LOTUS
# class keeps it; gallbladder (fluid) takes LOTUS's anechoic fluid class (vessel), as usbg's own table does;
# organs without a LOTUS class (GI, pancreas, spleen) take soft tissue.
LOTUS_IDS = (2, 3, 4, 6, 8, 9, 11, 12, 13)
USBG_TO_LOTUS_ORIG = {0: 9, 1: 11, 2: 4, 3: 4, 4: 6, 5: 12, 6: 12, 7: 12, 8: 8, 9: 3, 10: 13, 11: 2, 12: 12}
LOTUS_GEL_ID = 4        # coupling gel in the leading gap (anatomical seat): LOTUS's anechoic fluid class
_LOTUS_LUT = np.full(256, 12, np.int64)
for _u, _l in USBG_TO_LOTUS_ORIG.items():
    _LOTUS_LUT[_u] = _l


def lotus_dir():
    """The LOTUS checkout: $USBG_LOTUS_DIR, else <repo>/third_party/lotus, else the main checkout's
    third_party/lotus when this tree is a linked git working tree whose submodule is not checked out (read only)."""
    rel = os.path.join("models", "us_rendering_model.py")
    cands = []
    if os.environ.get("USBG_LOTUS_DIR"):
        cands.append(os.environ["USBG_LOTUS_DIR"])
    root = _repo_root()
    cands.append(os.path.join(root, "third_party", "lotus"))
    gitfile = os.path.join(root, ".git")
    if os.path.isfile(gitfile):                         # a linked working tree: ".git" is a file naming the main repository's git directory
        with open(gitfile) as fh:
            line = fh.read().strip()
        if line.startswith("gitdir:") and "/.git/" in line:
            main = line.split(":", 1)[1].strip().split("/.git/")[0]
            cands.append(os.path.join(main, "third_party", "lotus"))
    for c in cands:
        if os.path.exists(os.path.join(c, rel)):
            return os.path.abspath(c)
    raise FileNotFoundError(f"LOTUS models/us_rendering_model.py not found in {cands}; set USBG_LOTUS_DIR")


def lotus_source_info():
    d = lotus_dir()
    p = os.path.join(d, "models", "us_rendering_model.py")
    info = {"dir": d, "file": p, "sha256": _sha256(p), "license": "MIT"}
    try:
        import subprocess
        info["commit"] = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True,
                                        timeout=10).stdout.strip() or None
    except Exception:                                   # pragma: no cover
        info["commit"] = None
    return info


_REDIRECT_LOCK = threading.RLock()


@contextlib.contextmanager
def _cuda_redirect(device):
    """Scoped patch of ``torch.Tensor.to``: a 'cuda' target becomes ``device``; on MPS a float64 tensor is cast
    to float32 before the move (MPS has no float64). Active only inside the LOTUS import and forward."""
    import torch
    dev = torch.device(device)
    orig_to = torch.Tensor.to

    def _is_cuda(a):
        return (isinstance(a, str) and a.startswith("cuda")) or (isinstance(a, torch.device) and a.type == "cuda")

    def to(self, *args, **kw):
        hit = any(_is_cuda(a) for a in args) or _is_cuda(kw.get("device"))
        if not hit:
            return orig_to(self, *args, **kw)
        args = tuple(dev if _is_cuda(a) else a for a in args)
        if _is_cuda(kw.get("device")):
            kw["device"] = dev
        if dev.type == "mps" and self.dtype == torch.float64:
            self = orig_to(self, torch.float32)
        return orig_to(self, *args, **kw)

    with _REDIRECT_LOCK:
        torch.Tensor.to = to
        try:
            yield
        finally:
            torch.Tensor.to = orig_to


_LOTUS_MODELS: dict = {}


def _lotus_model(device):
    """(module, UltrasoundRendering) for ``device``: the original file executed under the CUDA redirect, so its
    module-level parameter tables live on ``device``. One module instance per device."""
    import importlib.util
    import types
    import torch
    key = str(torch.device(device))
    if key in _LOTUS_MODELS:
        return _LOTUS_MODELS[key]
    os.environ.setdefault("MPLBACKEND", "Agg")          # the file imports matplotlib.pyplot (debug plots only)
    path = os.path.join(lotus_dir(), "models", "us_rendering_model.py")
    name = "usbg_plug_lotus_orig_" + key.replace(":", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with _cuda_redirect(key), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
        model = mod.UltrasoundRendering(types.SimpleNamespace(debug=False), default_param=True)
    model.warp_img = lambda img: img                    # bypass LOTUS's own display warp (see module doc)
    model.eval()
    _LOTUS_MODELS[key] = (mod, model)
    return mod, model


def lotus_orig_labels(label_slice, gel=None):
    """usbg label slice (L,S) -> LOTUS ids (L,S) int64; under gel coupling the leading gap is LOTUS_GEL_ID."""
    lab = np.asarray(label_slice).astype(np.int64)
    ids = _LOTUS_LUT[np.clip(lab, 0, 255)]
    if gel is None:
        from ..seatmode import gel_coupling
        gel = gel_coupling()
    if gel:
        ids = np.where(_leading_gap_np(lab), LOTUS_GEL_ID, ids)
    return ids


def render_lotus_original(label_slice, *, device=None, cfg=None, seed=None, warp=False, gel=None, **kw):
    """The original LOTUS forward model (default parameters) on CPU or MPS. float32 (S, L) in [0,1]."""
    import torch
    dev = plug_device(device)
    _mod, model = _lotus_model(dev)
    ids = lotus_orig_labels(label_slice, gel)
    s = int(seed if seed is not None else (cfg.seed if cfg is not None else 0))
    with _rng_isolated(s), torch.no_grad(), _cuda_redirect(dev), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model(torch.as_tensor(ids, dtype=torch.long, device=dev))   # rot90(intensity, 3): warp bypassed
        inten = torch.rot90(out, 1)                                         # undo -> (L, S)
        img = inten.t().contiguous().float().cpu().numpy()                  # (S, L)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if warp:
        from .. import render_lotus as R
        img = R.warp_curvilinear(img)
    return img


# ===================================================================================== SIMUS (PyMUST)
@dataclass(frozen=True)
class SimusConfig:
    probe: str = "C5-2v"              # PyMUST probe whose fc, bandwidth, height and elevation focus are kept
    n_elements: int = 128
    density_per_mm2: float = 2.0      # scatterers per mm^2 of the sector
    n_tx: int = 1                     # diverging waves (tilts spread over +-tilt_deg when > 1)
    tilt_deg: float = 10.0
    attenuation: float = 0.54         # dB/cm/MHz, homogeneous (SIMUS); = soft tissue in the acoustic table
    shadow: bool = True               # fold Fresnel transmission + excess attenuation into the reflectivity
    spec_gain: float = 3.5            # interface reflector gain (render_lotus.REFL_GAIN)
    grid_samples: int = 256           # depth samples of the DAS grid (upsampled to N_SAMPLES)
    dyn_range_db: float = 50.0        # log-compression window (RenderConfig.dyn_range_db when a cfg is given)
    top_pctile: float = 0.99          # white level (render_lotus.TOP_PCTILE)
    db_thresh: float = -100.0         # SIMUS OPTIONS.dBThresh (its default)
    workers: int = 0                  # 0 = $USBG_SIMUS_WORKERS or min(8, cpu - 2)


def simus_config(**overrides) -> SimusConfig:
    """Defaults, then $USBG_SIMUS_DENSITY / $USBG_SIMUS_NTX / $USBG_SIMUS_WORKERS, then keyword overrides."""
    env = {}
    for key, name, typ in (("density_per_mm2", "USBG_SIMUS_DENSITY", float), ("n_tx", "USBG_SIMUS_NTX", int),
                           ("workers", "USBG_SIMUS_WORKERS", int)):
        v = os.environ.get(name, "").strip()
        if v:
            env[key] = typ(v)
    return replace(SimusConfig(), **{**env, **overrides})


def _n_workers(sc: SimusConfig) -> int:
    if sc.workers and sc.workers > 0:
        return int(sc.workers)
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def _simus_chunk(x, z, rc, delays, param, db_thresh):
    """Worker: SIMUS RF for one chunk of scatterers (top level so a spawned process can import it)."""
    import pymust
    from pymust import utils
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        opt = utils.Options()
        opt.dBThresh = float(db_thresh)
        rf, _ = pymust.simus(x.reshape(1, -1), z.reshape(1, -1), rc.reshape(1, -1), delays.reshape(1, -1),
                             param, opt)
    return np.asarray(rf, np.float64)


SIMUS_CHUNKS = 16
_POOL = {"ex": None, "n": 0}


def _pool(n):
    import atexit
    import concurrent.futures as cf
    import multiprocessing as mp
    if _POOL["ex"] is not None and _POOL["n"] == n:
        return _POOL["ex"]
    if _POOL["ex"] is not None:
        _POOL["ex"].shutdown(wait=True)
    _POOL["ex"] = cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
    _POOL["n"] = n
    atexit.register(lambda: _POOL["ex"] is not None and _POOL["ex"].shutdown(wait=False, cancel_futures=True))
    return _POOL["ex"]


class _SimusSetup:
    """Geometry, probe, transmit delays and DAS matrices for one SimusConfig (built once per process)."""

    def __init__(self, sc: SimusConfig):
        import pymust
        from .._reuse import us_render
        self.sc = sc
        self.L, self.S = int(us_render.N_LINES), int(us_render.N_SAMPLES)
        self.R = float(us_render.APEX_M)                       # convex radius = the slicer's virtual apex
        self.depth = float(us_render.IMG_DEPTH_M)
        self.half = np.deg2rad(float(us_render.FOV_DEG)) / 2.0
        p = pymust.getparam(sc.probe)
        n = int(sc.n_elements)
        pitch = 2.0 * self.R * np.sin(self.half / (n - 1))       # element arc spans exactly the slicer's FOV
        p.Nelements = n
        p.radius = self.R
        p.pitch = float(pitch)
        p.width = float(0.9 * pitch)
        p.kerf = float(0.1 * pitch)
        p.attenuation = float(sc.attenuation)
        p.fs = 4.0 * float(p.fc)
        self.param = p
        _xe, _ze, _the, h = p.getElementPositions()
        self.h = float(np.ravel(h)[0])
        self.z_off = self.R - self.h                            # slicer z (face center at 0) -> MUST z
        th = np.linspace(-self.half, self.half, self.L)
        dg = np.linspace(0.0, self.depth, int(sc.grid_samples))
        r = self.R + dg
        self.gx = (np.sin(th)[:, None] * r[None, :])
        self.gz = (np.cos(th)[:, None] * r[None, :]) - self.R + self.z_off
        r1 = self.R + self.depth                                # far corners fix the RF length (RC = 0)
        self.corner_x = np.array([np.sin(-self.half) * r1, np.sin(self.half) * r1, 0.0])
        self.corner_z = np.array([np.cos(-self.half) * r1, np.cos(self.half) * r1, r1]) - self.R + self.z_off
        if sc.n_tx <= 1:
            tilts = [0.0]
        else:
            tilts = list(np.linspace(-sc.tilt_deg, sc.tilt_deg, int(sc.n_tx)))
        self.tilts = tilts
        self.delays = []
        for t in tilts:
            if abs(t) < 1e-12:
                d = np.zeros((1, n))
            else:
                d = np.asarray(pymust.txdelay(p, float(np.deg2rad(t))), np.float64).reshape(1, n)
            self.delays.append(d)
        self.M = {}
        self.tgc = None
        self.param_dict = {k: getattr(p, k) for k in ("fc", "kerf", "width", "pitch", "Nelements", "bandwidth",
                                                       "radius", "height", "focus", "attenuation", "fs", "c")
                           if getattr(p, k, None) is not None}

    def fresh_param(self):
        """A copy of the probe parameters without the TXdelay that pymust.simus / dasmtx store as a side
        effect (dasmtx refuses a PARAM.TXdelay that differs from the delays of another transmit)."""
        import copy
        p = copy.deepcopy(self.param)
        for k in [k for k in list(p.keys()) if str(k).lower() == "txdelay"]:
            del p[k]
        return p

    def das(self, iq, k):
        import pymust
        if k not in self.M:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                self.M[k] = (pymust.dasmtx(1j * np.array(iq.shape), self.gx, self.gz, self.delays[k],
                                           self.fresh_param()), iq.shape)
        M, shp = self.M[k]
        if iq.shape != shp:
            raise RuntimeError(f"SIMUS I/Q shape changed {iq.shape} != {shp}")
        return (M @ iq.flatten(order="F")).reshape(self.gx.shape, order="F")


_SIMUS_SETUPS: dict = {}


def _setup_for(sc):
    if sc not in _SIMUS_SETUPS:
        _SIMUS_SETUPS[sc] = _SimusSetup(sc)
    return _SIMUS_SETUPS[sc]


def _acoustic(lab, gel):
    """Per-sample impedance, attenuation (dB/cm at 1 MHz) and mean backscatter from volume.ACOUSTIC_LOTUS."""
    from ..render_lotus import GEL_Z, GEL_MU
    from ..volume import ACOUSTIC_LOTUS
    mx = max(ACOUSTIC_LOTUS)
    Z = np.zeros(256, np.float64); MU = np.zeros(256, np.float64); BS = np.zeros(256, np.float64)
    Z[:] = ACOUSTIC_LOTUS[12][0]; MU[:] = ACOUSTIC_LOTUS[12][1]; BS[:] = ACOUSTIC_LOTUS[12][2]
    for k, (z, a, m0, _m1, _s0) in ACOUSTIC_LOTUS.items():
        if k <= mx:
            Z[k], MU[k], BS[k] = z, a, m0
    li = np.clip(lab, 0, 255)
    Zm, MUm, BSm = Z[li], MU[li], BS[li]
    if gel:
        g = _leading_gap_np(lab)
        Zm = np.where(g, GEL_Z, Zm); MUm = np.where(g, GEL_MU, MUm); BSm = np.where(g, 0.0, BSm)
    return Zm, MUm, BSm


def _simus_reflectivity(lab, sc, setup, rng, gel, homogeneous=False):
    """Scatterer positions (MUST frame) and reflectivity for one label slice (L,S).

    Transmission loss folded into the reflectivity (shadow=True), in physical units: the one-way Fresnel
    intensity transmission accumulated over the interfaces in front of a sample (round trip in amplitude),
    times the round-trip attenuation IN EXCESS of SIMUS's homogeneous medium,
    10^(-2 (alpha_tissue - alpha_SIMUS) f_c[MHz] z[cm] / 20), with alpha_tissue the acoustic table's
    attenuation (dB/cm/MHz, LOTUS's units) and alpha_SIMUS = SimusConfig.attenuation."""
    L, S = lab.shape
    Zm, MUm, BSm = _acoustic(lab, gel)
    if homogeneous:
        Zm = np.full_like(Zm, Zm.mean()); MUm = np.full_like(MUm, sc.attenuation); BSm = np.ones_like(BSm)
    Rc = np.zeros_like(Zm)
    Rc[:, 1:] = ((Zm[:, 1:] - Zm[:, :-1]) / (Zm[:, 1:] + Zm[:, :-1] + 1e-6)) ** 2
    Rc = np.clip(Rc, 0.0, 0.999)
    if sc.shadow:
        dz_cm = setup.depth / S * 100.0
        fc_mhz = float(setup.param.fc) / 1e6
        T_cum = np.cumprod(1.0 - Rc, axis=1)
        excess_db = 2.0 * fc_mhz * np.cumsum((MUm - sc.attenuation) * dz_cm, axis=1)
        amp = np.clip(T_cum * 10.0 ** (-excess_db / 20.0), 0.0, 10.0)
    else:
        amp = np.ones_like(Zm)
    r0, r1 = setup.R, setup.R + setup.depth
    area_mm2 = setup.half * (r1 ** 2 - r0 ** 2) * 1e6
    n = int(round(sc.density_per_mm2 * area_mm2))
    th = rng.uniform(-setup.half, setup.half, n)
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    li = np.clip(np.rint((th + setup.half) / (2 * setup.half) * (L - 1)).astype(np.int64), 0, L - 1)
    si = np.clip(np.rint((r - r0) / setup.depth * (S - 1)).astype(np.int64), 0, S - 1)
    rc = BSm[li, si] * amp[li, si] * rng.standard_normal(n)
    x = r * np.sin(th); z = r * np.cos(th) - setup.R + setup.z_off
    if not homogeneous and sc.spec_gain > 0:
        il, isamp = np.nonzero(Rc > 1e-3)                 # specular interfaces: one reflector per crossing
        if len(il):
            thl = -setup.half + il / (L - 1) * 2 * setup.half
            rl = r0 + isamp / (S - 1) * setup.depth
            x = np.r_[x, rl * np.sin(thl)]
            z = np.r_[z, rl * np.cos(thl) - setup.R + setup.z_off]
            rc = np.r_[rc, sc.spec_gain * Rc[il, isamp] * amp[il, isamp]]
    return x, z, rc


def _simus_rf(x, z, rc, k, setup, sc):
    """Summed RF of all chunks for transmit k (the far-corner RC = 0 points fix the RF length)."""
    nw = _n_workers(sc)
    cx, cz = setup.corner_x, setup.corner_z
    # a FIXED chunking (not tied to the worker count): SIMUS zeroes RF below 1e-5 of each call's maximum, so
    # the chunking, not the pool size, decides the result; any worker count then gives identical frames
    chunks = np.array_split(np.arange(len(x)), SIMUS_CHUNKS)
    p = setup.fresh_param()
    jobs = [(np.r_[x[c], cx], np.r_[z[c], cz], np.r_[rc[c], np.zeros(len(cx))], setup.delays[k], p,
             float(sc.db_thresh)) for c in chunks if len(c)]
    if nw > 1:
        ex = _pool(nw)
        outs = list(ex.map(_simus_chunk, *zip(*jobs)))
    else:
        outs = [_simus_chunk(*j) for j in jobs]
    n0 = min(o.shape[0] for o in outs)
    if any(o.shape[0] != n0 for o in outs):
        raise RuntimeError(f"SIMUS chunk RF lengths differ: {[o.shape for o in outs]}")
    return np.sum([o[:n0] for o in outs], axis=0)


def _simus_envelope(lab, sc, setup, seed, gel, homogeneous=False):
    import pymust
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 0x51A5])
    x, z, rc = _simus_reflectivity(lab, sc, setup, rng, gel, homogeneous=homogeneous)
    acc = None
    for k in range(len(setup.delays)):
        rf = _simus_rf(x, z, rc, k, setup, sc)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            iq = pymust.rf2iq(rf, setup.fresh_param())
            b = setup.das(iq, k)
        acc = b if acc is None else acc + b
    return np.abs(acc)                                    # (L, grid_samples)


def _simus_tgc(sc, setup, gel):
    """Depth gain from a homogeneous soft-tissue phantom (fixed seed): 1 / smoothed mean envelope per depth."""
    if setup.tgc is None:
        lab = np.full((setup.L, setup.S), 12, np.int64)
        env = _simus_envelope(lab, sc, setup, 20240601, gel=False, homogeneous=True)
        prof = env.mean(axis=0)
        k = 9
        prof = np.convolve(np.pad(prof, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        prof = np.maximum(prof, prof.max() * 1e-4)
        setup.tgc = (np.median(prof) / prof).astype(np.float64)
    return setup.tgc


def render_simus(label_slice, *, device=None, cfg=None, seed=None, warp=False, simus=None, gel=None, **kw):
    """SIMUS (PyMUST) B-mode of a canonical label slice. CPU only; float32 (S, L) in [0,1]."""
    sc = simus if isinstance(simus, SimusConfig) else simus_config(**(simus or {}))
    if cfg is not None and getattr(cfg, "dyn_range_db", None) is not None:
        sc = replace(sc, dyn_range_db=float(cfg.dyn_range_db))
    setup = _setup_for(replace(sc, dyn_range_db=50.0, top_pctile=0.99))
    lab = np.asarray(label_slice).astype(np.int64)
    if gel is None:
        from ..seatmode import gel_coupling
        gel = gel_coupling()
    s = int(seed if seed is not None else (cfg.seed if cfg is not None else 0))
    env = _simus_envelope(lab, setup.sc, setup, s, gel) * _simus_tgc(setup.sc, setup, gel)[None, :]
    db = 20.0 * np.log10(env + 1e-12)
    top = float(np.quantile(db, sc.top_pctile))
    img = np.clip((db - (top - sc.dyn_range_db)) / sc.dyn_range_db, 0.0, 1.0)      # (L, Sg)
    Sg = img.shape[1]
    if Sg != setup.S:                                     # linear upsampling along depth to N_SAMPLES
        src = np.linspace(0.0, 1.0, Sg); dst = np.linspace(0.0, 1.0, setup.S)
        img = np.stack([np.interp(dst, src, row) for row in img])
    out = img.T.astype(np.float32)                        # (S, L)
    if warp:
        from .. import render_lotus as R
        out = R.warp_curvilinear(out)
    return out


def simus_info(sc=None):
    import importlib.metadata as md
    sc = sc or simus_config()
    st = _SIMUS_SETUPS.get(replace(sc, dyn_range_db=50.0, top_pctile=0.99))
    d = {"config": asdict(sc), "workers_resolved": _n_workers(sc)}
    try:
        d["pymust_version"] = md.version("PyMUST")
    except Exception:                                    # pragma: no cover
        d["pymust_version"] = None
    if st is not None:
        d["param"] = {k: (float(v) if np.isscalar(v) else None) for k, v in st.param_dict.items()}
        d["tilts_deg"] = [float(t) for t in st.tilts]
    return d


# ===================================================================================== registry + install
_FN = {"lotus_reimpl": render_lotus_reimpl, "lotus_orig": render_lotus_original, "simus": render_simus}
_REGISTERED = {"done": False}


def register_all():
    """Register the plugged renderers with render_lotus.register_renderer (idempotent). After this,
    ``render_lotus.render(lab, "lotus_orig")`` etc. work directly; nothing else changes."""
    from .. import render_lotus as R
    if not _REGISTERED["done"]:
        for name, fn in _FN.items():
            R.register_renderer(name)(fn)
        _REGISTERED["done"] = True


_STATE = {"active": None, "orig_render": None, "stats": {}}


def _stat(kind, sec):
    s = _STATE["stats"].setdefault(kind, {"n": 0, "sec": 0.0, "max_ms": 0.0})
    s["n"] += 1
    s["sec"] += sec
    s["max_ms"] = max(s["max_ms"], sec * 1000.0)


def install(kind: str):
    """Route every ``render_lotus.render(..., "physics", cfg=...)`` call (the call ``view_match.curvi_render``
    and the landscape make) to the plugged renderer ``kind`` for the rest of the process, seeded by
    ``cfg.seed``. ``install("physics")`` installs the same timing wrapper with no rerouting (the reference
    row run through the same code path). Process-wide and not thread-safe, like the env's scoped swaps."""
    from .. import render_lotus as R
    kind = canonical_name(kind)
    register_all()
    if _STATE["orig_render"] is None:
        _STATE["orig_render"] = R.render
    orig = _STATE["orig_render"]

    def render(label_slice, renderer="physics", device=None, texture=None, cfg=None, **kw):
        k = _STATE["active"]
        if renderer != "physics" or k is None:
            return orig(label_slice, renderer, device=device, texture=texture, cfg=cfg, **kw)
        t0 = time.perf_counter()
        if k == "physics":
            img = orig(label_slice, "physics", device=device, texture=texture, cfg=cfg, **kw)
        else:
            c = cfg if cfg is not None else R.RenderConfig()
            seed = kw.pop("seed", c.seed)
            img = orig(label_slice, k, device=device, texture=texture, cfg=c, seed=seed, **kw)
        _stat(k, time.perf_counter() - t0)
        return img

    render.__usbg_plug__ = True
    R.render = render
    _STATE["active"] = kind
    return kind


def uninstall():
    from .. import render_lotus as R
    if _STATE["orig_render"] is not None:
        R.render = _STATE["orig_render"]
    _STATE["active"] = None


def active():
    return _STATE["active"]


@contextlib.contextmanager
def plugged(kind):
    """Temporarily route physics renders to ``kind`` (installs the wrapper on first use)."""
    prev = _STATE["active"]
    if _STATE["orig_render"] is None or prev is None:
        install(kind)
    _STATE["active"] = canonical_name(kind)
    try:
        yield
    finally:
        if prev is None:
            uninstall()
        else:
            _STATE["active"] = prev


def stats():
    return {k: {"n": v["n"], "sec": round(v["sec"], 3), "mean_ms": round(1000.0 * v["sec"] / max(v["n"], 1), 3),
                "max_ms": round(v["max_ms"], 3)} for k, v in _STATE["stats"].items()}


def plug_info(kind=None, **extra):
    """Provenance block for a result that used a plugged renderer."""
    import importlib.metadata as md
    kind = canonical_name(kind or _STATE["active"] or "physics")
    d = {"slot": "renderer", "renderer": kind, "citation": CITATIONS[kind], "analysis_status": PLUG_STATUS,
         "routing": "render_lotus.render(..., 'physics', cfg) -> plugged renderer (usbg.plug.renderers.install)",
         "geometry": "canonical convex label slice -> (N_SAMPLES, N_LINES) -> view_match 128 px measured sector",
         "seed_source": "cfg.seed (RenderConfig), shared by target and observation",
         "dr_knobs_ignored": kind != "physics", "render_stats": stats()}
    if kind in ("lotus_orig", "lotus_reimpl"):
        d["device"] = plug_device()
    if kind == "lotus_orig":
        try:
            d["lotus_source"] = lotus_source_info()
        except Exception as e:                            # pragma: no cover
            d["lotus_source"] = {"error": repr(e)}
        d["label_map_usbg_to_lotus"] = {str(k): v for k, v in USBG_TO_LOTUS_ORIG.items()}
        d["gel_gap_lotus_id"] = LOTUS_GEL_ID
        d["deviations"] = ["device redirect cuda -> cpu/mps (scoped torch.Tensor.to patch, no file edit)",
                           "on MPS the float64 TGC ramp is cast to float32 (CPU keeps float64)",
                           "LOTUS display warp bypassed; usbg measured-sector warp applied downstream",
                           "default (published) parameters; LOTUS joint training not run"]
    if kind == "simus":
        d["simus"] = simus_info()
        d["deviations"] = ["testbed footprint: radius 40 mm, 60 deg arc (C5-2v fc/bandwidth/height/focus kept)",
                           "tissue transmission loss folded into reflectivity (shadow=True)",
                           "TGC calibrated once on a homogeneous soft-tissue phantom",
                           "evaluation only (seconds per frame)"]
    try:
        import torch
        d["torch"] = torch.__version__
    except Exception:                                     # pragma: no cover
        pass
    try:
        d["pymust"] = md.version("PyMUST")
    except Exception:                                     # pragma: no cover
        d["pymust"] = None
    d.update(extra)
    return d


__all__ = ["install", "uninstall", "plugged", "active", "register_all", "plug_info", "stats",
           "render_lotus_original", "render_lotus_reimpl", "render_simus", "SimusConfig", "simus_config",
           "canonical_name", "lotus_dir", "lotus_source_info", "USBG_TO_LOTUS_ORIG", "PLUG_STATUS",
           "CITATIONS", "INFEASIBLE", "PLUG_NAMES"]
