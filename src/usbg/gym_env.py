"""Gymnasium environments for USB Gym.

``USBGym/ViewMatch-v0`` (:class:`ViewMatchEnv`) is the curvilinear view-matching task of the paper as a
standard ``gymnasium.Env``. Importing this module registers it::

    import gymnasium as gym
    import usbg.gym_env  # noqa: F401  (registers USBGym/ViewMatch-v0)

    env = gym.make("USBGym/ViewMatch-v0", texture="physics", mask="ct", device="cpu")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The environment reproduces the evaluation code path of ``scripts/train_view_match_rl.py`` (``evaluate``
for poses and match scores, ``rollout`` for the reward): the same setup sampler
(``goal_rl.sample_setup``), the same MuJoCo probe (``mujoco_env.USProbeEnv``), the same curvilinear
render (``view_match.curvi_render``), the same image-servo base and the same score. A fixed action
sequence gives the same poses and rewards as that code path (``tests/test_gym_env.py``).

Episode
    One reset seats one patient's CT at a screened acoustic window of a target organ (anatomical seat)
    and starts the probe from a random perturbation of that window pose: up to +-``lateral_m`` lateral
    and elevational, +-``axial_m`` axial, +-``fan_deg`` about the lateral and elevational axes and
    +-4 deg of roll. A start below the contact plane is lifted out of the patient. The target frame is
    rendered by the same pipeline at the window pose. Episodes last ``n_steps`` steps (default 12) and
    then truncate; there is no terminal state.

Observation (``spaces.Dict``)
    ``current``  ``(history, S, S)``: the last ``history`` frames (default 4, S = ``image_size`` =
                 128), oldest first; at reset the start frame is repeated.
    ``target``   ``(1, S, S)``: the target frame.
    ``current_mask``, ``target_mask``  uint8 ``(1, S, S)`` in {0..12}, only with
                 ``include_masks=True``: the organ masks of the selected mask source (CT label ids, or
                 {0,1,2,3,4,7} from a learned segmenter). Under ``mask="ct"`` these masks are CT
                 labels, which no physical rig provides.
    Frames are quantized to 8 bits as in the trainer (``usbg.eval._prep``) and given as float32 in
    [0, 1] (``frame_dtype="float32"``, the input of the paper's policies) or, losslessly, as uint8 in
    [0, 255] (``frame_dtype="uint8"``, for image policies of RL libraries). The reward uses the
    unquantized frames.

Action (``spaces.Box``, float32, shape (6,), bounds [-1, 1])
    A normalized body-frame twist ``[vx, vy, vz, wx, wy, wz]``. It is scaled by the bound
    ``a_bound``: ``goal_rl.MAX_TWIST`` = (0.03 m/s x3, 0.5 rad/s x3) without a base, or
    ``residual_frac * MAX_TWIST`` (default half) as the residual of a base. The executed twist is
    ``clip(base + a * a_bound, -MAX_TWIST, MAX_TWIST)``, integrated over 0.1 s.

Reward
    The paper's step reward (Eq. 1 and the reward gain): ``r_t = S_t - S_{t-1} - effort * |rho_t|^2``
    with ``S = w_seg * S_seg + w_app * S_app`` (defaults 0.6 and 0.4), ``S_seg`` the mean per-organ IoU
    of the two organ masks (``view_match.seg_organ_iou``), ``S_app`` the sector-interior NCC
    (``view_match.sector_appearance``) and ``rho_t = a_t * a_bound`` the agent's twist (the residual
    when a base is on). ``effort`` defaults to 0.003. The reward uses no pose.

Info
    ``match``, ``seg_iou``, ``appearance``: S and its two terms for the current frame.
    ``contact_force``: probe-skin normal force (N). ``step``, ``patient``, ``target_organ``, ``window``,
    ``n_windows``, ``start_clamped``, ``clamp_mm``, ``contact``. On steps also ``residual_twist``,
    ``base_twist`` and ``executed_twist`` (SI units).
    ``privileged`` (only with ``privileged_info=True``): simulator state that no physical rig provides,
    for evaluation, imitation teachers and the privileged pose-oracle base only. A non-privileged
    controller must not read it: ``pose_error_mm`` and ``rotation_error_deg`` (true pose error to the
    target, ``view_match.pose_error``), ``probe_pose`` and ``target_pose`` (4x4 world poses),
    ``analytic_twist`` (the clipped corrective twist from the true poses), ``ct_mask`` and
    ``target_ct_mask`` (CT label slices of the two frames, uint8).

Configuration (constructor keywords)
    ``patients``  "train" (``benchmark.TRAIN_PATIENTS``, default), "test" (the six held-out
                  ``TEST_PATIENTS``) or a list of case ids in ``data/cache``.
    ``targets``   target organs, default ("liver", "kidney").
    ``texture``   texture stage: "physics" (none), "cut", "cyclegan", "sonospade" (checkpoint from
                  ``tex_manifest``, default ``outputs/v2/textures/manifest.json``), a checkpoint path
                  (``texture_ckpt``) or any ``interfaces.TextureTranslator`` callable.
    ``mask``      mask source of the reward and the base: "ct" (CT label slice), "learned" (a
                  texture-matched segmenter, ``seg_ckpt`` or $US_SEG_CKPT; the withdrawn shared
                  checkpoint is refused) or any ``interfaces.MaskSource`` object.
    ``base``      None (the action is the full twist), "servo" (image-servo base on mask moments; pass
                  ``servo_base`` as a (F, 6) matrix, a v2 checkpoint's ``ibvs`` dict or a path to a v2
                  policy ``.pt``, or leave it None to calibrate with :func:`calibrate_servo_base`) or
                  "pose_oracle" (the analytic twist from the true pose: privileged).
    ``contact``   probe-skin contact on (default) or off. ``randomize`` "off" (one fixed render
                  configuration with seed ``render_seed``) or "episode" (one domain-randomization draw
                  per episode, shared by target and observations). ``render_config`` replaces the
                  default ``render_lotus.RenderConfig`` (impedance, attenuation and backscatter scale
                  per label; gain, time-gain slope, dynamic range, PSF, speckle, noise, shadow gain);
                  its seed is replaced by ``render_seed``, and "episode" randomizes around it.
                  ``device`` "cpu", "mps" or None (``render_lotus.get_device()``: MPS if present, else
                  the CPU; never CUDA).
    Reward weights ``w_seg``, ``w_app``, ``effort``; horizon ``n_steps``; ``history``; ``image_size``;
    start perturbation ``lateral_m``, ``axial_m``, ``fan_deg``; ``min_windows`` (a (patient, organ)
    cell needs this many screened windows to be drawn, default 2 as in the v2 runs).

Reset options
    ``seed`` (``reset(seed=)``) seeds the setup draw. ``options`` may fix ``patient``, ``target``
    (organ name), ``window`` (index, modulo the cell's window count) and ``contact`` (bool, this episode
    only). ``options={"eval_episode": k}`` reproduces episode k of the paper's crossed evaluation
    design over this environment's patients: setup seed ``eval_seed + k`` (default 999), cell
    ``k % n_cells`` in patient-major order, window ``k // n_cells``.

Limits
    The render path pins the PyTorch device and the mask source through scoped module-level swaps
    (``render_lotus.get_device`` and ``view_match._US``), so one process must not step two
    environments from different threads at once. Use process-based vector environments
    (``gymnasium.vector.AsyncVectorEnv``) for parallel rollouts. The anatomical seat is required
    (``USBG_SEAT=anatomical``, the default); ``USBG_SEAT=canonical`` only reproduces v1 numbers.

``USProbeGymEnv`` (below) is the older generic adapter over ``USProbeEnv`` with a caller-supplied
reward; it is kept unchanged for existing users.
"""
from __future__ import annotations

import contextlib
import os
from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
    _Base = gym.Env
except ImportError:  # gymnasium not installed: keep the module importable
    _HAVE_GYM = False
    _Base = object

from . import goal_rl as G
from . import render_lotus as R
from . import view_match as VM
from . import volume as V
from .eval import _prep
from .goal_rl import MAX_TWIST
from .mujoco_env import USProbeEnv

ENV_ID = "USBGym/ViewMatch-v0"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEX_MANIFEST = os.path.join(_REPO, "outputs", "v2", "textures", "manifest.json")
TEXTURE_STAGES = ("physics", "cut", "cyclegan", "sonospade")
ORGANS = {"liver": V.LIVER, "kidney": V.KIDNEY, "spleen": V.SPLEEN}
DR_SALT = 7919          # = scripts/train_view_match_rl._DR_SALT: the separate stream of DR draws
MASK_HIGH = V.N_LABELS - 1


# ===================================================================================== scoped swaps
@contextlib.contextmanager
def _render_device(device):
    """Pin the device of ``render_lotus.render`` (``view_match.curvi_render`` passes none) for the
    duration of one render. None keeps ``get_device()``. Not thread-safe (module-level swap)."""
    if device is None:
        yield
        return
    saved = R.get_device
    R.get_device = lambda prefer="mps": device
    try:
        yield
    finally:
        R.get_device = saved


@contextlib.contextmanager
def _ct_label_output():
    """Make ``view_match.curvi_render`` return the CT label slice (its ``_US is None`` branch) for the
    duration of one render, whatever mask source the process bound. Not thread-safe."""
    saved = VM._US
    VM._US = None
    try:
        yield
    finally:
        VM._US = saved


# ===================================================================================== image-servo base
def organ_moments(mask, organ, min_px=20):
    """Shape descriptors of one organ in normalized image coordinates: centroid, area, presence,
    principal-axis orientation and elongation. Same arithmetic as
    ``scripts/train_view_match_rl._organ_moments``."""
    m = (mask == organ); a = int(m.sum()); H, W = mask.shape
    if a < min_px:
        return dict(cx=0.5, cy=0.5, area=0.0, present=0.0, theta=0.0, elong=0.0)
    ys, xs = np.nonzero(m); x = xs / W; y = ys / H
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    cxx, cyy, cxy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy + 1e-12)
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    elong = (l1 - l2) / (l1 + l2 + 1e-9)
    return dict(cx=cx, cy=cy, area=a / (H * W), present=1.0, theta=float(theta), elong=float(elong))


def servo_features(cur_mask, tgt_mask, rich=True):
    """Image-only servo features (no pose): per matched organ the centroid offset, area change and
    co-presence; ``rich`` adds the elongation change and the orientation change as sin and cos deltas;
    then a constant. 36 features (rich) or 21. Same as ``train_view_match_rl.ibvs_features``."""
    f = []
    for o in VM.MATCH_ORGANS:
        c = organ_moments(cur_mask, o); t = organ_moments(tgt_mask, o)
        both = c["present"] * t["present"]
        f += [(t["cx"] - c["cx"]) * both, (t["cy"] - c["cy"]) * both, (t["area"] - c["area"]) * both, both]
        if rich:
            f += [(t["elong"] - c["elong"]) * both,
                  (np.sin(t["theta"]) - np.sin(c["theta"])) * both,
                  (np.cos(t["theta"]) - np.cos(c["theta"])) * both]
    f.append(1.0)
    return np.asarray(f, np.float64)


def _as_servo_base(spec):
    """(F, 6) matrix | {"kind": "linear", "model", "rich"} | path to a v2 policy .pt -> base dict."""
    if isinstance(spec, (str, os.PathLike)):
        import torch
        ck = torch.load(os.fspath(spec), map_location="cpu", weights_only=False)
        if not ck.get("ibvs"):
            raise ValueError(f"{spec} carries no calibrated servo base ('ibvs')")
        spec = ck["ibvs"]
    if isinstance(spec, dict):
        if spec.get("kind", "linear") != "linear":
            raise ValueError(f"only the linear servo base is supported here, got {spec.get('kind')!r}")
        model = np.asarray(spec["model"], np.float64)
        rich = bool(spec.get("rich", model.shape[0] == 36))
    else:
        model = np.asarray(spec, np.float64)
        rich = model.shape[0] == 36
    n_feat = 5 * (7 if rich else 4) + 1
    if model.shape != (n_feat, 6):
        raise ValueError(f"servo base must be ({n_feat}, 6) for rich={rich}, got {model.shape}")
    return {"kind": "linear", "model": model, "rich": rich}


def servo_base_twist(base, cur_mask, tgt_mask):
    """Base twist of a linear servo base (SI units, clipped to +-MAX_TWIST)."""
    return np.clip(servo_features(cur_mask, tgt_mask, base["rich"]) @ base["model"], -MAX_TWIST, MAX_TWIST)


# ===================================================================================== render pipeline
def _resolve_texture_ckpt(stage, texture_ckpt=None, tex_manifest=None):
    """Checkpoint path of a named texture stage (None for physics), as train_view_match_rl.resolve_texture
    under the anatomical seat: ``texture_ckpt`` first, then the v2 manifest."""
    if texture_ckpt is not None:
        p = None if str(texture_ckpt).lower() == "none" else os.fspath(texture_ckpt)
    elif stage == "physics":
        p = None
    else:
        import json
        man = tex_manifest or DEFAULT_TEX_MANIFEST
        if not os.path.exists(man):
            raise FileNotFoundError(f"texture manifest {man} missing; pass texture_ckpt= or tex_manifest=")
        m = json.load(open(man))
        if stage not in m:
            raise ValueError(f"{man} has no entry for {stage!r} (keys {sorted(m)})")
        p = m[stage]
    if p is not None and not os.path.isabs(p):
        p = os.path.join(_REPO, p)
    if (p is None) != (stage == "physics"):
        raise ValueError(f"texture {stage!r} with checkpoint {p!r}")
    if p is not None and not os.path.exists(p):
        raise FileNotFoundError(f"texture checkpoint {p} does not exist")
    return p


class ViewPipeline:
    """One bound (texture stage, mask source, device, image size): renders a probe pose to the
    curvilinear frame, the arm's organ mask and the CT label slice. Shared by :class:`ViewMatchEnv`
    and :func:`calibrate_servo_base`, so the target, every observation and the servo calibration go
    through the identical pipeline (the same-pipeline rule)."""

    def __init__(self, texture="physics", mask="ct", *, texture_ckpt=None, tex_manifest=None,
                 seg_ckpt=None, device=None, image_size=128):
        from . import provenance as PROV
        self.device = device or R.get_device()
        self.image_size = int(image_size)
        # texture stage
        if callable(texture):
            name = getattr(texture, "__name__", None) or type(texture).__name__
            self.texture_name, self.texture_ckpt, self.translator = name, None, texture
        else:
            name = str(texture)
            if name not in TEXTURE_STAGES:
                raise ValueError(f"texture {name!r}: expected one of {TEXTURE_STAGES} or a callable")
            ckpt = _resolve_texture_ckpt(name, texture_ckpt, tex_manifest)
            why = PROV.is_private_descendant(ckpt) if ckpt else None
            if why:
                raise ValueError(f"refusing texture {ckpt}: private or a private-pool descendant ({why})")
            self.texture_name, self.texture_ckpt = name, ckpt
            self.translator = None
            if ckpt is not None:
                from .texture_gan import load_translator
                self.translator = load_translator(ckpt, self.device)
        # mask source
        self.seg_ckpt = None
        if isinstance(mask, str):
            if mask == "ct":
                self.mask_name, self.mask_source = "ct", None
            elif mask == "learned":
                ck = seg_ckpt or os.environ.get("US_SEG_CKPT")
                if not ck:
                    raise ValueError("mask='learned' needs seg_ckpt= (or $US_SEG_CKPT): the implicit default "
                                     "segmenter is withdrawn (D2)")
                md5 = PROV.md5(ck)
                if md5 in PROV.WITHDRAWN_SEG_MD5:
                    raise ValueError(f"refusing segmenter {ck}: the withdrawn shared checkpoint (D2)")
                why = PROV.is_private_descendant(ck, md5)
                if why:
                    raise ValueError(f"refusing segmenter {ck}: private or a private-pool descendant ({why})")
                from .us_segmenter import load_segmenter
                self.mask_name, self.mask_source = "learned", load_segmenter(ck, device=self.device)
                self.seg_ckpt = os.path.abspath(ck)
            else:
                raise ValueError(f"mask {mask!r}: expected 'ct', 'learned' or an object with .segment(img01)")
        elif hasattr(mask, "segment"):
            self.mask_name, self.mask_source = getattr(mask, "name", type(mask).__name__), mask
        else:
            raise ValueError("mask must be 'ct', 'learned' or a MaskSource (an object with .segment(img01))")

    def render(self, lv, T, world_from_volume, cfg):
        """-> (frame float32 (S,S) in [0,1], arm mask int64 (S,S), CT label slice int64 (S,S)).
        The frame is ``view_match.curvi_render`` at pose T; the arm mask is the CT label slice or the
        mask source applied to the frame, exactly as ``curvi_render`` does with that source bound."""
        with _render_device(self.device), _ct_label_output():
            img, ct = VM.curvi_render(lv, T, world_from_volume, self.translator, cfg, self.image_size)
        mask = ct if self.mask_source is None else np.asarray(self.mask_source.segment(img), np.int64)
        return img, mask, ct

    def info(self):
        from . import provenance as PROV
        return {"texture": self.texture_name, "texture_ckpt": self.texture_ckpt,
                "texture_md5": PROV.md5(self.texture_ckpt), "mask": self.mask_name,
                "seg_ckpt": self.seg_ckpt, "seg_md5": PROV.md5(self.seg_ckpt), "device": self.device}


# ===================================================================================== patients and cells
_VOLUMES: dict = {}


def load_patient(case):
    """The cached label volume of a case id (``data/cache/<case>.npz``; ``volume.load_cache`` reads the
    bladder-free ``data/cache_anat`` twin under the anatomical seat). Cached per process."""
    from .seatmode import seat_mode
    key = (case, seat_mode())
    if key not in _VOLUMES:
        lv = V.load_cache(os.path.join(_REPO, "data", "cache", f"{case}.npz"))
        lv.meta.setdefault("case", case)
        _VOLUMES[key] = lv
    return _VOLUMES[key]


def _patient_list(patients):
    from .benchmark import TRAIN_PATIENTS, TEST_PATIENTS
    if isinstance(patients, str):
        if patients == "train":
            return list(TRAIN_PATIENTS)
        if patients == "test":
            return list(TEST_PATIENTS)
        return [patients]
    return [str(p) for p in patients]


def n_target_windows(lv, organ):
    """Screened windows of (case, organ name); 0 when the case has none (as the trainer counts them)."""
    from . import placement as P
    try:
        return int(P.n_windows(lv, ORGANS[organ]))
    except P.NoAcceptedWindow:
        return 0


def target_cells(lvs, targets, min_windows):
    """[(patient index, organ)] with at least ``min_windows`` screened windows, patient-major (the order
    the crossed evaluation design walks)."""
    return [(i, t) for i, lv in enumerate(lvs) for t in targets if n_target_windows(lv, t) >= min_windows]


def _require_anatomical():
    from .seatmode import seat_mode
    if seat_mode() != "anatomical":
        raise RuntimeError("USBGym/ViewMatch-v0 implements the anatomical-seat task; unset USBG_SEAT or set "
                           "USBG_SEAT=anatomical (USBG_SEAT=canonical only reproduces v1 numbers)")


def _draw_cell(rng, n_patients, targets, valid):
    """Uniform patient, then uniform target, redrawn while the cell is invalid: the training sampler
    ``train_view_match_rl.draw_cell``, so a seeded draw consumes the stream the same way."""
    while True:
        i = int(rng.integers(n_patients))
        t = targets[int(rng.integers(len(targets)))]
        if (i, t) in valid:
            return i, t


def _setup(lv, organ, rng, window, setup_kw):
    """(world_from_volume, T_target, T_start, meta) via goal_rl.sample_setup (train_view_match_rl.make_setup
    under the anatomical seat, without its training counters)."""
    return G.sample_setup(lv, ORGANS[organ], rng, window=window, return_meta=True, **setup_kw)


# ===================================================================================== calibration
def calibrate_servo_base(*, patients="train", targets=("liver", "kidney"), texture="physics", mask="ct",
                         texture_ckpt=None, tex_manifest=None, seg_ckpt=None, n=500, seed=0, render_seed=None,
                         randomize="off", rich=True, min_windows=2, lateral_m=0.02, axial_m=0.01,
                         fan_deg=18.0, image_size=128, device=None, pipeline=None, render_config=None):
    """Fit the linear image-servo base (features -> corrective twist, least squares) on ``n`` frame pairs,
    as ``train_view_match_rl.calibrate_ibvs`` does at the start of a run with seed ``seed``: the setup
    stream ``default_rng(seed)`` is consumed in the same order, so with the v2 cohort (TRAIN patients,
    liver and kidney, ``min_windows=2``), the same texture and mask source it gives the base of the
    seed-``seed`` v2 runs. Calibration reads the true pose (the analytic twist is the regression
    target), a simulator-only step; the fitted base needs only masks at run time."""
    _require_anatomical()
    pipe = pipeline or ViewPipeline(texture, mask, texture_ckpt=texture_ckpt, tex_manifest=tex_manifest,
                                    seg_ckpt=seg_ckpt, device=device, image_size=image_size)
    cases = _patient_list(patients)
    lvs = [load_patient(c) for c in cases]
    targets = list(targets)
    valid = set(target_cells(lvs, targets, min_windows))
    if not valid:
        raise ValueError("no (patient, organ) cell has enough screened windows")
    rng = np.random.default_rng(seed)
    dr_rng = np.random.default_rng([seed, DR_SALT])
    base_cfg = R.RenderConfig() if render_config is None else render_config
    cfg = base_cfg.with_seed(seed if render_seed is None else render_seed)
    setup_kw = dict(lateral_m=lateral_m, axial_m=axial_m, fan_deg=fan_deg)
    X, Y = [], []
    for _ in range(int(n)):
        i, organ = _draw_cell(rng, len(lvs), targets, valid)
        Wfv, T_tgt, T_cur, _m = _setup(lvs[i], organ, rng, None, setup_kw)
        c = R.domain_randomize(base_cfg, dr_rng) if randomize == "episode" else cfg
        _, cur_mask, _ = pipe.render(lvs[i], T_cur, Wfv, c)
        _, tgt_mask, _ = pipe.render(lvs[i], T_tgt, Wfv, c)
        X.append(servo_features(cur_mask, tgt_mask, rich)); Y.append(G.analytic_twist(T_cur, T_tgt))
    model = np.linalg.lstsq(np.asarray(X), np.asarray(Y), rcond=None)[0]
    return {"kind": "linear", "model": model, "rich": bool(rich), "n": int(n), "seed": int(seed),
            "patients": cases, "targets": targets, "pipeline": pipe.info()}


# ===================================================================================== the environment
class ViewMatchEnv(_Base):
    """``USBGym/ViewMatch-v0``: reproduce a target ultrasound view by moving a probe over a patient's
    CT. See the module docstring for the observation, action, reward, info and options."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, *, patients="train", targets=("liver", "kidney"), texture="physics", mask="ct",
                 texture_ckpt=None, tex_manifest=None, seg_ckpt=None, base=None, servo_base=None,
                 calib_pairs=500, residual_frac=0.5, contact=True, randomize="off", render_seed=0,
                 render_config=None,
                 n_steps=12, history=4, image_size=128, w_seg=0.6, w_app=0.4, effort=0.003,
                 lateral_m=0.02, axial_m=0.01, fan_deg=18.0, min_windows=2, eval_seed=999,
                 include_masks=False, frame_dtype="float32", privileged_info=True, device=None,
                 render_mode=None):
        if not _HAVE_GYM:
            raise ImportError("gymnasium is required for ViewMatchEnv: `pip install gymnasium`")
        super().__init__()
        _require_anatomical()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode {render_mode!r}: expected None or 'rgb_array'")
        if base not in (None, "servo", "pose_oracle"):
            raise ValueError(f"base {base!r}: expected None, 'servo' or 'pose_oracle'")
        if randomize not in ("off", "episode"):
            raise ValueError(f"randomize {randomize!r}: expected 'off' or 'episode'")
        if frame_dtype not in ("float32", "uint8"):
            raise ValueError(f"frame_dtype {frame_dtype!r}: expected 'float32' or 'uint8'")
        self.render_mode = render_mode
        self.pipeline = ViewPipeline(texture, mask, texture_ckpt=texture_ckpt, tex_manifest=tex_manifest,
                                     seg_ckpt=seg_ckpt, device=device, image_size=image_size)
        self.device = self.pipeline.device
        self.patients = _patient_list(patients)
        self.targets = list(targets)
        for t in self.targets:
            if t not in ORGANS:
                raise ValueError(f"target {t!r}: expected one of {sorted(ORGANS)}")
        self._lvs = [load_patient(c) for c in self.patients]
        self.min_windows = int(min_windows)
        self.cells = target_cells(self._lvs, self.targets, self.min_windows)
        if not self.cells:
            raise ValueError("no (patient, organ) cell has enough screened windows")
        self._valid = set(self.cells)
        self.base = base
        self.servo = None
        if base == "servo":
            if servo_base is None:
                servo_base = calibrate_servo_base(targets=self.targets, n=calib_pairs, seed=render_seed,
                                                  min_windows=self.min_windows, lateral_m=lateral_m,
                                                  axial_m=axial_m, fan_deg=fan_deg, randomize=randomize,
                                                  pipeline=self.pipeline, render_config=render_config)
            self.servo = _as_servo_base(servo_base)
        self.residual_frac = float(residual_frac)
        self.a_bound = (self.residual_frac * MAX_TWIST if base is not None else MAX_TWIST).astype(np.float32)
        self.contact = bool(contact)
        self.randomize = randomize
        self.render_seed = int(render_seed)
        self._base_cfg = R.RenderConfig() if render_config is None else render_config
        self.cfg = self._base_cfg.with_seed(self.render_seed)
        self.n_steps, self.history, self.image_size = int(n_steps), int(history), int(image_size)
        self.w_seg, self.w_app, self.effort = float(w_seg), float(w_app), float(effort)
        self.setup_kw = dict(lateral_m=float(lateral_m), axial_m=float(axial_m), fan_deg=float(fan_deg))
        self.eval_seed = int(eval_seed)
        self.include_masks = bool(include_masks)
        self.frame_dtype = frame_dtype
        self.privileged_info = bool(privileged_info)
        self._dr_rng = np.random.default_rng()
        self._probe = None
        S = self.image_size
        if frame_dtype == "uint8":
            frames = lambda shape: spaces.Box(0, 255, shape, np.uint8)          # noqa: E731
        else:
            frames = lambda shape: spaces.Box(0.0, 1.0, shape, np.float32)      # noqa: E731
        obs = {"current": frames((self.history, S, S)), "target": frames((1, S, S))}
        if self.include_masks:
            obs["current_mask"] = spaces.Box(0, MASK_HIGH, (1, S, S), np.uint8)
            obs["target_mask"] = spaces.Box(0, MASK_HIGH, (1, S, S), np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)

    # helpers
    def _score(self, img, mask):
        seg = VM.seg_organ_iou(mask, self._tgt_mask)
        app = VM.sector_appearance(img, self._tgt_img)
        return self.w_seg * seg + self.w_app * app, seg, app

    def _frames(self, x):
        x = np.asarray(x, np.float32)
        return np.rint(x * 255.0).astype(np.uint8) if self.frame_dtype == "uint8" else x

    def _obs(self):
        o = {"current": self._frames(np.stack(self._hist, 0)), "target": self._frames(self._tgt_prep[None])}
        if self.include_masks:
            o["current_mask"] = self._mask[None].astype(np.uint8)
            o["target_mask"] = self._tgt_mask[None].astype(np.uint8)
        return o

    def _info(self, seg, app):
        info = {"match": float(self._s), "seg_iou": float(seg), "appearance": float(app),
                "contact_force": float(self._probe.contact_force()), "step": self._t, "contact": self._contact,
                **self._episode}
        if self.privileged_info:
            pose = self._probe.world_from_probe
            pe, ae = VM.pose_error(pose, self._T_tgt)
            info["privileged"] = {"pose_error_mm": pe, "rotation_error_deg": ae, "probe_pose": pose,
                                  "target_pose": self._T_tgt.copy(),
                                  "analytic_twist": G.analytic_twist(pose, self._T_tgt),
                                  "ct_mask": self._ct.astype(np.uint8), "target_ct_mask": self._tgt_ct.astype(np.uint8)}
        return info

    def _pick_cell(self, rng, patient, organ):
        """(patient index, organ) for a reset: fixed by the options, else drawn from rng over the cells
        with at least ``min_windows`` windows (uniform patient, then uniform organ, redrawn while invalid,
        the training sampler's order). An explicitly fixed cell needs only one screened window."""
        if organ is not None and organ not in ORGANS:
            raise ValueError(f"target {organ!r}: expected one of {sorted(ORGANS)}")
        i = None
        if patient is not None:
            if patient not in self.patients:
                raise ValueError(f"patient {patient!r} is not in this environment's patients {self.patients}")
            i = self.patients.index(patient)
        if i is None and organ is None:
            return _draw_cell(rng, len(self._lvs), self.targets, self._valid)
        if i is None:
            if not any((j, organ) in self._valid for j in range(len(self._lvs))):
                raise ValueError(f"no patient has {self.min_windows} or more screened {organ} windows")
            while True:
                i = int(rng.integers(len(self._lvs)))
                if (i, organ) in self._valid:
                    return i, organ
        if organ is None:
            if not any((i, t) in self._valid for t in self.targets):
                raise ValueError(f"{patient} has no target with {self.min_windows} or more screened windows")
            while True:
                organ = self.targets[int(rng.integers(len(self.targets)))]
                if (i, organ) in self._valid:
                    return i, organ
        if n_target_windows(self._lvs[i], organ) < 1:
            raise ValueError(f"{patient} has no screened {organ} window")
        return i, organ

    def _base_twist(self):
        if self.base == "servo":
            return servo_base_twist(self.servo, self._mask, self._tgt_mask)
        if self.base == "pose_oracle":
            return G.analytic_twist(self._probe.world_from_probe, self._T_tgt)
        return np.zeros(6, np.float64)

    # API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = dict(options or {})
        unknown = set(opts) - {"patient", "target", "window", "contact", "eval_episode"}
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}")
        if seed is not None:
            self._dr_rng = np.random.default_rng([int(seed), DR_SALT])
        if self._probe is not None:
            self._probe.close()
        cfg = self.cfg
        if opts.get("eval_episode") is not None:
            k = int(opts["eval_episode"])
            rng = np.random.default_rng(self.eval_seed + k)
            i, organ = self.cells[k % len(self.cells)]
            window = k // len(self.cells)
            if self.randomize == "episode":
                cfg = R.domain_randomize(self._base_cfg, np.random.default_rng([self.eval_seed, k, DR_SALT]))
        else:
            rng = self.np_random
            i, organ = self._pick_cell(rng, opts.get("patient"), opts.get("target"))
            window = opts.get("window")
            if self.randomize == "episode":
                cfg = R.domain_randomize(self._base_cfg, self._dr_rng)
        lv = self._lvs[i]
        Wfv, T_tgt, T_start, meta = _setup(lv, organ, rng, window, self.setup_kw)
        self._contact = bool(opts.get("contact", self.contact))
        self._lv, self._Wfv, self._T_tgt, self._cfg_ep = lv, Wfv, T_tgt, cfg
        self._probe = USProbeEnv(lv, Wfv, T_start, renderer="physics", device=self.device, translator=None,
                                 randomize=False, rng=np.random.default_rng(0), contact=self._contact,
                                 observe=False)
        self._probe.reset()
        self._episode = {"patient": self.patients[i], "target_organ": organ, "window": meta.get("window"),
                         "n_windows": meta.get("n_windows"), "start_clamped": bool(meta.get("start_clamped")),
                         "clamp_mm": float(meta.get("clamp_mm") or 0.0)}
        self._tgt_img, self._tgt_mask, self._tgt_ct = self.pipeline.render(lv, T_tgt, Wfv, cfg)
        self._tgt_prep = _prep(self._tgt_img, self.image_size)
        self._img, self._mask, self._ct = self.pipeline.render(lv, self._probe.world_from_probe, Wfv, cfg)
        self._hist = deque([_prep(self._img, self.image_size)] * self.history, maxlen=self.history)
        self._t = 0
        self._s, seg, app = self._score(self._img, self._mask)
        return self._obs(), self._info(seg, app)

    def step(self, action):
        if self._probe is None:
            raise RuntimeError("call reset() before step()")
        if self._t >= self.n_steps:
            raise RuntimeError(f"the episode ended after {self.n_steps} steps; call reset()")
        a = np.clip(np.asarray(action, np.float32).reshape(6), -1.0, 1.0)
        rho = a * self.a_bound                                         # agent twist, SI units, float32
        base = self._base_twist()
        a_exec = np.clip(rho + base, -MAX_TWIST, MAX_TWIST)
        self._probe.step(a_exec)
        self._img, self._mask, self._ct = self.pipeline.render(self._lv, self._probe.world_from_probe,
                                                               self._Wfv, self._cfg_ep)
        self._hist.append(_prep(self._img, self.image_size))
        self._t += 1
        s_prev = self._s
        self._s, seg, app = self._score(self._img, self._mask)
        reward = (self._s - s_prev) - self.effort * float(rho @ rho)
        info = self._info(seg, app)
        info.update(residual_twist=rho, base_twist=base, executed_twist=a_exec)
        return self._obs(), float(reward), False, self._t >= self.n_steps, info

    def render(self):
        """``rgb_array``: the current frame (left) and the target frame (right), uint8 (S, 2S, 3)."""
        if self.render_mode != "rgb_array" or self._probe is None:
            return None
        g = np.concatenate([self._img, self._tgt_img], axis=1)
        g = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    def close(self):
        if self._probe is not None:
            self._probe.close()
            self._probe = None

    def provenance(self):
        """Configuration and input fingerprints of this environment (seat, windows file, texture and
        segmenter md5, cohort, reward and controller settings) for a result file."""
        from . import provenance as PROV
        return {"env_id": ENV_ID, "seat": PROV.seat_info(), "windows": PROV.windows_ref(),
                "pipeline": self.pipeline.info(), "patients": list(self.patients), "targets": list(self.targets),
                "min_windows": self.min_windows, "base": self.base,
                "servo_base_rich": None if self.servo is None else self.servo["rich"],
                "residual_frac": self.residual_frac, "contact": self.contact, "randomize": self.randomize,
                "render_seed": self.render_seed, "n_steps": self.n_steps, "history": self.history,
                "image_size": self.image_size, "frame_dtype": self.frame_dtype, "w_seg": self.w_seg,
                "w_app": self.w_app, "effort": self.effort, "setup": dict(self.setup_kw),
                "eval_seed": self.eval_seed, "git": PROV.git_info()}


# ===================================================================================== older adapter
class USProbeGymEnv(_Base):
    """A generic ``gymnasium.Env`` over ``USProbeEnv`` with a caller-supplied reward (kept for existing
    users; the view-matching task is :class:`ViewMatchEnv`).

    reward_fn: ``fn(obs_img, info) -> float`` (default: constant 0.0; supply your task reward).
    observation: stacked history ``(history, image_size, image_size)`` float32 in [0,1].
    action: 6-DoF body twist, ``Box(-MAX_TWIST, MAX_TWIST)``.
    """
    metadata = {"render_modes": []}

    def __init__(self, lv, world_from_volume, T0, *, reward_fn=None, history=4, image_size=128,
                 max_steps=16, renderer="physics", device=None, translator=None, **env_kw):
        if not _HAVE_GYM:
            raise ImportError("gymnasium is required for USProbeGymEnv: `pip install gymnasium`")
        super().__init__()
        self._env = USProbeEnv(lv, world_from_volume, T0, renderer=renderer, device=device,
                               translator=translator, **env_kw)
        self._reward_fn = reward_fn if reward_fn is not None else (lambda obs, info: 0.0)
        self.history = int(history)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self._hist = None
        self._t = 0
        mt = np.asarray(MAX_TWIST, np.float32)
        self.action_space = spaces.Box(low=-mt, high=mt, dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, (self.history, image_size, image_size), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self._env.reset()
        f = _prep(obs, self.image_size).astype(np.float32)
        self._hist = deque([f] * self.history, maxlen=self.history)
        self._t = 0
        return np.stack(self._hist, 0), info

    def step(self, action):
        obs, info = self._env.step(np.asarray(action, np.float64))
        self._hist.append(_prep(obs, self.image_size).astype(np.float32))
        self._t += 1
        reward = float(self._reward_fn(obs, info))
        terminated = False
        truncated = self._t >= self.max_steps
        return np.stack(self._hist, 0), reward, terminated, truncated, info

    def render(self):
        return self._env.render_obs()


if _HAVE_GYM and ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="usbg.gym_env:ViewMatchEnv")


__all__ = ["ENV_ID", "ViewMatchEnv", "ViewPipeline", "calibrate_servo_base", "servo_features",
           "servo_base_twist", "organ_moments", "load_patient", "target_cells", "USProbeGymEnv"]
