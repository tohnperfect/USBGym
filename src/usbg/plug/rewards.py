"""usbg.plug.rewards: published reward and similarity terms for the view-matching task (v2.1 pluggability,
exploratory, opt-in).

Nothing in the v2 or v2.1 job queues imports this module. It is reached only through
scripts/plug_reward_landscape.py, scripts/plug_reward_train.py and the gymnasium wrapper below, and every
default of the existing code stays as it was.

The paper's step reward is r_t = S_t - S_{t-1} - effort * |rho_t|^2 with the potential
S = w_seg * S_seg + w_app * S_app (view_match.combined_score): S_seg the mean per-organ IoU of the two
organ masks, S_app the normalized cross-correlation (NCC) of the lit sector pixels. A plug here replaces
the POTENTIAL S and keeps everything else (the gain form, the action cost, the learner, the evaluation).

Appearance terms (cur, tgt) -> [0, 1], higher = more similar. All but ncc_sector and ncc_frame use the
paper's border rule: the pixels lit (> 0.04) in either frame, so the shared black border of the
curvilinear frame never enters the score.

    ncc_lit     the paper's term, view_match.sector_appearance (the reference; the function itself)
    ncc_sector  NCC over the whole geometric sector (the fan of view_match.SECTOR, dark interior pixels
                included); isolates the lit-pixel rule
    ncc_frame   NCC over the whole frame, border included (the control the paper argues against)
    ssim        SSIM map (Wang et al. 2004: Gaussian window sigma 1.5, K1 0.01, K2 0.03) averaged over
                the lit pixels, mapped (s + 1) / 2
    nmi         normalized mutual information (Studholme et al. 1999), (H(A) + H(B)) / H(A, B) - 1,
                32-bin joint histogram of the lit pixels
    mi          mutual information (Viola and Wells 1997; Maes et al. 1997) of the same histogram,
                divided by log(32) so it lies in [0, 1]
    lpips       1 - LPIPS distance (Zhang et al. 2018, AlexNet trunk, v0.1 linear heads), the spatial
                LPIPS map averaged over the lit pixels; needs the lpips package (plugin venv) and the
                AlexNet ImageNet weights as a local file (never downloaded here)

Potentials (the name given to make_plug / --reward):

    paper        w_seg * S_seg + w_app * ncc_lit. Bit-identical to view_match.combined_score.
    <metric>     w_seg * S_seg + w_app * <metric> for any appearance term above (ncc_sector, ncc_frame,
                 ssim, nmi, mi, lpips): swaps only the appearance term.
    pose         PRIVILEGED. -(translation_mm + ang_w * rotation_deg) / pose_scale to the target pose
                 (view_match.pose_error; ang_w 2.0 is the trainer's ANG_W, pose_scale 100). Its gain is
                 the change in distance to the goal pose, the navigation reward of SonoGym (Ao et al.,
                 NeurIPS 2025) and the pose-improvement rewards of Li et al. (ICRA 2021) and Hase et al.
                 (IROS 2020). The policy still observes images only, so a policy trained on it is
                 non-privileged; the reward itself needs the simulator pose.
    pose_area    PRIVILEGED, adapted from VesNet-RL (Bi et al., 2022: probe distance to the vessel
                 centerline plus the segmented vessel area): the pose potential plus area_coef times the
                 visible fraction of each target-mask organ in the current mask (min(1, |cur o| / |tgt o|),
                 mean over the organs with >= 20 pixels in the target). An adaptation, not their reward.

Term ablations of the paper's reward need no plug: run "paper" with w_seg 1, w_app 0 (segmentation only)
or w_seg 0, w_app 1 (NCC only).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import os

import numpy as np

# The reference functions, bound at import time: the landscape wrapper replaces the view_match module
# attributes with recorders, and these names must keep pointing at the originals.
from usbg.view_match import (MATCH_ORGANS, SECTOR, _sector_coords, pose_error as _ref_pose_error,
                             sector_appearance as _ref_sector_appearance,
                             seg_organ_iou as _ref_seg_organ_iou)

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
LIT_THRESH = 0.04          # = view_match.sector_appearance's default threshold
MIN_PIX = 50               # = view_match.sector_appearance's minimum mask size
N_BINS = 32
ANG_W = 2.0                # = scripts/train_view_match_rl.ANG_W (deg -> mm-equivalent)
POSE_SCALE = 100.0
AREA_COEF = 0.5
AREA_MIN_PX = 20           # = view_match.seg_organ_iou's min_pixels
ALEXNET_FILE = "alexnet-owt-7be5be79.pth"      # torchvision release name; 7be5be79 = sha256 prefix
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
DEFAULT_ALEXNET = os.path.join(os.environ.get("USBG_PLUGIN_WEIGHTS", os.path.expanduser("~/.cache/usbg_plugin_weights")), "torch_home", "hub",
                               "checkpoints", ALEXNET_FILE)

CITATIONS = {
    "paper": "own (USB Gym view-matching score S = w_seg S_seg + w_app S_app)",
    "ncc_lit": "own; NCC: Lewis, Fast normalized cross-correlation, Vision Interface 1995",
    "ncc_sector": "own variant of ncc_lit (whole geometric sector); NCC: Lewis, Vision Interface 1995",
    "ncc_frame": "own control (whole frame); NCC: Lewis, Vision Interface 1995",
    "ssim": "Wang, Bovik, Sheikh, Simoncelli, IEEE Trans. Image Process. 13 (2004) 600-612, "
            "doi:10.1109/TIP.2003.819861",
    "nmi": "Studholme, Hill, Hawkes, Pattern Recognit. 32 (1999) 71-86, doi:10.1016/S0031-3203(98)00091-0",
    "mi": "Viola and Wells, Int. J. Comput. Vis. 24 (1997) 137-154, doi:10.1023/A:1007958904918; "
          "Maes et al., IEEE Trans. Med. Imaging 16 (1997) 187-198, doi:10.1109/42.563664",
    "lpips": "Zhang, Isola, Efros, Shechtman, Wang, CVPR 2018, 586-595, doi:10.1109/CVPR.2018.00068; "
             "lpips 0.1.4 (BSD-2-Clause); AlexNet trunk: Krizhevsky, Sutskever, Hinton, NeurIPS 2012",
    "pose": "change in distance to the goal pose as in SonoGym (Ao et al., NeurIPS 2025, "
            "doi:10.52202/085713-1343); pose-improvement rewards of Li et al., ICRA 2021, 8302-8308 and "
            "Hase et al., IROS 2020, 5534-5541, doi:10.1109/IROS45743.2020.9340913",
    "pose_area": "adapted from VesNet-RL (Bi et al., IEEE Robot. Autom. Lett. 7 (2022) 6638-6645, "
                 "doi:10.1109/LRA.2022.3176112): distance term plus a visible-structure area term",
}
APPEARANCE = ("ncc_lit", "ncc_sector", "ncc_frame", "ssim", "nmi", "mi", "lpips")
PRIVILEGED = ("pose", "pose_area")
REWARDS = ("paper",) + APPEARANCE[1:] + PRIVILEGED


# ======================================================================================= masks and NCC
def lit_mask(a, b, thresh=LIT_THRESH):
    """The paper's border rule: pixels lit in either frame (view_match.sector_appearance)."""
    return (a > thresh) | (b > thresh)


@functools.lru_cache(maxsize=8)
def _sector_mask_cached(H, W):
    return _sector_coords(H, W, SECTOR)[2]


def sector_mask(H, W):
    """The geometric fan of view_match.SECTOR at (H, W): where warp_to_sector writes image content."""
    return _sector_mask_cached(int(H), int(W))


def _ncc01(av, bv):
    av = av - av.mean(); bv = bv - bv.mean()
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    ncc = float((av * bv).sum() / denom) if denom > 1e-8 else 0.0
    return 0.5 * (ncc + 1.0)


def _pair(cur, tgt):
    return np.asarray(cur, np.float32), np.asarray(tgt, np.float32)


# ======================================================================================= appearance terms
def app_ncc_lit(cur, tgt):
    """The paper's appearance term, the original view_match.sector_appearance."""
    return float(_ref_sector_appearance(cur, tgt))


def app_ncc_sector(cur, tgt):
    a, b = _pair(cur, tgt)
    m = sector_mask(*a.shape)
    if int(m.sum()) < MIN_PIX:
        return 0.0
    return _ncc01(a[m], b[m])


def app_ncc_frame(cur, tgt):
    a, b = _pair(cur, tgt)
    return _ncc01(a.ravel(), b.ravel())


def app_ssim(cur, tgt):
    try:
        from skimage.metrics import structural_similarity
    except ImportError as e:                                        # pragma: no cover  (optional dependency)
        raise ImportError("the SSIM appearance term needs scikit-image: pip install \"usbg[plug]\"") from e
    a, b = _pair(cur, tgt)
    m = lit_mask(a, b)
    if int(m.sum()) < MIN_PIX:
        return 0.0
    _, smap = structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(np.clip(0.5 * (float(smap[m].mean()) + 1.0), 0.0, 1.0))


def _entropies(a, b, bins=N_BINS):
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    p = h / max(h.sum(), 1.0)
    pa, pb = p.sum(1), p.sum(0)

    def H(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())
    return H(pa), H(pb), H(p.ravel())


def app_nmi(cur, tgt):
    a, b = _pair(cur, tgt)
    m = lit_mask(a, b)
    if int(m.sum()) < MIN_PIX:
        return 0.0
    ha, hb, hab = _entropies(np.clip(a[m], 0, 1), np.clip(b[m], 0, 1))
    if hab <= 1e-12:
        return 1.0 if ha <= 1e-12 and hb <= 1e-12 else 0.0
    return float(np.clip((ha + hb) / hab - 1.0, 0.0, 1.0))


def app_mi(cur, tgt):
    a, b = _pair(cur, tgt)
    m = lit_mask(a, b)
    if int(m.sum()) < MIN_PIX:
        return 0.0
    ha, hb, hab = _entropies(np.clip(a[m], 0, 1), np.clip(b[m], 0, 1))
    return float(np.clip((ha + hb - hab) / np.log(N_BINS), 0.0, 1.0))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


class LPIPSAppearance:
    """1 - LPIPS (AlexNet, v0.1), the spatial distance map averaged over the lit pixels.

    The trunk weights are read from a LOCAL torchvision AlexNet file (alexnet_path, else
    $USBG_PLUG_ALEXNET, else $TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth, else the plugin
    weights directory) and checked against the sha256 prefix in the file name; nothing is downloaded.
    random_trunk=True keeps a randomly initialized trunk: a CODE-PATH SMOKE TEST ONLY, not LPIPS."""

    def __init__(self, alexnet_path=None, device="cpu", random_trunk=False):
        import torch
        import lpips
        self.torch, self.device, self.random_trunk = torch, device, bool(random_trunk)
        # pnet_rand=True: torchvision builds AlexNet with weights=None, so lpips never downloads; the v0.1
        # linear heads come from the lpips wheel (pretrained=True)
        net = lpips.LPIPS(net="alex", pretrained=True, pnet_rand=True, spatial=True, verbose=False)
        self.alexnet_path, self.alexnet_sha256 = None, None
        if not self.random_trunk:
            path = alexnet_path or os.environ.get("USBG_PLUG_ALEXNET")
            if not path and os.environ.get("TORCH_HOME"):
                path = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", ALEXNET_FILE)
            path = path or DEFAULT_ALEXNET
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"LPIPS needs the AlexNet ImageNet weights as a local file ({path} missing). They are "
                    f"not downloaded here: place {ALEXNET_URL} there (a one-time download).")
            sha = sha256_file(path)
            if not sha.startswith("7be5be79"):
                raise ValueError(f"{path}: sha256 {sha[:16]}... does not match the release tag 7be5be79")
            sd = torch.load(path, map_location="cpu", weights_only=True)
            with torch.no_grad():
                for sl in (net.net.slice1, net.net.slice2, net.net.slice3, net.net.slice4, net.net.slice5):
                    for name, p in sl.named_parameters():
                        p.copy_(sd[f"features.{name}"])
            self.alexnet_path, self.alexnet_sha256 = os.path.abspath(path), sha
        self.net = net.to(device).eval()
        self.lpips_version = _dist_version("lpips")

    def distance_map(self, cur, tgt):
        torch = self.torch
        a, b = _pair(cur, tgt)

        def t(x):
            return torch.from_numpy(np.repeat((x * 2.0 - 1.0)[None, None], 3, axis=1)).to(self.device)
        with torch.no_grad():
            d = self.net(t(a), t(b))                 # (1, 1, H, W): spatial=True upsamples to the input
        return d[0, 0].detach().cpu().numpy()

    def __call__(self, cur, tgt):
        a, b = _pair(cur, tgt)
        m = lit_mask(a, b)
        if int(m.sum()) < MIN_PIX:
            return 0.0
        d = self.distance_map(a, b)
        return float(np.clip(1.0 - float(d[m].mean()), 0.0, 1.0))

    def info(self):
        return {"net": "alex", "version": "0.1", "lpips_package": self.lpips_version,
                "random_trunk_smoke_only": self.random_trunk, "alexnet_path": self.alexnet_path,
                "alexnet_sha256": self.alexnet_sha256, "alexnet_url": ALEXNET_URL, "device": self.device}


_SIMPLE = {"ncc_lit": app_ncc_lit, "ncc_sector": app_ncc_sector, "ncc_frame": app_ncc_frame,
           "ssim": app_ssim, "nmi": app_nmi, "mi": app_mi}


def appearance_fn(name, lpips_kw=None):
    """Appearance term by name -> callable (cur, tgt) -> [0, 1]."""
    if name in _SIMPLE:
        return _SIMPLE[name]
    if name == "lpips":
        return LPIPSAppearance(**(lpips_kw or {}))
    raise ValueError(f"unknown appearance term {name!r} (expected one of {APPEARANCE})")


# ======================================================================================= potentials
def pose_potential(pose, tgt_pose, ang_w=ANG_W, pose_scale=POSE_SCALE):
    """-(translation_mm + ang_w * rotation_deg) / pose_scale (privileged: needs the simulator pose)."""
    pe, ae = _ref_pose_error(pose, tgt_pose)
    return -(pe + ang_w * ae) / pose_scale


def visible_area(cur_lab, tgt_lab, organs=MATCH_ORGANS, min_px=AREA_MIN_PX):
    """Mean over the target-mask organs (>= min_px) of min(1, |cur == o| / |tgt == o|); 0 if none."""
    c = np.asarray(cur_lab); t = np.asarray(tgt_lab)
    vals = []
    for o in organs:
        nt = int((t == o).sum())
        if nt >= min_px:
            vals.append(min(1.0, int((c == o).sum()) / nt))
    return float(np.mean(vals)) if vals else 0.0


class RewardPlug:
    """One reward potential S. potential(...) -> float; the step reward stays S_t - S_{t-1} - effort|rho|^2.

    w_seg, w_app: weights of the combined forms (paper and the swapped appearance terms); a caller that
    passes its own weights per call (the trainer does) overrides them."""

    def __init__(self, name="paper", *, w_seg=0.6, w_app=0.4, ang_w=ANG_W, pose_scale=POSE_SCALE,
                 area_coef=AREA_COEF, lpips_kw=None):
        if name not in REWARDS:
            raise ValueError(f"unknown reward {name!r} (expected one of {REWARDS})")
        self.name, self.w_seg, self.w_app = name, float(w_seg), float(w_app)
        self.ang_w, self.pose_scale, self.area_coef = float(ang_w), float(pose_scale), float(area_coef)
        self.privileged = name in PRIVILEGED
        self.app_name = None if self.privileged else ("ncc_lit" if name == "paper" else name)
        self._app = None if self.app_name is None else appearance_fn(self.app_name, lpips_kw)

    @property
    def needs_pose(self):
        return self.privileged

    def __call__(self, *args, **kw) -> float:
        """The potential S (so a RewardPlug satisfies usbg.interfaces.Reward); see potential()."""
        return self.potential(*args, **kw)

    def potential(self, cur_img, cur_lab, tgt_img, tgt_lab, *, pose=None, tgt_pose=None, w_seg=None,
                  w_app=None):
        if self.privileged:
            if pose is None or tgt_pose is None:
                raise ValueError(f"reward {self.name!r} is privileged: it needs the probe and target poses")
            s = pose_potential(pose, tgt_pose, self.ang_w, self.pose_scale)
            if self.name == "pose_area":
                s = s + self.area_coef * visible_area(cur_lab, tgt_lab)
            return float(s)
        ws = self.w_seg if w_seg is None else w_seg
        wa = self.w_app if w_app is None else w_app
        seg = _ref_seg_organ_iou(cur_lab, tgt_lab)
        app = self._app(cur_img, tgt_img)
        return ws * seg + wa * app                      # the arithmetic of view_match.combined_score

    def info(self):
        d = {"slot": "reward", "reward": self.name, "privileged": self.privileged,
             "appearance_term": self.app_name, "citation": CITATIONS[self.name],
             "step_reward": "S_t - S_{t-1} - effort * |rho_t|^2 (the paper's gain form; only S is swapped)",
             "analysis_status": ANALYSIS_STATUS}
        if self.privileged:
            d.update(potential=f"-(translation_mm + {self.ang_w:g} * rotation_deg) / {self.pose_scale:g}"
                     + (f" + {self.area_coef:g} * visible_area" if self.name == "pose_area" else ""),
                     ang_w=self.ang_w, pose_scale=self.pose_scale,
                     area_coef=self.area_coef if self.name == "pose_area" else None)
        else:
            d.update(potential=f"w_seg * S_seg + w_app * {self.app_name}", w_seg=self.w_seg, w_app=self.w_app,
                     appearance_citation=CITATIONS[self.app_name])
        if isinstance(self._app, LPIPSAppearance):
            d["lpips"] = self._app.info()
        return d


def make_plug(name, **kw):
    return RewardPlug(name, **kw)


def _dist_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def package_versions():
    """Installed distribution versions of the packages the plugged terms use (None = not installed)."""
    return {name: _dist_version(name) for name in ("numpy", "scipy", "scikit-image", "torch", "torchvision",
                                                     "lpips", "gymnasium")}


# ======================================================================================= scoped patch
@contextlib.contextmanager
def patched(obj, attr, value):
    """Temporarily set obj.attr (module-level swap; not thread-safe)."""
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


# ======================================================================================= similarity registry
def _register_similarity():
    """``usbg.similarity.load_similarity("plugapp:<term>")`` builds any appearance term above as a Similarity
    (an added registry entry; the existing specs are unchanged)."""
    from usbg import similarity as SIM

    class PlugSimilarity(SIM.Similarity):
        def __init__(self, term, lpips_kw=None):
            self.term, self._fn = term, appearance_fn(term, lpips_kw)

        def score(self, cur01, target01) -> float:
            return float(self._fn(cur01, target01))

    @SIM.register_similarity("plugapp")
    def _build(spec, device=None):
        term = spec.split(":", 1)[1] if ":" in spec else "ncc_lit"
        return PlugSimilarity(term, {"device": device or "cpu"} if term == "lpips" else None)

    return PlugSimilarity


PlugSimilarity = _register_similarity()


# ======================================================================================= gymnasium wrapper
def _gym_wrapper_base():
    try:
        import gymnasium as gym
        return gym.Wrapper
    except ImportError:                                  # pragma: no cover
        return object


class PluggedRewardWrapper(_gym_wrapper_base()):
    """A ``gymnasium.Wrapper`` that swaps the reward potential of ``USBGym/ViewMatch-v0``
    (usbg.plug.viewmatch_env.ViewMatchEnv, a copy of usbg.gym_env.ViewMatchEnv) without touching the
    environment: r_t = S_t - S_{t-1} - effort * |rho_t|^2 with S from the plug, on the environment's
    unquantized frames and masks (the frames its own reward uses). With reward="paper" the reward is
    bit-identical to the environment's. The privileged rewards read the simulator pose and therefore
    require the environment to be built with privileged_info=True, so the privilege is explicit.

        env = PluggedRewardWrapper(ViewMatchEnv(texture="physics", mask="ct"), reward="ssim")
    """

    def __init__(self, env, reward="paper", **plug_kw):
        super().__init__(env)
        u = env.unwrapped
        plug_kw.setdefault("w_seg", getattr(u, "w_seg", 0.6))
        plug_kw.setdefault("w_app", getattr(u, "w_app", 0.4))
        self.plug = reward if isinstance(reward, RewardPlug) else RewardPlug(reward, **plug_kw)
        if self.plug.privileged and not getattr(u, "privileged_info", False):
            raise ValueError(f"reward {self.plug.name!r} is privileged; build the environment with "
                             "privileged_info=True")
        self._s = None

    def _potential(self):
        u = self.env.unwrapped
        return self.plug.potential(u._img, u._mask, u._tgt_img, u._tgt_mask,
                                   pose=u._probe.world_from_probe, tgt_pose=u._T_tgt)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._s = self._potential()
        info["plug_reward"] = {"reward": self.plug.name, "S": float(self._s)}
        return obs, info

    def step(self, action):
        obs, r_env, term, trunc, info = self.env.step(action)
        u = self.env.unwrapped
        s_prev, self._s = self._s, self._potential()
        rho = info["residual_twist"]
        reward = (self._s - s_prev) - u.effort * float(rho @ rho)
        info["plug_reward"] = {"reward": self.plug.name, "S": float(self._s), "env_reward": float(r_env)}
        return obs, float(reward), term, trunc, info


__all__ = ["ANALYSIS_STATUS", "APPEARANCE", "REWARDS", "PRIVILEGED", "CITATIONS", "RewardPlug", "make_plug",
           "appearance_fn", "LPIPSAppearance", "PluggedRewardWrapper", "PlugSimilarity", "pose_potential",
           "visible_area", "lit_mask", "sector_mask", "package_versions", "patched"]
