"""plug_texture.py: v2.1 pluggability, texture slot (EXPLORATORY, opt-in).

Published unpaired sim-to-real translators added to the texture stage through the existing registries
(usbg.texture_gan.register_trainer and register_translator). Nothing in the library imports this module:
a process opts in by importing it (scripts/plug_texture_exec.py does this before running an existing
script), so every v2 and queued v2.1 code path is unchanged.

Stages (each one an in-repo reimplementation on the in-repo networks, trained with the v2 grayscale
recipe of scripts/run_textures_v2.sh step 5: 128 px, batch 4, 800 iterations, Adam 2e-4 (0.5, 0.999),
ResNet generator ngf 32 with 6 blocks, PatchGAN ndf 32, the 1540 TRAIN-only curvi frames of
data/dataset/v2/pairs_med_curvi.npz against the 404-frame public Vitale TRAIN pool, raw last iterate;
GAN objective as in each official code: LSGAN for FastCUT and UNSB, hinge for DCLGAN):

  fastcut    FastCUT (Park et al., ECCV 2020): the CUT objective without the identity PatchNCE term,
             lambda_NCE 10 and flip-equivariance augmentation, the CUT_mode=FastCUT defaults of the
             official code. Same generator, discriminator, projection head and PatchNCE as the in-repo CUT.
  dclgan     DCLGAN (Han et al., CVPRW 2021): two generators, two discriminators, two projection heads;
             PatchNCE in both directions where each direction embeds the source with one generator's
             encoder and the translation with the other's; L1 identity; lambda_GAN 1, lambda_NCE 2,
             lambda_IDT 1 and the hinge GAN objective (official DCL defaults). No image buffer (the
             official default pool_size 0).
  unsb       UNSB (Kim et al., ICLR 2024): unpaired neural Schroedinger bridge. A time- and noise-
             conditioned ResNet generator run over N = 5 bridge steps (tau 0.01), a time-conditioned
             PatchGAN, an energy network for the entropy term, PatchNCE with the identity term,
             lambda_SB 1 and lambda_NCE 1 (the authors' training command in their README and
             run_train.sh; the argparse default of lambda_SB is 0.1, which the authors override), sizes
             reduced to the in-repo ngf 32, 6 blocks. Inference draws its bridge noise from a fixed seed, so the stage is a
             deterministic function of its input like every other texture stage.
  histmatch  Global histogram matching (Gonzalez and Woods, Digital Image Processing): one monotone
             intensity map fitted once by quantile matching the in-sector pixels of the same sim frames
             to the in-sector pixels of the same real pool. Non-learned, no training randomness.

Inference: FastCUT and DCLGAN save {"G_s2r": ...} and load through the unchanged
texture_gan.TextureTranslator. UNSB and histogram matching save state["arch"] in {"unsb", "histmatch"},
which load_translator resolves through the loaders registered here (so the consumer process must have
imported this module).
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import texture_gan as TG

ANALYSIS_STATUS = "exploratory, v2.1 pluggability"
PLUG_STAGES = ("fastcut", "dclgan", "unsb", "histmatch")
LEARNED_STAGES = ("fastcut", "dclgan", "unsb")

METHODS = {
    "fastcut": {
        "method": "FastCUT (contrastive unpaired translation, fast variant)",
        "citation": "Park, Efros, Zhang, Zhu, Contrastive Learning for Unpaired Image-to-Image Translation, "
                    "ECCV 2020, LNCS 12354, 319-345, doi:10.1007/978-3-030-58545-7_19",
        "code": "official code github.com/taesungp/contrastive-unpaired-translation (CUT_mode FastCUT: "
                "nce_idt False, lambda_NCE 10, flip_equivariance True; checked against the upstream defaults block "
                "reproduced in cyclomon/UNSB d1f644f models/sb_model.py; the CUT copy vendored in third_party/lotus "
                "edits them to lambda_NCE 1 and no flip, so it is NOT the reference); reimplemented on the in-repo CUT",
        "deviation": "in-repo network sizes and 3 PatchNCE layers (as the in-repo CUT); constant lr for the "
                     "v2 800-iteration budget instead of the official 150+50 epoch schedule",
    },
    "dclgan": {
        "method": "DCLGAN (dual contrastive learning)",
        "citation": "Han, Shoeiby, Petersson, Armin, Dual Contrastive Learning for Unsupervised Image-to-Image "
                    "Translation, CVPR Workshops 2021, 746-755, doi:10.1109/CVPRW53098.2021.00084",
        "code": "official code github.com/JunlinHan/DCLGAN, checked at commit f7a7b8e (models/dcl_model.py: lambda_GAN 1, "
                "lambda_NCE 2, lambda_IDT 1, nce_idt True, pool_size 0; loss and D-then-G update order as here; "
                "options/train_options.py: gan_mode 'hinge', the GAN objective used here, in the form of its "
                "models/networks.py GANLoss); reimplemented on the in-repo networks",
        "deviation": "in-repo network sizes and 3 PatchNCE layers (official 4); constant lr for the v2 budget",
    },
    "unsb": {
        "method": "UNSB (unpaired neural Schroedinger bridge)",
        "citation": "Kim, Kwon, Kim, Ye, Unpaired Image-to-Image Translation via Neural Schrodinger Bridge, "
                    "ICLR 2024, arXiv:2305.15086",
        "code": "official code github.com/cyclomon/UNSB, checked at commit d1f644f (models/sb_model.py, "
                "options/base_options.py: sb mode, num_timesteps 5, tau 0.01, lambda_NCE 1, nce_idt True, n_mlp 3; "
                "schedule, bridge chain, D/E/G losses as here); lambda_SB 1.0 as in the authors' training command "
                "(README and run_train.sh: --mode sb --lambda_SB 1.0 --lambda_NCE 1.0; the argparse default is 0.1); "
                "reimplemented on the in-repo network sizes",
        "deviation": "ngf 32 with 6 conditioned residual blocks (official ngf 64, 9 blocks); 3 PatchNCE layers "
                     "(official 5); stride-2 "
                     "convolutions instead of anti-aliased pooling; the energy network's second source batch "
                     "is the previous iteration's independently drawn sim batch (the v2 trainer interface "
                     "draws one sim and one real batch per step); fixed-seed bridge noise at inference",
    },
    "histmatch": {
        "method": "Global histogram matching to the real pool (non-learned)",
        "citation": "Gonzalez and Woods, Digital Image Processing, 4th ed., Pearson, 2018 (histogram "
                    "specification)",
        "code": "in-repo quantile map (the global analogue of skimage.exposure.match_histograms)",
        "deviation": "one global map fitted on pooled in-sector pixels, not a per-frame match",
    },
}

EXCLUDED = {
    "diffusion": {
        "methods": "Echo from Noise (Stojanovski et al., ASMUS 2023, LNCS 14337, 34-43, "
                   "doi:10.1007/978-3-031-44521-7_4); semantic diffusion model SDM (Wang et al., arXiv:2207.00050)",
        "reason": "T denoising passes of an ADM-style U-Net per frame. Measured 2026-10-15 on this machine with "
                  "scripts/bench_diffusion_latency.py (61.8M parameters, 128 px, batch 1, n 20): 212 ms per pass on "
                  "CPU and 29.2 ms on MPS, so T = 50 costs 10.6 s (CPU) and 1.46 s (MPS) per frame, against 3.2 ms "
                  "(MPS) for one CUT-class generator pass and 21 ms for UNSB's 5 passes. One learned-mask base cell "
                  "renders about 2,600 frames (1,000 calibration renders plus 60 episodes x 13 frames x 2 contact "
                  "settings), about 62 min of texture alone on MPS at T = 50, and one RL training seed (57,600 "
                  "steps) about 23 h. No public abdominal CT-to-ultrasound diffusion weights exist (Echo from Noise "
                  "is cardiac), so a model would also have to be trained first (days on MPS). UNSB is the "
                  "diffusion-family translator that fits the loop.",
        "measured": {"bench": "scripts/bench_diffusion_latency.py --n 20", "cpu_ms_per_pass": 212.3,
                     "mps_ms_per_pass": 29.2, "t50_s_per_frame": {"cpu": 10.62, "mps": 1.46}},
    },
    "conpres": {"methods": "ConPres (Tomar et al., MICCAI 2021, doi:10.1007/978-3-030-87237-3_63)",
                "reason": "no public code found"},
    "optional_breadth": {"methods": "GcGAN (doi:10.1109/CVPR.2019.00253), F-LSeSim (doi:10.1109/CVPR46437.2021.01614), "
                                    "MUNIT (doi:10.1007/978-3-030-01219-9_11), UNIT (arXiv:1703.00848)",
                         "reason": "survey status optional (breadth only); not implemented"},
}


# ======================================================================================
# FastCUT
# ======================================================================================
@TG.register_trainer("fastcut")
class FastCUT(TG.CUT):
    """FastCUT on the in-repo CUT: no identity PatchNCE (lambda_Y = 0), lambda_NCE 10, flip equivariance.

    With probability 0.5 the source batch is flipped left-right before G; the translation's features are
    flipped back before PatchNCE against the unflipped source (the official calculate_NCE_loss). The
    curvilinear sector is left-right symmetric, so the flip keeps frames in-domain."""

    def __init__(self, device, ngf=32, ndf=32, n_blocks=6, lr=2e-4, lambda_nce=10.0, nce_tau=0.07,
                 flip_equivariance=True):
        super().__init__(device, ngf=ngf, ndf=ndf, n_blocks=n_blocks, lr=lr, lambda_nce=lambda_nce,
                         nce_tau=nce_tau)
        self.flip_equivariance = flip_equivariance

    def step(self, sim, real):
        import torch
        flipped = bool(self.flip_equivariance and torch.rand(1).item() < 0.5)
        src = torch.flip(sim, [3]) if flipped else sim
        fake = self.G_s2r(src)
        self.opt_d.zero_grad()
        loss_d = 0.5 * (self._adv(self.D_r(real), True) + self._adv(self.D_r(fake.detach()), False))
        loss_d.backward(); self.opt_d.step()
        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        loss_g_adv = self._adv(self.D_r(fake), True)
        fk = self.G_s2r.forward_features(fake)
        if flipped:
            fk = [torch.flip(f, [3]) for f in fk]
        fq = self.G_s2r.forward_features(sim)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        loss_nce = self._patchnce(q, k)
        loss_g = loss_g_adv + self.lambda_nce * loss_nce
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        return {"g": float(loss_g.detach()), "d": float(loss_d.detach()), "nce": float(loss_nce.detach()),
                "flip": float(flipped)}

    def save(self, path):
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"G_s2r": self.G_s2r.state_dict(), "plug_variant": "fastcut"}, path)


# ======================================================================================
# DCLGAN
# ======================================================================================
@TG.register_trainer("dclgan")
class DCLGAN:
    """DCLGAN on the in-repo networks. G_A = G_s2r (sim -> real), G_B = G_r2s, D_r judges real-domain
    frames, D_s sim-domain frames. H1 projects G_A encoder features, H2 G_B encoder features.
    NCE1: key G_A(sim) features (H1), query G_B-encoder features of G_A(sim) (H2).
    NCE2: key G_B(real) features (H2), query G_A-encoder features of G_B(real) (H1).
    loss_G = 0.5 lambda_GAN (adv_A + adv_B) + 0.5 lambda_NCE (NCE1 + NCE2) + 0.5 lambda_IDT (idt_A + idt_B).
    GAN objective: the official default gan_mode 'hinge' (DCLGAN options/train_options.py), in the form of
    its networks.GANLoss: real -> mean(relu(1 - D)), fake -> mean(relu(1 + D)); the generator term is the
    'real' form on D(G(x)). gan_mode='lsgan' gives the least-squares objective of the in-repo CUT instead."""

    def __init__(self, device, ngf=32, ndf=32, n_blocks=6, lr=2e-4, lambda_gan=1.0, lambda_nce=2.0,
                 lambda_idt=1.0, nce_tau=0.07, gan_mode="hinge"):
        import torch
        if gan_mode not in ("hinge", "lsgan"):
            raise ValueError("gan_mode must be 'hinge' (official DCLGAN default) or 'lsgan'")
        self.device = device
        self.gan_mode = gan_mode
        self.lambda_gan, self.lambda_nce, self.lambda_idt, self.tau = lambda_gan, lambda_nce, lambda_idt, nce_tau
        self.G_s2r = TG._resnet_generator(ngf=ngf, n_blocks=n_blocks).to(device)
        self.G_r2s = TG._resnet_generator(ngf=ngf, n_blocks=n_blocks).to(device)
        self.D_r = TG._patch_discriminator(ndf=ndf).to(device)
        self.D_s = TG._patch_discriminator(ndf=ndf).to(device)
        self.H1 = TG._PatchSampleF(device).build(self.G_s2r.feat_channels)
        self.H2 = TG._PatchSampleF(device).build(self.G_r2s.feat_channels)
        b = (0.5, 0.999)
        self.opt_g = torch.optim.Adam(list(self.G_s2r.parameters()) + list(self.G_r2s.parameters()), lr=lr, betas=b)
        self.opt_d = torch.optim.Adam(list(self.D_r.parameters()) + list(self.D_s.parameters()), lr=lr, betas=b)
        self.opt_h = torch.optim.Adam(self.H1.params + self.H2.params, lr=lr, betas=b)

    def _adv(self, d_out, real):
        import torch
        if self.gan_mode == "hinge":            # DCLGAN GANLoss('hinge'): -mean(min(+-D - 1, 0))
            return torch.relu(1.0 - d_out).mean() if real else torch.relu(1.0 + d_out).mean()
        t = torch.ones_like(d_out) if real else torch.zeros_like(d_out)
        return ((d_out - t) ** 2).mean()

    def _nce(self, enc_k, H_k, src, enc_q, H_q, tgt):
        fk = enc_k.forward_features(src)
        fq = enc_q.forward_features(tgt)
        k, ids = H_k(fk)
        q, _ = H_q(fq, patch_ids=ids)
        return TG._patchnce_loss(q, k, self.tau)

    def step(self, sim, real):
        fake_r = self.G_s2r(sim)
        fake_s = self.G_r2s(real)
        self.opt_d.zero_grad()
        loss_d = self.lambda_gan * (
            0.5 * (self._adv(self.D_r(real), True) + self._adv(self.D_r(fake_r.detach()), False))
            + 0.5 * (self._adv(self.D_s(sim), True) + self._adv(self.D_s(fake_s.detach()), False)))
        loss_d.backward(); self.opt_d.step()
        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        adv = self._adv(self.D_r(fake_r), True) + self._adv(self.D_s(fake_s), True)
        nce1 = self._nce(self.G_s2r, self.H1, sim, self.G_r2s, self.H2, fake_r)
        nce2 = self._nce(self.G_r2s, self.H2, real, self.G_s2r, self.H1, fake_s)
        idt = (self.G_s2r(real) - real).abs().mean() + (self.G_r2s(sim) - sim).abs().mean()
        loss_g = 0.5 * self.lambda_gan * adv + 0.5 * self.lambda_nce * (nce1 + nce2) + 0.5 * self.lambda_idt * idt
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        return {"g": float(loss_g.detach()), "d": float(loss_d.detach()), "nce1": float(nce1.detach()),
                "nce2": float(nce2.detach()), "idt": float(idt.detach())}

    def save(self, path):
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"G_s2r": self.G_s2r.state_dict(), "plug_variant": "dclgan"}, path)


# ======================================================================================
# UNSB
# ======================================================================================
def _timestep_embedding(t, dim):
    """Sinusoidal embedding of integer step indices t (B,) -> (B, dim) (DDPM / UNSB get_timestep_embedding)."""
    import torch
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1))
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def unsb_times(T):
    """The UNSB step schedule: times[0] = 0 and times[1..T] increasing to 1 (official sb_model.forward)."""
    incs = np.array([0.0] + [1.0 / (i + 1) for i in range(T - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return np.concatenate([np.zeros(1), times]).astype(np.float32)


def _unsb_generator(ngf=32, n_blocks=6, n_mlp=3, ch=1):
    """Time- and noise-conditioned ResNet generator (UNSB ResnetGenerator_ncsn, in-repo sizes). Encoder and
    decoder follow the in-repo ResNet generator; each residual block adds a time embedding and applies a
    noise-derived (AdaIN-style) affine modulation."""
    import torch
    import torch.nn as nn

    c_enc = ngf * 4
    zdim = tdim = ngf * 4

    class PixelNorm(nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    class CondBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv_block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim))
            self.style = nn.Linear(zdim, dim * 2)
            with torch.no_grad():
                self.style.bias[:dim] = 1.0
                self.style.bias[dim:] = 0.0
            self.dense_t = nn.Linear(tdim, dim)
            nn.init.zeros_(self.dense_t.bias)
            self.conv_fin = nn.Sequential(nn.ReLU(True), nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
                                          nn.InstanceNorm2d(dim))

        def forward(self, x, temb, zemb):
            out = self.conv_block(x)
            gamma, beta = self.style(zemb)[:, :, None, None].chunk(2, 1)
            out = gamma * out + beta
            out = out + self.dense_t(temb)[:, :, None, None]
            return x + self.conv_fin(out)

    class UNSBGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.ngf, self.zdim = ngf, zdim
            self.stem = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True))
            c, down = ngf, []
            for _ in range(2):
                down += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
                c *= 2
            self.down = nn.Sequential(*down)
            self.feat_channels = (ngf, c_enc, c_enc)
            self.res = nn.ModuleList([CondBlock(c_enc) for _ in range(n_blocks)])
            up = []
            for _ in range(2):
                up += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c // 2, 3, padding=1),
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
                c //= 2
            up += [nn.ReflectionPad2d(3), nn.Conv2d(c, ch, 7), nn.Tanh()]
            self.dec = nn.Sequential(*up)
            zl = [PixelNorm(), nn.Linear(zdim, zdim), nn.LeakyReLU(0.2)]
            for _ in range(n_mlp):
                zl += [nn.Linear(zdim, zdim), nn.LeakyReLU(0.2)]
            self.z_transform = nn.Sequential(*zl)
            self.time_embed = nn.Sequential(nn.Linear(ngf, tdim), nn.LeakyReLU(0.2), nn.Linear(tdim, tdim),
                                            nn.LeakyReLU(0.2))

        def _cond(self, t, z):
            return self.time_embed(_timestep_embedding(t, self.ngf)), self.z_transform(z)

        def forward(self, x, t, z):
            temb, zemb = self._cond(t, z)
            h = self.down(self.stem(x))
            for b in self.res:
                h = b(h, temb, zemb)
            return self.dec(h)

        def forward_features(self, x, t, z):
            temb, zemb = self._cond(t, z)
            f_early = self.stem(x)
            f_enc = self.down(f_early)
            h, f_mid = f_enc, f_enc
            for j, b in enumerate(self.res):
                h = b(h, temb, zemb)
                if j == len(self.res) // 2:
                    f_mid = h
            return [f_early, f_enc, f_mid]

    return UNSBGen()


def _cond_patch_discriminator(in_ch=1, ndf=32, n_layers=3):
    """Time-conditioned PatchGAN (UNSB NLayerDiscriminator_ncsn, in-repo sizes): the in-repo PatchGAN with a
    time embedding added as a per-channel bias after each intermediate convolution. With x2 the two inputs
    are concatenated along channels (the energy network's (pair, pair) call)."""
    import torch
    import torch.nn as nn

    class CondD(nn.Module):
        def __init__(self):
            super().__init__()
            self.ndf = ndf
            chans, convs, norms = [], [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1)], [None]
            c = ndf
            chans.append(c)
            for _ in range(1, n_layers):
                convs.append(nn.Conv2d(c, c * 2, 4, stride=2, padding=1)); norms.append(nn.InstanceNorm2d(c * 2))
                c *= 2; chans.append(c)
            convs.append(nn.Conv2d(c, c * 2, 4, stride=1, padding=1)); norms.append(nn.InstanceNorm2d(c * 2))
            chans.append(c * 2)
            self.convs = nn.ModuleList(convs)
            self.norms = nn.ModuleList([n if n is not None else nn.Identity() for n in norms])
            self.t_embed = nn.Sequential(nn.Linear(ndf, ndf * 4), nn.LeakyReLU(0.2), nn.Linear(ndf * 4, ndf * 4),
                                         nn.LeakyReLU(0.2))
            self.t_dense = nn.ModuleList([nn.Linear(ndf * 4, ch_) for ch_ in chans])
            self.out = nn.Conv2d(c * 2, 1, 4, stride=1, padding=1)

        def forward(self, x, t, x2=None):
            if x2 is not None:
                x = torch.cat([x, x2], dim=1)
            temb = self.t_embed(_timestep_embedding(t, self.ndf))
            h = x
            for conv, norm, dense in zip(self.convs, self.norms, self.t_dense):
                h = conv(h) + dense(temb)[:, :, None, None]
                h = nn.functional.leaky_relu(norm(h), 0.2)
            return self.out(h)

    return CondD()


class _UNSBSampler:
    """Callable inference chain (an nn.Module, so .eval() / .to() work): N bridge steps from the source,
    each a generator pass at step index t with noise z and a Gaussian bridge increment. The noise comes
    from a CPU torch.Generator re-seeded on every call, so the output is a deterministic function of the
    input (the same draws for every frame of one shape)."""

    def __new__(cls, G, T, tau, seed=0):
        import torch
        import torch.nn as nn

        class Sampler(nn.Module):
            def __init__(self):
                super().__init__()
                self.G = G
                self.T, self.tau, self.seed = int(T), float(tau), int(seed)
                self.register_buffer("times", torch.from_numpy(unsb_times(T)), persistent=False)

            def forward(self, x):
                g = torch.Generator(device="cpu").manual_seed(self.seed)
                dev, B = x.device, x.shape[0]
                times = self.times.to(dev)
                Xt, Xt_1 = x, None
                for t in range(self.T):
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = delta / denom
                        scale = delta * (1 - delta / denom)
                        noise = torch.randn(Xt.shape, generator=g).to(dev)
                        Xt = (1 - inter) * Xt + inter * Xt_1 + torch.sqrt(scale * self.tau) * noise
                    z = torch.randn((B, self.G.zdim), generator=g).to(dev)
                    Xt_1 = self.G(Xt, torch.full((B,), t, dtype=torch.long, device=dev), z)
                return Xt_1

        return Sampler()


@TG.register_trainer("unsb")
class UNSB:
    """UNSB trainer (official sb_model.optimize_parameters order: D, then E, then G and the projection head).

    Per step a bridge index t is drawn uniformly from {0..N-1}; the source, a second source batch and (for
    the identity term) the real batch are pushed through the current generator's bridge up to t without
    gradients; then
      D:  LSGAN, D(fake_t, t) vs D(real, t)
      E:  -E(XY, XY) + lse + lse^2, lse = logsumexp(E(XY, XY')) over all patch outputs
      G:  LSGAN + lambda_SB [-(N - t)/N tau (E(XY, XY) - lse) + tau ||X_t - fake_t||^2]
          + lambda_NCE 0.5 (PatchNCE(source, fake_t) + PatchNCE(real, idt_t))
    G_s2r is the inference chain, so train_texture_gan's preview and save work unchanged."""

    def __init__(self, device, ngf=32, ndf=32, n_blocks=6, lr=2e-4, num_timesteps=5, tau=0.01,
                 lambda_gan=1.0, lambda_sb=1.0, lambda_nce=1.0, nce_idt=True, n_mlp=3, nce_tau=0.07,
                 infer_seed=0):
        import torch
        self.device = device
        self.T, self.tau = int(num_timesteps), float(tau)
        self.lambda_gan, self.lambda_sb, self.lambda_nce, self.nce_idt = lambda_gan, lambda_sb, lambda_nce, nce_idt
        self.nce_tau = nce_tau
        self._cfg = {"ngf": ngf, "n_blocks": n_blocks, "n_mlp": n_mlp, "num_timesteps": self.T, "tau": self.tau,
                     "infer_seed": int(infer_seed), "lambda_gan": lambda_gan, "lambda_sb": lambda_sb,
                     "lambda_nce": lambda_nce, "nce_idt": nce_idt}
        self.G = _unsb_generator(ngf=ngf, n_blocks=n_blocks, n_mlp=n_mlp).to(device)
        self.D = _cond_patch_discriminator(in_ch=1, ndf=ndf).to(device)
        self.E = _cond_patch_discriminator(in_ch=4, ndf=ndf).to(device)
        self.H = TG._PatchSampleF(device).build(self.G.feat_channels)
        self.times = torch.from_numpy(unsb_times(self.T)).to(device)
        b = (0.5, 0.999)
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=b)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=b)
        self.opt_e = torch.optim.Adam(self.E.parameters(), lr=lr, betas=b)
        self.opt_h = torch.optim.Adam(self.H.params, lr=lr, betas=b)
        self._prev_sim = None
        self._sampler = _UNSBSampler(self.G, self.T, self.tau, infer_seed)

    @property
    def G_s2r(self):
        return self._sampler

    @staticmethod
    def _adv(d_out, real):
        import torch
        t = torch.ones_like(d_out) if real else torch.zeros_like(d_out)
        return ((d_out - t) ** 2).mean()

    @staticmethod
    def _req(net, flag):
        for p in net.parameters():
            p.requires_grad_(flag)

    def _z(self, n):
        import torch
        return torch.randn(n, self.G.zdim, device=self.device)

    def _nce(self, src, tgt):
        import torch
        n = src.shape[0]
        t0 = torch.zeros(n, dtype=torch.long, device=self.device)
        z = self._z(n)
        fk = self.G.forward_features(tgt, t0, z)
        fq = self.G.forward_features(src, t0, z)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        return TG._patchnce_loss(q, k, self.nce_tau)

    def step(self, sim, real):
        import torch
        B = sim.shape[0]
        sim2 = self._prev_sim if (self._prev_sim is not None and self._prev_sim.shape == sim.shape) \
            else torch.roll(sim, 1, dims=0)
        self._prev_sim = sim.detach()
        times, tau, T = self.times, self.tau, self.T
        t_sel = int(torch.randint(T, (1,)).item())
        with torch.no_grad():
            self.G.eval()
            Xt, Xt2, XtB = sim, sim2, real
            Xt_1 = Xt_12 = Xt_1B = None
            for t in range(t_sel + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = delta / denom
                    sd = torch.sqrt(delta * (1 - delta / denom) * tau)
                    Xt = (1 - inter) * Xt + inter * Xt_1 + sd * torch.randn_like(Xt)
                    Xt2 = (1 - inter) * Xt2 + inter * Xt_12 + sd * torch.randn_like(Xt2)
                    if self.nce_idt:
                        XtB = (1 - inter) * XtB + inter * Xt_1B + sd * torch.randn_like(XtB)
                tt = torch.full((B,), t, dtype=torch.long, device=self.device)
                Xt_1 = self.G(Xt, tt, self._z(B))
                Xt_12 = self.G(Xt2, tt, self._z(B))
                if self.nce_idt:
                    Xt_1B = self.G(XtB, tt, self._z(B))
            self.G.train()
        xa, xa2 = Xt.detach(), Xt2.detach()
        tt = torch.full((B,), t_sel, dtype=torch.long, device=self.device)
        if self.nce_idt:
            out = self.G(torch.cat([xa, XtB.detach()], 0), torch.cat([tt, tt], 0), self._z(2 * B))
            fake, idt = out[:B], out[B:]
        else:
            fake, idt = self.G(xa, tt, self._z(B)), None
        fake2 = self.G(xa2, tt, self._z(B))
        # D
        self._req(self.D, True)
        self.opt_d.zero_grad()
        loss_d = 0.5 * (self._adv(self.D(fake.detach(), tt), False) + self._adv(self.D(real, tt), True))
        loss_d.backward(); self.opt_d.step()
        # E
        self._req(self.E, True)
        self.opt_e.zero_grad()
        xy1 = torch.cat([xa, fake.detach()], 1); xy2 = torch.cat([xa2, fake2.detach()], 1)
        lse = torch.logsumexp(self.E(xy1, tt, xy2).reshape(-1), dim=0)
        loss_e = -self.E(xy1, tt, xy1).mean() + lse + lse ** 2
        loss_e.backward(); self.opt_e.step()
        # G (+ projection head)
        self._req(self.D, False); self._req(self.E, False)
        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        loss_gan = self._adv(self.D(fake, tt), True) * self.lambda_gan
        xy1 = torch.cat([xa, fake], 1); xy2 = torch.cat([xa2, fake2], 1)
        et_xy = self.E(xy1, tt, xy1).mean() - torch.logsumexp(self.E(xy1, tt, xy2).reshape(-1), dim=0)
        loss_sb = -(T - t_sel) / T * tau * et_xy + tau * torch.mean((xa - fake) ** 2)
        nce = self._nce(sim, fake)
        if self.nce_idt:
            nce = 0.5 * (nce + self._nce(real, idt))
        loss_g = loss_gan + self.lambda_sb * loss_sb + self.lambda_nce * nce
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        self._req(self.D, True); self._req(self.E, True)
        return {"g": float(loss_g.detach()), "d": float(loss_d.detach()), "e": float(loss_e.detach()),
                "sb": float(loss_sb.detach()), "nce": float(nce.detach()), "t": float(t_sel)}

    def save(self, path):
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"arch": "unsb", "G_unsb": self.G.state_dict(), "plug_variant": "unsb", **self._cfg}, path)


class UNSBTranslator:
    """Inference-time UNSB stage: img01 (H,W) [0,1] -> (H,W) [0,1] after N fixed-seed bridge steps."""

    wants_label = False

    def __init__(self, ckpt_path, device=None):
        import torch
        from .render_lotus import get_device
        self.device = device or get_device()
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.G = _unsb_generator(ngf=st.get("ngf", 32), n_blocks=st.get("n_blocks", 6),
                                 n_mlp=st.get("n_mlp", 3)).to(self.device)
        self.G.load_state_dict(st["G_unsb"])
        self.G.eval()
        self.chain = _UNSBSampler(self.G, st.get("num_timesteps", 5), st.get("tau", 0.01),
                                  st.get("infer_seed", 0)).to(self.device)
        self.chain.eval()

    def translate(self, img01):
        import torch
        a = torch.from_numpy(np.asarray(img01, np.float32) * 2 - 1)[None, None].to(self.device)
        with torch.no_grad():
            out = self.chain(a)[0, 0].cpu().numpy()
        return np.clip((out + 1) / 2, 0, 1).astype(np.float32)

    __call__ = translate


@TG.register_translator("unsb")
def _load_unsb(ckpt_path, device=None):
    return UNSBTranslator(ckpt_path, device=device)


# ======================================================================================
# global histogram matching (non-learned)
# ======================================================================================
def sector_mask(h=128, w=128, erode=0):
    """The measured Vitale sector mask (usbg.view_match.SECTOR) at 128 px, optionally eroded; None for any
    other frame size (the SECTOR parameters are in 128-px units)."""
    if (h, w) != (128, 128):
        return None
    from . import view_match as VM
    m = VM._sector_coords(h, w, VM.SECTOR)[2]
    if erode > 0:
        from scipy.ndimage import binary_erosion
        m = binary_erosion(m, iterations=int(erode))
    return m


def fit_histmatch(sim_frames, real_frames, erode=2, levels=256):
    """Quantile map LUT (levels,) from pooled in-sector sim pixels to pooled in-sector real pixels.

    LUT[v] = Q_real(F_sim(v)), with F_sim the mid-rank empirical CDF of the sim pixels quantized to
    `levels` grey levels. Monotone non-decreasing by construction."""
    sim = np.asarray(sim_frames, np.float32)
    real = np.asarray(real_frames, np.float32)
    m = sector_mask(*sim.shape[1:], erode=erode)
    s = sim[:, m].ravel() if m is not None else sim.ravel()
    mr = sector_mask(*real.shape[1:], erode=erode)
    r = real[:, mr].ravel() if mr is not None else real.ravel()
    sq = np.clip(np.rint(s * (levels - 1)), 0, levels - 1).astype(np.int64)
    cnt = np.bincount(sq, minlength=levels).astype(np.float64)
    below = np.concatenate([[0.0], np.cumsum(cnt)[:-1]])
    cdf = (below + 0.5 * cnt) / max(cnt.sum(), 1.0)
    lut = np.quantile(r, np.clip(cdf, 0.0, 1.0)).astype(np.float32)
    lut = np.maximum.accumulate(lut)
    stats = {"n_sim_px": int(s.size), "n_real_px": int(r.size), "erode_px": int(erode), "levels": int(levels),
             "sim_mean": float(s.mean()), "real_mean": float(r.mean()),
             "mapped_mean": float((lut[sq]).mean())}
    return lut, stats


def save_histmatch(path, lut, stats):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"arch": "histmatch", "lut": torch.from_numpy(np.asarray(lut, np.float32)), "stats": stats,
                "plug_variant": "histmatch", "mask": "usbg.view_match.SECTOR at 128 px (outside -> 0)"}, path)


class HistMatchTranslator:
    """Inference-time global histogram map: in-sector pixels -> LUT, outside the sector -> 0 (128 px curvi
    frames; any other size maps every pixel)."""

    wants_label = False

    def __init__(self, ckpt_path, device=None):
        import torch
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.lut = np.asarray(st["lut"].numpy() if hasattr(st["lut"], "numpy") else st["lut"], np.float32)
        self.levels = len(self.lut)
        self.device = "cpu"

    def translate(self, img01):
        img = np.asarray(img01, np.float32)
        idx = np.clip(np.rint(img * (self.levels - 1)), 0, self.levels - 1).astype(np.int64)
        out = self.lut[idx]
        m = sector_mask(*img.shape)
        if m is not None:
            out = out * m
        return np.clip(out, 0, 1).astype(np.float32)

    __call__ = translate


@TG.register_translator("histmatch")
def _load_histmatch(ckpt_path, device=None):
    return HistMatchTranslator(ckpt_path, device=device)


# ======================================================================================
# opt-in CLI plumbing: widen the stage choices of existing scripts in THIS process only
# ======================================================================================
_WIDENED = False


def widen_choices(extra=PLUG_STAGES):
    """Patch argparse in this process so an existing script's --variant / --translator choice list that
    already contains 'cut' also accepts the plug stages. Opt-in (scripts/plug_texture_exec.py calls it);
    no file is edited and no other process is affected."""
    global _WIDENED
    if _WIDENED:
        return
    import argparse
    orig = argparse._ActionsContainer.add_argument

    def add_argument(self, *args, **kw):
        ch = kw.get("choices")
        if ch is not None and any(a in ("--variant", "--translator") for a in args) and "cut" in list(ch):
            kw["choices"] = list(ch) + [s for s in extra if s not in ch]
        return orig(self, *args, **kw)

    argparse._ActionsContainer.add_argument = add_argument
    _WIDENED = True


def time_translator(tr, frames, n=30, warmup=3):
    """Mean / median ms per frame of a translator call on (H,W) [0,1] frames, batch 1 (MPS synchronized)."""
    import time
    import torch
    dev = str(getattr(tr, "device", "cpu"))
    fr = [np.asarray(frames[i % len(frames)], np.float32) for i in range(n + warmup)]
    ts = []
    for i, f in enumerate(fr):
        t0 = time.perf_counter()
        tr(f)
        if dev.startswith("mps"):
            torch.mps.synchronize()
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {"device": dev, "n": int(len(ts)), "ms_mean": float(ts.mean()), "ms_median": float(np.median(ts)),
            "ms_sd": float(ts.std())}


__all__ = ["ANALYSIS_STATUS", "PLUG_STAGES", "METHODS", "EXCLUDED", "FastCUT", "DCLGAN", "UNSB", "UNSBTranslator",
           "HistMatchTranslator", "fit_histmatch", "save_histmatch", "widen_choices", "time_translator", "unsb_times"]
