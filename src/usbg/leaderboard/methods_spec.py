"""usbg.leaderboard.methods_spec: which published method sits in which slot, its citations and its adapter.

The counts follow Table 2 of the USB Gym article (one entry per method, whatever its backbone, box prompt,
pretrained weights or residual form; ablations of the match score S left out): anatomy source 1, renderer 2,
texture stage 4, mask source 7, controller 8, reward 5; 27 in all. The full references, DOIs and license notes are
filled in data/methods.json from the article's reference list and its list of third-party code and weights.
"""
from __future__ import annotations

R_NOTE = ("the authors' reimplementation of the published method, in this repository (Apache-2.0); its weights "
          "are trained by the protocol and not distributed")
R_NOTE_NO_WEIGHTS = ("the authors' reimplementation of the published method, in this repository (Apache-2.0); no "
                     "trained weights")
R_NOTE_HISTMATCH = ("the authors' reimplementation of the published method, in this repository (Apache-2.0); its "
                    "intensity map is fitted by the protocol on the training pool and not distributed")
LOTUS_NOTE = ("the reimplementation (usbg.render_lotus) is the authors' code, adapted from the MIT-licensed LOTUS code "
              "(Copyright (c) 2023 Yordanka Velikova; notice in NOTICE)")
SLOT_OF_COMPONENT = {"Anatomy source": "anatomy", "Renderer": "renderer", "Texture stage": "texture",
                     "Mask source": "mask", "Controller": "ctrl", "Reward": "reward"}


def _m(name, cites, adapter, rows, license_from=(), license=None, note=None):
    return {"name": name, "cites": list(cites), "adapter": adapter, "rows": list(rows),
            "license_from": list(license_from), "license": license, "note": note}


PUBLISHED = {
    "anatomy": [
        _m("TotalSegmentator v2 labels of the same CTs, fast mode", ["totalseg"], "scripts/plug/totalseg_labels.py",
           ["anatomy/R1", "anatomy/R2", "anatomy/R3", "anatomy/R4", "anatomy/R5"], ["totalseg"])],
    "renderer": [
        _m("LOTUS forward model (original code with published default parameters; the authors' reimplementation)",
           ["velikova2023lotus", "salehi2015"], "usbg.plug.renderers (lotus_orig, lotus_reimpl)",
           ["renderer/lotus_orig", "renderer/lotus_reimpl"], ["velikova2023lotus"],
           note=LOTUS_NOTE),
        _m("SIMUS via PyMUST", ["garcia2022simus", "cigier2022simus"], "usbg.plug.renderers (simus)",
           ["renderer/simus"], ["garcia2022simus"])],
    "texture": [
        _m("FastCUT", ["cut"], "usbg.plug_texture.FastCUT", ["texture/fastcut"], license=R_NOTE),
        _m("DCLGAN", ["han2021dclgan"], "usbg.plug_texture.DCLGAN", ["texture/dclgan"], license=R_NOTE),
        _m("UNSB (neural Schrödinger bridge)", ["kim2024unsb"], "usbg.plug_texture.UNSB", ["texture/unsb"],
           license=R_NOTE),
        _m("Global histogram matching", ["gonzalez2018"], "usbg.plug_texture.HistMatchTranslator",
           ["texture/histmatch"], license=R_NOTE_HISTMATCH)],
    "mask": [
        _m("LR-ASPP (MobileNetV3-L backbone)", ["howard2019mobilenetv3"], "usbg.plug.masks.TorchvisionMaskSource",
           ["mask/lraspp_mbv3", "mask/lraspp_mbv3_imnet"], ["torchvision"]),
        _m("DeepLabV3 (MobileNetV3-L or ResNet-50 backbone)", ["chen2017deeplabv3", "howard2019mobilenetv3",
                                                               "he2016resnet"],
           "usbg.plug.masks.TorchvisionMaskSource", ["mask/deeplabv3_mbv3", "mask/deeplabv3_r50",
                                                     "mask/deeplabv3_mbv3_imnet", "mask/deeplabv3_r50_imnet"],
           ["torchvision"]),
        _m("FCN (ResNet-50 backbone)", ["long2015fcn", "he2016resnet"], "usbg.plug.masks.TorchvisionMaskSource",
           ["mask/fcn_r50", "mask/fcn_r50_imnet"], ["torchvision"]),
        _m("nnU-Net 2D", ["isensee2021nnunet"], "usbg.plug.masks.NnUNetMaskSource",
           ["mask/nnunet2d", "mask/nnunet2d_e20"], ["isensee2021nnunet"]),
        _m("SAM with box prompts", ["kirillov2023sam"], "usbg.plug.masks.SamBoxMaskSource",
           ["mask/sam_unet", "mask/sam_priv"], ["kirillov2023sam"]),
        _m("MedSAM with box prompts", ["ma2024medsam"], "usbg.plug.masks.SamBoxMaskSource (variant medsam)",
           ["mask/medsam_unet", "mask/medsam_priv"], ["ma2024medsam"]),
        _m("SAMUS with box prompts", ["lin2024samus"], "usbg.plug.samus_mask.SamusMaskSource",
           ["mask/samus_unet", "mask/samus_unet@s0-2", "mask/samus_priv", "mask/samus_priv@s0-2"],
           ["lin2024samus"])],
    "ctrl": [
        _m("PPO (Stable-Baselines3)", ["schulman2017ppo", "sb3"], "usbg.plug.controllers.SB3Controller "
           "(scripts/plug_train_sb3.py --algo ppo)", ["ctrl/ppo_res", "ctrl/ppo_scratch"], ["sb3"]),
        _m("SAC (Stable-Baselines3)", ["haarnoja2018sac", "sb3"], "usbg.plug.controllers.SB3Controller "
           "(--algo sac)", ["ctrl/sac_res"], ["sb3"]),
        _m("TD3 (Stable-Baselines3)", ["fujimoto2018td3", "sb3"], "usbg.plug.controllers.SB3Controller "
           "(--algo td3)", ["ctrl/td3_res"], ["sb3"]),
        _m("A2C (Stable-Baselines3)", ["mnih2016a3c", "sb3"], "usbg.plug.controllers.SB3Controller (--algo a2c)",
           ["ctrl/a2c_res"], ["sb3"]),
        _m("DQN with 13 discrete moves (Stable-Baselines3)", ["mnih2015dqn", "sb3"],
           "usbg.plug.controllers.SB3Controller (--algo dqn)", ["ctrl/dqn_scratch"], ["sb3"]),
        _m("Behavior cloning (BC)", ["pomerleau1991"], "usbg.plug.controllers.GoalNetController (--algo bc)",
           ["ctrl/bc_res", "ctrl/bc_scratch", "ctrl/bc_res@bn_train", "ctrl/bc_scratch@bn_train"], license=R_NOTE),
        _m("DAgger", ["ross2011dagger"], "usbg.plug.controllers.GoalNetController (--algo dagger)",
           ["ctrl/dagger_res", "ctrl/dagger_scratch", "ctrl/dagger_res@bn_train", "ctrl/dagger_scratch@bn_train"],
           license=R_NOTE),
        _m("Cross-entropy method (CEM) look-ahead", ["rubinstein1999cem"], "usbg.plug.controllers.CEMController",
           ["ctrl/cem"], license=R_NOTE_NO_WEIGHTS)],
    "reward": [
        _m("SSIM appearance term", ["wang2004ssim"], "usbg.plug.rewards.RewardPlug('ssim')", ["reward/ssim"],
           ["wang2004ssim"]),
        _m("NMI appearance term", ["studholme1999nmi"], "usbg.plug.rewards.RewardPlug('nmi')", ["reward/nmi"],
           license=R_NOTE_NO_WEIGHTS),
        _m("LPIPS appearance term", ["zhang2018lpips"], "usbg.plug.rewards.RewardPlug('lpips')", ["reward/lpips"],
           ["zhang2018lpips"]),
        _m("Change in pose distance", ["sonogym", "hase2020"], "usbg.plug.rewards.RewardPlug('pose')",
           ["reward/pose"], license=R_NOTE_NO_WEIGHTS),
        _m("Pose distance plus organ area", ["bi2022vesnet"], "usbg.plug.rewards.RewardPlug('pose_area')",
           ["reward/pose_area"], license=R_NOTE_NO_WEIGHTS)],
}

REFERENCE_COMPONENTS = [
    {"slot": "anatomy", "name": "TotalSegmentator dataset masks (13 training and 6 held-out CTs)", "cites": ["totalseg"],
     "adapter": "usbg.volume, scripts/build_anat_cache.py", "license": "dataset under CC BY 4.0"},
    {"slot": "renderer", "name": "Physics renderer (authors')", "cites": [], "adapter": "usbg.render_lotus",
     "license": "this repository (Apache-2.0)", "reference": "USB Gym (this repository; CITATION.cff)"},
    {"slot": "texture", "name": "SonoSPADE (authors')", "cites": ["sonospade"], "adapter": "usbg.texture_gan",
     "license": "code in this repository (Apache-2.0); the stage checkpoint is not released"},
    {"slot": "texture", "name": "CUT (prior stage, secondary)", "cites": ["cut"], "adapter": "usbg.texture_gan",
     "license": "the authors' implementation in this repository (Apache-2.0)"},
    {"slot": "texture", "name": "CycleGAN (prior stage, secondary)", "cites": ["cyclegan"], "adapter": "usbg.texture_gan",
     "license": "the authors' implementation in this repository (Apache-2.0)"},
    {"slot": "mask", "name": "U-Net of the same stage and seed (authors')", "cites": ["ronneberger2015unet"],
     "adapter": "usbg.us_segmenter, segtrain/", "license": "code in this repository (Apache-2.0); the checkpoints "
                                                           "inherit the KidneyUS terms and are not released"},
    {"slot": "ctrl", "name": "REINFORCE residual over the servo base, imitation start (authors')", "cites": ["reinforce"],
     "adapter": "scripts/train_view_match_rl.py", "license": "this repository (Apache-2.0)"},
    {"slot": "reward", "name": "Match score S (authors')", "cites": [], "adapter": "usbg.view_match",
     "license": "this repository (Apache-2.0)", "reference": "USB Gym (this repository; CITATION.cff)"},
]

# arXiv works cited from arXiv: their DataCite DOI (10.48550/arXiv.<id>)
ARXIV_DOI = {"chen2017deeplabv3": "10.48550/arXiv.1706.05587", "schulman2017ppo": "10.48550/arXiv.1707.06347"}

# license notes parsed from the article that name more than one component: the part that applies to a slot's rows
# (the torchvision note also names the AlexNet trunk of LPIPS, which only the reward slot uses)
NOTE_TRIM = {"mask": ("supplies the AlexNet trunk of LPIPS and the ImageNet-1K backbones", "supplies the ImageNet-1K "
                                                                                          "backbones")}

__all__ = ["PUBLISHED", "REFERENCE_COMPONENTS", "SLOT_OF_COMPONENT", "ARXIV_DOI", "NOTE_TRIM", "R_NOTE",
           "R_NOTE_NO_WEIGHTS", "R_NOTE_HISTMATCH", "LOTUS_NOTE"]
