"""S_US segmenter tests: U-Net shapes, label alignment, Dice / CE+Dice loss, and that a tiny net
overfits a learnable set to Dice > 0.8 (the mechanism; full pretraining is in train_segmenter.py)."""
from __future__ import annotations

import numpy as np
import pytest

from usbg import segmenter as SEG


def test_unet_shapes():
    import torch
    from usbg.render_lotus import get_device
    net = SEG.build_segmenter(n_classes=13, base=8, depth=3).to(get_device())
    x = torch.rand(2, 1, 64, 64, device=get_device())
    assert net(x).shape == (2, 13, 64, 64)


def test_align_label_to_image_transpose_and_resize():
    # slicer label is (N_LINES, N_SAMPLES); the render is its transpose, so alignment transposes.
    lab = np.zeros((160, 512), np.int64)
    lab[:, :256] = 3                              # a depth band in the label (cols = depth)
    al = SEG.align_label_to_image(lab, (128, 128))
    assert al.shape == (128, 128) and al.dtype == np.int64
    # after transpose, the depth band is in the ROWS (top half), so rows carry the value 3
    assert (al[:64] == 3).mean() > 0.9 and (al[64:] == 3).mean() < 0.1


def test_dice_and_loss():
    import torch
    logits = torch.zeros(1, 13, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4] = 1
    logits[0, 1, :4] = 10.0                       # perfect on class 1
    logits[0, 0, 4:] = 10.0                       # perfect on class 0 (background, ignored)
    assert SEG.dice_score(logits, target, n_classes=13) > 0.99
    loss = SEG.seg_ce_dice_loss(logits, target, n_classes=13)
    assert float(loss) >= 0.0 and np.isfinite(float(loss))


def test_segmenter_overfits_learnable_set():
    """Mechanism: perfect labels make a small set easy to fit (Dice > 0.8)."""
    import torch
    from usbg.render_lotus import get_device
    dev = get_device()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H = 48
    imgs, labs = [], []
    for _ in range(4):
        lab = np.zeros((H, H), np.int64)
        lab[:H // 2] = 1; lab[H // 2:, :H // 2] = 4; lab[H // 2:, H // 2:] = 7
        img = lab.astype(np.float32) / 12.0 + rng.normal(0, 0.03, (H, H)).astype(np.float32)
        imgs.append(img); labs.append(lab)
    X = torch.from_numpy(np.stack(imgs)[:, None]).to(dev)
    Y = torch.from_numpy(np.stack(labs)).to(dev)
    net = SEG.build_segmenter(13, base=8, depth=2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(150):
        opt.zero_grad()
        SEG.seg_ce_dice_loss(net(X), Y, n_classes=13).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        assert SEG.dice_score(net(X), Y, n_classes=13) > 0.8


def test_grouped_scheme():
    from usbg.volume import LIVER, FAT, GI, PANCREAS, MUSCLE, SOFT_TISSUE, BONE, N_LABELS
    gmap = SEG.grouped_map_array()
    assert gmap.shape == (N_LABELS,)
    # the degenerate soft tissues all collapse to one class
    soft = {int(gmap[c]) for c in (FAT, GI, PANCREAS, MUSCLE, SOFT_TISSUE)}
    assert len(soft) == 1
    # a distinct tissue keeps its own class, different from the soft group and from bone
    assert gmap[LIVER] != next(iter(soft)) and gmap[BONE] != next(iter(soft))
    assert int(gmap.max()) == SEG.N_GROUPED - 1
    # remap collapses a label field into the grouped ids
    lab = np.array([[LIVER, FAT], [GI, BONE]])
    rem = SEG.remap_labels(lab, gmap)
    assert rem[0, 1] == rem[1, 0]                     # fat and gi -> same grouped id
    assert rem.max() < SEG.N_GROUPED


def test_grouped_segmenter_save_load_carries_map(tmp_path):
    from usbg.render_lotus import get_device
    seg = SEG.Segmenter(device=get_device(), n_classes=SEG.N_GROUPED, base=8, depth=2,
                        group_map=SEG.grouped_map_array())
    p = str(tmp_path / "grp.pt")
    seg.save(p)
    seg2 = SEG.Segmenter.load(p)
    assert seg2.n_classes == SEG.N_GROUPED and seg2.group_map is not None
    # remap on the loaded segmenter matches the scheme
    from usbg.volume import FAT, GI
    assert seg2.remap(np.array([[FAT]]))[0, 0] == seg2.remap(np.array([[GI]]))[0, 0]


def test_segmenter_save_load(tmp_path):
    from usbg.render_lotus import get_device
    seg = SEG.Segmenter(device=get_device(), base=8, depth=2)
    p = str(tmp_path / "s.pt")
    seg.save(p)
    seg2 = SEG.Segmenter.load(p)
    img = np.clip(np.random.default_rng(0).random((64, 64)).astype(np.float32), 0, 1)
    pred = seg2.predict(img)
    assert pred.shape == (64, 64) and pred.dtype == np.int64
    assert 0 <= pred.min() and pred.max() < 13


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
