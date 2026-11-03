"""v2.1 label-information controls (usbg.label_controls, view_match.curvi_render(cond_label=)).

The hook is opt-in: the default render path must be bit-identical, and render_parts must reproduce
curvi_render exactly, with and without a control. Runs on the synthetic phantom (canonical seat, CPU).
"""
import numpy as np
import pytest

from usbg import label_controls as LC
from usbg import volume as V


@pytest.fixture(scope="module")
def scene():
    from usbg import goal_rl as G, render_lotus as R
    lv = V.synthetic_phantom()
    Wfv, T_tgt, _T_start = G.sample_setup(lv, V.GALLBLADDER, np.random.default_rng(0))
    return lv, Wfv, T_tgt, R.RenderConfig(seed=3)


class _LabelEcho:
    """A wants_label stub whose output depends on the label it is handed."""
    wants_label = True

    def translate_aligned(self, img, lab):
        return np.clip(img * 0.5 + (np.asarray(lab) == V.GALLBLADDER) * 0.4, 0, 1).astype(np.float32)


def test_default_path_unchanged_and_parts_match(scene, monkeypatch):
    from usbg import view_match as VM
    monkeypatch.setattr(VM, "_US", None)
    lv, Wfv, T, cfg = scene
    tr = _LabelEcho()
    a = VM.curvi_render(lv, T, Wfv, tr, cfg, 64)
    b = VM.curvi_render(lv, T, Wfv, tr, cfg, 64, cond_label=None)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for spec in ("none", "elev3", "elev-6", "permute"):
        c = LC.parse_control(spec)
        img, lab = VM.curvi_render(lv, T, Wfv, tr, cfg, 64, cond_label=None if c.is_identity else c)
        p = LC.render_parts(lv, T, Wfv, tr, cfg, 64, control=c)
        assert np.array_equal(img, p["tex"]) and np.array_equal(lab, p["lab_sec"])   # returned label stays true
    # a single-input stage ignores the control
    plain = VM.curvi_render(lv, T, Wfv, None, cfg, 64)
    ctrl = VM.curvi_render(lv, T, Wfv, None, cfg, 64, cond_label=LC.parse_control("elev6"))
    assert all(np.array_equal(x, y) for x, y in zip(plain, ctrl))


def test_controls_change_only_the_conditioning_label(scene):
    lv, Wfv, T, cfg = scene
    tr = _LabelEcho()
    p0 = LC.render_parts(lv, T, Wfv, tr, cfg, 64)
    p6 = LC.render_parts(lv, T, Wfv, tr, cfg, 64, control=LC.parse_control("elev6"))
    assert np.array_equal(p0["img_sec"], p6["img_sec"]) and np.array_equal(p0["lab_sec"], p6["lab_sec"])
    assert not np.array_equal(p0["cond_sec"], p6["cond_sec"])


def test_permutation_is_a_derangement_of_the_match_organs():
    from usbg.view_match import MATCH_ORGANS
    assert set(LC.PERMUTATION) == set(MATCH_ORGANS) == set(LC.PERMUTATION.values())
    assert all(k != v for k, v in LC.PERMUTATION.items())
    lab = np.array([[0, V.LIVER, V.KIDNEY, V.SPLEEN, V.GALLBLADDER, V.VESSEL, V.BONE]], np.uint8)
    out = LC.parse_control("permute")(None, None, None, lab)
    assert out.tolist() == [[0, V.KIDNEY, V.SPLEEN, V.LIVER, V.VESSEL, V.GALLBLADDER, V.BONE]]


def test_painted_translator_identity_lines_and_loader(tmp_path):
    import torch
    from usbg.texture_gan import load_translator
    img = np.full((32, 32), 0.5, np.float32)
    lab = np.full((32, 32), V.SOFT_TISSUE, np.int64)
    lab[8:20, 8:20] = V.KIDNEY
    ident = LC.LabelPaintedTranslator({"label_gains": {}, "capsule_db": 0.0})
    assert np.allclose(ident.translate_aligned(img, lab), img, atol=1e-6)
    tr = LC.LabelPaintedTranslator({"label_gains": {int(V.KIDNEY): 0.8}, "capsule_db": 12.0})
    out = tr.translate_aligned(img, lab)
    edge = LC.capsule_edges(lab)
    assert edge.sum() > 0 and not edge[0].any()                     # lines only at the kidney boundary
    assert (out[edge] > img[edge]).all()                             # echogenic line
    inner = np.zeros_like(edge); inner[10:18, 10:18] = True
    assert np.allclose(out[inner], 0.4, atol=1e-5)                   # gain 0.8 inside the organ
    assert np.allclose(out[lab == V.SOFT_TISSUE], 0.5, atol=1e-6)    # other labels unchanged
    ck = tmp_path / "painted.pt"
    torch.save({"arch": LC.PAINT_ARCH, "params": tr.params}, ck)
    tr2 = load_translator(str(ck), device="cpu")
    assert isinstance(tr2, LC.LabelPaintedTranslator) and tr2.wants_label
    assert np.array_equal(tr2.translate_aligned(img, lab), out)


def test_dice_definitions():
    ref = np.zeros((1, 4, 4), np.int64); ref[0, :2, :2] = V.LIVER
    pred = np.zeros((1, 4, 4), np.int64); pred[0, :2, :1] = V.LIVER
    acc = LC.DiceAccumulator()
    acc.add(LC.to_six(pred[0]), LC.to_six(ref[0]))
    r = acc.result()
    assert r["support"] == {"liver": 1} and abs(r["mean_fg"] - 2 * 2 / (2 + 4)) < 1e-4
    assert LC.to_six(np.array([V.SPLEEN, V.PANCREAS, V.KIDNEY])).tolist() == [5, 0, 4]
