"""v2.1 pluggability (exploratory): unit tests of usbg.plug.samus_mask (CPU, synthetic data, no patient data).
The model tests are skipped when the SAMUS clone or weights are absent."""
import os
import sys

import numpy as np
import pytest

from usbg.plug import masks as M
from usbg.plug import samus_mask as SM

HAVE = os.path.isfile(SM.DEFAULT_WEIGHTS) and os.path.isdir(SM.DEFAULT_CODE)


def _scene():
    m = np.zeros((128, 128), np.int64)
    m[20:70, 10:90] = 1
    m[80:110, 60:100] = 4
    m[30:36, 100:106] = 3
    m[0:3, 0:3] = 5                       # GI (not a match organ)
    img = np.full((128, 128), 0.3, np.float32)
    img[m == 1] = 0.7
    img[m == 4] = 0.15
    return img, m


def test_components_match_boxes_from_mask():
    _, m = _scene()
    m[40:44, 40:46] = 3
    for kw in ({}, {"max_boxes": 1}, {"mode": "organ"}):
        comps = SM.components_from_mask(m, **kw)
        assert [(o, SM.fixed_bbox(c)) for o, c in comps] == [(o, b) for o, b in M.boxes_from_mask(m, **kw)]


def test_samus_prompt_functions():
    c = np.zeros((10, 10), bool)
    c[2:5, 3:8] = True
    assert SM.fixed_bbox(c) == [3, 2, 7, 4]
    idx = np.argwhere(c)
    assert SM.fixed_click(c) == [int(idx[len(idx) // 2][1]), int(idx[len(idx) // 2][0])]
    assert c[SM.fixed_click(c)[1], SM.fixed_click(c)[0]]
    assert SM.fixed_bbox(np.zeros((4, 4), bool)) is None and SM.fixed_click(np.zeros((4, 4), bool)) is None


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SM.load_samus_source({"kind": "samus", "weights": "/x", "oops": "1"})


@pytest.mark.skipif(not HAVE, reason="SAMUS clone or weights not present")
@pytest.mark.parametrize("ptype", ["box", "point"])
def test_samus_gt_prompts_cpu(ptype):
    img, m = _scene()
    src = M.load_mask_source(f"kind=samus,weights={SM.DEFAULT_WEIGHTS},prompt=gt,ptype={ptype},device=cpu")
    assert src.privileged and src.kind == "samus" and src.ckpt_path == os.path.abspath(SM.DEFAULT_WEIGHTS)
    src.register_prompts(img[None], M.canon_to_six(m)[None])
    out = src.segment(img)
    assert out.shape == m.shape and set(np.unique(out).tolist()) <= {0, 1, 2, 3, 4, 7}
    assert np.array_equal(M.to_repo(src.segment6(img)), out)
    p = src.probs(img, SM.components_from_mask(m))
    assert p.shape == (3, 128, 128) and np.isfinite(p).all()
    d = src.describe()
    assert d["code"]["commit_ok"] and d["prompt_type"] == ptype and d["privileged"] is True
    for mod in ("tkinter", "turtle", "einops"):          # import-time stubs never stay in sys.modules
        assert getattr(sys.modules.get(mod), "__doc__", None) != "usbg.plug.samus_mask import-time stub"
