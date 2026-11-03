"""v2.1 pluggability (exploratory): unit tests of usbg.plug.masks (CPU, synthetic data, no patient data).

The CT-label tap is exercised in a subprocess so it never instruments view_match inside this pytest
process."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from usbg.plug import masks as M

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parse_spec():
    s = M.parse_spec("kind=sam,weights=/x/y.pth,prompt=ct")
    assert s == {"kind": "sam", "weights": "/x/y.pth", "prompt": "ct"}
    with pytest.raises(ValueError):
        M.parse_spec("weights=/x")
    with pytest.raises(ValueError):
        M.parse_spec("kind=tv,oops")


def test_label_maps():
    six = np.arange(6)
    assert M.to_repo(six).tolist() == [0, 1, 2, 3, 4, 7]
    canon = np.arange(13)
    assert M.canon_to_six(canon).tolist() == [0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 0, 0, 0]
    assert M.canon_to_six_repo(canon).tolist() == [0, 1, 2, 3, 4, 0, 0, 7, 0, 0, 0, 0, 0]


def test_boxes_from_mask():
    m = np.zeros((64, 64), np.int64)
    m[5:15, 5:25] = 1                    # liver component, 200 px
    m[40:44, 40:44] = 3                  # vessel, 16 px < min_px 20
    m[30:36, 2:8] = 3                    # vessel, 36 px
    m[50:60, 50:60] = 7                  # spleen
    m[0:3, 60:63] = 5                    # GI (not a match organ)
    b = M.boxes_from_mask(m)
    assert (1, [5, 5, 24, 14]) in b
    assert (3, [2, 30, 7, 35]) in b
    assert (7, [50, 50, 59, 59]) in b
    assert all(o in (1, 2, 3, 4, 7) for o, _ in b) and len(b) == 3
    m2 = m.copy(); m2[40:44, 40:46] = 3   # second vessel component now 24 px
    assert sum(o == 3 for o, _ in M.boxes_from_mask(m2)) == 2
    assert sum(o == 3 for o, _ in M.boxes_from_mask(m2, max_boxes=1)) == 1
    assert sum(o == 3 for o, _ in M.boxes_from_mask(m2, mode="organ")) == 1


def test_tv_segmenter_roundtrip(tmp_path):
    pytest.importorskip("torchvision")
    s = M.TVSegmenter(device="cpu", n_classes=6, arch="lraspp_mbv3", seed=0)
    x = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    p0 = s.predict(x)
    ck = str(tmp_path / "tv.pt")
    s.save(ck)
    src = M.TorchvisionMaskSource(ck, device="cpu")
    assert np.array_equal(src.segment6(x), p0)
    assert set(np.unique(src.segment(x)).tolist()) <= {0, 1, 2, 3, 4, 7}
    assert src.timing()["n_calls"] == 2 and src.describe()["privileged"] is False
    assert src.ckpt_path == os.path.abspath(ck)


_TAP_SCRIPT = r"""
import json, numpy as np
from usbg import view_match as VM
from usbg.plug import masks as M
def fake_render(lv, T, Wfv, translator, cfg=None, size=128, cond_label=None):
    img = np.full((8, 8), 0.5, np.float32); lab = np.zeros((8, 8), np.int64); lab[2:5, 2:5] = int(T)
    img_sec, lab_sec = VM.warp_to_sector(img, lab)
    if cond_label is not None:
        VM.warp_to_sector(img, lab * 0 + 9)              # a second (label-control) warp must be ignored
    tex = img_sec
    out = VM._US.segment(tex) if VM._US is not None else lab_sec
    return tex, out
VM.curvi_render = fake_render
VM.warp_to_sector = lambda img, label, p=None: (img.copy(), label.copy())
class Probe:
    def segment(self, img):
        return M.current_ct_label(img).copy()
M.install_ct_label_tap(); M.install_ct_label_tap()        # idempotent
VM.set_mask_source(Probe())
res = {}
img, lab = VM.curvi_render(None, 4, None, None)
res["inside"] = int(lab.max())
img, lab = VM.curvi_render(None, 7, None, None, cond_label=True)
res["first_warp"] = int(lab.max())
VM.set_mask_source(None)
img, lab = VM.curvi_render(None, 3, None, None)
res["after_identity"] = int(M.current_ct_label(img).max())
try:
    M.current_ct_label(img.copy()); res["copy_raises"] = False
except RuntimeError:
    res["copy_raises"] = True
print(json.dumps(res))
"""


def test_ct_label_tap_subprocess():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"), USBG_LEARNED_SEG="0")
    out = subprocess.run([sys.executable, "-c", _TAP_SCRIPT], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"inside": 4, "first_warp": 7, "after_identity": 3, "copy_raises": True}
