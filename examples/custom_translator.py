"""A custom sim-to-real texture translator: any callable ``img01 -> img01`` plugs straight into the
render loop with no library edit (the edit-free open slot).

Run:  python examples/custom_translator.py
"""
import numpy as np

from usbg.render_lotus import render


def gamma_translator(img01):
    """A trivial appearance tweak. Real translators are trained GANs (CUT, SonoSPADE); this shows the
    interface, not a good translator."""
    return np.clip(np.asarray(img01, np.float32) ** 1.5, 0.0, 1.0)


# For a PER-TISSUE (dual-input) translator, mark it and accept the label slice:
#   gamma_translator.wants_label = True
#   def gamma_translator(img01, label_slice): ...


if __name__ == "__main__":
    lab = np.zeros((128, 128), np.int64)
    lab[40:90, 30:100] = 3                        # a soft-tissue block
    out = render(lab, "physics", texture=gamma_translator)   # plugged into the loop, no lib edit
    print("rendered + translated:", out.shape, out.dtype, "range",
          round(float(out.min()), 3), round(float(out.max()), 3))
