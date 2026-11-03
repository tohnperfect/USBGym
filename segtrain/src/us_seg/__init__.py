"""us_seg: cross-domain (sim/real) abdominal ultrasound organ segmenter for USB Gym.

Trains one MPS-native 2D U-Net on SonoSPADE-textured simulated B-mode (labeled) and evaluates
zero-shot on real US, so it can serve as the single, self-consistent mask source everywhere in the
USB Gym view-matching controller. See segtrain/README.md.
"""
__version__ = "0.1.0"
