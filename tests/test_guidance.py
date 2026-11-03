"""Tests for the prescriptive directional guidance (twist -> instruction)."""
from __future__ import annotations

import numpy as np

from usbg.guidance import describe_twist


def test_directions_and_units():
    assert describe_twist([0, 0, 0, 0, 0, 0.40], "rotate", 2.0) == "rotate clockwise ~46 deg"
    assert describe_twist([0, 0, 0, 0, 0, -0.40], "rotate", 2.0) == "rotate counter-clockwise ~46 deg"
    assert describe_twist([-0.02, 0, 0, 0, 0, 0], "slide", 2.0) == "slide left ~4 cm"
    assert describe_twist([0.02, 0, 0, 0, 0, 0], "slide", 2.0) == "slide right ~4 cm"
    assert describe_twist([0, 0, 0, 0, 0.20, 0], "rock", 2.0) == "rock forward ~23 deg"
    assert describe_twist([0, 0, 0, -0.26, 0, 0], "tilt", 2.0) == "fan left ~30 deg"
    assert describe_twist([0, 0, 0.010, 0, 0, 0], "compress", 2.0) == "press in ~2 cm"
    assert describe_twist([0, 0, -0.010, 0, 0, 0], "compress", 2.0) == "release ~2 cm"


def test_hold_and_horizon_scaling():
    assert describe_twist([0, 0, 0, 0, 0, 0.4], "hold", 2.0) == "hold steady"
    # magnitude scales with the horizon
    a = describe_twist([0, 0, 0, 0, 0, 0.4], "rotate", 1.0)
    b = describe_twist([0, 0, 0, 0, 0, 0.4], "rotate", 2.0)
    assert "23 deg" in a and "46 deg" in b


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
