"""guidance.py: turn a predicted 6-DOF twist into a structured prescriptive move for AI guidance.

A prescriptive move is (maneuver, direction, amplitude): which probe manipulation, which way, and
how much. Each maneuver has a dominant twist channel (slide->vx, rock->wy, tilt/fan->wx,
rotate->wz, compress->vz); the sign of that channel gives the direction and the rate integrated
over a horizon gives the amplitude, so an image-conditioned policy's twist (for the current move)
or the anticipation head's predicted next twist (for the next move) becomes a human instruction like
"rotate clockwise ~45 deg", "slide left ~4 cm", "rock forward ~20 deg", "fan left ~30 deg", or
"press in ~2 cm". Directions follow the probe-frame convention (x lateral, y elevation, z axial with
+z into tissue); they are a fixed convention, not a clinical laterality claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# maneuver -> (twist channel index, kind, positive word, negative word, verb)
_SPEC = {
    "slide":    (0, "lin", "right", "left", "slide"),            # vx
    "compress": (2, "lin", "in", "out", "press"),               # vz (into tissue)
    "tilt":     (3, "ang", "right", "left", "fan"),             # wx (tilt == fan)
    "rock":     (4, "ang", "forward", "back", "rock"),          # wy
    "rotate":   (5, "ang", "clockwise", "counter-clockwise", "rotate"),  # wz
}


@dataclass
class Move:
    """A structured prescriptive move: which manipulation, which direction, how much."""
    maneuver: str
    direction: str
    amplitude: float           # magnitude in `unit`
    unit: str                  # "deg" or "cm" ("" for hold)
    phrase: str                # human-readable instruction

    def as_dict(self):
        return {"maneuver": self.maneuver, "direction": self.direction,
                "amplitude": round(float(self.amplitude), 1), "unit": self.unit, "phrase": self.phrase}


def move_from_twist(twist, maneuver: str, horizon_s: float = 2.0) -> Move:
    """Structured move for `maneuver` from a body twist [vx,vy,vz,wx,wy,wz] (SI) over a horizon (s).
    Amplitude is |rate| * horizon in cm (linear) or degrees (angular)."""
    if maneuver == "hold" or maneuver not in _SPEC:
        return Move("hold", "steady", 0.0, "", "hold steady")
    ci, kind, pos, neg, verb = _SPEC[maneuver]
    rate = float(np.asarray(twist).reshape(-1)[ci])
    direction = pos if rate >= 0 else neg
    if kind == "ang":
        amp, unit = abs(rate) * horizon_s * 180.0 / np.pi, "deg"
    else:
        amp, unit = abs(rate) * horizon_s * 100.0, "cm"
    if maneuver == "compress":                                   # phrase press in / release
        phrase = f"press {direction} ~{amp:.0f} {unit}" if rate >= 0 else f"release ~{amp:.0f} {unit}"
        direction = direction if rate >= 0 else "out"
    else:
        phrase = f"{verb} {direction} ~{amp:.0f} {unit}"
    return Move(maneuver, direction, amp, unit, phrase)


def describe_twist(twist, maneuver: str, horizon_s: float = 2.0) -> str:
    """Back-compatible string form of the prescriptive move (see move_from_twist)."""
    return move_from_twist(twist, maneuver, horizon_s).phrase
