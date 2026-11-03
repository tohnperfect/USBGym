"""usbg.leaderboard: the plug-slot leaderboard of USB Gym.

The leaderboard holds every row of Table 5 of the USB Gym article (the published methods in the plug slots) with
the per-episode values behind it, and the rows of methods added since. A new method is run under the unchanged
plug protocol (usbg.leaderboard.protocol) by ``python -m usbg.leaderboard run``, which writes a submission.json;
``validate`` checks it and recomputes its summary; the maintainers ``add`` a validated submission and ``render``
LEADERBOARD.md. See SUBMITTING.md at the repository root.

    python -m usbg.leaderboard {check,run,validate,render,add,recompute,fetch,template} --help
"""
from __future__ import annotations

from .protocol import PROTOCOL_VERSION, SUBMISSION_SCHEMA  # noqa: F401

__all__ = ["PROTOCOL_VERSION", "SUBMISSION_SCHEMA"]
