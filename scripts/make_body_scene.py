#!/usr/bin/env python3
"""Write a MuJoCo scene of the probe resting on a human body at a screened acoustic window.

The body is the patient's own CT skin ("ct_skin") or the packaged generic torso fitted to that skin ("avatar"); see
usbg.body_scene. The CT is seated as in the gym (placement.seat_volume_under_probe), so the probe images the same
view as the gym's episode at that window.

  python scripts/make_body_scene.py --case s0011 --organ liver --window 0 --surface both --png
  python -m mujoco.viewer --mjcf=outputs/body_scene/s0011_liver_w0_ct_skin.xml

Needs the label cache of the case (README, Data preparation) and the body extra (pip install -e ".[body]"). Writes,
per surface, <out>/<case>_<organ>_w<window>_<surface>.xml (self-contained, inline meshes), the matching _seat.json
(world_from_volume, world_from_body, probe start, start lift, signed gap of the window face to the body surface,
slab count, fit residual and scale for the avatar) and with --png an image from the scene's "overview" camera.
In the viewer the collision slabs are geom group 3 (hidden until that group is switched on).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ORGANS = ("liver", "kidney", "spleen")


def render_png(xml, path, camera="overview"):
    """Offscreen image of a scene from one of its cameras (1280 x 960)."""
    import mujoco
    from PIL import Image
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=960, width=1280) as r:
        r.update_scene(data, camera=camera)
        Image.fromarray(r.render()).save(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="case id with a label cache in data/cache (for example s0011)")
    ap.add_argument("--organ", default="liver", choices=ORGANS, help="target organ of the screened window")
    ap.add_argument("--window", type=int, default=0, help="screened window index (modulo the window count)")
    ap.add_argument("--surface", default="ct_skin", choices=("ct_skin", "avatar", "both"))
    ap.add_argument("--avatar-mesh", default=None, help="another torso mesh for the avatar (metres, +y anterior, "
                                                        "+z superior); default: the packaged one")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "body_scene"))
    ap.add_argument("--no-visual", action="store_true", help="collision slabs only (no drawn body mesh)")
    ap.add_argument("--png", action="store_true", help="also save an image from the overview camera")
    args = ap.parse_args(argv)

    from usbg import body_scene as BS
    from usbg import placement as P
    from usbg import volume as V
    from usbg.seatmode import seat_mode
    if seat_mode() != "anatomical":
        raise SystemExit("make_body_scene needs USBG_SEAT=anatomical (the default): the windows are anatomical")
    lv = V.load_cache(os.path.join(args.cache_dir, f"{args.case}.npz"))
    lv.meta.setdefault("case", args.case)
    label = {"liver": V.LIVER, "kidney": V.KIDNEY, "spleen": V.SPLEEN}[args.organ]
    n = P.n_windows(lv, label)
    if n == 0:
        raise SystemExit(f"{args.case} has no screened {args.organ} window")
    window = args.window % n
    Wfv, T_target = P.seat_volume_under_probe(lv, label, window=window)
    os.makedirs(args.out, exist_ok=True)
    kinds = ("ct_skin", "avatar") if args.surface == "both" else (args.surface,)
    for kind in kinds:
        xml, rec = BS.scene_xml(lv, Wfv, T_target, kind, visual=not args.no_visual,
                                mesh=args.avatar_mesh if kind == "avatar" else None)
        stem = os.path.join(args.out, f"{args.case}_{args.organ}_w{window}_{kind}")
        with open(stem + ".xml", "w") as fh:
            fh.write(xml)
        rec = {"case": args.case, "organ": args.organ, "window": window, "n_windows": n, **rec}
        with open(stem + "_seat.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        msg = (f"{kind}: {stem}.xml  slabs {rec['n_slabs']}, start lift {rec['start_lift_mm']:.1f} mm, "
               f"window face {rec['window_gap_mm']:+.1f} mm from the body surface")
        if kind == "avatar":
            msg += f", fit residual {rec['info']['reg_rms_mm']:.1f} mm at scale {rec['info']['reg_scale']:.3f}"
        print(msg)
        if args.png:
            print(f"  image: {render_png(xml, stem + '.png')}")
        print(f"  view:  python -m mujoco.viewer --mjcf={stem}.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
