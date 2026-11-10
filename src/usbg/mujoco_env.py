"""mujoco_env.py: a probe pressed against a patient surface, rendering the B-mode it sees.

A minimal MuJoCo (CPU) environment for Phase 3. The probe is a dynamic body welded to a mocap
target (the standard teleoperation pattern): we drive the mocap each step with a 6-DOF twist
action, the weld pulls the probe there, and contact with the patient surface pushes back so the
normal force is a real, sane signal (a purely kinematic mocap probe against a static box would
report zero force, since neither body has DOFs). The observation is the LOTUS B-mode rendered
at the probe's ACTUAL pose (which lags the target when pressing), so image and force are
consistent.

Frames: the patient skin is a box whose top face is at world z = 0; tissue is below. The probe
starts just above the skin, imaging axis (+z probe) pointing down into the patient. The CT is
seated under the probe by placement.seat_volume_under_probe (axis-aligned start).

v2.1.2 (opt-in): USProbeEnv(contact_plane=(p, n)) re-poses that box after compiling the
unchanged model XML: its top face then lies in the plane of point p and normal n, and the press cap acts along n
(placement.clamp_start_outside_plane). Without the argument nothing changes.

v2.3.0 (opt-in): USProbeEnv(body=...) replaces the box by a human body surface (body_scene.place: the patient's CT
skin or the packaged avatar), collided through convex slabs; the press cap then acts along the normal of the slab
facet nearest the probe face. Without the argument nothing changes.
"""
from __future__ import annotations

import os

import numpy as np

from . import slicer as S
from . import render_lotus as R
from .contracts_bridge import se3_exp, DT_C

CONTACT_ENV = "USBG_CONTACT"      # on (default) | off: the env-wide default for the contact switch


def contact_default() -> bool:
    """Patient-surface contact switch default: $USBG_CONTACT (on/off, default on)."""
    v = os.environ.get(CONTACT_ENV, "on").strip().lower()
    if v in ("", "1", "on", "true", "yes"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"{CONTACT_ENV}={v!r}; expected on|off")


PATIENT_SOLREF = "0.02 1"            # contact softness of the patient surface (box or body slabs)
PATIENT_SOLIMP = "0.8 0.9 0.001"


def _mjcf(patient_half=(0.12, 0.15, 0.08), probe_half=(0.008, 0.008, 0.012),
          weld_solref=(0.02, 1.0), assets_xml="", patient_xml=None) -> str:
    """Build the model XML. Patient box top face at z=0; probe welded to a mocap target.

    assets_xml / patient_xml (body_scene.BodySurface.mjcf): an <asset> block and the worldbody geoms that replace
    the box (geoms named "patient", "patient_1", ... are the contact surface). With both at their defaults the
    string is the box model, unchanged."""
    px, py, pz = patient_half
    bx, by, bz = probe_half
    if patient_xml is None:
        patient_xml = f"""    <!-- patient: static box, top (skin) at z=0, tissue below -->
    <geom name="patient" type="box" pos="0 0 {-pz}" size="{px} {py} {pz}"
          rgba="0.8 0.6 0.55 1" solref="{PATIENT_SOLREF}" solimp="{PATIENT_SOLIMP}"/>"""
    return f"""
<mujoco model="us_probe_gym">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <compiler autolimits="true"/>{assets_xml}
  <default>
    <geom friction="1 0.05 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 0.5" dir="0 0 -1"/>
{patient_xml}
    <!-- mocap target the probe is welded to -->
    <body name="target" mocap="true" pos="0 0 0.02">
      <geom type="box" size="0.002 0.002 0.002" contype="0" conaffinity="0"
            rgba="0 1 0 0.3"/>
    </body>
    <!-- probe: dynamic body with a free joint. The imaging face is at the body origin
         (transducer face centre, +z probe images into tissue); the housing geom sits BEHIND
         the face (local -z) so the face just touches the skin at contact, no pre-penetration. -->
    <body name="probe" pos="0 0 0.02">
      <freejoint name="probe_free"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="probe_tip" type="box" size="{bx} {by} {bz}" pos="0 0 {-bz}"
            rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="probe_weld" body1="probe" body2="target"
          solref="{weld_solref[0]} {weld_solref[1]}" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
"""


def _plane_tuple(plane):
    """(p (3,), n (3,) unit) from a (p, n) pair; refuses a zero normal."""
    p, n = plane
    p = np.asarray(p, dtype=np.float64).reshape(3).copy()
    n = np.asarray(n, dtype=np.float64).reshape(3).copy()
    nn = float(np.linalg.norm(n))
    if not np.isfinite(nn) or nn < 1e-9 or not np.all(np.isfinite(p)):
        raise ValueError("contact_plane needs a finite point and a non-zero normal")
    return p, n / nn


def plane_rotation(n):
    """Rotation whose third column is the unit normal n: the minimal rotation taking world +z onto n (Rodrigues),
    so the box keeps its in-plane orientation as close to the world axes as the tilt allows."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ n)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])                  # half turn about x
    v = np.cross(z, n)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _pose_patient_box(model, p, n):
    """Re-pose the compiled static patient box: its top face then lies in the plane of point p and outward normal
    n (top centre at p, box centre at p - n * half height). Only the box pose changes; the MJCF string is untouched."""
    import mujoco
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "patient")
    hz = float(model.geom_size[gid][2])
    model.geom_pos[gid] = np.asarray(p, np.float64) - hz * np.asarray(n, np.float64)
    model.geom_quat[gid] = _quat_from_mat(plane_rotation(n).reshape(9))
    # the compiler marked the box as sharing its body's orientation (identity quat), so the kinematics would copy
    # the body frame and ignore the new quat; clear that flag so geom_xmat follows geom_quat
    model.geom_sameframe[gid] = 0


def _mat_from_T(T):
    return np.asarray(T[:3, :3], dtype=np.float64).reshape(9)


def _quat_from_mat(mat9):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64))
    return q                                   # (w, x, y, z)


class USProbeEnv:
    """Probe-vs-patient environment. obs = LOTUS B-mode at the probe's actual pose.

    Parameters:
      lv                : volume.LabeledVolume to image.
      world_from_volume : 4x4 placing the CT in the world (see placement.seat_volume_under_probe).
      T0                : initial world_from_probe (probe above skin, imaging down).
      renderer          : "lotus" (MPS) or "numpy".
      settle_steps      : mj_step iterations per env step (let the weld and contact settle).
      contact           : patient-surface contact. None = $USBG_CONTACT (default on). False disables
                          collision on every static (world-body) geom before the first reset, so the
                          weld alone drives the probe and contact_force() is 0.
      observe           : render the B-mode observation on reset/step (default). False returns obs=None
                          (callers that render their own view, e.g. the view-match trainer, skip a
                          physics render per step; the dynamics are unchanged).
      contact_plane     : v2.1.2 (opt-in). None (default) keeps the box top at z = 0 (the
                          plane anchored at the target), as before. (p, n) = a world point and unit normal: the
                          compiled patient box is re-posed: its top face lies in the plane of point p and
                          outward normal n (top-face centre at p), and the press cap applies along n.
      body              : v2.3.0 (opt-in). None (default) keeps the box. A body_scene.BodySurface (body_scene.place)
                          replaces the box by that body's convex collision slabs, already in world coordinates;
                          the press cap applies along BodySurface.surface_plane. Cannot be combined with
                          contact_plane.
    """

    def __init__(self, lv, world_from_volume, T0, *, renderer="physics", device=None,
                 settle_steps=10, cfg=None, translator=None, randomize=False, rng=None,
                 contact=None, observe=True, max_press_m=None, contact_plane=None, body=None):
        import mujoco
        self._mj = mujoco
        if body is not None and contact_plane is not None:
            raise ValueError("body and contact_plane cannot be combined: the body is its own contact surface")
        self.body = body
        if body is None:
            self.model = mujoco.MjModel.from_xml_string(_mjcf())
        else:
            assets_xml, patient_xml = body.mjcf(visual=False)
            self.model = mujoco.MjModel.from_xml_string(_mjcf(assets_xml=assets_xml, patient_xml=patient_xml))
        self.contact_plane = None
        if contact_plane is not None:
            self.contact_plane = _plane_tuple(contact_plane)
            _pose_patient_box(self.model, *self.contact_plane)
        self.contact = contact_default() if contact is None else bool(contact)
        if not self.contact:
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) == 0:          # static scene geoms (patient skin)
                    self.model.geom_contype[gid] = 0
                    self.model.geom_conaffinity[gid] = 0
        self.observe = bool(observe)
        if max_press_m is None:
            from .seatmode import seat_mode
            if seat_mode() == "anatomical":
                max_press_m = float(os.environ.get("USBG_MAX_PRESS_MM", "5")) / 1000.0
        self.max_press_m = max_press_m            # None = unlimited (v1 canonical physics)
        self.data = mujoco.MjData(self.model)
        self.lv = lv
        self.world_from_volume = np.asarray(world_from_volume, dtype=np.float64)
        self.T0 = np.asarray(T0, dtype=np.float64)
        self.renderer = renderer
        self.device = device or (R.get_device() if renderer in ("physics", "lotus") else "cpu")
        self.settle_steps = settle_steps
        # sim-to-real texture is a same-pipeline stage: one Renderer renders every observation (and,
        # for the goal reward, the target via render_at) through the identical cfg + translator draw.
        self._base_cfg = cfg if cfg is not None else (R.RenderConfig() if renderer == "physics"
                                                      else None)
        self.translator = translator
        self.randomize = randomize
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._renderer = R.Renderer(kind=renderer, cfg=self._base_cfg, translator=translator,
                                    device=self.device)

        self._probe_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "probe")
        self._probe_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        self._patient_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "patient")
        # every geom of the contact surface: the box alone, or the slabs of a body ("patient", "patient_1", ...)
        self._patient_gids = frozenset(
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").split("_")[0] == "patient")
        self._target = None            # commanded target pose (world_from_probe)
        self.reset()

    # pose helpers
    def _press_limited(self, T):
        """Force-capped press: the COMMANDED pose may put the probe face at most max_press_m below the
        skin plane (z=0), as a force-limited robot would (about 8 N at 5 mm in this contact model).
        Tissue is rigid, so deeper commands only grow the solver's penalty force (40+ N) or, with
        contact off, drive the probe into the body. Anatomical seat only (max_press_m None there means
        $USBG_MAX_PRESS_MM, default 5 mm); the v1 canonical physics is unchanged."""
        if self.max_press_m is None:
            return T
        if self.body is not None:                          # v2.3.0 opt-in: the cap along the nearest slab facet
            return self.body.clamp(T, min_clear_m=-self.max_press_m)[0]
        if self.contact_plane is not None:                 # v2.1.2 opt-in: the cap along the plane normal
            from .placement import clamp_start_outside_plane
            p, n = self.contact_plane
            return clamp_start_outside_plane(T, p, n, min_clear_m=-self.max_press_m)[0]
        from .placement import clamp_start_outside_skin
        return clamp_start_outside_skin(T, min_clear_m=-self.max_press_m)[0]

    def _set_mocap(self, T):
        self.data.mocap_pos[0] = T[:3, 3]
        self.data.mocap_quat[0] = _quat_from_mat(_mat_from_T(T))

    def _place_probe(self, T):
        """Hard-set the free joint to T (used on reset so we start already at the pose)."""
        self.data.qpos[:3] = T[:3, 3]
        self.data.qpos[3:7] = _quat_from_mat(_mat_from_T(T))
        self.data.qvel[:] = 0.0

    @property
    def world_from_probe(self) -> np.ndarray:
        """The probe body's ACTUAL pose in the world (metres)."""
        T = np.eye(4)
        T[:3, 3] = self.data.xpos[self._probe_bid]
        T[:3, :3] = self.data.xmat[self._probe_bid].reshape(3, 3)
        return T

    # contact
    def contact_force(self) -> float:
        """Total normal contact force (N) between the probe tip and the patient surface."""
        mujoco = self._mj
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._probe_gid in (c.geom1, c.geom2) and (c.geom1 in self._patient_gids
                                                          or c.geom2 in self._patient_gids):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])                    # normal component
        return float(total)

    # observation
    @property
    def cfg_ep(self):
        """The render config in force this episode (the per-episode domain-randomization draw)."""
        return self._renderer.cfg

    def render_at(self, T_world_from_probe) -> np.ndarray:
        """Render a B-mode at an arbitrary world pose through the SAME pipeline as the observation
        (same renderer, cfg draw, and translator). Used to render the goal target consistently."""
        lab = S.slice_at_pose(self.lv, np.asarray(T_world_from_probe, dtype=np.float64),
                              self.world_from_volume)
        return self._renderer.render(lab)

    def render_obs(self, **kw) -> np.ndarray:
        return self.render_at(self.world_from_probe)

    def _maybe_obs(self):
        return self.render_obs() if self.observe else None

    def _obs_info(self):
        return {"pose": self.world_from_probe, "target": self._target.copy(),
                "contact_force": self.contact_force()}

    # gym-style API
    def reset(self, T0=None):
        self._mj.mj_resetData(self.model, self.data)
        # draw one per-episode render config (domain randomization), held fixed for the whole
        # episode so target and every observation share it (same-pipeline rule)
        if self.randomize and self.renderer == "physics":
            self._renderer.cfg = R.domain_randomize(self._base_cfg or R.RenderConfig(), self._rng)
        T = np.asarray(T0, dtype=np.float64) if T0 is not None else self.T0
        self._target = T.copy()
        self._set_mocap(T)
        self._place_probe(T)
        self._mj.mj_forward(self.model, self.data)
        return self._maybe_obs(), self._obs_info()

    def step(self, twist):
        """Advance by a body-frame 6-DOF twist [vx,vy,vz,wx,wy,wz] over one DT_C.

        Integrates the commanded target pose T <- T @ exp(twist*DT_C), drives the mocap, steps
        the physics so the weld and contact settle, then renders the B-mode at the actual pose.
        """
        twist = np.asarray(twist, dtype=np.float64).reshape(6)
        self._target = self._press_limited(self._target @ se3_exp(twist * DT_C))
        self._set_mocap(self._target)
        for _ in range(self.settle_steps):
            self._mj.mj_step(self.model, self.data)
        obs = self._maybe_obs()
        return obs, self._obs_info()

    def step_to_pose(self, T_world_from_probe):
        """Drive the probe toward an absolute world pose (used by scripted / demonstrator poses)."""
        self._target = self._press_limited(np.asarray(T_world_from_probe, dtype=np.float64).copy())
        self._set_mocap(self._target)
        for _ in range(self.settle_steps):
            self._mj.mj_step(self.model, self.data)
        return self._maybe_obs(), self._obs_info()

    def force_at_pose(self, T_world_from_probe) -> float:
        """Hard-set the probe to a pose and return the contact force, without stepping dynamics.

        The probe has a free joint (DOFs), so mj_forward's constraint solver computes a nonzero
        normal force from any penetration. Used by dataset generation to attach a QC force to a
        kinematically rendered frame at the exact commanded pose.
        """
        self._place_probe(np.asarray(T_world_from_probe, dtype=np.float64))
        self._set_mocap(np.asarray(T_world_from_probe, dtype=np.float64))
        self._mj.mj_forward(self.model, self.data)
        return self.contact_force()

    def close(self):
        pass
