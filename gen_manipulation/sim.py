"""Tabletop MuJoCo simulation with a 7-DoF Franka Panda arm."""

from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "tabletop.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0  # actuator8 ctrl range, maps to 0..0.04 m per finger
TCP_SITE = "tcp"
HOME_KEY = "home"


class TabletopSim:
    """Thin wrapper over MjModel/MjData exposing the arm, gripper and TCP.

    The arm uses position actuators, so `set_arm_target` takes joint angles (rad).
    """

    def __init__(self, xml_path: str | Path = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.arm_qpos_ids = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof_ids = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_act_ids = np.array([m.actuator(a).id for a in ARM_ACTUATORS])
        self.gripper_act_id = m.actuator(GRIPPER_ACTUATOR).id
        self.tcp_site_id = m.site(TCP_SITE).id
        self.target_mocap_id = m.body("target").mocapid[0]
        self.home_key_id = m.key(HOME_KEY).id

        self.reset()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def reset(self) -> None:
        """Reset to the home keyframe (arm home, gripper open, objects at spawn)."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    # --- Arm / gripper -------------------------------------------------------
    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_ids].copy()

    def set_arm_target(self, q: np.ndarray) -> None:
        self.data.ctrl[self.arm_act_ids] = q

    def set_gripper(self, opening: float) -> None:
        """opening in [0, 1]: 0 = closed, 1 = fully open."""
        opening = float(np.clip(opening, 0.0, 1.0))
        self.data.ctrl[self.gripper_act_id] = GRIPPER_CLOSED + opening * (GRIPPER_OPEN - GRIPPER_CLOSED)

    # --- Poses ---------------------------------------------------------------
    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (m) and orientation quaternion (wxyz) in the world frame."""
        pos = self.data.site_xpos[self.tcp_site_id].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp_site_id])
        return pos, quat

    def set_target_marker(self, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        """Move the visual-only target marker (mocap body)."""
        self.data.mocap_pos[self.target_mocap_id] = pos
        if quat is not None:
            self.data.mocap_quat[self.target_mocap_id] = quat
