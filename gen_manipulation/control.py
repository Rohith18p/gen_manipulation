"""Differential IK (via mink) and Cartesian TCP motion for the Panda."""

from collections.abc import Callable

import mink
import numpy as np

from gen_manipulation.sim import TCP_SITE, TabletopSim


class PandaIK:
    """Solves TCP pose -> arm joint angles with mink's QP-based differential IK.

    Keeps its own kinematic configuration, so successive calls warm-start from
    the previous solution (smooth streaming IK). Call `sync` to re-seed it from
    the simulator state.
    """

    def __init__(self, sim: TabletopSim, solver: str = "daqp", dt: float = 0.01):
        self.sim = sim
        self.solver = solver
        self.dt = dt
        self.configuration = mink.Configuration(sim.model)
        self.tcp_task = mink.FrameTask(
            frame_name=TCP_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-3,
        )
        # Pulls the redundant (7th) DoF toward the home posture.
        self.posture_task = mink.PostureTask(sim.model, cost=1e-2)
        self.posture_task.set_target(sim.model.key_qpos[sim.home_key_id])
        self.limits = [mink.ConfigurationLimit(sim.model)]
        self.sync()

    def sync(self) -> None:
        self.configuration.update(self.sim.data.qpos.copy())

    def solve(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ) -> tuple[np.ndarray, bool]:
        """Return (arm joint angles, converged) for a TCP target (pos m, quat wxyz)."""
        target = mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(quat)), np.asarray(pos))
        self.tcp_task.set_target(target)
        tasks = [self.tcp_task, self.posture_task]

        converged = False
        for _ in range(max_iters):
            vel = mink.solve_ik(
                self.configuration, tasks, self.dt, self.solver, damping=1e-3, limits=self.limits
            )
            self.configuration.integrate_inplace(vel, self.dt)
            current = self.configuration.get_transform_frame_to_world(TCP_SITE, "site")
            pos_err = np.linalg.norm(current.translation() - target.translation())
            rot_err = np.linalg.norm((current.rotation().inverse() @ target.rotation()).log())
            if pos_err < pos_tol and rot_err < rot_tol:
                converged = True
                break
        return self.configuration.q[self.sim.arm_qpos_ids].copy(), converged


def move_tcp(
    sim: TabletopSim,
    ik: PandaIK,
    pos: np.ndarray,
    quat: np.ndarray,
    duration: float,
    control_hz: float = 50.0,
    on_step: Callable[[], None] | None = None,
) -> None:
    """Move the TCP along a straight line to `pos` (fixed orientation `quat`).

    Interpolates the Cartesian path at `control_hz`, solves IK per waypoint and
    sends the result to the arm's position actuators. `on_step` runs after every
    control tick (e.g. viewer sync / real-time pacing).
    """
    start, _ = sim.tcp_pose()
    goal = np.asarray(pos, dtype=float)
    sim.set_target_marker(goal, quat)

    substeps = max(1, round(1.0 / (control_hz * sim.dt)))
    n_ticks = max(1, round(duration * control_hz))
    for i in range(1, n_ticks + 1):
        s = i / n_ticks
        s = 3 * s**2 - 2 * s**3  # smoothstep: zero velocity at both ends
        q, _ = ik.solve(start + s * (goal - start), quat, max_iters=20)
        sim.set_arm_target(q)
        sim.step(substeps)
        if on_step is not None:
            on_step()


def hold(sim: TabletopSim, duration: float, on_step: Callable[[], None] | None = None,
         control_hz: float = 50.0) -> None:
    """Keep current actuator targets and simulate for `duration` seconds."""
    substeps = max(1, round(1.0 / (control_hz * sim.dt)))
    for _ in range(max(1, round(duration * control_hz))):
        sim.step(substeps)
        if on_step is not None:
            on_step()
