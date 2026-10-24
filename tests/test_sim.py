"""Smoke tests for the tabletop scene, IK and Cartesian motion."""

import numpy as np
import pytest

from gen_manipulation import PandaIK, TabletopSim, hold, move_tcp


@pytest.fixture
def sim():
    return TabletopSim()


def test_scene_loads(sim):
    assert len(sim.arm_qpos_ids) == 7
    assert sim.model.nu == 8  # 7 arm position actuators + 1 gripper
    np.testing.assert_allclose(sim.arm_q, sim.model.key_qpos[sim.home_key_id][:7])


def test_home_is_stable_and_cubes_rest_on_table(sim):
    hold(sim, 2.0)
    assert np.abs(sim.arm_q - sim.model.key_qpos[sim.home_key_id][:7]).max() < 1e-3
    for name in ("cube_red", "cube_green", "cube_blue"):
        assert sim.data.body(name).xpos[2] == pytest.approx(0.02, abs=2e-3)


def test_ik_converges(sim):
    ik = PandaIK(sim)
    _, quat = sim.tcp_pose()
    q, converged = ik.solve(np.array([0.6, 0.2, 0.2]), quat, max_iters=300)
    assert converged
    assert q.shape == (7,)


def test_move_tcp_tracks_target(sim):
    ik = PandaIK(sim)
    _, quat = sim.tcp_pose()
    target = np.array([0.65, -0.25, 0.15])
    move_tcp(sim, ik, target, quat, duration=2.0)
    hold(sim, 0.5)
    pos, q_now = sim.tcp_pose()
    assert np.linalg.norm(pos - target) < 2e-3
    assert abs(np.dot(q_now, quat)) > 0.999  # orientation held (gripper down)


def test_gripper_open_close(sim):
    finger = sim.model.joint("finger_joint1").qposadr[0]
    sim.set_gripper(0.0)
    hold(sim, 1.0)
    assert sim.data.qpos[finger] < 0.002
    sim.set_gripper(1.0)
    hold(sim, 1.0)
    assert sim.data.qpos[finger] > 0.038
