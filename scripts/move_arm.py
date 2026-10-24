"""Move the Panda TCP through a tour of tabletop workspace waypoints.

Interactive viewer (macOS requires mjpython; loops until the window is closed):
    uv run mjpython -m scripts.move_arm
Headless, one pass, prints tracking error per waypoint:
    uv run python -m scripts.move_arm --headless
"""

import argparse
import time

import numpy as np

from gen_manipulation import PandaIK, TabletopSim, hold, move_tcp

# (name, TCP position in world frame [m], gripper opening after arrival [0..1]).
# Table top is z=0; cubes are 4 cm tall at x~0.55-0.6.
WAYPOINTS = [
    ("left_high", (0.45, 0.30, 0.35), 1.0),
    ("left_far", (0.70, 0.30, 0.15), 1.0),
    ("right_far", (0.70, -0.30, 0.15), 1.0),
    ("right_high", (0.45, -0.30, 0.35), 1.0),
    ("above_green", (0.60, 0.00, 0.15), 1.0),
    ("near_green", (0.60, 0.00, 0.07), 0.0),  # close gripper (no grasp, just actuate)
    ("above_green", (0.60, 0.00, 0.15), 1.0),
]
SEGMENT_TIME = 2.0  # s per waypoint
SETTLE_TIME = 0.5  # s hold after each waypoint


def run_tour(sim: TabletopSim, ik: PandaIK, home_pos, down_quat, on_step=None, verbose=True):
    """One pass over WAYPOINTS and back home. Returns per-waypoint TCP errors (m)."""
    errors = []
    for name, pos, gripper in [*WAYPOINTS, ("home", tuple(home_pos), 1.0)]:
        target = np.array(pos)
        move_tcp(sim, ik, target, down_quat, SEGMENT_TIME, on_step=on_step)
        sim.set_gripper(gripper)
        hold(sim, SETTLE_TIME, on_step=on_step)
        err = np.linalg.norm(sim.tcp_pose()[0] - target)
        errors.append(err)
        if verbose:
            print(f"{name:<12} target {target.round(3)}  err {err * 1000:5.1f} mm  gripper {gripper:.0f}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="run one pass without the viewer")
    args = parser.parse_args()

    sim = TabletopSim()
    ik = PandaIK(sim)
    home_pos, down_quat = sim.tcp_pose()  # keep the home (gripper-down) orientation
    print(f"Loaded {sim.model.nq} qpos / {sim.model.nu} actuators; TCP home {home_pos.round(3)}")

    if args.headless:
        errors = run_tour(sim, ik, home_pos, down_quat)
        print(f"Max tracking error {max(errors) * 1000:.1f} mm over {sim.data.time:.1f} s sim time")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        wall_start, sim_start = time.perf_counter(), sim.data.time

        def on_step():
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            # Pace the simulation to real time.
            lag = (sim.data.time - sim_start) - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)

        try:
            while viewer.is_running():
                run_tour(sim, ik, home_pos, down_quat, on_step=on_step)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
