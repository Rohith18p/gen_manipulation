# gen_manipulation

MuJoCo tabletop manipulation with a Franka Emika Panda (7-DoF arm + gripper).

## Setup

```bash
uv sync
```

## Run

```bash
uv run mjpython -m scripts.move_arm             # interactive viewer (macOS needs mjpython)
uv run python -m scripts.move_arm --headless    # no viewer, prints tracking error
uv run pytest -q                                # tests
```

## Layout

- `assets/tabletop.xml` is the scene (table, cubes, target marker). It includes the Panda from `assets/franka_emika_panda/`, which is vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under Apache-2.0.
- `gen_manipulation/sim.py` has `TabletopSim`, which wraps the model and data and handles arm/gripper control and the TCP pose.
- `gen_manipulation/control.py` has `PandaIK` (mink differential IK) and the Cartesian `move_tcp` helper.
- `scripts/move_arm.py` moves the TCP on a tour of workspace waypoints.

The world origin is the robot base on the table surface. +x points forward, +z points up, and the table top is at z=0.
