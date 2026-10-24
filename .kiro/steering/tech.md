# Tech

- Python 3.12, managed with `uv` (`pyproject.toml` + `uv.lock`). Pin exact versions.
  Ask the user before adding a new dependency.
- Simulation: `mujoco` 3.17 (CPU). `mujoco-mjx` + `jax` installed for later batched
  sim; JAX runs on CPU on this Mac. "Failed to import warp" messages on import are
  harmless (optional NVIDIA backend).
- IK: `mink` (QP differential IK, `daqp` solver).
- Tests: `pytest`.

## Commands
- Install / sync env: `uv sync`
- Tests: `uv run pytest -q`
- Headless demo: `uv run python -m scripts.move_arm --headless`
- Viewer demo (macOS needs mjpython): `uv run mjpython -m scripts.move_arm`
- Inspect a scene: `uv run mjpython -m mujoco.viewer --mjcf assets/tabletop.xml`

## Conventions
- World frame: origin = robot base on the table surface, +x forward, +z up; table top z=0.
- Quaternions are wxyz (MuJoCo convention). Units: m, rad, s.
- Arm actuators are position servos: `ctrl` = joint angle targets. Gripper ctrl 0..255
  (wrapped by `TabletopSim.set_gripper(0..1)`).
- Robot links have `gravcomp="1"` (like the real Franka controller); keep it, or the
  servos sag ~1 cm.
- Scripts run as modules from the repo root (`python -m scripts.<name>`); the package
  is not installed, and pytest gets `pythonpath = ["."]` from pyproject.
- Verify changes with `uv run pytest -q` and the headless demo (tracking error < 2 mm).
