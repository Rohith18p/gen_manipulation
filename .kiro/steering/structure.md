# Structure

```
assets/                    MJCF scenes + robot models
  tabletop.xml             scene: table, cubes, target marker; includes the Panda
  franka_emika_panda/      vendored from mujoco_menagerie (see header in panda.xml for local edits)
gen_manipulation/          library code
  sim.py                   TabletopSim: load scene, reset, step, arm/gripper control, TCP pose
  control.py               PandaIK (mink) + move_tcp / hold motion helpers
scripts/                   runnable entry points (python -m scripts.<name>)
  move_arm.py              workspace tour demo, viewer or --headless
tests/                     pytest smoke tests
.kiro/steering/            agent context (this folder)
```

## Guidelines
- Keep it flat and small: add a module only when a new responsibility appears
  (e.g. `tasks.py` for pick-and-place), not per class.
- New scenes go in `assets/` next to `tabletop.xml` and `<include>` the robot.
  MuJoCo resolves `meshdir` relative to the top-level scene, so scenes must live in `assets/`.
- Don't edit Menagerie files except for documented local edits in the file header.
- Library code has no viewer/GUI logic; scripts own visualization and CLI flags.
- Add a test in `tests/` for new sim or control behavior.
