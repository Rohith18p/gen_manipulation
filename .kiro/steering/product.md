# Product

Research codebase for generative robot manipulation in simulation.

- Robot: Franka Emika Panda (7-DoF arm + parallel gripper) in MuJoCo.
- Scene: robot mounted on a tabletop with small objects (cubes).
- Current phase: simulation setup and testing — scene loads, arm moves to
  Cartesian targets via IK, gripper actuates. No grasping/learning yet.
- Next phases (planned): pick-and-place, data collection, learned/generative
  policies, batched simulation with MJX.
