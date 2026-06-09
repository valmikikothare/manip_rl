"""Planning interfaces: goal specifications and the Planner protocol.

Planners run on CPU MuJoCo (mujoco.MjModel/MjData) outside the JAX JIT
boundary; the HierarchicalAgent (hierarchy.py) composes them with learned
policies in ordinary Python rollout loops.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class GoalSpec:
    """A high-level goal a planner refines into a joint trajectory.

    Exactly one of joint_target / ee_pos should be set. ee_* goals are
    resolved to joint space via inverse kinematics before planning.
    Equality is approximate, so replan-on-change is robust to float noise.
    """

    joint_target: np.ndarray | None = None      # (n_arm_joints,)
    ee_pos: np.ndarray | None = None            # (3,)
    ee_quat: np.ndarray | None = None           # (4,) wxyz, optional with ee_pos
    # Joints the planner may move (defaults to the arm; gripper stays put).
    metadata: dict = field(default_factory=dict)

    def approx_equals(self, other: "GoalSpec | None", tol: float = 1e-3) -> bool:
        if other is None:
            return False

        def close(a, b):
            if (a is None) != (b is None):
                return False
            return a is None or bool(np.allclose(a, b, atol=tol))

        return (close(self.joint_target, other.joint_target)
                and close(self.ee_pos, other.ee_pos)
                and close(self.ee_quat, other.ee_quat))


@dataclass
class PlanResult:
    """A joint-space trajectory: waypoints[i] is a full arm configuration."""

    waypoints: np.ndarray   # (num_waypoints, n_joints)
    success: bool
    info: dict = field(default_factory=dict)


class Planner(Protocol):
    def plan(
        self,
        model,                  # mujoco.MjModel (CPU)
        start_q: np.ndarray,    # current full qpos
        goal: GoalSpec,
    ) -> PlanResult:
        """Plan a collision-free joint trajectory from start_q to goal."""
        ...
