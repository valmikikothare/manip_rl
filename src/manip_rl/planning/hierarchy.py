"""Hierarchical control: learned high-level goals -> classical planner -> tracking.

The high-level policy emits a GoalSpec (or None to hand control to the
low-level policy). Whenever the GoalSpec changes, the planner re-plans from
the current configuration; while a plan is active, actions track its
waypoints through the env's delta-position action space. Runs in ordinary
Python (outside JIT) — intended for evaluation/deployment and CPU rollouts.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from manip_rl.planning.base import GoalSpec, Planner
from manip_rl.robots.base import RobotConfig

# high_level(obs, t) -> GoalSpec | None; low_level(obs, rng) -> action
HighLevelFn = Callable[[Any, int], GoalSpec | None]
LowLevelFn = Callable[[Any, Any], np.ndarray]


@dataclass
class HierarchicalAgent:
    env: Any                      # ManipulationEnv (for model/action conventions)
    robot: RobotConfig
    planner: Planner
    high_level: HighLevelFn
    low_level: LowLevelFn | None = None
    waypoint_tol: float = 0.05    # rad, advance when this close

    _goal: GoalSpec | None = field(default=None, init=False)
    _plan: np.ndarray | None = field(default=None, init=False)
    _wp_idx: int = field(default=0, init=False)

    def reset(self):
        self._goal, self._plan, self._wp_idx = None, None, 0

    def act(self, state, t: int, rng=None) -> np.ndarray:
        """Compute an env action for the current (host-side) state."""
        goal = self.high_level(state.obs, t)

        if goal is None:
            if self.low_level is None:
                raise ValueError("high_level returned None but no low_level policy set")
            self._plan = None
            return np.asarray(self.low_level(state.obs, rng))

        if not goal.approx_equals(self._goal):
            self._goal = goal
            qpos = np.asarray(state.data.qpos)
            result = self.planner.plan(self.env.mj_model, qpos, goal)
            self._plan = result.waypoints if result.success else None
            self._wp_idx = 0
            if not result.success:
                # Planner failed: fall back to low-level policy if present.
                if self.low_level is not None:
                    return np.asarray(self.low_level(state.obs, rng))
                return np.zeros(self.env.action_size)

        return self._track_waypoint(state)

    def _track_waypoint(self, state) -> np.ndarray:
        """Delta-position action steering arm ctrl toward the active waypoint."""
        arm_q = np.asarray(state.data.qpos)[self.env.arm_qposadr]
        while (self._wp_idx < len(self._plan) - 1
               and np.linalg.norm(self._plan[self._wp_idx] - arm_q) < self.waypoint_tol):
            self._wp_idx += 1
        waypoint = self._plan[self._wp_idx]

        # Env convention: ctrl += action * action_scale (per ctrl joint).
        ctrl = np.asarray(state.data.ctrl)
        action = np.zeros(self.env.action_size)
        n_arm = len(self.robot.arm_joints)
        action[:n_arm] = (waypoint - ctrl[:n_arm]) / self.env._action_scale
        return np.clip(action, -1.0, 1.0)

    @property
    def plan_active(self) -> bool:
        return self._plan is not None and self._wp_idx < len(self._plan)
