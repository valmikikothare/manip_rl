"""Joint-space RRT with shortcut smoothing, on CPU MuJoCo collision checks.

Good-enough classical baseline for approach planning; swap in OMPL/TrajOpt
behind the same Planner protocol later if needed.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from manip_rl.planning.base import GoalSpec, PlanResult
from manip_rl.robots.base import RobotConfig


def ik_solve(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_qadr: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    max_iters: int = 200,
    tol: float = 1e-3,
    damping: float = 1e-2,
) -> tuple[np.ndarray, bool]:
    """Damped least-squares IK for an end-effector site; returns (qpos, ok)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    # Map arm qpos addresses to dof indices (hinge/slide joints: 1 dof each).
    arm_dofadr = np.array([
        model.jnt_dofadr[j] for j in range(model.njnt)
        if model.jnt_qposadr[j] in set(arm_qadr.tolist())
    ])

    for _ in range(max_iters):
        mujoco.mj_fwdPosition(model, data)
        err_pos = target_pos - data.site_xpos[site_id]
        err = err_pos
        if target_quat is not None:
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
            err_quat = np.zeros(3)
            dq = np.zeros(4)
            mujoco.mju_negQuat(dq, site_quat)
            mujoco.mju_mulQuat(dq, target_quat, dq)
            mujoco.mju_quat2Vel(err_quat, dq, 1.0)
            err = np.concatenate([err_pos, err_quat])
        if np.linalg.norm(err_pos) < tol:
            return data.qpos.copy(), True

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = jacp[:, arm_dofadr]
        if target_quat is not None:
            jac = np.vstack([jacp[:, arm_dofadr], jacr[:, arm_dofadr]])
        dq_arm = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * np.eye(jac.shape[0]), err
        )
        data.qpos[arm_qadr] += np.clip(dq_arm, -0.2, 0.2)
    return data.qpos.copy(), False


@dataclass
class RRTPlanner:
    robot: RobotConfig
    max_iters: int = 2000
    step_size: float = 0.15          # rad, joint-space extension step
    goal_bias: float = 0.2
    goal_tol: float = 0.05
    shortcut_iters: int = 100
    collision_check_resolution: float = 0.05
    seed: int = 0

    def plan(self, model: mujoco.MjModel, start_q: np.ndarray, goal: GoalSpec) -> PlanResult:
        rng = np.random.default_rng(self.seed)
        data = mujoco.MjData(model)
        arm_qadr = np.array([
            model.jnt_qposadr[model.joint(j).id] for j in self.robot.arm_joints
        ])
        arm_jids = [model.joint(j).id for j in self.robot.arm_joints]
        lowers = model.jnt_range[arm_jids, 0]
        uppers = model.jnt_range[arm_jids, 1]
        robot_geoms = self._robot_geom_ids(model)

        # Resolve goal to an arm configuration.
        if goal.joint_target is not None:
            goal_q = np.asarray(goal.joint_target)
        else:
            data.qpos[:] = start_q
            full_q, ok = ik_solve(
                model, data, model.site(self.robot.ee_site).id, arm_qadr,
                np.asarray(goal.ee_pos),
                None if goal.ee_quat is None else np.asarray(goal.ee_quat),
            )
            if not ok:
                return PlanResult(np.empty((0, len(arm_qadr))), False, {"reason": "ik_failed"})
            goal_q = full_q[arm_qadr]
        goal_q = np.clip(goal_q, lowers, uppers)

        def in_collision(arm_q: np.ndarray) -> bool:
            data.qpos[:] = start_q          # everything else stays put
            data.qpos[arm_qadr] = arm_q
            mujoco.mj_fwdPosition(model, data)
            for i in range(data.ncon):
                con = data.contact[i]
                if con.dist > 0:
                    continue
                g1, g2 = con.geom1, con.geom2
                if (g1 in robot_geoms) != (g2 in robot_geoms):
                    return True             # robot touching world
                if g1 in robot_geoms and g2 in robot_geoms:
                    b1 = model.geom_bodyid[g1]
                    b2 = model.geom_bodyid[g2]
                    if b1 != b2:
                        return True         # self-collision
            return False

        def segment_free(q_a: np.ndarray, q_b: np.ndarray) -> bool:
            dist = np.linalg.norm(q_b - q_a)
            n = max(2, int(dist / self.collision_check_resolution))
            for t in np.linspace(0.0, 1.0, n):
                if in_collision(q_a + t * (q_b - q_a)):
                    return False
            return True

        start_arm_q = start_q[arm_qadr]
        if in_collision(start_arm_q):
            return PlanResult(np.empty((0, len(arm_qadr))), False, {"reason": "start_in_collision"})
        if in_collision(goal_q):
            return PlanResult(np.empty((0, len(arm_qadr))), False, {"reason": "goal_in_collision"})

        # --- RRT -------------------------------------------------------------
        nodes = [start_arm_q]
        parents = [-1]
        goal_idx = -1
        for _ in range(self.max_iters):
            sample = goal_q if rng.random() < self.goal_bias else rng.uniform(lowers, uppers)
            near_idx = int(np.argmin([np.linalg.norm(n - sample) for n in nodes]))
            near = nodes[near_idx]
            direction = sample - near
            dist = np.linalg.norm(direction)
            new = sample if dist <= self.step_size else near + direction / dist * self.step_size
            if not segment_free(near, new):
                continue
            nodes.append(new)
            parents.append(near_idx)
            if np.linalg.norm(new - goal_q) < self.goal_tol and segment_free(new, goal_q):
                nodes.append(goal_q)
                parents.append(len(nodes) - 2)
                goal_idx = len(nodes) - 1
                break
        if goal_idx < 0:
            return PlanResult(np.empty((0, len(arm_qadr))), False,
                              {"reason": "max_iters", "tree_size": len(nodes)})

        path = []
        idx = goal_idx
        while idx >= 0:
            path.append(nodes[idx])
            idx = parents[idx]
        path = path[::-1]

        # --- shortcut smoothing -------------------------------------------------
        for _ in range(self.shortcut_iters):
            if len(path) <= 2:
                break
            i, j = sorted(rng.choice(len(path), size=2, replace=False))
            if j - i > 1 and segment_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]

        return PlanResult(np.array(path), True, {"tree_size": len(nodes)})

    def _robot_geom_ids(self, model: mujoco.MjModel) -> set[int]:
        """Geoms attached to the robot's kinematic chain (via arm joint bodies)."""
        root = model.jnt_bodyid[model.joint(self.robot.arm_joints[0]).id]
        ids = set()
        for b in range(model.nbody):
            cur = b
            while cur != 0:
                if cur == root:
                    ids.add(b)
                    break
                cur = model.body_parentid[cur]
        ids.add(root)
        geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in ids}
        return geoms
