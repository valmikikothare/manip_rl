"""Pick-and-place: bring an object to a goal position (optionally pose)."""

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from manip_rl.envs.tasks.base import Task


class PickPlace(Task):
    """Move the object to a (possibly rotated) goal pose above the table."""

    name = "pick_place"
    object_body = "box"
    goal_mocap = "mocap_target"

    def __init__(
        self,
        sample_orientation: bool = False,
        object_pos_range: float = 0.2,
        goal_height_range: tuple[float, float] = (0.2, 0.4),
        success_pos_tol: float = 0.05,
    ):
        self._sample_orientation = sample_orientation
        self._object_pos_range = object_pos_range
        self._goal_height_range = goal_height_range
        self._success_pos_tol = success_pos_tol

    def default_reward_scales(self) -> config_dict.ConfigDict:
        return config_dict.create(
            gripper_obj=4.0,        # reach: gripper approaches the object
            obj_goal=8.0,           # transport: object approaches the goal
            no_floor_collision=0.25,
            robot_home_qpos=0.3,    # regularize arm toward home pose
        )

    def sample_object_pos(self, rng, env):
        r = self._object_pos_range
        offset = jax.random.uniform(
            rng, (3,), minval=jp.array([-r, -r, 0.0]), maxval=jp.array([r, r, 0.0])
        )
        return env.init_obj_pos + offset

    def sample_goal(self, rng, env):
        rng_pos, rng_axis, rng_theta = jax.random.split(rng, 3)
        r = self._object_pos_range
        lo, hi = self._goal_height_range
        pos = env.init_obj_pos + jax.random.uniform(
            rng_pos, (3,), minval=jp.array([-r, -r, lo]), maxval=jp.array([r, r, hi])
        )
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
        if self._sample_orientation:
            axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            axis = axis / math.norm(axis)
            theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            quat = math.axis_angle_to_quat(axis, theta)
        return pos, quat

    def reward_terms(self, env, data: mjx.Data, info: dict[str, Any]):
        obj_pos = data.xpos[env.obj_body_id]
        gripper_pos = data.site_xpos[env.ee_site_id]

        pos_err = jp.linalg.norm(info["target_pos"] - obj_pos)
        target_mat = math.quat_to_mat(info["target_quat"])
        rot_err = jp.linalg.norm(
            target_mat.ravel()[:6] - data.xmat[env.obj_body_id].ravel()[:6]
        )
        rot_weight = 0.1 if self._sample_orientation else 0.0
        obj_goal = 1 - jp.tanh(5 * ((1 - rot_weight) * pos_err + rot_weight * rot_err))

        gripper_obj = 1 - jp.tanh(5 * jp.linalg.norm(obj_pos - gripper_pos))

        robot_home_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[env.arm_qposadr] - env.init_q[env.arm_qposadr]
            )
        )
        no_floor_collision = 1.0 - env.floor_collision(data)

        # Gate the transport term on having reached the object once, so the
        # policy can't farm obj_goal reward by nudging the object along the floor.
        info["reached_obj"] = jp.maximum(
            info["reached_obj"],
            (jp.linalg.norm(obj_pos - gripper_pos) < 0.012).astype(float),
        )

        return {
            "gripper_obj": gripper_obj,
            "obj_goal": obj_goal * info["reached_obj"],
            "no_floor_collision": no_floor_collision,
            "robot_home_qpos": robot_home_qpos,
        }

    def success(self, env, data: mjx.Data, info: dict[str, Any]):
        pos_err = jp.linalg.norm(info["target_pos"] - data.xpos[env.obj_body_id])
        return (pos_err < self._success_pos_tol).astype(float)
