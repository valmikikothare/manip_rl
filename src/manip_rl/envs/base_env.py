"""Generic manipulation environment: RobotConfig + Task -> MjxEnv.

Observations are a dict:
  - "state":            policy observation (proprio + object/goal relatives)
  - "privileged_state": critic observation (state + ground-truth object pose,
                        velocity, and goal) for asymmetric actor-critic.
Brax PPO selects keys via network_factory(policy_obs_key=..., value_obs_key=...).
"""

from typing import Any, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
import numpy as np

from manip_rl.envs.tasks.base import Task
from manip_rl.robots.base import RobotConfig


def make_config(task: Task) -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(scales=task.default_reward_scales()),
        impl="jax",
        naconmax=24 * 2048,
        naccdmax=24 * 2048,
        njmax=128,
    )


class ManipulationEnv(mjx_env.MjxEnv):
    """Robot-and-task-parameterized manipulation environment."""

    def __init__(
        self,
        robot: RobotConfig,
        task: Task,
        config: config_dict.ConfigDict,
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._robot = robot
        self._task = task

        self._xml_path = robot.scene_xml.as_posix()
        mj_model = mujoco.MjModel.from_xml_string(
            robot.scene_xml.read_text(), assets=robot.assets()
        )
        mj_model.opt.timestep = self.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        self._action_scale = self._config.action_scale

        # Resolve ids/addresses once, as numpy, so everything below is jittable.
        jnt_qposadr = lambda name: mj_model.jnt_qposadr[mj_model.joint(name).id]
        self.arm_qposadr = np.array([jnt_qposadr(j) for j in robot.arm_joints])
        self.robot_qposadr = np.array([jnt_qposadr(j) for j in robot.joints])
        self.ctrl_qposadr = np.array([jnt_qposadr(j) for j in robot.ctrl_joints])
        self.ee_site_id = mj_model.site(robot.ee_site).id
        self.obj_body_id = mj_model.body(task.object_body).id
        self._obj_qposadr = mj_model.jnt_qposadr[
            mj_model.body(task.object_body).jntadr[0]
        ]
        self._goal_mocapid = mj_model.body(task.goal_mocap).mocapid
        self._floor_sensor_adr = np.array([
            mj_model.sensor_adr[mj_model.sensor(s).id]
            for s in robot.floor_contact_sensors
        ])

        key = mj_model.keyframe(robot.home_keyframe)
        self.init_q = jp.array(key.qpos)
        self._init_ctrl = jp.array(key.ctrl)
        self.init_obj_pos = jp.array(
            key.qpos[self._obj_qposadr : self._obj_qposadr + 3], dtype=jp.float32
        )
        self._lowers, self._uppers = mj_model.actuator_ctrlrange.T

    # --- helpers used by tasks -------------------------------------------------

    def floor_collision(self, data: mjx.Data) -> jax.Array:
        """1.0 if any registered hand geom touches the floor."""
        if len(self._floor_sensor_adr) == 0:
            return jp.array(0.0)
        return jp.any(data.sensordata[self._floor_sensor_adr] > 0).astype(float)

    # --- MjxEnv API --------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, rng_obj, rng_goal = jax.random.split(rng, 3)

        obj_pos = self._task.sample_object_pos(rng_obj, self)
        target_pos, target_quat = self._task.sample_goal(rng_goal, self)

        init_q = self.init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(obj_pos)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocapid, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._goal_mocapid, :].set(target_quat),
        )

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "reached_obj": 0.0,
        }
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            "success": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        ctrl = state.data.ctrl + action * self._action_scale
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        raw_rewards = self._task.reward_terms(self, data, state.info)
        scales = self._config.reward_config.scales
        reward = jp.clip(
            sum(v * scales[k] for k, v in raw_rewards.items()), -1e4, 1e4
        )

        obj_pos = data.xpos[self.obj_body_id]
        out_of_bounds = jp.any(jp.abs(obj_pos) > 1.0) | (obj_pos[2] < 0.0)
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        state.metrics.update(
            **raw_rewards,
            out_of_bounds=out_of_bounds.astype(float),
            success=self._task.success(self, data, state.info),
        )
        obs = self._get_obs(data, state.info)
        return State(data, obs, reward, done.astype(float), state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        ee_pos = data.site_xpos[self.ee_site_id]
        ee_mat = data.site_xmat[self.ee_site_id].ravel()
        obj_mat = data.xmat[self.obj_body_id].ravel()
        obj_pos = data.xpos[self.obj_body_id]
        target_mat = math.quat_to_mat(info["target_quat"])

        state = jp.concatenate([
            data.qpos,
            data.qvel,
            ee_pos,
            ee_mat[3:],
            obj_mat[3:],
            obj_pos - ee_pos,
            info["target_pos"] - obj_pos,
            target_mat.ravel()[:6] - obj_mat[:6],
            data.ctrl - data.qpos[self.ctrl_qposadr],
        ])
        privileged = jp.concatenate([
            state,
            obj_pos,
            data.xquat[self.obj_body_id],
            data.cvel[self.obj_body_id],
            info["target_pos"],
            info["target_quat"],
        ])
        return {"state": state, "privileged_state": privileged}

    # --- properties ---------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
