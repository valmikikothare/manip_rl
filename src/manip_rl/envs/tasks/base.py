"""Task protocol: defines objects, goals, rewards, and success for an env.

Tasks are pure descriptions; all mutable state lives in the env's `info` dict
so everything stays jittable.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import jax
from ml_collections import config_dict
from mujoco import mjx

if TYPE_CHECKING:
    from manip_rl.envs.base_env import ManipulationEnv


class Task(abc.ABC):
    """A manipulation task within a robot scene."""

    name: str
    # Body name of the manipulated object in the scene MJCF.
    object_body: str
    # Mocap body marking the goal pose (also serves as visualization).
    goal_mocap: str

    @abc.abstractmethod
    def default_reward_scales(self) -> config_dict.ConfigDict:
        """Scales for each term returned by reward_terms."""

    @abc.abstractmethod
    def sample_object_pos(self, rng: jax.Array, env: "ManipulationEnv") -> jax.Array:
        """Initial object position for an episode."""

    @abc.abstractmethod
    def sample_goal(
        self, rng: jax.Array, env: "ManipulationEnv"
    ) -> tuple[jax.Array, jax.Array]:
        """Goal (pos, quat) for an episode."""

    @abc.abstractmethod
    def reward_terms(
        self, env: "ManipulationEnv", data: mjx.Data, info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        """Unscaled reward terms; keys must match default_reward_scales."""

    @abc.abstractmethod
    def success(
        self, env: "ManipulationEnv", data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        """1.0 when the task is solved (used for eval metrics)."""
