"""Registers manip_rl environments into the Playground manipulation registry.

Importing this module is enough; afterwards `mujoco_playground.registry.load`
(or `load_env` below) resolves our names alongside the built-ins.

Use `load_env` instead of `registry.load` directly: recent Playground versions
default `impl` to "warp" (NVIDIA-only); we always want the JAX implementation,
which runs on CPU today and ROCm/CUDA via the JAX backend.
"""

from typing import Any, Optional, Union

from ml_collections import config_dict
from mujoco_playground import registry
from mujoco_playground._src import manipulation

from manip_rl.envs.base_env import ManipulationEnv, make_config
from manip_rl.envs.tasks.pick_place import PickPlace
from manip_rl.robots.panda import PANDA


class PandaPickPlace(ManipulationEnv):
    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ):
        task = PickPlace()
        super().__init__(PANDA, task, config or make_config(task), config_overrides)


class PandaPickPlaceOrientation(ManipulationEnv):
    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ):
        task = PickPlace(sample_orientation=True)
        super().__init__(PANDA, task, config or make_config(task), config_overrides)


def _register():
    from manip_rl.envs.randomize import domain_randomize

    manipulation.register_environment(
        "ManipPickPlace", PandaPickPlace, lambda: make_config(PickPlace())
    )
    manipulation.register_environment(
        "ManipPickPlaceOrientation",
        PandaPickPlaceOrientation,
        lambda: make_config(PickPlace(sample_orientation=True)),
    )
    # register_environment doesn't take a randomizer, but the module-level
    # dict backs registry.get_domain_randomizer; env ids are resolved lazily
    # inside the randomizer itself (it takes env=None for the generic case).
    manipulation._randomizer["ManipPickPlace"] = domain_randomize
    manipulation._randomizer["ManipPickPlaceOrientation"] = domain_randomize


_register()


def load_env(env_name: str, config_overrides: dict | None = None):
    overrides = {"impl": "jax", **(config_overrides or {})}
    return registry.load(env_name, config_overrides=overrides)


def get_config(env_name: str):
    cfg = registry.get_default_config(env_name)
    cfg.impl = "jax"
    return cfg
