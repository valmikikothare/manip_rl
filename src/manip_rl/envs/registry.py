"""Registers manip_rl environments into the Playground manipulation registry.

Importing this module is enough; afterwards `mujoco_playground.registry.load`
can resolve our environment names alongside the built-ins.

Use `load_env` instead of `registry.load` directly: recent Playground versions
default `impl` to "warp" (NVIDIA-only); we always want the JAX implementation,
which runs on CPU today and ROCm/CUDA via the JAX backend.
"""

from mujoco_playground import registry

# Custom envs land here in the robot/task abstraction milestone:
#   manipulation.register_environment("ManipPickPlace", PickPlaceEnv, pick_place.default_config)


def load_env(env_name: str, config_overrides: dict | None = None):
    overrides = {"impl": "jax", **(config_overrides or {})}
    return registry.load(env_name, config_overrides=overrides)


def get_config(env_name: str):
    cfg = registry.get_default_config(env_name)
    cfg.impl = "jax"
    return cfg
