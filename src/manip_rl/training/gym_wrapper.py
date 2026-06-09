"""Gymnasium adapter for MjxEnv environments (CPU, single env).

Lets CPU-based libraries (Stable-Baselines3, etc.) train on any registered
env. Each Gym step runs the jitted MJX step for one world — fine for SB3-style
workflows, but the JAX trainers (ppo.py) are the fast path.

Usage:
    from manip_rl.training.gym_wrapper import MjxGymEnv
    env = MjxGymEnv("ManipPickPlace")  # obs = flat "state" key
    # SB3: model = PPO("MlpPolicy", env); model.learn(100_000)
"""

import jax
import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("Install the sb3 extra: uv sync --extra sb3") from e

from manip_rl.envs.registry import load_env


class MjxGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name: str, obs_key: str = "state", seed: int = 0):
        self._env = load_env(env_name)
        self._obs_key = obs_key
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        self._rng = jax.random.PRNGKey(seed)
        self._state = None
        self._t = 0
        self._episode_length = self._env._config.episode_length

        obs_size = self._env.observation_size
        if isinstance(obs_size, dict):
            obs_size = obs_size[obs_key][0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self._env.action_size,), np.float32)

    def _obs(self):
        obs = self._state.obs
        if isinstance(obs, dict):
            obs = obs[self._obs_key]
        return np.asarray(obs, dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, reset_rng = jax.random.split(self._rng)
        self._state = self._reset_fn(reset_rng)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._state = self._step_fn(self._state, action.astype(np.float32))
        self._t += 1
        terminated = bool(self._state.done)
        truncated = self._t >= self._episode_length
        info = {k: float(v) for k, v in self._state.metrics.items()}
        return self._obs(), float(self._state.reward), terminated, truncated, info

    def render(self):
        return self._env.render([self._state])[0]
