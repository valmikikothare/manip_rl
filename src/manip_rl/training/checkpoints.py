"""Save/load brax PPO policies as inference functions."""

import functools
import pathlib
import pickle

from brax.training.agents.ppo import networks as ppo_networks
import jax


def save_policy(ckpt_dir, params, network_factory_kwargs, obs_size, act_size):
    """Persist PPO params plus what's needed to rebuild the inference net."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "network_factory_kwargs": network_factory_kwargs,
        "obs_size": obs_size,
        "act_size": act_size,
    }
    with open(ckpt_dir / "policy.pkl", "wb") as f:
        pickle.dump(payload, f)


def load_policy(ckpt_dir, env, deterministic: bool = True):
    """Rebuild a jitted `policy(obs, rng) -> action` from a checkpoint dir."""
    with open(pathlib.Path(ckpt_dir) / "policy.pkl", "rb") as f:
        payload = pickle.load(f)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **payload["network_factory_kwargs"]
    )
    ppo_network = network_factory(payload["obs_size"], payload["act_size"])
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = jax.jit(make_policy(payload["params"], deterministic=deterministic))

    def policy(obs, rng):
        action, _ = inference_fn(obs, rng)
        return action

    return policy
