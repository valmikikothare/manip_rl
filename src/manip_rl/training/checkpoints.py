"""Save/load trained policies as `policy(obs, rng) -> action` functions.

Two formats share the policy.pkl convention: "brax_ppo" (ppo_brax.py) and
"cleanrl_ppo" (ppo.py). load_policy dispatches on the payload's format tag.
"""

import functools
import pathlib
import pickle

from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp


def save_policy(ckpt_dir, params, network_factory_kwargs, obs_size, act_size):
    """Persist brax PPO params plus what's needed to rebuild the inference net."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "brax_ppo",
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
    fmt = payload.get("format", "brax_ppo")
    if fmt == "brax_ppo":
        return _load_brax(payload, deterministic)
    if fmt == "cleanrl_ppo":
        return _load_cleanrl(payload)
    raise ValueError(f"Unknown policy format: {fmt}")


def _load_brax(payload, deterministic):
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


def _load_cleanrl(payload):
    from manip_rl.training.ppo import Actor, norm_apply

    args = payload["args"]
    actor = Actor(tuple(args["policy_hidden"]), payload["act_dim"])
    params, norms = payload["actor_params"], payload["norms"]
    obs_key = args["policy_obs_key"]

    @jax.jit
    def policy(obs, rng):
        del rng  # deterministic: act at the mean
        x = obs[obs_key] if isinstance(obs, dict) else obs
        mean, _ = actor.apply(params, norm_apply(norms["policy"], x))
        return jp.clip(mean, -1.0, 1.0)

    return policy
