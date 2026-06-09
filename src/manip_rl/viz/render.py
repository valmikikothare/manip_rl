"""Render environment rollouts to mp4.

Usage:
    uv run python -m manip_rl.viz.render --env PandaPickCube
    uv run python -m manip_rl.viz.render --env PandaPickCube --policy <ckpt-dir>

Without --policy, actions are random — useful as a stack smoke test.
"""

import argparse
import pathlib

import jax
import jax.numpy as jp
import mediapy as media

from manip_rl.envs.registry import load_env


def rollout(env, policy_fn, rng, num_steps: int):
    """Roll out policy_fn(obs, rng) -> action for num_steps; returns states."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(rng)
    states = [state]
    for _ in range(num_steps):
        rng, act_rng = jax.random.split(rng)
        action = policy_fn(state.obs, act_rng)
        state = jit_step(state, action)
        states.append(state)
        if state.done.all():
            break
    return states


def random_policy(env):
    def policy(obs, rng):
        return jax.random.uniform(rng, (env.action_size,), minval=-1.0, maxval=1.0)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="PandaPickCube")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output mp4 path")
    parser.add_argument("--policy", default=None, help="checkpoint dir (brax PPO params)")
    args = parser.parse_args()

    env = load_env(args.env)
    print(f"Loaded {args.env}: obs={env.observation_size}, act={env.action_size}")

    if args.policy:
        from manip_rl.training.checkpoints import load_policy
        policy_fn = load_policy(args.policy, env)
    else:
        policy_fn = random_policy(env)

    states = rollout(env, policy_fn, jax.random.PRNGKey(args.seed), args.steps)
    returns = float(jp.sum(jp.stack([s.reward for s in states])))
    print(f"Rolled out {len(states)} steps, return = {returns:.2f}")

    frames = env.render(states)
    out = pathlib.Path(args.out or f"outputs/{args.env}_rollout.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(out, frames, fps=1.0 / env.dt)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
