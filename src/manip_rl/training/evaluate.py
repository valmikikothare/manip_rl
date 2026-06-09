"""Deterministic policy evaluation: success rate, returns, optional video.

Usage:
    uv run python -m manip_rl.training.evaluate --env ManipPickPlace \
        --policy runs/<dir> [--episodes 50] [--video]
"""

import argparse

import jax
import jax.numpy as jp
import numpy as np

from manip_rl.envs.registry import load_env
from manip_rl.training.checkpoints import load_policy


def evaluate(env, policy_fn, num_episodes: int, seed: int = 0):
    """Sequential episodes (jitted reset/step); returns per-episode stats."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    episode_length = env._config.episode_length

    returns, successes = [], []
    rng = jax.random.PRNGKey(seed)
    for _ in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        ep_return, ep_success = 0.0, 0.0
        for _ in range(episode_length):
            rng, act_rng = jax.random.split(rng)
            state = jit_step(state, policy_fn(state.obs, act_rng))
            ep_return += float(state.reward)
            ep_success = max(ep_success, float(state.metrics.get("success", 0.0)))
            if bool(state.done):
                break
        returns.append(ep_return)
        successes.append(ep_success)
    return np.array(returns), np.array(successes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="ManipPickPlace")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true",
                        help="also render one episode to mp4")
    args = parser.parse_args()

    env = load_env(args.env)
    policy_fn = load_policy(args.policy, env)
    returns, successes = evaluate(env, policy_fn, args.episodes, args.seed)

    print(f"{args.env} x {args.episodes} episodes")
    print(f"  return:  {returns.mean():8.2f} ± {returns.std():.2f} "
          f"(min {returns.min():.2f}, max {returns.max():.2f})")
    print(f"  success: {successes.mean() * 100:5.1f}%")

    if args.video:
        import mediapy as media
        from manip_rl.viz.render import rollout
        states = rollout(env, policy_fn, jax.random.PRNGKey(args.seed),
                         env._config.episode_length)
        out = f"outputs/{args.env}_eval.mp4"
        media.write_video(out, env.render(states), fps=1.0 / env.dt)
        print(f"  video:   {out}")


if __name__ == "__main__":
    main()
