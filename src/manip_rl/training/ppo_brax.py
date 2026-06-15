"""Baseline trainer: brax PPO on a Playground env (mirrors the official notebook).

This is the known-good reference path; the hackable trainer lives in ppo.py.

Usage:
    uv run python -m manip_rl.training.ppo_brax --env PandaPickCube
    # quick CPU sanity run:
    uv run python -m manip_rl.training.ppo_brax --env PandaPickCube \
        --num-timesteps 500000 --num-envs 256
"""

import argparse
import functools
import json
import pathlib
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params

from manip_rl.envs.registry import load_env
from manip_rl.training.checkpoints import save_policy

# Do NOT enable jax_enable_x64 here. MJX/Playground/brax PPO all run in float32;
# float64 kernels hard-crash (SIGSEGV) on the ROCm RDNA2 card (gfx1032 ->
# gfx1030 impersonation), whose FP64 path is unusable. x64 is not needed for
# correct physics — the float32 GPU step matches the CPU step to ~1e-6.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="PandaPickCubeOrientation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out", default=None, help="run dir (default runs/<env>-<time>)"
    )
    # CPU-friendly overrides; None keeps the env's published hyperparameters.
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-evals", type=int, default=None)
    args = parser.parse_args()

    run_dir = pathlib.Path(
        args.out or f"runs/{args.env}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    env = load_env(args.env)
    eval_env = load_env(args.env)

    ppo_params = manipulation_params.brax_ppo_config(args.env)
    for key in ("num_timesteps", "num_envs", "num_evals"):
        if getattr(args, key) is not None:
            ppo_params[key] = getattr(args, key)

    network_factory_kwargs = {}
    ppo_training_params = dict(ppo_params)
    if "network_factory" in ppo_params:
        network_factory_kwargs = dict(ppo_params.network_factory)
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_kwargs
    )

    history = []
    t_start = time.monotonic()

    def progress(num_steps, metrics):
        entry = {
            "steps": int(num_steps),
            "reward": float(metrics["eval/episode_reward"]),
            "reward_std": float(metrics["eval/episode_reward_std"]),
            "wall_time": time.monotonic() - t_start,
        }
        history.append(entry)
        print(
            f"[{entry['wall_time']:7.1f}s] steps={entry['steps']:>10,} "
            f"reward={entry['reward']:8.2f} ± {entry['reward_std']:.2f}",
            flush=True,
        )

    def checkpoint(current_step, make_policy, params):
        # Called by brax at every eval; keeps progress if the run is killed.
        del current_step, make_policy
        save_policy(
            run_dir,
            params,
            network_factory_kwargs,
            env.observation_size,
            env.action_size,
        )

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=checkpoint,
        seed=args.seed,
    )
    _, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    save_policy(
        run_dir, params, network_factory_kwargs, env.observation_size, env.action_size
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved policy + history to {run_dir}")
    print(
        f"Render with: uv run python -m manip_rl.viz.render --env {args.env} "
        f"--policy {run_dir}"
    )


if __name__ == "__main__":
    main()
