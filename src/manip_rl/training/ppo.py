"""CleanRL-style PPO in JAX for MJX/Playground envs — single file, hackable.

The algorithm lives entirely in this file (networks, GAE, clipped objective,
update loop). Environment batching/episode handling reuses Playground's brax
training wrappers, and observation normalization is a simple running mean/std.

Usage:
    uv run python -m manip_rl.training.ppo --env ManipPickPlace
    # quick CPU sanity run:
    uv run python -m manip_rl.training.ppo --env ManipPickPlace \
        --total-timesteps 300000 --num-envs 128
"""

import argparse
import dataclasses
import json
import pathlib
import pickle
import time
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jp
import optax
from flax.training.train_state import TrainState
from mujoco_playground import wrapper

from manip_rl.envs.registry import load_env


@dataclasses.dataclass
class Args:
    env: str = "ManipPickPlace"
    seed: int = 0
    out: str | None = None

    total_timesteps: int = 5_000_000
    num_envs: int = 256
    num_steps: int = 10            # rollout length per iteration (unroll)
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 1e-3
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    policy_hidden: Sequence[int] = (32, 32, 32, 32)
    value_hidden: Sequence[int] = (256, 256, 256, 256, 256)
    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"

    eval_every: int = 20           # iterations between eval/log lines
    eval_envs: int = 32


class MLP(nn.Module):
    hidden: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h)(x))
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)


class Actor(nn.Module):
    hidden: Sequence[int]
    act_dim: int

    @nn.compact
    def __call__(self, x):
        mean = MLP(self.hidden, self.act_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        return mean, log_std


# --- running observation normalization ---------------------------------------


def norm_init(size):
    return {"mean": jp.zeros(size), "var": jp.ones(size), "count": jp.array(1e-4)}


def norm_update(state, batch):
    """Welford-style running mean/var update over a flattened batch."""
    batch = batch.reshape(-1, batch.shape[-1])
    b_mean, b_var = batch.mean(0), batch.var(0)
    b_count = batch.shape[0]
    delta = b_mean - state["mean"]
    tot = state["count"] + b_count
    new_mean = state["mean"] + delta * b_count / tot
    m_a = state["var"] * state["count"]
    m_b = b_var * b_count
    m2 = m_a + m_b + delta**2 * state["count"] * b_count / tot
    return {"mean": new_mean, "var": m2 / tot, "count": tot}


def norm_apply(state, x):
    return jp.clip((x - state["mean"]) / jp.sqrt(state["var"] + 1e-8), -10, 10)


# --- PPO ----------------------------------------------------------------------


def gaussian_logprob(mean, log_std, action):
    var = jp.exp(2 * log_std)
    logp = -0.5 * ((action - mean) ** 2 / var + 2 * log_std + jp.log(2 * jp.pi))
    return logp.sum(-1)


def train(args: Args):
    run_dir = pathlib.Path(args.out or f"runs/{args.env}-ppo-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_env = load_env(args.env)
    episode_length = raw_env._config.episode_length
    env = wrapper.wrap_for_brax_training(raw_env, episode_length=episode_length)
    act_dim = raw_env.action_size

    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key, reset_key = jax.random.split(key, 4)

    state0 = env.reset(jax.random.split(reset_key, args.num_envs))
    pobs_dim = state0.obs[args.policy_obs_key].shape[-1]
    vobs_dim = state0.obs[args.value_obs_key].shape[-1]

    actor = Actor(tuple(args.policy_hidden), act_dim)
    critic = MLP(tuple(args.value_hidden), 1)
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate),
    )
    actor_ts = TrainState.create(
        apply_fn=actor.apply, params=actor.init(actor_key, jp.zeros(pobs_dim)), tx=tx)
    critic_ts = TrainState.create(
        apply_fn=critic.apply, params=critic.init(critic_key, jp.zeros(vobs_dim)), tx=tx)
    norms = {"policy": norm_init(pobs_dim), "value": norm_init(vobs_dim)}

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    def policy_action(actor_params, norms, obs, rng):
        mean, log_std = actor.apply(actor_params, norm_apply(norms["policy"], obs))
        action = mean + jp.exp(log_std) * jax.random.normal(rng, mean.shape)
        return action, gaussian_logprob(mean, log_std, action)

    @jax.jit
    def training_iteration(carry, _):
        actor_ts, critic_ts, norms, env_state, key = carry

        # --- rollout -----------------------------------------------------------
        def rollout_step(carry, _):
            env_state, key = carry
            key, act_key = jax.random.split(key)
            pobs = env_state.obs[args.policy_obs_key]
            vobs = env_state.obs[args.value_obs_key]
            action, logp = policy_action(actor_ts.params, norms, pobs, act_key)
            value = critic.apply(
                critic_ts.params, norm_apply(norms["value"], vobs)).squeeze(-1)
            next_state = env.step(env_state, jp.clip(action, -1.0, 1.0))
            transition = dict(
                pobs=pobs, vobs=vobs, action=action, logp=logp, value=value,
                reward=next_state.reward, done=next_state.done,
                truncation=next_state.info["truncation"],
            )
            return (next_state, key), transition

        (env_state, key), traj = jax.lax.scan(
            rollout_step, (env_state, key), None, length=args.num_steps)

        # --- normalizer update ---------------------------------------------------
        norms = {
            "policy": norm_update(norms["policy"], traj["pobs"]),
            "value": norm_update(norms["value"], traj["vobs"]),
        }

        # --- GAE -----------------------------------------------------------------
        last_value = critic.apply(
            critic_ts.params,
            norm_apply(norms["value"], env_state.obs[args.value_obs_key]),
        ).squeeze(-1)

        def gae_step(carry, t):
            # Terminal dones cut the bootstrap; truncations (time limits) mask
            # the step out entirely since V(s_final) isn't available (brax
            # convention — the next value belongs to the auto-reset state).
            next_adv, next_value = carry
            termination = traj["done"][t] * (1.0 - traj["truncation"][t])
            trunc_mask = 1.0 - traj["truncation"][t]
            delta = (traj["reward"][t]
                     + args.gamma * next_value * (1.0 - termination)
                     - traj["value"][t]) * trunc_mask
            adv = delta + (args.gamma * args.gae_lambda
                           * (1.0 - termination) * trunc_mask * next_adv)
            return (adv, traj["value"][t]), adv

        (_, _), advantages = jax.lax.scan(
            gae_step, (jp.zeros_like(last_value), last_value),
            jp.arange(args.num_steps - 1, -1, -1))
        advantages = advantages[::-1]
        returns = advantages + traj["value"]

        # --- PPO update -----------------------------------------------------------
        flat = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), {
            "pobs": traj["pobs"], "vobs": traj["vobs"], "action": traj["action"],
            "logp": traj["logp"], "adv": advantages.reshape(batch_size),
            "ret": returns.reshape(batch_size),
        })

        def update_epoch(carry, _):
            actor_ts, critic_ts, key = carry
            key, perm_key = jax.random.split(key)
            idx = jax.random.permutation(perm_key, batch_size)
            idx = idx.reshape(args.num_minibatches, minibatch_size)

            def update_minibatch(carry, mb_idx):
                actor_ts, critic_ts = carry
                mb = jax.tree.map(lambda x: x[mb_idx], flat)

                def actor_loss_fn(params):
                    mean, log_std = actor.apply(
                        params, norm_apply(norms["policy"], mb["pobs"]))
                    logp = gaussian_logprob(mean, log_std, mb["action"])
                    ratio = jp.exp(logp - mb["logp"])
                    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                    pg = -jp.minimum(
                        ratio * adv,
                        jp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv,
                    ).mean()
                    entropy = (log_std + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
                    return pg - args.ent_coef * entropy

                def critic_loss_fn(params):
                    value = critic.apply(
                        params, norm_apply(norms["value"], mb["vobs"])).squeeze(-1)
                    return args.vf_coef * ((value - mb["ret"]) ** 2).mean()

                actor_grads = jax.grad(actor_loss_fn)(actor_ts.params)
                critic_grads = jax.grad(critic_loss_fn)(critic_ts.params)
                return (actor_ts.apply_gradients(grads=actor_grads),
                        critic_ts.apply_gradients(grads=critic_grads)), None

            (actor_ts, critic_ts), _ = jax.lax.scan(
                update_minibatch, (actor_ts, critic_ts), idx)
            return (actor_ts, critic_ts, key), None

        (actor_ts, critic_ts, key), _ = jax.lax.scan(
            update_epoch, (actor_ts, critic_ts, key), None,
            length=args.update_epochs)

        metrics = {
            "reward_per_step": traj["reward"].mean(),
            "done_frac": traj["done"].mean(),
        }
        return (actor_ts, critic_ts, norms, env_state, key), metrics

    # --- eval ---------------------------------------------------------------------
    @jax.jit
    def eval_rollout(actor_params, norms, rng):
        eval_state = env.reset(jax.random.split(rng, args.eval_envs))

        def step(carry, _):
            state = carry
            mean, _ = actor.apply(
                actor_params,
                norm_apply(norms["policy"], state.obs[args.policy_obs_key]))
            state = env.step(state, jp.clip(mean, -1.0, 1.0))
            return state, (state.reward, state.metrics.get("success", state.reward * 0))

        _, (rewards, successes) = jax.lax.scan(
            step, eval_state, None, length=episode_length)
        return rewards.sum(0).mean(), successes.max(0).mean()

    # --- main loop ------------------------------------------------------------------
    print(f"{args.env}: {num_iterations} iterations x {batch_size} steps "
          f"({args.num_envs} envs), policy obs {pobs_dim}, value obs {vobs_dim}")
    carry = (actor_ts, critic_ts, norms, state0, key)
    history = []
    t_start = time.monotonic()
    for it in range(num_iterations):
        carry, metrics = training_iteration(carry, None)
        if it % args.eval_every == 0 or it == num_iterations - 1:
            actor_ts, critic_ts, norms = carry[0], carry[1], carry[2]
            key, eval_key = jax.random.split(carry[4])
            ep_reward, success = eval_rollout(actor_ts.params, norms, eval_key)
            entry = {
                "steps": (it + 1) * batch_size,
                "reward": float(ep_reward),
                "success": float(success),
                "wall_time": time.monotonic() - t_start,
            }
            history.append(entry)
            print(f"[{entry['wall_time']:7.1f}s] steps={entry['steps']:>10,} "
                  f"eval_reward={entry['reward']:8.2f} success={entry['success']:.2f}",
                  flush=True)

    actor_ts, critic_ts, norms = carry[0], carry[1], carry[2]
    payload = {
        "format": "cleanrl_ppo",
        "actor_params": jax.device_get(actor_ts.params),
        "norms": jax.device_get(norms),
        "args": dataclasses.asdict(args),
        "pobs_dim": pobs_dim,
        "act_dim": act_dim,
    }
    with open(run_dir / "policy.pkl", "wb") as f:
        pickle.dump(payload, f)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved policy + history to {run_dir}")
    print(f"Render with: uv run python -m manip_rl.viz.render --env {args.env} "
          f"--policy {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Args):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("policy_hidden", "value_hidden"):
            parser.add_argument(flag, type=int, nargs="+", default=f.default)
        else:
            ftype = type(f.default) if f.default is not None else str
            parser.add_argument(flag, type=ftype, default=f.default)
    train(Args(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
