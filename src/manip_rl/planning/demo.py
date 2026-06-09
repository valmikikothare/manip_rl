"""Hierarchical demo: RRT-planned approach + scripted (or learned) grasp.

The scripted high-level emits a pregrasp GoalSpec above the object; the RRT
planner produces a collision-free arm trajectory; tracking executes it through
the env's action space. After the approach, control hands off to a grasp
policy: a trained checkpoint if --policy is given, otherwise a scripted
close-and-lift.

Usage:
    uv run python -m manip_rl.planning.demo [--policy runs/<dir>] [--video]
"""

import argparse

import jax
import numpy as np

from manip_rl.envs.registry import load_env
from manip_rl.planning.base import GoalSpec
from manip_rl.planning.hierarchy import HierarchicalAgent
from manip_rl.planning.rrt import RRTPlanner
from manip_rl.robots.panda import PANDA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="ManipPickPlace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default=None, help="grasp policy checkpoint dir")
    parser.add_argument("--approach-steps", type=int, default=80)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    env = load_env(args.env)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(args.seed))
    obj_pos0 = np.asarray(state.data.xpos[env.obj_body_id])

    def high_level(obs, t):
        if t < args.approach_steps:
            # Pregrasp: hover above the object's initial position.
            return GoalSpec(ee_pos=obj_pos0 + np.array([0.0, 0.0, 0.12]))
        return None  # hand off to low-level grasp policy

    if args.policy:
        from manip_rl.training.checkpoints import load_policy
        low_level = load_policy(args.policy, env)
    else:
        def low_level(obs, rng):
            # Scripted: descend, close the gripper, lift.
            action = np.zeros(env.action_size)
            action[-1] = -1.0  # close gripper (panda: last actuator)
            action[3] = 0.3    # crude lift via elbow — demo placeholder
            return action

    agent = HierarchicalAgent(
        env=env, robot=PANDA, planner=RRTPlanner(PANDA, seed=args.seed),
        high_level=high_level, low_level=low_level,
    )

    rng = jax.random.PRNGKey(args.seed + 1)
    states, total_reward = [state], 0.0
    pregrasp = obj_pos0 + np.array([0.0, 0.0, 0.12])
    approach_err = None
    for t in range(env._config.episode_length):
        rng, act_rng = jax.random.split(rng)
        action = agent.act(state, t, act_rng)
        state = jit_step(state, jax.numpy.asarray(action, dtype=jax.numpy.float32))
        states.append(state)
        total_reward += float(state.reward)
        if t == args.approach_steps - 1:
            ee = np.asarray(state.data.site_xpos[env.ee_site_id])
            approach_err = np.linalg.norm(ee - pregrasp)
        if bool(state.done):
            break

    print(f"episode return: {total_reward:.2f}")
    print(f"approach: ee-to-pregrasp error {approach_err:.3f} m "
          f"({'OK' if approach_err is not None and approach_err < 0.05 else 'MISSED'})")
    print(f"plan active at end: {agent.plan_active}")

    if args.video:
        import mediapy as media
        out = "outputs/hierarchical_demo.mp4"
        media.write_video(out, env.render(states), fps=1.0 / env.dt)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
