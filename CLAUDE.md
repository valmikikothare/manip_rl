# CLAUDE.md

Guidance for working in **manip-rl** — an RL manipulation research platform built on
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (MJX/JAX).
The platform handles env/config/training plumbing; the research happens in models and
training paradigms. Embodiments (arm/hand combos), tasks, and trainers are swappable
via config.

## Environment & tooling

- **uv-managed**, Python pinned to **3.12** (`.python-version`); the system Python is
  3.14 and won't work. Always run via `uv run ...`.
- `jax` is pinned to **0.8.\*** in `pyproject.toml`. Do not bump it casually: brax
  (≤0.14) uses APIs removed in jax 0.10, and the `jax-rocm7` plugin wheels target 0.8.x.
- Optional extras: `rocm` (AMD GPU plugin), `sb3` (Gymnasium + Stable-Baselines3).

## Commands

```bash
uv sync                                                          # install
uv run python -m manip_rl.viz.render --env ManipPickPlace        # smoke test → mp4 (random policy)
uv run python -m manip_rl.training.ppo --env ManipPickPlace      # CleanRL-style JAX PPO (primary trainer)
uv run python -m manip_rl.training.ppo_brax --env PandaPickCube  # brax PPO baseline (known-good reference)
uv run python -m manip_rl.training.evaluate --env ManipPickPlace --policy runs/<dir> --video
uv run python -m manip_rl.planning.demo --video                  # RRT approach + grasp demo
```

Quick CPU sanity runs: pass small `--num-envs` / `--total-timesteps` (ppo) or
`--num-timesteps` / `--num-envs` (ppo_brax). There is no test runner; `scripts/test.sh`
and the smoke-test render are the de-facto checks.

**Compilation cost:** the first run of any trainer spends a long time (tens of minutes
on CPU) in XLA compilation before printing progress — this is expected. Prefer the
lighter custom `ppo.py` for iteration; brax PPO on the Panda scene can take 25+ min just
to compile on CPU.

## GPU (AMD ROCm)

This is the dev machine's only GPU path — see `docs/rocm_setup.md` and
`docs/gpu_verification.md`. No NVIDIA, so CUDA-only stacks (Isaac Lab/Gym, ManiSkill GPU
sim, MuJoCo-Warp `impl=warp`) are unusable.

```bash
./scripts/install_rocm_arch.sh        # system prerequisites (sudo; user runs it, then re-login)
uv sync --extra rocm
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 uv run python scripts/check_gpu.py
```

The `HSA_OVERRIDE_GFX_VERSION=10.3.0` (gfx1030 impersonation) is required for the
gfx1032 card.

## Architecture

A `ManipulationEnv` (`mjx_env.MjxEnv`) is fully determined by a **`RobotConfig` + a
`Task`**. Adding an arm/hand combo means writing a `RobotConfig` (+ an MJCF scene);
adding a task means writing a `Task`. No env-class changes either way.

- `src/manip_rl/robots/` — `RobotConfig` (`base.py`): joints, actuator→joint map
  (`ctrl_joints`, length `nu`), `ee_site`, `home_keyframe`, floor-contact sensors, scene
  XML + assets. `panda.py` reuses Playground's Franka scene.
- `src/manip_rl/envs/`
  - `base_env.py` — `ManipulationEnv` and `make_config`. Observations are a dict:
    `"state"` (policy: proprio + object/goal relatives) and `"privileged_state"`
    (critic: state + ground-truth object pose/vel/goal) for **asymmetric actor-critic**.
    Action space is **delta-position** control (`ctrl + action * action_scale`, clipped).
  - `tasks/` — `Task` protocol (`base.py`): `object_body`, `goal_mocap`,
    `sample_object_pos`, `sample_goal`, `reward_terms`, `success`,
    `default_reward_scales`. Tasks are pure descriptions; mutable per-episode state lives
    in the env `info` dict so everything stays jittable. `pick_place.py` is the current
    task.
  - `registry.py` — registers envs into Playground's manipulation registry. Registered
    ids: **`ManipPickPlace`**, **`ManipPickPlaceOrientation`**.
  - `randomize.py` — domain randomization (object physics + actuator gains per world),
    Playground randomizer signature.
- `src/manip_rl/training/`
  - `ppo.py` — **primary** trainer: CleanRL-style single-file JAX PPO (networks, GAE,
    clipped objective, update loop all in one file). Reuses Playground brax wrappers for
    env batching.
  - `ppo_brax.py` — brax PPO baseline / reference path.
  - `evaluate.py` — deterministic eval: success rate, returns, optional video.
  - `checkpoints.py` — save/load policies as `policy(obs, rng) -> action`. Both trainers
    write `policy.pkl`; `load_policy` dispatches on a `"format"` tag
    (`brax_ppo` / `cleanrl_ppo`).
  - `gym_wrapper.py` — `MjxGymEnv`, a single-env CPU Gymnasium adapter for SB3-style
    libs (needs the `sb3` extra). The JAX trainers are the fast path.
- `src/manip_rl/planning/` — classical layer, runs on **CPU MuJoCo outside the JIT
  boundary**. `rrt.py` (joint-space RRT + shortcut smoothing, damped-least-squares IK),
  `base.py` (`GoalSpec`, `Planner` protocol), `hierarchy.py` (`HierarchicalAgent`:
  learned high-level emits a `GoalSpec` → planner re-plans on change → tracks waypoints
  through the env's delta-position actions), `demo.py`.
- `src/manip_rl/viz/render.py` — roll out a policy (or random) to mp4.

**Direction (context, not yet built):** dexterous grasping of complex objects (arm+hand
combos like LEAP) and hierarchical control where a learned high-level policy emits
`GoalSpec`s refined by classical planners with replan-on-change.

## Conventions & gotchas

- **Always load envs via `registry.load_env` / `get_config`, never `registry.load`
  directly.** Playground ≥0.2 defaults `impl` to `"warp"` (NVIDIA-only); `load_env`
  forces `impl="jax"`, which runs on CPU and on ROCm/CUDA via the JAX backend.
- `naconmax` / `naccdmax` / `njmax` config fields are **warp-only** — the JAX impl
  ignores them.
- Keep the GPU path open and prefer CleanRL-style hackable code over framework magic;
  the config-driven swappable-embodiment/task design is deliberate.
- `outputs/`, `runs/`, `checkpoints/`, and `*.mp4` are gitignored (training artifacts).
