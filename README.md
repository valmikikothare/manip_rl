# manip-rl

RL manipulation research platform built on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (MJX/JAX).
Starts with Franka Panda pick-and-place; designed so embodiments (arm/hand combos),
tasks, and trainers are swappable via config.

## Quick start (CPU)

```bash
uv sync
uv run python -m manip_rl.viz.render --env PandaPickCube        # smoke test → mp4
uv run python -m manip_rl.training.ppo_brax --env PandaPickCube # baseline PPO
```

## GPU (AMD ROCm)

See [docs/rocm_setup.md](docs/rocm_setup.md). Short version:

```bash
./scripts/install_rocm_arch.sh        # system prerequisites (needs sudo)
# re-login, then:
uv sync --extra rocm
HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python scripts/check_gpu.py
```

## Layout

- `src/manip_rl/robots/` — embodiment configs (`RobotConfig`)
- `src/manip_rl/envs/` — `MjxEnv` task environments + Playground registry glue
- `src/manip_rl/training/` — CleanRL-style JAX PPO (primary), brax PPO, Gymnasium adapter
- `src/manip_rl/planning/` — classical planners (RRT*) + hierarchical policy/planner agent
- `src/manip_rl/viz/` — rollout rendering
- `experiments/configs/` — experiment configs
- `notebooks/` — reference notebook exports
