# Post-ROCm verification checklist

Run these in order once `scripts/install_rocm_arch.sh` has been run (and you've
re-logged-in). Everything was verified mechanically on CPU on 2026-06-09;
this checklist proves the same code paths on GPU and does the training
verification that was too slow for CPU.

```bash
# 0. Env vars (persist these; see docs/rocm_setup.md)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCR_VISIBLE_DEVICES=0

# 1. Backend + physics parity (GPU vs CPU, catches silent gfx-override breakage)
uv sync --extra rocm
uv run python scripts/check_gpu.py

# 2. Env smoke test on GPU
uv run python -m manip_rl.viz.render --env ManipPickPlace

# 3. Short training run — reward should trend upward within a few minutes
uv run python -m manip_rl.training.ppo --env ManipPickPlace \
    --total-timesteps 2000000 --num-envs 1024

# 4. Evaluate + render the checkpoint it saves (checkpoints at every eval)
uv run python -m manip_rl.training.evaluate --env ManipPickPlace \
    --policy runs/<run-dir> --episodes 20 --video

# 5. brax PPO reference on the same env (parity check vs step 3)
uv run python -m manip_rl.training.ppo_brax --env ManipPickPlace

# 6. Hierarchical demo with the trained policy as the grasp phase
uv run python -m manip_rl.planning.demo --policy runs/<run-dir> --video
```

Expected outcomes:
- Step 1 prints a GPU device, a speedup factor, and "physics matches CPU".
  If it warns about divergence, do not trust GPU training (gfx1032 override risk).
- Step 3: `eval_reward` should climb well above the ~110-130 an untrained
  policy gets from shaping; `success` should become nonzero as transport learns.
- Steps 3 vs 5 should reach broadly similar rewards (different algorithms/
  hyperparameters, so trends matter, not exact numbers).
- If JAX falls back to CPU silently, `scripts/check_gpu.py` is the tell —
  rerun it whenever performance seems off.
