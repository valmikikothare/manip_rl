"""manip-rl: RL manipulation research platform on MuJoCo Playground/MJX."""

import os

import jax

# Persistent XLA compilation cache: recompiling the Panda scene costs minutes
# on CPU; with the cache, subsequent processes reuse compiled executables.
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax"),
)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
