"""manip-rl: RL manipulation research platform on MuJoCo Playground/MJX."""

import os

# ROCm/RDNA2 fix (must run before `import jax`, i.e. before the XLA backend is
# created and reads XLA_FLAGS): XLA captures jitted scan/while bodies into HIP
# graphs (command buffers), but HIP-graph stream updates are broken on this
# stack — they raise "UpdateStreams failed" (hip_graph_internal.cpp) and SIGSEGV
# the process the moment a graph-captured step runs (e.g. the brax PPO training
# step; plain eval, which isn't graph-captured, runs fine). Disabling command
# buffers stops graph capture. No-op on CPU/CUDA, so it's always safe to set.
if "xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()

import jax

# Persistent XLA compilation cache: recompiling the Panda scene costs minutes
# on CPU; with the cache, subsequent processes reuse compiled executables.
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax"),
)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
