"""Check which JAX backend is active and benchmark/sanity-check MJX on it.

Usage:
    uv run python scripts/check_gpu.py

On a GPU install, also verifies physics parity against the CPU backend,
since the gfx1030 impersonation (HSA_OVERRIDE_GFX_VERSION) is unofficial.
"""

import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

_TEST_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body pos="0.02 0 0.6">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

NUM_ENVS = 2048
NUM_STEPS = 200


def rollout(device, num_envs: int, num_steps: int):
    """Vmapped MJX rollout on the given device; returns (qpos, seconds)."""
    model = mujoco.MjModel.from_xml_string(_TEST_XML)
    with jax.default_device(device):
        mjx_model = mjx.put_model(model)
        data = mjx.make_data(mjx_model)
        batch = jax.vmap(lambda dq: data.replace(qpos=data.qpos + 0.001 * dq))(
            jp.arange(num_envs, dtype=jp.float32)
        )

        @jax.jit
        def run(d):
            def body(d, _):
                d = jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, d)
                return d, None

            d, _ = jax.lax.scan(body, d, None, length=num_steps)
            return d

        out = jax.block_until_ready(run(batch))  # compile
        t0 = time.perf_counter()
        out = jax.block_until_ready(run(out))
        dt = time.perf_counter() - t0
    return np.asarray(out.qpos), dt


def main():
    print(f"jax {jax.__version__}, default backend: {jax.default_backend()}")
    for d in jax.devices():
        print(f"  device: {d!r}")

    gpus = [d for d in jax.devices() if d.platform != "cpu"]
    cpu = jax.devices("cpu")[0]

    qpos_cpu, t_cpu = rollout(cpu, NUM_ENVS, NUM_STEPS)
    steps = NUM_ENVS * NUM_STEPS
    print(f"\nCPU:  {steps / t_cpu:,.0f} env-steps/s ({t_cpu:.2f}s)")

    if not gpus:
        print(
            "\nNo GPU backend. CPU-only install — this is fine; see docs/rocm_setup.md to enable ROCm."
        )
        return

    qpos_gpu, t_gpu = rollout(gpus[0], NUM_ENVS, NUM_STEPS)
    print(
        f"GPU:  {steps / t_gpu:,.0f} env-steps/s ({t_gpu:.2f}s)  [{t_cpu / t_gpu:.1f}x vs CPU]"
    )

    # Physics parity: identical inputs, loose tolerance (different float orders).
    err = np.max(np.abs(qpos_cpu - qpos_gpu))
    print(f"\nmax |qpos_cpu - qpos_gpu| = {err:.2e}")
    if err > 1e-2 or not np.isfinite(qpos_gpu).all():
        print(
            "WARNING: GPU physics diverges from CPU — do not trust GPU training "
            "(known risk of the gfx version override)."
        )
    else:
        print("GPU physics matches CPU within tolerance. Good to go.")


if __name__ == "__main__":
    main()
