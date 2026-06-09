# ROCm setup on Arch Linux (RX 6650 XT / 6800S, gfx1032)

Goal: run JAX (and therefore MJX / Playground training) on the AMD dGPU.
Everything in this repo works on CPU without any of this; ROCm is purely an
acceleration path.

## The catch, stated up front

- The dGPU is **gfx1032** (Navi 23, RDNA2). ROCm does **not** officially support
  it. The standard workaround is to impersonate gfx1030 (which *is* supported)
  via `HSA_OVERRIDE_GFX_VERSION=10.3.0`. This works for many users on RDNA2
  cards but is unofficial — kernel-level crashes or wrong results are possible.
  Verify with the parity check in `scripts/check_gpu.py` before trusting
  training runs.
- JAX ROCm plugin wheels are built for **specific jax versions and ROCm major
  versions**. Installing the `[rocm]` extra will downgrade/pin `jax` to whatever
  the plugin requires. Check compatibility at
  <https://github.com/ROCm/rocm-jax/releases> and
  <https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html>.
- Arch's ROCm packages track upstream loosely. The plugin's ROCm major version
  (currently `rocm7`) must match the system ROCm major version
  (`pacman -Si rocm-hip-sdk` → Version). If Arch is still on ROCm 6.x, use the
  `jax-rocm6-*` wheels instead (edit the `[rocm]` extra in `pyproject.toml`).

## 1. System prerequisites

Run the script (reviews before running encouraged — it uses sudo):

```bash
./scripts/install_rocm_arch.sh
```

What it does:

| Step | Why |
|---|---|
| `pacman -S rocm-hip-sdk rocm-smi-lib rocminfo` | HIP runtime + libs JAX's PJRT plugin dlopens; `rocminfo`/`rocm-smi` for diagnostics |
| add user to `render` and `video` groups | unprivileged access to `/dev/kfd` and `/dev/dri/*` |
| prints the env vars to persist | see below |

Then **log out and back in** (group membership) and reboot if `amdgpu` was just
installed.

## 2. Environment variables

Persist these (e.g. in `~/.config/environment.d/rocm.conf`, your shell rc, or a
project `.envrc`):

```bash
# Impersonate gfx1030 so ROCm accepts the gfx1032 card
export HSA_OVERRIDE_GFX_VERSION=10.3.0
# Target only the dGPU; the 680M iGPU (gfx1035) confuses ROCm device enumeration
export ROCR_VISIBLE_DEVICES=0
# Headless GPU rendering for mujoco (optional, for rendering not physics)
export MUJOCO_GL=egl
```

If `rocminfo` lists the iGPU first, swap the index in `ROCR_VISIBLE_DEVICES`
(check with `rocminfo | grep -A5 'Marketing Name'`).

## 3. Python side

```bash
uv sync --extra rocm
```

If resolution fails because PyPI doesn't host the plugin wheels for your
combo, install from AMD's index instead:

```bash
uv pip install jax jax-rocm7-plugin jax-rocm7-pjrt \
  --find-links https://github.com/ROCm/rocm-jax/releases
# or wheels from https://repo.radeon.com (see AMD's JAX-on-ROCm docs)
```

## 4. Verify

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python scripts/check_gpu.py
```

The script:
1. prints `jax.devices()` — expect a `RocmDevice`/`gpu` entry, not just `CpuDevice`;
2. steps a small MJX model and times it on the default backend vs CPU;
3. compares GPU vs CPU physics outputs (loose tolerance) — a sanity check that
   the gfx1030 impersonation isn't silently producing garbage.

## 5. If it doesn't work

- `rocminfo` errors / no `gfx*` agent → kernel/driver issue: check `dmesg | grep -i kfd`,
  confirm `render` group, reboot.
- JAX falls back to CPU with a plugin load error → ROCm version mismatch between
  system and wheels; align major versions (section 0).
- GPU hangs/resets mid-training → known failure mode of unofficial gfx overrides;
  nothing to tune — fall back to CPU locally and treat GPU runs as
  better-hardware-later (the code paths are identical).
- Docker escape hatch: AMD publishes prebuilt JAX-ROCm images
  (`rocm/jax`, see <https://hub.docker.com/r/rocm/jax>) — only the kernel driver
  must work on the host.

Sources: [AMD JAX-on-ROCm install docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html),
[ROCm/rocm-jax](https://github.com/ROCm/rocm-jax),
[gfx1032-on-Arch gist](https://gist.github.com/viraj-s15/9a43f10b0f937d6787e2756cc7358bb1),
[AMD MuJoCo Playground on ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/rocm-jax-mujoco/README.html).
