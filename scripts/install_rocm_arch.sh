#!/usr/bin/env bash
# Install ROCm prerequisites for JAX on Arch Linux (see docs/rocm_setup.md).
# Idempotent; uses sudo for pacman and group changes.
set -euo pipefail

echo "==> System ROCm version available in repos:"
pacman -Si rocm-hip-sdk | grep -E '^(Repository|Version)' || {
    echo "rocm-hip-sdk not found in repos; enable the [extra] repository." >&2
    exit 1
}
echo
echo "    NOTE: the jax-rocmN plugin major version in pyproject.toml [rocm]"
echo "    must match the ROCm major version above (rocm7 wheels <-> ROCm 7.x)."
echo

echo "==> Installing ROCm packages (HIP runtime/SDK, diagnostics)..."
sudo pacman -S --needed rocm-hip-sdk rocm-smi-lib rocminfo

echo "==> Adding $USER to render and video groups..."
sudo usermod -aG render,video "$USER"

echo "==> Checking GPU visibility (may fail until reboot/re-login):"
rocminfo 2>/dev/null | grep -E 'Marketing Name|gfx' | head -8 || \
    echo "    rocminfo failed - expected before re-login/reboot."

cat <<'EOF'

Done. Next steps:
  1. Log out and back in (group membership), reboot if drivers were new.
  2. Persist environment variables (e.g. ~/.config/environment.d/rocm.conf):
       HSA_OVERRIDE_GFX_VERSION=10.3.0
       ROCR_VISIBLE_DEVICES=0
  3. uv sync --extra rocm
  4. HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python scripts/check_gpu.py
EOF
