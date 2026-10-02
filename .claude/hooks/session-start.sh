#!/bin/bash
# Prepares a Claude Code cloud session: installs dependencies and the vgpu CPU software renderer so
# tests, shader validation and the browser smoke test can run without a physical GPU.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Install dependencies (pnpm, as pinned by "packageManager"; the container caches the result).
pnpm install --frozen-lockfile

# One-off download of the CPU Vulkan renderer into ~/.cache/vgpu (skipped when already present).
npx --no-install vgpu install-software-renderer

# Make headless GPU tests fail loudly (instead of skipping) when no adapter is available.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export REQUIRE_GPU=1' >> "$CLAUDE_ENV_FILE"
fi
