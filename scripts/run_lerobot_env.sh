#!/usr/bin/env bash
set -euo pipefail

LEROBOT_VENV="${LEROBOT_VENV:-/workspace/hyh/.venvs/lerobot-smolvla}"
PYTHONPATH= "${LEROBOT_VENV}/bin/python" "$@"
