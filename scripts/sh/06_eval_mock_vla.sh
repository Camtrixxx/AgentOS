#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend mock \
  --num-episodes "${NUM_EPISODES:-18}" \
  --randomize-layout \
  --max-steps "${MAX_STEPS:-100}" \
  --write-report

