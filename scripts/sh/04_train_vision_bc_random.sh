#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python learning/train_vision_bc.py \
  --data-dir "${DATA_DIR:-data/vision_demos_random}" \
  --epochs "${EPOCHS:-120}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --output "${OUTPUT:-checkpoints/vision_bc_random_policy.pt}"

