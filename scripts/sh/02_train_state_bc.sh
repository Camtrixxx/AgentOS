#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python learning/train_bc.py \
  --data-dir "${DATA_DIR:-data/demos}" \
  --epochs "${EPOCHS:-300}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --output "${OUTPUT:-checkpoints/bc_policy.pt}"

