#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/collect_demo.py \
  --num-episodes "${NUM_EPISODES:-60}" \
  --output-dir "${OUTPUT_DIR:-data/demos}" \
  ${RANDOMIZE_LAYOUT:+--randomize-layout}

