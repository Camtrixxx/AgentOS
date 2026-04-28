#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/collect_vision_demo.py \
  --num-episodes "${NUM_EPISODES:-120}" \
  --output-dir "${OUTPUT_DIR:-data/vision_demos_random}" \
  --randomize-layout

