#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[1/4] Teleop/control pipeline"
python examples/run_fake_pipeline.py

echo "[2/4] Scripted embodied agent"
python scripts/run_agent.py

echo "[3/4] RGB render preview"
python scripts/render_fake_env.py --output outputs/fake_env.ppm

echo "[4/4] Mock VLA evaluation report"
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend mock \
  --num-episodes 3 \
  --randomize-layout \
  --max-steps 100 \
  --write-report \
  --report-dir outputs/eval_reports_smoke

