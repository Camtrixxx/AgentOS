#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "This removes generated local artifacts: data/, outputs/, checkpoints/."
echo "Source code and docs are not removed."

rm -rf data outputs checkpoints

echo "Generated artifacts removed."

