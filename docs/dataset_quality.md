# Dataset Quality Workflow

Before fine-tuning SmolVLA or comparing policies, inspect the demonstration data.

## Inspect Vision Demos

```bash
python scripts/inspect_dataset.py \
  --data-dir data/vision_demos_random \
  --expect-images \
  --output-dir outputs/dataset_quality_random
```

Outputs:

```text
outputs/dataset_quality_random/dataset_quality.json
outputs/dataset_quality_random/dataset_quality.md
```

The report includes:

- number of episodes
- number of transitions
- success rate
- target color distribution
- action min/max/mean/std
- missing images
- bad action shapes
- missing transition keys

## Export LeRobot-Style Manifest

```bash
python scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir data/lerobot_fake_manipulation \
  --format manifest
```

Native LeRobot export is gated until a LeRobot version is installed and pinned:

```bash
python scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir data/lerobot_fake_manipulation_native \
  --format native
```

## Small Policy Benchmark

```bash
python scripts/benchmark_policies.py \
  --num-episodes 3 \
  --max-steps 100 \
  --output-dir outputs/policy_benchmark
```

This evaluates:

- scripted
- mock VLA
- SmolVLA dry-run
- RL scripted baseline
- RL random baseline

