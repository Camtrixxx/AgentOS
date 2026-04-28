# Episode Dataset Schema

Each recorded rollout is stored as one directory:

```text
data/demos/episode_000000/
├── metadata.json
├── transitions.jsonl
└── arrays.npz
```

## `metadata.json`

```json
{
  "initial_observation": {},
  "summary": {
    "steps": 31,
    "total_reward": 0.7,
    "success": true
  },
  "num_steps": 31
}
```

## `transitions.jsonl`

One JSON object per environment step:

```json
{
  "observation": {},
  "action": [0.03, 0.02, 1.0],
  "reward": -0.01,
  "next_observation": {},
  "done": false,
  "info": {}
}
```

## `arrays.npz`

Contains dense arrays for quick training scripts:

- `actions`: shape `(T, action_dim)`
- `rewards`: shape `(T,)`
- `dones`: shape `(T,)`

This simple format is intentionally close to imitation learning and VLA fine-tuning datasets. Later versions can add camera images, robot proprioception, language embeddings, and simulator states.

The fake environment can already produce RGB observations through `env.render_rgb()` or `FakeManipulationConfig(include_image=True)`. The current recorder stores observations as JSON-compatible data, so image recording should be treated as a lightweight debugging mode until a dedicated image dataset writer is added.

## Current BC Feature Format

The first behavior cloning baseline does not train from raw JSON observations directly. It converts each observation into a compact state vector in `learning/features.py`:

```text
[
  ee_x, ee_y,
  target_x, target_y,
  bowl_x, bowl_y,
  target_minus_ee_x, target_minus_ee_y,
  bowl_minus_ee_x, bowl_minus_ee_y,
  gripper_closed,
  holding_target,
  target_color_red,
  target_color_blue,
  target_color_green
]
```

The action target remains:

```text
[dx, dy, gripper]
```

## Vision Demo Format

Vision demonstrations are stored separately under `data/vision_demos`:

```text
data/vision_demos/episode_000000/
├── images/
│   ├── 000000.npy
│   ├── 000000_next.npy
│   └── ...
├── metadata.json
├── transitions.jsonl
└── arrays.npz
```

Each image file is an RGB `uint8` NumPy array with shape `(128, 128, 3)`. `transitions.jsonl` stores `image_path` and `next_image_path` instead of embedding image arrays directly.

For generalization experiments, collect demonstrations with randomized layouts:

```bash
python scripts/collect_vision_demo.py --num-episodes 120 --randomize-layout
```
