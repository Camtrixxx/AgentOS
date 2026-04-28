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
