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

