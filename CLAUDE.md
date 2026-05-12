# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A simulation-first embodied AI lab with two parallel pipelines:

1. **Teleop → Control**: synthetic hand keypoints → stereo triangulation → hand retargeting → safety limiter → fake robot backend
2. **Embodied Agent Loop**: language instruction → RGB image + state observation → policy/VLA backend → action [dx, dy, gripper] → fake manipulation env → evaluation report

No real robot or simulator required. No package manager — pure Python scripts with `numpy` dependency. Tests use `pytest`.

## Quick Verification

```bash
bash scripts/sh/00_smoke_test.sh          # full sanity check
python examples/run_fake_pipeline.py       # teleop → control demo
python scripts/run_agent.py                # scripted embodied agent
python scripts/render_fake_env.py --output outputs/fake_env.ppm  # RGB render
```

## Common Commands

```bash
# Collect demos → train → evaluate (substitute env vars to override defaults)
NUM_EPISODES=30 bash scripts/sh/03_collect_vision_demos_random.sh
EPOCHS=40 bash scripts/sh/04_train_vision_bc_random.sh
CHECKPOINT=checkpoints/vision_bc_random_policy.pt bash scripts/sh/05_eval_vision_bc_random.sh

# Evaluate any policy with the unified entry point
python learning/evaluate_policy.py --policy scripted --write-report
python learning/evaluate_policy.py --policy bc --checkpoint checkpoints/bc_policy.pt --write-report
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_random_policy.pt --write-report
python learning/evaluate_policy.py --policy vla --vla-backend mock --write-report

# Clean generated artifacts (data/, outputs/, checkpoints/)
bash scripts/sh/99_clean_generated.sh
```

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_safety_limiter.py -v   # single test file
```

## Architecture

### Agent loop (`agent/agent_loop.py`)

Central orchestration. `AgentLoop(env, policy, recorder?)` runs environment rollouts. All policies implement the `Policy` protocol (single `act(observation) -> np.ndarray` method), and all envs implement `Env` (reset/step). A `recorder` can be attached to capture transitions.

### Policies (`agent/`)

| Policy | File | What it does |
|--------|------|-------------|
| `ScriptedPickPlacePolicy` | `scripted_policy.py` | Hardcoded expert for data collection |
| `BCPolicy` | `bc_policy.py` | State-only BC model (no vision) |
| `VisionBCPolicy` | `vision_bc_policy.py` | BC with CNN image encoder |
| `VLAPolicy` | `vla_policy.py` | Delegates to a `VLABackend` via `FakeEnvVLAAdapter` |

### VLA-ready interface (`vla/`, `adapters/`)

`VLABackend` is a Protocol with one method: `predict(observation: VLAObservation) -> VLAAction`. `FakeEnvVLAAdapter` translates `FakeManipulationEnv` observations/actions to the VLA contract. To add a real VLA model, implement a new backend following the same `predict` signature.

### Fake environment (`envs/fake_manipulation_env.py`)

2D pick-and-place: 3 colored blocks, 1 bowl. Actions are `[dx, dy, gripper]`. Supports RGB rendering, randomized layouts, and language instructions via `TaskSpec(instruction, target_color)`.

### Learning pipeline (`learning/`)

- `demo_dataset.py` / `vision_demo_dataset.py` — state and vision dataset loaders
- `features.py` — feature extraction (end-effector position, object positions)
- `models.py` / `vision_models.py` — MLP and CNN policy networks
- `train_bc.py` — train state BC from demonstrations
- `train_vision_bc.py` — train VisionBC from vision demonstrations
- `evaluate_policy.py` — unified evaluation entry point for all policy types

### Teleop-to-control pipeline

- `perception/stereo_triangulation.py` — stereo 3D reconstruction from 2D keypoints
- `retargeting/simple_hand_retargeter.py` — human hand skeleton → robot joint commands
- `control/safety_limiter.py` — joint limit clamping and delta limiting
- `control/fake_robot_backend.py` — no-hardware robot backend
- `kinematics/ik_solver.py` — placeholder IK solver (shape only)

### Generated directories (gitignored)

`data/`, `outputs/`, `checkpoints/` — all runtime artifacts. Clean with `scripts/sh/99_clean_generated.sh`.

## Key Design Conventions

- All modules use `__future__ import annotations` and type hints throughout
- `sys.path.insert(0, str(PROJECT_ROOT))` at the top of entry points; imports use project-root-relative paths (e.g., `from agent.agent_loop import AgentLoop`)
- Configuration via frozen dataclasses (e.g., `FakeManipulationConfig`, `SafetyConfig`)
- Seeds are passed explicitly to constructors (no global RNG state)
- The `Policy` and `Env` protocols in `agent_loop.py` accept any duck-typed object, so policies and envs don't formally subclass anything
