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
python scripts/run_fake_pipeline.py        # teleop → control demo
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
python learning/evaluate_policy.py --policy vla --vla-backend smolvla_dry_run --write-report
python learning/evaluate_policy.py --policy rl --rl-backend scripted --write-report

# RL training (requires stable-baselines3)
python scripts/train_rl.py --backend sb3 --timesteps 10000 --randomize-layout

# Dataset inspection and LeRobot export
python scripts/inspect_dataset.py --data-dir data/vision_demos_random
python scripts/export_lerobot_dataset.py --data-dir data/vision_demos_random --output-dir outputs/lerobot_export

# Policy benchmark across all backends
python scripts/benchmark_policies.py --num-episodes 3

# Tool-based agent with file-backed workspace protocol
python scripts/run_tool_agent.py "pick up the red block and place it in the bowl"
python scripts/run_planner_agent.py "pick up the green block and place it in the bowl" --randomize-layout

# Standalone watchdog polling ACTION.md
python scripts/run_watchdog.py --once

# Clean generated artifacts (data/, outputs/, checkpoints/, workspace/)
bash scripts/sh/99_clean_generated.sh
```

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_safety_limiter.py -v   # single test file
```

## Architecture

### Agent loop (`agent/agent_loop.py`)

Central orchestration. `run_episode(env, policy, *, recorder, task, max_steps)` runs environment rollouts. All policies implement the `Policy` protocol (single `act(observation) -> np.ndarray` method), and all envs implement `Env` (reset/step). A `recorder` can be attached to capture transitions.

### Policies (`agent/`)

| Policy | File | What it does |
|--------|------|-------------|
| `ScriptedPickPlacePolicy` | `scripted_policy.py` | Hardcoded expert for data collection |
| `BCPolicy` | `bc_policy.py` | State-only BC model (no vision) |
| `VisionBCPolicy` | `vision_bc_policy.py` | BC with CNN image encoder |
| `VLAPolicy` | `vla_policy.py` | Delegates to a `VLABackend` via `FakeEnvVLAAdapter` |
| `RLPolicy` | `rl_policy.py` | Wraps scripted/random/SB3 backends behind the Policy protocol |

### VLA-ready interface (`agent/`, `hal/`)

`VLABackend` is a Protocol with one method: `predict(observation: VLAObservation) -> VLAAction`. `FakeEnvVLAAdapter` translates `FakeManipulationEnv` observations/actions to the VLA contract. Three backends exist:
- `MockVLABackend` — deterministic mock, always moves toward the target color
- `SmolVLABackend` — loads LeRobot SmolVLA; falls back to mock on import/load failure. Supports `dry_run=True` to test the integration point without the model

### Fake environment (`envs/fake_manipulation_env.py`)

2D pick-and-place: 3 colored blocks, 1 bowl. Actions are `[dx, dy, gripper]`. Supports RGB rendering, randomized layouts, and language instructions via `TaskSpec(instruction, target_color)`.

### Learning pipeline (`learning/`)

- `demo_dataset.py` / `vision_demo_dataset.py` — state and vision dataset loaders
- `features.py` — feature extraction (end-effector position, object positions)
- `models.py` / `vision_models.py` — MLP and CNN policy networks
- `train_bc.py` — train state BC from demonstrations
- `train_vision_bc.py` — train VisionBC from vision demonstrations
- `evaluate_policy.py` — unified evaluation entry point for all policy types
- `devices.py` — resolve `cpu`/`cuda`/`npu`/`auto` torch devices (auto-detects NPU)

### RL integration (`learning/`)

`FakeManipulationGymEnv` wraps `FakeManipulationEnv` as a Gymnasium-style env (reset/step/render) without requiring the `gymnasium` package. Used by `scripts/train_rl.py` with stable-baselines3 PPO.

### HAL — Hardware Abstraction Layer (`hal/`)

`BaseDriver` is an ABC defining the driver contract: `load_environment()`, `execute_action()`, `get_environment()`, `get_runtime_state()`, plus connect/disconnect/health_check lifecycle methods. `FakeManipulationDriver` implements it over `FakeManipulationEnv` and is the runtime's single point of contact with the environment.

Drivers are registered in `hal/drivers.py` (`register_driver`, `load_driver`, `list_drivers`). The watchdog merges `get_runtime_state()` (connection, health, step progress) into ENVIRONMENT.md after each poll cycle.

Also contains the teleop-to-control pipeline: `stereo_triangulation.py`, `simple_hand_retargeter.py`, `safety_limiter.py`, `fake_robot_backend.py`, `ik_solver.py`, and the VLA adapter (`vla_adapter.py`).

### Runtime system (`runtime/`)

A file-backed embodied agent runtime using a workspace of Markdown files as the inter-module protocol:

```
workspace/
├── ACTION.md       # JSON-fenced action queue; watchdog consumes first pending item
├── ENVIRONMENT.md  # JSON-fenced environment state (robot, objects, receptacles, episode, runtime)
├── EMBODIED.md     # Static driver profile (supported actions, constraints)
├── LESSONS.md      # Human-readable failure log for post-mortem
├── TASK.md         # Current task state
├── SKILL.md        # Reusable workflow recipes
├── PLAN.md         # Task plan document
└── REPORT.md       # Execution report
```

Key components:
- **Planner** (`planner.py`) — `RuleBasedPlanner` produces a `TaskPlan` of `PlannedStep`s (reset_task, scripted_pick_place_loop, render)
- **Executor** (`executor.py`) — runs a `TaskPlan` using `run_episode()` directly with an env and policy, then renders and writes a REPORT.md
- **Watchdog** (`watchdog.py`) — polls `ACTION.md` for pending actions, validates them via `ActionValidator`, executes through the HAL driver, writes results back
- **Action queue** (`action_queue.py`) — JSON-fenced Markdown queue with normalize/append/poll/save primitives
- **Action validator** (`action_validator.py`) — validates action type, parameter shape, workspace bounds, and step delta limits
- **Environment I/O** (`environment_io.py`) — bi-directional conversion between env observations and the `ENVIRONMENT.md` document

### Tools system (`tools/`)

Plug-in tools implementing a `Tool` protocol (`name`, `description`, `run(parameters) -> ToolResponse`). Registered in `ToolRegistry` with optional tracing. Tools are the runtime's composable action units:

| Tool | What it does |
|------|-------------|
| `ReadEnvironmentTool` | Loads `ENVIRONMENT.md` |
| `AppendActionTool` | Validates and queues an action in `ACTION.md` |
| `RunWatchdogOnceTool` | Executes the first pending action |
| `ResetTaskTool` | Chains append_action → watchdog for `reset` |
| `StepEnvTool` | Chains append_action → watchdog for `env_step` |
| `RenderFakeEnvTool` | Renders the environment to PPM |
| `CreatePlanTool` | Writes a `TaskPlan` to `PLAN.md` |
| `EvaluateScriptedPolicyTool` | Runs scripted policy evaluation |

### Teleop-to-control pipeline (`hal/`)

- `hal/stereo_triangulation.py` — stereo 3D reconstruction from 2D keypoints
- `hal/simple_hand_retargeter.py` — human hand skeleton → robot joint commands
- `hal/safety_limiter.py` — joint limit clamping and delta limiting
- `hal/fake_robot_backend.py` — no-hardware robot backend
- `hal/ik_solver.py` — placeholder IK solver (shape only)
- `hal/vla_adapter.py` — observation/action adapters for VLA backends

### Datasets (`datasets/`)

- `episode_recorder.py` / `vision_episode_recorder.py` — record episodes to `data/`
- `inspector.py` — validate dataset quality (missing images, bad shapes, success rate, action stats)
- `lerobot_exporter.py` — export to LeRobot JSONL manifest and optional native `LeRobotDataset` format

### Generated directories (gitignored)

`data/`, `outputs/`, `checkpoints/`, `workspace/` — all runtime artifacts. Clean with `scripts/sh/99_clean_generated.sh`.

## Key Design Conventions

- All modules use `__future__ import annotations` and type hints throughout
- `sys.path.insert(0, str(PROJECT_ROOT))` at the top of entry points; imports use project-root-relative paths (e.g., `from agent.agent_loop import run_episode`)
- Configuration via frozen dataclasses (e.g., `FakeManipulationConfig`, `SafetyConfig`)
- Seeds are passed explicitly to constructors (no global RNG state)
- The `Policy` and `Env` protocols in `agent_loop.py` accept any duck-typed object, so policies and envs don't formally subclass anything
