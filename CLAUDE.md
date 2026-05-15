# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project

This repository is now focused on a simulation-first embodied AgentOS runtime:

```text
language task
-> planner
-> tools
-> ACTION.md
-> watchdog + validator
-> HAL driver
-> ENVIRONMENT.md
-> REPORT.md / TRACE.jsonl / LESSONS.md
```

The project started as a teleop-control lab, but the old synthetic teleop runtime has been removed from the main code path. Historical reference material remains under `docs/reference/`.

## Quick Verification

```bash
bash scripts/sh/00_smoke_test.sh
python scripts/run_agentos.py "pick up the red block and place it in the bowl"
python scripts/render_fake_env.py --output outputs/fake_env.ppm
python -m pytest tests/ -v
```

## Common Commands

```bash
# Main file-backed AgentOS runtime
python scripts/run_agentos.py "pick up the green block and place it in the bowl" \
  --workspace workspace/agentos_demo \
  --render-output outputs/agentos_demo.ppm

# Optional DeepSeek LLM planner. Do not commit API keys.
export DEEPSEEK_API_KEY="sk-..."
python scripts/run_agentos.py "pick up the blue block and place it in the bowl" --planner deepseek

# Skill library planner backed by runtime/skills/default_skill.md
python scripts/run_agentos.py "pick up the blue block and place it in the bowl" --planner skill

# Multi-episode full AgentOS benchmark
python scripts/benchmark_agentos.py --planner skill --num-episodes 9 --randomize-layout

# Direct policy debug path
python scripts/run_agent.py

# Standalone watchdog consuming ACTION.md
python scripts/run_watchdog.py --once

# Collect demos -> train -> evaluate
NUM_EPISODES=30 bash scripts/sh/03_collect_vision_demos_random.sh
EPOCHS=40 bash scripts/sh/04_train_vision_bc_random.sh
CHECKPOINT=checkpoints/vision_bc_random_policy.pt bash scripts/sh/05_eval_vision_bc_random.sh

# Unified policy evaluation
python learning/evaluate_policy.py --policy scripted --write-report
python learning/evaluate_policy.py --policy bc --checkpoint checkpoints/bc_policy.pt --write-report
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_random_policy.pt --write-report
python learning/evaluate_policy.py --policy vla --vla-backend mock --write-report
python learning/evaluate_policy.py --policy rl --rl-backend scripted --write-report

# Dataset inspection and LeRobot export
python scripts/inspect_dataset.py --data-dir data/vision_demos_random
python scripts/export_lerobot_dataset.py --data-dir data/vision_demos_random --output-dir outputs/lerobot_export

# Clean generated artifacts
bash scripts/sh/99_clean_generated.sh
```

## Architecture

### Runtime (`runtime/`)

The core AgentOS runtime is file-backed. Workspace files are Markdown documents with JSON fenced payloads:

```text
workspace/
├── ACTION.md       # action queue; watchdog consumes first pending item
├── ENVIRONMENT.md  # task, robot, scene, episode, runtime state
├── EMBODIED.md     # driver profile and constraints
├── LESSONS.md      # failure notes
├── TASK.md         # current task summary
├── SKILL.md        # reusable workflow notes
├── PLAN.md         # planner output
└── REPORT.md       # execution report
```

Key runtime modules:

- `executor.py` — `AgentOSExecutor` runs `TaskPlan`s through `ToolRegistry`
- `watchdog.py` — validates and executes queued actions through a HAL driver
- `action_queue.py` — action lifecycle, markdown parsing, status transitions
- `repository.py` — workspace I/O and locked `ACTION.md` read-modify-write
- `file_watcher.py` — inotify/mtime file change watcher for watchdog wakeups
- `action_validator.py` — validates action schema, step delta, workspace bounds
- `environment_io.py` — converts observations to/from `ENVIRONMENT.md`
- `skill_planner.py` — markdown-backed workflow templates for repeatable plans
- `llm_planner.py` — optional DeepSeek workflow planner with local fallback

### HAL (`hal/`)

`BaseDriver` is the execution boundary. It includes:

- `CommandDriver`, `QueryDriver`, `RuntimeDriver` protocol boundaries
- driver state machine: `disconnected`, `idle`, `executing`, `fault`, `closed`
- `execute_action()` template method that wraps `_execute_action()`
- capabilities and runtime state surfaced into `ENVIRONMENT.md`

Current driver:

- `FakeManipulationDriver` — drives `FakeManipulationEnv` through the AgentOS protocol

### Tools (`tools/`)

Tools implement `run(parameters) -> ToolResponse` and are registered in `ToolRegistry`.

Important tools:

- `ReadEnvironmentTool`
- `AppendActionTool`
- `RunWatchdogOnceTool`
- `ResetTaskTool`
- `StepEnvTool`
- `ScriptedPickPlaceLoopTool`
- `RenderFakeEnvTool`
- `CreatePlanTool`
- `EvaluateScriptedPolicyTool`

### Environment (`envs/`)

`FakeManipulationEnv` is a 2D language-conditioned pick-and-place environment:

- 3 colored blocks
- 1 bowl
- action `[dx, dy, gripper]`
- optional RGB rendering
- randomized layouts

Rendering lives in `envs/fake_manipulation_render.py`.

### Policies (`agent/`)

All policies expose `act(observation) -> np.ndarray`.

- `ScriptedPickPlacePolicy`
- `BCPolicy`
- `VisionBCPolicy`
- `VLAPolicy`
- `RLPolicy`

`VLABackend` predicts `VLAAction` from `VLAObservation`. `FakeEnvVLAAdapter` converts between fake env observations and the VLA contract.

### Learning (`learning/`)

Offline learning intentionally uses direct env rollouts instead of the AgentOS workspace for speed:

- dataset loaders
- BC / VisionBC training
- unified evaluation
- Gym-style wrapper for RL
- evaluation report generation

This is a separate path from the online AgentOS runtime.

## Generated Directories

These are runtime artifacts and are gitignored:

```text
data/
outputs/
checkpoints/
workspace/
```

## Design Conventions

- Keep `scripts/run_agentos.py` as the primary AgentOS entry point.
- Keep `scripts/run_agent.py` as the direct policy debug path.
- Do not add new execution paths that bypass `ACTION.md` for online runtime behavior.
- `SkillLibraryPlanner` reads JSON skill libraries from Markdown. Keep skills workflow-level.
- `DeepSeekPlanner` is optional. No API key means fallback to `RuleBasedPlanner`.
- LLM planners may only emit workflow-level `TaskPlan` JSON. They must not emit low-level actions.
- Offline learning may use direct `env.step()` loops for efficiency.
- Keep HAL focused on driver contracts and concrete drivers.
- Historical teleop reference material belongs in `docs/reference/`, not in runtime code.
