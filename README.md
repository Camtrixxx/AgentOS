# Embodied AgentOS Lab

一个 simulation-first 的具身 AgentOS 项目，用来验证：

```text
language task
-> planner
-> tools
-> ACTION.md
-> watchdog + safety validator
-> HAL driver
-> ENVIRONMENT.md
-> report / trace / lessons
```

项目最初来自 teleop-control lab 原型，现在已经收敛为 file-backed embodied AgentOS runtime。早期 synthetic teleop demo 已从主线移除，相关原始资料保留在 `docs/reference/`，作为未来 human-input provider 的参考。

## What It Does

当前主线是一个可审计、可恢复、模型与执行层解耦的具身运行时：

- file-backed workspace protocol: `ACTION.md`、`ENVIRONMENT.md`、`PLAN.md`、`REPORT.md`
- `AgentOSExecutor` 通过 `ToolRegistry` 执行计划步骤
- `AppendActionTool` 把动作写入 `ACTION.md`
- `watchdog` 监听 action queue，校验后通过 HAL driver 执行
- `BaseDriver` 提供状态机、capabilities、runtime state 和 CQRS 协议边界
- `FakeManipulationDriver` 用 fake pick-and-place 环境验证完整闭环
- BC / VisionBC / RL / VLA policy 作为可插拔能力和离线学习路径

默认环境不依赖真实机器人即可运行。

## Quick Start

```bash
cd /workspace/hyh/embodied-teleop-control-lab
bash scripts/sh/00_smoke_test.sh
```

主 AgentOS 入口：

```bash
python scripts/run_agentos.py "pick up the red block and place it in the bowl"
```

运行后重点查看：

```text
workspace/.../ACTION.md       action queue and results
workspace/.../ENVIRONMENT.md  latest runtime state
workspace/.../PLAN.md         planner output
workspace/.../REPORT.md       execution summary
outputs/traces/*.jsonl        tool/runtime trace
```

## Common Commands

运行 AgentOS 闭环：

```bash
python scripts/run_agentos.py \
  "pick up the green block and place it in the bowl" \
  --workspace workspace/agentos_demo \
  --render-output outputs/agentos_demo.ppm
```

使用 DeepSeek 作为可选 LLM Planner：

```bash
export DEEPSEEK_API_KEY="sk-..."
python scripts/run_agentos.py \
  "pick up the blue block and place it in the bowl" \
  --planner deepseek
```

没有 `DEEPSEEK_API_KEY` 时，DeepSeek planner 会自动 fallback 到 deterministic `RuleBasedPlanner`。LLM 只生成 workflow-level `TaskPlan`，不会直接输出低层动作。

使用可复用 skill library 作为 Planner：

```bash
python scripts/run_agentos.py \
  "pick up the blue block and place it in the bowl" \
  --planner skill
```

默认 skill 定义在 `runtime/skills/default_skill.md`。如需试验自定义 workflow，可传入 `--skill-path path/to/SKILL.md`。

运行完整 AgentOS 多回合 benchmark：

```bash
python scripts/benchmark_agentos.py \
  --planner skill \
  --num-episodes 9 \
  --randomize-layout
```

运行 direct policy debug 路径：

```bash
python scripts/run_agent.py
```

导出 RGB 环境图：

```bash
python scripts/render_fake_env.py --output outputs/fake_env.ppm
```

采集随机化视觉 demonstrations：

```bash
bash scripts/sh/03_collect_vision_demos_random.sh
```

训练随机化 VisionBC：

```bash
bash scripts/sh/04_train_vision_bc_random.sh
```

评估随机化 VisionBC 并生成报告：

```bash
bash scripts/sh/05_eval_vision_bc_random.sh
```

运行 VLA-ready mock backend：

```bash
bash scripts/sh/06_eval_mock_vla.sh
```

清理本地生成物：

```bash
bash scripts/sh/99_clean_generated.sh
```

## Project Layout

```text
agent/        policy wrappers and VLA backends
recorders/    episode recorders and dataset schema
docs/         architecture, running guide, and reference notes
envs/         fake manipulation environment and renderer
hal/          AgentOS driver contracts, driver registry, fake driver, VLA adapter
learning/     BC / VisionBC / RL models, datasets, training, evaluation reports
runtime/      workspace protocol, executor, watchdog, action queue, planner, trace
scripts/      Python entrypoints and shell shortcuts
tests/        regression tests
tools/        plug-in tools for the AgentOS runtime
```

## VLA-Ready Design

VLA models plug in behind the same policy boundary:

```text
FakeManipulationEnv observation
-> FakeEnvVLAAdapter
-> VLAObservation(image, instruction, state)
-> VLABackend.predict(...)
-> VLAAction(ee_delta, gripper)
-> env action [dx, dy, gripper]
```

当前 mock backend 可以这样运行：

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend mock \
  --randomize-layout \
  --max-steps 100 \
  --write-report
```

以后接 OpenVLA、LeRobot、SmolVLA、Pi0 或远程模型服务时，主要新增一个 backend，实现：

```python
predict(observation: VLAObservation) -> VLAAction
```

## Generated Artifacts

以下目录是运行后生成的，不进入 Git：

```text
data/
outputs/
checkpoints/
workspace/
```

## Documentation

- [docs/running_guide.md](docs/running_guide.md): 怎么运行主要模块
- [docs/project_overview.md](docs/project_overview.md): 项目定位和架构说明
- [docs/architecture_diagram.md](docs/architecture_diagram.md): AgentOS runtime 数据流
- [recorders/schema.md](recorders/schema.md): demonstration 数据格式

## Resume Angle

> Built a simulation-first embodied AgentOS runtime with file-backed action/state protocols, a watchdog-driven HAL execution loop, driver state management, policy/VLA boundaries, offline imitation learning pipelines, and auditable reports/traces.
