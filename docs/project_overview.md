# Embodied AgentOS Lab 项目介绍

## 项目定位

`Embodied AgentOS Lab` 是一个 simulation-first 的具身 AgentOS runtime。它关注的不是单个策略模型，而是一个真实具身系统需要的执行边界：

```text
任务 / 策略意图
-> 计划
-> 工具调用
-> 动作队列
-> 安全校验
-> HAL driver
-> 环境状态回写
-> 报告、trace、lessons
```

项目最早来自 teleop-control lab 原型。早期 synthetic hand keypoint / stereo / retargeting demo 已从主线移除，相关原始资料保留在 `docs/reference/`。当前主线正式收敛为 file-backed embodied AgentOS。

## 为什么这个项目适合具身智能方向

具身智能系统的难点不只是“预测一个动作”，而是如何让模型、工具、环境和执行层在安全边界内协作。这个项目把这些边界显式化：

| 层级 | 当前模块 |
| --- | --- |
| 任务规划 | `runtime/planner.py` |
| 工具执行 | `tools/`, `tools/registry.py` |
| 动作协议 | `runtime/action_queue.py`, `workspace/ACTION.md` |
| 环境状态 | `runtime/environment_io.py`, `workspace/ENVIRONMENT.md` |
| 安全校验 | `runtime/action_validator.py` |
| 执行守护 | `runtime/watchdog.py` |
| HAL driver | `hal/base_driver.py`, `hal/fake_manipulation_driver.py` |
| 策略能力 | `agent/`, `learning/` |
| 数据闭环 | `recorders/`, `learning/`, `outputs/` |

这个设计让 Agent 不直接控制机器人或环境。所有动作都要落入 `ACTION.md`，经过 validator 和 watchdog，再由 driver 执行。

## 当前 AgentOS 数据流

```text
用户任务
  "pick up the green block and place it in the bowl"
        |
        v
scripts/run_agentos.py
        |
        v
RuleBasedPlanner
        |
        v
workspace/PLAN.md
        |
        v
AgentOSExecutor
        |
        v
ToolRegistry
        |
        +-- ResetTaskTool
        +-- ScriptedPickPlaceLoopTool
        +-- StepEnvTool
        +-- RenderFakeEnvTool
        |
        v
AppendActionTool 写入 ACTION.md
        |
        v
Watchdog 监听 action queue
        |
        v
ActionValidator 校验 action schema、动作范围、workspace bounds
        |
        v
FakeManipulationDriver
        |
        v
FakeManipulationEnv.step(...)
        |
        v
ENVIRONMENT.md / REPORT.md / TRACE.jsonl
```

## Workspace Protocol

默认 workspace 是一组 Markdown 文件，每个文件内包含 JSON fenced block，便于人类和 Agent 同时检查。

```text
workspace/
├── ACTION.md       # pending/running/completed/failed action queue
├── ENVIRONMENT.md  # task, robot, scene, episode, runtime state
├── EMBODIED.md     # driver profile and constraints
├── LESSONS.md      # failure notes and postmortem notes
├── TASK.md         # current task summary
├── SKILL.md        # reusable workflow notes
├── PLAN.md         # planner output
└── REPORT.md       # execution report
```

`ACTION.md` 使用加锁 read-modify-write，避免 tool agent 和 watchdog 并发写入时互相覆盖。写文件使用 atomic replace，避免读到半写入文件。

## HAL Driver

`BaseDriver` 是 AgentOS 的执行边界。它提供：

- `CommandDriver`: `execute_action(...)`
- `QueryDriver`: `get_environment()`, `get_runtime_state()`, `health_check()`
- driver 状态机: `disconnected`, `idle`, `executing`, `fault`, `closed`
- capabilities: supported actions, workspace bounds, max step delta

当前默认实现是 `FakeManipulationDriver`，它把 file-backed AgentOS runtime 接到 `FakeManipulationEnv`。

## Learning 与 Runtime 的关系

离线学习管线保留 direct rollout 路径：

```text
FakeManipulationEnv
-> policy.act(...)
-> env.step(...)
-> recorder / dataset / evaluation report
```

这条路径用于批量采集、训练和评估，避免每个训练 step 都写 workspace 文件。AgentOS runtime 用于需要安全审计、trace 和 driver 边界的在线执行。

## 主要入口

主 AgentOS 入口：

```bash
python scripts/run_agentos.py "pick up the red block and place it in the bowl"
```

direct policy debug：

```bash
python scripts/run_agent.py
```

统一评估入口：

```bash
python learning/evaluate_policy.py --policy scripted --write-report
```

## 项目结构

```text
agent/        policy wrappers and VLA backends
docs/         architecture docs and reference material
envs/         fake manipulation environment
hal/          driver contracts, registry, fake driver, VLA adapter
learning/     training, evaluation, BC/VisionBC/RL utilities
recorders/    episode recorders and LeRobot exporter
runtime/      workspace protocol, executor, watchdog, action queue, planner
scripts/      runtime and experiment entrypoints
tests/        regression tests
tools/        ToolRegistry and runtime tools
```

## 下一步方向

- 增加 AgentOS benchmark，统计多回合 success rate、latency、action failure count。
- 把 `SKILL.md` 升级成可解析 workflow library。
- 增加真实 simulator driver，例如 ManiSkill / Isaac / MuJoCo。
- 增加 remote VLA provider 和 human approval gate。
