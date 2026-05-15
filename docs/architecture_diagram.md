# 项目架构图

这份文档用于快速建立项目全局认知。当前主线是 file-backed embodied AgentOS runtime；早期 synthetic teleop demo 已从主运行时代码中移除，历史资料保留在 `docs/reference/`。

## 一句话总结

这是一个 simulation-first 的具身 AgentOS 项目：它包含 fake manipulation 环境、policy/VLA 能力、离线学习管线，以及一个基于 Markdown workspace 的安全执行 runtime，包括 planner、tools、action queue、watchdog 和 HAL driver。

## 全局系统图

```text
                                  +----------------------+
                                  |      User / AI       |
                                  +----------+-----------+
                                             |
                                             v
+--------------------------------------------------------------------------------+
|                                  Entry Points                                  |
| scripts/run_agentos.py       主 AgentOS runtime                                |
| scripts/run_agent.py         direct policy debug                               |
| learning/evaluate_policy.py  offline policy evaluation                         |
| scripts/collect_*            demonstration collection                          |
| scripts/train_*              policy / VLA training wrappers                    |
| scripts/run_watchdog.py      standalone ACTION.md watchdog                     |
+--------------------------------------------------------------------------------+
        |                                  |                                  |
        v                                  v                                  v
+----------------------+        +----------------------+        +----------------------+
| AgentOS Runtime      |        | Policy / VLA         |        | Offline Learning     |
| 文件协议、安全执行     |        | 可替换动作生成能力      |        | 数据、训练、评估       |
+----------+-----------+        +----------+-----------+        +----------+-----------+
           |                               |                               |
           v                               v                               v
+----------------------+        +----------------------+        +----------------------+
| runtime/planner.py   |        | agent/*.py           |        | recorders/           |
| TaskPlan             |        | Policy.act(...)      |        | episode datasets     |
+----------+-----------+        +----------+-----------+        +----------+-----------+
           |                               |                               |
           v                               v                               v
+----------------------+        +----------------------+        +----------------------+
| tools/registry.py    |        | hal/vla_adapter.py   |        | learning/*.py        |
| ToolRegistry         |        | VLA boundary         |        | train/eval reports   |
+----------+-----------+        +----------+-----------+        +----------+-----------+
           |
           v
+----------------------+
| workspace/ACTION.md  |
| pending actions      |
+----------+-----------+
           |
           v
+----------------------+
| runtime/watchdog.py  |
| validate + execute   |
+----------+-----------+
           |
           v
+----------------------+
| hal/BaseDriver       |
| FakeManipulationDriver|
+----------+-----------+
           |
           v
+----------------------+
| envs/FakeManipulation|
| [dx, dy, gripper]    |
+----------+-----------+
           |
           v
+----------------------+
| ENVIRONMENT/REPORT   |
| trace / lessons      |
+----------------------+
```

## AgentOS Runtime 流程

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
  PlannedStep(reset_task)
  PlannedStep(scripted_pick_place_loop)
  PlannedStep(render_fake_env)
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
AppendActionTool writes ACTION.md
        |
        v
Watchdog waits for ACTION.md changes
        |
        v
ActionValidator
  action type / parameter shape / workspace bounds / max delta
        |
        v
HAL Driver
  BaseDriver state machine + FakeManipulationDriver
        |
        v
FakeManipulationEnv.step(...)
        |
        v
EnvironmentIO writes ENVIRONMENT.md
        |
        v
REPORT.md / TRACE.jsonl / LESSONS.md
```

## Policy / VLA 数据流

```text
FakeManipulationEnv observation
  instruction, ee_position, gripper_closed,
  objects, receptacles, image(optional), step_count
        |
        +---------------------------------------------------------+
        |                                                         |
        v                                                         v
Scripted / BC / VisionBC / RLPolicy                         VLAPolicy
        |                                                         |
        |                                                FakeEnvVLAAdapter
        |                                                         |
        |                                                VLABackend.predict
        |                                                mock / SmolVLA
        |                                                         |
        +---------------- action [dx, dy, gripper] <--------------+
```

在在线 AgentOS 路径中，policy 只提出动作；动作必须经过 `AppendActionTool -> ACTION.md -> watchdog -> HAL driver` 执行。离线训练和评估可以直接使用 `run_episode()`，避免批量 ML 工作流产生大量 workspace 写入。

## Workspace Protocol

```text
workspace/ACTION.md       pending/running/completed/failed actions
workspace/ENVIRONMENT.md  latest task, robot, scene, episode, runtime state
workspace/EMBODIED.md     driver capabilities and constraints
workspace/PLAN.md         planner output
workspace/TASK.md         current task status
workspace/SKILL.md        reusable workflow notes
workspace/LESSONS.md      failure notes and postmortems
workspace/REPORT.md       execution summary
```

`ACTION.md` 使用文件锁包住 read-modify-write，写入使用 atomic replace，watchdog 使用 `FileWatcher` 等待文件变化并保留 timeout fallback。

## HAL Driver

`hal/base_driver.py` 提供：

- `CommandDriver`, `QueryDriver`, `RuntimeDriver` 协议
- `DriverState`: `disconnected`, `idle`, `executing`, `fault`, `closed`
- `execute_action()` 模板方法，统一管理状态转换
- `get_capabilities()` 和 `get_runtime_state()`

当前 driver：

```text
hal/fake_manipulation_driver.py -> envs/FakeManipulationEnv
```

## 数据集 / 训练流程

```text
ScriptedPickPlacePolicy
        |
        v
scripts/collect_demo.py
scripts/collect_vision_demo.py
        |
        v
data/demos/
data/vision_demos/
data/vision_demos_random/
        |
        +----------------------------+
        |                            |
        v                            v
learning/train_bc.py          learning/train_vision_bc.py
        |                            |
        v                            v
checkpoints/*.pt              outputs/eval_reports_*
```

Vision demos can be exported to LeRobot format through `scripts/export_lerobot_dataset.py`, then used by `scripts/train_smolvla.py`.

## 模块职责矩阵

| 层级 | 主要文件 | 职责 |
|------|----------|------|
| Runtime | `runtime/*.py` | file-backed AgentOS workspace, executor, watchdog, action queue |
| Tools | `tools/*.py` | Agent 可调用工具和统一响应 |
| HAL | `hal/base_driver.py`, `hal/fake_manipulation_driver.py` | driver 状态机、capabilities、环境执行 |
| Environment | `envs/fake_manipulation_env.py` | 2D 语言条件 pick-and-place 环境 |
| Policies | `agent/*.py` | scripted、BC、VisionBC、VLA、RL policy wrappers |
| VLA boundary | `hal/vla_adapter.py`, `agent/vla_*.py` | fake env 与 VLA contract 的转换 |
| Learning | `learning/*.py` | 数据加载、训练、评估、报告 |
| Data | `recorders/*.py` | episode 记录、检查、LeRobot 导出 |
| Scripts | `scripts/*.py`, `scripts/sh/*.sh` | CLI 入口和可复现实验流程 |
| Tests | `tests/*.py` | runtime、tools、HAL、env、learning 回归测试 |

## 一分钟讲清这个项目

这是一个 simulation-first embodied AgentOS reference implementation。它用一个小型 fake manipulation 世界验证具身系统最关键的工程边界：Agent 不直接操作环境或硬件，而是把动作写入 file-backed action queue；watchdog 监听、校验并通过 HAL driver 执行；环境状态和结果回写到 workspace，形成可审计、可恢复、可测试的闭环。离线学习管线继续提供 BC/VisionBC/RL/VLA 能力，但在线执行主线统一走 AgentOS runtime。
