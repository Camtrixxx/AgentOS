# Upgrade Mapping For Embodied Teleop Control Lab

本文说明如何把 `PhyAgentOS` 和 `HelloAgents` 的架构思想映射到当前项目。

当前项目路径：

```text
/data/heyuhang/hyh/embodied-teleop-control-lab
```

参考项目路径：

```text
/data/heyuhang/hyh/reference-repos/PhyAgentOS
/data/heyuhang/hyh/reference-repos/HelloAgents
```

## Current Project Baseline

当前项目已经具备两个可运行主线：

```text
Teleop -> Control
  synthetic hand keypoints
  -> stereo triangulation
  -> hand retargeting
  -> safety limiter
  -> fake robot backend

Embodied Agent Loop
  language instruction
  -> RGB image + state observation
  -> policy / VLA backend
  -> action [dx, dy, gripper]
  -> fake manipulation env
  -> evaluation report
```

当前强项：

- 模块边界清楚
- fake manipulation env 已经能形成语言条件闭环
- BC / VisionBC / VLA mock 后端已经有统一 evaluation entry
- 控制安全层已有 `SafetyLimiter`
- tests 覆盖了关键基础模块

当前缺口：

- Agent 和物理执行层还没有协议解耦
- 没有 workspace runtime 文件
- 没有 HAL driver 抽象
- 没有 watchdog 进程
- 工具调用没有统一响应协议
- 缺少运行时 trace
- 服务器环境文档需要补充 Ascend/NPU 注意事项

## Recommended Target Architecture

建议升级后的最小架构：

```text
embodied-teleop-control-lab/
  workspace/
    ACTION.md
    ENVIRONMENT.md
    EMBODIED.md
    LESSONS.md
    TASK.md
  runtime/
    action_queue.py
    environment_io.py
    workspace.py
    watchdog.py
    trace.py
  hal/
    base_driver.py
    fake_manipulation_driver.py
    fake_robot_driver.py
  tools/
    response.py
    base.py
    registry.py
    embodied_tools.py
    evaluation_tools.py
  scripts/
    init_workspace.py
    run_watchdog.py
```

数据流：

```text
User / Agent / Policy
-> tools.execute_action(...)
-> ACTION.md
-> runtime.watchdog
-> HAL driver
-> FakeManipulationEnv or FakeRobotBackend
-> ENVIRONMENT.md
-> ToolResponse / TraceLogger / EvaluationReport
```

## Mapping From PhyAgentOS

| PhyAgentOS concept | Current project target | Why |
| --- | --- | --- |
| `ACTION.md` | `workspace/ACTION.md` | 让 Agent/Policy 和执行层解耦 |
| `ENVIRONMENT.md` | `workspace/ENVIRONMENT.md` | 把 fake env observation 变成可读、可复盘状态 |
| `EMBODIED.md` | `workspace/EMBODIED.md` | 描述 fake robot / fake manipulation env 的能力约束 |
| `LESSONS.md` | `workspace/LESSONS.md` | 记录失败经验和安全规则 |
| `hal/base_driver.py` | `hal/base_driver.py` | 统一 fake env、fake robot、未来 ROS2/真实机器人 |
| `hal_watchdog.py` | `runtime/watchdog.py` + `scripts/run_watchdog.py` | 轮询动作、执行、更新环境 |
| `EmbodiedActionTool` | `tools/embodied_tools.py` | 把动作执行包装为 Agent tool |
| Critic validation | `runtime/action_validator.py` 或 `SafetyLimiter` 扩展 | 动作落地前做安全检查 |

## Mapping From HelloAgents

| HelloAgents concept | Current project target | Why |
| --- | --- | --- |
| `ToolResponse` | `tools/response.py` | 所有工具返回统一结构 |
| `Tool` | `tools/base.py` | 定义工具协议 |
| `ToolRegistry` | `tools/registry.py` | 统一注册和调用工具 |
| `TraceLogger` | `runtime/trace.py` | 记录 episode/action/tool/env 事件 |
| `HistoryManager` | `runtime/session_log.py` | append-only 会话日志 |
| `TodoWrite` | `workspace/TASK.md` | 长程任务拆解和状态跟踪 |
| `SkillLoader` | `workspace/SKILL.md` | 成功流程沉淀为 SOP |

## Phase 1: Minimal Embodied OS Runtime

目标：不接 LLM，不接真实机器人，先完成文件协议闭环。

新增模块：

```text
runtime/action_queue.py
runtime/environment_io.py
runtime/workspace.py
hal/base_driver.py
hal/fake_manipulation_driver.py
scripts/init_workspace.py
scripts/run_watchdog.py
tests/test_action_queue.py
tests/test_environment_io.py
tests/test_fake_manipulation_driver.py
```

最小功能：

```text
scripts/init_workspace.py
  -> 创建 workspace/*.md

tools or simple script
  -> 往 ACTION.md 写 pending action

scripts/run_watchdog.py
  -> 读取 ACTION.md
  -> 调用 FakeManipulationDriver.execute_action()
  -> 更新 ENVIRONMENT.md
  -> 更新 ACTION.md status/result
```

推荐动作协议：

```json
{
  "schema_version": "embodied_lab.action_queue.v1",
  "actions": [
    {
      "id": "move_001",
      "action_type": "env_step",
      "parameters": {
        "action": [0.02, 0.0, -1.0]
      },
      "status": "pending"
    }
  ]
}
```

推荐环境协议：

```json
{
  "schema_version": "embodied_lab.environment.v1",
  "updated_at": "2026-05-12T00:00:00Z",
  "task": {
    "instruction": "pick up the red block and place it in the bowl",
    "target_color": "red"
  },
  "robot": {
    "ee_position": [0.0, -0.75],
    "gripper_closed": false,
    "held_object": null
  },
  "objects": {
    "red_block": {"color": "red", "position": [-0.55, 0.15]}
  },
  "receptacles": {
    "bowl": {"position": [0.55, 0.65]}
  },
  "episode": {
    "step_count": 0,
    "success": false,
    "done": false,
    "last_reward": 0.0
  }
}
```

## Phase 2: Tool Layer

目标：把当前脚本能力变成可被 Agent 调用的工具。

新增模块：

```text
tools/response.py
tools/base.py
tools/registry.py
tools/embodied_tools.py
tools/evaluation_tools.py
tools/render_tools.py
runtime/trace.py
```

工具候选：

```text
read_environment
append_action
execute_action_once
render_fake_env
run_scripted_episode
evaluate_policy
collect_demo
train_bc
train_vision_bc
```

工具返回格式：

```text
status: success | partial | error
text: 给 Agent/用户看的摘要
data: 结构化结果
error: 错误码和错误信息
stats: time_ms 等
context: tool_name、params 等
```

## Phase 3: Planner / Critic / VLA Integration

目标：引入更像 Agent OS 的任务规划层。

新增能力：

- Rule-based critic: 不依赖 LLM 的安全校验
- Optional LLM planner: 把自然语言任务转成 action queue
- VLA tool: 把 `VLABackend.predict()` 包成工具
- Session trace: 记录 planner 输入、动作、环境变化、评估结果

推荐流程：

```text
instruction
-> planner proposes actions
-> critic validates against EMBODIED.md + ENVIRONMENT.md
-> ACTION.md
-> watchdog executes
-> trace + report
```

## Phase 4: Real Robot / Simulator Readiness

目标：为真实机器人和仿真器保留接口。

未来 driver：

```text
ROS2ManipulationDriver
UnitreeDDSDriver
IsaacSimDriver
RealSensePerceptionDriver
InspireHandDriver
```

这个阶段再考虑：

- ROS2 node
- DDS bridge
- 相机/深度传感器
- 真实机器人状态反馈
- emergency stop
- command timeout
- 多机器人 workspace

## Ascend/NPU Environment Note

当前服务器是华为 Ascend 环境：

```text
npu-smi available
nvidia-smi unavailable
python3 available
python unavailable
torch / torch_npu / pytest not installed in current environment
```

因此文档和脚本后续应注意：

- 命令优先写 `python3`，或说明需要激活虚拟环境让 `python` 指向 Python 3。
- BC/VisionBC 依赖 PyTorch，不是仅依赖 numpy。
- 如果要用 NPU 训练，需要安装匹配版本的 `torch` + `torch_npu`。
- 短期 smoke test 可先跑不依赖 torch 的 teleop/control、fake env、mock VLA 部分。

## Implementation Priority

建议先做这 6 个最小任务：

1. 增加 `runtime/action_queue.py`。
2. 增加 `runtime/environment_io.py`。
3. 增加 `hal/base_driver.py`。
4. 增加 `hal/fake_manipulation_driver.py`。
5. 增加 `scripts/init_workspace.py` 和 `scripts/run_watchdog.py`。
6. 增加对应 tests，保证 action queue 和 environment IO 稳定。

完成后，本项目就会从“可训练 fake embodied pipeline”升级为“有 Agent OS 雏形的具身智能实验框架”。

