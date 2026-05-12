# PhyAgentOS Architecture Notes

参考源码路径：

```text
/data/heyuhang/hyh/reference-repos/PhyAgentOS
```

GitHub:

```text
https://github.com/Camtrixxx/PhyAgentOS
```

## Project Positioning

`PhyAgentOS` 是一个具身智能 Agent OS 框架，核心思想是把“大脑 Agent”和“物理执行层”解耦。大模型不直接调用硬件 API，而是通过本地工作区中的协议文件表达任务、动作、环境状态和机器人能力。

它的关键词是：

- State-as-a-File
- Markdown protocol
- Track A / Track B 双轨架构
- HAL watchdog
- Driver abstraction
- Embodied profile
- Critic validation
- Fleet / multi-robot workspace

## Top-Level Structure

主要目录：

```text
PhyAgentOS/
  PhyAgentOS/
    agent/          # Planner、loop、memory、subagent、tools
    templates/      # ACTION.md、ENVIRONMENT.md、EMBODIED.md 等协议模板
    providers/      # LLM provider 抽象和适配
    session/        # 会话管理
    skills/         # Agent skills
    config/         # 配置与路径管理
  hal/
    base_driver.py  # 所有硬件/仿真 driver 的抽象接口
    hal_watchdog.py # 轮询 ACTION.md 并执行动作
    drivers/        # 不同机器人和仿真 driver
    simulation/     # 仿真和 ENVIRONMENT.md IO
    profiles/       # 机器人 EMBODIED.md profile
  docs/
  examples/
  tests/
```

## Core Runtime Model

PhyAgentOS 把系统分成两条轨道：

```text
Track A: cognitive agent
  user instruction
  -> planner
  -> critic
  -> write ACTION.md

Track B: physical execution
  hal_watchdog.py
  -> read ACTION.md
  -> driver.execute_action(...)
  -> update ENVIRONMENT.md
  -> update ACTION.md status/result
```

这个设计的好处是：

- Agent 侧和硬件侧可以独立运行。
- 出错时可以直接查看 Markdown 文件定位问题。
- 真实机器人接入时，可以把风险集中在 HAL driver 和 watchdog 层。
- 同一套 Agent 协议可以切换不同机器人 profile。

## Workspace Protocol

PhyAgentOS 的关键文件都放在 workspace 中：

```text
workspace/
  ACTION.md       # 待执行动作队列
  ENVIRONMENT.md  # 环境状态、scene graph、robot runtime state
  EMBODIED.md     # 当前机器人能力、约束、支持动作
  LESSONS.md      # 失败经验和安全提示
  SKILL.md        # 成功工作流 SOP
```

### ACTION.md

`ACTION.md` 中的核心 payload 是一个 JSON code block：

```json
{
  "schema_version": "PhyAgentOS.action_queue.v1",
  "actions": [
    {
      "id": "start_demo",
      "action_type": "start",
      "parameters": {},
      "status": "pending"
    }
  ]
}
```

watchdog 会执行第一个 `status = pending` 的动作，并把结果写回：

```json
{
  "id": "start_demo",
  "action_type": "start",
  "parameters": {},
  "status": "completed",
  "result": "..."
}
```

### ENVIRONMENT.md

`ENVIRONMENT.md` 是结构化环境状态。它支持：

- `objects`: 当前物体状态
- `robots`: 每个机器人运行时状态
- `scene_graph`: 语义场景图
- `map`: 导航地图摘要
- `tf`: 坐标系状态摘要
- `updated_at`: 最近更新时间

这对当前项目很有价值，因为 `FakeManipulationEnv` 的 observation 可以被写成类似结构：

```text
objects.red_block.position
objects.blue_block.position
objects.green_block.position
receptacles.bowl.position
robot.ee_position
robot.gripper_closed
task.instruction
```

### EMBODIED.md

`EMBODIED.md` 是机器人能力 profile，包含：

- Identity
- Degrees of Freedom
- Sensors
- Supported Actions
- Physical Constraints
- Connection
- Runtime Protocol

它的用途不是给代码解析所有细节，而是给 Agent/Critic 读取，让大模型知道当前机器人能做什么、不能做什么。

## HAL Driver Design

`hal/base_driver.py` 定义了所有 driver 的最小接口：

```python
class BaseDriver(ABC):
    def get_profile_path(self) -> Path: ...
    def load_scene(self, scene: dict[str, dict]) -> None: ...
    def execute_action(self, action_type: str, params: dict) -> str: ...
    def get_scene(self) -> dict[str, dict]: ...
```

可选生命周期接口：

```python
connect()
disconnect()
is_connected()
health_check()
get_runtime_state()
close()
```

这个抽象非常适合当前项目未来扩展：

```text
FakeManipulationDriver
FakeRobotHandDriver
ROS2RobotDriver
UnitreeDriver
IsaacSimDriver
```

## Watchdog Design

`hal/hal_watchdog.py` 的核心流程：

```text
load driver
install EMBODIED.md profile
load ENVIRONMENT.md scene
loop:
  health_check
  save current environment
  read ACTION.md
  find first pending action
  result = driver.execute_action(action_type, params)
  save ENVIRONMENT.md
  update ACTION.md status/result
```

这可以作为本项目 `scripts/run_watchdog.py` 的参考。

## Agent Tools

PhyAgentOS 的 `EmbodiedActionTool` 做了三件关键事：

1. 根据 robot_id 选择 workspace。
2. 读取 `EMBODIED.md` 和 `ENVIRONMENT.md`。
3. 用 Critic 校验动作是否安全可行，校验通过后写入 `ACTION.md`。

本项目短期不需要完整 LLM Critic，但可以先做一个规则版 Critic：

```text
action_type 是否被支持
dx/dy 是否超过 step_size
target_color 是否存在
是否超出 workspace bounds
gripper 参数是否合法
```

## Fleet Mode

PhyAgentOS 支持多机器人 workspace：

```text
workspaces/
  shared/
    ENVIRONMENT.md
    LESSONS.md
  robot_001/
    ACTION.md
    EMBODIED.md
  robot_002/
    ACTION.md
    EMBODIED.md
```

当前项目暂时不需要 Fleet，但可以提前保留接口命名，比如 `robot_id`、`driver_name`、`workspace`。

## What To Borrow

适合借鉴：

- `ACTION.md` 动作队列协议
- `ENVIRONMENT.md` 环境状态协议
- `EMBODIED.md` 机器人能力 profile
- `BaseDriver` HAL 抽象
- `watchdog` 执行动作和更新环境的闭环
- `Critic` 在动作落盘前做安全校验的思想

不建议现在照搬：

- 多平台 channel 系统
- 完整 provider/LLM 框架
- Fleet registry 的复杂实现
- 大量真实机器人 driver
- InternUtopia/Isaac 复杂仿真层

## License Note

PhyAgentOS 仓库声明为 MIT License。当前文档只做架构学习和设计归纳，不复制其大段源码。

