# 项目架构图

这份文档用于给没有上下文的人或 AI 快速建立项目全局认知。代码路径、命令、类名保持英文，说明文字使用中文。

## 一句话总结

这是一个 simulation-first 的具身智能实验室项目：它包含 fake manipulation 环境、策略学习管线、VLA/SmolVLA 接口、LeRobot 数据导出与训练，以及一个基于 Markdown 文件协议的 Agent OS runtime，包括 planner、tools、watchdog 和 HAL driver。

## 全局系统图

```text
                                      +----------------------------------------+
                                      |                User / AI               |
                                      | 任务、命令、实验、检查、调试            |
                                      +---------------------+------------------+
                                                            |
                                                            v
+----------------------------------------------------------------------------------------------+
|                                      Entry Points                                             |
| scripts/run_agent.py                 运行 scripted / BC / VisionBC / VLA 策略                 |
| learning/evaluate_policy.py           统一策略评估入口                                        |
| scripts/collect_demo.py               采集 state demonstration                                |
| scripts/collect_vision_demo.py        采集 RGB demonstration                                  |
| scripts/export_lerobot_dataset.py     导出 LeRobot manifest / native dataset                  |
| scripts/train_smolvla.py              LeRobot SmolVLA 训练封装                                |
| scripts/run_tool_agent.py             tool agent smoke / e2e                                  |
| scripts/run_planner_agent.py          planner + executor + watchdog 闭环                      |
| scripts/run_watchdog.py               单独消费 ACTION.md 的 watchdog                          |
+----------------------------------------------------------------------------------------------+
        |                         |                            |                         |
        v                         v                            v                         v
+------------------+     +----------------------+    +----------------------+   +---------------------+
| Teleop / Control |     | Embodied Learning    |    | Agent OS Runtime     |   | Dataset / Training  |
| 遥操作控制管线    |     | 具身学习闭环          |    | 文件协议运行时        |   | 数据、训练、报告      |
+--------+---------+     +----------+-----------+    +----------+-----------+   +----------+----------+
         |                          |                           |                          |
         v                          v                           v                          v
+------------------+     +----------------------+    +----------------------+   +---------------------+
| perception/      |     | agent/agent_loop.py  |    | runtime/workspace.py |   | datasets/           |
| 双目 3D 重建      |     | Env + Policy rollout |    | ACTION/ENV/PLAN md  |   | 记录/导出/检查       |
+--------+---------+     +----------+-----------+    +----------+-----------+   +----------+----------+
         |                          |                           |                          |
         v                          v                           v                          v
+------------------+     +----------------------+    +----------------------+   +---------------------+
| retargeting/     |     | agent/*Policy        |    | tools/registry.py    |   | learning/           |
| 手部到机器人命令  |     | scripted/BC/VLA/RL   |    | Tool 协议与注册表     |   | 训练/评估            |
+--------+---------+     +----------+-----------+    +----------+-----------+   +----------+----------+
         |                          |                           |                          |
         v                          v                           v                          v
+------------------+     +----------------------+    +----------------------+   +---------------------+
| control/         |     | envs/                |    | runtime/executor.py  |   | vla/                |
| 安全限制器        |     | FakeManipulationEnv  |    | 执行 TaskPlan        |   | mock + SmolVLA      |
+--------+---------+     +----------+-----------+    +----------+-----------+   +----------+----------+
         |                          |                           |                          |
         v                          v                           v                          v
+------------------+     +----------------------+    +----------------------+   +---------------------+
| fake robot       |     | observation / action |    | runtime/watchdog.py  |   | outputs/            |
| backend          |     | [dx, dy, gripper]    |    | 验证并执行 action    |   | checkpoint/report   |
+------------------+     +----------------------+    +----------+-----------+   +---------------------+
                                                                 |
                                                                 v
                                                       +----------------------+
                                                       | hal/                 |
                                                       | BaseDriver + fake    |
                                                       +----------+-----------+
                                                                  |
                                                                  v
                                                       +----------------------+
                                                       | envs/                |
                                                       | FakeManipulationEnv  |
                                                       +----------------------+
```

## 主学习 / 策略数据流

英文/代码视角：

```text
TaskSpec(
  instruction="pick up the red block and place it in the bowl",
  target_color="red"
)
        |
        v
FakeManipulationEnv.reset(task)
        |
        v
Observation dict
  instruction, ee_position, gripper_closed,
  objects, receptacles, image(optional), step_count
        |
        +-----------------------------------------------------------------+
        |                                                                 |
        v                                                                 v
ScriptedPickPlacePolicy / BCPolicy / VisionBCPolicy / RLPolicy      VLAPolicy
        |                                                                 |
        |                                                                 v
        |                                                     FakeEnvVLAAdapter
        |                                                       observation ->
        |                                                       VLAObservation
        |                                                                 |
        |                                                                 v
        |                                               VLABackend.predict(...)
        |                                               mock or SmolVLABackend
        |                                                                 |
        |                                                                 v
        +------------------------------ action [dx, dy, gripper] <--------+
                                           |
                                           v
                                 FakeManipulationEnv.step(action)
                                           |
                                           v
                           reward / done / success / next observation
                                           |
                                           v
                         EpisodeRecorder / VisionEpisodeRecorder / Report
```

中文语义版：

```text
任务描述
  “把红色方块拿起来并放进碗里”
        |
        v
FakeManipulationEnv 初始化任务
        |
        v
环境产生 observation
  包括语言指令、末端执行器位置、夹爪状态、
  物体位置、目标容器位置、RGB 图像、当前步数
        |
        +-------------------------------------------------------------+
        |                                                             |
        v                                                             v
普通策略路线                                                   VLA 策略路线
  scripted / BC / VisionBC / RL                                 VLAPolicy
        |                                                             |
        |                                                             v
        |                                                   环境 observation
        |                                                   转成 VLAObservation
        |                                                             |
        |                                                             v
        |                                                   VLA backend 推理
        |                                                   mock 或 SmolVLA
        |                                                             |
        |                                                             v
        +------------------------- 输出动作 [dx, dy, gripper] <--------+
                                           |
                                           v
                               环境执行一步动作
                                           |
                                           v
                           得到 reward / done / success / 下一帧状态
                                           |
                                           v
                           记录 episode、生成评估报告或训练数据
```

这个项目的策略边界刻意设计得很简单：每个策略只需要实现 `act(observation) -> np.ndarray`。因此环境本身不用改，就可以替换为 scripted expert、state-only BC、RGB VisionBC、RL、mock VLA 或 LeRobot SmolVLA。

## Agent OS Runtime 流程

英文/代码视角：

```text
用户任务
  "pick up the green block and place it in the bowl"
        |
        v
scripts/run_planner_agent.py
        |
        v
RuleBasedPlanner
        |
        v
workspace/PLAN.md
  PlannedStep(reset_task)
  PlannedStep(scripted_pick_place_loop)
  PlannedStep(render)
        |
        v
Executor
        |
        v
ToolRegistry
        |
        +-- ResetTaskTool
        +-- StepEnvTool
        +-- ReadEnvironmentTool
        +-- RenderFakeEnvTool
        +-- EvaluateScriptedPolicyTool
        |
        v
AppendActionTool 把 JSON action 写入 workspace/ACTION.md
        |
        v
Watchdog 轮询 ACTION.md
        |
        v
ActionValidator
  检查 action 类型、向量形状、workspace 边界、单步最大位移
        |
        v
HAL Driver
  FakeManipulationDriver
        |
        v
FakeManipulationEnv.step(...)
        |
        v
EnvironmentIO 写回 workspace/ENVIRONMENT.md
        |
        v
Executor 更新 PLAN.md / REPORT.md / trace files
```

中文语义版：

```text
用户给出具身任务
        |
        v
planner 根据任务生成执行计划
        |
        v
计划写入 PLAN.md
        |
        v
executor 按计划调用工具
        |
        v
工具把动作追加到 ACTION.md
        |
        v
watchdog 读取待执行动作
        |
        v
validator 做安全检查
  动作类型是否合法
  动作参数是否完整
  位移是否超过限制
  是否越出 workspace
        |
        v
HAL driver 执行动作
        |
        v
fake manipulation 环境状态改变
        |
        v
EnvironmentIO 把最新状态写回 ENVIRONMENT.md
        |
        v
executor 继续下一步计划，最后写 REPORT.md
```

这个 runtime 是 file-backed 的。`workspace/` 下的 Markdown 文件不只是日志，而是模块之间的协议：

```text
workspace/ACTION.md       pending 和 completed actions
workspace/ENVIRONMENT.md  最新环境状态
workspace/EMBODIED.md     driver 能力说明
workspace/PLAN.md         planner 输出
workspace/TASK.md         当前任务状态
workspace/SKILL.md        可复用流程配方
workspace/LESSONS.md      失败记录与复盘
workspace/REPORT.md       执行总结
```

这个设计的好处是：人类、LLM、tool agent 都可以读写同一套状态文件，不需要数据库或消息队列，也能完成可检查、可恢复的具身任务执行。

## 数据集 / 训练流程

英文/代码视角：

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
checkpoints/bc_policy.pt      checkpoints/vision_bc_policy.pt
        |                            |
        +--------------+-------------+
                       |
                       v
              learning/evaluate_policy.py
                       |
                       v
              outputs/eval_reports_*

Vision demos
        |
        v
scripts/export_lerobot_dataset.py
        |
        +-- frames.jsonl + metadata.json
        |
        +-- native_lerobot/
              meta/info.json
              meta/stats.json
              data/chunk-000/file-000.parquet
        |
        v
scripts/train_smolvla.py
        |
        v
outputs/smolvla_*/checkpoints/*/pretrained_model/
```

中文语义版：

```text
scripted expert 自动完成任务
        |
        v
采集 demonstration
        |
        v
保存为本地数据集
  state demo
  vision demo
  randomized vision demo
        |
        +------------------------------+
        |                              |
        v                              v
训练 state BC                  训练 VisionBC
        |                              |
        v                              v
保存 BC checkpoint             保存 VisionBC checkpoint
        |                              |
        +---------------+--------------+
                        |
                        v
                  统一评估入口
                        |
                        v
                  生成评估报告

vision demo 数据
        |
        v
导出 LeRobot 格式
        |
        +-- 轻量 manifest
        |
        +-- 原生 LeRobotDataset
        |
        v
用 LeRobot / SmolVLA 训练
        |
        v
保存 SmolVLA checkpoint
```

当前重要的已生成数据与 checkpoint：

```text
data/vision_demos_random/
  150 episodes, 5027 frames

outputs/lerobot_smolvla_formal_dataset/native_lerobot/
  由 vision_demos_random 导出的 native LeRobotDataset

outputs/smolvla_formal_pretrained_10step/checkpoints/000010/pretrained_model/
  真实 LeRobot/SmolVLA 预训练权重微调 10 step 后保存的 checkpoint
```

## SmolVLA / LeRobot 接入方式

SmolVLA 是通过 VLA 边界接入的，而不是侵入式改造整个项目：

英文/代码视角：

```text
FakeManipulationEnv observation
        |
        v
FakeEnvVLAAdapter
        |
        v
VLAObservation(image, instruction, state)
        |
        v
SmolVLABackend.predict(...)
        |
        v
LeRobot SmolVLAPolicy.select_action(batch)
        |
        v
VLAAction(ee_delta, gripper)
        |
        v
env action [dx, dy, gripper]
```

中文语义版：

```text
fake 环境产生当前观测
        |
        v
适配器把环境观测转换成 VLA 输入格式
        |
        v
VLA 输入包含图像、语言指令、状态向量
        |
        v
SmolVLA backend 调用 LeRobot policy
        |
        v
LeRobot SmolVLAPolicy 预测动作
        |
        v
动作转换成环境可执行的 [dx, dy, gripper]
        |
        v
fake 环境执行动作并进入下一步
```

关键文件：

```text
vla/smolvla_backend.py
datasets/lerobot_exporter.py
scripts/export_lerobot_dataset.py
scripts/train_smolvla.py
scripts/run_lerobot_env.sh
docs/smolvla_rl_integration.md
```

LeRobot 环境是隔离安装的：

```text
/workspace/hyh/.venvs/lerobot-smolvla
```

隔离原因：如果把 `lerobot[smolvla]` 直接装进主 Ascend 环境，它会拉取更新的 `torch`/`numpy`，可能破坏当前可用的 `torch_npu`。目前隔离的 LeRobot venv 可以完成数据导出和训练，但使用的是 CPU PyTorch。

已确认的环境状态：

```text
Main Docker Python:
  torch 2.1.0
  torch_npu 2.1.0.post18.dev20251112
  numpy 1.26.4
  NPU visible

LeRobot venv:
  torch 2.10.0+cpu
  no cuda
  no torch.npu
```

运行 LeRobot 相关命令时，需要清空 `PYTHONPATH`，或使用封装脚本：

```bash
scripts/run_lerobot_env.sh scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir outputs/lerobot_export \
  --format native
```

训练 SmolVLA：

```bash
python scripts/train_smolvla.py \
  --vision-data-dir data/vision_demos_random \
  --dataset-output-dir outputs/lerobot_smolvla_formal_dataset \
  --output-dir outputs/smolvla_formal_pretrained_10step \
  --steps 10 \
  --load-vlm-weights \
  --overwrite
```

`scripts/train_smolvla.py` 默认配置：

```text
HF_ENDPOINT=https://hf-mirror.com
device=cpu
num_vlm_layers=1
num_expert_layers=1
chunk_size=4
n_action_steps=4
image_size=128
```

## 模块职责矩阵

| 层级 | 主要文件 | 职责 |
|------|----------|------|
| Environment | `envs/fake_manipulation_env.py` | 2D 语言条件 pick-and-place 环境 |
| Agent rollout | `agent/agent_loop.py` | 通用 env-policy episode loop |
| Policies | `agent/*.py` | Scripted、BC、VisionBC、VLA、RL 策略封装 |
| VLA boundary | `adapters/vla_adapter.py`, `vla/*.py` | 环境 observation/action 与 VLA contract 的转换 |
| Learning | `learning/*.py` | 数据加载、特征提取、训练、评估 |
| RL | `rl/gym_fake_manipulation.py`, `scripts/train_rl.py` | Gym-style wrapper 与 SB3 训练入口 |
| Data | `datasets/*.py` | episode 记录、质量检查、LeRobot 导出 |
| Runtime | `runtime/*.py` | 文件协议 planner、action queue、watchdog、executor、trace |
| Tools | `tools/*.py` | Tool 协议和可执行 agent actions |
| HAL | `hal/*.py` | runtime 与环境之间的 driver 抽象 |
| Teleop | `perception/`, `retargeting/`, `control/`, `kinematics/` | 早期 teleop-to-control demo 路径 |
| Scripts | `scripts/*.py`, `scripts/sh/*.sh` | CLI 入口和可复现实验流程 |
| Tests | `tests/*.py` | env、learning、runtime、tools、VLA、RL 回归测试 |

## 一分钟讲清这个项目

这是一个 simulation-first 的具身智能实验室项目。它从一个小型 2D pick-and-place 世界开始，让 observation、action、语言任务、demonstration、训练、评估和安全执行这些核心概念都可以在没有真实机器人的情况下开发。环境之上，项目支持多种策略：scripted expert、state behavior cloning、vision behavior cloning、RL wrapper、mock VLA 和 LeRobot SmolVLA。与此同时，项目还有一个 file-backed Agent OS runtime：planner 写计划，tools 追加动作，watchdog 验证并通过 HAL driver 执行动作，环境状态再写回 Markdown 文件。未来如果要接真实机器人、真实 VLA 或真实仿真器，可以逐步替换 fake HAL driver、增加新的 VLA backend，或把 toy environment 换成真实 simulator，同时保留现有的 policy/runtime contract。
