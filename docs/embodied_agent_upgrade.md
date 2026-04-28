# Embodied Agent Upgrade

本次升级把项目从单条遥操作控制 demo 扩展成了一个小型具身 Agent 原型。

新增主线：

```text
language instruction -> observation -> policy -> action -> environment -> recorded episode
```

## 新增目录

```text
agent/
├── agent_loop.py
├── scripted_policy.py
└── vla_policy.py

envs/
└── fake_manipulation_env.py

datasets/
├── episode_recorder.py
└── schema.md

scripts/
├── run_agent.py
└── collect_demo.py
```

## 运行方式

运行一个 scripted embodied agent：

```bash
python scripts/run_agent.py
```

运行并记录数据：

```bash
python scripts/run_agent.py --record
```

采集多条 demonstrations：

```bash
python scripts/collect_demo.py --num-episodes 3
```

训练 behavior cloning policy：

```bash
python learning/train_bc.py --epochs 200
```

评估训练好的 BC policy：

```bash
python learning/evaluate_policy.py --policy bc --num-episodes 9
```

## 运行后发生的事

运行：

```bash
python scripts/run_agent.py
```

背后会发生下面这些步骤：

1. 创建 `FakeManipulationEnv`。
2. 创建 `ScriptedPickPlacePolicy`。
3. 创建任务：`pick up the red block and place it in the bowl`。
4. 调用 `env.reset(task)` 初始化环境。
5. 环境返回第一帧 `observation`。
6. `policy` 读取当前 `observation`。
7. `policy` 判断当前应该向 `red_block` 移动。
8. `policy` 输出动作 `[dx, dy, gripper]`。
9. `AgentLoop` 把动作传给 `env.step(action)`。
10. 环境更新末端执行器位置 `ee_position`。
11. 多次重复移动，直到末端执行器靠近 `red_block`。
12. `policy` 输出闭合夹爪动作。
13. 环境判断距离足够近，于是设置 `held_object = red_block`。
14. `policy` 判断已经抓住物体，开始向 `bowl` 移动。
15. 环境让 `red_block` 跟随 `ee_position` 移动。
16. 到达 `bowl` 附近后，`policy` 输出打开夹爪动作。
17. 环境释放 `red_block`。
18. 环境检查 `red_block` 是否已经在 `bowl` 附近。
19. 如果满足成功条件，环境返回 `done=True` 和 `success=True`。
20. `AgentLoop` 结束当前 episode。
21. 脚本打印 `success`、`steps` 和 `total_reward`。

可以把这条流程理解成最小版具身 Agent 闭环：

```text
observation -> policy -> action -> environment -> next_observation
```

当前系统已经具备：

- 读取任务语言。
- 观察世界状态。
- 根据状态选择动作。
- 通过动作改变环境。
- 判断任务是否完成。
- 可选地记录整条行为数据。

## 当前环境

`FakeManipulationEnv` 是一个二维桌面 pick-and-place 环境。

环境中包含：

- 一个末端执行器 `ee_position`
- 一个二值夹爪 `gripper_closed`
- 三个物体：红色、蓝色、绿色 block
- 一个目标容器 `bowl`

动作格式：

```text
[dx, dy, gripper]
```

其中：

- `dx, dy` 控制末端执行器移动。
- `gripper > 0` 表示闭合夹爪。
- `gripper <= 0` 表示打开夹爪。

观测格式包含：

- language instruction
- ee position
- gripper state
- held object
- object positions
- receptacle positions
- step count

## 当前 Agent

`ScriptedPickPlacePolicy` 是一个规则策略。

它会：

1. 从语言指令中解析目标颜色。
2. 移动到目标物体。
3. 闭合夹爪抓取。
4. 移动到 bowl。
5. 打开夹爪放置。

这个 policy 的价值不是智能本身，而是提供一个稳定的数据生成器和接口基线。后续可以用同样的 `act(observation) -> action` 接口替换为：

- BC policy
- Transformer policy
- Diffusion Policy
- OpenVLA adapter
- remote VLA inference service

## 数据记录

`EpisodeRecorder` 会把每条 rollout 保存成：

```text
data/demos/episode_xxxxxx/
├── metadata.json
├── transitions.jsonl
└── arrays.npz
```

这为后续 imitation learning 和 VLA fine-tuning 打基础。

## Behavior Cloning

当前项目已经加入第一版 BC imitation learning 闭环。

数据来源：

```text
ScriptedPickPlacePolicy -> EpisodeRecorder -> data/demos
```

训练目标：

```text
observation features -> action [dx, dy, gripper]
```

当前使用的 observation features 是状态特征，不是图像：

- 末端执行器位置 `ee_position`
- 目标物体位置
- `bowl` 位置
- 目标物体相对末端执行器的位置
- `bowl` 相对末端执行器的位置
- 夹爪是否闭合
- 当前是否抓住目标物体
- 目标颜色 one-hot

对应代码：

```text
learning/features.py
learning/demo_dataset.py
learning/models.py
learning/train_bc.py
learning/evaluate_policy.py
agent/bc_policy.py
```

这一步的意义是把项目从“规则 Agent”推进到“可学习 Agent”：

```text
采集专家数据 -> 训练策略模型 -> 替换规则策略 -> 在环境中评估成功率
```

## 后续建议

下一步最自然的升级是：

1. 增加更多随机初始位置，避免 BC 只记住固定轨迹。
2. 加 image observation，把 fake environment 渲染成简单 RGB 图。
3. 把 `FakeManipulationEnv` 替换或并列接入 LIBERO / robosuite / ManiSkill。
4. 实现 `VLAPolicy`，对接 OpenVLA 或其它 VLA 模型。
5. 加统一 `RobotAction`，把 Agent、teleop 和后续 C++ 控制 bridge 接到同一套动作接口。
