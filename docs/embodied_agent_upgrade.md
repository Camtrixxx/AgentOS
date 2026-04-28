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

开启 RGB image observation：

```bash
python scripts/run_agent.py --include-image
```

导出当前 fake 环境的 RGB 预览图：

```bash
python scripts/render_fake_env.py --output outputs/fake_env.ppm
```

采集视觉 demonstrations：

```bash
python scripts/collect_vision_demo.py --num-episodes 60
```

训练 VisionBCPolicy：

```bash
python learning/train_vision_bc.py --epochs 80
```

评估 VisionBCPolicy：

```bash
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_policy.pt
```

采集随机化布局 demonstrations：

```bash
python scripts/collect_vision_demo.py --num-episodes 120 --randomize-layout
```

在随机化布局上评估：

```bash
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_policy.pt --randomize-layout --max-steps 100
```

生成评估报告：

```bash
python learning/evaluate_policy.py \
  --policy vision_bc \
  --checkpoint checkpoints/vision_bc_random_policy.pt \
  --randomize-layout \
  --max-steps 100 \
  --write-report
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
- optional RGB image observation

## RGB Render 和 Image Observation

当前环境已经支持 top-down RGB 渲染：

```python
image = env.render_rgb()
```

图像格式：

```text
shape: (128, 128, 3)
dtype: uint8
layout: RGB
```

画面中包含：

- 浅色桌面背景和网格。
- 红、蓝、绿三个 block。
- 黄色 `bowl`。
- 黑色末端执行器和夹爪状态。

默认 observation 不包含 image，避免让当前 JSONL demonstration 过大。需要图像时，可以通过 `FakeManipulationConfig(include_image=True)` 打开：

```python
config = FakeManipulationConfig(
    workspace_low=np.array([-1.0, -1.0]),
    workspace_high=np.array([1.0, 1.0]),
    include_image=True,
)
env = FakeManipulationEnv(config=config)
observation = env.reset(task)
image = observation["image"]
```

这一步的意义是让项目开始具备 VLA 需要的输入形态：

```text
language instruction + RGB image + state -> action
```

当前 BC policy 仍然使用状态特征训练，没有使用图像。下一步可以新增 `VisionBCPolicy` 或 `VLAPolicy`，读取 `observation["image"]` 和 `observation["instruction"]`。

## Vision Behavior Cloning

当前项目已经加入第一版视觉模仿学习闭环。

新增数据流：

```text
ScriptedPickPlacePolicy
-> FakeManipulationEnv(include_image=True)
-> VisionEpisodeRecorder
-> data/vision_demos
-> train_vision_bc.py
-> VisionBCPolicy
-> evaluate_policy.py --policy vision_bc
```

视觉数据目录结构：

```text
data/vision_demos/episode_000000/
├── images/
│   ├── 000000.npy
│   ├── 000000_next.npy
│   └── ...
├── metadata.json
├── transitions.jsonl
└── arrays.npz
```

`transitions.jsonl` 中不会直接保存大图像数组，而是保存相对路径：

```json
{
  "image_path": "images/000000.npy",
  "action": [0.03, 0.02, -1.0]
}
```

`VisionBCPolicy` 使用一个小 CNN 编码 RGB 图像，再拼接任务颜色 one-hot，最后输出动作：

```text
RGB image + target color -> [dx, dy, gripper]
```

这一步比直接接真实 VLA 更稳，因为它先把视觉输入、语言/任务条件、动作输出和评估闭环打通了。

## Layout Randomization

当前 fake 环境已经支持随机化初始布局。

打开方式：

```python
config = FakeManipulationConfig(
    workspace_low=np.array([-1.0, -1.0]),
    workspace_high=np.array([1.0, 1.0]),
    randomize_layout=True,
)
env = FakeManipulationEnv(config=config, seed=0)
```

随机化内容：

- 红、蓝、绿 block 会在各自基础位置附近随机扰动。
- `bowl` 会在基础位置附近随机扰动。
- 物体之间保持最小距离，避免完全重叠。
- 所有位置都会裁剪到 workspace 内部。

这一步的意义是从“固定轨迹模仿”推进到“分布内泛化”：

```text
fixed layout demos -> randomized layout demos -> randomized layout evaluation
```

如果模型只在固定布局上训练，它很容易记住轨迹；如果在随机化布局上训练和评估，它必须学会根据 observation 调整动作。

## Evaluation Report

当前评估脚本支持生成 JSON 和 Markdown 报告。

命令：

```bash
python learning/evaluate_policy.py \
  --policy vision_bc \
  --checkpoint checkpoints/vision_bc_random_policy.pt \
  --num-episodes 18 \
  --randomize-layout \
  --max-steps 100 \
  --write-report
```

输出目录：

```text
outputs/eval_reports/
├── eval_YYYYMMDD_HHMMSS_vision_bc.json
└── eval_YYYYMMDD_HHMMSS_vision_bc.md
```

报告内容包括：

- policy 名称
- checkpoint 路径
- episode 数量
- max steps
- 是否随机化布局
- success rate
- average steps
- average reward
- 每个 episode 的 target color、success、steps、reward、failure reason

这一步让项目开始具备“实验记录”能力。后续比较 `scripted`、`bc`、`vision_bc`、`vla` 时，不需要只靠终端输出判断结果。

## VLA-Ready Interface

当前项目已经具备可替换的 VLA 接口层。

新增结构：

```text
adapters/
└── vla_adapter.py

vla/
├── base.py
└── mock_backend.py

agent/
└── vla_policy.py
```

核心流程：

```text
FakeManipulationEnv observation
-> FakeEnvVLAAdapter
-> VLAObservation(image, instruction, state)
-> VLABackend.predict(...)
-> VLAAction(ee_delta, gripper)
-> FakeEnv action [dx, dy, gripper]
```

`VLAPolicy` 现在不再是空占位，而是可以接入 `AgentLoop` 的 policy wrapper。

当前可用 backend：

```text
MockVLABackend
```

它模拟真实 VLA 后端的接口：

```python
predict(observation: VLAObservation) -> VLAAction
```

运行方式：

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend mock \
  --randomize-layout \
  --max-steps 100 \
  --write-report
```

这一步的意义是：以后接 OpenVLA、LeRobot、SmolVLA、Pi0 或远程模型服务时，不需要改 `AgentLoop`，只需要新增一个 backend，实现同样的 `predict()` 接口。

未来真实 VLA backend 可以长这样：

```python
class OpenVLABackend:
    name = "openvla"

    def predict(self, observation: VLAObservation) -> VLAAction:
        image = observation.image
        instruction = observation.instruction
        raw_action = self.model.predict(image=image, instruction=instruction)
        return self.action_adapter(raw_action)
```

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
