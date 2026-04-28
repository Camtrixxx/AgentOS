# Running Guide

这份文档说明项目各个功能模块怎么运行，以及每条命令在做什么。

## 0. 进入项目

```bash
cd /workspace/hyh/embodied-teleop-control-lab
```

当前项目不需要真实机器人即可运行。默认使用 fake environment、fake robot backend 和本地数据。

## 1. 一键 Smoke Test

最快确认项目能跑：

```bash
bash scripts/sh/00_smoke_test.sh
```

它会依次运行：

1. 遥操作到控制管线 demo。
2. scripted embodied agent。
3. fake 环境 RGB 渲染。
4. mock VLA 随机布局评估并生成报告。

输出产物：

```text
outputs/fake_env.ppm
outputs/eval_reports_smoke/
```

## 2. 遥操作到控制管线

命令：

```bash
python examples/run_fake_pipeline.py
```

功能：

```text
synthetic 3D hand
-> stereo projection
-> triangulation
-> hand retargeting
-> safety limiter
-> FakeRobotBackend
```

你会看到：

```text
reconstruction_error_m ...
safe_command [...]
robot_frame_id 1
```

这个 demo 对应项目里的早期机器人控制主线。

## 3. Embodied Agent Demo

运行 scripted agent：

```bash
python scripts/run_agent.py
```

运行带 RGB image observation 的 agent：

```bash
python scripts/run_agent.py --include-image
```

运行随机化布局：

```bash
python scripts/run_agent.py --randomize-layout --max-steps 100
```

功能：

```text
language instruction
-> FakeManipulationEnv
-> ScriptedPickPlacePolicy
-> action [dx, dy, gripper]
-> success / failure
```

## 4. RGB Render

导出 fake 环境预览图：

```bash
python scripts/render_fake_env.py --output outputs/fake_env.ppm
```

输出：

```text
outputs/fake_env.ppm
```

这张图是 top-down RGB observation，后续 VisionBC / VLA 都可以使用类似输入。

## 5. State BC: 采集、训练、评估

### 5.1 采集 state demonstrations

```bash
bash scripts/sh/01_collect_state_demos.sh
```

等价于：

```bash
python scripts/collect_demo.py --num-episodes 60 --output-dir data/demos
```

输出：

```text
data/demos/episode_xxxxxx/
```

### 5.2 训练 state BC

```bash
bash scripts/sh/02_train_state_bc.sh
```

等价于：

```bash
python learning/train_bc.py \
  --data-dir data/demos \
  --epochs 300 \
  --batch-size 128 \
  --output checkpoints/bc_policy.pt
```

输出：

```text
checkpoints/bc_policy.pt
```

### 5.3 评估 state BC

```bash
python learning/evaluate_policy.py \
  --policy bc \
  --checkpoint checkpoints/bc_policy.pt \
  --num-episodes 9 \
  --write-report
```

输出：

```text
outputs/eval_reports/
```

## 6. VisionBC: 随机布局视觉模仿学习

这是当前项目最推荐展示的学习链路。

### 6.1 采集随机布局视觉 demonstrations

```bash
bash scripts/sh/03_collect_vision_demos_random.sh
```

等价于：

```bash
python scripts/collect_vision_demo.py \
  --num-episodes 120 \
  --output-dir data/vision_demos_random \
  --randomize-layout
```

输出：

```text
data/vision_demos_random/episode_xxxxxx/
├── images/
├── metadata.json
├── transitions.jsonl
└── arrays.npz
```

### 6.2 训练随机布局 VisionBC

```bash
bash scripts/sh/04_train_vision_bc_random.sh
```

等价于：

```bash
python learning/train_vision_bc.py \
  --data-dir data/vision_demos_random \
  --epochs 120 \
  --batch-size 128 \
  --output checkpoints/vision_bc_random_policy.pt
```

输出：

```text
checkpoints/vision_bc_random_policy.pt
```

### 6.3 评估随机布局 VisionBC

```bash
bash scripts/sh/05_eval_vision_bc_random.sh
```

等价于：

```bash
python learning/evaluate_policy.py \
  --policy vision_bc \
  --checkpoint checkpoints/vision_bc_random_policy.pt \
  --num-episodes 18 \
  --randomize-layout \
  --max-steps 100 \
  --write-report
```

输出：

```text
outputs/eval_reports/
```

## 7. VLA-Ready Mock Backend

当前项目已经具备 VLA 接口层，但还没有接真实大模型。

运行 mock VLA：

```bash
bash scripts/sh/06_eval_mock_vla.sh
```

等价于：

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend mock \
  --num-episodes 18 \
  --randomize-layout \
  --max-steps 100 \
  --write-report
```

流程：

```text
observation
-> FakeEnvVLAAdapter
-> VLAObservation(image, instruction, state)
-> MockVLABackend.predict()
-> VLAAction(ee_delta, gripper)
-> env action [dx, dy, gripper]
```

真实 VLA 后续只需要新增 backend，实现：

```python
predict(observation: VLAObservation) -> VLAAction
```

## 8. Evaluation Reports

所有 policy 都可以用同一个评估入口：

```bash
python learning/evaluate_policy.py --policy scripted --write-report
python learning/evaluate_policy.py --policy bc --checkpoint checkpoints/bc_policy.pt --write-report
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_random_policy.pt --write-report
python learning/evaluate_policy.py --policy vla --vla-backend mock --write-report
```

报告输出：

```text
outputs/eval_reports/
├── eval_YYYYMMDD_HHMMSS_policy.json
└── eval_YYYYMMDD_HHMMSS_policy.md
```

报告包含：

- policy 名称
- checkpoint
- success rate
- average steps
- average reward
- 每个 episode 的成功/失败情况

## 9. 常用环境变量

快捷脚本支持用环境变量覆盖默认参数。

示例：

```bash
NUM_EPISODES=30 bash scripts/sh/03_collect_vision_demos_random.sh
EPOCHS=40 bash scripts/sh/04_train_vision_bc_random.sh
CHECKPOINT=checkpoints/vision_bc_random_policy.pt bash scripts/sh/05_eval_vision_bc_random.sh
```

## 10. 生成文件说明

以下目录是运行后生成的，本地存在但不会进入 Git：

```text
data/
outputs/
checkpoints/
```

如果你刚 clone 项目，这些目录可能不存在。运行采集、训练或评估命令后会自动创建。

如果磁盘空间不足，可以清理这些可再生本地产物：

```bash
bash scripts/sh/99_clean_generated.sh
```

注意：这会删除本地采集数据、评估报告和训练 checkpoint；需要时可以重新运行采集和训练脚本生成。
