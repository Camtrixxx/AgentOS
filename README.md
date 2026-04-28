# Embodied Teleop Control Lab

一个 simulation-first 的具身智能学习项目，用来从零搭建：

```text
teleoperation/control -> embodied agent loop -> imitation learning -> vision policy -> VLA-ready interface
```

当前项目不依赖真实机器人即可运行。它包含 fake manipulation 环境、RGB observation、demonstration collection、BC / VisionBC 训练、随机化布局评估、evaluation report，以及可替换的 VLA backend 接口。

## What It Does

项目目前有两条主线。

第一条是遥操作到控制管线：

```text
synthetic hand keypoints
-> stereo triangulation
-> hand retargeting
-> safety limiter
-> fake robot backend
```

第二条是具身 Agent 学习闭环：

```text
language instruction
-> RGB image + state observation
-> policy / VLA backend
-> action [dx, dy, gripper]
-> fake manipulation environment
-> evaluation report
```

已经实现的能力：

- 双目手部 3D 重建 demo
- 手部 retargeting baseline
- safety limiter 和 fake robot backend
- 语言条件 pick-and-place fake 环境
- RGB render 和 image observation
- scripted expert policy
- state BC imitation learning
- VisionBC imitation learning
- 随机化布局训练和评估
- JSON / Markdown evaluation report
- VLA-ready adapter 和 mock VLA backend

## Quick Start

进入项目：

```bash
cd /workspace/hyh/embodied-teleop-control-lab
```

最快确认项目能跑：

```bash
bash scripts/sh/00_smoke_test.sh
```

它会运行：

```text
teleop/control demo
scripted embodied agent
RGB render
mock VLA randomized evaluation report
```

详细运行说明见：

[docs/running_guide.md](docs/running_guide.md)

## Common Commands

运行遥操作到控制 demo：

```bash
python examples/run_fake_pipeline.py
```

运行具身 Agent：

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
adapters/     observation/action adapters for VLA-style backends
agent/        scripted, BC, VisionBC, and VLA policy wrappers
configs/      camera, robot, and safety configs
control/      safety limiter and fake robot backend
datasets/     episode recorders and dataset schema
docs/         project docs and running guide
envs/         fake manipulation environment
evaluation/   JSON / Markdown evaluation reports
examples/     runnable small demos
kinematics/   IK placeholders
learning/     BC / VisionBC models, datasets, training, evaluation
perception/   stereo triangulation
retargeting/  hand retargeting baseline
scripts/      Python entrypoints and shell shortcuts
sim/          future simulator integrations
tests/        lightweight tests
vla/          VLA backend protocol and mock backend
```

## VLA-Ready Design

真实 VLA 模型还没有接入，但接口已经准备好：

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
```

如果磁盘空间不足，可以清理：

```bash
bash scripts/sh/99_clean_generated.sh
```

## Documentation

- [docs/running_guide.md](docs/running_guide.md): 怎么运行每个模块
- [docs/project_overview.md](docs/project_overview.md): 项目整体介绍和设计目标
- [docs/embodied_agent_upgrade.md](docs/embodied_agent_upgrade.md): Agent / BC / VisionBC / VLA-ready 详细演进
- [datasets/schema.md](datasets/schema.md): demonstration 数据格式

## Resume Angle

一句话介绍：

> Built a simulation-first embodied intelligence lab with stereo hand reconstruction, safety-constrained control, language-conditioned fake manipulation, demonstration collection, BC/VisionBC training, randomized evaluation, and a VLA-ready backend interface.
