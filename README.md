# Embodied Teleop Control Lab

一个面向具身智能入门和作品集展示的遥操作到机器人控制项目。

项目主线：

```text
stereo hand keypoints -> 3D hand skeleton -> retargeting -> safety limiter -> robot backend
```

当前版本先提供一个不依赖真实硬件的 MVP。它用合成手部数据模拟双目相机观测，恢复 3D 手部关键点，生成机器人手部控制量，并发送给 `FakeRobotBackend`。

## Why This Project

这类项目比单独写一个控制节点更适合作为具身智能作品，因为它覆盖了完整链路：

- 感知：双目 2D 关键点到 3D 手部骨架
- 动作表示：人手骨架到机器人手部命令
- 控制安全：关节限幅、速度限制
- 后端抽象：先接 fake backend，后续可替换为 Unitree / Inspire Hand / ROS2 backend
- 可扩展：后续可以接 IK、仿真、模仿学习和真实机器人

## Quick Start

```bash
cd /workspace/hyh/embodied-teleop-control-lab
python examples/run_fake_pipeline.py
```

期望输出类似：

```text
reconstruction_error_m 0.0
safe_command [...]
robot_frame_id 1
```

## Project Layout

```text
configs/       calibration, robot, safety configs
perception/    stereo triangulation and future keypoint detectors
retargeting/   human hand/arm pose to robot command mapping
kinematics/    IK solver adapters
control/       safety limiter and robot backend abstractions
ros_nodes/     future ROS2 nodes
sim/           future MeshCat / MuJoCo / Isaac Sim runners
tools/         recording, replay, visualization scripts
tests/         small correctness tests
examples/      runnable demos
```

## Roadmap

1. Replace synthetic hand keypoints with webcam or recorded detector output.
2. Add MeshCat visualization for hand skeleton and robot command.
3. Add ROS2 message bridge for command/state topics.
4. Replace `SimpleHandRetargeter` with robot-specific DexRetargeting.
5. Add Unitree G1 and Inspire Hand backends behind the same interface.
6. Record demonstrations and train a small policy for replay.

## Resume Angle

一句话介绍：

> Built an embodied teleoperation pipeline that converts stereo hand observations into safety-constrained robot commands, with modular perception, retargeting, and backend abstractions for simulation and real hardware.

