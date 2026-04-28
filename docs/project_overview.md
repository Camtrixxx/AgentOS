# Embodied Teleop Control Lab 项目介绍

## 1. 项目定位

`Embodied Teleop Control Lab` 是一个面向具身智能学习、实验和作品集展示的机器人遥操作控制项目。

项目目标是构建一条清晰的机器人动作生成链路：

```text
人类动作观测 -> 3D 动作重建 -> 机器人动作重定向 -> 安全控制约束 -> 机器人后端执行
```

当前版本是一个最小可运行 MVP。它暂时不依赖真实摄像头、真实机器人或 ROS2 环境，而是用合成数据模拟双目相机输入，完整跑通从手部 3D 重建到机器人控制命令输出的流程。

这让项目具备两个优点：

- 可以在普通开发环境中快速验证核心算法和软件架构。
- 后续可以逐步替换模块，接入真实感知、仿真器、ROS2、Unitree G1 或 Inspire Hand。

## 2. 为什么这个项目适合具身智能方向

具身智能不是单独的视觉模型、控制器或大模型接口，而是一个完整闭环系统。一个机器人需要感知环境，理解动作目标，将人类或策略模型的意图转换成机器人可执行的动作，并在安全约束下执行。

本项目覆盖了具身智能工程中的几个关键环节：

| 环节 | 当前项目中的对应模块 |
| --- | --- |
| 感知输入 | 双目 2D 手部关键点 |
| 3D 表示 | 手部 3D skeleton |
| 动作重定向 | hand retargeting |
| 运动控制 | joint command |
| 安全约束 | joint limit 和 max delta limiter |
| 执行后端 | FakeRobotBackend |
| 后续扩展 | ROS2、DDS、仿真器、真实机器人 |

因此，它不是一个孤立脚本，而是一个可以逐步扩展成真实机器人系统的工程框架。

## 3. 当前数据流

当前 demo 的数据流如下：

```text
make_synthetic_hand()
        |
        v
project() 生成左右相机 2D 关键点
        |
        v
StereoHandTriangulation.triangulate()
        |
        v
SimpleHandRetargeter.retarget()
        |
        v
SafetyLimiter.limit()
        |
        v
FakeRobotBackend.send_joint_command()
```

对应运行入口是：

```bash
python examples/run_fake_pipeline.py
```

输出示例：

```text
reconstruction_error_m 2.1877147063396337e-16
safe_command [1.0, 0.8576, 0.7154, 0.5922, 0.4594, 0.3285]
robot_frame_id 1
```

其中：

- `reconstruction_error_m` 表示双目三角化恢复出的 3D 点和原始合成 3D 点之间的平均误差。
- `safe_command` 是经过动作重定向和安全限幅后的机器人手部命令。
- `robot_frame_id` 表示 fake robot 已经接收了第几帧控制命令。

## 4. 项目目录结构

```text
embodied-teleop-control-lab/
├── README.md
├── configs/
│   ├── camera.yaml
│   ├── robot.yaml
│   └── safety.yaml
├── control/
│   ├── fake_robot_backend.py
│   └── safety_limiter.py
├── docs/
│   └── project_overview.md
├── examples/
│   └── run_fake_pipeline.py
├── kinematics/
│   └── ik_solver.py
├── perception/
│   └── stereo_triangulation.py
├── retargeting/
│   └── simple_hand_retargeter.py
├── ros_nodes/
├── sim/
├── tests/
│   ├── test_safety_limiter.py
│   └── test_stereo_triangulation.py
└── tools/
```

## 5. 核心模块说明

### 5.1 `perception/stereo_triangulation.py`

这个模块负责双目三角化。

输入：

- 左相机 2D 关键点 `uv_left`
- 右相机 2D 关键点 `uv_right`
- 双目相机标定参数

输出：

- 3D 手部关键点，形状为 `(N, 3)`，单位为米

核心类：

```python
StereoCameraConfig
StereoHandTriangulation
```

设计重点：

- 不绑定具体 2D 检测器。
- 可以接入 MediaPipe、手部关键点模型、录制数据或其他视觉模型。
- 内部带有异常点修复逻辑，如果部分三角化点无效，会优先使用上一帧结果或有效点均值进行修复。

后续可以扩展：

- 加入相机畸变校正。
- 加入关键点置信度。
- 加入时间滤波，例如 One Euro Filter 或 Kalman Filter。
- 支持左右手同时三角化。

### 5.2 `retargeting/simple_hand_retargeter.py`

这个模块负责将人的 21 点手部骨架映射为机器人手部控制命令。

当前版本输出 6 维命令：

```text
[thumb, index, middle, ring, little, spread]
```

其中：

- 前 5 维表示五根手指的弯曲程度。
- 第 6 维表示手掌展开程度。

当前实现是一个启发式 baseline：

- 使用手腕到各个指尖的距离估计手指弯曲程度。
- 使用食指根部和小指根部距离估计手掌展开程度。

这个版本的意义不是追求最终控制精度，而是提供一个可运行、可替换的动作重定向接口。

后续可以替换为：

- DexRetargeting
- Inspire Hand 专用映射
- 基于优化的 retargeting
- 基于学习的 retargeting policy

### 5.3 `control/safety_limiter.py`

这个模块负责机器人控制安全。

核心功能：

- 关节命令上下限裁剪。
- 单步最大变化量限制。
- 保留上一帧命令，避免控制命令突变。

核心类：

```python
SafetyConfig
SafetyLimiter
```

在真实机器人系统中，安全模块非常重要。即使上游感知或策略模型输出异常，安全层也应该尽量避免机器人突然大幅运动。

后续可以扩展：

- command timeout 后自动回零。
- emergency stop。
- 不同关节使用不同速度限制。
- 速度、力矩、电流等多层限制。
- 与真实机器人状态反馈闭环结合。

### 5.4 `control/fake_robot_backend.py`

这个模块是一个无硬件机器人后端。

它的作用是：

- 接收机器人 joint command。
- 更新内部机器人状态。
- 记录命令日志。
- 让上层算法在没有真实机器人时也能开发和测试。

核心类：

```python
FakeRobotBackend
RobotState
```

后续真实机器人接入时，可以保持相同接口，实现：

```python
UnitreeG1Backend
InspireHandBackend
ROS2RobotBackend
MuJoCoRobotBackend
```

这样上层 pipeline 不需要关心底层到底是真机、仿真还是 fake backend。

### 5.5 `kinematics/ik_solver.py`

当前是一个占位 IK solver：

```python
IdentityIKSolver
```

它只是保持接口形状，方便后续接入真实逆解器。

后续可以接入：

- PinkIK
- CuroboIK
- Pinocchio
- MuJoCo inverse kinematics
- 自定义优化 IK

## 6. 配置文件说明

### 6.1 `configs/camera.yaml`

保存双目相机标定参数：

- 左相机内参 `k_left`
- 右相机内参 `k_right`
- 左到右旋转矩阵 `r_left_to_right`
- 左到右平移向量 `t_left_to_right_m`

当前是合成 demo 使用的简单标定。

### 6.2 `configs/robot.yaml`

保存机器人关节信息。

当前 fake hand 有 6 个关节：

```text
thumb
index
middle
ring
little
spread
```

后续可以增加：

- Unitree G1 双臂关节名
- Inspire Hand 关节名
- 腰部、头部、移动底盘等关节

### 6.3 `configs/safety.yaml`

保存安全控制参数：

- `joint_lower`
- `joint_upper`
- `max_delta_per_step`
- `command_timeout_s`

后续真实机器人部署时，应该把这些参数改成和具体机器人硬件一致。

## 7. 当前 MVP 的价值

当前版本虽然小，但已经具备一个完整系统的雏形：

```text
perception -> representation -> retargeting -> safety -> backend
```

这比单独写一个算法文件更重要，因为机器人系统真正难的地方往往是模块之间的数据边界、接口稳定性和安全约束。

当前 MVP 可以用于：

- 验证双目三角化逻辑。
- 验证手部动作重定向接口。
- 验证安全限幅行为。
- 作为接入真实相机、仿真器、ROS2 的基础。
- 作为具身智能项目的作品集雏形。

## 8. 推荐下一步开发计划

### 阶段一：可视化

目标：让项目可以直观看到手部骨架和控制命令。

建议添加：

- `sim/simple_viewer.py`
- 使用 Matplotlib 或 MeshCat 显示 3D hand skeleton。
- 显示 fake robot 的 6 维控制条。

完成后，项目会更适合展示。

### 阶段二：真实感知输入

目标：从合成数据切换到真实或录制数据。

建议添加：

- `tools/record_keypoints.py`
- `tools/replay_keypoints.py`
- `perception/hand_keypoint_detector.py`

可以先使用录制的 `.npy` 或 `.jsonl` 数据，不急着直接接摄像头。

### 阶段三：仿真后端

目标：把命令送到可视化机器人或仿真环境。

建议添加：

- `sim/meshcat_viewer.py`
- `control/mujoco_backend.py`

仿真比真实机器人更适合早期调试，因为可以反复测试异常情况。

### 阶段四：ROS2 / DDS 接入

目标：让项目开始接近真实机器人系统。

建议添加：

- `ros_nodes/teleop_node.py`
- `ros_nodes/control_bridge_node.py`
- `control/ros2_backend.py`

这一阶段可以参考你已有代码中的 Unitree DDS runner、ROS2 node、full body bridge 等设计。

### 阶段五：模仿学习

目标：从遥操作数据中学习一个简单策略。

建议添加：

- `tools/record_demo.py`
- `tools/replay_demo.py`
- `learning/train_policy.py`
- `learning/policy_inference.py`

数据格式可以先定义为：

```text
observation: hand_joints_3d / robot_state
action: safe_joint_command
timestamp: time
```

这一步会让项目更贴近具身智能中的 Learning from Demonstration。

## 9. 简历和面试表达

中文表达：

> 我实现了一个具身智能遥操作控制原型系统，包含双目手部 3D 重建、动作重定向、安全限幅和机器人后端抽象。系统当前支持无硬件 fake backend，后续可以扩展到 ROS2、Unitree G1、Inspire Hand 和仿真环境。

英文表达：

> Built an embodied teleoperation pipeline that converts stereo hand observations into safety-constrained robot commands, with modular perception, retargeting, and backend abstractions for simulation and real hardware.

可以强调的技术点：

- Stereo triangulation
- Hand pose retargeting
- Safety-constrained control
- Modular robot backend
- Simulation-first development
- ROS2 / DDS ready architecture

## 10. 当前项目的核心原则

这个项目后续扩展时建议坚持以下原则：

1. 每个模块只做一件事。
2. 算法和硬件接口解耦。
3. 先 fake backend，再仿真，最后真机。
4. 所有真实机器人控制前都经过 safety limiter。
5. 每个新增模块都尽量提供一个最小可运行 demo。
6. 数据格式要稳定，便于录制、回放和训练。

这样项目会逐步长成一个可靠的具身智能工程，而不是一组临时脚本。

## 11. 当前 VLA 接入状态

项目已经进入 VLA-ready 状态。

当前已经具备：

- RGB image observation
- language instruction
- state observation
- action `[dx, dy, gripper]`
- policy evaluation report
- VLA observation/action adapter
- mock VLA backend
- 可替换的 `VLAPolicy`

也就是说，真实 VLA 模型还没有接入，但工程接口已经准备好。下一步接真实模型时，主要工作会集中在新增 backend，而不是重写环境、`AgentLoop` 或评估系统。
