# Documentation Index

这个目录分成两类内容：

## Project Docs

- [project_overview.md](project_overview.md): 项目整体定位、架构、模块说明和路线图。
- [embodied_agent_upgrade.md](embodied_agent_upgrade.md): 具身 Agent 升级说明，包括 fake 环境、policy loop 和数据记录。

## Reference Materials

原始机器人代码和系统资料放在 [reference/](reference/) 下，用于后续迁移、重构和对照学习。

- [original_robot_control_framework_structure.txt](reference/original_robot_control_framework_structure.txt): 原始机器人控制系统目录结构说明。
- [original_stereo_hand_retargeting_pipeline.txt](reference/original_stereo_hand_retargeting_pipeline.txt): 双目手部三角化、retargeting、IK 和控制桥接相关代码资料。
- [original_realtime_rl_unitree_dds_node.txt](reference/original_realtime_rl_unitree_dds_node.txt): ROS2、Any2Track、Unitree DDS 实时节点相关代码资料。
- [original_full_body_arm_bridge_cpp.txt](reference/original_full_body_arm_bridge_cpp.txt): C++ full-body / arm SDK / Inspire Hand bridge 相关代码资料。

