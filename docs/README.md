# Documentation Index

这个目录分成两类内容：

## Project Docs

- [project_overview.md](project_overview.md): 项目整体定位、架构、模块说明和路线图。
- [embodied_agent_upgrade.md](embodied_agent_upgrade.md): 具身 Agent 升级说明，包括 fake 环境、policy loop 和数据记录。
- [running_guide.md](running_guide.md): 项目各功能模块的运行命令、快捷脚本和输出说明。
- [smolvla_rl_integration.md](smolvla_rl_integration.md): SmolVLA 和强化学习接入路线、脚本和评估入口。
- [dataset_quality.md](dataset_quality.md): Demonstration 数据质量检查、LeRobot 导出和小规模 policy benchmark。
- [architecture_diagram.md](architecture_diagram.md): 项目架构图、核心数据流、Agent OS 运行流和 SmolVLA/LeRobot 接入说明。

## Reference Materials

原始机器人代码和系统资料放在 [reference/](reference/) 下，用于后续迁移、重构和对照学习。

- [original_robot_control_framework_structure.txt](reference/original_robot_control_framework_structure.txt): 原始机器人控制系统目录结构说明。
- [original_stereo_hand_retargeting_pipeline.txt](reference/original_stereo_hand_retargeting_pipeline.txt): 双目手部三角化、retargeting、IK 和控制桥接相关代码资料。
- [original_realtime_rl_unitree_dds_node.txt](reference/original_realtime_rl_unitree_dds_node.txt): ROS2、Any2Track、Unitree DDS 实时节点相关代码资料。
- [original_full_body_arm_bridge_cpp.txt](reference/original_full_body_arm_bridge_cpp.txt): C++ full-body / arm SDK / Inspire Hand bridge 相关代码资料。

## External Reference Project Architecture

外部参考项目的源码放在本仓库外的 `/data/heyuhang/hyh/reference-repos/`，详细架构分析记录在 [reference_projects/](reference_projects/) 下。

- [reference_projects/README.md](reference_projects/README.md): 参考项目索引和总体结论。
- [reference_projects/phyagentos_architecture.md](reference_projects/phyagentos_architecture.md): PhyAgentOS 具身 Agent OS 架构拆解。
- [reference_projects/helloagents_architecture.md](reference_projects/helloagents_architecture.md): HelloAgents 通用 Agent 工程架构拆解。
- [reference_projects/upgrade_mapping.md](reference_projects/upgrade_mapping.md): 两个参考项目对当前项目的升级映射。
