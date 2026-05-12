# Reference Projects Architecture Notes

这个目录记录两个外部参考项目的架构设计，以及它们对本项目后续升级的启发。

源码没有复制进本仓库，避免把大型第三方项目和许可证边界混进当前项目。当前本地参考路径是：

```text
/data/heyuhang/hyh/reference-repos/PhyAgentOS
/data/heyuhang/hyh/reference-repos/HelloAgents
```

## Documents

- [phyagentos_architecture.md](phyagentos_architecture.md): PhyAgentOS 的具身智能 Agent OS 架构拆解。
- [helloagents_architecture.md](helloagents_architecture.md): HelloAgents 的通用多智能体工程架构拆解。
- [upgrade_mapping.md](upgrade_mapping.md): 两个参考项目如何映射到当前 `embodied-teleop-control-lab`。

## High-Level Takeaway

`PhyAgentOS` 更适合参考具身智能运行时设计：Markdown 文件协议、HAL driver、watchdog、机器人 profile、安全校验和多机器人 workspace。

`HelloAgents` 更适合参考通用 Agent 工程化：工具响应协议、工具注册表、上下文管理、会话持久化、任务进度、trace 可观测性和技能外化。

本项目当前不应该直接照搬两个框架，而应该抽取轻量设计，形成自己的最小闭环：

```text
Policy / Agent
-> ACTION.md
-> watchdog
-> HAL driver
-> FakeManipulationEnv / FakeRobotBackend
-> ENVIRONMENT.md
-> evaluation report
```

