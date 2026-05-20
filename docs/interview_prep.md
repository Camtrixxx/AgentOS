# AgentOS 面试准备资料

---

## 一、30 秒项目简介

> 我独立设计并实现了一个仿真优先的具身 Agent 执行框架。它的核心贡献是：在 AI Agent 的"决策"和物理环境的"执行"之间，建立了一套文件驱动的多层安全协议。12 个迭代 commit，110+ 文件，8800+ 行 Python，79 个测试全部通过。支持 2D 和 3D 双仿真后端，三层 Planner 体系（模板匹配 / LLM+校验 / 硬编码 fallback），外加完整的帧序列可视化和 benchmark 评估系统。

---

## 二、3 分钟项目讲解

### 2.1 问题

具身 AI 系统有一个被忽视的安全问题：**LLM/Agent 产出的动作，在被物理执行之前，应该经过什么样的防线？**

当前业界两极分化：
- 研究代码（ManiSkill/robosuite 示例）：Agent 直接调 `env.step()`，零安全边界
- 工业中间件（ROS 2）：完整通信/安全/生命周期，但不关心 Agent 决策层

### 2.2 我的方案

我做了一个中间层——用文件协议把"决策"和"执行"解耦：

```
Planner → Tool → ACTION.md (fcntl锁) → Watchdog → Validator → Driver State Machine → 物理执行
```

每层一个职责，每层独立的失败路径。换 Planner 不改执行链，换仿真后端不改协议。

### 2.3 技术亮点

- **三层 Planner 体系**：SkillLibraryPlanner（模板匹配，可审计）+ DeepSeekPlanner（LLM + 9 点校验 + 自动 fallback）+ RuleBasedPlanner（最后防线）
- **驱动状态机**：DriverState 枚举，BaseDriver 用 Template Method 模式包裹所有物理操作，不允许绕过
- **文件锁 + 原子写**：ACTION.md 的 read-modify-write 用 fcntl.LOCK_EX 保护，写入用 mkstemp + fsync + os.replace，断电安全
- **动态能力校验**：ActionValidator 从 ENVIRONMENT.md 的 runtime.capabilities 读取当前 driver 的能力声明（action_dims、workspace_bounds、receptacles），换 driver 不用改 validator
- **双仿真后端**：FakeManipulationEnv（2D toy，快速验证）+ Robosuite + MuJoCo（3D 物理，Panda 机械臂）
- **完整可视化**：执行过程每 N 步渲染一帧，加 stage/action 标注，自动生成 GIF/WebP/HTML 播放器

### 2.4 关键数字

- 12 个增量 commit，每次保留全部已有功能
- 79 个测试，零 TODO/FIXME，零函数重复
- DAG 单向依赖，10 层目录结构，无循环引用
- 从 `pip install` 到 `run_agentos.py` 成功执行：< 5 分钟

---

## 三、常见面试问题 & 回答要点

### Q1: 为什么用文件作为 Agent 和环境的通信协议，而不是直接函数调用？

**回答要点：**
1. **可审计**：每一步都有落盘记录，出问题可以回溯到具体的 ACTION.md 条目
2. **可恢复**：系统崩溃后重启，workspace 文件还在，可以从上次状态继续
3. **安全解耦**：Agent 只能写文件，不能直接调硬件。watchdog 是唯一消费 ACTION.md 的进程
4. **多进程天然支持**：Planner、Watchdog、Driver 可以跑在不同进程甚至不同机器上
5. **不是数据库**：文件协议是正确抽象层。后续如果需要高性能，可以换 Redis/NATS backend，保持协议语义

### Q2: 为什么有三个 Planner？是不是过度设计？

**回答要点：**
1. **SkillLibraryPlanner**：模板匹配，O(1) 决策，可审计可版本化。适合已知任务
2. **DeepSeekPlanner**：LLM 语义理解。适合模糊指令和未见过的任务组合。但当 tool 数量少时，它和 SkillLibraryPlanner 输出几乎相同
3. **RuleBasedPlanner**：最后防线，永远可用。当 LLM API 挂了、skill 没匹配时，保证系统不崩溃

关键是 fallback 链：`skill → deepseek → rule`。每一层失败自动降级，对用户透明。

不是过度设计——三层各解决不同问题，各有各的不可替代性。

### Q3: LLM 输出的安全性怎么保证？

**回答要点：**
LLM planner 的核心安全机制是 **9 点本地校验**：

1. target_color 必须在合法集合内
2. 不能有 target_color 不一致
3. steps 不能为空
4. tool 名必须在白名单内
5. 禁止 LLM 输出 `[dx, dy, gripper]` 低层动作（`LOW_LEVEL_PARAMETER_NAMES` 检查）
6. reset_task 必须存在
7. 执行步骤必须存在
8. reset 参数必须正确（target_color、receptacle_name）
9. render 步骤位置合法

任何一条不通过 → `PlanValidationError` → fallback 到 RuleBasedPlanner。

最坏情况：LLM 被 prompt-injected 输出恶意 JSON → 校验拒绝 → 系统仍然用一个安全的三步计划正常执行。

### Q4: fcntl 文件锁在这个项目里具体是怎么用的？

**回答要点：**
`ACTION.md` 是 AgentOS 的核心数据竞争点——Agent 往里面写新动作，Watchdog 从中取 pending 动作并改状态。

`WorkspaceRepository.update_actions(mutator)` 的流程：
1. 打开 `.ACTION.md.lock`
2. `fcntl.flock(LOCK_EX)` — 排他锁
3. `load_action_document()` — 读
4. `mutator(document)` — 改（例如标记 action 从 pending→running）
5. `save_action_document()` — 原子写（mkstemp + fsync + os.replace）
6. `fcntl.flock(LOCK_UN)` — 释放锁

多个进程/线程同时操作 ACTION.md 时，锁保证不会有 lost-update。

### Q5: 这个项目和你见过的其他 Agent 项目最大的区别是什么？

**回答要点：**
大部分 Agent 项目是一个巨大的 `Agent.run()` 方法，里面混着 prompt 构造、工具调用、环境交互、日志打印。

AgentOS 是 5 层独立职责：
- Planner 是纯函数（instruction → TaskPlan）
- Tool 是执行器（不关心谁调的）
- Watchdog 是仲裁者（唯一的 ACTION.md 消费方）
- Driver 有状态机 guard（不允许绕过）
- Validator 从 driver capabilities 动态读规则

换任何一层不影响其他层。这种可替换性在 AI Agent 代码里非常罕见。

### Q6: 如果有 100 个并发的 Agent 任务，现在的架构怎么扩展？

**回答要点：**
两个层面：

**水平扩展**：每个 Agent 任务一个独立 workspace 目录。workspace 之间完全隔离。benchmark 已经在做这件事（每个 episode 一个 workspace/benchmarks/{run_id}/episode_{N}/）。

**垂直扩展**：ACTION.md 的文件 I/O 在单任务下 < 0.5ms/步，95 步的 Lift 任务总 I/O 约 50ms。并发 100 个任务时文件系统会成为瓶颈——但 WorkspaceRepository 的接口设计支持替换 backend。`get_actions()` / `save_actions()` / `update_actions()` 三个方法，换成 Redis 的 LPUSH/BRPOP 即可，协议语义不变。

### Q7: 为什么 robosuite + MuJoCo，不用 Isaac Lab？

**回答要点：**
当前硬件是华为 Ascend NPU（aarch64 + CANN），没有 NVIDIA GPU。
- MuJoCo CPU 模式直接可用，单臂操作物理计算 < 3ms/步
- robosuite 预置 PickPlace/Lift/Stack 任务，安装 `pip install mujoco robosuite`
- Isaac Lab 需要 CUDA + PhysX GPU，当前环境不兼容

但架构已经为 Isaac Lab 做好了准备：`RobosuiteEnvAdapter` 和 `RobosuiteDriver` 可以平行替换为 `IsaacLabEnvAdapter` 和 `IsaacLabDriver`，不改任何 runtime 代码。等迁移到 A100 后接上。

### Q8: 项目里你个人最满意的设计是什么？

**回答要点（选 2-3 个根据自己的真实感受）：**

1. **Skill 录制闭环**：执行成功 → 自动推导 pattern → 参数化模板 → 去重 → 写回 SKILL.md。`_template_colors` 和 `_replace_placeholders` 是精确的逆操作。这个闭环让系统能从经验中学习。

2. **RobosuiteLiftPolicy 的 5 阶段 FSM**：不是玩具——有比例控制 `clip((target - ee) * gain, -max_delta, max_delta)`，有状态记忆 `_close_count`，有自动复位。简洁但完整。

3. **`_stabilize_orientation`**：MuJoCo OSMesa 渲染有时翻转图像。这个函数逐帧对比翻转前后的差异，自动选和上一帧更接近的方向。30 行解决了一个 tricky 的工程问题。

4. **12 个 commit 的增长方式**：每次都是增量，每次保留全部已有功能。没有一次推倒重写。

---

## 四、面试中可能用到的具体数字

| 指标 | 数值 | 讲的时候怎么用 |
|------|------|---------------|
| Python 文件 | 110+ | "代码组织在 10 层目录里" |
| 总代码行 | 8,800+ | "中等体量，一个人可以完整理解" |
| 测试数 | 79 | "全部通过，含 mock/集成/环境 guard" |
| commits | 12 | "每次增量，保留全部已有功能" |
| Planner 层数 | 3 | "模板匹配 → LLM+校验 → 硬编码 fallback" |
| LLM 校验点数 | 9 | "任何一条失败→自动降级" |
| Lift FSM 阶段数 | 5 | "对齐→下探→闭合→上抬→保持" |
| 仿真后端 | 2 | "2D fake + 3D MuJoCo robosuite" |
| TODO/FIXME | 0 | "零临时方案，零技术债" |

---

## 五、准备建议

### 自我介绍框架

```
1. 我做了什么 (30秒)
   → "我独立设计并实现了一个仿真优先的具身 Agent 执行框架"

2. 为什么做 (30秒)  
   → "AI Agent 输出动作 → 物理执行 之间有安全真空"

3. 怎么做 (60秒)
   → 文件协议解耦 + 三层规划 + 状态机执行 + 动态校验

4. 技术亮点 (60秒)
   → 选 2-3 个你最熟悉的（建议：LLM 安全链路 + Skill 闭环 + 5 阶段 FSM）

5. 结果 (30秒)
   → 12 commits, 79 tests, 双后端, 零技术债
```

### 带一个 demo

如果面试允许屏幕共享，准备这个命令：

```bash
python scripts/run_agentos.py "lift the cube" \
  --driver robosuite --planner skill \
  --skill-path runtime/skills/robosuite_skill.md
```

然后打开 `outputs/robosuite_lift_viz/viewer.html` 展示可视化回放。

### 如果被问到"你做的东西能落地吗"

```
"这个项目定位是参考实现（reference implementation），不是产品。
它验证了协议的可替换性——换个 Planner、换个仿真后端、
换个通信 backend，不改任何 runtime 代码。

如果有人要在真实的 Franka Panda 机器人上部署，
只需要写一个 ~150 行的 PandaDriver（继承 BaseDriver），
实现 _execute_reset 和 _execute_env_step 两个方法。
整个 AgentOS 协议栈可以原封不动地用。"
```

### 如果被问到"你和 ROS 的区别是什么"

```
"ROS 解决的是传感器→控制器的通信问题（topic/service/action），
AgentOS 解决的是 AI Agent→控制器的安全边界问题。

两者不冲突，可以叠加：
AgentOS 做 Planner → Watchdog 这一层，
ROS 做 Driver → 电机控制这一层。

事实上 BaseDriver 的 CQRS 协议（CommandDriver/QueryDriver/RuntimeDriver）
就是为这种叠加设计的。"
```
