# AgentOS 架构图

## 1. 整体分层架构

```text
                         ┌──────────────────────────────┐
                         │         scripts/              │
                         │  run_agentos  benchmark       │
                         │  run_agent    run_watchdog    │
                         │  test_robosuite_env           │
                         └──────────┬───────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          v                         v                         v
  ┌───────────────┐     ┌───────────────────┐     ┌──────────────────┐
  │   learning/   │     │     runtime/      │     │     tools/       │
  │               │     │                   │     │                  │
  │ train_bc      │     │ planner           │     │ embodied_tools   │
  │ train_vis_bc  │     │ executor          │     │ robosuite_tools  │
  │ evaluate      │     │ watchdog          │     │ evaluation_tools │
  │ benchmark     │     │ action_queue      │     │ render_tools     │
  │ eval_report   │     │ action_validator  │     │ planner_tools    │
  │               │     │ repository        │     │ registry         │
  └───────┬───────┘     │ environment_io    │     └────────┬─────────┘
          │             │ file_io / watcher │              │
          │             │ trace / lessons   │              │
          │             │ skill_planner     │              │
          │             │ llm_planner       │              │
          │             │ plan_utils        │              │
          │             └────────┬──────────┘              │
          │                      │                         │
          v                      v                         v
  ┌───────────────┐     ┌───────────────────┐     ┌──────────────────┐
  │    agent/     │     │      hal/         │     │     envs/        │
  │               │     │                   │     │                  │
  │ scripted      │     │ BaseDriver ◄──────┼─────┤ FakeManipulation │
  │ robosuite_lift│     │ DriverRegistry    │     │ RobosuiteEnv     │
  │ bc / vision_bc│     │ FakeDriver        │     │ render / ppm     │
  │ vla / rl      │     │ RobosuiteDriver   │     │ task_utils       │
  └───────────────┘     └────────┬──────────┘     └──────────────────┘
                                 │
                                 v
                        ┌───────────────────┐
                        │    recorders/     │
                        │                   │
                        │ EpisodeRecorder   │
                        │ LeRobot Exporter  │
                        │ Dataset Inspector │
                        └───────────────────┘
```

**依赖方向：单向向上。`envs/` 和 `agent/` 为叶节点，无内部依赖。**

---

## 2. AgentOS 在线执行流程

```text
                         用户 / Benchmark
                              │
                              │  "lift the cube"
                              v
              ┌───────────────────────────────┐
              │          Planner               │
              │                               │
              │  SkillLibraryPlanner           │
              │    │ match "lift the cube"     │
              │    │ → robosuite_skill.md      │
              │    v                           │
              │  DeepSeekPlanner (fallback)    │
              │    │                           │
              │    v                           │
              │  RuleBasedPlanner (last resort)│
              └───────────────┬───────────────┘
                              │
                              │  TaskPlan
                              v
              ┌───────────────────────────────┐
              │       AgentOSExecutor          │
              │                               │
              │  for step in plan.steps:       │
              │    registry.run(tool, params)  │
              └───────────────┬───────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          v                   v                   v
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ reset_task   │   │ lift_loop    │   │ render_env   │
  │              │   │              │   │              │
  │ AppendAction │   │ StepEnvTool  │   │ driver.render│
  │  → reset     │   │  x N steps   │   │  → PPM image │
  │ RunWatchdog  │   │              │   │              │
  └──────┬───────┘   └──────┬───────┘   └──────────────┘
         │                  │
         v                  v
  ┌──────────────────────────────────────────────┐
  │                ACTION.md                      │
  │  ┌────────────────────────────────────┐      │
  │  │ actions:                           │      │
  │  │   - id: 1  type: reset  status: ✅ │      │
  │  │   - id: 2  type: env_step  ▶️      │      │
  │  │   - id: 3  type: env_step  ⏳      │      │
  │  └────────────────────────────────────┘      │
  └──────────────────────┬───────────────────────┘
                         │
                         v
  ┌──────────────────────────────────────────────┐
  │              Watchdog                         │
  │                                              │
  │  poll_once(driver, repo):                     │
  │    1. claim first pending action  (fcntl 锁) │
  │    2. validate_action()                       │
  │    3. driver.execute_action()                 │
  │    4. finish action status            (锁)   │
  └──────────────────────┬───────────────────────┘
                         │
                         v
  ┌──────────────────────────────────────────────┐
  │            Driver State Machine               │
  │                                              │
  │  DISCONNECTED → IDLE ⇄ EXECUTING             │
  │                    ↘         ↗               │
  │                     FAULT                     │
  │                      ↓                       │
  │                    CLOSED                     │
  │                                              │
  │  execute_action() wraps _execute_action():    │
  │    begin_action → IDLE→EXECUTING              │
  │    try: _execute_action(reset/env_step)       │
  │    except: mark_fault                         │
  │    finish_action → EXECUTING→IDLE             │
  └──────────────────────┬───────────────────────┘
                         │
                         v
  ┌──────────────────────────────────────────────┐
  │        ENVIRONMENT.md (回写)                  │
  │  ┌────────────────────────────────────┐      │
  │  │ robot:                             │      │
  │  │   ee_position: [0.12, -0.05, 1.1]  │      │
  │  │   gripper_closed: true             │      │
  │  │ objects:                           │      │
  │  │   cube: {position: [0.12, -0.05,..]}│     │
  │  │ episode:                           │      │
  │  │   success: false  step_count: 23   │      │
  │  │ runtime:                           │      │
  │  │   driver_state: executing          │      │
  │  │   capabilities: {...}              │      │
  │  └────────────────────────────────────┘      │
  └──────────────────────────────────────────────┘
```

---

## 3. Planner 层级与 Fallback 链

```text
                     ┌─────────────────────┐
                     │   Planner Protocol   │
                     │  plan(instruction,   │
                     │   target_color=None) │
                     │      → TaskPlan      │
                     └──────────┬──────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          v                     v                     v
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ SkillLibrary    │   │  DeepSeek       │   │  RuleBased      │
│ Planner         │   │  Planner        │   │  Planner        │
│                 │   │                 │   │                 │
│ SKILL.md ──→    │   │ API call ──→    │   │ instruction ──→ │
│ parse_skills()  │   │ _parse_json()   │   │ parse_target_   │
│ match_skill()   │   │                 │   │   color()       │
│ instantiate()   │   │ 9-point validate│   │                 │
│                 │   │                 │   │ fixed 3-step:   │
│ no match ───────┼───→ fallback ───────┼───→ reset_task      │
│                 │   │                 │   │ pick_place_loop │
│  170 lines      │   │  218 lines      │   │ render_fake_env │
└─────────────────┘   └─────────────────┘   │                 │
                                            │  60 lines       │
                                            └─────────────────┘

Fallback 链:  Skill  →  DeepSeek  →  RuleBased
              (模板)    (LLM+校验)    (硬编码)
```

---

## 4. 安全 / 校验链路

```text
        LLM 输出 (JSON text)
              │
              v
    ┌─────────────────────┐
    │  _parse_json()       │  提取 JSON (支持 fence)
    │  plan_from_dict()    │  构造 TaskPlan
    └─────────┬───────────┘
              │
              v
    ┌─────────────────────────────────────────────┐
    │         _validate_plan() — 9 点校验          │
    │                                             │
    │  1. target_color ∈ {red, blue, green}       │
    │  2. target_color_mismatch?                  │
    │  3. steps 非空                              │
    │  4. tool name ∈ WORKFLOW_TOOL_NAMES         │
    │  5. 无低层参数 {dx, dy, gripper}            │
    │  6. reset_task 存在                         │
    │  7. scripted_pick_place_loop 存在           │
    │  8. reset 参数: target_color / receptacle   │
    │  9. render_fake_env 位置合法 (最后)         │
    └─────────┬───────────────────────────────────┘
              │
              v
    ┌─────────────────────┐
    │  ToolRegistry.run()  │
    │     ↓                │
    │  AppendActionTool    │
    │     ↓                │
    │  ActionValidator     │
    │   - action_type 合法性│
    │   - action shape 合法 │
    │   - delta 不超限     │
    │   - workspace bounds │
    │   - capabilities 驱动│
    │     ↓                │
    │  ACTION.md (fcntl)   │
    │     ↓                │
    │  Watchdog            │
    │     ↓                │
    │  Driver State Machine│
    └─────────────────────┘

  任何一步失败 → 上一步 fallback
  LLM 输出 [dx, dy, gripper] → 第 5 点直接拒绝
```

---

## 5. 双执行路径

```text
  ┌─────────────────────────────────────────────────────────┐
  │            Online AgentOS Path                          │
  │                                                         │
  │  run_agentos.py / benchmark_agentos.py                  │
  │    → Planner.plan()                                     │
  │    → AgentOSExecutor.execute()                          │
  │    → ToolRegistry → ACTION.md → Watchdog → Driver       │
  │    → ENVIRONMENT.md → REPORT.md → TRACE.jsonl           │
  │                                                         │
  │  特点: 文件审计、可恢复、fcntl 安全、全栈               │
  │  用途: 生产执行、系统验证、benchmark                     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │            Offline Direct Path                           │
  │                                                         │
  │  evaluate_policy.py / train_bc.py / run_agent.py        │
  │    → env.reset() → policy.act(obs) → env.step(action)  │
  │    → run_episode()                                      │
  │                                                         │
  │  特点: 零文件 I/O、高速、批量                            │
  │  用途: 训练、数据采集、策略快速迭代                      │
  └─────────────────────────────────────────────────────────┘
```

---

## 6. Workspace 文件协议

```text
  workspace/{run_id}/episode_000/
  │
  ├── ACTION.md          "命令队列 —— fcntl 锁保护"
  │   ┌──────────────────────────────────────┐
  │   │ ```json                               │
  │   │ { "actions": [                        │
  │   │     { "id": "a1",                     │
  │   │       "action_type": "env_step",      │
  │   │       "parameters": {                 │
  │   │         "action": [0.02, 0, 0.01, 1]  │
  │   │       },                              │
  │   │       "status": "pending"  },         │
  │   │     { "id": "a2", "status": "..." }   │
  │   │   ] }                                 │
  │   │ ```                                   │
  │   └──────────────────────────────────────┘
  │
  ├── ENVIRONMENT.md     "运行时状态"
  │   ┌──────────────────────────────────────┐
  │   │ ```json                               │
  │   │ { "task": {"instruction": "lift..."},  │
  │   │   "robot": {"ee_position": [0,0,1]},  │
  │   │   "objects": {"cube": {...}},          │
  │   │   "runtime": {"driver_state": "idle", │
  │   │               "capabilities": {...}}   │
  │   │ }                                     │
  │   │ ```                                   │
  │   └──────────────────────────────────────┘
  │
  ├── EMBODIED.md        "驱动 profile"
  ├── LESSONS.md         "失败经验积累"
  ├── TASK.md            "当前任务摘要"
  ├── SKILL.md           "可复用 workflow 模板"
  ├── PLAN.md            "Planner 输出"
  └── REPORT.md          "执行报告"
```

---

## 7. Driver 双后端架构

```text
              ┌──────────────────────────┐
              │      BaseDriver (ABC)     │
              │                          │
              │  execute_action()         │
              │  _execute_action()        │
              │  get_environment()        │
              │  get_capabilities()       │
              │  get_runtime_state()      │
              │  load_environment()       │
              │  State Machine            │
              └──────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         v                               v
┌─────────────────────┐     ┌─────────────────────┐
│ FakeManipulation    │     │ RobosuiteDriver     │
│ Driver              │     │                     │
│                     │     │  task_name: "Lift"   │
│ action: [dx,dy,grp] │     │  robot: "Panda"     │
│ workspace: 2D       │     │  action: [dx,dy,dz, │
│ colors: red/blue/   │     │          grp]       │
│   green             │     │  workspace: 3D      │
│ env: FakeEnv        │     │  env: RobosuiteEnv  │
│                     │     │    Adapter          │
│ 183 lines           │     │                     │
└─────────────────────┘     │ 151 lines           │
                            └─────────────────────┘

通过 DriverRegistry 动态发现:
  driver_registry.register("fake_manipulation", FakeManipulationDriver)
  driver_registry.register("robosuite", RobosuiteDriver)  # ImportError guarded

CLI:
  --driver fake_manipulation   # 默认，2D
  --driver robosuite           # 3D MuJoCo
```

---

## 8. RobosuiteLiftPolicy — 5 阶段 FSM

```text
                    ┌──────────┐
                    │   init   │
                    └────┬─────┘
                         │
                         v
              ┌─────────────────────┐
              │  move_above_cube    │  ee xy → cube xy
              │                     │  gripper: OPEN
              │  xy_error > 0.025?  │
              └────────┬────────────┘
                       │ xy aligned
                       v
              ┌─────────────────────┐
              │  descend            │  ee z → cube z + offset
              │                     │  gripper: OPEN
              │  z > grasp_z?       │
              └────────┬────────────┘
                       │ z at grasp height
                       v
              ┌─────────────────────┐
              │  close_gripper      │  25 steps of GRIPPER_CLOSE
              │                     │
              │  close_count < 25?  │
              └────────┬────────────┘
                       │ 25 steps done
                       v
              ┌─────────────────────┐
              │  lift               │  ee z → cube z + 0.28
              │                     │  gripper: CLOSE
              │  z < lift_z?        │
              └────────┬────────────┘
                       │ z at target height
                       v
              ┌─────────────────────┐
              │  hold               │  [0, 0, 0, CLOSE]
              │  wait for success?  │
              └─────────────────────┘

每个阶段输出: [dx, dy, dz, gripper]
  dx/dy/dz = clip((target - ee) * position_gain, -max_delta, max_delta)
```

---

## 9. Skill 录制闭环

```text
     ┌────────────────────────────────────────────────────┐
     │                                                    │
     │   instruction ──→ SkillLibraryPlanner.plan()       │
     │                      │                             │
     │                      │ match?                      │
     │                      v                             │
     │               ┌──────────────┐                     │
     │               │ instantiate  │                     │
     │               │ skill →      │                     │
     │               │ TaskPlan     │                     │
     │               └──────┬───────┘                     │
     │                      │                             │
     │                      v                             │
     │               AgentOSExecutor                      │
     │                 .execute(plan)                     │
     │                      │                             │
     │                      │ success?                    │
     │                      v                             │
     │               ┌──────────────┐                     │
     │               │ record_plan  │                     │
     │               │ _as_skill()  │                     │
     │               │              │                     │
     │               │ derive       │                     │
     │               │ pattern:     │                     │
     │               │ "lift the    │                     │
     │               │  cube"       │                     │
     │               │              │                     │
     │               │ _template_   │                     │
     │               │ colors()     │                     │
     │               │              │                     │
     │               │ dedup?       │                     │
     │               │  pattern     │                     │
     │               │  already     │                     │
     │               │  exists?     │                     │
     │               └──────┬───────┘                     │
     │                      │                             │
     │                      v                             │
     │               ┌──────────────┐                     │
     │               │ SKILL.md     │  ◄── 写回 skill    │
     │               │  updated     │       library       │
     │               └──────────────┘                     │
     │                                                    │
     └────────────────────────────────────────────────────┘

  record_plan_as_skill() 逆向:
    执行:  {color} → "blue"   (_replace_placeholders)
    录制:  "blue" → {color}   (_template_colors)
```

---

## 10. 能力矩阵与演进

```text
       2D Fake Env               3D robosuite            Isaac Lab (A100)
  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │ PickPlace ✅     │    │ Lift ✅          │    │ Parallel envs    │
  │ Scripted ✅      │    │ PickPlace ⏳     │    │ Multi-camera     │
  │ VisionBC ✅      │    │ Stack ⏳         │    │ VLA inference    │
  │ VLA mock ✅      │    │ VisionBC ⏳      │    │ Large benchmark  │
  └──────────────────┘    │ VLA ⏳           │    └──────────────────┘
                           └──────────────────┘

  当前进度: ████████████░░░░░░░░  60%
  骨架+Planner+Skill+Benchmark+robosuite 全部完成
  下一步: PickPlace/Stack 策略 → 视觉策略 → Isaac Lab 并行
```

---

## 11. 项目统计一览

```text
  ┌─────────────────────────────────────────────┐
  │                                             │
  │   109 Python files    8,466 lines of code   │
  │    79 tests           17 markdown docs      │
  │     8 shell scripts   11 commits            │
  │                                             │
  │   ┌─────────┐  ┌─────────┐  ┌─────────┐    │
  │   │ runtime │  │  tools  │  │  agent  │    │
  │   │ 1,748 L │  │  590 L  │  │  552 L  │    │
  │   └─────────┘  └─────────┘  └─────────┘    │
  │   ┌─────────┐  ┌─────────┐  ┌─────────┐    │
  │   │ scripts │  │ learning│  │  envs   │    │
  │   │ 1,121 L │  │  929 L  │  │  699 L  │    │
  │   └─────────┘  └─────────┘  └─────────┘    │
  │   ┌─────────┐  ┌─────────┐  ┌─────────┐    │
  │   │   hal   │  │recorders│  │  tests  │    │
  │   │  609 L  │  │  600 L  │  │ 1,516 L │    │
  │   └─────────┘  └─────────┘  └─────────┘    │
  │                                             │
  │   0 TODO/FIXME    0 函数重复                │
  │   DAG 单向依赖    无循环引用                 │
  │                                             │
  └─────────────────────────────────────────────┘
```

---

## 12. 设计模式分布

| 模式 | 位置 | 示例 |
|------|------|------|
| Protocol | `planner.py`, `base_driver.py` | `Planner`, `CommandDriver`, `RuntimeDriver` |
| Template Method | `base_driver.py` | `execute_action() → _execute_action()` |
| Repository | `repository.py` | `WorkspaceRepository` + fcntl |
| Registry | `registry.py`, `drivers.py` | `ToolRegistry`, `DriverRegistry` |
| Chain of Responsibility | `llm_planner.py` | 9 checks → fallback |
| Strategy | `agent/` | `Policy.act(obs) → np.ndarray` |
| Observer | `trace.py` | `TraceLogger` JSONL |
| Facade | `executor.py` | `AgentOSExecutor` |
| CQRS | `base_driver.py` | `CommandDriver` / `QueryDriver` |
