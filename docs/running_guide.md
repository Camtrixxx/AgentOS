# Running Guide

这份文档说明项目各个功能模块怎么运行，以及每条命令在做什么。

## 0. 进入项目

```bash
cd /workspace/hyh/embodied-teleop-control-lab
```

当前项目不需要真实机器人即可运行。默认使用 fake manipulation environment、AgentOS workspace 和本地数据。

## 1. 一键 Smoke Test

最快确认项目能跑：

```bash
bash scripts/sh/00_smoke_test.sh
```

它会依次运行：

1. AgentOS runtime 闭环。
2. direct scripted embodied agent。
3. fake 环境 RGB 渲染。
4. mock VLA 随机布局评估并生成报告。

输出产物：

```text
outputs/fake_env.ppm
outputs/eval_reports_smoke/
```

## 1.1 Embodied Runtime Workspace

初始化 Markdown workspace：

```bash
python3 scripts/init_workspace.py
```

这会创建运行时协议文件：

```text
workspace/ACTION.md
workspace/ENVIRONMENT.md
workspace/EMBODIED.md
workspace/LESSONS.md
workspace/TASK.md
workspace/SKILL.md
workspace/PLAN.md
workspace/REPORT.md
```

排队一个动作：

```bash
python3 - <<'PY'
from runtime.action_queue import append_action, load_action_document, save_action_document
from runtime.workspace import initialize_workspace

paths = initialize_workspace("workspace")
doc = load_action_document(paths.action)
doc = append_action(doc, action_type="env_step", parameters={"action": [0.02, 0.0, -1.0]})
save_action_document(paths.action, doc)
PY
```

让 watchdog 执行一个 pending action：

```bash
python3 scripts/run_watchdog.py --once
```

持续监听：

```bash
python3 scripts/run_watchdog.py --poll-interval 1.0
```

运行主 AgentOS 入口。它会通过 `ToolRegistry` 调用 plan step，把动作写入 `ACTION.md`，再由 watchdog + HAL driver 执行，并把工具调用写入 `outputs/traces/*.jsonl`：

```bash
python3 scripts/run_agentos.py "pick up the red block and place it in the bowl"
```

在 Docker 中运行：

```bash
docker exec heyuhang-dl bash -lc '
cd /workspace/hyh/embodied-teleop-control-lab
python scripts/run_agentos.py "pick up the red block and place it in the bowl" \
  --workspace workspace/agentos_smoke \
  --render-output outputs/agentos_smoke.ppm
'
```

也可以直接换任务、workspace 和输出路径：

```bash
python3 scripts/run_agentos.py "pick up the blue block and place it in the bowl" \
  --workspace workspace/agentos_blue \
  --render-output outputs/agentos_blue.ppm
```

### 1.2 Skill Library Planner

`SkillLibraryPlanner` 从 Markdown skill library 读取 workflow 模板，把自然语言任务映射为 `TaskPlan`。默认 skill 文件是 `runtime/skills/default_skill.md`：

```bash
python3 scripts/run_agentos.py \
  "pick up the blue block and place it in the bowl" \
  --planner skill
```

使用自定义 skill library：

```bash
python3 scripts/run_agentos.py \
  "pick up the green block and place it in the bowl" \
  --planner skill \
  --skill-path runtime/skills/default_skill.md
```

skill library 只表达 workflow-level tools，例如 `reset_task`、`scripted_pick_place_loop`、`render_fake_env`。它不直接写 `[dx, dy, gripper]`，低层动作仍然由 tool、watchdog、validator 和 HAL driver 处理。

成功执行后，可以把这次 plan 录制为 workspace-local skill。默认写入当前 workspace 的 `SKILL.md`，不会修改仓库内置的 `runtime/skills/default_skill.md`：

```bash
python3 scripts/run_agentos.py \
  "pick up the blue block and place it in the bowl" \
  --planner skill \
  --record-skill
```

如果需要写到指定 skill library：

```bash
python3 scripts/run_agentos.py \
  "pick up the green block and place it in the bowl" \
  --record-skill \
  --record-skill-path workspace/my_skills/SKILL.md
```

### 1.3 DeepSeek LLM Planner

`run_agentos.py` 默认使用 deterministic `RuleBasedPlanner`。如果配置了 DeepSeek API key，可以让 LLM 生成 workflow-level `TaskPlan`：

```bash
export DEEPSEEK_API_KEY="sk-..."
python3 scripts/run_agentos.py \
  "pick up the blue block and place it in the bowl" \
  --planner deepseek
```

可选环境变量：

```text
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

没有 API key、API 调用失败或 LLM 输出未通过本地校验时，会自动 fallback 到 `RuleBasedPlanner`。LLM planner 只允许输出 `reset_task`、`scripted_pick_place_loop`、`render_fake_env` 这类 workflow step，不允许输出 `[dx, dy, gripper]` 等低层动作。

运行后重点查看：

```text
workspace/.../PLAN.md      # Planner 生成的任务计划
workspace/.../REPORT.md    # 执行报告
workspace/.../LESSONS.md   # 失败或被 Critic 拒绝的经验记录
outputs/traces/*.jsonl     # 工具调用和执行事件 trace
```

### 1.4 AgentOS Benchmark

多回合 benchmark 会跑完整在线路径：Planner → Tools → `ACTION.md` → Watchdog → HAL driver，并汇总成功率、平均步数、耗时、fallback 分布和失败原因：

```bash
python3 scripts/benchmark_agentos.py \
  --planner skill \
  --num-episodes 9 \
  --randomize-layout
```

benchmark 也支持录制成功 plan。未指定 `--record-skill-path` 时，每个 episode 会写入自己的 workspace `SKILL.md`，避免实验互相污染：

```bash
python3 scripts/benchmark_agentos.py \
  --planner skill \
  --num-episodes 9 \
  --record-skill
```

对比 deterministic planner 与 DeepSeek planner：

```bash
python3 scripts/benchmark_agentos.py --planner rule --num-episodes 9
python3 scripts/benchmark_agentos.py --planner deepseek --num-episodes 9
```

报告默认写入：

```text
outputs/agentos_benchmarks/
workspace/benchmarks/
```

### 1.5 Robosuite / MuJoCo 3D Backend

项目提供可选 robosuite backend，用于把 AgentOS 从 2D fake env 迁移到 3D MuJoCo manipulation。当前环境没有安装 `mujoco` / `robosuite` 时，其它功能和测试不受影响。

先安装可选依赖：

```bash
pip install mujoco robosuite
```

验证 robosuite 能 reset / step：

```bash
python3 scripts/test_robosuite_env.py --task Lift --robot Panda
```

如需离屏渲染：

```bash
python3 scripts/test_robosuite_env.py \
  --task Lift \
  --robot Panda \
  --offscreen \
  --render-output outputs/robosuite_lift.ppm
```

AgentOS CLI 已预留 driver 选择：

```bash
python3 scripts/run_agentos.py \
  "lift the cube" \
  --driver robosuite \
  --sim-task Lift \
  --robot Panda \
  --planner skill \
  --skill-path runtime/skills/robosuite_skill.md
```

当前 robosuite 接入包含 `robosuite_lift_loop` scripted workflow，可完成 `Lift` 的对齐、下探、闭合、上抬和成功检查。它仍然通过 `ACTION.md`、watchdog、validator 和 HAL driver 执行，不绕过 AgentOS 协议。

当前华为 Ascend 服务器上，系统 Python 可能尚未安装 `numpy`、`pytest`、`torch`、`torch_npu`。workspace 初始化不依赖这些包，但 fake manipulation driver 需要 `numpy`，BC/VisionBC 训练需要 PyTorch。

如果使用当前 Docker 环境，项目路径是：

```bash
docker exec heyuhang-dl bash -lc 'cd /workspace/hyh/embodied-teleop-control-lab && python scripts/init_workspace.py'
```

容器内 `torch_npu` 可用。如果只使用后四张 NPU，可以这样运行训练：

```bash
docker exec heyuhang-dl bash -lc '
cd /workspace/hyh/embodied-teleop-control-lab
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python learning/train_vision_bc.py \
  --data-dir data/vision_demos_random \
  --output checkpoints/vision_bc_random_policy.pt \
  --device npu
'
```

## 2. Embodied Agent Demo

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

## 3. RGB Render

导出 fake 环境预览图：

```bash
python scripts/render_fake_env.py --output outputs/fake_env.ppm
```

输出：

```text
outputs/fake_env.ppm
```

这张图是 top-down RGB observation，后续 VisionBC / VLA 都可以使用类似输入。

## 4. State BC: 采集、训练、评估

### 4.1 采集 state demonstrations

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

### 4.2 训练 state BC

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

### 4.3 评估 state BC

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

## 5. VisionBC: 随机布局视觉模仿学习

这是当前项目最推荐展示的学习链路。

### 5.1 采集随机布局视觉 demonstrations

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

### 5.2 训练随机布局 VisionBC

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

### 5.3 评估随机布局 VisionBC

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

## 6. VLA-Ready Mock Backend

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

### 6.1 SmolVLA Bridge

训练前先检查数据质量：

```bash
python scripts/inspect_dataset.py \
  --data-dir data/vision_demos_random \
  --expect-images \
  --output-dir outputs/dataset_quality_random
```

导出当前 vision demos 为 LeRobot-style manifest：

```bash
python scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir data/lerobot_fake_manipulation \
  --format manifest
```

不安装 LeRobot 时，可以先跑 SmolVLA dry-run 后端，验证 VLA 接口和评估链路：

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend smolvla_dry_run \
  --num-episodes 3 \
  --write-report
```

安装 LeRobot/SmolVLA 后，可切换为真实后端：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend smolvla \
  --smolvla-model lerobot/smolvla_base \
  --device npu
```

### 6.2 Reinforcement Learning

先 smoke-test Gym-style fake env wrapper：

```bash
python scripts/train_rl.py --backend smoke
```

评估 RL policy wrapper 的 deterministic baseline：

```bash
python learning/evaluate_policy.py \
  --policy rl \
  --rl-backend scripted \
  --num-episodes 3 \
  --write-report
```

安装 `stable-baselines3` 后，可以训练 PPO：

```bash
python scripts/train_rl.py \
  --backend sb3 \
  --timesteps 10000 \
  --output checkpoints/rl_ppo_fake_manipulation.zip
```

小规模 policy 对比：

```bash
python scripts/benchmark_policies.py \
  --num-episodes 3 \
  --max-steps 100 \
  --output-dir outputs/policy_benchmark
```

## 7. Evaluation Reports

所有 policy 都可以用同一个评估入口：

```bash
python learning/evaluate_policy.py --policy scripted --write-report
python learning/evaluate_policy.py --policy bc --checkpoint checkpoints/bc_policy.pt --write-report
python learning/evaluate_policy.py --policy vision_bc --checkpoint checkpoints/vision_bc_random_policy.pt --write-report
python learning/evaluate_policy.py --policy vla --vla-backend mock --write-report
python learning/evaluate_policy.py --policy rl --rl-backend scripted --write-report
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

## 8. 常用环境变量

快捷脚本支持用环境变量覆盖默认参数。

示例：

```bash
NUM_EPISODES=30 bash scripts/sh/03_collect_vision_demos_random.sh
EPOCHS=40 bash scripts/sh/04_train_vision_bc_random.sh
CHECKPOINT=checkpoints/vision_bc_random_policy.pt bash scripts/sh/05_eval_vision_bc_random.sh
```

## 9. 生成文件说明

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
