# HelloAgents Architecture Notes

参考源码路径：

```text
/data/heyuhang/hyh/reference-repos/HelloAgents
```

GitHub:

```text
https://github.com/Camtrixxx/HelloAgents
```

## Project Positioning

`HelloAgents` 是一个通用多智能体工程框架，重点不是具身机器人，而是 Agent 应用的工程化能力。

它的核心能力包括：

- ToolResponse 工具响应协议
- ToolRegistry 工具注册表
- Function calling Agent
- ReAct / Reflection / Plan-and-Solve agents
- HistoryManager 上下文历史管理
- ContextBuilder 上下文构建
- SessionStore 会话持久化
- TraceLogger 可观测性
- CircuitBreaker 工具熔断
- TodoWrite 任务进度
- DevLog 决策日志
- Skills 知识外化

## Top-Level Structure

主要目录：

```text
hello_agents/
  core/
    agent.py          # Agent 基类
    llm.py            # LLM 封装
    llm_adapters.py   # OpenAI / Anthropic / Gemini 等适配
    message.py        # Message 对象
    session_store.py  # 会话持久化
    lifecycle.py      # 异步生命周期事件
    streaming.py      # SSE 流式输出
  agents/
    simple_agent.py
    react_agent.py
    reflection_agent.py
    plan_solve_agent.py
  tools/
    base.py           # Tool 基类
    response.py       # ToolResponse 协议
    registry.py       # ToolRegistry
    circuit_breaker.py
    tool_filter.py
    builtin/
      file_tools.py
      task_tool.py
      todowrite_tool.py
      devlog_tool.py
      skill_tool.py
  context/
    history.py
    token_counter.py
    truncator.py
    builder.py
  observability/
    trace_logger.py
  skills/
    loader.py
```

## ToolResponse Protocol

HelloAgents 把工具返回值统一成结构化对象：

```text
ToolResponse
  status: success | partial | error
  text: 给 LLM 阅读的文本
  data: 结构化数据
  error_info: 错误信息
  stats: 运行统计
  context: 调用上下文
```

这个设计适合当前项目，因为具身系统里的工具结果不能只靠字符串表达。比如：

```text
render_fake_env
  text: rendered image saved
  data: {path, shape, dtype}
  stats: {time_ms}

execute_action
  text: action completed
  data: {success, reward, done, info}
  context: {action_type, params}

evaluate_policy
  text: success_rate=0.8
  data: {success_rate, avg_steps, report_path}
```

## Tool Base Class

HelloAgents 的 Tool 抽象包含：

```text
name
description
get_parameters()
run(parameters) -> ToolResponse
run_with_timing()
arun()
to_openai_schema()
```

这个抽象比当前项目需要的稍重。当前项目可以先做轻量版本：

```python
class Tool(Protocol):
    name: str
    description: str
    def run(self, params: dict[str, Any]) -> ToolResponse: ...
```

## ToolRegistry

`ToolRegistry` 管理工具注册与调用：

```text
register_tool(tool)
register_function(func)
execute_tool(name, input)
get_tool(name)
get_tools_description()
```

它还集成了 `CircuitBreaker`，当某个工具连续失败时暂时禁用，避免 Agent 死循环调用坏工具。

当前项目可以先实现最小注册表：

```text
ToolRegistry
  register(tool)
  run(name, params)
  list_tools()
```

后续再加熔断和 OpenAI schema。

## Agent Base

HelloAgents 的 `Agent` 基类整合了：

- LLM
- tool registry
- history manager
- observation truncator
- token counter
- trace logger
- skill loader
- session store
- subagent
- TodoWrite
- DevLog

这个类很完整，但对当前项目来说太重。更适合借鉴它的分层思想，而不是直接复制。

本项目可采用更小的分层：

```text
EmbodiedAgent
  planner: optional LLM / scripted planner
  tools: ToolRegistry
  memory: simple session log
  trace: TraceLogger
```

## Context Management

HelloAgents 的 `HistoryManager` 特点：

- 消息只追加，不编辑
- 支持保留最近 N 轮
- 支持压缩旧历史为 summary
- 支持序列化/反序列化

当前项目短期可用更简单的 episode/session log：

```text
memory/sessions/session-xxxx.jsonl
  user instruction
  observation summary
  selected action
  tool response
  environment update
```

当接入真实 LLM 后，再考虑 token-aware 历史压缩。

## Observability

`TraceLogger` 是 HelloAgents 很值得借鉴的一块。它输出：

- JSONL: 机器可读，适合分析
- HTML: 人类可读，适合复盘

当前项目已经有 evaluation report，但缺少运行时 trace。可以新增：

```text
outputs/traces/
  trace-YYYYMMDD-HHMMSS.jsonl
```

记录事件：

```text
episode_start
observation
policy_action
tool_call
tool_result
env_step
episode_end
evaluation_summary
```

这对具身智能尤其重要，因为失败通常来自感知、规划、控制、环境状态不一致中的某一环。

## TodoWrite / DevLog / Skills

这些能力适合后续阶段：

- TodoWrite: 适合长程任务规划。
- DevLog: 适合记录 Agent 为什么选择某个工具或动作。
- Skills: 适合把成功任务流程沉淀为可复用 SOP。

当前项目可以先通过 Markdown 文件模拟：

```text
workspace/TASK.md
workspace/SKILL.md
workspace/LESSONS.md
outputs/traces/*.jsonl
```

## What To Borrow

适合借鉴：

- `ToolResponse` 结构化工具返回协议
- `ToolRegistry` 工具注册和统一调用
- `TraceLogger` JSONL 可观测性
- `HistoryManager` 的 append-only 思路
- `TodoWrite` 的“最多一个 in_progress”任务约束
- `Skills` 把经验外化为文件的思路

不建议现在照搬：

- 完整 LLM provider 适配层
- 完整 function calling agent
- SSE 服务
- 子代理系统
- 大量内置文件工具
- 复杂 token budget 逻辑

## License Note

HelloAgents 仓库声明为 CC BY-NC-SA 4.0。这个许可证包含非商业限制。当前项目应避免直接复制其源码；建议只参考架构思想，并自行实现轻量版本。

