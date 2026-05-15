from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from runtime.plan_io import plan_from_dict
from runtime.plan_utils import force_target_color
from runtime.planner import PlannedStep, Planner, RuleBasedPlanner, TaskPlan


SUPPORTED_COLORS = {"red", "blue", "green"}
WORKFLOW_TOOL_NAMES = {"reset_task", "scripted_pick_place_loop", "render_fake_env"}
LOW_LEVEL_PARAMETER_NAMES = {"action", "dx", "dy", "gripper"}

WORKFLOW_TOOLS: list[dict[str, Any]] = [
    {
        "name": "reset_task",
        "description": "Initialize the fake pick-and-place task.",
        "parameters": {
            "instruction": "natural language task",
            "target_color": "one of: red, blue, green",
            "receptacle_name": "always 'bowl'",
        },
    },
    {
        "name": "scripted_pick_place_loop",
        "description": "Run the scripted expert policy loop until success or timeout.",
        "parameters": {"max_steps": "int between 1 and 500, default 80"},
    },
    {
        "name": "render_fake_env",
        "description": "Render final environment state to a PPM image.",
        "parameters": {"output": "optional output path string"},
    },
]


class PlanValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class DeepSeekPlanner:
    """LLM-backed workflow planner with deterministic fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        fallback: Planner | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
        self.timeout = timeout
        self.fallback = fallback or RuleBasedPlanner()
        self.last_fallback_reason: str | None = None
        self._client = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def plan(self, instruction: str, *, target_color: str | None = None) -> TaskPlan:
        if self._client is None:
            return self._fallback("no_api_key", instruction, target_color)

        try:
            raw_text = self._call_api(
                self._build_system_prompt(),
                self._build_user_message(instruction, target_color),
            )
            payload = self._parse_json(raw_text)
            plan = plan_from_dict(payload)
            if target_color is not None:
                plan = force_target_color(plan, target_color)
            self._validate_plan(plan, target_color=target_color)
            self.last_fallback_reason = None
            return plan
        except PlanValidationError as exc:
            return self._fallback(exc.reason, instruction, target_color)
        except json.JSONDecodeError:
            return self._fallback("invalid_json", instruction, target_color)
        except Exception as exc:
            return self._fallback("api_error", instruction, target_color, details=str(exc))

    def _build_system_prompt(self) -> str:
        tools_json = json.dumps(WORKFLOW_TOOLS, indent=2, ensure_ascii=False)
        return f"""You are an embodied AgentOS planner.

Your job is to map a natural language task to a workflow-level TaskPlan.
You must only use the available tools.

Available tools:
{tools_json}

Output ONLY a JSON object matching this schema:
{{
  "instruction": "<original or normalized task text>",
  "target_color": "<red|blue|green>",
  "steps": [
    {{
      "tool": "<tool name from available tools>",
      "parameters": {{}},
      "description": "<one-line summary>"
    }}
  ]
}}

Rules:
- Use only listed tools.
- Do not invent tools.
- target_color must be one of: red, blue, green.
- receptacle_name is always "bowl".
- Include reset_task before execution.
- Include scripted_pick_place_loop for task execution.
- render_fake_env may be last.
- Do not output low-level actions such as [dx, dy, gripper].
- Do not include append_action, step_env, or watchdog tools.
"""

    def _build_user_message(self, instruction: str, target_color: str | None) -> str:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Target color override:\n{target_color or 'none'}\n\n"
            "Return a TaskPlan JSON."
        )

    def _call_api(self, system_prompt: str, user_message: str) -> str:
        assert self._client is not None
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content or ""

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fenced is not None:
            text = fenced.group(1).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise PlanValidationError("invalid_plan_shape")
        return payload

    def _validate_plan(self, plan: TaskPlan, *, target_color: str | None = None) -> None:
        if plan.target_color not in SUPPORTED_COLORS:
            raise PlanValidationError("invalid_color")
        if target_color is not None and plan.target_color != target_color:
            raise PlanValidationError("target_color_mismatch")

        if not plan.steps:
            raise PlanValidationError("invalid_plan_shape")

        tool_names = [step.tool for step in plan.steps]
        for step in plan.steps:
            if step.tool not in WORKFLOW_TOOL_NAMES:
                raise PlanValidationError("invalid_tool")
            if any(name in step.parameters for name in LOW_LEVEL_PARAMETER_NAMES):
                raise PlanValidationError("invalid_parameters")

        if "reset_task" not in tool_names:
            raise PlanValidationError("missing_reset_task")
        if "scripted_pick_place_loop" not in tool_names:
            raise PlanValidationError("missing_execution_step")

        for step in plan.steps:
            if step.tool == "reset_task":
                self._validate_reset_step(step, plan.target_color)
            elif step.tool == "scripted_pick_place_loop":
                self._validate_loop_step(step)

        render_indices = [index for index, step in enumerate(plan.steps) if step.tool == "render_fake_env"]
        if render_indices and render_indices[-1] != len(plan.steps) - 1:
            raise PlanValidationError("invalid_plan_shape")

    def _validate_reset_step(self, step: PlannedStep, target_color: str) -> None:
        if step.parameters.get("target_color") != target_color:
            raise PlanValidationError("target_color_mismatch")
        if step.parameters.get("receptacle_name", "bowl") != "bowl":
            raise PlanValidationError("invalid_parameters")
        instruction = str(step.parameters.get("instruction") or "").strip()
        if not instruction:
            raise PlanValidationError("invalid_parameters")

    def _validate_loop_step(self, step: PlannedStep) -> None:
        max_steps = step.parameters.get("max_steps", 80)
        try:
            max_steps_int = int(max_steps)
        except (TypeError, ValueError):
            raise PlanValidationError("invalid_parameters") from None
        if max_steps_int < 1 or max_steps_int > 500:
            raise PlanValidationError("invalid_parameters")

    def _fallback(
        self,
        reason: str,
        instruction: str,
        target_color: str | None,
        *,
        details: str | None = None,
    ) -> TaskPlan:
        self.last_fallback_reason = reason
        suffix = f": {details}" if details else ""
        print(f"DeepSeekPlanner: {reason}{suffix}, falling back to RuleBasedPlanner")
        return self.fallback.plan(instruction, target_color=target_color)
