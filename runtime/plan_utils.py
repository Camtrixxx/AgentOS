from __future__ import annotations

import re

from runtime.planner import PlannedStep, TaskPlan


def force_target_color(plan: TaskPlan, target_color: str) -> TaskPlan:
    instruction = rewrite_instruction_color(plan.instruction, target_color)
    forced_steps: list[PlannedStep] = []
    for step in plan.steps:
        parameters = dict(step.parameters)
        if step.tool == "reset_task":
            parameters["target_color"] = target_color
            parameters["receptacle_name"] = "bowl"
            parameters["instruction"] = rewrite_instruction_color(
                str(parameters.get("instruction") or instruction),
                target_color,
            )
        forced_steps.append(PlannedStep(step.tool, parameters, step.description))
    return TaskPlan(instruction=instruction, target_color=target_color, steps=forced_steps)


def rewrite_instruction_color(instruction: str, target_color: str) -> str:
    text = instruction.strip()
    if not text:
        return f"pick up the {target_color} block and place it in the bowl"
    updated = re.sub(r"\b(red|blue|green)\b", target_color, text, count=1, flags=re.IGNORECASE)
    if updated == text and target_color not in text.lower():
        return f"pick up the {target_color} block and place it in the bowl"
    return updated
