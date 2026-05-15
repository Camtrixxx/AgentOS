from __future__ import annotations

from dataclasses import dataclass

from envs.task_utils import parse_target_color


@dataclass(frozen=True)
class PlannedStep:
    tool: str
    parameters: dict
    description: str


@dataclass(frozen=True)
class TaskPlan:
    instruction: str
    target_color: str
    steps: list[PlannedStep]


class RuleBasedPlanner:
    """Small deterministic planner for the fake pick-and-place domain."""

    def plan(self, instruction: str, *, target_color: str | None = None) -> TaskPlan:
        color = target_color or parse_target_color(instruction)
        normalized_instruction = instruction.strip() or f"pick up the {color} block and place it in the bowl"
        return TaskPlan(
            instruction=normalized_instruction,
            target_color=color,
            steps=[
                PlannedStep(
                    tool="reset_task",
                    parameters={
                        "instruction": normalized_instruction,
                        "target_color": color,
                        "receptacle_name": "bowl",
                    },
                    description="Initialize the fake manipulation task.",
                ),
                PlannedStep(
                    tool="scripted_pick_place_loop",
                    parameters={"max_steps": 80},
                    description="Repeatedly observe, critique, and execute policy actions until done.",
                ),
                PlannedStep(
                    tool="render_fake_env",
                    parameters={},
                    description="Render the final environment state for inspection.",
                ),
            ],
        )



