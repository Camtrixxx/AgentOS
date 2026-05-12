from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from runtime.environment_io import to_jsonable, utc_now_iso
from runtime.planner import PlannedStep, TaskPlan


PLAN_SCHEMA_VERSION = "embodied_lab.plan.v1"
REPORT_SCHEMA_VERSION = "embodied_lab.execution_report.v1"
FENCE_OPEN = "```json"
FENCE_CLOSE = "```"


def plan_to_dict(plan: TaskPlan) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "instruction": plan.instruction,
        "target_color": plan.target_color,
        "steps": [asdict(step) for step in plan.steps],
    }


def plan_from_dict(payload: dict[str, Any]) -> TaskPlan:
    steps = [
        PlannedStep(
            tool=str(item.get("tool", "")),
            parameters=dict(item.get("parameters", {})),
            description=str(item.get("description", "")),
        )
        for item in payload.get("steps", [])
        if isinstance(item, dict)
    ]
    return TaskPlan(
        instruction=str(payload.get("instruction", "")),
        target_color=str(payload.get("target_color", "red")),
        steps=steps,
    )


def dump_plan_document(plan: TaskPlan) -> str:
    payload = json.dumps(to_jsonable(plan_to_dict(plan)), indent=2, ensure_ascii=False)
    return "# Task Plan\n\nPlanner-generated execution plan.\n\n" f"{FENCE_OPEN}\n{payload}\n{FENCE_CLOSE}\n"


def save_plan_document(path: Path, plan: TaskPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_plan_document(plan), encoding="utf-8")


def dump_execution_report(report: dict[str, Any]) -> str:
    payload = dict(report)
    payload.setdefault("schema_version", REPORT_SCHEMA_VERSION)
    payload.setdefault("created_at", utc_now_iso())
    report_json = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False)
    return "# Execution Report\n\nRuntime result for the latest task plan.\n\n" f"{FENCE_OPEN}\n{report_json}\n{FENCE_CLOSE}\n"


def save_execution_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_execution_report(report), encoding="utf-8")

