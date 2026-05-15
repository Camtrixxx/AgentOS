import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.llm_planner import DeepSeekPlanner


def _response(payload: dict | str):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _client_with_response(payload: dict | str):
    client = MagicMock()
    client.chat.completions.create.return_value = _response(payload)
    return client


def _valid_payload(target_color: str = "red") -> dict:
    instruction = f"pick up the {target_color} block and place it in the bowl"
    return {
        "instruction": instruction,
        "target_color": target_color,
        "steps": [
            {
                "tool": "reset_task",
                "parameters": {
                    "instruction": instruction,
                    "target_color": target_color,
                    "receptacle_name": "bowl",
                },
                "description": "Initialize the task.",
            },
            {
                "tool": "scripted_pick_place_loop",
                "parameters": {"max_steps": 80},
                "description": "Run the expert loop.",
            },
            {
                "tool": "render_fake_env",
                "parameters": {},
                "description": "Render the final state.",
            },
        ],
    }


def test_parses_valid_json_response(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = _client_with_response(_valid_payload("green"))
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the green block and place it in the bowl")

    assert planner.last_fallback_reason is None
    assert plan.target_color == "green"
    assert [step.tool for step in plan.steps] == [
        "reset_task",
        "scripted_pick_place_loop",
        "render_fake_env",
    ]


def test_falls_back_when_no_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    planner = DeepSeekPlanner(api_key="")

    plan = planner.plan("pick up the red block and place it in the bowl")

    assert planner.last_fallback_reason == "no_api_key"
    assert plan.target_color == "red"
    assert [step.tool for step in plan.steps] == [
        "reset_task",
        "scripted_pick_place_loop",
        "render_fake_env",
    ]


def test_rejects_invalid_tool_name(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = _valid_payload("red")
    payload["steps"].append({"tool": "fly_robot", "parameters": {}, "description": "Invalid."})
    client = _client_with_response(payload)
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the red block and place it in the bowl")

    assert planner.last_fallback_reason == "invalid_tool"
    assert plan.target_color == "red"


def test_rejects_unknown_color(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = _valid_payload("red")
    payload["target_color"] = "yellow"
    payload["steps"][0]["parameters"]["target_color"] = "yellow"
    client = _client_with_response(payload)
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the red block and place it in the bowl")

    assert planner.last_fallback_reason == "invalid_color"
    assert plan.target_color == "red"


def test_falls_back_on_api_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the blue block and place it in the bowl")

    assert planner.last_fallback_reason == "api_error"
    assert plan.target_color == "blue"


def test_rejects_plan_without_execution_step(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = _valid_payload("red")
    payload["steps"] = [payload["steps"][0], payload["steps"][2]]
    client = _client_with_response(payload)
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the red block and place it in the bowl")

    assert planner.last_fallback_reason == "missing_execution_step"
    assert plan.target_color == "red"


def test_target_color_override_syncs_all_related_fields(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = _client_with_response(_valid_payload("red"))
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the red block and place it in the bowl", target_color="blue")
    reset_step = next(step for step in plan.steps if step.tool == "reset_task")

    assert planner.last_fallback_reason is None
    assert plan.target_color == "blue"
    assert "blue" in plan.instruction
    assert reset_step.parameters["target_color"] == "blue"
    assert "blue" in reset_step.parameters["instruction"]
    assert reset_step.parameters["receptacle_name"] == "bowl"


def test_parses_fenced_json_response(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    fenced = "```json\n" + json.dumps(_valid_payload("blue")) + "\n```"
    client = _client_with_response(fenced)
    with patch("runtime.llm_planner.OpenAI", return_value=client):
        planner = DeepSeekPlanner(api_key="test-key")

    plan = planner.plan("pick up the blue block and place it in the bowl")

    assert planner.last_fallback_reason is None
    assert plan.target_color == "blue"
