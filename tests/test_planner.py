import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.planner import RuleBasedPlanner


def test_rule_based_planner_extracts_target_color_and_steps():
    plan = RuleBasedPlanner().plan("pick up the green block and place it in the bowl")

    assert plan.target_color == "green"
    assert [step.tool for step in plan.steps] == ["reset_task", "scripted_pick_place_loop", "render_fake_env"]
    assert plan.steps[0].parameters["target_color"] == "green"

