import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.plan_io import dump_execution_report, dump_plan_document, plan_from_dict, plan_to_dict
from runtime.planner import RuleBasedPlanner


def test_plan_document_round_trip_payload():
    plan = RuleBasedPlanner().plan("pick up the blue block and place it in the bowl")
    payload = plan_to_dict(plan)
    restored = plan_from_dict(payload)
    markdown = dump_plan_document(plan)

    assert payload["schema_version"] == "embodied_lab.plan.v1"
    assert restored.target_color == "blue"
    assert "```json" in markdown


def test_execution_report_document_contains_schema():
    markdown = dump_execution_report({"success": True, "steps": 3})

    assert "embodied_lab.execution_report.v1" in markdown
    assert '"success": true' in markdown

