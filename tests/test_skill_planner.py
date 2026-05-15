import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.planner import RuleBasedPlanner
from runtime.skill_planner import ParsedSkill, SkillLibraryPlanner, instantiate_skill, match_skill, parse_skills


SKILL_MD = """
# Skill Library

```json
{
  "schema_version": "embodied_lab.skill_library.v1",
  "skills": [
    {
      "name": "pick_place",
      "description": "Pick up a colored block and place it in the bowl",
      "pattern": "pick up the {color} block and place it in the bowl",
      "parameters": {
        "color": {"type": "string", "values": ["red", "blue", "green"]}
      },
      "target_color_param": "color",
      "steps": [
        {
          "tool": "reset_task",
          "parameters": {
            "instruction": "pick up the {color} block and place it in the bowl",
            "target_color": "{color}",
            "receptacle_name": "bowl"
          },
          "description": "Initialize."
        },
        {
          "tool": "scripted_pick_place_loop",
          "parameters": {"max_steps": 80},
          "description": "Execute."
        },
        {
          "tool": "render_fake_env",
          "parameters": {},
          "description": "Render."
        }
      ]
    }
  ]
}
```
"""


def test_parse_skill_from_markdown():
    skills = parse_skills(SKILL_MD)

    assert "pick_place" in skills
    assert skills["pick_place"].pattern == "pick up the {color} block and place it in the bowl"


def test_parse_skills_skips_non_skill_json_blocks():
    markdown = """
```json
{"example": true}
```

""" + SKILL_MD

    skills = parse_skills(markdown)

    assert "pick_place" in skills


def test_match_skill_extracts_params():
    skill = parse_skills(SKILL_MD)["pick_place"]

    params = match_skill(skill, "Pick up the blue block and place it in the bowl.")

    assert params == {"color": "blue"}


def test_instantiate_replaces_placeholders():
    skill = parse_skills(SKILL_MD)["pick_place"]

    plan = instantiate_skill(skill, {"color": "red"}, instruction="pick up the red block and place it in the bowl")

    assert plan.target_color == "red"
    assert plan.steps[0].parameters["target_color"] == "red"
    assert "red" in plan.steps[0].parameters["instruction"]


def test_no_match_falls_back(tmp_path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(SKILL_MD, encoding="utf-8")
    planner = SkillLibraryPlanner(skill_path=skill_path, fallback=RuleBasedPlanner())

    plan = planner.plan("do something unknown")

    assert plan.target_color == "red"
    assert [step.tool for step in plan.steps] == [
        "reset_task",
        "scripted_pick_place_loop",
        "render_fake_env",
    ]


def test_target_color_override_beats_skill_match(tmp_path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(SKILL_MD, encoding="utf-8")
    planner = SkillLibraryPlanner(skill_path=skill_path)

    plan = planner.plan("pick up the red block and place it in the bowl", target_color="blue")
    reset_step = next(step for step in plan.steps if step.tool == "reset_task")

    assert plan.target_color == "blue"
    assert "blue" in plan.instruction
    assert reset_step.parameters["target_color"] == "blue"
    assert "blue" in reset_step.parameters["instruction"]
    assert reset_step.parameters["receptacle_name"] == "bowl"


def test_invalid_skill_param_value_does_not_match():
    skill = ParsedSkill(
        name="pick_place",
        description="",
        pattern="pick up the {color} block and place it in the bowl",
        parameters={"color": {"type": "string", "values": ["red", "blue", "green"]}},
        target_color_param="color",
        steps=[],
    )

    assert match_skill(skill, "pick up the yellow block and place it in the bowl") is None
