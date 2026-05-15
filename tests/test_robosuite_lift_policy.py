import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.robosuite_scripted_policy import GRIPPER_CLOSE, GRIPPER_OPEN, RobosuiteLiftPolicy
from runtime.skill_planner import SkillLibraryPlanner


def _env(ee, cube):
    return {
        "robot": {"ee_position": ee},
        "objects": {"cube": {"position": cube}},
        "episode": {"success": False, "done": False},
    }


def test_lift_policy_moves_above_cube():
    policy = RobosuiteLiftPolicy(max_delta=0.04)

    action = policy.act(_env([0.0, 0.0, 1.0], [0.2, -0.1, 0.8]))

    assert policy.last_stage == "move_above_cube"
    assert action[0] > 0
    assert action[1] < 0
    assert action[3] == GRIPPER_OPEN


def test_lift_policy_descends_when_xy_aligned():
    policy = RobosuiteLiftPolicy(max_delta=0.04)

    action = policy.act(_env([0.2, -0.1, 1.2], [0.2, -0.1, 0.8]))

    assert policy.last_stage == "descend"
    assert action[2] < 0
    assert action[3] == GRIPPER_OPEN


def test_lift_policy_closes_then_lifts():
    policy = RobosuiteLiftPolicy(max_delta=0.04, close_steps=2)
    near_grasp = _env([0.2, -0.1, 0.77], [0.2, -0.1, 0.8])

    first = policy.act(near_grasp)
    second = policy.act(near_grasp)
    third = policy.act(near_grasp)

    assert first[3] == GRIPPER_CLOSE
    assert second[3] == GRIPPER_CLOSE
    assert policy.last_stage == "lift"
    assert third[2] > 0
    assert third[3] == GRIPPER_CLOSE


def test_robosuite_skill_preserves_lift_instruction():
    planner = SkillLibraryPlanner(skill_path=PROJECT_ROOT / "runtime" / "skills" / "robosuite_skill.md")

    plan = planner.plan("lift the cube")

    assert plan.instruction == "lift the cube"
    assert [step.tool for step in plan.steps] == ["reset_task", "robosuite_lift_loop", "render_fake_env"]
