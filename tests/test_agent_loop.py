import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_loop import AgentLoop
from agent.scripted_policy import ScriptedPickPlacePolicy
from envs.fake_manipulation_env import FakeManipulationEnv, TaskSpec


def test_scripted_agent_solves_fake_pick_place():
    env = FakeManipulationEnv(seed=0)
    policy = ScriptedPickPlacePolicy()
    task = TaskSpec("pick up the red block and place it in the bowl", "red")

    result = AgentLoop(env, policy).run_episode(task=task, max_steps=80)

    assert result.success
    assert result.steps < 80

