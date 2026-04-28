import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.report import EpisodeEval, EvaluationSummary, write_evaluation_report


def test_write_evaluation_report(tmp_path):
    summary = EvaluationSummary(
        policy="scripted",
        checkpoint=None,
        num_episodes=1,
        max_steps=80,
        randomize_layout=False,
        success_rate=1.0,
        avg_steps=10.0,
        avg_reward=0.9,
        episodes=[
            EpisodeEval(
                episode=0,
                target_color="red",
                success=True,
                steps=10,
                total_reward=0.9,
                failure_reason="none",
            )
        ],
    )

    json_path, md_path = write_evaluation_report(summary, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert "Evaluation Report" in md_path.read_text(encoding="utf-8")

