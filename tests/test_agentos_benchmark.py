import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_agentos import AgentOSBenchmarkConfig, parse_target_colors, run_benchmark


def test_benchmark_rule_planner_two_episodes(tmp_path):
    summary = run_benchmark(
        AgentOSBenchmarkConfig(
            planner="rule",
            num_episodes=2,
            max_steps=80,
            seed=0,
            target_colors=("red", "blue"),
            workspace_root=tmp_path / "workspace",
            output_dir=tmp_path / "outputs",
            write_report=False,
            run_id="test_run",
        )
    )

    assert summary.run_id == "test_run"
    assert summary.planner == "rule"
    assert summary.num_episodes == 2
    assert summary.success_rate == 1.0
    assert summary.fallback_count == 0
    assert len(summary.episodes) == 2
    assert (tmp_path / "workspace" / "test_run" / "episode_000" / "ACTION.md").exists()
    assert (tmp_path / "workspace" / "test_run" / "episode_001" / "REPORT.md").exists()


def test_benchmark_no_write_report_keeps_report_paths_empty(tmp_path):
    summary = run_benchmark(
        AgentOSBenchmarkConfig(
            planner="rule",
            num_episodes=1,
            max_steps=80,
            target_colors=("green",),
            workspace_root=tmp_path / "workspace",
            output_dir=tmp_path / "outputs",
            write_report=False,
            run_id="no_report",
        )
    )

    assert summary.report_json is None
    assert summary.report_markdown is None
    assert not (tmp_path / "outputs" / "benchmark_no_report.json").exists()


def test_benchmark_output_structure(tmp_path):
    summary = run_benchmark(
        AgentOSBenchmarkConfig(
            planner="rule",
            num_episodes=1,
            max_steps=80,
            target_colors=("red",),
            workspace_root=tmp_path / "workspace",
            output_dir=tmp_path / "outputs",
            write_report=True,
            run_id="with_report",
        )
    )

    assert summary.report_json is not None
    assert summary.report_markdown is not None
    json_path = Path(summary.report_json)
    md_path = Path(summary.report_markdown)
    assert json_path.exists()
    assert md_path.exists()
    assert '"run_id": "with_report"' in json_path.read_text(encoding="utf-8")
    assert "AgentOS Benchmark Report" in md_path.read_text(encoding="utf-8")


def test_parse_target_colors_rejects_unknown_color():
    try:
        parse_target_colors("red,yellow")
    except ValueError as exc:
        assert "unsupported target colors" in str(exc)
    else:
        raise AssertionError("expected ValueError")
