from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentOSEpisodeResult:
    episode: int
    target_color: str
    success: bool
    steps: int
    last_reward: float
    failure_reason: str
    planner: str
    fallback_reason: str | None
    workspace: str
    report_path: str
    trace_path: str | None
    render_path: str | None
    duration_ms: float
    planner_time_ms: float


@dataclass(frozen=True)
class AgentOSBenchmarkSummary:
    run_id: str
    planner: str
    num_episodes: int
    max_steps: int
    randomize_layout: bool
    seed: int
    target_colors: list[str]
    success_rate: float
    avg_steps: float
    avg_reward: float
    avg_duration_ms: float
    avg_planner_time_ms: float
    fallback_count: int
    fallback_reasons: dict[str, int]
    failure_reasons: dict[str, int]
    report_json: str | None
    report_markdown: str | None
    episodes: list[AgentOSEpisodeResult]


def build_benchmark_summary(
    *,
    run_id: str,
    planner: str,
    num_episodes: int,
    max_steps: int,
    randomize_layout: bool,
    seed: int,
    target_colors: list[str],
    episodes: list[AgentOSEpisodeResult],
    report_json: str | None = None,
    report_markdown: str | None = None,
) -> AgentOSBenchmarkSummary:
    success_count = sum(int(ep.success) for ep in episodes)
    fallback_reasons = _count_values(ep.fallback_reason for ep in episodes if ep.fallback_reason)
    failure_reasons = _count_values(ep.failure_reason for ep in episodes)
    denominator = max(len(episodes), 1)
    return AgentOSBenchmarkSummary(
        run_id=run_id,
        planner=planner,
        num_episodes=num_episodes,
        max_steps=max_steps,
        randomize_layout=randomize_layout,
        seed=seed,
        target_colors=target_colors,
        success_rate=success_count / denominator,
        avg_steps=sum(ep.steps for ep in episodes) / denominator,
        avg_reward=sum(ep.last_reward for ep in episodes) / denominator,
        avg_duration_ms=sum(ep.duration_ms for ep in episodes) / denominator,
        avg_planner_time_ms=sum(ep.planner_time_ms for ep in episodes) / denominator,
        fallback_count=sum(1 for ep in episodes if ep.fallback_reason),
        fallback_reasons=fallback_reasons,
        failure_reasons=failure_reasons,
        report_json=report_json,
        report_markdown=report_markdown,
        episodes=episodes,
    )


def write_agentos_benchmark_report(summary: AgentOSBenchmarkSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"benchmark_{summary.run_id}.json"
    md_path = output_dir / f"benchmark_{summary.run_id}.md"
    json_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    md_path.write_text(render_benchmark_markdown(summary), encoding="utf-8")
    return json_path, md_path


def render_benchmark_markdown(summary: AgentOSBenchmarkSummary) -> str:
    episode_rows = [
        "| Episode | Target | Success | Steps | Reward | Failure | Fallback | Duration ms | Planner ms |",
        "| ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for ep in summary.episodes:
        episode_rows.append(
            f"| {ep.episode} | {ep.target_color} | {ep.success} | {ep.steps} | "
            f"{ep.last_reward:.3f} | {ep.failure_reason} | {ep.fallback_reason or 'none'} | "
            f"{ep.duration_ms:.1f} | {ep.planner_time_ms:.1f} |"
        )

    return "\n".join(
        [
            "# AgentOS Benchmark Report",
            "",
            "## Summary",
            "",
            f"- Run ID: `{summary.run_id}`",
            f"- Planner: `{summary.planner}`",
            f"- Episodes: `{summary.num_episodes}`",
            f"- Max steps: `{summary.max_steps}`",
            f"- Randomize layout: `{summary.randomize_layout}`",
            f"- Seed: `{summary.seed}`",
            f"- Target colors: `{', '.join(summary.target_colors)}`",
            f"- Success rate: `{summary.success_rate:.3f}`",
            f"- Average steps: `{summary.avg_steps:.2f}`",
            f"- Average reward: `{summary.avg_reward:.3f}`",
            f"- Average duration ms: `{summary.avg_duration_ms:.1f}`",
            f"- Average planner ms: `{summary.avg_planner_time_ms:.1f}`",
            f"- Fallback count: `{summary.fallback_count}`",
            "",
            "## Fallback Reasons",
            "",
            *_distribution_lines(summary.fallback_reasons),
            "",
            "## Failure Reasons",
            "",
            *_distribution_lines(summary.failure_reasons),
            "",
            "## Episodes",
            "",
            *episode_rows,
            "",
        ]
    )


def infer_agentos_failure_reason(success: bool, steps: int, max_steps: int) -> str:
    if success:
        return "none"
    if steps >= max_steps:
        return "timeout"
    return "execution_failed"


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _distribution_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {count}" for key, count in sorted(values.items())]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, AgentOSBenchmarkSummary):
        data = asdict(value)
        data["episodes"] = [asdict(ep) for ep in value.episodes]
        return data
    if isinstance(value, AgentOSEpisodeResult):
        return asdict(value)
    return value
