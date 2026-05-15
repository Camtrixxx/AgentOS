from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeEval:
    episode: int
    target_color: str
    success: bool
    steps: int
    total_reward: float
    failure_reason: str


@dataclass(frozen=True)
class EvaluationSummary:
    policy: str
    checkpoint: str | None
    num_episodes: int
    max_steps: int
    randomize_layout: bool
    success_rate: float
    avg_steps: float
    avg_reward: float
    episodes: list[EpisodeEval]


def infer_failure_reason(success: bool, steps: int, max_steps: int) -> str:
    if success:
        return "none"
    if steps >= max_steps:
        return "timeout"
    return "failed"


def write_evaluation_report(summary: EvaluationSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{timestamp}_{summary.policy}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return json_path, md_path


def render_markdown_report(summary: EvaluationSummary) -> str:
    checkpoint = summary.checkpoint if summary.checkpoint else "none"
    rows = [
        "| Episode | Target | Success | Steps | Reward | Failure Reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for ep in summary.episodes:
        rows.append(
            f"| {ep.episode} | {ep.target_color} | {ep.success} | "
            f"{ep.steps} | {ep.total_reward:.3f} | {ep.failure_reason} |"
        )

    return "\n".join(
        [
            "# Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Policy: `{summary.policy}`",
            f"- Checkpoint: `{checkpoint}`",
            f"- Episodes: `{summary.num_episodes}`",
            f"- Max steps: `{summary.max_steps}`",
            f"- Randomize layout: `{summary.randomize_layout}`",
            f"- Success rate: `{summary.success_rate:.3f}`",
            f"- Average steps: `{summary.avg_steps:.2f}`",
            f"- Average reward: `{summary.avg_reward:.3f}`",
            "",
            "## Episodes",
            "",
            *rows,
            "",
        ]
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, EvaluationSummary):
        data = asdict(value)
        data["episodes"] = [asdict(ep) for ep in value.episodes]
        return data
    if isinstance(value, EpisodeEval):
        return asdict(value)
    return value

