from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.fake_manipulation_env import FakeManipulationConfig, FakeManipulationEnv
from hal.fake_manipulation_driver import FakeManipulationDriver
from learning.agentos_benchmark_report import (
    AgentOSBenchmarkSummary,
    AgentOSEpisodeResult,
    build_benchmark_summary,
    infer_agentos_failure_reason,
    write_agentos_benchmark_report,
)
from runtime.executor import AgentOSExecutor
from runtime.llm_planner import DeepSeekPlanner
from runtime.planner import Planner, RuleBasedPlanner
from runtime.trace import TraceLogger


SUPPORTED_COLORS = {"red", "blue", "green"}


@dataclass(frozen=True)
class AgentOSBenchmarkConfig:
    planner: str = "rule"
    num_episodes: int = 30
    max_steps: int = 80
    randomize_layout: bool = False
    seed: int = 0
    target_colors: tuple[str, ...] = ("red", "blue", "green")
    workspace_root: Path = Path("workspace/benchmarks")
    output_dir: Path = Path("outputs/agentos_benchmarks")
    write_report: bool = True
    run_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-episode AgentOS benchmark.")
    parser.add_argument("--planner", choices=["rule", "deepseek"], default="rule")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--randomize-layout", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-colors", default="red,blue,green")
    parser.add_argument("--workspace-root", default="workspace/benchmarks")
    parser.add_argument("--output-dir", default="outputs/agentos_benchmarks")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AgentOSBenchmarkConfig:
    return AgentOSBenchmarkConfig(
        planner=args.planner,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        randomize_layout=args.randomize_layout,
        seed=args.seed,
        target_colors=tuple(parse_target_colors(args.target_colors)),
        workspace_root=PROJECT_ROOT / args.workspace_root,
        output_dir=PROJECT_ROOT / args.output_dir,
        write_report=not args.no_write_report,
    )


def run_benchmark(config: AgentOSBenchmarkConfig) -> AgentOSBenchmarkSummary:
    run_id = config.run_id or _new_run_id()
    planner = _build_planner(config.planner)
    episodes: list[AgentOSEpisodeResult] = []

    for episode_idx in range(config.num_episodes):
        color = config.target_colors[episode_idx % len(config.target_colors)]
        instruction = f"pick up the {color} block and place it in the bowl"
        workspace = config.workspace_root / run_id / f"episode_{episode_idx:03d}"

        env_config = FakeManipulationConfig(
            workspace_low=np.array([-1.0, -1.0], dtype=float),
            workspace_high=np.array([1.0, 1.0], dtype=float),
            randomize_layout=config.randomize_layout,
        )
        env = FakeManipulationEnv(config=env_config, seed=config.seed + episode_idx)
        driver = FakeManipulationDriver(env=env)
        trace = TraceLogger(workspace / "traces")

        plan_start = time.monotonic()
        plan = planner.plan(instruction, target_color=color)
        planner_time_ms = (time.monotonic() - plan_start) * 1000.0
        fallback_reason = getattr(planner, "last_fallback_reason", None)

        render_output = workspace / "final.ppm"
        exec_start = time.monotonic()
        result = AgentOSExecutor(workspace=workspace, driver=driver, trace=trace).execute(
            plan,
            max_steps=config.max_steps,
            render_output=render_output,
        )
        duration_ms = (time.monotonic() - exec_start) * 1000.0
        failure_reason = infer_agentos_failure_reason(result.success, result.steps, config.max_steps)

        episodes.append(
            AgentOSEpisodeResult(
                episode=episode_idx,
                target_color=color,
                success=result.success,
                steps=result.steps,
                last_reward=result.last_reward,
                failure_reason=failure_reason,
                planner=config.planner,
                fallback_reason=fallback_reason,
                workspace=str(workspace),
                report_path=result.report_path,
                trace_path=result.trace_path,
                render_path=result.render_path,
                duration_ms=duration_ms,
                planner_time_ms=planner_time_ms,
            )
        )

    summary = build_benchmark_summary(
        run_id=run_id,
        planner=config.planner,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps,
        randomize_layout=config.randomize_layout,
        seed=config.seed,
        target_colors=list(config.target_colors),
        episodes=episodes,
    )
    if config.write_report:
        json_path, md_path = write_agentos_benchmark_report(summary, config.output_dir)
        summary = build_benchmark_summary(
            run_id=run_id,
            planner=config.planner,
            num_episodes=config.num_episodes,
            max_steps=config.max_steps,
            randomize_layout=config.randomize_layout,
            seed=config.seed,
            target_colors=list(config.target_colors),
            episodes=episodes,
            report_json=str(json_path),
            report_markdown=str(md_path),
        )
    return summary


def parse_target_colors(raw: str) -> list[str]:
    colors = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not colors:
        raise ValueError("target_colors must not be empty")
    unknown = [color for color in colors if color not in SUPPORTED_COLORS]
    if unknown:
        raise ValueError(f"unsupported target colors: {unknown}")
    return colors


def print_summary(summary: AgentOSBenchmarkSummary) -> None:
    print(f"run_id={summary.run_id}")
    print(f"planner={summary.planner}")
    print(f"episodes={summary.num_episodes}")
    print(f"success_rate={summary.success_rate:.3f}")
    print(f"avg_steps={summary.avg_steps:.2f}")
    print(f"avg_reward={summary.avg_reward:.3f}")
    print(f"avg_duration_ms={summary.avg_duration_ms:.1f}")
    print(f"avg_planner_time_ms={summary.avg_planner_time_ms:.1f}")
    print(f"fallback_count={summary.fallback_count}")
    if summary.report_json:
        print(f"report_json={summary.report_json}")
    if summary.report_markdown:
        print(f"report_markdown={summary.report_markdown}")


def main() -> None:
    args = parse_args()
    summary = run_benchmark(config_from_args(args))
    print_summary(summary)


def _build_planner(name: str) -> Planner:
    if name == "deepseek":
        return DeepSeekPlanner()
    return RuleBasedPlanner()


def _new_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:6]}"


if __name__ == "__main__":
    main()
