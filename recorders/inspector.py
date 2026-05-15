from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learning.features import parse_target_color
from runtime.environment_io import to_jsonable, utc_now_iso


@dataclass(frozen=True)
class DatasetInspection:
    data_dir: Path
    num_episodes: int
    num_transitions: int
    success_rate: float
    target_color_counts: dict[str, int]
    action_min: list[float]
    action_max: list[float]
    action_mean: list[float]
    action_std: list[float]
    missing_images: list[str]
    bad_action_shape: list[str]
    missing_keys: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_images and not self.bad_action_shape and not self.missing_keys


REQUIRED_TRANSITION_KEYS = {"observation", "action", "reward", "done", "info"}
VISION_IMAGE_KEYS = {"image_path", "next_image_path"}


def inspect_demo_dataset(data_dir: Path, *, expect_images: bool = False) -> DatasetInspection:
    data_dir = Path(data_dir)
    episode_dirs = sorted(path for path in data_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {data_dir}")

    successes = 0
    color_counts: Counter[str] = Counter()
    actions: list[np.ndarray] = []
    missing_images: list[str] = []
    bad_action_shape: list[str] = []
    missing_keys: list[str] = []
    num_transitions = 0

    for episode_dir in episode_dirs:
        metadata_path = episode_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            successes += int(bool(metadata.get("summary", {}).get("success", False)))

        transitions_path = episode_dir / "transitions.jsonl"
        if not transitions_path.exists():
            missing_keys.append(str(transitions_path))
            continue

        with transitions_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                transition = json.loads(line)
                num_transitions += 1
                missing = REQUIRED_TRANSITION_KEYS - set(transition)
                if missing:
                    missing_keys.append(f"{transitions_path}:{line_idx}: missing {sorted(missing)}")

                observation = transition.get("observation", {})
                if isinstance(observation, dict):
                    color_counts[parse_target_color(str(observation.get("instruction", "")))] += 1

                action = np.asarray(transition.get("action", []), dtype=float)
                if action.shape != (3,):
                    bad_action_shape.append(f"{transitions_path}:{line_idx}: {tuple(action.shape)}")
                else:
                    actions.append(action)

                if expect_images:
                    for key in VISION_IMAGE_KEYS:
                        image_rel = transition.get(key)
                        if not image_rel:
                            missing_images.append(f"{transitions_path}:{line_idx}: missing {key}")
                            continue
                        image_path = episode_dir / str(image_rel)
                        if not image_path.exists():
                            missing_images.append(str(image_path))

    action_array = np.asarray(actions, dtype=float) if actions else np.zeros((0, 3), dtype=float)
    return DatasetInspection(
        data_dir=data_dir,
        num_episodes=len(episode_dirs),
        num_transitions=num_transitions,
        success_rate=successes / max(len(episode_dirs), 1),
        target_color_counts=dict(color_counts),
        action_min=_stat(action_array, "min"),
        action_max=_stat(action_array, "max"),
        action_mean=_stat(action_array, "mean"),
        action_std=_stat(action_array, "std"),
        missing_images=missing_images,
        bad_action_shape=bad_action_shape,
        missing_keys=missing_keys,
    )


def write_inspection_report(inspection: DatasetInspection, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_quality.json"
    md_path = output_dir / "dataset_quality.md"
    payload = {
        "schema_version": "embodied_lab.dataset_quality.v1",
        "created_at": utc_now_iso(),
        "ok": inspection.ok,
        **to_jsonable(inspection.__dict__),
    }
    payload["data_dir"] = str(inspection.data_dir)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_inspection_markdown(inspection), encoding="utf-8")
    return json_path, md_path


def render_inspection_markdown(inspection: DatasetInspection) -> str:
    return "\n".join(
        [
            "# Dataset Quality Report",
            "",
            f"- Data dir: `{inspection.data_dir}`",
            f"- OK: `{inspection.ok}`",
            f"- Episodes: `{inspection.num_episodes}`",
            f"- Transitions: `{inspection.num_transitions}`",
            f"- Success rate: `{inspection.success_rate:.3f}`",
            f"- Target colors: `{inspection.target_color_counts}`",
            f"- Action min: `{inspection.action_min}`",
            f"- Action max: `{inspection.action_max}`",
            f"- Action mean: `{inspection.action_mean}`",
            f"- Action std: `{inspection.action_std}`",
            f"- Missing images: `{len(inspection.missing_images)}`",
            f"- Bad action shapes: `{len(inspection.bad_action_shape)}`",
            f"- Missing keys: `{len(inspection.missing_keys)}`",
            "",
        ]
    )


def _stat(actions: np.ndarray, name: str) -> list[float]:
    if actions.size == 0:
        return [0.0, 0.0, 0.0]
    fn = getattr(actions, name)
    return fn(axis=0).astype(float).tolist()

