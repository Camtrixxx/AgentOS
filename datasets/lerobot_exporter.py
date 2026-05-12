from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learning.features import extract_state_features, parse_target_color
from runtime.environment_io import to_jsonable, utc_now_iso


@dataclass(frozen=True)
class LeRobotExportSummary:
    output_dir: Path
    num_episodes: int
    num_frames: int
    action_dim: int
    state_dim: int
    native: bool = False


def export_vision_demos_to_lerobot_jsonl(
    *,
    data_dir: Path,
    output_dir: Path,
    dataset_name: str = "embodied_lab_fake_manipulation",
    copy_images: bool = False,
) -> LeRobotExportSummary:
    """Export local vision demos into a LeRobot-like JSONL manifest.

    This is a lightweight bridge, not a full Hugging Face LeRobotDataset writer.
    It preserves the key schema names used by LeRobot-style policies:
    `observation.images.*`, `observation.state`, `task`, and `action`.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "frames.jsonl"
    images_out = output_dir / "images"
    if copy_images:
        images_out.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(data_dir.glob("episode_*/transitions.jsonl"))
    if not episode_files:
        raise FileNotFoundError(f"No vision episodes found under {data_dir}")

    num_frames = 0
    action_dim = 0
    state_dim = 0
    with frames_path.open("w", encoding="utf-8") as handle:
        for episode_index, transitions_path in enumerate(episode_files):
            episode_dir = transitions_path.parent
            with transitions_path.open("r", encoding="utf-8") as f:
                for frame_index, line in enumerate(f):
                    transition = json.loads(line)
                    observation = transition["observation"]
                    action = np.asarray(transition["action"], dtype=float)
                    state = extract_state_features(observation)
                    image_rel = Path(str(transition["image_path"]))
                    image_path = episode_dir / image_rel
                    exported_image_path = image_path
                    if copy_images:
                        dst = images_out / f"episode_{episode_index:06d}_{frame_index:06d}.npy"
                        shutil.copy2(image_path, dst)
                        exported_image_path = dst

                    row: dict[str, Any] = {
                        "dataset": dataset_name,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "timestamp": frame_index,
                        "task": observation["instruction"],
                        "target_color": parse_target_color(observation["instruction"]),
                        "observation.images.front": str(exported_image_path),
                        "observation.state": state.tolist(),
                        "action": action.tolist(),
                        "reward": float(transition.get("reward", 0.0)),
                        "done": bool(transition.get("done", False)),
                    }
                    handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
                    num_frames += 1
                    action_dim = int(action.shape[0])
                    state_dim = int(state.shape[0])

    summary = LeRobotExportSummary(
        output_dir=output_dir,
        num_episodes=len(episode_files),
        num_frames=num_frames,
        action_dim=action_dim,
        state_dim=state_dim,
    )
    metadata = {
        "schema_version": "embodied_lab.lerobot_export.v1",
        "created_at": utc_now_iso(),
        "dataset_name": dataset_name,
        "source_data_dir": str(data_dir),
        "num_episodes": summary.num_episodes,
        "num_frames": summary.num_frames,
        "features": {
            "observation.images.front": {"dtype": "uint8", "shape": [128, 128, 3], "format": "npy"},
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            "task": {"dtype": "string"},
        },
        "notes": [
            "This export is a LeRobot-style bridge manifest.",
            "Use it to validate data mapping before converting to a native LeRobotDataset.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def export_vision_demos_to_lerobot_native(
    *,
    data_dir: Path,
    output_dir: Path,
    dataset_name: str = "embodied_lab_fake_manipulation",
    copy_images: bool = False,
) -> LeRobotExportSummary:
    """Export local vision demos to a native LeRobotDataset when available."""

    LeRobotDataset = _import_lerobot_dataset()
    _patch_hf_datasets_fingerprint()

    summary = export_vision_demos_to_lerobot_jsonl(
        data_dir=data_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
        copy_images=copy_images,
    )

    native_root = Path(output_dir) / "native_lerobot"
    if native_root.exists():
        shutil.rmtree(native_root)
    features = {
        "observation.images.front": {"dtype": "image", "shape": (3, 128, 128), "names": ["channel", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (summary.state_dim,), "names": [f"state_{i}" for i in range(summary.state_dim)]},
        "action": {"dtype": "float32", "shape": (summary.action_dim,), "names": ["dx", "dy", "gripper"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=10,
        features=features,
        root=native_root,
        robot_type="fake_manipulation",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    frames_by_episode = _load_manifest_frames(Path(output_dir) / "frames.jsonl")
    for frames in frames_by_episode:
        for row in frames:
            image = np.load(row["observation.images.front"]).astype(np.uint8)
            dataset.add_frame(
                {
                    "observation.images.front": image,
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": row["task"],
                }
            )
        dataset.save_episode()

    marker: dict[str, Any] = {
        "native_requested": True,
        "native_writer": "lerobot.LeRobotDataset.create",
        "native_root": str(native_root),
        "lerobot_version": _get_lerobot_version(),
        "message": "Native LeRobotDataset export completed.",
    }
    (Path(output_dir) / "native_export_status.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return LeRobotExportSummary(
        output_dir=summary.output_dir,
        num_episodes=summary.num_episodes,
        num_frames=summary.num_frames,
        action_dim=summary.action_dim,
        state_dim=summary.state_dim,
        native=True,
    )


def _load_manifest_frames(frames_path: Path) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    with frames_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            while len(episodes) <= episode_index:
                episodes.append([])
            episodes[episode_index].append(row)
    return episodes


def _import_lerobot_dataset() -> Any:
    """Import LeRobotDataset despite this repo's local `datasets` package name.

    Hugging Face LeRobot imports the external `datasets` package. This project
    also has a top-level `datasets/` package, so native export temporarily
    removes the project root from import resolution before importing LeRobot.
    """

    project_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    local_datasets_module = sys.modules.get("datasets")
    try:
        sys.modules.pop("datasets", None)
        sys.path = [
            path
            for path in sys.path
            if path not in ("", str(project_root), str(project_root.resolve()))
        ]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "LeRobot is not installed or could not be imported. "
            "Use manifest export now, or run native export with the LeRobot virtualenv."
        ) from exc
    finally:
        sys.path = original_path
        if local_datasets_module is not None:
            sys.modules["datasets"] = local_datasets_module


def _get_lerobot_version() -> str:
    try:
        import lerobot

        return str(getattr(lerobot, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _patch_hf_datasets_fingerprint() -> None:
    """Avoid a dill recursion bug seen in the Ascend container's Python stack."""

    try:
        import datasets.arrow_dataset as arrow_dataset
        import datasets.fingerprint as fingerprint
    except Exception:
        return

    def stable_fingerprint(_dataset: Any) -> str:
        return "embodied_lab_lerobot_export"

    fingerprint.generate_fingerprint = stable_fingerprint
    arrow_dataset.generate_fingerprint = stable_fingerprint
