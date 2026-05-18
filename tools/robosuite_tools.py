from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from envs.ppm_writer import write_ppm
from agent.robosuite_scripted_policy import RobosuiteLiftPolicy
from runtime.environment_io import to_jsonable
from runtime.repository import WorkspaceRepository, resolve_repo
from tools.embodied_tools import StepEnvTool
from tools.response import ToolResponse


class RobosuiteLiftLoopTool:
    name = "robosuite_lift_loop"
    description = "Run a scripted robosuite Lift policy through ACTION.md and the watchdog."

    def __init__(
        self,
        workspace: str | Path | WorkspaceRepository = "workspace",
        driver: Any | None = None,
        policy: RobosuiteLiftPolicy | None = None,
    ):
        self._repo_params: str | Path | WorkspaceRepository = workspace
        self.step_tool = StepEnvTool(workspace, driver=driver)
        self.policy = policy or RobosuiteLiftPolicy()

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        repo = resolve_repo(self._repo_params, parameters)
        repo.initialize()
        max_steps = int(parameters.get("max_steps", 120))
        render_every = max(0, int(parameters.get("render_every", 0)))
        frames_dir = Path(parameters["frames_dir"]) if parameters.get("frames_dir") else None
        video_output = Path(parameters["video_output"]) if parameters.get("video_output") else None
        gif_duration_ms = int(parameters.get("gif_duration_ms", 80))
        continue_after_success_steps = max(0, int(parameters.get("continue_after_success_steps", 0)))
        visual_cameras = _parse_visual_cameras(parameters.get("visual_cameras"))
        success_tail_steps = 0
        records: list[dict[str, Any]] = []
        frame_paths: list[str] = []
        previous_frame_image: np.ndarray | None = None
        trace_path = frames_dir / "trace.jsonl" if frames_dir is not None else None
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
            if trace_path is not None:
                trace_path.write_text("", encoding="utf-8")

        for _ in range(max_steps):
            environment = repo.get_environment()
            episode = environment.get("episode", {}) if isinstance(environment.get("episode"), dict) else {}
            if bool(episode.get("success", False)) or bool(episode.get("done", False)):
                if bool(episode.get("success", False)) and success_tail_steps < continue_after_success_steps:
                    success_tail_steps += 1
                else:
                    gif_path = _write_gif(frame_paths, video_output, gif_duration_ms)
                    return ToolResponse.success(
                        "robosuite lift loop already complete",
                        data=_response_data(
                            bool(episode.get("success", False)),
                            records,
                            environment,
                            frame_paths,
                            trace_path,
                            gif_path,
                        ),
                    )

            if bool(episode.get("done", False)):
                return ToolResponse.success(
                    "robosuite lift loop already complete",
                    data={"success": bool(episode.get("success", False)), "step_records": records},
                )

            action = self.policy.act(environment)
            response = self.step_tool.run({"workspace": repo.paths.root, "action": action})
            updated_environment = repo.get_environment()
            updated_episode = updated_environment.get("episode", {}) if isinstance(updated_environment, dict) else {}
            frame_path = None
            if render_every and frames_dir is not None and len(records) % render_every == 0:
                frame_path, previous_frame_image = _render_frame(
                    self.step_tool.watchdog_tool.driver,
                    frames_dir,
                    len(records),
                    label=f"{len(records):03d} {self.policy.last_stage}",
                    cameras=visual_cameras,
                    previous_image=previous_frame_image,
                )
                if frame_path is not None:
                    frame_paths.append(str(frame_path))
            record = {
                "stage": self.policy.last_stage,
                "action": action,
                "tool_status": response.status.value,
                "tool_text": response.text,
                "step_count": updated_episode.get("step_count"),
                "reward": updated_episode.get("last_reward"),
                "success": bool(updated_episode.get("success", False)),
                "done": bool(updated_episode.get("done", False)),
            }
            if frame_path is not None:
                record["frame"] = str(frame_path)
            if response.error is not None:
                record["error"] = response.error
            records.append(record)
            if trace_path is not None:
                _append_trace(trace_path, record, updated_environment)
            if response.error is not None:
                return ToolResponse.failure(
                    response.error.get("code", "robosuite_lift_failed"),
                    response.error.get("message", response.text),
                    data=_response_data(False, records, updated_environment, frame_paths, trace_path, video_output),
                )
            if bool(updated_episode.get("success", False)):
                success_tail_steps += 1
            if (
                bool(updated_episode.get("success", False))
                and success_tail_steps > continue_after_success_steps
            ) or bool(updated_episode.get("done", False)):
                gif_path = _write_gif(frame_paths, video_output, gif_duration_ms)
                return ToolResponse.success(
                    "robosuite lift loop completed",
                    data=_response_data(
                        bool(updated_episode.get("success", False)),
                        records,
                        updated_environment,
                        frame_paths,
                        trace_path,
                        gif_path,
                    ),
                )

        environment = repo.get_environment()
        gif_path = _write_gif(frame_paths, video_output, gif_duration_ms)
        return ToolResponse.failure(
            "robosuite_lift_timeout",
            f"robosuite lift loop reached max_steps={max_steps}",
            data=_response_data(False, records, environment, frame_paths, trace_path, gif_path),
        )


def _parse_visual_cameras(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _render_frame(
    driver: Any,
    frames_dir: Path,
    index: int,
    *,
    label: str | None = None,
    cameras: list[str] | None = None,
    previous_image: np.ndarray | None = None,
) -> tuple[Path | None, np.ndarray | None]:
    if driver is None or not hasattr(driver, "env"):
        return None, previous_image
    try:
        image = _render_visual_image(driver, cameras or [])
        image = _stabilize_orientation(image, previous_image)
    except Exception:
        return None, previous_image
    path = frames_dir / f"frame_{index:04d}.ppm"
    write_ppm(path, image)
    png_path = path.with_suffix(".png")
    try:
        from PIL import Image

        image_obj = Image.open(path).convert("RGB")
        if label:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image_obj)
            draw.rectangle((0, 0, image_obj.width, 22), fill=(0, 0, 0))
            draw.text((6, 5), label, fill=(255, 255, 255))
            _draw_camera_labels(draw, cameras or [], image_obj.width, image_obj.height)
        image_obj.save(png_path)
        return png_path, image
    except Exception:
        return path, image


def _render_visual_image(driver: Any, cameras: list[str]) -> np.ndarray:
    sim = getattr(getattr(driver.env, "env", None), "sim", None)
    if not cameras or sim is None:
        return driver.env.render_rgb()
    width = int(getattr(getattr(driver.env, "config", None), "image_size", 256))
    height = width
    images = []
    for camera in cameras:
        image = sim.render(camera_name=camera, width=width, height=height, depth=False)
        image_array = np.asarray(image, dtype=np.uint8)[::-1]
        images.append(_normalize_camera_orientation(image_array, camera))
    if not images:
        return driver.env.render_rgb()
    return np.concatenate(images, axis=1)


def _normalize_camera_orientation(image: np.ndarray, camera: str) -> np.ndarray:
    if camera != "frontview" or image.ndim != 3 or image.shape[0] < 20:
        return image
    height = image.shape[0]
    gray = image.astype(float).mean(axis=2)
    top = float(gray[: height // 3].mean())
    bottom = float(gray[-height // 3 :].mean())
    if top > bottom + 20.0:
        return image[::-1]
    return image


def _stabilize_orientation(image: np.ndarray, previous_image: np.ndarray | None) -> np.ndarray:
    if previous_image is None or previous_image.shape != image.shape:
        return image
    flipped = image[::-1]
    original_score = float(np.mean(np.abs(image.astype(float) - previous_image.astype(float))))
    flipped_score = float(np.mean(np.abs(flipped.astype(float) - previous_image.astype(float))))
    if flipped_score + 2.0 < original_score:
        return flipped
    return image


def _draw_camera_labels(draw: Any, cameras: list[str], image_width: int, image_height: int) -> None:
    if not cameras:
        return
    cell_width = image_width // max(1, len(cameras))
    for index, camera in enumerate(cameras):
        x0 = index * cell_width
        y0 = image_height - 22
        draw.rectangle((x0, y0, x0 + cell_width, image_height), fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 5), camera, fill=(255, 255, 255))


def _append_trace(path: Path, record: dict[str, Any], environment: dict[str, Any]) -> None:
    robot = environment.get("robot", {}) if isinstance(environment.get("robot"), dict) else {}
    objects = environment.get("objects", {}) if isinstance(environment.get("objects"), dict) else {}
    cube = objects.get("cube", {}) if isinstance(objects.get("cube"), dict) else {}
    payload = {
        "frame": record.get("frame"),
        "stage": record.get("stage"),
        "action": record.get("action"),
        "step_count": record.get("step_count"),
        "reward": record.get("reward"),
        "success": record.get("success"),
        "done": record.get("done"),
        "ee_position": robot.get("ee_position"),
        "cube_position": cube.get("position"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def _write_gif(frame_paths: list[str], video_output: Path | None, duration_ms: int) -> str | None:
    if video_output is None or not frame_paths:
        return str(video_output) if video_output is not None else None
    try:
        from PIL import Image

        images = [Image.open(path).convert("RGB").copy() for path in frame_paths if Path(path).exists()]
        if not images:
            return None
        video_output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            video_output,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        _write_visual_sidecars(frame_paths, video_output, images, duration_ms)
        return str(video_output)
    except Exception:
        return None


def _write_visual_sidecars(frame_paths: list[str], video_output: Path, images: list[Any], duration_ms: int) -> None:
    try:
        webp_path = video_output.with_suffix(".webp")
        images[0].save(
            webp_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            quality=92,
            method=4,
        )
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw

        samples = _sample_indices(len(images), count=10)
        thumbs = []
        for index in samples:
            thumb = images[index].copy()
            thumb.thumbnail((256, 256))
            canvas = Image.new("RGB", (256, 256), (245, 245, 245))
            canvas.paste(thumb, ((256 - thumb.width) // 2, (256 - thumb.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 256, 22), fill=(0, 0, 0))
            draw.text((6, 5), Path(frame_paths[index]).stem, fill=(255, 255, 255))
            thumbs.append(canvas)
        sheet = Image.new("RGB", (256 * len(thumbs), 256), (255, 255, 255))
        for column, thumb in enumerate(thumbs):
            sheet.paste(thumb, (column * 256, 0))
        sheet.save(video_output.parent / "contact_sheet.png")
    except Exception:
        pass
    try:
        frame_names = [Path(path).name for path in frame_paths]
        html = _viewer_html(frame_names, duration_ms)
        (video_output.parent / "viewer.html").write_text(html, encoding="utf-8")
    except Exception:
        pass


def _sample_indices(size: int, *, count: int) -> list[int]:
    if size <= count:
        return list(range(size))
    return [round(index * (size - 1) / (count - 1)) for index in range(count)]


def _viewer_html(frame_names: list[str], duration_ms: int) -> str:
    frame_list = json.dumps(frame_names)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Robosuite Lift Viewer</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; display: grid; place-items: center; min-height: 100vh; }}
    .wrap {{ width: min(960px, 96vw); }}
    img {{ width: 100%; background: #222; }}
    .controls {{ display: flex; gap: 12px; align-items: center; margin: 12px 0; }}
    button {{ padding: 8px 14px; }}
    input {{ flex: 1; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Robosuite Lift Viewer</h2>
    <img id="frame" src="{frame_names[0] if frame_names else ""}">
    <div class="controls">
      <button id="play">Pause</button>
      <input id="slider" type="range" min="0" max="0" value="0">
      <span id="label"></span>
    </div>
    <p>Expected motion: move above the cube, descend, close the gripper, then lift the cube off the table.</p>
  </div>
  <script>
    const frames = {frame_list};
    const delay = {duration_ms};
    const img = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const label = document.getElementById("label");
    const play = document.getElementById("play");
    let index = 0;
    let playing = true;
    slider.max = Math.max(0, frames.length - 1);
    function show(next) {{
      index = Math.max(0, Math.min(frames.length - 1, next));
      img.src = frames[index];
      slider.value = index;
      label.textContent = `${{index + 1}}/${{frames.length}} ${{frames[index]}}`;
    }}
    slider.oninput = () => {{ playing = false; play.textContent = "Play"; show(Number(slider.value)); }};
    play.onclick = () => {{ playing = !playing; play.textContent = playing ? "Pause" : "Play"; }};
    setInterval(() => {{ if (playing && frames.length) show((index + 1) % frames.length); }}, delay);
    show(0);
  </script>
</body>
</html>
"""


def _response_data(
    success: bool,
    records: list[dict[str, Any]],
    environment: dict[str, Any],
    frame_paths: list[str],
    trace_path: Path | None,
    video_output: str | Path | None,
) -> dict[str, Any]:
    return {
        "success": success,
        "step_records": records,
        "environment": environment,
        "frames": frame_paths,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "video_output": str(video_output) if video_output is not None else None,
    }
