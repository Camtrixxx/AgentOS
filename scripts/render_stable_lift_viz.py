from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw

from agent.robosuite_scripted_policy import RobosuiteLiftPolicy
from envs.robosuite_env import RobosuiteEnvAdapter, RobosuiteEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a stable robosuite Lift visualization.")
    parser.add_argument("--output-dir", default="outputs/robosuite_lift_viz")
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--continue-after-success", type=int, default=24)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("frame_*.png", "frame_*.ppm"):
        for path in out.glob(pattern):
            path.unlink()

    config = RobosuiteEnvConfig(
        task_name=args.task,
        robot=args.robot,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_name=args.camera,
        image_size=args.image_size,
    )
    env = RobosuiteEnvAdapter(config)
    obs = env.reset(instruction="lift the cube", target_color="red")
    policy = RobosuiteLiftPolicy()
    frames: list[Path] = []
    trace: list[dict[str, object]] = []
    first_success_step: int | None = None

    for step in range(140):
        action = policy.act({"robot": {"ee_position": obs["ee_position"]}, "objects": obs["objects"]})
        obs, reward, _done, info = env.step(action)
        cube_position = obs.get("objects", {}).get("cube", {}).get("position")
        record = {
            "step": step,
            "stage": policy.last_stage,
            "action": action,
            "reward": reward,
            "success": bool(info.get("success")),
            "ee_position": obs.get("ee_position"),
            "cube_position": cube_position,
        }
        trace.append(record)
        if first_success_step is None and info.get("success"):
            first_success_step = step
        if step % args.render_every == 0 or info.get("success"):
            image = env.env.sim.render(
                camera_name=args.camera,
                width=args.image_size,
                height=args.image_size,
                depth=False,
            )[::-1]
            image_obj = Image.fromarray(image).convert("RGB")
            _draw_overlay(image_obj, step, policy.last_stage, args.camera, cube_position)
            path = out / f"frame_{step:04d}.png"
            image_obj.save(path)
            frames.append(path)
        if first_success_step is not None and step >= first_success_step + args.continue_after_success:
            break

    env.close()
    (out / "viz_trace.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in trace),
        encoding="utf-8",
    )
    _write_contact_sheet(out, frames)
    _write_animations(out, frames)
    _write_viewers(out, frames)
    print(f"frames={len(frames)} viewer={out / 'viewer.html'}")


def _draw_overlay(image: Image.Image, step: int, stage: str, camera: str, cube_position: object) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), f"{step:03d} {stage}", fill=(255, 255, 255))
    draw.rectangle((0, image.height - 22, image.width, image.height), fill=(0, 0, 0))
    cube_z = 0.0
    if isinstance(cube_position, list) and len(cube_position) >= 3:
        cube_z = float(cube_position[2])
    draw.text((6, image.height - 17), f"{camera} cube_z={cube_z:.3f}", fill=(255, 255, 255))


def _write_contact_sheet(out: Path, frames: list[Path]) -> None:
    if not frames:
        return
    samples = [round(index * (len(frames) - 1) / 9) for index in range(10)] if len(frames) > 10 else list(range(len(frames)))
    thumbs = [Image.open(frames[index]).convert("RGB") for index in samples]
    sheet = Image.new("RGB", (thumbs[0].width * len(thumbs), thumbs[0].height), (255, 255, 255))
    for column, thumb in enumerate(thumbs):
        sheet.paste(thumb, (column * thumb.width, 0))
    sheet.save(out / "contact_sheet.png")


def _write_animations(out: Path, frames: list[Path]) -> None:
    if not frames:
        return
    images = [Image.open(path).convert("RGB").copy() for path in frames]
    images[0].save(out / "lift.webp", save_all=True, append_images=images[1:], duration=140, loop=0, quality=92, method=4)
    images[0].save(out / "lift.gif", save_all=True, append_images=images[1:], duration=140, loop=0, optimize=False, disposal=2)


def _write_viewers(out: Path, frames: list[Path]) -> None:
    if not frames:
        return
    frame_names = [path.name for path in frames]
    frames_json = json.dumps(frame_names)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Robosuite Lift Viewer</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh}}
.wrap{{width:min(900px,96vw)}}img{{width:100%;background:#222}}
.controls{{display:flex;gap:12px;align-items:center;margin:12px 0}}input{{flex:1}}button{{padding:8px 14px}}
</style></head><body><div class="wrap">
<h2>Robosuite Lift Viewer - stable offline render</h2>
<img id="frame" src="{frame_names[0]}">
<div class="controls"><button id="play">Pause</button><input id="slider" type="range" min="0" max="0" value="0"><span id="label"></span></div>
<p>Stable offline render. AgentOS still produced the execution trace; this viewer avoids headless live-render artifacts.</p>
</div><script>
const frames={frames_json};
const img=document.getElementById('frame');
const slider=document.getElementById('slider');
const label=document.getElementById('label');
const play=document.getElementById('play');
let i=0,playing=true;
slider.max=frames.length-1;
function show(n){{i=Math.max(0,Math.min(frames.length-1,n));img.src=frames[i];slider.value=i;label.textContent=(i+1)+'/'+frames.length+' '+frames[i];}}
slider.oninput=()=>{{playing=false;play.textContent='Play';show(Number(slider.value));}};
play.onclick=()=>{{playing=!playing;play.textContent=playing?'Pause':'Play';}};
setInterval(()=>{{if(playing)show((i+1)%frames.length);}},140);
show(0);
</script></body></html>
"""
    (out / "viewer.html").write_text(html, encoding="utf-8")
    data = base64.b64encode((out / "lift.webp").read_bytes()).decode("ascii")
    standalone = (
        "<!doctype html><meta charset='utf-8'><title>Robosuite Lift</title>"
        "<body style='margin:0;background:#111;color:#eee;font-family:system-ui'>"
        "<div style='width:min(900px,96vw);margin:32px auto'>"
        "<h2>Robosuite Lift Stable Render</h2>"
        f"<img style='width:100%' src='data:image/webp;base64,{data}'>"
        "<p>Standalone stable render.</p></div></body>"
    )
    (out / "viewer_standalone.html").write_text(standalone, encoding="utf-8")


if __name__ == "__main__":
    main()
