from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = Path("/workspace/hyh/.venvs/lerobot-smolvla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LeRobot SmolVLA training job on exported demos.")
    parser.add_argument("--vision-data-dir", default="data/vision_demos_random_smoke")
    parser.add_argument("--dataset-output-dir", default="outputs/lerobot_smolvla_dataset")
    parser.add_argument("--dataset-root", default=None, help="Existing native_lerobot dataset root.")
    parser.add_argument("--output-dir", default="outputs/smolvla_train")
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-vlm-layers", type=int, default=1)
    parser.add_argument("--num-expert-layers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--load-vlm-weights", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv = Path(args.venv)
    python_bin = venv / "bin" / "python"
    train_bin = venv / "bin" / "lerobot-train"
    if not train_bin.exists():
        raise FileNotFoundError(f"LeRobot train entrypoint not found: {train_bin}")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is None:
        dataset_output_dir = PROJECT_ROOT / args.dataset_output_dir
        if args.overwrite and dataset_output_dir.exists():
            shutil.rmtree(dataset_output_dir)
        if not (dataset_output_dir / "native_lerobot" / "meta" / "info.json").exists():
            export_cmd = [
                str(python_bin),
                str(PROJECT_ROOT / "scripts" / "export_lerobot_dataset.py"),
                "--data-dir",
                str(PROJECT_ROOT / args.vision_data_dir),
                "--output-dir",
                str(dataset_output_dir),
                "--format",
                "native",
            ]
            run(export_cmd, cwd=PROJECT_ROOT, env=lerobot_env(args))
        dataset_root = dataset_output_dir / "native_lerobot"
    elif not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    train_cmd = [
        str(train_bin),
        "--dataset.repo_id",
        "embodied_lab_fake_manipulation",
        "--dataset.root",
        str(dataset_root),
        "--policy.type",
        "smolvla",
        "--policy.device",
        args.device,
        "--policy.push_to_hub",
        "false",
        "--policy.load_vlm_weights",
        str(bool(args.load_vlm_weights)).lower(),
        "--policy.num_vlm_layers",
        str(args.num_vlm_layers),
        "--policy.num_expert_layers",
        str(args.num_expert_layers),
        "--policy.resize_imgs_with_padding",
        f"[{args.image_size}, {args.image_size}]",
        "--policy.chunk_size",
        str(args.chunk_size),
        "--policy.n_action_steps",
        str(args.n_action_steps),
        "--policy.num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--num_workers",
        str(args.num_workers),
        "--eval_freq",
        "0",
        "--log_freq",
        "1",
        "--save_freq",
        str(args.steps),
        "--output_dir",
        str(output_dir),
        "--wandb.enable",
        "false",
    ]
    run(train_cmd, cwd=Path("/workspace"), env=lerobot_env(args))
    print(f"checkpoint_dir={output_dir / 'checkpoints' / f'{args.steps:06d}' / 'pretrained_model'}")


def lerobot_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    return env


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
