# SmolVLA and RL Integration

This project keeps SmolVLA and RL inside the same repository because both share the fake manipulation environment, datasets, VLA adapter, runtime evaluation, and reports.

## SmolVLA Path

SmolVLA should enter through the existing VLA boundary:

```text
FakeManipulationEnv observation
-> FakeEnvVLAAdapter
-> VLAObservation(image, instruction, state)
-> SmolVLABackend.predict(...)
-> VLAAction(ee_delta, gripper)
-> env action [dx, dy, gripper]
```

Current files:

```text
datasets/lerobot_exporter.py
scripts/export_lerobot_dataset.py
vla/smolvla_backend.py
```

Export local vision demos to a LeRobot-style manifest:

```bash
python scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir data/lerobot_fake_manipulation
```

LeRobot is installed in the Docker container as an isolated environment at:

```text
/workspace/hyh/.venvs/lerobot-smolvla
```

This keeps the base Ascend stack on `torch 2.1.0 + torch_npu 2.1.0`. Use the wrapper below for commands that need the LeRobot Python packages. It clears the container-level `PYTHONPATH` so LeRobot uses its own Hugging Face dependencies instead of the system packages:

```bash
scripts/run_lerobot_env.sh scripts/export_lerobot_dataset.py \
  --data-dir data/vision_demos_random \
  --output-dir data/lerobot_fake_manipulation \
  --format native
```

Native export writes both the stable manifest bridge and a LeRobotDataset under:

```text
data/lerobot_fake_manipulation/native_lerobot
```

Run a first real SmolVLA training smoke test:

```bash
python scripts/train_smolvla.py \
  --vision-data-dir data/vision_demos_random_smoke \
  --dataset-output-dir outputs/lerobot_smolvla_dataset \
  --output-dir outputs/smolvla_real_smoke \
  --steps 1 \
  --overwrite
```

By default this uses `HF_ENDPOINT=https://hf-mirror.com`, `device=cpu`, one VLM layer, one expert layer, and `load_vlm_weights=false`. That keeps the first run small enough to validate data loading, SmolVLA forward/backward, and checkpoint saving. For a real fine-tune from pretrained weights, increase data volume and pass `--load-vlm-weights`.

Evaluate the SmolVLA integration without requiring LeRobot installation:

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend smolvla_dry_run \
  --num-episodes 3 \
  --write-report
```

When LeRobot is installed, switch to:

```bash
python learning/evaluate_policy.py \
  --policy vla \
  --vla-backend smolvla \
  --smolvla-model lerobot/smolvla_base \
  --device npu
```

On the current Ascend Docker environment, use the last four NPUs with:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python ...
```

## RL Path

RL enters as another policy backend, not as a replacement for the environment:

```text
FakeManipulationGymEnv
-> RL trainer / checkpoint
-> RLPolicy
-> learning/evaluate_policy.py --policy rl
```

Current files:

```text
rl/gym_fake_manipulation.py
agent/rl_policy.py
scripts/train_rl.py
```

Smoke-test the Gym-style wrapper:

```bash
python scripts/train_rl.py --backend smoke
```

Evaluate deterministic RL wrapper baseline:

```bash
python learning/evaluate_policy.py \
  --policy rl \
  --rl-backend scripted \
  --num-episodes 3 \
  --write-report
```

Evaluate random RL baseline:

```bash
python learning/evaluate_policy.py \
  --policy rl \
  --rl-backend random \
  --num-episodes 3
```

When `stable-baselines3` is installed:

```bash
python scripts/train_rl.py \
  --backend sb3 \
  --timesteps 10000 \
  --output checkpoints/rl_ppo_fake_manipulation.zip

python learning/evaluate_policy.py \
  --policy rl \
  --rl-backend sb3 \
  --checkpoint checkpoints/rl_ppo_fake_manipulation.zip
```
