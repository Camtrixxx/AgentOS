from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.demo_dataset import DemoTransitionDataset
from learning.devices import resolve_torch_device
from learning.features import FEATURE_DIM
from learning.models import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy from recorded demos.")
    parser.add_argument("--data-dir", default="data/demos")
    parser.add_argument("--output", default="checkpoints/bc_policy.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--xy-loss-weight", type=float, default=10.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, or npu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    dataset = DemoTransitionDataset(PROJECT_ROOT / args.data_dir)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Need at least two transitions for train/validation split")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MLPPolicy(FEATURE_DIM, action_dim=3, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    action_weights = torch.tensor(
        [args.xy_loss_weight, args.xy_loss_weight, args.gripper_loss_weight],
        dtype=torch.float32,
    ).to(device)
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, actions in train_loader:
            features = features.to(device)
            actions = actions.to(device)
            pred = model(features)
            loss = weighted_mse_loss(pred, actions, action_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * features.shape[0]
            train_count += features.shape[0]

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            val_loss = evaluate_loss(model, val_loader, action_weights)
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss / train_count:.6f} "
                f"val_loss={val_loss:.6f}"
            )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "feature_dim": FEATURE_DIM,
            "action_dim": 3,
            "hidden_dim": args.hidden_dim,
            "num_transitions": len(dataset),
            "epochs": args.epochs,
            "xy_loss_weight": args.xy_loss_weight,
            "gripper_loss_weight": args.gripper_loss_weight,
        },
        output_path,
    )
    print(f"saved_checkpoint={output_path}")


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights.to(pred.device)).mean()


def evaluate_loss(model: MLPPolicy, loader: DataLoader, action_weights: torch.Tensor) -> float:
    model.eval()
    total = 0.0
    count = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            actions = actions.to(device)
            pred = model(features)
            loss = weighted_mse_loss(pred, actions, action_weights)
            total += float(loss.item()) * features.shape[0]
            count += features.shape[0]
    return total / max(count, 1)


if __name__ == "__main__":
    main()
