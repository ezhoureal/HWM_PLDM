#!/usr/bin/env python3
"""Train a small action student from HWM/PLDM teacher labels.

Expected data format is a torch-saved dict with:
  - z_t: [N, D]
  - action: [N, A]
and, depending on --mode:
  - z_subgoal: [N, D]
  - z_g: [N, D]

Use --synthetic for a local smoke test without collected teacher rollouts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pldm.planning.distillation import FEATURE_KEYS, PolicyMLP


@dataclass
class TrainStats:
    train_mse: float
    val_mse: float
    n_train: int
    n_val: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/distill_policy.pt"))
    parser.add_argument(
        "--mode",
        choices=sorted(FEATURE_KEYS),
        default="low",
        help="Student conditioning choice.",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=8192)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_synthetic_data(
    n_samples: int,
    latent_dim: int,
    action_dim: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z_t = torch.randn(n_samples, latent_dim, generator=generator)
    z_subgoal = torch.randn(n_samples, latent_dim, generator=generator)
    z_g = torch.randn(n_samples, latent_dim, generator=generator)

    # Smooth bounded teacher with both local subgoal and global-goal signal.
    w_low = torch.randn(latent_dim * 2, action_dim, generator=generator)
    w_goal = torch.randn(latent_dim * 2, action_dim, generator=generator)
    low_in = torch.cat([z_t, z_subgoal], dim=-1)
    goal_in = torch.cat([z_t, z_g], dim=-1)
    action = 0.75 * torch.tanh(low_in @ w_low / latent_dim**0.5)
    action += 0.25 * torch.tanh(goal_in @ w_goal / latent_dim**0.5)
    action += 0.03 * torch.randn(n_samples, action_dim, generator=generator)
    action = action.clamp(-1.0, 1.0)

    return {
        "z_t": z_t,
        "z_subgoal": z_subgoal,
        "z_g": z_g,
        "action": action,
    }


def load_data(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    if args.synthetic:
        return make_synthetic_data(
            n_samples=args.synthetic_samples,
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            seed=args.seed,
        )
    if args.data is None:
        raise ValueError("Provide --data or pass --synthetic.")
    data = torch.load(args.data, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {args.data}, got {type(data)!r}")
    return data


def build_tensors(
    data: Dict[str, torch.Tensor],
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = [key for key in (*FEATURE_KEYS[mode], "action") if key not in data]
    if missing:
        raise KeyError(f"Missing keys for mode={mode}: {missing}")
    features = [data[key].float().flatten(start_dim=1) for key in FEATURE_KEYS[mode]]
    x = torch.cat(features, dim=-1)
    y = data["action"].float().flatten(start_dim=1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/action count mismatch: {x.shape[0]} vs {y.shape[0]}")
    return x, y


def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            total_loss += float(loss.item()) * x.shape[0]
            total_count += x.shape[0]
    return total_loss / max(total_count, 1)


def train(args: argparse.Namespace) -> tuple[nn.Module, TrainStats]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    data = load_data(args)
    x, y = build_tensors(data, args.mode)

    dataset = TensorDataset(x, y)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = PolicyMLP(
        input_dim=x.shape[-1],
        action_dim=y.shape[-1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = torch.mean((pred - batch_y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_mse = evaluate(model, train_loader, device)
        val_mse = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch + 1:03d} train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f}"
        )

    stats = TrainStats(
        train_mse=evaluate(model, train_loader, device),
        val_mse=evaluate(model, val_loader, device),
        n_train=n_train,
        n_val=n_val,
    )
    return model.cpu(), stats


def main() -> None:
    args = parse_args()
    model, stats = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": args.mode,
            "feature_keys": FEATURE_KEYS[args.mode],
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "stats": stats.__dict__,
        },
        args.out,
    )
    print(f"saved={args.out}")
    print(
        f"final train_mse={stats.train_mse:.6f} val_mse={stats.val_mse:.6f} "
        f"n_train={stats.n_train} n_val={stats.n_val}"
    )


if __name__ == "__main__":
    main()
