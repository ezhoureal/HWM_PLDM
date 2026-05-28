import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from pldm.policy.l1 import (
    L1LatentSubgoalPolicy,
    L1PlanningTraceDataset,
    make_policy_config_from_trace as make_l1_policy_config_from_trace,
    save_policy_checkpoint as save_l1_policy_checkpoint,
)
from pldm.policy.l2 import (
    L2LatentGoalPolicy,
    L2PlanningTraceDataset,
    make_policy_config_from_trace as make_l2_policy_config_from_trace,
    save_policy_checkpoint as save_l2_policy_checkpoint,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


POLICY_REGISTRY = {
    "l1": {
        "dataset_cls": L1PlanningTraceDataset,
        "policy_cls": L1LatentSubgoalPolicy,
        "config_fn": make_l1_policy_config_from_trace,
        "save_fn": save_l1_policy_checkpoint,
        "predict_fn": lambda policy, inputs: policy(
            inputs["current_latents"],
            inputs["subgoal_latents"],
            inputs["final_goal_latents"],
        ),
        "target_fn": lambda actions: actions,
    },
    "l2": {
        "dataset_cls": L2PlanningTraceDataset,
        "policy_cls": L2LatentGoalPolicy,
        "config_fn": make_l2_policy_config_from_trace,
        "save_fn": save_l2_policy_checkpoint,
        "predict_fn": lambda policy, inputs: policy(
            inputs["current_latents"],
            inputs["final_goal_latents"],
        ),
        "target_fn": lambda actions: actions[:, 0],
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an offline latent policy from planner traces"
    )
    parser.add_argument("--policy_level", choices=sorted(POLICY_REGISTRY), required=True)
    parser.add_argument("--trace_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry_entry = POLICY_REGISTRY[args.policy_level]

    dataset = registry_entry["dataset_cls"](args.trace_path)
    val_len = int(len(dataset) * args.val_fraction)
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("Trace dataset is too small for the requested validation split")

    generator = torch.Generator().manual_seed(args.seed)
    if val_len > 0:
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )

    policy_config = registry_entry["config_fn"](
        dataset.data,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    policy = registry_entry["policy_cls"](policy_config).to(device)
    input_keys = tuple(dataset.input_keys)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        policy.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in input_keys
            }
            actions = registry_entry["target_fn"](
                batch["actions"].to(device, non_blocking=True)
            )

            pred = registry_entry["predict_fn"](policy, inputs)
            loss = F.mse_loss(pred, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = None
        if val_loader is not None:
            policy.eval()
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                    inputs = {
                        key: batch[key].to(device, non_blocking=True)
                        for key in input_keys
                    }
                    actions = registry_entry["target_fn"](
                        batch["actions"].to(device, non_blocking=True)
                    )
                    pred = registry_entry["predict_fn"](policy, inputs)
                    val_losses.append(F.mse_loss(pred, actions).item())
            val_loss = float(np.mean(val_losses))

        score = val_loss if val_loss is not None else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss}")

        if score < best_val:
            best_val = score
            registry_entry["save_fn"](
                policy,
                args.output_path,
                extra={
                    "epoch": epoch,
                    "best_loss": best_val,
                    "policy_level": args.policy_level,
                    "trace_path": str(Path(args.trace_path).resolve()),
                    "history": history,
                },
            )

    print(f"saved best policy to {args.output_path} with loss {best_val:.6f}")


if __name__ == "__main__":
    main()
