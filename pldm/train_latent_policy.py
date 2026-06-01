import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from pldm.policy.concat_traces import concat_policy_traces
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
    parser.add_argument(
        "--trace_path",
        help="Path to a single trace .pt file (or a combined file). "
             "Ignored if --trace_dir is provided.",
    )
    parser.add_argument(
        "--trace_dir",
        help="Directory containing multiple trace files (e.g. checkpoint/policy_traces). "
             "Will auto-discover and concatenate all matching L1 or L2 traces for --policy_level.",
    )
    parser.add_argument(
        "--trace_pattern",
        default=None,
        help="Glob pattern inside --trace_dir (default: l1_latent_*.pt or l2_latent_*.pt).",
    )
    parser.add_argument(
        "--combined_trace_cache",
        default=None,
        help="Optional path to write the concatenated trace (avoids re-concat on every run). "
             "If not given, a combined cache file is written inside --trace_dir.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resume training from this .pt checkpoint (loads policy weights, optimizer state if present, epoch, best_loss, and history). Continues toward --epochs.",
    )
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

    # Resolve the effective trace path (single file or auto-concat from directory)
    if args.trace_dir:
        level = args.policy_level
        d = Path(args.trace_dir).expanduser().resolve()
        pattern = args.trace_pattern or (f"{level}_latent_*.pt")

        if args.combined_trace_cache:
            cache_path = Path(args.combined_trace_cache).expanduser().resolve()
        else:
            cache_path = d / f"{level}_latent_combined_from_dir.pt"

        # Collect candidates while excluding generated cache files. If a caller
        # explicitly asks for a combined pattern, keep those matches except the
        # cache path we are about to overwrite.
        pattern_mentions_combined = "combined" in pattern.lower()
        candidates = [
            c
            for c in sorted(d.glob(pattern))
            if ".partial" not in c.name
            and c.resolve() != cache_path
            and (pattern_mentions_combined or "combined" not in c.name.lower())
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No {level} trace files found in {d} matching {pattern}"
            )

        print(f"[trace] Found {len(candidates)} {level.upper()} trace file(s) in {d}")
        print(f"[trace] Concatenating into cache: {cache_path}")

        concat_policy_traces(
            candidates,
            cache_path,
            level=level,
            source_tag="train-from-dir",
            verbose=True,
        )
        effective_trace_path = str(cache_path)
    else:
        if not args.trace_path:
            raise ValueError("--trace_path is required unless --trace_dir is used")
        effective_trace_path = args.trace_path

    dataset = registry_entry["dataset_cls"](effective_trace_path)
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

    # Resume logic (supports checkpoints written by this trainer)
    start_epoch = 0
    best_val = float("inf")
    history = []
    ckpt_arg = getattr(args, "checkpoint", None)
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg).expanduser().resolve()
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            policy.load_state_dict(ckpt["policy_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_val = float(ckpt.get("best_loss", float("inf")))
            history = list(ckpt.get("history", []))
            print(f"[resume] Loaded {ckpt_path.name} (epoch {start_epoch}, best_val={best_val:.6f}, {len(history)} history entries)")
        else:
            print(f"[warn] --checkpoint {ckpt_path} not found; starting from scratch")

    for epoch in range(start_epoch, args.epochs):
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
                    "trace_path": str(Path(effective_trace_path).resolve()),
                    "history": history,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )

    print(f"saved best policy to {args.output_path} with loss {best_val:.6f}")


if __name__ == "__main__":
    main()
