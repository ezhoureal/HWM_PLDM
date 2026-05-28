from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class L1PolicyConfig:
    current_dim: int
    subgoal_dim: int
    final_goal_dim: int
    action_dim: int = 2
    horizon: int = 10
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0


class L1LatentSubgoalPolicy(nn.Module):
    """Amortized L1 controller in latent space.

    Inputs are the current world latent, the L2-selected subgoal latent, and the
    final-goal latent. The output is a full primitive action sequence that can
    replace the online L1 sub-trajectory planner.
    """

    def __init__(self, config: L1PolicyConfig):
        super().__init__()
        self.config = config
        input_dim = config.current_dim + config.subgoal_dim + config.final_goal_dim
        output_dim = config.horizon * config.action_dim

        layers = []
        dim = input_dim
        for _ in range(config.num_layers):
            layers += [nn.Linear(dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.GELU()]
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        current_latents: torch.Tensor,
        subgoal_latents: torch.Tensor,
        final_goal_latents: torch.Tensor,
    ):
        batch_size = current_latents.shape[0]
        x = torch.cat(
            [
                current_latents.float().flatten(start_dim=1),
                subgoal_latents.float().flatten(start_dim=1),
                final_goal_latents.float().flatten(start_dim=1),
            ],
            dim=-1,
        )
        actions = self.net(x)
        return actions.view(batch_size, self.config.horizon, self.config.action_dim)


class L1PlanningTraceDataset(Dataset):
    """Offline dataset of latent L1 planner decisions."""

    def __init__(self, path: str):
        self.path = path
        self.data = torch.load(path, map_location="cpu", weights_only=True)
        self.current_latents = self.data["current_latents"].float()
        self.subgoal_latents = self.data["subgoal_latents"].float()
        self.final_goal_latents = self.data["final_goal_latents"].float()
        self.actions = self.data["actions"].float()

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return {
            "current_latents": self.current_latents[idx],
            "subgoal_latents": self.subgoal_latents[idx],
            "final_goal_latents": self.final_goal_latents[idx],
            "actions": self.actions[idx],
        }


def flatten_l1_policy_trace_chunks(
    trace_chunks,
    max_samples: Optional[int] = None,
    success_only: bool = True,
):
    if not trace_chunks:
        raise ValueError("No L1 latent policy traces were collected")

    filtered_chunks = []
    dropped_examples = 0
    filter_type = "none"
    for chunk in trace_chunks:
        actions = chunk["actions"].float().cpu()
        keep = torch.ones(actions.shape[0], dtype=torch.bool)

        if success_only:
            if "subgoal_success" in chunk:
                keep = chunk["subgoal_success"].cpu().bool()
                filter_type = "subgoal_success"
            else:
                raise ValueError(
                    "Cannot filter L1 policy traces by success: trace chunk is "
                    f"missing subgoal_success for chunk batch {actions.shape[0]}"
                )

        dropped_examples += int((~keep).sum().item())
        if keep.any():
            filtered_chunks.append(
                {
                    "current_latents": chunk["current_latents"].float().cpu()[keep],
                    "subgoal_latents": chunk["subgoal_latents"].float().cpu()[keep],
                    "final_goal_latents": chunk["final_goal_latents"].float().cpu()[keep],
                    "actions": actions[keep],
                }
            )

    if not filtered_chunks:
        raise ValueError("Success filtering removed all L1 latent policy traces")

    current_latents = torch.cat([x["current_latents"] for x in filtered_chunks], dim=0)
    subgoal_latents = torch.cat([x["subgoal_latents"] for x in filtered_chunks], dim=0)
    final_goal_latents = torch.cat([x["final_goal_latents"] for x in filtered_chunks], dim=0)
    actions = torch.cat([x["actions"] for x in filtered_chunks], dim=0)

    pre_sample_examples = int(actions.shape[0])
    if max_samples is not None and actions.shape[0] > max_samples:
        keep = torch.randperm(actions.shape[0])[:max_samples]
        current_latents = current_latents[keep]
        subgoal_latents = subgoal_latents[keep]
        final_goal_latents = final_goal_latents[keep]
        actions = actions[keep]

    return {
        "version": 2,
        "kind": "l1_latent_subgoal_policy_trace",
        "current_latents": current_latents.contiguous(),
        "subgoal_latents": subgoal_latents.contiguous(),
        "final_goal_latents": final_goal_latents.contiguous(),
        "actions": actions.contiguous(),
        "metadata": {
            "num_examples": int(actions.shape[0]),
            "num_examples_before_sampling": pre_sample_examples,
            "num_dropped_failed_examples": dropped_examples,
            "success_only": success_only,
            "success_filter": filter_type,
            "current_shape": tuple(current_latents.shape[1:]),
            "subgoal_shape": tuple(subgoal_latents.shape[1:]),
            "final_goal_shape": tuple(final_goal_latents.shape[1:]),
            "action_shape": tuple(actions.shape[1:]),
            "max_samples": max_samples,
        },
}


def _atomic_torch_save(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def save_l1_planning_trace(
    trace_chunks,
    path: str,
    *,
    max_samples: Optional[int] = None,
    source: str = "h_l1_mpc",
    success_only: bool = True,
):
    trace = flatten_l1_policy_trace_chunks(
        trace_chunks,
        max_samples=max_samples,
        success_only=success_only,
    )
    trace["source"] = source
    output_path = Path(path)
    _atomic_torch_save(trace, output_path)
    print(
        f"saved {trace['metadata']['num_examples']} L1 latent policy trace examples "
        f"to {output_path}"
    )
    return trace


def make_policy_config_from_trace(
    trace_data: dict,
    *,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
):
    return L1PolicyConfig(
        current_dim=int(torch.tensor(trace_data["current_latents"].shape[1:]).prod().item()),
        subgoal_dim=int(torch.tensor(trace_data["subgoal_latents"].shape[1:]).prod().item()),
        final_goal_dim=int(torch.tensor(trace_data["final_goal_latents"].shape[1:]).prod().item()),
        horizon=int(trace_data["actions"].shape[1]),
        action_dim=int(trace_data["actions"].shape[2]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def save_policy_checkpoint(policy: L1LatentSubgoalPolicy, path: str, extra: Optional[dict] = None):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "policy_config": asdict(policy.config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def load_policy_checkpoint(path: str, map_location: Optional[str] = None):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    policy_config = L1PolicyConfig(**payload["policy_config"])
    policy = L1LatentSubgoalPolicy(policy_config)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, payload
