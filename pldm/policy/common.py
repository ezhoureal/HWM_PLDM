from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class LatentPolicyConfig:
    input_keys: tuple[str, ...]
    input_dims: dict[str, int]
    action_dim: int
    horizon: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0


class LatentActionSequencePolicy(nn.Module):
    """Generic latent-conditioned sequence policy."""

    def __init__(self, config: LatentPolicyConfig):
        super().__init__()
        self.latent_policy_config = config
        input_dim = sum(config.input_dims[key] for key in config.input_keys)
        output_dim = config.horizon * config.action_dim

        layers = []
        dim = input_dim
        for _ in range(config.num_layers):
            layers.extend(
                [
                    nn.Linear(dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                ]
            )
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward_inputs(self, **inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs[self.latent_policy_config.input_keys[0]].shape[0]
        flat_inputs = [
            inputs[key].float().flatten(start_dim=1)
            for key in self.latent_policy_config.input_keys
        ]
        x = torch.cat(flat_inputs, dim=-1)
        actions = self.net(x)
        return actions.view(
            batch_size,
            self.latent_policy_config.horizon,
            self.latent_policy_config.action_dim,
        )


class LatentPlanningTraceDataset(Dataset):
    """Offline dataset of latent planner decisions."""

    def __init__(self, path: str, input_keys: Optional[Sequence[str]] = None):
        self.path = path
        self.data = torch.load(path, map_location="cpu", weights_only=True)
        saved_input_keys = self.data.get("input_keys")
        if saved_input_keys is None:
            if input_keys is None:
                raise ValueError(
                    "Trace file is missing input_keys metadata and no fallback input_keys "
                    "were provided."
                )
            self.input_keys = tuple(input_keys)
        else:
            self.input_keys = tuple(saved_input_keys)
        self.inputs = {
            key: self.data[key].float()
            for key in self.input_keys
        }
        self.actions = self.data["actions"].float()

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.inputs.items()}
        item["actions"] = self.actions[idx]
        return item


def flatten_policy_trace_chunks(
    trace_chunks: Sequence[Mapping[str, torch.Tensor]],
    *,
    input_keys: Sequence[str],
    kind: str,
    success_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    success_only: bool = True,
):
    if not trace_chunks:
        raise ValueError(f"No {kind} traces were collected")

    filtered_chunks = []
    dropped_examples = 0
    filter_type = "none"
    input_keys = tuple(input_keys)

    for chunk in trace_chunks:
        actions = chunk["actions"].float().cpu()
        keep = torch.ones(actions.shape[0], dtype=torch.bool)

        if success_only:
            if success_key is None:
                raise ValueError(f"{kind} does not define a success signal to filter on")
            if success_key not in chunk:
                raise ValueError(
                    f"Cannot filter {kind} traces by success: missing {success_key} "
                    f"for chunk batch {actions.shape[0]}"
                )
            keep = chunk[success_key].cpu().bool()
            filter_type = success_key

        dropped_examples += int((~keep).sum().item())
        if keep.any():
            filtered_chunks.append(
                {
                    key: chunk[key].float().cpu()[keep]
                    for key in input_keys
                }
                | {"actions": actions[keep]}
            )

    if not filtered_chunks:
        raise ValueError(f"Success filtering removed all {kind} traces")

    flattened = {
        key: torch.cat([chunk[key] for chunk in filtered_chunks], dim=0)
        for key in input_keys
    }
    actions = torch.cat([chunk["actions"] for chunk in filtered_chunks], dim=0)

    pre_sample_examples = int(actions.shape[0])
    if max_samples is not None and actions.shape[0] > max_samples:
        keep = torch.randperm(actions.shape[0])[:max_samples]
        flattened = {key: value[keep] for key, value in flattened.items()}
        actions = actions[keep]

    metadata = {
        "num_examples": int(actions.shape[0]),
        "num_examples_before_sampling": pre_sample_examples,
        "num_dropped_failed_examples": dropped_examples,
        "success_only": success_only,
        "success_filter": filter_type,
        "action_shape": tuple(actions.shape[1:]),
        "max_samples": max_samples,
    }
    for key, value in flattened.items():
        metadata[f"{key}_shape"] = tuple(value.shape[1:])

    return {
        "version": 2,
        "kind": kind,
        "input_keys": list(input_keys),
        **{key: value.contiguous() for key, value in flattened.items()},
        "actions": actions.contiguous(),
        "metadata": metadata,
    }


def make_latent_policy_config_from_trace(
    trace_data: dict,
    *,
    input_keys: Optional[Sequence[str]] = None,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
) -> LatentPolicyConfig:
    saved_input_keys = trace_data.get("input_keys")
    if saved_input_keys is None:
        if input_keys is None:
            raise ValueError(
                "Trace data is missing input_keys metadata and no fallback input_keys "
                "were provided."
            )
        input_keys = tuple(input_keys)
    else:
        input_keys = tuple(saved_input_keys)
    input_dims = {
        key: int(torch.tensor(trace_data[key].shape[1:]).prod().item())
        for key in input_keys
    }
    return LatentPolicyConfig(
        input_keys=input_keys,
        input_dims=input_dims,
        horizon=int(trace_data["actions"].shape[1]),
        action_dim=int(trace_data["actions"].shape[2]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def _atomic_torch_save(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def save_policy_checkpoint(policy: nn.Module, path: str, extra: Optional[dict] = None):
    if not hasattr(policy, "latent_policy_config"):
        raise ValueError("Policy is missing latent_policy_config for checkpoint export")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "policy_config": asdict(policy.latent_policy_config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)
