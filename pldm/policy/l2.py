from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from pldm.policy.common import (
    LatentActionSequencePolicy,
    LatentPlanningTraceDataset,
    LatentPolicyConfig,
    _atomic_torch_save,
    flatten_policy_trace_chunks,
    make_latent_policy_config_from_trace,
    save_policy_checkpoint as save_generic_policy_checkpoint,
)

L2_INPUT_KEYS = ("current_latents", "final_goal_latents")
L2_TRACE_KIND = "l2_latent_goal_policy_trace"


@dataclass
class L2PolicyConfig:
    current_dim: int
    final_goal_dim: int
    action_dim: int
    horizon: int = 1
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0


def _to_latent_config(config: L2PolicyConfig) -> LatentPolicyConfig:
    return LatentPolicyConfig(
        input_keys=L2_INPUT_KEYS,
        input_dims={
            "current_latents": config.current_dim,
            "final_goal_latents": config.final_goal_dim,
        },
        action_dim=config.action_dim,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


class L2LatentGoalPolicy(LatentActionSequencePolicy):
    """Amortized L2 controller that predicts the next latent macro action."""

    def __init__(self, config: L2PolicyConfig):
        super().__init__(_to_latent_config(config))
        self.config = config

    def forward(
        self,
        current_latents: torch.Tensor,
        final_goal_latents: torch.Tensor,
    ):
        return self.forward_inputs(
            current_latents=current_latents,
            final_goal_latents=final_goal_latents,
        ).squeeze(1)


class L2PlanningTraceDataset(LatentPlanningTraceDataset):
    def __init__(self, path: str):
        super().__init__(path, input_keys=L2_INPUT_KEYS)


def flatten_l2_policy_trace_chunks(
    trace_chunks,
    max_samples: Optional[int] = None,
    success_only: bool = True,
):
    return flatten_policy_trace_chunks(
        trace_chunks,
        input_keys=L2_INPUT_KEYS,
        kind=L2_TRACE_KIND,
        success_key="episode_success",
        max_samples=max_samples,
        success_only=success_only,
    )


def save_l2_planning_trace(
    trace_chunks,
    path: str,
    *,
    max_samples: Optional[int] = None,
    source: str = "h_l2_mpc",
    success_only: bool = True,
):
    trace = flatten_l2_policy_trace_chunks(
        trace_chunks,
        max_samples=max_samples,
        success_only=success_only,
    )
    trace["source"] = source
    output_path = Path(path)
    _atomic_torch_save(trace, output_path)
    print(
        f"saved {trace['metadata']['num_examples']} L2 latent policy trace examples "
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
    if int(trace_data["actions"].shape[1]) != 1:
        raise ValueError(
            "L2 policy traces must contain exactly one latent action per example."
        )
    config = make_latent_policy_config_from_trace(
        trace_data,
        input_keys=L2_INPUT_KEYS,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return L2PolicyConfig(
        current_dim=config.input_dims["current_latents"],
        final_goal_dim=config.input_dims["final_goal_latents"],
        action_dim=config.action_dim,
        horizon=1,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def save_policy_checkpoint(
    policy: L2LatentGoalPolicy,
    path: str,
    extra: Optional[dict] = None,
):
    save_generic_policy_checkpoint(policy, path, extra=extra)


def load_policy_checkpoint(path: str, map_location: Optional[str] = None):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    latent_config = LatentPolicyConfig(**payload["policy_config"])
    policy_config = L2PolicyConfig(
        current_dim=latent_config.input_dims["current_latents"],
        final_goal_dim=latent_config.input_dims["final_goal_latents"],
        action_dim=latent_config.action_dim,
        horizon=latent_config.horizon,
        hidden_dim=latent_config.hidden_dim,
        num_layers=latent_config.num_layers,
        dropout=latent_config.dropout,
    )
    policy = L2LatentGoalPolicy(policy_config)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, payload
