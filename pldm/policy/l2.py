from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

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
    encoder_layers: int = 2
    use_relative_features: bool = True


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


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim * 4, hidden_dim),
            ]
        )
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _make_encoder(input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
    layers = []
    dim = input_dim
    for _ in range(max(1, num_layers)):
        layers.extend(
            [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
        )
        if dropout:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    return nn.Sequential(*layers)


class _LegacyL2LatentGoalPolicy(LatentActionSequencePolicy):
    """Compatibility wrapper for checkpoints saved with the shared flat MLP."""

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


class L2LatentGoalPolicy(nn.Module):
    """Amortized L2 controller that predicts the next latent macro action."""

    def __init__(self, config: L2PolicyConfig):
        super().__init__()
        self.config = config
        self.latent_policy_config = _to_latent_config(config)
        self.current_encoder = _make_encoder(
            config.current_dim,
            config.hidden_dim,
            config.encoder_layers,
            config.dropout,
        )
        self.goal_encoder = _make_encoder(
            config.final_goal_dim,
            config.hidden_dim,
            config.encoder_layers,
            config.dropout,
        )

        fusion_dim = config.hidden_dim * 4
        if config.use_relative_features and config.current_dim == config.final_goal_dim:
            fusion_dim += config.current_dim * 3

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            *[
                _ResidualBlock(config.hidden_dim, dropout=config.dropout)
                for _ in range(config.num_layers)
            ],
            nn.LayerNorm(config.hidden_dim),
        )
        self.head = nn.Linear(config.hidden_dim, config.action_dim)

    def _flatten_inputs(
        self,
        current_latents: torch.Tensor,
        final_goal_latents: torch.Tensor,
    ):
        return (
            current_latents.float().flatten(start_dim=1),
            final_goal_latents.float().flatten(start_dim=1),
        )

    def forward(
        self,
        current_latents: torch.Tensor,
        final_goal_latents: torch.Tensor,
    ):
        current, goal = self._flatten_inputs(current_latents, final_goal_latents)
        current_emb = self.current_encoder(current)
        goal_emb = self.goal_encoder(goal)
        emb_delta = goal_emb - current_emb
        features = [current_emb, goal_emb, emb_delta, current_emb * goal_emb]

        if self.config.use_relative_features and current.shape[-1] == goal.shape[-1]:
            raw_delta = goal - current
            features.extend([raw_delta, raw_delta.abs(), current * goal])

        fused = self.fusion(torch.cat(features, dim=-1))
        return self.head(self.trunk(fused))


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
    if isinstance(policy, _LegacyL2LatentGoalPolicy):
        save_generic_policy_checkpoint(policy, path, extra=extra)
        return

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
    saved_config = payload["policy_config"]
    if "input_dims" in saved_config:
        latent_config = LatentPolicyConfig(**saved_config)
        policy_config = L2PolicyConfig(
            current_dim=latent_config.input_dims["current_latents"],
            final_goal_dim=latent_config.input_dims["final_goal_latents"],
            action_dim=latent_config.action_dim,
            horizon=latent_config.horizon,
            hidden_dim=latent_config.hidden_dim,
            num_layers=latent_config.num_layers,
            dropout=latent_config.dropout,
        )
        policy = _LegacyL2LatentGoalPolicy(policy_config)
    else:
        policy_config = L2PolicyConfig(**saved_config)
        policy = L2LatentGoalPolicy(policy_config)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, payload
