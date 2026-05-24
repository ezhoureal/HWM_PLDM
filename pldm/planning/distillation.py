from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn

from pldm.models.utils import flatten_conv_output


FEATURE_KEYS = {
    "low": ("z_t", "z_subgoal"),
    "goal": ("z_t", "z_g"),
    "goal_subgoal": ("z_t", "z_g", "z_subgoal"),
}


class PolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flat_cpu(x: torch.Tensor) -> torch.Tensor:
    return flatten_conv_output(x.detach()).float().cpu()


def build_policy_input(
    mode: str,
    z_t: torch.Tensor,
    z_subgoal: torch.Tensor,
    z_g: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    values = {
        "z_t": flatten_conv_output(z_t.float()),
        "z_subgoal": flatten_conv_output(z_subgoal.float()),
        "z_g": flatten_conv_output(z_g.float()) if z_g is not None else None,
    }
    missing = [key for key in FEATURE_KEYS[mode] if values[key] is None]
    if missing:
        raise ValueError(f"Missing policy inputs for mode={mode}: {missing}")
    return torch.cat([values[key] for key in FEATURE_KEYS[mode]], dim=-1)


def infer_mlp_dims(state_dict: Dict[str, torch.Tensor]) -> tuple[int, int]:
    first_weight = state_dict["net.0.weight"]
    last_weight_key = sorted(
        key for key in state_dict if key.startswith("net.") and key.endswith(".weight")
    )[-1]
    last_weight = state_dict[last_weight_key]
    return first_weight.shape[1], last_weight.shape[0]


def load_policy_checkpoint(
    path: str,
    device: torch.device,
    mode_override: Optional[str] = None,
) -> tuple[PolicyMLP, str, Sequence[str]]:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    input_dim, action_dim = infer_mlp_dims(state_dict)
    hidden_dim = int(ckpt["hidden_dim"])
    depth = int(ckpt["depth"])
    mode = mode_override or ckpt.get("mode", "low")
    feature_keys = ckpt.get("feature_keys", FEATURE_KEYS[mode])

    policy = PolicyMLP(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        depth=depth,
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy, mode, feature_keys


@dataclass
class DistilledPolicyController:
    policy: PolicyMLP
    mode: str
    device: torch.device
    max_action_abs: Optional[float] = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        mode_override: Optional[str] = None,
        max_action_abs: Optional[float] = None,
    ) -> "DistilledPolicyController":
        policy, mode, _ = load_policy_checkpoint(
            path=path,
            device=device,
            mode_override=mode_override,
        )
        return cls(
            policy=policy,
            mode=mode,
            device=device,
            max_action_abs=max_action_abs,
        )

    @torch.no_grad()
    def act(
        self,
        z_t: torch.Tensor,
        z_subgoal: torch.Tensor,
        z_g: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = build_policy_input(self.mode, z_t=z_t, z_subgoal=z_subgoal, z_g=z_g)
        action = self.policy(x.to(self.device)).detach()
        fallback_mask = torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
        if self.max_action_abs is not None:
            fallback_mask = action.abs().amax(dim=-1) > self.max_action_abs
        return action, fallback_mask


class DistillationTraceCollector:
    def __init__(
        self,
        out_dir: str,
        shard_size: int = 10000,
        prefix: str = "hwm_low_level",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self.records = []
        self.shard_idx = 0

    def __len__(self) -> int:
        return len(self.records)

    def record_batch(
        self,
        z_t: torch.Tensor,
        z_subgoal: torch.Tensor,
        action: torch.Tensor,
        z_g: Optional[torch.Tensor],
        target: torch.Tensor,
        reward: torch.Tensor,
        success: torch.Tensor,
        step: int,
        env_offset: int,
        source: str,
    ) -> None:
        z_t = _flat_cpu(z_t)
        z_subgoal = _flat_cpu(z_subgoal)
        action = action.detach().float().cpu()
        z_g = _flat_cpu(z_g) if z_g is not None else None
        target = target.detach().float().cpu()
        reward = reward.detach().float().cpu()
        success = success.detach().float().cpu()

        for batch_idx in range(action.shape[0]):
            record = {
                "z_t": z_t[batch_idx],
                "z_subgoal": z_subgoal[batch_idx],
                "action": action[batch_idx],
                "target": target[batch_idx],
                "reward": reward[batch_idx],
                "success": success[batch_idx],
                "step": torch.tensor(step),
                "env_id": torch.tensor(env_offset + batch_idx),
                "source": source,
            }
            if z_g is not None:
                record["z_g"] = z_g[batch_idx]
            self.records.append(record)

            if len(self.records) >= self.shard_size:
                self.flush()

    def flush(self) -> Optional[Path]:
        if not self.records:
            return None

        keys = [key for key in self.records[0] if key != "source"]
        shard = {key: torch.stack([record[key] for record in self.records]) for key in keys}
        shard["source"] = [record["source"] for record in self.records]
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:05d}.pt"
        torch.save(shard, path)
        self.records.clear()
        self.shard_idx += 1
        return path

    def close(self) -> Optional[Path]:
        return self.flush()
