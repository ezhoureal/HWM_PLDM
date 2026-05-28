from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import torch

from pldm.models.encoders.enums import BackboneOutput
from pldm.planning.planners.mppi_planner import LearnedDynamics
from pldm.planning.planners.planner import PlanningResult


class LatentPolicyPlanner(ABC):
    is_policy_planner = True

    def __init__(
        self,
        checkpoint_path: str,
        model,
        normalizer,
        prober=None,
        *,
        load_policy_checkpoint: Callable,
    ):
        self.model = model
        self.normalizer = normalizer
        self.prober = prober
        self.device = next(model.parameters()).device
        self.policy, self.checkpoint = load_policy_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.policy = self.policy.to(self.device)
        self.dynamics = LearnedDynamics(
            model,
            state_dim=model.spatial_repr_dim,
        )

    def _encode_current_state(
        self,
        current_state: Union[torch.Tensor, BackboneOutput],
        *,
        repr_input: bool,
        curr_proprio_pos: Optional[torch.Tensor],
        curr_proprio_vel: Optional[torch.Tensor],
        curr_locations: Optional[torch.Tensor],
    ) -> BackboneOutput:
        if repr_input:
            return current_state

        if self.model.backbone.using_proprio:
            if curr_proprio_vel is not None and curr_proprio_pos is not None:
                curr_proprio_states = torch.cat(
                    [curr_proprio_pos, curr_proprio_vel], dim=-1
                )
            elif curr_proprio_vel is not None:
                curr_proprio_states = curr_proprio_vel
            elif curr_proprio_pos is not None:
                curr_proprio_states = curr_proprio_pos
            else:
                raise ValueError("Need proprio states to plan")

            curr_proprio_states = curr_proprio_states.to(self.device)
        else:
            curr_proprio_states = None

        return self.model.backbone(
            current_state.to(self.device),
            proprio=curr_proprio_states,
            locations=curr_locations.to(self.device)
            if curr_locations is not None
            else None,
        )

    def _build_planning_result(
        self,
        backbone_output: BackboneOutput,
        actions: torch.Tensor,
        *,
        dynamics_actions: Optional[torch.Tensor] = None,
    ) -> PlanningResult:
        rollout_actions = actions if dynamics_actions is None else dynamics_actions

        ensemble_state_input = self.model.predictor._prepare_ensemble_input(
            backbone_output.encodings.detach()
        )
        ensemble_proprio_input = self.model.predictor._prepare_ensemble_input(
            backbone_output.proprio_component
        )
        ensemble_location_input = self.model.predictor._prepare_ensemble_input(
            backbone_output.location_component
        )
        ensemble_raw_location_input = self.model.predictor._prepare_ensemble_input(
            backbone_output.raw_locations
        )

        dynamics_output = self.dynamics(
            state=ensemble_state_input,
            proprio=ensemble_proprio_input,
            location=ensemble_location_input,
            raw_location=ensemble_raw_location_input,
            action=rollout_actions.permute(1, 0, 2),
            only_return_last=False,
            flatten_output=False,
        )

        pred_raw_locations = dynamics_output.raw_locations
        if pred_raw_locations is not None:
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_raw_locations
            ).detach()
        elif self.prober is not None:
            pred_locs = torch.stack(
                [self.prober(x) for x in dynamics_output.obs_component]
            )
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_locs
            ).detach()
        else:
            unnormed_locations = None

        return PlanningResult(
            ensemble_predictions=dynamics_output.ensemble_predictions,
            ensemble_obs_component=dynamics_output.ensemble_obs_component,
            ensemble_proprio_component=dynamics_output.ensemble_proprio_component,
            ensemble_raw_locations=dynamics_output.ensemble_raw_locations,
            pred_obs=dynamics_output.obs_component,
            pred_proprio=dynamics_output.proprio_component,
            pred_location=dynamics_output.location_component,
            raw_locations=pred_raw_locations,
            actions=actions,
            locations=unnormed_locations,
            losses=[0],
        )

    @abstractmethod
    def plan(
        self,
        current_state: Union[torch.Tensor, BackboneOutput],
        plan_size: int,
        repr_input: bool = True,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        del targets, repr_input
