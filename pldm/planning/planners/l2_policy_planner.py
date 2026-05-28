from __future__ import annotations

from typing import Optional, Union

import torch

from pldm.models.encoders.enums import BackboneOutput
from pldm.policy.l2 import load_policy_checkpoint
from pldm.planning.planners.latent_policy_planner import LatentPolicyPlanner


class L2PolicyPlanner(LatentPolicyPlanner):
    def __init__(self, checkpoint_path: str, model, normalizer, prober=None):
        super().__init__(
            checkpoint_path=checkpoint_path,
            model=model,
            normalizer=normalizer,
            prober=prober,
            load_policy_checkpoint=load_policy_checkpoint,
        )

    @torch.no_grad()
    def plan(
        self,
        current_state: Union[torch.Tensor, BackboneOutput],
        plan_size: int,
        repr_input: bool = True,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        final_goal_latents: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.Tensor] = None,
    ):
        del diff_loss_idx, plan_size

        if final_goal_latents is None:
            raise ValueError("L2PolicyPlanner requires final_goal_latents.")

        self.dynamics.before_planning_callback()
        try:
            backbone_output = self._encode_current_state(
                current_state,
                repr_input=repr_input,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
                curr_locations=curr_locations,
            )

            action = self.policy(
                backbone_output.obs_component.detach(),
                final_goal_latents.to(self.device).detach(),
            )
            actions = action.unsqueeze(1)
            return self._build_planning_result(backbone_output, actions)
        finally:
            self.dynamics.after_planning_callback()
