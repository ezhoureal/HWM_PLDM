from typing import NamedTuple, Optional
from .planner import Planner, PlanningResult
import torch


class TwoLvlPlanningResult(NamedTuple):
    level1: PlanningResult
    level2: PlanningResult
    l1_policy_trace: Optional[dict] = None
    l2_policy_trace: Optional[dict] = None


class TwoLvlPlanner:
    def __init__(
        self,
        l1_planner: Planner,
        l2_planner: Planner,
        l2_step_skip: int,
    ):
        self.l1_planner = l1_planner
        self.l2_planner = l2_planner
        self.l2_step_skip = l2_step_skip

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.final_goal_latents = targets.detach()
        self.l2_planner.reset_targets(targets, repr_input=repr_input)

    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        repr_input: bool = False,
        mock_l1: bool = False,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        batch_size = current_state.shape[0]

        proprio_l1 = None
        if curr_proprio_pos is not None and curr_proprio_vel is not None:
            proprio_l1 = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
        elif curr_proprio_pos is not None:
            proprio_l1 = curr_proprio_pos.cuda()
        elif curr_proprio_vel is not None:
            proprio_l1 = curr_proprio_vel.cuda()

        locations_cuda = curr_locations.cuda() if curr_locations is not None else None

        backbone_output = self.l1_planner.model.backbone(
            current_state.cuda(), proprio=proprio_l1, locations=locations_cuda
        )
        l2_plan_kwargs = {
            "current_state": backbone_output,
            "plan_size": plan_size,
            "repr_input": True,
        }
        if getattr(self.l2_planner, "is_policy_planner", False):
            l2_plan_kwargs["final_goal_latents"] = self.final_goal_latents.detach()
        l2_result = self.l2_planner.plan(
            # diff_loss_idx=diff_loss_idx.to(enc2.device),
            **l2_plan_kwargs,
        )

        if mock_l1:
            # ONLY MAKES SENSE FOR WALL DATASET
            actions_l1 = (
                l2_result.actions[:, 0]
                .detach()
                .unsqueeze(1)
                .repeat(1, self.l2_step_skip, 1)
                .to(encs2.device)
            )
            actions_l1 = actions_l1 / self.l2_step_skip * 2
            locations_l1 = torch.zeros(
                (self.l2_step_skip + 1, batch_size, 1, 2),
                device=encs2.device,
            )
        else:
            if getattr(self.l1_planner, "is_policy_planner", False):
                l1_result = self.l1_planner.plan(
                    current_state=backbone_output,
                    plan_size=self.l2_step_skip,
                    repr_input=True,
                    curr_proprio_pos=curr_proprio_pos,
                    curr_proprio_vel=curr_proprio_vel,
                    curr_locations=curr_locations,
                    subgoal_latents=l2_result.pred_obs[1].detach(),
                    final_goal_latents=self.final_goal_latents.detach(),
                )
            else:
                self.l1_planner.reset_targets(
                    l2_result.pred_obs[1].detach(), repr_input=True
                )

                l1_result = self.l1_planner.plan(
                    current_state=backbone_output,
                    plan_size=self.l2_step_skip,
                    repr_input=True,
                    curr_proprio_pos=curr_proprio_pos,
                    curr_proprio_vel=curr_proprio_vel,
                )

        l1_policy_trace = {
            "current_latents": backbone_output.obs_component.detach(),
            "subgoal_latents": l2_result.pred_obs[1].detach(),
            "final_goal_latents": self.final_goal_latents.detach(),
            "actions": l1_result.actions.detach(),
        }
        if l2_result.locations is not None:
            l1_policy_trace["subgoal_locations"] = l2_result.locations[1].detach()

        l2_policy_trace = {
            "current_latents": backbone_output.obs_component.detach(),
            "final_goal_latents": self.final_goal_latents.detach(),
            "actions": l2_result.actions[:, :1].detach(),
        }

        return TwoLvlPlanningResult(
            level1=l1_result,
            level2=l2_result,
            l1_policy_trace=l1_policy_trace,
            l2_policy_trace=l2_policy_trace,
        )
