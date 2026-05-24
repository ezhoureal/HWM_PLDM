from typing import NamedTuple, Optional
from .planner import Planner, PlanningResult
import torch

from pldm.planning.distillation import DistilledPolicyController


class TwoLvlPlanningResult(NamedTuple):
    level1: PlanningResult
    level2: PlanningResult


class TwoLvlPlanner:
    def __init__(
        self,
        l1_planner: Planner,
        l2_planner: Planner,
        l2_step_skip: int,
        policy_controller: Optional[DistilledPolicyController] = None,
    ):
        self.l1_planner = l1_planner
        self.l2_planner = l2_planner
        self.l2_step_skip = l2_step_skip
        self.policy_controller = policy_controller
        self.cached_subgoal = None
        self.cached_z_g = None
        self.last_call_stats = {}

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.l2_planner.reset_targets(targets, repr_input=repr_input)
        self.cached_z_g = targets.detach()

    def set_policy_goal(self, z_g: torch.Tensor):
        self.cached_z_g = z_g.detach()

    @property
    def uses_distilled_policy(self) -> bool:
        return self.policy_controller is not None

    def _encode_l1(
        self,
        current_state: torch.Tensor,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
    ):
        proprio_l1 = None
        if curr_proprio_pos is not None and curr_proprio_vel is not None:
            proprio_l1 = torch.cat([curr_proprio_pos, curr_proprio_vel], dim=-1).cuda()
        elif curr_proprio_pos is not None:
            proprio_l1 = curr_proprio_pos.cuda()
        elif curr_proprio_vel is not None:
            proprio_l1 = curr_proprio_vel.cuda()

        locations_cuda = curr_locations.cuda() if curr_locations is not None else None

        return self.l1_planner.model.backbone(
            current_state.cuda(), proprio=proprio_l1, locations=locations_cuda
        )

    def _empty_policy_result(
        self,
        action: torch.Tensor,
        fallback_result: Optional[PlanningResult] = None,
    ) -> PlanningResult:
        if fallback_result is not None:
            return PlanningResult(
                ensemble_predictions=fallback_result.ensemble_predictions,
                ensemble_obs_component=fallback_result.ensemble_obs_component,
                ensemble_proprio_component=fallback_result.ensemble_proprio_component,
                ensemble_raw_locations=fallback_result.ensemble_raw_locations,
                pred_obs=fallback_result.pred_obs,
                pred_proprio=fallback_result.pred_proprio,
                pred_location=fallback_result.pred_location,
                raw_locations=fallback_result.raw_locations,
                actions=action,
                locations=fallback_result.locations,
                losses=fallback_result.losses,
            )

        device = action.device
        empty = torch.empty(0, 1, device=device)
        return PlanningResult(
            ensemble_predictions=empty,
            ensemble_obs_component=empty,
            ensemble_proprio_component=None,
            ensemble_raw_locations=None,
            pred_obs=empty,
            pred_proprio=None,
            pred_location=None,
            raw_locations=None,
            actions=action,
            locations=None,
            losses=[0],
        )

    def _policy_plan_from_backbone(
        self,
        backbone_output,
        plan_size: int,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
    ) -> PlanningResult:
        if self.cached_subgoal is None:
            raise RuntimeError("Cannot run low-level policy before an L2 subgoal exists")

        action, fallback_mask = self.policy_controller.act(
            z_t=backbone_output.encodings.detach(),
            z_subgoal=self.cached_subgoal,
            z_g=self.cached_z_g,
        )
        action = action.unsqueeze(1)

        fallback_result = None
        if fallback_mask.any():
            self.l1_planner.reset_targets(self.cached_subgoal, repr_input=True)
            fallback_result = self.l1_planner.plan(
                current_state=backbone_output,
                plan_size=plan_size,
                repr_input=True,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
            )
            fallback_action = fallback_result.actions[:, :1].to(action.device)
            action = torch.where(
                fallback_mask.view(-1, 1, 1),
                fallback_action,
                action,
            )

        self.last_call_stats = {
            "policy_calls": int(action.shape[0]),
            "policy_fallback_l1_calls": int(fallback_mask.sum().item()),
            "l1_planner_calls": int(fallback_mask.any().item()),
            "l1_planner_env_calls": int(fallback_mask.sum().item()),
        }
        return self._empty_policy_result(action=action, fallback_result=fallback_result)

    def act_low_level(
        self,
        current_state: torch.Tensor,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        plan_size: Optional[int] = None,
    ) -> PlanningResult:
        backbone_output = self._encode_l1(
            current_state=current_state,
            curr_proprio_pos=curr_proprio_pos,
            curr_proprio_vel=curr_proprio_vel,
            curr_locations=curr_locations,
        )
        return self._policy_plan_from_backbone(
            backbone_output=backbone_output,
            plan_size=plan_size or self.l2_step_skip,
            curr_proprio_pos=curr_proprio_pos,
            curr_proprio_vel=curr_proprio_vel,
        )

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
        backbone_output = self._encode_l1(
            current_state=current_state,
            curr_proprio_pos=curr_proprio_pos,
            curr_proprio_vel=curr_proprio_vel,
            curr_locations=curr_locations,
        )
        l2_result = self.l2_planner.plan(
            current_state=backbone_output,
            plan_size=plan_size,
            repr_input=True,
            # diff_loss_idx=diff_loss_idx.to(enc2.device),
        )
        self.cached_subgoal = l2_result.pred_obs[1].detach()

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
        elif self.uses_distilled_policy:
            l1_result = self._policy_plan_from_backbone(
                backbone_output=backbone_output,
                plan_size=self.l2_step_skip,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
            )
            self.last_call_stats["l2_planner_calls"] = 1
            self.last_call_stats["l2_planner_env_calls"] = batch_size
        else:
            self.l1_planner.reset_targets(
                self.cached_subgoal, repr_input=True
            )

            l1_result = self.l1_planner.plan(
                current_state=backbone_output,
                plan_size=self.l2_step_skip,
                repr_input=True,
                curr_proprio_pos=curr_proprio_pos,
                curr_proprio_vel=curr_proprio_vel,
            )
            self.last_call_stats = {
                "l2_planner_calls": 1,
                "l1_planner_calls": 1,
                "l2_planner_env_calls": batch_size,
                "l1_planner_env_calls": batch_size,
                "policy_calls": 0,
                "policy_fallback_l1_calls": 0,
            }

        return TwoLvlPlanningResult(level1=l1_result, level2=l2_result)
