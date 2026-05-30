import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class FakeBackboneOutput:
    def __init__(
        self,
        encodings,
        obs_component=None,
        proprio_component=None,
        location_component=None,
        raw_locations=None,
    ):
        self.encodings = encodings
        self._obs_component = obs_component
        self.proprio_component = proprio_component
        self.location_component = location_component
        self.raw_locations = raw_locations

    @property
    def obs_component(self):
        return self._obs_component if self._obs_component is not None else self.encodings


def _install_import_stubs():
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.OmegaConf = object
    omegaconf_module.MISSING = "???"
    sys.modules.setdefault("omegaconf", omegaconf_module)

    encoder_enums_module = types.ModuleType("pldm.models.encoders.enums")
    encoder_enums_module.BackboneOutput = FakeBackboneOutput
    sys.modules.setdefault("pldm.models.encoders.enums", encoder_enums_module)


_install_import_stubs()

from pldm.planning.planners import l1_policy_planner as l1_module  # noqa: E402
from pldm.planning.planners import l2_policy_planner as l2_module  # noqa: E402
from pldm.planning.planners import latent_policy_planner as latent_module  # noqa: E402
from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner  # noqa: E402
from pldm.planning.trace_paths import level_scoped_path  # noqa: E402
from pldm.policy.l2 import (  # noqa: E402
    L2LatentGoalPolicy,
    L2PolicyConfig,
    load_policy_checkpoint as load_l2_policy_checkpoint,
    save_policy_checkpoint as save_l2_policy_checkpoint,
)


class RecordingDynamics:
    instances = []

    def __init__(self, model, state_dim=None):
        self.model = model
        self.state_dim = state_dim
        self.before_calls = 0
        self.after_calls = 0
        self.actions = []
        RecordingDynamics.instances.append(self)

    def before_planning_callback(self):
        self.before_calls += 1

    def after_planning_callback(self):
        self.after_calls += 1

    def __call__(
        self,
        state,
        proprio,
        location,
        raw_location,
        action,
        only_return_last=True,
        flatten_output=True,
    ):
        del state, proprio, location, raw_location, only_return_last, flatten_output
        self.actions.append(action.detach().clone())

        horizon, batch, _ = action.shape
        obs_dim = self.model.spatial_repr_dim
        pred_obs = torch.zeros(horizon + 1, batch, obs_dim)
        for step in range(horizon + 1):
            pred_obs[step] = step

        return SimpleNamespace(
            ensemble_predictions=pred_obs.unsqueeze(0),
            ensemble_obs_component=pred_obs.unsqueeze(0),
            ensemble_proprio_component=None,
            ensemble_raw_locations=None,
            obs_component=pred_obs,
            proprio_component=None,
            location_component=None,
            raw_locations=None,
        )


class FakePredictor:
    ensemble_size = 1

    def _prepare_ensemble_input(self, value):
        if value is None:
            return None
        return value.unsqueeze(0)


class FakeBackbone:
    using_proprio = False

    def __init__(self):
        self.calls = []

    def __call__(self, current_state, proprio=None, locations=None):
        self.calls.append((current_state, proprio, locations))
        return FakeBackboneOutput(
            encodings=current_state + 10,
            obs_component=current_state + 20,
            proprio_component=None,
            location_component=None,
            raw_locations=None,
        )


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.spatial_repr_dim = 3
        self.backbone = FakeBackbone()
        self.predictor = FakePredictor()
        self.config = SimpleNamespace(action_dim=2)


class FakeNormalizer:
    def __init__(self):
        self.normalized_actions = []

    def normalize_action(self, actions):
        self.normalized_actions.append(actions.detach().clone())
        return actions + 100

    def unnormalize_location(self, locations):
        return locations


class FakeL1Policy(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.config = SimpleNamespace(horizon=actions.shape[1])
        self.actions = actions
        self.calls = []

    def forward(self, current_latents, subgoal_latents, final_goal_latents):
        self.calls.append((current_latents, subgoal_latents, final_goal_latents))
        return self.actions


class FakeL2Policy(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.calls = []

    def forward(self, current_latents, final_goal_latents):
        self.calls.append((current_latents, final_goal_latents))
        return self.action


def _patch_policy_planner_dependencies(monkeypatch):
    RecordingDynamics.instances.clear()
    monkeypatch.setattr(latent_module, "LearnedDynamics", RecordingDynamics)


def test_level_scoped_path_prevents_policy_trace_overwrites():
    assert level_scoped_path(None, "medium") is None
    assert (
        level_scoped_path("/tmp/policy_traces/l2_latent.pt", "medium")
        == "/tmp/policy_traces/l2_latent_medium.pt"
    )
    assert (
        level_scoped_path("/tmp/policy_traces/l2_{level}.pt", "hard")
        == "/tmp/policy_traces/l2_hard.pt"
    )
    assert (
        level_scoped_path("/tmp/policy_traces/l2_latent", "easy")
        == "/tmp/policy_traces/l2_latent_easy"
    )


def test_l1_policy_planner_uses_policy_actions_and_normalized_rollout(monkeypatch):
    _patch_policy_planner_dependencies(monkeypatch)
    model = FakeModel()
    normalizer = FakeNormalizer()
    policy_actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    policy = FakeL1Policy(policy_actions)
    monkeypatch.setattr(
        l1_module,
        "load_policy_checkpoint",
        lambda path, map_location=None: (policy, {"path": path}),
    )

    planner = l1_module.L1PolicyPlanner("l1.ckpt", model=model, normalizer=normalizer)
    current = FakeBackboneOutput(
        encodings=torch.tensor([[0.0, 1.0, 2.0]]),
        obs_component=torch.tensor([[10.0, 11.0, 12.0]]),
    )
    subgoal = torch.tensor([[20.0, 21.0, 22.0]])
    goal = torch.tensor([[30.0, 31.0, 32.0]])

    result = planner.plan(
        current_state=current,
        plan_size=2,
        repr_input=True,
        subgoal_latents=subgoal,
        final_goal_latents=goal,
    )

    assert result.actions.shape == (1, 2, 2)
    torch.testing.assert_close(result.actions, policy_actions[:, :2])
    torch.testing.assert_close(normalizer.normalized_actions[0], policy_actions[:, :2])
    torch.testing.assert_close(
        RecordingDynamics.instances[0].actions[0],
        (policy_actions[:, :2] + 100).permute(1, 0, 2),
    )
    torch.testing.assert_close(policy.calls[0][0], current.obs_component)
    torch.testing.assert_close(policy.calls[0][1], subgoal)
    torch.testing.assert_close(policy.calls[0][2], goal)
    assert RecordingDynamics.instances[0].before_calls == 1
    assert RecordingDynamics.instances[0].after_calls == 1


def test_l2_policy_planner_predicts_single_macro_action(monkeypatch):
    _patch_policy_planner_dependencies(monkeypatch)
    model = FakeModel()
    normalizer = FakeNormalizer()
    policy_action = torch.tensor([[7.0, 8.0]])
    policy = FakeL2Policy(policy_action)
    monkeypatch.setattr(
        l2_module,
        "load_policy_checkpoint",
        lambda path, map_location=None: (policy, {"path": path}),
    )

    planner = l2_module.L2PolicyPlanner("l2.ckpt", model=model, normalizer=normalizer)
    current = FakeBackboneOutput(
        encodings=torch.tensor([[0.0, 1.0, 2.0]]),
        obs_component=torch.tensor([[10.0, 11.0, 12.0]]),
    )
    goal = torch.tensor([[30.0, 31.0, 32.0]])

    result = planner.plan(
        current_state=current,
        plan_size=4,
        repr_input=True,
        final_goal_latents=goal,
    )

    assert result.actions.shape == (1, 1, 2)
    torch.testing.assert_close(result.actions, policy_action.unsqueeze(1))
    assert normalizer.normalized_actions == []
    torch.testing.assert_close(
        RecordingDynamics.instances[0].actions[0], policy_action.unsqueeze(1).permute(1, 0, 2)
    )
    torch.testing.assert_close(policy.calls[0][0], current.obs_component)
    torch.testing.assert_close(policy.calls[0][1], goal)


class RecordingPolicyPlanner:
    is_policy_planner = True

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.reset_calls = []
        self.model = FakeModel()

    def reset_targets(self, targets, repr_input=True):
        self.reset_calls.append((targets, repr_input))

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _planning_result(actions, pred_obs):
    return SimpleNamespace(
        actions=actions,
        pred_obs=pred_obs,
        locations=None,
    )


def test_two_level_planner_replaces_regular_l1_and_l2_with_policy_planners(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    current_state = torch.tensor([[1.0, 2.0, 3.0]])
    final_goal = torch.tensor([[9.0, 8.0, 7.0]])
    l2_pred_obs = torch.stack(
        [
            torch.tensor([[2.0, 2.0, 2.0]]),
            torch.tensor([[4.0, 4.0, 4.0]]),
        ]
    )
    l2_result = _planning_result(
        actions=torch.tensor([[[0.5, 0.75]]]),
        pred_obs=l2_pred_obs,
    )
    l1_result = _planning_result(
        actions=torch.tensor([[[1.0, 1.5], [2.0, 2.5]]]),
        pred_obs=torch.zeros(3, 1, 3),
    )
    l2_planner = RecordingPolicyPlanner(l2_result)
    l1_planner = RecordingPolicyPlanner(l1_result)
    planner = TwoLvlPlanner(l1_planner=l1_planner, l2_planner=l2_planner, l2_step_skip=2)

    planner.reset_targets(final_goal, repr_input=True)
    result = planner.plan(current_state=current_state, plan_size=5)

    assert l2_planner.reset_calls == [(final_goal, True)]
    assert l2_planner.calls[0]["repr_input"] is True
    assert l2_planner.calls[0]["plan_size"] == 5
    torch.testing.assert_close(l2_planner.calls[0]["final_goal_latents"], final_goal)

    assert l1_planner.reset_calls == []
    assert l1_planner.calls[0]["repr_input"] is True
    assert l1_planner.calls[0]["plan_size"] == 2
    torch.testing.assert_close(l1_planner.calls[0]["subgoal_latents"], l2_pred_obs[1])
    torch.testing.assert_close(l1_planner.calls[0]["final_goal_latents"], final_goal)

    torch.testing.assert_close(result.level1.actions, l1_result.actions)
    torch.testing.assert_close(result.level2.actions, l2_result.actions)
    torch.testing.assert_close(result.l1_policy_trace["actions"], l1_result.actions)
    torch.testing.assert_close(
        result.l1_policy_trace["subgoal_latents"], l2_pred_obs[1]
    )
    torch.testing.assert_close(
        result.l2_policy_trace["actions"], l2_result.actions[:, :1]
    )


def test_policy_planners_report_missing_required_latents(monkeypatch):
    _patch_policy_planner_dependencies(monkeypatch)
    model = FakeModel()
    normalizer = FakeNormalizer()
    monkeypatch.setattr(
        l1_module,
        "load_policy_checkpoint",
        lambda path, map_location=None: (
            FakeL1Policy(torch.zeros(1, 2, 2)),
            {"path": path},
        ),
    )
    monkeypatch.setattr(
        l2_module,
        "load_policy_checkpoint",
        lambda path, map_location=None: (
            FakeL2Policy(torch.zeros(1, 2)),
            {"path": path},
        ),
    )

    l1_planner = l1_module.L1PolicyPlanner("l1.ckpt", model=model, normalizer=normalizer)
    l2_planner = l2_module.L2PolicyPlanner("l2.ckpt", model=model, normalizer=normalizer)
    current = FakeBackboneOutput(encodings=torch.zeros(1, 3))

    with pytest.raises(ValueError, match="subgoal_latents and final_goal_latents"):
        l1_planner.plan(current_state=current, plan_size=1, repr_input=True)
    with pytest.raises(ValueError, match="final_goal_latents"):
        l2_planner.plan(current_state=current, plan_size=1, repr_input=True)


def test_l2_policy_uses_goal_conditioned_architecture_and_round_trips(tmp_path):
    config = L2PolicyConfig(
        current_dim=4,
        final_goal_dim=4,
        action_dim=2,
        hidden_dim=16,
        num_layers=2,
        encoder_layers=1,
    )
    policy = L2LatentGoalPolicy(config)
    current = torch.randn(3, 4)
    goal = torch.randn(3, 4)

    pred = policy(current, goal)

    assert pred.shape == (3, 2)
    assert hasattr(policy, "current_encoder")
    assert hasattr(policy, "goal_encoder")
    assert hasattr(policy, "trunk")

    checkpoint_path = tmp_path / "l2_policy.pt"
    save_l2_policy_checkpoint(policy, str(checkpoint_path))
    loaded, payload = load_l2_policy_checkpoint(str(checkpoint_path), map_location="cpu")

    loaded_pred = loaded(current, goal)
    torch.testing.assert_close(loaded_pred, pred)
    assert payload["policy_config"]["use_relative_features"] is True
