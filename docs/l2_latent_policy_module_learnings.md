# L2 Latent Policy Next Step

The next policy-improvement direction is DAgger-style relabeling, not a
periodic hybrid of the distilled L2 policy and full L2 MPC. Mixing controllers
during evaluation should not be treated as a supported strategy.

## Current Diagnosis

- The L2 policy can imitate planner trace actions open-loop, but closed-loop
success is lower than the original planner.
- The likely failure mode is covariate shift: once the policy makes a slightly
imperfect macro-action, later states can drift away from the planner-trace
distribution used for supervised training.
- The original planner has online search and recovery at every L2 decision. The
distilled policy emits one deterministic action and has no equivalent recovery
mechanism.

## Replace Hybrid Fallback With DAgger-Style Relabeling

The next experiment should train on the states induced by the policy itself:

1. Run the current L2 policy closed-loop on the target start-goal split using the
fast proprio-only evaluation path.
2. Save policy-induced L2 decision states, including failed or low-progress
episodes.
3. Relabel those visited states with the full L2 planner as the teacher.
4. Append the relabeled examples to the existing planner-trace dataset.
5. Retrain the L2 policy and evaluate pure-policy success against the full
planner baseline.

This directly targets the observed planner-policy gap by teaching the policy
what the planner would do from off-distribution states that the policy actually
visits.

## Training Improvements To Pair With Relabeling

- Include failed and recovery-heavy rollouts instead of filtering only for
successful episodes.
- Add small latent perturbations during training so the policy is robust to
normal closed-loop drift.
- Track policy-induced state coverage separately from planner-trace coverage.
- Report pure-policy success by split: OG medium, OG hard, and OOD/probe maps.

## Removed Direction

The periodic hybrid L2 policy plus full-MPC fallback path is intentionally not
part of the current planning path. If planner assistance is revisited later, it
should be introduced as a separate confidence-gated recovery experiment with
explicit uncertainty or progress signals.
