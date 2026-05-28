Yes — that is a much bigger and more interesting framing.

I would describe it as:

> **automatic cognitive compilation**: the agent first solves unfamiliar situations through slow model-based planning, then consolidates successful solutions into a fast reactive policy so similar situations no longer require conscious planning.

This is strongly inspired by human learning: we initially solve tasks deliberately, then after enough practice they become automatic habits or skills.

## Is this already done?

**Partially, but not in the full form you are describing.**

Existing work covers pieces of it:

1. **Model-based planning distilled to policy** already exists. For example, MPDP explicitly studies distilling optimized model-based planning behavior into a policy to reduce online planning cost. ([arXiv][1])

2. **MPC imitation learning** already exists. Papers in this area treat MPC as the expert and train a policy to imitate the planner, often with on-policy correction to reduce distribution shift. ([arXiv][1])

3. **Dyna-style learning** already uses a world model to generate imagined experience for training a policy. Deep Dyna-Q, for instance, integrates a world model into deep RL by generating simulated experience for policy learning. ([ACL Anthology][2])

4. **Neuroscience/RL already has the goal-directed vs habitual distinction.** Model-based control is often associated with flexible, goal-directed behavior, while model-free control is often associated with habit-like behavior, though the neuroscience literature is more nuanced than the simple mapping. ([PMC][3])

5. **Replay/consolidation is an established bridge.** Cognitive and RL literature studies replay as a mechanism for learning compact predictive representations and converting experience into longer-lasting structure. ([PMC][4])

But the **full autonomous loop** you are describing is not, as far as I can tell, a solved standard framework:

> detect when planning is needed → solve with hierarchical world-model planning → store useful plan traces → distill them into a reactive network → later bypass planning when confidence is high → fall back to planning when reactive policy is uncertain or fails.

That complete loop, especially on top of **hierarchical PLDM/HWM-style latent world models**, still looks like a real research opportunity.

## The core mechanism

I would formulate it as two systems:

### Mode-2: deliberative planner

This is the current hierarchical PLDM/HWM mechanism:

[
(z_t, z_g) \xrightarrow{\text{hierarchical planning}} z_{\text{subgoal}}, a_{t:t+H}
]

It samples or optimizes trajectories in latent space.

### Mode-1: compiled reactive policy

This is the learned habitual controller:

[
\pi_\theta(z_t, z_g, z_{\text{subgoal}}) \rightarrow a_t
]

or simpler:

[
\pi_\theta(z_t, z_{\text{subgoal}}) \rightarrow a_t
]

After enough planner-generated examples, the policy should act directly without online search.

The key is that the reactive policy is **not trained from human demonstrations**. It is trained from the agent’s own successful thinking/trying.

## What makes your version more novel

The novelty is not “distill a planner.” That exists.

The novelty is the **automatic life cycle**:

| Phase                          | Human analogy                       | Machine mechanism                |
| ------------------------------ | ----------------------------------- | -------------------------------- |
| Encounter unfamiliar situation | conscious thought                   | invoke world-model planner       |
| Try candidate solutions        | mental simulation / trial-and-error | latent MPC / PLDM rollouts       |
| Successful resolution          | reward / goal achievement           | store trajectory + latent states |
| Practice/replay                | consolidation                       | train policy from planner traces |
| Automatic execution            | habit / intuition                   | reactive policy forward pass     |
| Surprise/failure               | conscious re-engagement             | fall back to planner             |

That is closer to a cognitive architecture than a single algorithm.

## A concrete framework

I would build it as:

### 1. Planner-as-teacher

When the policy is uncertain or unsuccessful, invoke hierarchical PLDM:

[
\tau^* = {z_t, z_g, z_{\text{subgoal}}, a_t, C_t}_{t=1}^{T}
]

Store the selected action, predicted subgoals, planner costs, and whether the episode succeeded.

### 2. Consolidation buffer

Maintain a memory of successful and informative traces:

[
\mathcal{D}*{\text{consolidate}} =
{(z_t, z_g, z*{\text{subgoal}}, a_t^*, w_t)}
]

where (w_t) could depend on:

* planner confidence,
* trajectory success,
* cost margin over alternatives,
* novelty of state,
* repeated usefulness.

### 3. Reactive policy training

Train a student policy:

[
\mathcal{L}*{\text{BC}} =
-\log \pi*\theta(a_t^* \mid z_t, z_g, z_{\text{subgoal}})
]

For continuous actions:

[
\mathcal{L} =
|\pi_\theta(z_t,z_g,z_{\text{subgoal}}) - a_t^*|^2
]

You can also train from soft planner scores:

[
p(a) \propto \exp(-C(a)/\tau)
]

This is better than hard imitation because it preserves the planner’s uncertainty.

### 4. Gating between Mode-1 and Mode-2

This is critical.

At runtime, the system should not always trust the reactive policy. It should have a gate:

[
\text{use Mode-1 if } U_\theta(z_t,z_g) < \epsilon
]

where uncertainty (U) can come from:

* policy entropy,
* ensemble disagreement,
* predicted goal-reaching error,
* low value estimate,
* novelty in latent state,
* mismatch between predicted and observed next latent.

Otherwise, call Mode-2.

This gives you:

> plan when unfamiliar, react when familiar.

### 5. Continual consolidation

After each episode:

* successful planned traces train the policy;
* failed reactive traces are relabeled by planner;
* old skills are replayed to avoid forgetting.

This becomes a lifelong skill-learning loop.

## The strongest research question

I would frame the central hypothesis as:

> **Can hierarchical latent planning automatically compile itself into reusable reactive skills, while preserving the ability to fall back to planning on novel states?**

Because it introduces:

* automatic planner invocation,
* skill consolidation,
* reactive/planning arbitration,
* lifelong improvement.

## Relation to existing work

The nearest ancestors are:

| Existing area               | What it covers                           | What it misses                                                 |
| --------------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| MPDP / planner distillation | planning → policy                        | usually not hierarchical cognitive consolidation               |
| MPC imitation               | controller compression                   | often task-specific, not autonomous skill acquisition          |
| Dyna                        | world model generates training data      | not necessarily planner-to-habit compilation                   |
| Dreamer                     | imagined rollouts train actor            | less explicit Mode-2-to-Mode-1 distillation                    |
| habit RL literature         | model-based vs habitual control          | usually cognitive modeling, not PLDM/HWM implementation        |
| replay/consolidation        | experience becomes stable representation | not necessarily action-policy compilation from latent planning |

So your idea sits at the intersection of these, but the **HWM/PLDM + automatic Mode-1 consolidation** angle still seems underexplored.

## Implementation status in this repo

We have now implemented a first working slice of this idea inside the PLDM/HWM
codebase.

### Implemented

1. **Shared latent policy module**

   We refactored a generic latent-policy stack that is shared by both hierarchy
   levels instead of keeping separate ad hoc codepaths.

   Current pieces:

   * shared latent MLP policy core,
   * shared latent trace dataset loader,
   * shared trace flatten/save utilities,
   * shared training entrypoint with level-specific wrappers.

2. **L1 policy distillation**

   The L1 planner can already be distilled offline from hierarchical MPC traces.

   Current supervision:

   [
   (z_t^{l1}, z_{\text{subgoal}}^{l2}, z_g^{l1}) \rightarrow a_{t:t+H-1}
   ]

   That policy is trained by behavior cloning and can already be used at
   inference time through a dedicated `L1PolicyPlanner` that replaces the online
   L1 planner.

3. **L2 policy distillation**

   We now also collect and train an L2 policy.

   Importantly, the current design predicts **only the first L2 latent action**,
   not an entire latent-action sequence. This matches the hierarchical MPC loop
   more closely, because only the first L2 action is consumed before replanning.

   Current supervision:

   [
   (z_t^{l2}, z_g^{l2}) \rightarrow u_t^{l2}
   ]

   where (u_t^{l2}) is the first latent macro-action chosen by the L2 planner.

4. **Inference-side L2 policy planner**

   We added an `L2PolicyPlanner` for inference. It:

   * loads a trained L2 policy checkpoint,
   * predicts one latent macro-action,
   * rolls that single latent action through the L2 world model,
   * returns the predicted first subgoal latent for the downstream L1 policy or
     planner.

5. **Latent trace collection**

   Hierarchical evaluation can now save both:

   * L1 latent-policy traces,
   * L2 latent-policy traces.

   The L1 trace path is filtered by **subgoal success** for each L1 segment.
   The L2 trace path is filtered by **episode success**.

### Partially implemented

1. **Planner-to-policy compilation**

   We do have planner trace collection, offline policy training, and inference
   substitution for both L1 and L2.

   So the basic pipeline

   > plan -> store traces -> train policy -> reuse policy at inference

   is now implemented.

2. **Automatic cognitive compilation**

   The broader autonomous loop is still only partially implemented.

   What exists:

   * hierarchical planner as teacher,
   * offline consolidation into L1/L2 policies,
   * optional policy use at inference.

   What does **not** yet exist:

   * automatic uncertainty-based arbitration between policy and planner,
   * automatic fallback from policy to planner on novel states,
   * continual online relabeling / DAgger-style correction,
   * replay scheduling based on novelty, confidence, or usefulness,
   * long-term skill memory management.

### Current limitations / observations

1. **L1 speedup is limited so far**

   We already implemented the L1 policy according to the design, but the
   wall-clock speedup effect has not been very strong yet. That is one reason we
   moved on to L2 compilation, since replacing high-level latent planning has a
   better chance of reducing total planning cost.

2. **L2 is currently single-step reactive, not full-horizon**

   This is intentional for now. Predicting only the first L2 action keeps the
   training target aligned with actual MPC usage, but it also means we are not
   yet compiling an entire high-level plan into one policy forward pass.

3. **No policy gate yet**

   At the moment, whether to use the policy is configured manually
   (`use_l1_policy`, `use_l2_policy`) rather than decided by uncertainty or
   novelty.

### Immediate next steps

The most important next experiments are:

1. evaluate `use_l2_policy=true` with `use_l1_policy=false` to isolate whether
   L2 compilation gives meaningful speedup;
2. compare success rate vs planning latency for:
   * full hierarchical MPC,
   * L1 policy only,
   * L2 policy only,
   * both L1 and L2 policies;
3. add a gating mechanism so the agent can choose between reactive execution
   and planning instead of relying on a fixed config switch.
