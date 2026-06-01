# Auto-learned Policy Module Design

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