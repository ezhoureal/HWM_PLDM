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

That is a stronger paper than:

> “We distill HWM into a policy.”

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

## What I would implement first

A minimal version could be:

1. Use HWM/PLDM as the teacher planner.
2. Train a subgoal-conditioned policy:

[
\pi_{\text{low}}(z_t, z_{\text{subgoal}}) \rightarrow a_t
]

3. Add a confidence gate:

   * use policy when confident;
   * otherwise call planner.
4. Add DAgger-style relabeling:

   * when policy fails or visits novel states, query planner.
5. Measure:

   * success rate,
   * planning calls per episode,
   * latency,
   * generalization to unseen mazes/tasks,
   * whether policy gradually replaces planner.

The cleanest empirical curve would be:

> planning calls decrease over training while success rate remains high.

That would demonstrate automatic compilation from deliberate planning into reactive skill.

## The punchline

Yes — I think this is a genuinely promising direction.

The crisp name could be something like:

**Hierarchical Planner-to-Policy Consolidation**

or more cognitively:

**From Deliberation to Habit: Amortizing Hierarchical World-Model Planning into Reactive Policies**

The key insight is:

> A world-model planner should not merely choose actions. It should produce training data for its own future intuition.

That is very LeCun-compatible, biologically inspired, and technically implementable.

[1]: https://arxiv.org/abs/2307.12933?utm_source=chatgpt.com "Theoretically Guaranteed Policy Improvement Distilled from Model-Based Planning"
[2]: https://aclanthology.org/P18-1203/?utm_source=chatgpt.com "Deep Dyna-Q: Integrating Planning for Task-Completion ..."
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4526597/?utm_source=chatgpt.com "Model-based learning protects against forming habits - PMC"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9004662/?utm_source=chatgpt.com "Learning Structures: Predictive Representations, Replay, and ..."
