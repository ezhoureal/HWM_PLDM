## Diverse Mazes

First, download the pretrained world model weights by running 
```
python download_ckpt_from_hf.py --out-dir /workspace/HWM_PLDM/pldm/pretrained
```

Which will download two ckpts:

* PLDM (1 level) ckpt: `3-9-1-seed248_epoch=3_sample_step=15465472.ckpt`
* HWM (2 levels) ckpt: `load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt`

To evaluate hierarchical planning on the downloaded HWM ckpt:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml
```

### Probe-data HWM evaluation caveats

For the OOD probe evaluation, use the `maze2d_large_diverse_probe` dataset paths
in `configs/diverse_maze/icml/large_diverse_25maps_l2.yaml`. The 25-map dataset
is the in-distribution training/eval source; the probe dataset contains the held
out probe maps and the `starts_targets_9_12.pt` / `starts_targets_13_16.pt`
planning splits used by the L2 `medium` and `hard` settings.

Full command used for the HWM L2 probe eval:

```
python train.py \
  --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values \
    output_dir=<xxx> \
    eval_cfg.probing.load_prober_l2=true \
    eval_cfg.probing.visualize_probing=false
```

To evaluate flat planning on the downloaded PLDM ckpt:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml --values load_checkpoint_path=/workspace/HWM_PLDM/pldm/pretrained/3-9-1-seed248_epoch=3_sample_step=15465472.ckpt
```

## Offline Latent Planner-to-Policy Distillation

The policy-compilation path now uses shared latent-policy and trace-collector
modules for both hierarchy levels.

### Fast Proprio-Only Path for L2 Policy Eval on OG (25maps) Data

**Key insight**: For *planning evaluation* (and trace collection) of a trained L2 policy
on the in-distribution og/25maps dataset, you do **not** need any pre-rendered
`images.npy` files.

- The offline datasets (for normalizers and any probing) can be pointed at
  `maze2d_large_diverse_25maps/data.p` with `images_path: null`. Only proprio
  (position + velocity) statistics are needed.
- Live environments created from the `starts_targets_*_trace_*.pt` files render
  images **on the fly** during rollouts when `image_obs=true` (exactly as done
  during the original hierarchical trace collection runs).
- This is a much lighter "fast path" that completely skips the heavy image
  pre-rendering and caching step.

This is how the original `og_trace_*` collection runs were performed.

Example command to run the full L2 policy planning eval on the og (25maps) medium+hard
splits using a trained policy (e.g. the custom combined v2 policy):

```bash
python train.py \
  --config pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values \
    output_dir=l2_og_fast_policy_eval_v2 \
    "data.d4rl_config.path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p" \
    "data.d4rl_config.images_path=null" \
    "eval_cfg.probing.train_path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p" \
    "eval_cfg.probing.train_images_path=null" \
    "eval_cfg.probing.val_path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/data.p" \
    "eval_cfg.probing.val_images_path=null" \
    "eval_cfg.probing.load_prober=false" \
    "eval_cfg.probing.load_prober_l2=false" \
    "eval_cfg.probing.probe_preds=false" \
    "eval_cfg.probing.epochs=0" \
    "eval_cfg.h_d4rl_planning.use_l2_policy=true" \
    "eval_cfg.h_d4rl_planning.l2_policy_checkpoint_path=${root_path}/checkpoint/policies/l2_latent_custom_combined_v2.pt" \
    "eval_cfg.h_d4rl_planning.data_path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps" \
    "eval_cfg.h_d4rl_planning.levels=medium,hard" \
    "eval_cfg.h_d4rl_planning.medium.set_start_target_path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/starts_targets_9_12_trace_n500_seed0.pt" \
    "eval_cfg.h_d4rl_planning.hard.set_start_target_path=${root_path}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/starts_targets_13_16_trace_n300_seed0.pt" \
    "eval_cfg.h_d4rl_planning.image_obs=true"
```

The resulting `planning_l2_mpc_report_l2_d4rl_medium` (and hard) files contain the
success rates, avg/median steps, etc., using the fast reactive L2 policy instead
of expensive L2-level MPC search at every replan step.

This fast path is the natural way to evaluate (and collect traces for) any
L2 policy you distill on the original 25-map diverse maze distribution.

> **Next policy-improvement step**: use the fast path to collect closed-loop
> policy failures, relabel those states with the full L2 planner, and retrain.
> See [docs/l2_latent_policy_module_learnings.md](../docs/l2_latent_policy_module_learnings.md).

### Policy modules L1

The L1 policy distills the expensive sub-trajectory planner used inside
hierarchical MPC. It collects latent supervision of the form:

```
(current_l1_latent, l2_subgoal_latent, final_goal_latent) -> primitive_action_sequence
```

Collect traces from a hierarchical HWM planning eval by setting
`eval_cfg.h_d4rl_planning.l1_policy_trace_path`.

By default, traces are filtered to keep only episodes that eventually reached
the goal, so failed planner rollouts are not used as policy targets.

Train the offline latent L1 policy from that trace file:

```
python train_l1_policy.py \
  --trace_path /workspace/HWM_PLDM/checkpoint/policy_traces/l1_latent_medium.pt \
  --output_path /workspace/HWM_PLDM/checkpoint/policies/l1_latent_policy.pt
```

Then to enable L1 policy, set
`eval_cfg.h_d4rl_planning.use_l1_policy=true` and
`eval_cfg.h_d4rl_planning.l1_policy_checkpoint_path=` points to a trained checkpoint.

### L2 policy

The L2 policy distills the first latent macro-action selected at each
hierarchical replanning step:

```
(current_l2_latent, final_goal_latent) -> next_l2_latent_action
```

Only the first latent action is supervised because hierarchical MPC replans
after each L1 segment, so that is the part of the L2 plan that is actually
consumed online.

Collect traces alongside the same hierarchical eval by setting
`eval_cfg.h_d4rl_planning.l2_policy_trace_path` to a path to save the traces.

Then train the offline latent L2 policy from that trace file:

```bash
python train_l2_policy.py \
  --trace_path /workspace/HWM_PLDM/checkpoint/policy_traces/l2_latent_medium.pt \
  --output_path /workspace/HWM_PLDM/checkpoint/policies/l2_latent_policy.pt \
  --epochs 60 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --val_fraction 0.15 \
  --hidden_dim 256 \
  --num_layers 3 \
  --dropout 0.05 \
  --seed 42
```

## Training

To train the HWM (2 levels) on the large-maze setting by loading the downloaded level 1 PLDM model , run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml
```

If you prefer to train the level-1 PLDM world model from scratch, run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml
```

Then later if you want to train a HWM model by loading the newly trained level 1 WM, update `load_checkpoint_path` in `configs/diverse_maze/icml/large_diverse_25maps_l2.yaml` to point to your trained level-1 checkpoint, and run the level-2 training command above.

## Hyperparameter tuning

Hyperparameters ($\alpha, \beta, \lambda, \delta, \omega$) should be tuned for any new environment.

Within a given environment, hyperparameters should be tuned for different offline datasets that have significant differences in data distributions.

To reduce the hyperparamter search space for a given setting, one idea is to take the hyperparameters for the closest setting, get lower and upper bounds for each parameter by dividing and multiplying it by a factor (eg: 3) respectively, and perform a random search within the lower and upper bounds of all parameters.
