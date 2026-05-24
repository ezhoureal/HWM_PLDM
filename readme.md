
<h1 align="center"><em>Hierarchical Planning with Latent World Models</em></h1>

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2604.03208">Paper</a> | 🌐 <a href="https://kevinghst.github.io/HWM/">Website</a>
</p>

<p align="center">
  <a href="https://kevinghst.github.io">Wancong Zhang</a>, <a href="https://scholar.google.com/citations?user=qUB-__0AAAAJ&hl=en">Basile Terver</a>, <a href="https://artemzholus.github.io">Artem Zholus</a>, <a href="https://soham-chitnis10.github.io">Soham Chitnis</a>, <a href="http://harsh-sutariya.github.io/">Harsh Sutaria</a>,<br/>
  <a href="https://www.midoassran.ca">Mido Assran</a>, <a href="https://www.amirbar.net">Amir Bar</a>, <a href="https://randallbalestriero.github.io">Randall Balestriero</a>, <a href="https://scholar.google.com/citations?user=SvRU8F8AAAAJ&hl=en">Adrien Bardes</a>, <a href="https://yann.lecun.org/ex/">Yann LeCun</a>*, <a href="https://scholar.google.com/citations?user=euUV4iUAAAAJ&hl=en">Nicolas Ballas</a>*
</p>


<p align="center">
  <img src="assets/episode_3.gif" alt="Episode 3" />
</p>

<p align="center">
  <img src="assets/episode_17.gif" alt="Episode 17" />
</p>


# Overview

- Implements **Hierarchical Planning with Latent World Models (HWM)**
- Demonstrates **long-horizon planning** in Diverse Maze (PLDM)
- Achieves higher success and lower planning cost vs flat planners

<em>Disclaimer: While HWM is evaluated across multiple world models (VJEPA2, DINO-WM, and PLDM), this repository provides a minimal implementation on PLDM (Diverse Maze). For full results across additional world models and tasks, see the [project page](https://kevinghst.github.io/HWM/) and [paper](https://arxiv.org/pdf/2604.03208).</em>

---

<p>
  Figure 1a: <strong>Hierarchical planning in latent space.</strong> A high-level planner optimizes macro-actions using a long-horizon world model to reach the goal; the first predicted latent state serves as a subgoal for a low-level planner, which optimizes primitive actions with a short-horizon world model. 
</p>
<p align="center">
  <img src="assets/figure_1a.png" alt="Figure 1a" />
</p>


<p>
Figure 1b: Hierarchical planning improves success on non-greedy, long-horizon tasks across multiple latent world models.
</p>
<p align="center">
  <img src="assets/figure_1b.png" alt="Figure 1b" />
</p>



# Remote Setup

The recommended remote dependency stack is Python 3.10, PyTorch 2.7.1, CUDA
12.8, and MuJoCo 2.1.0 for `mujoco-py` / D4RL. Use an NVIDIA driver that
supports CUDA 12.8; a 570-series or newer driver is recommended.

```bash
git clone git@github.com:kevinghst/HWM_PLDM.git
cd HWM_PLDM
export REPO_ROOT="$PWD"

conda create -n pldm python=3.10 -y
conda activate pldm

python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .
```

If PyPI is slow from your remote region, configure a mirror before the `pip`
commands:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## MuJoCo 2.1 for D4RL + mujoco-py

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf

mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

Add the runtime variables to your shell startup file if you will run multiple
jobs:

```bash
cat >> "$HOME/.bashrc" <<'EOF'
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
```

## Sanity Check

```bash
cd "$REPO_ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda runtime", torch.version.cuda)
PY
```

Expected output includes `torch 2.7.1`, `cuda available True`, and CUDA runtime
`12.8`.

## Download Checkpoints and Data

```bash
cd "$REPO_ROOT"

python pldm/download_ckpt_from_hf.py --out-dir "$REPO_ROOT/pldm/pretrained"

python pldm_envs/diverse_maze/data_generation/download_ds_from_hf.py \
  --out-dir "$REPO_ROOT/pldm_envs/diverse_maze/datasets"
```

# Run Experiments

1. Go to `pldm_envs/`, follow instructions to set up dataset for the environment of your choice
2. Go to `pldm/`, follow instruction to run training or evaluation

## Evaluate Pretrained PLDM / HWM

Flat PLDM MPC:

```bash
cd "$REPO_ROOT/pldm"

python train.py \
  --configs configs/diverse_maze/icml/large_diverse_25maps.yaml \
  --values \
    root_path="$REPO_ROOT" \
    wandb=false \
    eval_only=true \
    load_checkpoint_path="$REPO_ROOT/pldm/pretrained/3-9-1-seed248_epoch=3_sample_step=15465472.ckpt"
```

Hierarchical HWM MPC:

```bash
cd "$REPO_ROOT/pldm"

python train.py \
  --configs configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values \
    root_path="$REPO_ROOT" \
    wandb=false \
    eval_only=true \
    load_l1_only=false \
    load_checkpoint_path="$REPO_ROOT/pldm/pretrained/load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt"
```

For a quick remote flat PLDM smoke test, append these overrides to the flat
command:

```bash
quick_debug=true \
eval_cfg.d4rl_planning.level1.mppi.num_samples=16 \
eval_cfg.d4rl_planning.n_envs=1 \
eval_cfg.d4rl_planning.n_steps=20
```

For a quick remote HWM smoke test, append these overrides to the HWM command:

```bash
quick_debug=true \
eval_cfg.h_d4rl_planning.level1.mppi.num_samples=16 \
eval_cfg.h_d4rl_planning.level2.mppi.num_samples=32 \
eval_cfg.h_d4rl_planning.n_envs=1 \
eval_cfg.h_d4rl_planning.n_steps=20
```

Do not use the smoke settings for benchmark numbers.

## Planner-to-Policy Distillation

Collect low-level HWM teacher traces. This runs HWM as usual and writes compact
latent/action shards to `outputs/distill_traces/`.

```bash
cd "$REPO_ROOT/pldm"

python train.py \
  --configs configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values \
    root_path="$REPO_ROOT" \
    wandb=false \
    eval_only=true \
    load_l1_only=false \
    load_checkpoint_path="$REPO_ROOT/pldm/pretrained/load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt" \
    eval_cfg.h_d4rl_planning.distill_trace_dir="$REPO_ROOT/outputs/distill_traces" \
    eval_cfg.h_d4rl_planning.distill_trace_shard_size=20000
```

Train the low-level distilled policy:

```bash
cd "$REPO_ROOT"

python scripts/distill_policy.py \
  --data outputs/distill_traces/hwm_low_level_00000.pt \
  --mode low \
  --epochs 50 \
  --batch-size 1024 \
  --hidden-dim 256 \
  --depth 3 \
  --out outputs/pi_low.pt
```

Train ablation policies from the same trace shard:

```bash
python scripts/distill_policy.py \
  --data outputs/distill_traces/hwm_low_level_00000.pt \
  --mode goal \
  --epochs 50 \
  --batch-size 1024 \
  --hidden-dim 256 \
  --depth 3 \
  --out outputs/pi_goal.pt

python scripts/distill_policy.py \
  --data outputs/distill_traces/hwm_low_level_00000.pt \
  --mode goal_subgoal \
  --epochs 50 \
  --batch-size 1024 \
  --hidden-dim 256 \
  --depth 3 \
  --out outputs/pi_goal_subgoal.pt
```

Evaluate HWM with the learned low-level policy replacing L1 MPC:

```bash
cd "$REPO_ROOT/pldm"

python train.py \
  --configs configs/diverse_maze/icml/large_diverse_25maps_l2.yaml \
  --values \
    root_path="$REPO_ROOT" \
    wandb=false \
    eval_only=true \
    load_l1_only=false \
    load_checkpoint_path="$REPO_ROOT/pldm/pretrained/load_from_l1248-seed248_epoch=5_sample_step=10789632.ckpt" \
    eval_cfg.h_d4rl_planning.distill_policy_path="$REPO_ROOT/outputs/pi_low.pt" \
    eval_cfg.h_d4rl_planning.distill_policy_mode=low
```

To enable the simple fallback gate, add an action-magnitude threshold. Actions
above the threshold use L1 MPC for that step:

```bash
eval_cfg.h_d4rl_planning.distill_policy_max_action_abs=1.25
```

The planning report logs success rate plus planner/policy counters such as
`l2_planner_calls`, `l1_planner_calls`, `l1_planner_env_calls`, `policy_calls`,
and `policy_fallback_l1_calls`.


# Datasets

To see the datasets we used to train our models, see folders inside pldm_envs/. The readmes there will guide you on how to download and set up the datasets
