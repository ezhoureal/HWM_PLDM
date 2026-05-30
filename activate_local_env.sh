#!/usr/bin/env bash
set -e

export VENV=/tmp/hwm_venv
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/hf-cache
export PIP_CACHE_DIR=/tmp/pip-cache

if [ ! -d "$VENV" ]; then
  uv venv "$VENV" --python 3.9 --system-site-packages
fi

source "$VENV/bin/activate"

export D4RL_SUPPRESS_IMPORT_ERROR=1 \
  PYTHONFAULTHANDLER=1 \
  CUDA_LAUNCH_BLOCKING=1 \
  GPUS=1 \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 \
  LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64