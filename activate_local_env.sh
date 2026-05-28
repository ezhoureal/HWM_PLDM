#!/usr/bin/env bash
set -e

export VENV=/tmp/hwm_venv
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/hf-cache
export PIP_CACHE_DIR=/tmp/pip-cache

if [ ! -d "$VENV" ]; then
  uv venv "$VENV" --python 3.9 --system-site-packages
  uv pip install -r requirements.txt
  uv pip install -e .
fi

source "$VENV/bin/activate"

