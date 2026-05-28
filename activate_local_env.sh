#!/usr/bin/env bash
set -e

export VENV=/tmp/hwm_venv
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/hf-cache
export PIP_CACHE_DIR=/tmp/pip-cache

if [ ! -d "$VENV" ]; then
  python -m venv "$VENV" --system-site-packages
fi

source "$VENV/bin/activate"

