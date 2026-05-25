#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Render Diverse Maze image observations and optionally build images.npy.

Usage:
  scripts/render_diverse_maze_images.sh [options]

Options:
  --data-path PATH       Dataset directory containing data.p and metadata.pt.
                         Defaults to the probe dataset under this repo.
  --repo-root PATH       Repository root. Defaults to this script's parent repo.
  --parallel N          Number of per-episode render jobs to run at once. Default: 6.
  --start N             First episode index to consider. Default: 0.
  --end N               Last episode index to consider, inclusive. Default: all.
  --postprocess         Run postprocess_images.py after rendering completes.
  --log-dir PATH        Render log directory. Default: <repo_root>/.logs.
  --help                Show this help.

Environment:
  MUJOCO_GL             Defaults to osmesa for headless/cloud rendering.
  MUJOCO_DIR            Defaults to $HOME/.mujoco/mujoco210.
  LOCAL_SYSROOT         Optional local sysroot with OSMesa headers/libs.
                         Defaults to <repo_root>/.deps/sysroot if present.
  UV                    uv executable. Default: uv.

The renderer is resumable. It skips an episode once all expected PNG frames
for that episode already exist.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
data_path=""
parallel=6
start_idx=0
end_idx=""
postprocess=false
log_dir=""
uv_bin="${UV:-uv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path)
      data_path="$2"
      shift 2
      ;;
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --parallel)
      parallel="$2"
      shift 2
      ;;
    --start)
      start_idx="$2"
      shift 2
      ;;
    --end)
      end_idx="$2"
      shift 2
      ;;
    --postprocess)
      postprocess=true
      shift
      ;;
    --log-dir)
      log_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$repo_root" && pwd)"
if [[ -z "$data_path" ]]; then
  data_path="$repo_root/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
fi
data_path="$(cd "$data_path" && pwd)"
if [[ -z "$log_dir" ]]; then
  log_dir="$repo_root/.logs"
fi
mkdir -p "$log_dir" "$data_path/images"

read -r total_episodes episode_length < <(
  cd "$repo_root"
  "$uv_bin" run python - "$data_path" <<'PY'
import sys
from pathlib import Path

import torch

metadata = torch.load(Path(sys.argv[1]) / "metadata.pt")
n_episodes = int(metadata["n_episodes"])
train_maps_n = int(metadata.get("train_maps_n", 1))
episode_length = int(metadata["episode_length"])
print(n_episodes * train_maps_n, episode_length)
PY
)

if [[ -z "$end_idx" ]]; then
  end_idx=$((total_episodes - 1))
fi

if (( start_idx < 0 || end_idx < start_idx || end_idx >= total_episodes )); then
  echo "Invalid range: start=$start_idx end=$end_idx total_episodes=$total_episodes" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"

mujoco_dir="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
if [[ -d "$mujoco_dir/bin" ]]; then
  export LD_LIBRARY_PATH="$mujoco_dir/bin:${LD_LIBRARY_PATH:-}"
fi

local_sysroot="${LOCAL_SYSROOT:-$repo_root/.deps/sysroot}"
if [[ -d "$local_sysroot/usr" ]]; then
  export CPATH="$local_sysroot/usr/include:${CPATH:-}"
  export LIBRARY_PATH="$local_sysroot/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$local_sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

export RENDER_REPO_ROOT="$repo_root"
export RENDER_DATA_PATH="$data_path"
export RENDER_LOG_DIR="$log_dir"
export RENDER_EPISODE_LENGTH="$episode_length"
export RENDER_UV_BIN="$uv_bin"

echo "Rendering episodes $start_idx..$end_idx for $data_path"
echo "parallel=$parallel episode_length=$episode_length MUJOCO_GL=$MUJOCO_GL"

seq "$start_idx" "$end_idx" | xargs -n1 -P "$parallel" bash -c '
  set -euo pipefail
  idx="$1"
  expected=$((RENDER_EPISODE_LENGTH + 1))
  existing=$(find "$RENDER_DATA_PATH/images" -maxdepth 1 -type f -name "${idx}_*.png" 2>/dev/null | wc -l)
  if (( existing >= expected )); then
    exit 0
  fi

  cd "$RENDER_REPO_ROOT"
  "$RENDER_UV_BIN" run python pldm_envs/diverse_maze/data_generation/render_data.py \
    --data_path "$RENDER_DATA_PATH" \
    --workers_num 1000 \
    --worker_id "$idx" \
    > "$RENDER_LOG_DIR/render_probe_episode_${idx}.log" 2>&1
' _

rendered=$(find "$data_path/images" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
expected_total=$((total_episodes * (episode_length + 1)))
echo "Rendered PNGs: $rendered / $expected_total"

if [[ "$postprocess" == true ]]; then
  cd "$repo_root"
  "$uv_bin" run python pldm_envs/diverse_maze/data_generation/postprocess_images.py \
    --data_path "$data_path"
fi
