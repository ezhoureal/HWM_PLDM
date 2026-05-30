# This file downloads the original datasets and render them

# Root path of the project. CHANGE TO YOUR OWN.
PROJECT_ROOT=/workspace/HWM_PLDM
RENDER_WORKERS=${RENDER_WORKERS:-1}
POSTPROCESS_WORKERS=${POSTPROCESS_WORKERS:-4}

run_render_workers() {
    local data_path="$1"
    local workers="$2"
    local worker_id
    local pids=()

    for ((worker_id = 0; worker_id < workers; worker_id++)); do
        python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py"             --data_path "${data_path}"             --workers_num "${workers}"             --worker_id "${worker_id}" &
        pids+=($!)
    done

    local status=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done

    return "${status}"
}

# Download datasets from HF into pldm_envs/diverse_maze/datasets.
python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/download_ds_from_hf.py" \
    --out-dir "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets"

DATA_PATHS=(
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)

# render the datasets. save images as numpy
set -e

for DATA_PATH in "${DATA_PATHS[@]}"; do
    run_render_workers "$DATA_PATH" "$RENDER_WORKERS"
    python "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py"         --data_path "$DATA_PATH"         --num_workers "$POSTPROCESS_WORKERS"
done
