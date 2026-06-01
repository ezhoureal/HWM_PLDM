"""
Generate a dictionary with keys values:
- starts: lists of (4,) numpy arrays
- targets: lists of (4,) numpy arrays
- map_layouts: list of strings of example '######\\##O###\\#OOOO#\\#OO#O#\\#OO#O#\\######'
- block_dists: list of integers
"""

import argparse
from pathlib import Path

import torch
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate starts/targets trial files for diverse maze evaluation"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to dataset folder (containing train_maps.pt + data.p + metadata.pt)",
    )
    parser.add_argument("--n_envs", type=int, default=40, help="trials per band (default 40 matches paper evals)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_block", type=int, default=None, help="if set, generate only this band")
    parser.add_argument("--max_block", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="explicit output .pt filename (else auto: starts_targets_MIN_MAX.pt or _trace_nN_seedS for non-defaults)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    env_name = "maze2d_large_diverse"
    n = args.n_envs
    seed = args.seed

    block_radii = [(5, 8), (9, 12), (13, 16)]
    if args.min_block is not None:
        maxb = args.max_block if args.max_block is not None else args.min_block + 3
        block_radii = [(args.min_block, maxb)]
    elif args.output:
        raise ValueError("--output is only valid when generating a single band with --min_block")

    from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import (
        Maze2DEnvsGenerator,
    )

    for min_r, max_r in block_radii:
        config = {
            "env_name": env_name,
            "n_envs": n,
            "min_block_radius": min_r,
            "max_block_radius": max_r,
            "action_repeat": 4,
            "action_repeat_mode": "id",
            "stack_states": 1,
            "image_obs": True,
            "data_path": str(data_path),
            "set_start_target_path": None,
            "unique_shortest_path": False,
            "seed": seed,
        }

        from types import SimpleNamespace

        cfg = SimpleNamespace(**config)

        envs_generator = Maze2DEnvsGenerator(
            env_name=cfg.env_name,
            n_envs=cfg.n_envs,
            min_block_radius=cfg.min_block_radius,
            max_block_radius=cfg.max_block_radius,
            action_repeat=cfg.action_repeat,
            action_repeat_mode=cfg.action_repeat_mode,
            seed=cfg.seed,
            stack_states=cfg.stack_states,
            image_obs=cfg.image_obs,
            data_path=cfg.data_path,
            trials_path=cfg.set_start_target_path,
            unique_shortest_path=cfg.unique_shortest_path,
            normalizer=None,
            build_envs=False,
        )

        envs, trials = envs_generator()

        if args.output:
            out_name = args.output
        else:
            suffix = f"_trace_n{n}_seed{seed}" if (n != 40 or seed != 42) else ""
            out_name = f"starts_targets_{min_r}_{max_r}{suffix}.pt"
        torch.save(trials, data_path / out_name)
        print(f"wrote {data_path / out_name} with {len(trials.get('starts', []))} trials (seed={seed})")

if __name__ == "__main__":
    main()
