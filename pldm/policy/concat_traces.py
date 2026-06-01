#!/usr/bin/env python
"""
Concatenate multiple latent policy trace files (L1 or L2) into a single combined dataset.

Usage examples:
  # Explicit files
  python -m pldm.policy.concat_traces --level l1 \
    --output checkpoint/policy_traces/l1_latent_combined.pt \
    --input checkpoint/policy_traces/l1_latent_og.pt \
    --input checkpoint/policy_traces/l1_latent_medium.pt \
    --input checkpoint/policy_traces/l1_latent_25maps_9_12_n500_seed0.pt

  # Auto-discover in a directory (recommended)
  python -m pldm.policy.concat_traces --level l1 \
    --output checkpoint/policy_traces/l1_latent_combined.pt \
    --dir checkpoint/policy_traces --pattern "l1_latent_*.pt"

  # L2 example
  python -m pldm.policy.concat_traces --level l2 \
    --output checkpoint/policy_traces/l2_latent_combined.pt \
    --dir checkpoint/policy_traces --pattern "l2_latent_*.pt"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import torch

from pldm.policy.common import _atomic_torch_save

L1_KIND = "l1_latent_subgoal_policy_trace"
L2_KIND = "l2_latent_goal_policy_trace"

L1_INPUT_KEYS = ("current_latents", "subgoal_latents", "final_goal_latents")
L2_INPUT_KEYS = ("current_latents", "final_goal_latents")


def _load_trace(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict trace")
    if "actions" not in data or "metadata" not in data:
        raise ValueError(f"{path} does not look like a flattened policy trace")
    return data


def _get_kind(data: dict) -> str:
    return data.get("kind", "")


def concat_policy_traces(
    input_files: list[Path],
    output_file: Path,
    *,
    level: Optional[str] = None,
    source_tag: str = "multi-run-concat",
    verbose: bool = True,
) -> dict:
    """Concatenate several trace .pt files (already in flattened form) into one.

    All input files must be of the same level (L1 or L2). The 'kind' field is used
    to validate consistency.
    """
    if not input_files:
        raise ValueError("No input files provided for concatenation")

    input_files = [Path(p).expanduser().resolve() for p in input_files]
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    expected_kind = None
    if level == "l1":
        expected_kind = L1_KIND
    elif level == "l2":
        expected_kind = L2_KIND

    all_data = []
    sources = []
    total_dropped = 0
    total_before_sampling = 0

    for p in input_files:
        if not p.exists():
            raise FileNotFoundError(f"Trace file not found: {p}")

        data = _load_trace(p)
        kind = _get_kind(data)

        if expected_kind is None:
            if kind == L1_KIND:
                expected_kind = L1_KIND
            elif kind == L2_KIND:
                expected_kind = L2_KIND
            else:
                raise ValueError(f"Unknown trace kind in {p}: {kind}")

        if kind != expected_kind:
            raise ValueError(
                f"Kind mismatch: expected {expected_kind} but {p} has kind={kind}"
            )

        all_data.append(data)
        sources.append(str(p))

        md = data.get("metadata", {})
        total_dropped += int(md.get("num_dropped_failed_examples", 0))
        total_before_sampling += int(md.get("num_examples_before_sampling", data["actions"].shape[0]))

        if verbose:
            n = data["actions"].shape[0]
            print(f"  Loaded {p.name:50s}  {n:6d} examples")

    # Determine tensor keys to concatenate (from the first file's input_keys + actions)
    first = all_data[0]
    input_keys = tuple(first.get("input_keys", []))
    if not input_keys:
        # Fallbacks
        if expected_kind == L1_KIND:
            input_keys = L1_INPUT_KEYS
        else:
            input_keys = L2_INPUT_KEYS

    # Concatenate tensors
    concatenated = {}
    for key in input_keys:
        if key not in first:
            raise KeyError(f"Missing key {key} in first trace file")
        tensors = [d[key] for d in all_data]
        concatenated[key] = torch.cat(tensors, dim=0)

    actions = torch.cat([d["actions"] for d in all_data], dim=0)

    # Build merged metadata
    num_examples = int(actions.shape[0])
    merged_metadata = {
        "num_examples": num_examples,
        "num_examples_before_sampling": total_before_sampling,
        "num_dropped_failed_examples": total_dropped,
        "success_only": first.get("metadata", {}).get("success_only", True),
        "success_filter": first.get("metadata", {}).get("success_filter", "unknown"),
        "action_shape": tuple(actions.shape[1:]),
        "max_samples": None,  # concatenation already happened
        "num_sources": len(all_data),
        "sources": sources,
        "concat_source_tag": source_tag,
    }

    # Copy shape metadata from first file (they must be compatible)
    for key in input_keys:
        shape_key = f"{key}_shape"
        if shape_key in first.get("metadata", {}):
            merged_metadata[shape_key] = first["metadata"][shape_key]

    combined = {
        "version": 2,
        "kind": expected_kind,
        "input_keys": list(input_keys),
        **{k: v.contiguous() for k, v in concatenated.items()},
        "actions": actions.contiguous(),
        "metadata": merged_metadata,
    }

    _atomic_torch_save(combined, output_file)

    if verbose:
        print(f"\nConcatenated {len(all_data)} files → {num_examples} total examples")
        print(f"Saved combined {expected_kind} trace to {output_file}")

    return combined


def _expand_inputs(dir_path: Optional[str], pattern: Optional[str], inputs: list[str]) -> list[Path]:
    files: list[Path] = []

    if inputs:
        for item in inputs:
            p = Path(item)
            if p.is_dir():
                # treat as dir with default patterns
                files.extend(sorted(p.glob("l1_latent_*.pt")))
                files.extend(sorted(p.glob("l2_latent_*.pt")))
            else:
                files.extend(Path(x).expanduser().resolve() for x in glob.glob(str(item)))

    if dir_path:
        d = Path(dir_path).expanduser().resolve()
        if not d.is_dir():
            raise NotADirectoryError(f"--dir must be a directory: {d}")
        pat = pattern or "*latent_*.pt"
        matches = sorted(d.glob(pat))
        # Filter out partials by default unless the pattern explicitly asks for them
        if "partial" not in pat.lower():
            matches = [m for m in matches if ".partial" not in m.name]
        files.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Concatenate multiple L1 or L2 latent policy trace files into one dataset."
    )
    p.add_argument(
        "--level",
        choices=["l1", "l2"],
        help="Which policy level (auto-detected from files if omitted)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the combined .pt file (e.g. l1_latent_combined.pt)",
    )
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input trace file or glob (can be repeated). Also accepts directories.",
    )
    p.add_argument(
        "--dir",
        help="Directory to scan for trace files (commonly checkpoint/policy_traces)",
    )
    p.add_argument(
        "--pattern",
        help="Glob pattern inside --dir (default: *latent_*.pt, auto-excludes .partial files)",
    )
    p.add_argument(
        "--tag",
        default="multi-run-concat",
        help="Tag to record in metadata['concat_source_tag']",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the files that would be concatenated, do not write output",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    input_files = _expand_inputs(args.dir, args.pattern, args.input)

    if not input_files:
        print("No trace files found. Use --input, --dir + --pattern, or both.")
        return 1

    print(f"Found {len(input_files)} candidate trace file(s):")
    for f in input_files:
        print(f"  {f}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    concat_policy_traces(
        input_files,
        args.output,
        level=args.level,
        source_tag=args.tag,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
