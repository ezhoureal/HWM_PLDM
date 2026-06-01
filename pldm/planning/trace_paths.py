"""Utilities for scoping trace and policy checkpoint paths by difficulty level."""

from pathlib import Path
from typing import Optional


def level_scoped_path(path: Optional[str], level: str) -> Optional[str]:
    """Append or substitute the difficulty level into a trace/policy path.

    - If path is None, returns None (no tracing for this level).
    - If the filename contains the literal "{level}", performs .format(level=level).
    - Otherwise appends "_{level}" to the stem (before any suffix).
    - Preserves the original directory and suffix (even if missing).

    Examples:
        level_scoped_path("/tmp/traces/l2.pt", "medium") -> "/tmp/traces/l2_medium.pt"
        level_scoped_path("/tmp/traces/l2_{level}.pt", "hard") -> "/tmp/traces/l2_hard.pt"
        level_scoped_path("/tmp/traces/l2_latent", "easy") -> "/tmp/traces/l2_latent_easy"
    """
    if path is None:
        return None
    p = Path(path)
    name = p.name
    if "{level}" in name:
        new_name = name.format(level=level)
        return str(p.with_name(new_name))
    stem = p.stem
    suffix = p.suffix or ""
    if not stem.endswith(f"_{level}") and not stem.endswith(level):
        stem = f"{stem}_{level}"
    return str(p.with_name(f"{stem}{suffix}"))
