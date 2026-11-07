"""Device selection, explicit RNG streams and environment capture.

No module-level mutable state: every random draw takes a generator built
from an explicit seed, and the device is chosen by a function call.
"""

import functools
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import lucid
import numpy as np

from affdrift._typing import Device

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def default_device() -> Device:
    """``"metal"`` when the Metal backend answers, else ``"cpu"``.

    Lucid only knows these two names; the CLAUDE.md order cuda > mps > cpu
    maps onto metal > cpu (docs/deviations.md D1).
    """
    try:
        lucid.ones((1,), device="metal").sum().item()
    except Exception:  # noqa: BLE001 — any backend failure means no Metal
        return "cpu"
    return "metal"


def numpy_rng(seed: int, *stream: int) -> np.random.Generator:
    """A NumPy generator for ``seed`` and an optional sub-stream path.

    Distinct ``stream`` tuples give independent streams of the same seed, so
    e.g. particle init and data generation never share draws.
    """
    return np.random.default_rng([seed, *stream])


def to_tensor(array: np.ndarray, device: Device, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    """Copy a NumPy array onto ``device`` as ``dtype`` (float32 by default)."""
    np_dtype = np.float64 if dtype == lucid.float64 else np.float32
    return lucid.tensor(np.ascontiguousarray(array, dtype=np_dtype), dtype=dtype, device=device)


def to_numpy(tensor: lucid.Tensor) -> np.ndarray:
    """Host copy of a tensor as float64."""
    return np.asarray(tensor.detach().numpy(), dtype=np.float64)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_is_dirty(*paths: str) -> bool:
    return bool(_git("status", "--porcelain", "--", *paths))


def environment() -> dict[str, str]:
    """Everything needed to say where and with what a run was produced."""
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": str(git_is_dirty()),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "lucid": lucid.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": default_device(),
    }
