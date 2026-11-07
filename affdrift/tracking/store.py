"""Read-only access to finished runs, for aggregation and figures."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from affdrift.config import Experiment
from affdrift.tracking.run import RunState, read_status, run_dir


@dataclass(frozen=True)
class RunRecord:
    exp: Experiment
    arm: str
    seed: int
    path: Path

    @cached_property
    def status(self) -> dict[str, Any]:
        return read_status(self.path) or {"state": "missing"}

    @property
    def state(self) -> str:
        return self.status["state"]

    @cached_property
    def metrics(self) -> dict[str, Any]:
        return json.loads((self.path / "metrics.json").read_text())

    @cached_property
    def config(self) -> dict[str, Any]:
        return yaml.safe_load((self.path / "config.yaml").read_text())

    @cached_property
    def config_hash(self) -> str:
        return (self.path / "config_hash.txt").read_text().strip()

    @cached_property
    def arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.path / "arrays.npz") as data:
            return {k: data[k] for k in data.files}

    @cached_property
    def log(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.path / "log.jsonl").read_text().splitlines() if line]

    @property
    def name(self) -> str:
        return f"{self.exp.id}/{self.arm}/seed{self.seed}"


def records(exp: Experiment, *, smoke: bool = False) -> dict[str, dict[int, RunRecord]]:
    """Every declared (arm, seed) of the experiment, present or not."""
    seeds = exp.seeds[:1] if smoke else exp.seeds
    return {
        arm: {s: RunRecord(exp, arm, s, run_dir(exp, arm, s, smoke=smoke)) for s in seeds}
        for arm in exp.arms
    }


def complete(exp: Experiment, *, smoke: bool = False) -> dict[str, dict[int, RunRecord]]:
    """Only runs whose state is ``complete`` (``invalid`` runs are excluded, CLAUDE.md §2.5)."""
    return {
        arm: done
        for arm, runs in records(exp, smoke=smoke).items()
        if (done := {s: r for s, r in runs.items() if r.state == RunState.COMPLETE})
    }
