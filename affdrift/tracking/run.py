"""One run = one (experiment, arm, seed), in its own directory.

``experiments/<Level>/<ID>/runs/<arm>/seed<k>/`` (``runs_smoke/`` for smoke runs)::

    config.yaml      resolved config (B0 ⊕ Level ⊕ common ⊕ arm)
    config_hash.txt  12-hex SHA-256 of the resolved config
    delta.json       Δ against B0 and against the reference arm
    env.json         git commit/dirty, versions, device, time
    status.json      running | complete | failed | invalid (+ reason)
    log.jsonl        every logged step
    metrics.json     final metrics
    arrays.npz       raw arrays the figures are drawn from (git-ignored)
    figs/            per-run figures

A run directory is created only after the Δ check passes, so a refused
config leaves no trace but the error.
"""

import json
import shutil
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import yaml

from affdrift.config import Delta, Experiment, config_hash
from affdrift.runtime import environment
from affdrift.tracking.logger import MultiSink, Sink


class RunState(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    #: Finished, but excluded from aggregation (e.g. Gate G2 scale check).
    INVALID = "invalid"


def run_dir(exp: Experiment, arm: str, seed: int, *, smoke: bool = False) -> Path:
    return exp.dir / ("runs_smoke" if smoke else "runs") / arm / f"seed{seed}"


def read_status(path: Path) -> dict[str, Any] | None:
    status = path / "status.json"
    return json.loads(status.read_text()) if status.exists() else None


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_default) + "\n")


def _default(obj: Any) -> Any:
    match obj:
        case np.generic():
            return obj.item()
        case np.ndarray():
            return obj.tolist()
        case Path():
            return str(obj)
        case _:
            raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


@dataclass
class Run:
    """Context manager for one run.  Use :meth:`finish` to record metrics."""

    exp: Experiment
    arm: str
    seed: int
    smoke: bool = False
    force: bool = False
    use_wandb: bool = True
    config: dict[str, Any] = field(init=False)
    delta: Delta = field(init=False)
    path: Path = field(init=False)
    _sink: Sink | None = field(init=False, default=None)
    _finished: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.delta = self.exp.check_delta(self.arm)  # refuses before anything is written
        self.config = self.exp.resolve(self.arm)
        self.path = run_dir(self.exp, self.arm, self.seed, smoke=self.smoke)

    @property
    def name(self) -> str:
        return f"{self.exp.id}/{self.arm}/seed{self.seed}" + ("/smoke" if self.smoke else "")

    def is_complete(self) -> bool:
        return (s := read_status(self.path)) is not None and s["state"] in {RunState.COMPLETE, RunState.INVALID}

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        if self.path.exists():
            if self.is_complete() and not self.force:
                raise FileExistsError(f"{self.name} is complete; pass force=True to redo it")
            shutil.rmtree(self.path)
        (self.path / "figs").mkdir(parents=True)
        (self.path / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        (self.path / "config_hash.txt").write_text(config_hash(self.config) + "\n")
        _write_json(self.path / "delta.json", self.delta.to_json())
        _write_json(self.path / "env.json", environment() | {"seed": self.seed, "smoke": self.smoke})
        self._set_state(RunState.RUNNING)
        wandb_kwargs = (
            {
                "name": self.name,
                "group": self.exp.id,
                "job_type": self.arm,
                "tags": [self.exp.level, self.exp.id, *(["smoke"] if self.smoke else [])],
                "config": self.config | {"seed": self.seed, "config_hash": config_hash(self.config)},
            }
            if self.use_wandb
            else None
        )
        self._sink = MultiSink.for_run(self.path / "log.jsonl", wandb_kwargs=wandb_kwargs)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        failed = exc is not None
        if failed:
            self._set_state(RunState.FAILED, reason="".join(traceback.format_exception(exc)))
        elif not self._finished:
            self._set_state(RunState.FAILED, reason="run exited without finish()")
        if self._sink is not None:
            self._sink.close(failed=failed or not self._finished)

    # ── recording ────────────────────────────────────────────────────────

    def log(self, step: int, **scalars: float) -> None:
        assert self._sink is not None, "log() outside the run context"
        self._sink.log(step, scalars)

    def save_arrays(self, **arrays: np.ndarray) -> None:
        np.savez_compressed(self.path / "arrays.npz", **arrays)

    def finish(self, metrics: Mapping[str, Any], *, valid: bool = True, reason: str = "") -> None:
        _write_json(self.path / "metrics.json", dict(metrics))
        assert self._sink is not None
        self._sink.summary(metrics)
        self._set_state(RunState.COMPLETE if valid else RunState.INVALID, reason=reason)
        self._finished = True

    def _set_state(self, state: RunState, *, reason: str = "") -> None:
        _write_json(self.path / "status.json", {"state": state.value, "reason": reason, **environment()})
