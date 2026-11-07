"""Metric sinks.  ``log.jsonl`` is the record; wandb is a mirror.

wandb runs ``offline`` by default (``AFFD_WANDB`` = ``offline`` | ``online`` |
``disabled``): nothing leaves the machine until ``wandb sync`` is run by hand.
A wandb failure never fails a run — the JSONL file is the source of truth.
"""

import json
import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self, override

from affdrift.runtime import ROOT

type WandbMode = Literal["offline", "online", "disabled"]
WANDB_PROJECT = "affinity-drifting"


def wandb_mode() -> WandbMode:
    match mode := os.environ.get("AFFD_WANDB", "offline"):
        case "offline" | "online" | "disabled":
            return mode
        case _:
            raise ValueError(f"AFFD_WANDB must be offline, online or disabled, got {mode!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Sink(Protocol):
    def log(self, step: int, scalars: Mapping[str, float]) -> None: ...
    def summary(self, metrics: Mapping[str, Any]) -> None: ...
    def close(self, *, failed: bool = False) -> None: ...


class JsonlSink:
    """One JSON object per line: ``{"step": t, <name>: value, ...}``."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("a", buffering=1)

    def log(self, step: int, scalars: Mapping[str, float]) -> None:
        self._fh.write(json.dumps({"step": step} | {k: _jsonable(v) for k, v in scalars.items()}) + "\n")

    def summary(self, metrics: Mapping[str, Any]) -> None:
        pass

    def close(self, *, failed: bool = False) -> None:
        self._fh.close()


class WandbSink:
    """Mirror of the JSONL log.  Group = experiment ID, job type = arm."""

    def __init__(self, *, name: str, group: str, job_type: str, tags: list[str], config: Mapping[str, Any],
                 mode: WandbMode) -> None:
        import wandb

        self._run = wandb.init(
            project=WANDB_PROJECT, name=name, group=group, job_type=job_type, tags=tags,
            config=dict(config), mode=mode, dir=str(ROOT), reinit="finish_previous",
            settings=wandb.Settings(silent=True, console="off"),
        )

    def log(self, step: int, scalars: Mapping[str, float]) -> None:
        self._run.log(dict(scalars), step=step)

    def summary(self, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, int | float | bool | str):
                self._run.summary[key] = value

    def close(self, *, failed: bool = False) -> None:
        self._run.finish(exit_code=1 if failed else 0)


class MultiSink:
    """Fan-out to several sinks; a wandb error is downgraded to a warning."""

    def __init__(self, *sinks: Sink) -> None:
        self._sinks = list(sinks)

    @classmethod
    def for_run(cls, log_path: Path, *, wandb_kwargs: Mapping[str, Any] | None) -> Self:
        sinks: list[Sink] = [JsonlSink(log_path)]
        if wandb_kwargs is not None and (mode := wandb_mode()) != "disabled":
            try:
                sinks.append(WandbSink(**wandb_kwargs, mode=mode))
            except Exception as exc:  # noqa: BLE001 — wandb is optional
                warnings.warn(f"wandb disabled for this run: {exc}", RuntimeWarning, stacklevel=2)
        return cls(*sinks)

    def _each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for sink in self._sinks:
            try:
                getattr(sink, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if isinstance(sink, JsonlSink):
                    raise
                warnings.warn(f"wandb {method} failed: {exc}", RuntimeWarning, stacklevel=3)

    def log(self, step: int, scalars: Mapping[str, float]) -> None:
        self._each("log", step, scalars)

    def summary(self, metrics: Mapping[str, Any]) -> None:
        self._each("summary", metrics)

    def close(self, *, failed: bool = False) -> None:
        self._each("close", failed=failed)


class NullSink(JsonlSink):
    """For tests: accepts everything, writes nothing."""

    def __init__(self) -> None:
        pass

    @override
    def log(self, step: int, scalars: Mapping[str, float]) -> None:
        pass

    @override
    def close(self, *, failed: bool = False) -> None:
        pass
