"""Run directory layout and lifecycle (CLAUDE.md §1, §8)."""

import json
from pathlib import Path

import pytest

from affdrift.config import Experiment
from affdrift.tracking.run import Run


def _exp(tmp_path: Path) -> Experiment:
    return Experiment(tmp_path, {"id": "EX", "level": "L0", "runner": "particles", "seeds": [0],
                                 "arms": {"ref": {}}})


def test_run_writes_the_layout(tmp_path: Path) -> None:
    run = Run(_exp(tmp_path), "ref", 0, use_wandb=False)
    with run:
        run.log(0, rms_v=1.0)
        run.finish({"x": 1})
    files = {p.name for p in run.path.iterdir()}
    assert {"config.yaml", "config_hash.txt", "delta.json", "env.json", "status.json", "log.jsonl",
            "metrics.json", "figs"} <= files
    assert json.loads((run.path / "status.json").read_text())["state"] == "complete"
    assert run.path == tmp_path / "runs" / "ref" / "seed0"


def test_failure_is_recorded(tmp_path: Path) -> None:
    run = Run(_exp(tmp_path), "ref", 0, use_wandb=False)
    with pytest.raises(RuntimeError), run:
        raise RuntimeError("boom")
    status = json.loads((run.path / "status.json").read_text())
    assert status["state"] == "failed" and "boom" in status["reason"]


def test_complete_run_is_not_overwritten(tmp_path: Path) -> None:
    exp = _exp(tmp_path)
    with (run := Run(exp, "ref", 0, use_wandb=False)):
        run.finish({})
    with pytest.raises(FileExistsError), Run(exp, "ref", 0, use_wandb=False):
        pass
