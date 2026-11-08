"""L1 runner pieces that do not need a training run."""

import json
from pathlib import Path

import pytest

from affdrift.config import Experiment
from affdrift.levels import level1


@pytest.mark.parametrize("rule", ["knn_median", "multiscale"])
def test_frozen_sigma_round_trips(tmp_path: Path, rule: str) -> None:
    """σ is computed once, written to sigma.json, and read back identically (list for multi-scale)."""
    exp = Experiment(tmp_path, {"id": "EX", "level": "L1", "runner": "train", "seeds": [0], "reference_arm": "a",
                                "allowed_delta": ["*"], "arms": {"a": {"set": {"x_kernel.sigma_rule": rule}}}})
    first = level1.frozen_sigma(exp)
    again = level1.frozen_sigma(exp)
    assert first == again
    assert isinstance(again, list) == (rule == "multiscale")
    assert json.loads((tmp_path / "sigma.json").read_text())["rule"] == rule
