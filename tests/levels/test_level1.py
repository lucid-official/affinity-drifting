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


def test_neighbourhood_truncation_is_exact() -> None:
    """The top-K speed-up gives the same field as the whole bank (B0 bandwidths), and switches itself off when wide."""
    import numpy as np

    from affdrift.config import BASE_PATH, deep_merge, load_yaml
    from affdrift.core.drift import AffinityDrift, DriftSpec, TargetSet
    from affdrift.runtime import numpy_rng, to_numpy, to_tensor

    b0 = load_yaml(BASE_PATH)
    cfg = deep_merge(b0, b0.pop("levels")["L1"])
    bank = level1.bank_of(cfg)
    pos = level1.Positives(cfg, bank, "cpu", level1.bandwidths(cfg, bank), numpy_rng(0, 0))
    assert pos.neighbourhood is not None
    idx = numpy_rng(1, 0).integers(0, len(bank.c), size=8)
    x = to_tensor(numpy_rng(2, 0).normal(size=(8, 16, 2)), "cpu")
    drift = AffinityDrift(DriftSpec(sigma=0.07, repulsion=False, drift_norm=False))
    fast = to_numpy(drift.field(x, pos(idx, bank.c[idx])).v)
    full = to_numpy(drift.field(x, TargetSet(pos.bank_y, pos.log_weights(idx))).v)
    assert np.abs(fast - full).max() < 1e-5
    wide = {**cfg, "h_p": {**cfg["h_p"], "scale": 8.0}}
    assert level1.Positives(wide, bank, "cpu", level1.bandwidths(wide, bank), numpy_rng(0, 0)).neighbourhood is None
