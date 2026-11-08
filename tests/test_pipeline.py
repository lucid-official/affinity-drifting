"""Automatic report flags (affdrift/pipeline.py)."""

from types import SimpleNamespace

from affdrift.pipeline import _anomalies


def _runs(metrics: dict) -> dict:
    return {"arm": {0: SimpleNamespace(metrics=metrics)}}


def test_step_cap_flag_only_for_particle_runs() -> None:
    assert _anomalies(_runs({"steps": 100, "stable": False, "final_rms_v": 1e-3}))  # particle run at the cap
    assert not _anomalies(_runs({"steps": 100, "stable": True}))
    assert not _anomalies(_runs({"steps": 20000}))  # trained generator: fixed steps, no stability notion
