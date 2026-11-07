"""The single-factor Δ check refuses anything the experiment did not declare (CLAUDE.md §2.1)."""

from pathlib import Path

import pytest

from affdrift.config import DeltaViolation, Experiment, deep_merge, diff, expand_dotted, flatten


def _exp(tmp_path: Path, **spec: object) -> Experiment:
    base = {"id": "EX", "level": "L0", "runner": "particles", "seeds": [0], "arms": {"ref": {}}}
    return Experiment(tmp_path, base | spec)


def test_dict_utilities() -> None:
    assert expand_dotted({"a.b": 1, "c": {"d.e": 2}}) == {"a": {"b": 1}, "c": {"d": {"e": 2}}}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    assert flatten({"a": {"b": 1}, "c": [1, 2]}) == {"a.b": 1, "c": [1, 2]}
    assert diff({"a": {"b": 1}}, {"a": {"b": 2}}) == {"a.b": (1, 2)}


def test_reference_arm_has_empty_delta(tmp_path: Path) -> None:
    d = _exp(tmp_path).check_delta("ref")
    assert d.vs_base == {} and d.vs_ref == {}


def test_undeclared_change_is_refused(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"h_q.ratio": 1.0}}}, allowed_delta=["h_p.*"])
    with pytest.raises(DeltaViolation, match="not in allowed_delta"):
        exp.check_delta("x")


def test_two_knobs_against_reference_is_refused(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"h_q.ratio": 1.0, "softmax": "joint"}}},
               allowed_delta=["h_q.ratio", "softmax"])
    with pytest.raises(DeltaViolation, match="at most 1"):
        exp.check_delta("x")


def test_factorial_allows_two_knobs(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"h_q.ratio": 1.0, "softmax": "joint"}}},
               allowed_delta=["h_q.ratio", "softmax"], max_knobs=2)
    assert set(exp.check_delta("x").vs_ref) == {"h_q.ratio", "softmax"}


def test_per_arm_reference(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "a": {"set": {"h_q.ratio": 1.0}},
                               "b": {"set": {"h_q.ratio": 1.0, "softmax": "joint"}, "ref": "a"}},
               allowed_delta=["h_q.ratio", "softmax"])
    assert list(exp.check_delta("b").vs_ref) == ["softmax"]


def test_unknown_key_is_refused(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"h_q.ratoi": 1.0}}}, allowed_delta=["h_q.*"])
    with pytest.raises(DeltaViolation, match="not in B0"):
        exp.check_delta("x")


def test_glob_group_counts_as_one_knob(tmp_path: Path) -> None:
    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"negative.type": "counterfactual", "negative.lambda": 0.5}}},
               allowed_delta=["negative.*"])
    assert exp.check_delta("x").knobs == ("negative.*",)


def test_baseline_arm_is_required(tmp_path: Path) -> None:
    from affdrift.config import MissingBaseline

    with pytest.raises(MissingBaseline):
        _exp(tmp_path).check_baseline()
    ok = _exp(tmp_path, arms={"ref": {}, "drifting": {"set": {"field": "release"}}}, allowed_delta=["field"])
    ok.check_baseline()
    assert list(ok.check_delta("drifting").vs_ref) == ["field"]


def test_run_refuses_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from affdrift.tracking.run import Run

    exp = _exp(tmp_path, arms={"ref": {}, "x": {"set": {"softmax": "joint"}}}, allowed_delta=[])
    with pytest.raises(DeltaViolation):
        Run(exp, "x", 0, use_wandb=False)
    assert not (tmp_path / "runs").exists()
