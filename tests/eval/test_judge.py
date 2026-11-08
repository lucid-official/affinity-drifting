"""Bootstrap CIs and the pre-registered judges."""

import numpy as np

from affdrift.eval.judge import Verdict, judge
from affdrift.eval.stats import bootstrap_ci, paired_bootstrap_ci


def _table(**arms: list[dict]) -> dict:
    return {a: dict(enumerate(rows)) for a, rows in arms.items()}


def test_bootstrap_ci_contains_mean() -> None:
    ci = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ci.lo <= ci.mean == 3.0 <= ci.hi
    d = paired_bootstrap_ci([2, 3, 4], [1, 2, 3])
    assert d.lo == d.hi == d.mean == 1.0


def test_threshold() -> None:
    t = _table(b0=[{"tv": 0.05}, {"tv": 0.07}])
    assert judge({"kind": "threshold", "arm": "b0", "metric": "tv", "fail_above": 0.1}, t).verdict == Verdict.PASS
    t = _table(b0=[{"tv": 0.2}, {"tv": 0.3}])
    assert judge({"kind": "threshold", "arm": "b0", "metric": "tv", "fail_above": 0.1}, t).verdict == Verdict.FAIL


def test_memorization() -> None:
    rule = {"kind": "memorization", "arms": ["s"], "metric": "self", "min": 0.95}
    ok = _table(s=[{"self": 1.0, "stable": True, "steps": 1}])
    bad = _table(s=[{"self": 0.5, "stable": True, "steps": 1}])
    open_ = _table(s=[{"self": 0.5, "stable": False, "steps": 1}])
    assert judge(rule, ok).verdict == Verdict.PASS
    assert judge(rule, bad).verdict == Verdict.FAIL
    assert judge(rule, open_).verdict == Verdict.INCONCLUSIVE


def test_monotone() -> None:
    rule = {"kind": "monotone", "order": ["a", "b"], "metrics": {"m": "increasing"}}
    up = _table(a=[{"m": i} for i in range(5)], b=[{"m": i + 1} for i in range(5)])
    down = _table(a=[{"m": i + 1} for i in range(5)], b=[{"m": i} for i in range(5)])
    assert judge(rule, up).verdict == Verdict.PASS
    assert judge(rule, down).verdict == Verdict.FAIL


def test_equivalence() -> None:
    rule = {"kind": "equivalence", "pairs": [["a", "b"]], "metrics": ["m"], "margin": 0.02}
    rng = np.random.default_rng(0)
    base = rng.uniform(size=5)
    same = _table(a=[{"m": v + 0.001} for v in base], b=[{"m": v} for v in base])
    far = _table(a=[{"m": v + 0.5} for v in base], b=[{"m": v} for v in base])
    assert judge(rule, same).verdict == Verdict.PASS
    assert judge(rule, far).verdict == Verdict.FAIL


def test_closed_form_requires_divergence() -> None:
    rule = {"kind": "closed_form", "converge_arms": ["l0"], "diverge_arms": ["l11"],
            "metrics": ["e"], "max_rel_err": 0.05}
    t = _table(l0=[{"e": 0.01}], l11=[{"diverged": False}])
    assert judge(rule, t).verdict == Verdict.FAIL


def test_standard_metrics_sanity() -> None:
    from affdrift.eval.metrics import density_coverage, mmd_rbf, precision_recall

    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 2))
    same = rng.normal(size=(400, 2))
    collapsed = rng.normal(scale=0.05, size=(400, 2))
    p, r = precision_recall(real, same)
    assert p > 0.9 and r > 0.9
    p, r = precision_recall(real, collapsed)
    assert p > 0.9 and r < 0.3          # mode collapse: precise but no recall
    _, cov = density_coverage(real, collapsed)
    assert cov < 0.3
    assert mmd_rbf(real, same) < mmd_rbf(real, collapsed)


def test_parallel_condition_metrics_equal_serial() -> None:
    from affdrift.eval.metrics import evaluate_conditions

    rng = np.random.default_rng(0)
    c, n = 7, 64
    args = (rng.normal(size=(c, n, 2)), rng.normal(size=(c, n, 2)), rng.normal(size=(c, n, 2)),
            rng.normal(size=(c, n)), rng.normal(size=(c, n)), rng.normal(size=(c, 3, 2)), rng.normal(size=(300, 2)), 0.05)
    serial = evaluate_conditions(*args, workers=1)
    parallel = evaluate_conditions(*args, workers=3)
    for k in serial:
        np.testing.assert_allclose(serial[k], parallel[k])
