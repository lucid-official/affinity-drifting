"""Pre-registered verdicts.

The rule an experiment is judged by is the ``judge:`` block of its
``experiment.yaml``, committed with ``prereg.md`` before any run
(CLAUDE.md §2.9).  This module only evaluates that block; it never takes a
threshold from anywhere else.

Every judge returns PASS, FAIL or INCONCLUSIVE with the evidence (values and
CIs) that produced it.  Several checks combine as the worst of them,
FAIL > INCONCLUSIVE > PASS.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from affdrift.eval.stats import CI, bootstrap_ci, paired_bootstrap_ci

type Table = Mapping[str, Mapping[int, Mapping[str, Any]]]  # arm → seed → metrics


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"

    @staticmethod
    def worst(verdicts: Sequence["Verdict"]) -> "Verdict":
        order = (Verdict.FAIL, Verdict.INCONCLUSIVE, Verdict.PASS)
        return next(v for v in order if v in verdicts) if verdicts else Verdict.INCONCLUSIVE


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    verdict: Verdict
    detail: str
    ci: CI | None = None


@dataclass(frozen=True, slots=True)
class Judgement:
    verdict: Verdict
    kind: str
    checks: list[Check] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "kind": self.kind,
            "checks": [
                {"name": c.name, "verdict": c.verdict.value, "detail": c.detail,
                 "ci": None if c.ci is None else c.ci.to_json()}
                for c in self.checks
            ],
        }


def _values(table: Table, arm: str, metric: str) -> list[float]:
    if arm not in table:
        raise KeyError(f"no complete runs for arm {arm!r}")
    return [float(m[metric]) for _, m in sorted(table[arm].items())]


def _paired(table: Table, a: str, b: str, metric: str) -> CI:
    seeds = sorted(table[a].keys() & table[b].keys())
    return paired_bootstrap_ci([table[a][s][metric] for s in seeds], [table[b][s][metric] for s in seeds])


def judge(rule: Mapping[str, Any], table: Table) -> Judgement:
    kind = rule["kind"]
    match kind:
        case "max_below":
            return _max_below(rule, table)
        case "memorization":
            return _memorization(rule, table)
        case "threshold":
            return _threshold(rule, table)
        case "monotone":
            return _monotone(rule, table)
        case "closed_form":
            return _closed_form(rule, table)
        case "equivalence":
            return _equivalence(rule, table)
        case _:
            raise ValueError(f"unknown judge kind {kind!r}")


def _done(kind: str, checks: list[Check]) -> Judgement:
    return Judgement(Verdict.worst([c.verdict for c in checks]), kind, checks)


def _max_below(rule: Mapping[str, Any], table: Table) -> Judgement:
    """FAIL if the largest value of ``metric`` over the judged arms' runs exceeds ``threshold``."""
    checks = []
    for metric in rule["metrics"]:
        values = [v for arm in rule.get("arms", list(table)) for v in _values(table, arm, metric)]
        worst = max(values)
        ok = worst <= rule["threshold"]
        checks.append(Check(metric, Verdict.PASS if ok else Verdict.FAIL,
                            f"max over {len(values)} runs = {worst:.3e} (threshold {rule['threshold']:.0e})"))
    return _done("max_below", checks)


def _memorization(rule: Mapping[str, Any], table: Table) -> Judgement:
    """E0.2: FAIL if a run sits stably at ‖V‖ ≈ 0 without memorising; PASS if every run memorises."""
    checks = []
    for arm in rule["arms"]:
        for seed, m in sorted(table[arm].items()):
            memorised = m[rule["metric"]] >= rule["min"]
            if memorised:
                v = Verdict.PASS
            elif m["stable"]:
                v = Verdict.FAIL
            else:
                v = Verdict.INCONCLUSIVE
            checks.append(Check(f"{arm}/seed{seed}", v,
                                f"{rule['metric']} = {m[rule['metric']]:.4f}, stable = {m['stable']}, steps = {m['steps']}"))
    return _done("memorization", checks)


def _threshold(rule: Mapping[str, Any], table: Table) -> Judgement:
    """FAIL if the seed mean of ``metric`` is above ``fail_above``."""
    ci = bootstrap_ci(_values(table, rule["arm"], rule["metric"]))
    ok = ci.mean <= rule["fail_above"]
    detail = f"seed mean {ci} vs fail_above {rule['fail_above']}"
    return _done("threshold", [Check(f"{rule['arm']}:{rule['metric']}", Verdict.PASS if ok else Verdict.FAIL, detail, ci)])


def _monotone(rule: Mapping[str, Any], table: Table) -> Judgement:
    """Adjacent arms along ``order`` must move each metric in its stated direction.

    FAIL if a paired CI lies entirely on the wrong side of zero;
    INCONCLUSIVE if only the mean does; PASS otherwise.
    """
    checks = []
    order: list[str] = rule["order"]
    for metric, direction in rule["metrics"].items():
        sign = {"increasing": 1.0, "decreasing": -1.0}[direction]
        for prev, nxt in zip(order, order[1:]):
            ci = _paired(table, nxt, prev, metric)
            lo, hi = sorted((sign * ci.lo, sign * ci.hi))
            if hi < 0:
                v = Verdict.FAIL
            elif sign * ci.mean < 0:
                v = Verdict.INCONCLUSIVE
            else:
                v = Verdict.PASS
            checks.append(Check(f"{metric}: {nxt} − {prev}", v, f"paired diff {ci} (expected {direction})", ci))
    return _done("monotone", checks)


def _closed_form(rule: Mapping[str, Any], table: Table) -> Judgement:
    """E0.5: relative errors below ``max_rel_err`` where λ < 1; divergence where λ ≥ 1."""
    checks = []
    for arm in rule["converge_arms"]:
        for metric in rule["metrics"]:
            ci = bootstrap_ci(_values(table, arm, metric))
            ok = ci.mean <= rule["max_rel_err"]
            checks.append(Check(f"{arm}:{metric}", Verdict.PASS if ok else Verdict.FAIL,
                                f"seed mean {ci} vs {rule['max_rel_err']}", ci))
    for arm in rule["diverge_arms"]:
        flags = [bool(m["diverged"]) for m in table[arm].values()]
        checks.append(Check(f"{arm}:diverged", Verdict.PASS if all(flags) else Verdict.FAIL,
                            f"{sum(flags)}/{len(flags)} seeds diverged"))
    return _done("closed_form", checks)


def _equivalence(rule: Mapping[str, Any], table: Table) -> Judgement:
    """Paired CI within ±margin → PASS; entirely outside → FAIL; straddling → INCONCLUSIVE."""
    checks = []
    m = float(rule["margin"])
    for a, b in rule["pairs"]:
        for metric in rule["metrics"]:
            ci = _paired(table, a, b, metric)
            if ci.within(-m, m):
                v = Verdict.PASS
            elif ci.outside(-m, m):
                v = Verdict.FAIL
            else:
                v = Verdict.INCONCLUSIVE
            checks.append(Check(f"{metric}: {a} − {b}", v, f"paired diff {ci}, margin ±{m}", ci))
    return _done("equivalence", checks)


def seed_summary(table: Table, metrics: Sequence[str]) -> dict[str, dict[str, dict[str, float | int]]]:
    """``{arm: {metric: CI}}`` for the report table; non-numeric metrics are skipped."""
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for arm, runs in table.items():
        out[arm] = {}
        for metric in metrics:
            vals = [m.get(metric) for m in runs.values()]
            if all(isinstance(v, int | float) and not isinstance(v, bool) and np.isfinite(v) for v in vals):
                out[arm][metric] = bootstrap_ci([float(v) for v in vals]).to_json()
    return out
