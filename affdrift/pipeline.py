"""The experiment workflow behind ``scripts/``: run → sweep → aggregate → figures → report.

Order and gates (CLAUDE.md §6, §8):

1. ``prereg.md`` and ``experiment.yaml`` must be committed and unmodified
   before any run of the experiment, smoke included.
2. Every run passes the Δ check (:meth:`Experiment.check_delta`).
3. Aggregation judges with the committed ``judge:`` block only.
4. Each step appends to ``EXPERIMENTS.md``.
"""

import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from affdrift.config import Experiment
from affdrift.eval.figures.level0 import BUILDERS as L0_FIGURES
from affdrift.eval.figures.level1 import BUILDERS as L1_FIGURES
from affdrift.eval.figures.writer import FigureWriter, rel
from affdrift.eval.judge import Judgement, judge, seed_summary
from affdrift.levels.level0 import RUNNERS as L0_RUNNERS
from affdrift.levels.level1 import RUNNERS as L1_RUNNERS
from affdrift.runtime import ROOT, git_commit
from affdrift.tracking import ledger
from affdrift.tracking.run import Run
from affdrift.tracking.store import complete, records

RUNNERS = {"L0": L0_RUNNERS, "L1": L1_RUNNERS}
FIGURES = {"L0": L0_FIGURES, "L1": L1_FIGURES}


class NotPreregistered(RuntimeError):
    """The experiment's pre-registration is missing, uncommitted or modified."""


def assert_preregistered(exp: Experiment) -> None:
    files = [exp.dir / "prereg.md", exp.dir / "experiment.yaml"]
    for f in files:
        if not f.exists():
            raise NotPreregistered(f"{rel(f)} does not exist")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=ROOT, capture_output=True)
        if tracked.returncode != 0:
            raise NotPreregistered(f"{rel(f)} is not committed")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, files)], cwd=ROOT)
    if dirty.returncode != 0:
        raise NotPreregistered(f"{exp.id}: prereg.md / experiment.yaml differ from HEAD")


# ── running ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Outcome:
    name: str
    state: str
    seconds: float


def run_one(exp: Experiment, arm: str, seed: int, *, smoke: bool = False, force: bool = False,
            use_wandb: bool = True) -> Outcome:
    runner = RUNNERS[exp.level][exp.runner]
    run = Run(exp, arm, seed, smoke=smoke, force=force, use_wandb=use_wandb)
    if run.is_complete() and not force:
        return Outcome(run.name, "skipped (complete)", 0.0)
    start = time.perf_counter()
    with run:
        metrics = runner(run)
        run.finish(metrics)
    return Outcome(run.name, "complete", time.perf_counter() - start)


def sweep(exp: Experiment, *, arms: Iterable[str] | None = None, seeds: Iterable[int] | None = None,
          smoke: bool = False, force: bool = False, use_wandb: bool = True) -> list[Outcome]:
    assert_preregistered(exp)
    exp.check_baseline()
    arms = list(arms or exp.arms)
    seeds = list(seeds or (exp.seeds[:1] if smoke else exp.seeds))
    for arm in arms:  # refuse the whole sweep before the first run if any arm violates Δ
        exp.check_delta(arm)
    outcomes: list[Outcome] = []
    start = time.perf_counter()
    for arm in arms:
        for seed in seeds:
            try:
                outcomes.append(o := run_one(exp, arm, seed, smoke=smoke, force=force, use_wandb=use_wandb))
            except Exception as exc:
                outcomes.append(o := Outcome(f"{exp.id}/{arm}/seed{seed}", f"failed: {type(exc).__name__}: {exc}", 0.0))
            print(f"{o.name:40s} {o.state} ({o.seconds:.1f}s)", flush=True)
    ledger.append("smoke" if smoke else "sweep", exp.id, [
        f"arms {arms} × seeds {seeds}",
        f"{sum(o.state == 'complete' for o in outcomes)} complete, "
        f"{sum(o.state.startswith('skipped') for o in outcomes)} skipped, "
        f"{sum(o.state.startswith('failed') for o in outcomes)} failed",
        f"wall time {time.perf_counter() - start:.0f} s",
        *[f"FAILED {o.name}: {o.state}" for o in outcomes if o.state.startswith("failed")],
    ])
    ledger.rebuild_index()
    return outcomes


# ── aggregation and report ──────────────────────────────────────────────────


def aggregate(exp: Experiment, *, allow_partial: bool = False) -> Judgement:
    runs = complete(exp)
    declared = records(exp)
    missing = [r.name for arm in declared.values() for r in arm.values() if r.state != "complete"]
    if missing and not allow_partial:
        raise RuntimeError(f"{exp.id}: {len(missing)} runs not complete: {missing[:5]}…")
    table = {arm: {s: r.metrics for s, r in rs.items()} for arm, rs in runs.items()}
    judgement = judge(exp.spec["judge"], table)
    summary = seed_summary(table, exp.spec.get("report_metrics", []))
    out = {
        "id": exp.id, "time_utc": datetime.now(UTC).isoformat(timespec="seconds"), "commit": git_commit(),
        "judgement": judgement.to_json(), "summary": summary, "missing": missing,
        "runs": {arm: {s: {"config_hash": r.config_hash, "state": r.state} for s, r in rs.items()}
                 for arm, rs in declared.items() if all(r.path.exists() for r in rs.values())},
    }
    (exp.dir / "aggregate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    write_report(exp, judgement, summary, runs, missing)
    ledger.append("aggregate", exp.id, [
        f"verdict **{judgement.verdict}** ({judgement.kind})",
        *[f"{c.verdict}: {c.name} — {c.detail}" for c in judgement.checks if c.verdict != "PASS"][:10],
        f"report: `{rel(exp.dir / 'report.md')}`",
    ])
    ledger.rebuild_index()
    return judgement


def _anomalies(runs: dict[str, dict[int, Any]]) -> list[str]:
    """Automatic flags; people add their own below them in the report."""
    flags = []
    for arm, rs in runs.items():
        for s, r in rs.items():
            m = r.metrics
            if m.get("steps") is not None and not m.get("stable") and not m.get("diverged"):
                flags.append(f"`{arm}/seed{s}`: step cap reached without a stable ‖V‖ < tol "
                             f"(final RMS‖V‖ = {m.get('final_rms_v', float('nan')):.2e})")
            if m.get("diverged") and not str(arm).startswith("lam_1"):
                flags.append(f"`{arm}/seed{s}`: diverged")
            for note in m.get("notes", []):
                if "eager" in note and "matches" not in note:
                    flags.append(f"`{arm}/seed{s}`: {note}")
    return flags


def write_report(exp: Experiment, judgement: Judgement, summary: dict[str, Any], runs: dict[str, Any],
                 missing: list[str]) -> None:
    metrics = exp.spec.get("report_metrics", [])
    head = "| arm | " + " | ".join(metrics) + " |\n|" + " --- |" * (len(metrics) + 1)
    rows = [f"| {arm} | " + " | ".join(
        (f"{c['mean']:.4g} [{c['lo']:.4g}, {c['hi']:.4g}]" if (c := summary.get(arm, {}).get(m)) else "—")
        for m in metrics) + " |" for arm in exp.arms if arm in summary]
    checks = "\n".join(f"| {c.verdict} | {c.name} | {c.detail} |" for c in judgement.checks)
    figs = sorted(exp.figure_dir.glob("*/*.png"))
    fig_lines = "\n".join(f"- [{p.stem}]({os.path.relpath(p, exp.dir)})" for p in figs if not p.stem.endswith("_blind"))
    anomalies = "\n".join(f"- {a}" for a in _anomalies(runs)) or "- (자동 감지 없음)"
    report = f"""# {exp.id} — {exp.title}: 결과 보고서

> 자동 생성: `scripts/aggregate.py {exp.id}` · commit `{git_commit()[:10]}` · {datetime.now(UTC):%Y-%m-%d %H:%M UTC}
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **{judgement.verdict}**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
{checks}

## 수치 (seed 평균, 95% bootstrap CI, {len(exp.seeds)} seeds)

{head}
{chr(10).join(rows)}

누락 run: {len(missing)}개{(' — ' + ', '.join(missing[:10])) if missing else ''}

## Figures

{fig_lines or '- (아직 없음: `scripts/make_figures.py ' + exp.id + '`)'}

## 이상 징후

{anomalies}

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
"""
    (exp.dir / "report.md").write_text(report)


# ── figures ─────────────────────────────────────────────────────────────────


def make_figures(exp: Experiment, *, only: Iterable[str] | None = None) -> list[str]:
    runs = complete(exp)
    writer = FigureWriter(exp.figure_dir)
    wanted = dict(exp.spec.get("figures", {}))
    made = []
    for fig_id, opts in wanted.items():
        if only and fig_id not in only:
            continue
        built = FIGURES[exp.level][fig_id](exp, runs, opts or {})
        path = writer.save(fig_id, built.render, built.panels, blind=built.blind, sources=built.sources,
                           extra=built.extra)
        made.append(rel(path))
    # per-run FC.1 (CLAUDE.md §3: every run gets the common figures it can have)
    if "FC.1" in wanted:
        for arm, rs in runs.items():
            for seed, r in rs.items():
                single = {arm: {seed: r}}
                built = FIGURES[exp.level]["FC.1"](exp, single, {"arms": [arm]})
                FigureWriter(exp.run_figure_dir(arm, seed)).save("FC.1", built.render, built.panels, blind=False,
                                                   sources=built.sources)
    ledger.append("figures", exp.id, [f"`{m}`" for m in made] or ["(none)"])
    return made
