"""``EXPERIMENTS.md``: the lab notebook.

Two parts (CLAUDE.md §8):

* **Index** — regenerated between ``<!-- index:start -->`` and
  ``<!-- index:end -->`` from every ``experiment.yaml`` and ``aggregate.json``.
  Never edited by hand.
* **Log** — append-only, newest last.  Scripts append one entry per
  pre-registration, smoke run, sweep and aggregation; people append notes.
  Existing entries are never edited; a correction is a new entry.
"""

import contextlib
import fcntl
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from affdrift.config import Experiment
from affdrift.runtime import ROOT, git_commit

LEDGER = ROOT / "EXPERIMENTS.md"
START, END = "<!-- index:start -->", "<!-- index:end -->"


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive lock so concurrent sweeps never lose each other's entries."""
    with (path.parent / f".{path.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append(kind: str, exp_id: str, lines: Iterable[str], *, path: Path = LEDGER) -> None:
    """Append one log entry: a dated heading and bullet lines."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join(f"- {line}" for line in lines)
    entry = f"\n### {stamp} · {exp_id} · {kind}\n\n- commit `{git_commit()[:10]}`\n{body}\n"
    with _locked(path), path.open("a") as fh:
        fh.write(entry)


def _status(exp: Experiment) -> tuple[str, str]:
    agg = exp.dir / "aggregate.json"
    if agg.exists():
        data = json.loads(agg.read_text())
        return "aggregated", data["judgement"]["verdict"]
    if any((exp.dir / "runs").glob("*/seed*/metrics.json")):
        return "running", "—"
    if (exp.dir / "prereg.md").exists():
        return "pre-registered", "—"
    return "planned", "—"


def rebuild_index(*, path: Path = LEDGER) -> None:
    """Regenerate the index table from the experiment directories."""
    rows = ["| Level | ID | Claim | Title | Status | Verdict | Report |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for exp in Experiment.all():
        status, verdict = _status(exp)
        report = exp.dir / "report.md"
        link = f"[report]({report.relative_to(ROOT)})" if report.exists() else "—"
        rows.append(
            f"| {exp.level} | [{exp.id}]({exp.dir.relative_to(ROOT)}/) | {exp.spec.get('claim', '')} | "
            f"{exp.title} | {status} | {verdict} | {link} |"
        )
    with _locked(path):
        text = path.read_text()
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        path.write_text(f"{head}{START}\n" + "\n".join(rows) + f"\n{END}{tail}")
