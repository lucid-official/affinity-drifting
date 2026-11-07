"""Exploratory, NOT pre-registered: L0 fixed points compared at the resolution the field can see.

The pre-registered E0.2/E0.3/E0.6 metrics assign each particle to its nearest positive, a sub-σ quantity. This script
compares σ-smoothed densities instead (TV between Gaussian KDEs on a grid) and tracks the E0.2 self ratio over
snapshots. Output: experiments/L0/analysis/smoothed.json. It never feeds a verdict.
"""

import json
from datetime import UTC, datetime

import numpy as np

from affdrift.config import EXPERIMENTS_ROOT, Experiment
from affdrift.eval.metrics import assignment_matrix
from affdrift.runtime import git_commit
from affdrift.tracking.store import RunRecord, complete

GRID = np.linspace(-2.2, 2.2, 160)
POINTS = np.stack([a.ravel() for a in np.meshgrid(GRID, GRID)], -1)
CELL = (GRID[1] - GRID[0]) ** 2
CONDITIONS = range(0, 32, 4)  # every 4th condition


def kde(centres: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    d2 = ((POINTS[:, None, :] - centres[None]) ** 2).sum(-1)
    return (np.exp(-d2 / (2 * sigma**2)) * weights).sum(1) / (2 * np.pi * sigma**2)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum() * CELL)


def particles(rec: RunRecord, k: int) -> np.ndarray:
    return kde(rec.arrays["final"][k], np.full(rec.arrays["final"].shape[1], 1 / rec.arrays["final"].shape[1]),
               rec.metrics["sigma"])


def vs_target(rec: RunRecord) -> float:
    """Mean over conditions of TV(p̂_k, q̂_k), both smoothed with the experiment's σ."""
    y, w, s = rec.arrays["y"], rec.arrays["pos_weights"], rec.metrics["sigma"]
    return float(np.mean([tv(kde(y, w[k], s), particles(rec, k)) for k in CONDITIONS]))


def between(a: RunRecord, b: RunRecord) -> float:
    return float(np.mean([tv(particles(a, k), particles(b, k)) for k in CONDITIONS]))


def main() -> None:
    out: dict = {"note": __doc__.split("\n\n")[0], "commit": git_commit()[:10],
                 "time_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    for eid in ("E0.2", "E0.3", "E0.4", "E0.6"):
        runs = complete(Experiment.load(eid))
        out[f"{eid}: smoothed TV(p̂, q̂) per seed"] = {
            arm: [round(vs_target(r), 4) for _, r in sorted(rs.items())] for arm, rs in runs.items()}
    e6 = complete(Experiment.load("E0.6"))
    out["E0.6: smoothed TV(q̂_joint, q̂_sep) per seed"] = {
        f"{a} vs {b}": [round(between(e6[a][s], e6[b][s]), 4) for s in sorted(e6[a])]
        for a, b in (("joint_hq0", "sep_hq0"), ("joint_sym", "sep_sym"))}
    e2 = complete(Experiment.load("E0.2"))
    traj = {}
    for arm, rs in e2.items():
        r = rs[min(rs)]
        steps, snaps, y = r.arrays["snapshot_steps"], r.arrays["snapshots"], r.arrays["y"]
        traj[arm] = {int(t): round(float(np.diag(assignment_matrix(x, y)).mean()), 4)
                     for t, x in zip(steps, snaps) if t in (0, 1, 2) or t > 100}
    out["E0.2: self_mean over snapshots (seed 0)"] = traj
    path = EXPERIMENTS_ROOT / "L0" / "analysis" / "smoothed.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
