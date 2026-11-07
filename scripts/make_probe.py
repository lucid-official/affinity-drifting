"""Create the fixed probe sets (plan §10.1).  Run once, commit, never regenerate.

Refuses to overwrite an existing probe file.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "probe"
SEED = 20261107


def level0() -> dict:
    k = 32
    conditions = np.round(np.linspace(0, k - 1, 16)).astype(int).tolist()
    psi = np.linspace(0, 1, k)
    quiver = min(conditions, key=lambda i: (abs(psi[i] - 0.5), i))
    return {
        "level": "L0",
        "rule": "16 of the 32 conditions, evenly spaced on ψ including both ends; "
                "quiver_condition = the probe condition nearest ψ = 0.5 (lower index on a tie)",
        "conditions": conditions,
        "quiver_condition": int(quiver),
    }


def level1() -> dict:
    rng = np.random.default_rng(SEED)
    easy = np.c_[rng.uniform(0, 1, 6), rng.uniform(0.4, 0.6, 6)]
    hard = np.c_[rng.uniform(0, 1, 6), np.r_[rng.uniform(0.05, 0.2, 3), rng.uniform(0.8, 0.95, 3)]]
    tail = np.array([[0.02, 0.02], [0.02, 0.98], [0.98, 0.02], [0.98, 0.98]])
    conds = np.r_[easy, hard, tail]
    groups = ["easy"] * 6 + ["hard"] * 6 + ["tail"] * 4
    eps_seeds = rng.integers(0, 2**31 - 1, size=16).tolist()
    pairs = rng.integers(0, 2**31 - 1, size=(16, 2)).tolist()
    return {
        "level": "L1",
        "rule": "easy: c₂ ∈ [0.4, 0.6] (balanced mixture); hard: c₂ ∈ [0.05, 0.2] ∪ [0.8, 0.95] (a rare mode); "
                "tail: the four corners of [0, 1]² (fewest neighbours, largest h_p). "
                "ε are stored as seeds so the probe does not depend on the noise dimension.",
        "conditions": [{"c": [round(float(a), 6), round(float(b), 6)], "group": g} for (a, b), g in zip(conds, groups)],
        "eps_seeds": eps_seeds,
        "fc4_pairs": pairs,
        "fc4_steps": 16,
    }


if __name__ == "__main__":
    PROBE.mkdir(exist_ok=True)
    for name, make in (("probe_L0.json", level0), ("probe_L1.json", level1)):
        path = PROBE / name
        if path.exists():
            raise SystemExit(f"{path} exists: probe sets are immutable")
        data = make() | {"generator": "scripts/make_probe.py", "seed": SEED,
                         "created_utc": datetime.now(UTC).isoformat(timespec="seconds")}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print("wrote", path)
