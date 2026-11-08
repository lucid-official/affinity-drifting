"""Exploratory, NOT pre-registered: reference values for interpreting the E1.4 and E1.5 failures.

1. E1.5 — TV between the σ-smoothed Nadaraya–Watson mixture Σ w̃_i δ(y_i) and σ-smoothed samples of the TRUE
   conditional p(x|c), on the same 64 training conditions and with the same σ as the pre-registered metric
   (`level1._nw_tv`), plus the finite-sample floor TV(true, true). A generator that learns p(x|c) instead of the
   mixture of the 16 pooled points sits near the first value.
2. E1.3/E1.5 — memorization share (`metrics.memorization`, ε = ½ median nearest-neighbor spacing of the training
   positives) of TRUE samples from p(x|c) on training conditions: the chance level of "landing on a training
   positive" for a generator that has learned p(x|c) exactly.
3. E1.4 — on the 16 stored training conditions of each seed, distance from the mean of the one-positive samples to
   the conditional mean E[x|c] and to the condition's own training positive y_i, for every quantization step δ.

Output: experiments/L1/analysis/reference.json. It never feeds a verdict.
"""

import json
from datetime import UTC, datetime

import numpy as np
from scipy.spatial.distance import cdist

from affdrift.config import EXPERIMENTS_ROOT, Experiment
from affdrift.core.kernels import caption_log_weights
from affdrift.data.toy2d import conditional_mean
from affdrift.levels.level1 import bandwidths, bank_of
from affdrift.runtime import git_commit
from affdrift.tracking.store import complete

GRID = np.linspace(-2.6, 2.6, 120)  # same grid as level1._nw_tv
POINTS = np.stack([a.ravel() for a in np.meshgrid(GRID, GRID)], -1)
CELL = (GRID[1] - GRID[0]) ** 2


def kde(centres: np.ndarray, weights: np.ndarray | float, sigma: float) -> np.ndarray:
    k = np.exp(-cdist(POINTS, centres, "sqeuclidean") / (2 * sigma**2))
    return (k * weights).sum(1) / (2 * np.pi * sigma**2)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum() * CELL)


def nw_reference() -> dict[str, float]:
    exp = Experiment.load("E1.5")
    cfg = exp.resolve("hq0")
    bank = bank_of(cfg)
    s = json.loads((exp.dir / "sigma.json").read_text())["sigma"]
    sigma = float(min(s)) if isinstance(s, list) else float(s)  # finest scale, as passed to _nw_tv
    idx = bank.eval_train[:64]
    hp = bandwidths({**cfg, "h_p": {**cfg["h_p"], "rule": "n_eff", "scale": 1.0, "neighbors": "semantic"}}, bank)
    w = np.exp(caption_log_weights(bank.c[idx], bank.c, hp[idx]))
    real, _ = bank.true_samples(bank.c[idx], 512, 7)
    real2, _ = bank.true_samples(bank.c[idx], 512, 8)
    nw_true, true_true = [], []
    for i in range(len(idx)):
        keep = w[i] > 1e-6 * w[i].max()
        p, q, q2 = kde(bank.y[keep], w[i][keep], sigma), kde(real[i], 1 / 512, sigma), kde(real2[i], 1 / 512, sigma)
        nw_true.append(tv(p, q))
        true_true.append(tv(q2, q))
    return {"sigma": sigma, "tv_nw_vs_true": float(np.mean(nw_true)), "tv_true_vs_true": float(np.mean(true_true))}


def memorization_chance() -> dict[str, float]:
    from scipy.spatial import cKDTree

    from affdrift.eval.metrics import memorization

    bank = bank_of(Experiment.load("E1.3").resolve("AffD"))
    eps = 0.5 * float(np.median(cKDTree(bank.y).query(bank.y, k=2)[0][:, 1]))
    idx = bank.eval_train[:256]
    real, _ = bank.true_samples(bank.c[idx], 512, 7)
    return {"eps": eps, "true_samples": float(np.mean([memorization(r, bank.y, eps) for r in real]))}


def collapse_target() -> dict[str, dict[str, float]]:
    exp = Experiment.load("E1.4")
    runs = complete(exp)
    out = {}
    for d in ("d0", "d64", "d16", "d4"):
        arm = f"{d}_h0"
        bank = bank_of(exp.resolve(arm))
        to_mean, to_own = [], []
        for rec in runs[arm].values():
            c, xbar = rec.arrays["c_train"], rec.arrays["fake_train"].mean(1)
            j = np.argmin(cdist(c, bank.c), axis=1)
            assert np.abs(bank.c[j] - c).max() < 1e-9
            to_mean.append(float(np.linalg.norm(xbar - conditional_mean(c, mix_beta=bank.mix_beta), axis=1).mean()))
            to_own.append(float(np.linalg.norm(xbar - bank.y[j], axis=1).mean()))
        out[arm] = {"to_conditional_mean": float(np.mean(to_mean)), "to_own_positive": float(np.mean(to_own))}
    return out


def main() -> None:
    out = {"note": __doc__.split("\n\n")[0], "commit": git_commit()[:10],
           "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "E1.5 NW reference": nw_reference(),
           "memorization chance level": memorization_chance(), "E1.4 one-positive sample mean": collapse_target()}
    path = EXPERIMENTS_ROOT / "L1" / "analysis" / "reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
