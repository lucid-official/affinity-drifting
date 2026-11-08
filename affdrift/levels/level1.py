"""Level 1 runners: a conditional generator trained with each arm's field (plan §5).

Training uses the standard recipe of the drifting / diffusion line (docs: affd-standard-methodology): Adam with
constant learning rate, an EMA of the generator weights (decay 0.999, `lucid.optim.swa_utils`), a fixed step
budget for every arm (D19), and evaluation with the EMA weights. The learning rate is chosen once at Gate G4 (D17).

Arms are options of the same code path (CLAUDE.md §2.2):

=================  ======================================================================
AffD (B0)          caption weights from the N_eff rule over the whole bank, Gaussian k_x
single             ``h_p.rule: zero`` — one-hot on the condition's own positive
K1                 ``x_kernel.kind: none``
K2                 ``h_p.neighbors: random`` — one global permutation of the bank columns
K3                 ``positives.noise: matched`` — single + Gaussian noise of the borrowed variance
K4                 ``positives.source: oracle`` — fresh samples of p(x|c) each step
K5                 ``h_p.rule: inf`` — uniform over the bank (unconditional)
drifting           ``field: release`` — the Lucid release loss, own positive only
=================  ======================================================================
"""

import functools
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import lucid
import numpy as np
from lucid import Tensor
from lucid.optim import Adam
from lucid.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from affdrift._typing import Config, FloatArray, Metrics
from affdrift.config import Experiment
from affdrift.core.bandwidth import solve_hp
from affdrift.core.drift import AffinityDrift, DriftLoss, DriftSpec, MultiScaleAffinityDrift, RawDriftLoss, Softmax, TargetSet
from affdrift.core.kernels import XKernel, caption_log_weights, log_normalize
from affdrift.data.toy2d import Bank, make_bank
from affdrift.eval import metrics as M
from affdrift.models.generator import GeneratorSpec, MLPGenerator
from affdrift.runtime import ROOT, default_device, numpy_rng, to_numpy, to_tensor
from affdrift.tracking.run import Run

SMOKE_STEPS = 300
EVAL_SEED = 777


# ── data, bandwidth, σ ──────────────────────────────────────────────────────


@functools.cache
def _bank(n_train: int, n_heldout: int, n_eval_train: int, seed: int, mode_std: float, mix_beta: float) -> Bank:
    return make_bank(n_train=n_train, n_heldout=n_heldout, n_eval_train=n_eval_train, seed=seed,
                     mode_std=mode_std, mix_beta=mix_beta)


def bank_of(cfg: Config) -> Bank:
    l1 = cfg["level1"]
    return _bank(l1["n_train"], l1["n_heldout"], l1["n_eval_train"], l1["data_seed"], l1["mode_std"], l1["mix_beta"])


@functools.cache
def _n_eff_bandwidth(n_train: int, seed: int, n_star: float, key: tuple) -> FloatArray:
    bank = _bank(*key)
    return np.concatenate([solve_hp(bank.c[i:i + 500], bank.c, n_star) for i in range(0, len(bank.c), 500)])


def bandwidths(cfg: Config, bank: Bank) -> FloatArray:
    """h_p for every bank condition (``rule: zero``/``inf`` are the single and K5 limits)."""
    hp, l1 = cfg["h_p"], cfg["level1"]
    match hp["rule"]:
        case "n_eff":
            key = (l1["n_train"], l1["n_heldout"], l1["n_eval_train"], l1["data_seed"], l1["mode_std"], l1["mix_beta"])
            return _n_eff_bandwidth(l1["n_train"], l1["data_seed"], float(hp["n_star"]), key) * float(hp["scale"])
        case "fixed":
            return np.full(len(bank.c), float(hp["fixed_value"]))
        case "zero":
            return np.zeros(len(bank.c))
        case "inf":
            return np.full(len(bank.c), np.inf)
        case other:
            raise ValueError(f"unknown h_p.rule {other!r}")


def knn_median_sigma(points: FloatArray, k: int) -> float:
    """D15: median over the bank of the distance to the k-th nearest other positive."""
    d, _ = cKDTree(points).query(points, k=k + 1)
    return float(np.median(d[:, k]))


def multiscale_sigmas(points: FloatArray, k: int, n_scales: int = 3) -> list[float]:
    """D20 candidate: geometric ladder from the kNN width (D15) to the median heuristic (Gretton et al., 2012)."""
    sub = numpy_rng(0, 90).choice(len(points), size=min(3000, len(points)), replace=False)
    from scipy.spatial.distance import pdist

    lo, hi = knn_median_sigma(points, k), float(np.median(pdist(points[sub])))
    return [float(v) for v in np.geomspace(lo, hi, n_scales)]


def frozen_sigma(exp: Experiment) -> float | list[float]:
    """σ frozen once per experiment in ``sigma.json`` from the reference arm's bank (CLAUDE.md §2.3)."""
    path = exp.dir / "sigma.json"
    ref = exp.resolve(exp.reference_arm)
    rule = ref["x_kernel"]["sigma_rule"]
    if path.exists():
        rec = json.loads(path.read_text())
        if rec["rule"] != rule:
            raise ValueError(f"{path} frozen with {rec['rule']!r}, config asks {rule!r}")
        return [float(v) for v in rec["sigma"]] if isinstance(rec["sigma"], list) else float(rec["sigma"])
    match rule:
        case "knn_median":
            sigma = knn_median_sigma(bank_of(ref).y, int(ref["h_p"]["n_star"]))
        case "multiscale":
            sigma = multiscale_sigmas(bank_of(ref).y, int(ref["h_p"]["n_star"]))
        case _:
            raise ValueError(f"L1 sigma_rule must be knn_median (D15) or multiscale, got {rule!r}")
    path.write_text(json.dumps({"sigma": sigma, "rule": rule, "computed_from_arm": exp.reference_arm,
                                "time_utc": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n")
    return sigma


# ── positives per arm ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Positives:
    """Builds each step's positive :class:`TargetSet` for a batch of bank indices."""

    cfg: Config
    bank: Bank
    device: str
    h_p: FloatArray
    rng: np.random.Generator

    @functools.cached_property
    def bank_c(self) -> Tensor:
        return to_tensor(self.bank.c, self.device)

    @functools.cached_property
    def bank_y(self) -> Tensor:
        return to_tensor(self.bank.y[None], self.device)

    @functools.cached_property
    def permutation(self) -> np.ndarray:
        """K2: one global permutation of bank columns, fixed by the data seed."""
        return numpy_rng(self.cfg["level1"]["data_seed"], 7).permutation(len(self.bank.c))

    @functools.cached_property
    def matched_noise_std(self) -> float:
        """K3: √(mean per-dimension variance of the positives AffD would borrow), on 512 bank conditions."""
        hp = _n_eff_bandwidth(self.cfg["level1"]["n_train"], self.cfg["level1"]["data_seed"],
                              float(self.cfg["h_p"]["n_star"]),
                              tuple(self.cfg["level1"][k] for k in ("n_train", "n_heldout", "n_eval_train",
                                                                    "data_seed", "mode_std", "mix_beta")))
        idx = numpy_rng(self.cfg["level1"]["data_seed"], 8).choice(len(self.bank.c), 512, replace=False)
        w = np.exp(caption_log_weights(self.bank.c[idx], self.bank.c, hp[idx]))
        mean = w @ self.bank.y
        var = (w * ((self.bank.y[None] - mean[:, None]) ** 2).sum(-1)).sum(1) / 2
        return float(np.sqrt(var.mean()))

    def log_weights(self, idx: np.ndarray) -> Tensor:
        """``(B, 1, N)`` normalised caption log-weights of the batch conditions over the bank."""
        rule = self.cfg["h_p"]["rule"]
        b, n = len(idx), len(self.bank.c)
        match rule:
            case "zero":
                lw = np.full((b, n), -np.inf)
                lw[np.arange(b), idx] = 0.0
                logw = to_tensor(lw, self.device)
            case "inf":
                logw = lucid.full((b, n), -math.log(n), device=self.device)
            case _:
                c = self.bank_c[lucid.tensor(idx, device=self.device)]
                d2 = ((c.unsqueeze(1) - self.bank_c.unsqueeze(0)) ** 2).sum(dim=-1)
                h = to_tensor(self.h_p[idx], self.device).unsqueeze(1)
                logits = d2 * (-0.5) / (h * h)
                logw = logits - lucid.logsumexp(logits, dim=-1, keepdim=True)
        if self.cfg["h_p"]["neighbors"] == "random":
            logw = logw[:, lucid.tensor(self.permutation, device=self.device)]
        return logw.unsqueeze(1)

    #: Neighbours kept when the caption weights beyond them are exactly negligible (see :meth:`neighbourhood`).
    TOP_K = 1024
    #: Largest weight dropped, relative to the row maximum, for the neighbourhood to be used.
    DROP_TOL = 1e-12

    @functools.cached_property
    def neighbourhood(self) -> np.ndarray | None:
        """``(N, K)`` caption-space neighbours of every bank condition, or ``None`` when truncation is not exact.

        A pure speed-up: the field is computed over the K nearest conditions only when every dropped caption weight
        is below ``DROP_TOL`` × the row maximum, i.e. zero in float32. Rules ``inf``, ``zero``, random neighbours and
        bandwidths too wide for that fall back to the whole bank.
        """
        hp = self.cfg["h_p"]
        if hp["rule"] not in ("n_eff", "fixed") or hp["neighbors"] != "semantic":
            return None
        d, nbr = cKDTree(self.bank.c).query(self.bank.c, k=self.TOP_K)
        dropped = np.exp(-(d[:, -1] ** 2) / (2 * self.h_p**2))      # weight just outside the kept set
        return nbr if float(dropped.max()) < self.DROP_TOL else None

    def __call__(self, idx: np.ndarray, c: FloatArray) -> TargetSet:
        match self.cfg["positives"]["source"]:
            case "oracle":
                n = int(self.cfg["positives"]["oracle_n"])
                y, _ = self.bank.true_samples(c, n, seed=int(self.rng.integers(1 << 31)))
                return TargetSet.uniform(to_tensor(y, self.device))
            case "bank" if (nbr := self.neighbourhood) is not None:
                rows = nbr[idx]                                                # (B, K)
                d2 = ((self.bank.c[rows] - self.bank.c[idx][:, None, :]) ** 2).sum(-1)
                logits = -d2 / (2 * self.h_p[idx][:, None] ** 2)
                logw = logits - logits.max(-1, keepdims=True)
                logw -= np.log(np.exp(logw).sum(-1, keepdims=True))
                return TargetSet(to_tensor(self.bank.y[rows], self.device), to_tensor(logw[:, None, :], self.device))
            case "bank":
                return TargetSet(self.bank_y, self.log_weights(idx))
            case other:
                raise ValueError(f"unknown positives.source {other!r}")


# ── training ────────────────────────────────────────────────────────────────


def generator_spec(cfg: Config) -> GeneratorSpec:
    m, l1 = cfg["model"], cfg["level1"]
    return GeneratorSpec(eps_dim=m["eps_dim"], hidden=m["hidden"], depth=m["depth"], cond_hidden=m["cond_hidden"],
                         delta=float(l1["delta"]), lipschitz=None if l1["lipschitz"] is None else float(l1["lipschitz"]))


type LossFn = Callable[[Tensor, np.ndarray, FloatArray], tuple[Tensor, dict[str, Tensor]]]


def make_loss(cfg: Config, bank: Bank, sigma: float | list[float], device: str, seed: int) -> tuple[LossFn, dict[str, Any]]:
    """The arm's loss ``(x (B, M, 2), idx, c) -> (scalar loss, info)`` and what it resolved to."""
    rng = numpy_rng(seed, 30)
    match cfg["field"]:
        case "release":
            temps = tuple(cfg["release"]["temperatures"])
            drift: DriftLoss = DriftLoss(temps) if cfg["normalization"]["drift"] else RawDriftLoss(temps)
            y = bank.y

            def release_loss(x: Tensor, idx: np.ndarray, c: FloatArray) -> tuple[Tensor, dict[str, Tensor]]:
                loss, info = drift(x, to_tensor(y[idx][:, None, :], device))
                return loss.mean(), info

            return release_loss, {"field": "release"}
        case "affd":
            pos = Positives(cfg, bank, device, bandwidths(cfg, bank), rng)
            noise = 0.0 if cfg["positives"]["noise"] == "none" else pos.matched_noise_std
            sigmas = sigma if isinstance(sigma, list) else [sigma]
            spec = DriftSpec(sigma=sigmas[0], kernel=XKernel(cfg["x_kernel"]["kind"]), softmax=Softmax(cfg["softmax"]),
                             repulsion=True, drift_norm=bool(cfg["normalization"]["drift"]), pos_noise_std=noise)
            gen = lucid.Generator(device=device) if noise > 0 else None
            if gen is not None:
                gen.manual_seed(seed)
            affd = (MultiScaleAffinityDrift(spec, sigmas, generator=gen) if len(sigmas) > 1
                    else AffinityDrift(spec, generator=gen))
            exclude_self = bool(cfg["h_q"]["exclude_self"])
            ratio = float(cfg["h_q"]["ratio"])

            def affd_loss(x: Tensor, idx: np.ndarray, c: FloatArray) -> tuple[Tensor, dict[str, Tensor]]:
                gen_targets = None
                if ratio > 0:
                    gen_targets = _cross_condition_repulsion(x, idx, pos, ratio, exclude_self)
                elif not exclude_self:
                    gen_targets = TargetSet.own_samples(x, exclude_self=False)
                loss, info = affd(x, pos(idx, c), gen_targets=gen_targets)
                return loss.mean(), info

            return affd_loss, {"field": "affd", "pos_noise_std": noise,
                               "h_p_median": float(np.median(pos.h_p)) if np.isfinite(pos.h_p).all() else float("inf")}
        case other:
            raise ValueError(f"unknown field {other!r}")


def _cross_condition_repulsion(x: Tensor, idx: np.ndarray, pos: Positives, ratio: float,
                               exclude_self: bool) -> TargetSet:
    """h_q > 0: repel from every generated sample in the batch, weighted by the caption kernel at h_q = ratio·h_p."""
    b, m, d = (int(s) for s in x.shape)
    c = pos.bank.c[idx]
    log_wq = caption_log_weights(c, c, ratio * pos.h_p[idx])                     # (B, B)
    full = np.repeat(log_wq - math.log(m), m, axis=1)[:, None, :].repeat(m, axis=1)  # (B, M, B·M)
    if exclude_self:
        rows = np.arange(m)
        for k in range(b):
            full[k, rows, k * m + rows] = -np.inf
    return TargetSet(x.reshape(1, b * m, d), to_tensor(log_normalize(full), pos.device))


def sample_conditions(cfg: Config, bank: Bank, b: int, rng: np.random.Generator) -> np.ndarray:
    """Bank indices for one batch.

    ``uniform`` (B0): i.i.d. over the bank. ``clustered`` (E1.5, every arm alike): ``b / group`` anchors, each with
    ``group − 1`` neighbours drawn from its caption weights w̃(c_anchor), so that repulsion with h_q > 0 has
    neighbouring conditions' samples in the batch to act on.
    """
    n = len(bank.c)
    match cfg["train"]["batch_mode"]:
        case "uniform":
            return rng.integers(0, n, size=b)
        case "clustered":
            group = int(cfg["train"]["group"])
            anchors = rng.integers(0, n, size=b // group)
            hp = bandwidths({**cfg, "h_p": {**cfg["h_p"], "rule": "n_eff", "scale": 1.0}}, bank)
            w = np.exp(caption_log_weights(bank.c[anchors], bank.c, hp[anchors]))
            return np.concatenate([np.r_[a, rng.choice(n, size=group - 1, p=row)] for a, row in zip(anchors, w)])
        case other:
            raise ValueError(f"unknown train.batch_mode {other!r}")


@dataclass(slots=True)
class Trained:
    model: MLPGenerator
    ema: AveragedModel
    log: list[dict[str, float]]
    probe_snapshots: dict[int, FloatArray]
    seconds: float


def train(run: Run, cfg: Config, bank: Bank, sigma: float | list[float], *, steps: int, on_log: Callable[[int, Mapping[str, float]], None]
          ) -> tuple[Trained, dict[str, Any]]:
    device = default_device()
    seed = run.seed
    lucid.manual_seed(seed)  # parameter init only; every data draw below uses an explicit generator
    model = MLPGenerator(generator_spec(cfg)).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(cfg["train"]["ema"])))
    opt = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    loss_fn, resolved = make_loss(cfg, bank, sigma, device, seed)
    rng = numpy_rng(seed, 31)
    g = lucid.Generator(device=device)
    g.manual_seed(seed)
    b, m, eps_dim = int(cfg["train"]["batch_conditions"]), int(cfg["gen"]["M"]), int(cfg["model"]["eps_dim"])
    eval_every = int(cfg["train"]["eval_every"])
    probe = probe_inputs(cfg, device)
    snaps_at = {0, steps // 4, steps // 2, steps}
    snaps: dict[int, FloatArray] = {}
    log: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(steps + 1):
        if step in snaps_at:
            snaps[step] = sample(ema.module if step else model, probe["eps"], probe["c"])
        if step == steps:
            break
        idx = sample_conditions(cfg, bank, b, rng)
        c_np = bank.c[idx]
        eps = lucid.randn(b, m, eps_dim, generator=g, device=device)
        x = model(eps, to_tensor(c_np, device).unsqueeze(1))
        loss, info = loss_fn(x, idx, c_np)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)
        if step % eval_every == 0 or step == steps - 1:
            _materialize(loss, model, ema, opt)
        if step % int(cfg["train"]["log_every"]) == 0:
            row = {"loss": float(loss.item())} | {k: float(v.item()) for k, v in info.items() if v.ndim == 0}
            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            log.append({"step": step, **row})
            on_log(step, row)
    return Trained(model, ema, log, snaps, time.perf_counter() - start), resolved


def _materialize(loss: Tensor, model: MLPGenerator, ema: AveragedModel, opt: Adam) -> None:
    """Evaluate everything a step produced (CLAUDE.md §7.4).

    Metal is lazy: evaluating only the loss leaves the updated parameters, the EMA copy and Adam's moments as
    pending graphs, which grow until Metal's resource limit (seen at ~18k steps in the first G4 run). Called every
    ``train.eval_every`` steps: every step costs 11.3 ms/step, every 10 steps 3.7 ms/step (measured), and a
    10-step graph is far below the limit.
    """
    state = [t for st in opt.state.values() for t in st.values() if isinstance(t, Tensor)]
    lucid.eval(loss, *model.parameters(), *ema.module.parameters(), *state)


# ── sampling and evaluation ─────────────────────────────────────────────────


def probe_inputs(cfg: Config, device: str) -> dict[str, Tensor]:
    """The committed probe set (plan §10.1): 16 conditions × 16 ε seeds."""
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    eps_dim = int(cfg["model"]["eps_dim"])
    eps = np.stack([np.random.default_rng(s).normal(size=eps_dim) for s in probe["eps_seeds"]])
    c = np.array([p["c"] for p in probe["conditions"]])
    return {"eps": to_tensor(np.broadcast_to(eps, (len(c), *eps.shape)).copy(), device),
            "c": to_tensor(c[:, None, :], device), "c_np": c, "eps_np": eps}


def sample(model: MLPGenerator, eps: Tensor, c: Tensor) -> FloatArray:
    with lucid.no_grad():
        out = model(eps, c)
        out.eval()
    return to_numpy(out)


def generate(model: MLPGenerator, c: FloatArray, n: int, eps_dim: int, seed: int, device: str) -> FloatArray:
    eps = numpy_rng(seed, 40).normal(size=(len(c), n, eps_dim))
    out = []
    for i in range(0, len(c), 64):
        out.append(sample(model, to_tensor(eps[i:i + 64], device), to_tensor(c[i:i + 64, None, :], device)))
    return np.concatenate(out)


def evaluate(model: MLPGenerator, cfg: Config, bank: Bank, *, device: str, sigma: float, n_nw: int = 64
             ) -> tuple[Metrics, dict]:
    """Plan §5.1 metrics (judged) and standard metrics (reported), on training and held-out conditions."""
    l1, eps_dim = cfg["level1"], int(cfg["model"]["eps_dim"])
    n = int(l1["eval_samples"])
    nn1 = float(np.median(cKDTree(bank.y).query(bank.y, k=2)[0][:, 1]))
    mem_eps = 0.5 * nn1
    out: Metrics = {"memorization_eps": mem_eps}
    arrays: dict[str, np.ndarray] = {}
    for split, conds, seed in (("train", bank.c[bank.eval_train], 1), ("heldout", bank.c_heldout, 2)):
        fake = generate(model, conds, n, eps_dim, EVAL_SEED + seed, device)
        real, _ = bank.true_samples(conds, n, seed=EVAL_SEED + 10 + seed)
        centres = bank.modes(conds)
        per: dict[str, list[float]] = {k: [] for k in ("w2", "mci", "off_manifold", "mode_coverage", "memorization",
                                                          "precision", "recall", "density", "coverage", "mmd")}
        real2, _ = bank.true_samples(conds, n, seed=EVAL_SEED + 20 + seed)   # independent draw: data–data floor
        lp_fake = bank.log_density(fake, conds)
        lp_real = bank.log_density(real, conds)
        per["w2_datadata"] = []
        for i in range(len(conds)):
            f, r = fake[i], real[i]
            per["w2"].append(_w2(f, r))
            per["w2_datadata"].append(_w2(real2[i], r))
            per["mci"].append(M.mci(f, r))
            per["off_manifold"].append(M.off_manifold(lp_fake[i], float(np.quantile(lp_real[i], 0.01))))
            per["mode_coverage"].append(M.mode_coverage(M.nearest_mode(f, centres[i]), 3))
            per["memorization"].append(M.memorization(f, bank.y, mem_eps))
            p_, r_ = M.precision_recall(r, f)
            d_, c_ = M.density_coverage(r, f)
            per["precision"].append(p_)
            per["recall"].append(r_)
            per["density"].append(d_)
            per["coverage"].append(c_)
            per["mmd"].append(M.mmd_rbf(f, r))
        for k, v in per.items():
            out[f"{k}_{split}"] = float(np.mean(v))
        pool = numpy_rng(EVAL_SEED, 60 + seed).choice(len(conds) * n, size=2048, replace=False)
        out[f"w2_marginal_{split}"] = _w2(fake.reshape(-1, 2)[pool], real.reshape(-1, 2)[pool])
        out[f"w2_marginal_datadata_{split}"] = _w2(real2.reshape(-1, 2)[pool], real.reshape(-1, 2)[pool])
        arrays[f"fake_{split}"] = fake[:16]
        arrays[f"real_{split}"] = real[:16]
        arrays[f"c_{split}"] = conds[:16]
    out |= _nw_tv(model, cfg, bank, device=device, n_cond=n_nw, sigma=sigma)
    return out, arrays


def _w2(a: FloatArray, b: FloatArray) -> float:
    cost = cdist(a, b, "sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def _nw_tv(model: MLPGenerator, cfg: Config, bank: Bank, *, device: str, n_cond: int, sigma: float) -> Metrics:
    """E1.5: TV between σ-smoothed densities of the samples and of the NW mixture Σ w̃_i δ(y_i) (training conditions).

    Compared at the resolution the field acts on (its fixed point is q̂ = p̂, Lemma 1); a point-level assignment
    is below σ and was uninformative in L0 (experiments/L0/analysis).
    """
    idx = bank.eval_train[:n_cond]
    hp = bandwidths({**cfg, "h_p": {**cfg["h_p"], "rule": "n_eff", "scale": 1.0, "neighbors": "semantic"}}, bank)
    w = np.exp(caption_log_weights(bank.c[idx], bank.c, hp[idx]))
    fake = generate(model, bank.c[idx], int(cfg["level1"]["eval_samples"]), int(cfg["model"]["eps_dim"]),
                    EVAL_SEED + 3, device)
    g = np.linspace(-2.6, 2.6, 120)
    grid = np.stack([a.ravel() for a in np.meshgrid(g, g)], -1)
    cell = (g[1] - g[0]) ** 2
    norm = 1.0 / (2 * np.pi * sigma**2)
    tvs = []
    for i, f in enumerate(fake):
        keep = w[i] > 1e-6 * w[i].max()
        p = (np.exp(-cdist(grid, bank.y[keep], "sqeuclidean") / (2 * sigma**2)) * w[i][keep]).sum(1) * norm
        q = np.exp(-cdist(grid, f, "sqeuclidean") / (2 * sigma**2)).mean(1) * norm
        tvs.append(0.5 * float(np.abs(p - q).sum() * cell))
    own = [float((np.linalg.norm(f - bank.y[j], axis=1) <= 0.5 * float(np.median(
        cKDTree(bank.y).query(bank.y, k=2)[0][:, 1]))).mean()) for f, j in zip(fake, idx)]
    return {"nw_tv_train": float(np.mean(tvs)), "own_positive_share_train": float(np.mean(own))}


def raw_drift_energy(model: MLPGenerator, cfg: Config, bank: Bank, sigma: float | list[float], *, device: str) -> float:
    """E1.6: mean ‖V‖² of the arm's field without drift normalisation on a fixed evaluation batch (diagnostic)."""
    if cfg["field"] != "affd":
        return float("nan")
    idx = bank.eval_train[:64]
    eps = to_tensor(numpy_rng(EVAL_SEED, 41).normal(size=(64, int(cfg["gen"]["M"]), int(cfg["model"]["eps_dim"]))),
                    device)
    with lucid.no_grad():
        x = model(eps, to_tensor(bank.c[idx], device).unsqueeze(1))
        pos = Positives(cfg, bank, device, bandwidths(cfg, bank), numpy_rng(0, 0))
        sigmas = sigma if isinstance(sigma, list) else [sigma]
        spec = DriftSpec(sigma=sigmas[0], kernel=XKernel(cfg["x_kernel"]["kind"]), drift_norm=False)
        v = MultiScaleAffinityDrift(spec, sigmas).field(x, pos(idx, bank.c[idx]),
                                      TargetSet.own_samples(x, exclude_self=bool(cfg["h_q"]["exclude_self"]))).v
        e = (v * v).sum(dim=-1).mean()
        e.eval()
    return float(e.item())


def interpolation_strip(model: MLPGenerator, cfg: Config, device: str) -> FloatArray:
    """FC.4: ε₀ → ε₁ in 16 steps for every probe condition and committed seed pair."""
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    eps_dim, steps = int(cfg["model"]["eps_dim"]), int(probe["fc4_steps"])
    c = np.array([p["c"] for p in probe["conditions"]])
    t = np.linspace(0, 1, steps)[:, None]
    strips = []
    for s0, s1 in probe["fc4_pairs"][: len(c)]:
        e0, e1 = (np.random.default_rng(s).normal(size=eps_dim) for s in (s0, s1))
        strips.append((1 - t) * e0 + t * e1)
    eps = np.stack(strips)                                                         # (16, steps, eps_dim)
    return sample(model, to_tensor(eps, device), to_tensor(c[:, None, :], device))


def epsilon_map(model: MLPGenerator, cfg: Config, device: str, n: int = 64) -> tuple[FloatArray, FloatArray]:
    """F1.3: outputs over a 2-D slice of ε (dims 0 and 1 on [−2.5, 2.5]², the rest 0) for probe conditions 0 and 6."""
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    c = np.array([probe["conditions"][i]["c"] for i in (0, 6)])
    g = np.linspace(-2.5, 2.5, n)
    eps = np.zeros((n * n, int(cfg["model"]["eps_dim"])))
    eps[:, 0], eps[:, 1] = (a.ravel() for a in np.meshgrid(g, g))
    out = sample(model, to_tensor(np.broadcast_to(eps, (2, *eps.shape)).copy(), device), to_tensor(c[:, None, :], device))
    return out, c


# ── runners ─────────────────────────────────────────────────────────────────


def run_train(run: Run) -> Metrics:
    cfg, device = run.config, default_device()
    bank = bank_of(cfg)
    sigma = frozen_sigma(run.exp)
    steps = int(cfg["train"]["steps"]) * int(cfg["train"]["steps_multiplier"])
    if run.smoke:
        steps = min(steps, SMOKE_STEPS)
    trained, resolved = train(run, cfg, bank, sigma, steps=steps, on_log=lambda t, row: run.log(t, **row))
    model = trained.ema.module
    finest = sigma[0] if isinstance(sigma, list) else sigma
    metrics, arrays = evaluate(model, cfg, bank, device=device, sigma=finest)
    metrics |= {"sigma": sigma, "steps": steps, "train_seconds": trained.seconds,
                "raw_drift_energy": raw_drift_energy(model, cfg, bank, sigma, device=device), **resolved}
    run.save_arrays(
        **arrays,
        probe_steps=np.array(sorted(trained.probe_snapshots)),
        probe_samples=np.stack([trained.probe_snapshots[t] for t in sorted(trained.probe_snapshots)]),
        fc4=interpolation_strip(model, cfg, device),
        eps_map=(em := epsilon_map(model, cfg, device))[0],
        eps_map_c=em[1],
    )
    return metrics


def run_lemma0(run: Run) -> Metrics:
    """E1.1 / G2: single positive, repulsion off — the drift gradient equals the L2-regression gradient."""
    cfg, device = run.config, default_device()
    bank = bank_of(cfg)
    lucid.manual_seed(run.seed)
    model = MLPGenerator(generator_spec(cfg)).to(device)
    release = cfg["field"] == "release"
    idx = numpy_rng(run.seed, 50).integers(0, len(bank.c), size=64)
    eps = to_tensor(numpy_rng(run.seed, 51).normal(size=(64, 1, int(cfg["model"]["eps_dim"]))), device)
    c = to_tensor(bank.c[idx], device).unsqueeze(1)
    y = to_tensor(bank.y[idx][:, None, :], device)
    out: Metrics = {}
    for name, loss_of in (
        ("drift", (lambda x: RawDriftLoss(tuple(cfg["release"]["temperatures"]))(x, y)[0].mean()) if release else
                  (lambda x: AffinityDrift(DriftSpec(sigma=0.1, repulsion=False, drift_norm=False))(x, y)[0].mean())),
        ("l2", lambda x: ((x - y) ** 2).mean(dim=(1, 2)).mean()),
    ):
        for p in model.parameters():
            p.grad = None
        loss = loss_of(model(eps, c))
        loss.backward()
        out[name] = [to_numpy(p.grad).ravel() for p in model.parameters()]
    err = max(float(np.abs(a - b).max()) for a, b in zip(out["drift"], out["l2"], strict=True))
    scale = max(float(np.abs(b).max()) for b in out["l2"])
    run.log(0, max_grad_err=err, max_grad=scale)
    # baseline arm: the release field (one generated sample, so no repulsion) is not an L2 regression; reported only
    return {"max_grad_err": err, "max_grad_abs": scale, "max_grad_rel_err": err / max(scale, 1e-12)}


RUNNERS: Mapping[str, Any] = {"train": run_train, "lemma0": run_lemma0}
