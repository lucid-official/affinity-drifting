"""Level 0 runners: resolved config + seed → metrics (plan §4).

Two runners, selected by ``runner:`` in ``experiment.yaml``:

* ``lemma1`` — E0.1, the field against autograd's σ²∇log(p̂/q̂).
* ``particles`` — E0.2–E0.6, particle dynamics to their fixed point.

σ is frozen per experiment in ``sigma.json`` the first time any arm needs
it, computed from the *reference arm's* positive bank, and only read
afterwards (CLAUDE.md §2.3).
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import lucid
import lucid.autograd
import numpy as np
from lucid import Tensor

from affdrift._typing import Config, FloatArray, Metrics
from affdrift.config import Experiment
from affdrift.core.bandwidth import n_eff, solve_hp
from affdrift.core.drift import AffinityDrift, DriftLoss, DriftSpec, RawDriftLoss, Softmax, TargetSet
from affdrift.core.identities import prop_e_equilibrium, score_ratio_field
from affdrift.core.kernels import XKernel, caption_log_weights, log_normalize, shuffle_neighbors
from affdrift.data.particles import (
    ParticleProblem,
    ParticleSystem,
    Trajectory,
    affd_field,
    half_median_pairwise,
    init_particles,
    make_gaussian_pair,
    make_single_bank,
    release_field,
)
from affdrift.eval import metrics as M
from affdrift.runtime import default_device, numpy_rng, to_numpy, to_tensor
from affdrift.tracking.run import Run

#: Smoke runs stop here; the config is untouched so the Δ check is the real one.
SMOKE_MAX_STEPS = 2000

# ── shared pieces ───────────────────────────────────────────────────────────


def positive_bank(cfg: Config) -> FloatArray:
    """All positive points of the Level-0 problem, for the σ rule."""
    l0 = cfg["level0"]
    match l0["problem"]:
        case "single_bank":
            return make_single_bank(l0["K"], seed=l0["data_seed"], box=l0["y_box"], min_sep=l0["y_min_sep"]).y
        case "gaussian_pair":
            g = l0["gaussian_pair"]
            return make_gaussian_pair(tuple(g["mu1"]), tuple(g["mu2"]), g["s"], g["n_pos"], seed=l0["data_seed"])
        case other:
            raise ValueError(f"unknown level0.problem {other!r}")


def frozen_sigma(exp: Experiment) -> float:
    """Read σ from ``sigma.json``; compute and write it once if absent."""
    path = exp.dir / "sigma.json"
    ref_cfg = exp.resolve(exp.reference_arm)
    rule = ref_cfg["x_kernel"]["sigma_rule"]
    if path.exists():
        record = json.loads(path.read_text())
        if record["rule"] != rule:
            raise ValueError(f"{path} was frozen with rule {record['rule']!r}, config asks {rule!r}")
        return float(record["sigma"])
    match rule:
        case "half_median_pairwise":
            sigma = half_median_pairwise(positive_bank(ref_cfg))
        case _:
            raise ValueError(f"unknown sigma_rule {rule!r}")
    path.write_text(json.dumps({
        "sigma": sigma, "rule": rule, "computed_from_arm": exp.reference_arm,
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    return sigma


def drift_spec(cfg: Config, sigma: float) -> DriftSpec:
    if cfg["positives"]["noise"] != "none":
        raise NotImplementedError("K3 noise matching is defined from borrowed positives (L1+)")
    return DriftSpec(
        sigma=sigma,
        kernel=XKernel(cfg["x_kernel"]["kind"]),
        softmax=Softmax(cfg["softmax"]),
        repulsion=True,
        neg_lambda=float(cfg["negative"]["lambda"]) if cfg["negative"]["type"] == "counterfactual" else 0.0,
        drift_norm=bool(cfg["normalization"]["drift"]),
    )


def release_drift(cfg: Config) -> DriftLoss:
    """The original Drifting field (baseline arm ``drifting``): the Lucid release, raw under D4."""
    temps = tuple(cfg["release"]["temperatures"])
    return DriftLoss(temps) if cfg["normalization"]["drift"] else RawDriftLoss(temps)


def bandwidths(cfg: Config, psi: FloatArray) -> FloatArray:
    """h_p per condition from the configured rule (single positive = 0, K5 = ∞)."""
    hp = cfg["h_p"]
    match hp["rule"]:
        case "n_eff":
            return solve_hp(psi, psi, hp["n_star"]) * hp["scale"]
        case "zero":
            return np.zeros(len(psi))
        case "inf":
            return np.full(len(psi), np.inf)
        case other:
            raise ValueError(f"unknown h_p.rule {other!r}")


# ── E0.1: Lemma 1 ───────────────────────────────────────────────────────────


def run_lemma1(run: Run) -> Metrics:
    """Max |V − σ²∇log(p̂/q̂)| on random problems.

    Checked for both repulsion forms (leave-one-out, B0; self-included, L0's
    D6), in float64 on CPU (the judged number) and float32 on Metal (reported).
    The autograd reference is always float64 on CPU.
    """
    if run.config["field"] == "release":
        return _release_identity(run)
    l1 = run.config["level0"]["lemma1"]
    rng = numpy_rng(run.seed, 10)
    out: Metrics = {}
    step = 0
    for sigma in l1["sigmas"]:
        b, c, p, d = l1["groups"], l1["queries"], l1["positives"], l1["dim"]
        x = rng.normal(size=(b, c, d))
        y = rng.normal(size=(1, p, d))
        log_w = caption_log_weights(rng.uniform(size=(b, 1)), rng.uniform(size=(p, 1)), rng.uniform(0.05, 0.5))
        spec = DriftSpec(sigma=sigma, drift_norm=False)

        def problem(device: str, dtype: lucid.dtype, exclude_self: bool) -> tuple[Any, TargetSet, TargetSet]:
            xt = to_tensor(x, device, dtype)
            pos = TargetSet(to_tensor(y, device, dtype), to_tensor(log_w[:, None, :], device, dtype))
            return xt, pos, TargetSet.own_samples(xt, exclude_self=exclude_self)

        for exclude_self in (True, False):
            ref = score_ratio_field(*problem("cpu", lucid.float64, exclude_self), sigma)
            ref.eval()
            tag = "loo" if exclude_self else "self"
            for name, device, dtype in (("f64_cpu", "cpu", lucid.float64), ("f32_metal", default_device(), lucid.float32)):
                v = AffinityDrift(spec).field(*problem(device, dtype, exclude_self)).v.eval()
                err = float(np.abs(to_numpy(v) - to_numpy(ref)).max())
                out[f"max_err_{name}_{tag}_sigma{sigma:g}"] = err
                run.log(step, sigma=sigma, exclude_self=float(exclude_self), **{f"max_err_{name}": err})
                step += 1
    for name in ("f64_cpu", "f32_metal"):
        out[f"max_err_{name}"] = max(v for k, v in out.items() if k.startswith(f"max_err_{name}_"))
    return out


def _release_identity(run: Run) -> Metrics:
    """Baseline arm of E0.1: the baseline class is the Lucid release, loss and gradient."""
    from lucid.models.generative.drifting import _model as release

    rng = numpy_rng(run.seed, 11)
    temps = tuple(run.config["release"]["temperatures"])
    out: Metrics = {}
    for device in ("cpu", default_device()):
        gen = to_tensor(rng.normal(size=(4, 16, 6)), device).requires_grad_(True)
        pos = to_tensor(rng.normal(size=(4, 8, 6)), device)
        ours, _ = DriftLoss(temps)(gen, pos)
        ref, _ = release._drift_loss(gen, pos, None, None, temps)
        (g_ours,) = lucid.autograd.grad(ours.sum(), gen)
        (g_ref,) = lucid.autograd.grad(ref.sum(), gen)
        lucid.eval(ours, ref, g_ours, g_ref)
        out[f"release_loss_diff_{device}"] = float(np.abs(to_numpy(ours) - to_numpy(ref)).max())
        out[f"release_grad_diff_{device}"] = float(np.abs(to_numpy(g_ours) - to_numpy(g_ref)).max())
    out["release_max_diff"] = max(out.values())
    run.log(0, **out)
    return out


# ── E0.2–E0.6: particles ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Built:
    problem: ParticleProblem
    k: int
    m: int
    #: ``(K, P, D)`` each condition's own positives — all the baseline sees.
    class_positives: Tensor
    y: FloatArray | None = None
    pos_weights: FloatArray | None = None
    h_p: FloatArray | None = None


def build_single_bank(cfg: Config, device: str) -> Built:
    l0, m = cfg["level0"], int(cfg["gen"]["M"])
    bank = make_single_bank(l0["K"], seed=l0["data_seed"], box=l0["y_box"], min_sep=l0["y_min_sep"])
    k = len(bank.y)
    h_p = bandwidths(cfg, bank.psi)
    log_w = caption_log_weights(bank.psi, bank.psi, h_p)
    if cfg["h_p"]["neighbors"] == "random":
        log_w = shuffle_neighbors(log_w, numpy_rng(l0["data_seed"], 7))
    pos = TargetSet(to_tensor(bank.y[None], device), to_tensor(log_w[:, None, :], device))

    exclude_self = bool(cfg["h_q"]["exclude_self"])
    rep = None
    if (ratio := float(cfg["h_q"]["ratio"])) > 0:
        log_wq = caption_log_weights(bank.psi, bank.psi, ratio * h_p)            # (K, K)
        per_particle = np.repeat(log_wq - math.log(m), m, axis=1)                 # (K, K·M)
        full = np.broadcast_to(per_particle[:, None, :], (k, m, k * m)).copy()
        if exclude_self:
            rows = np.arange(m)
            for c in range(k):
                full[c, rows, c * m + rows] = -np.inf
        rep = to_tensor(log_normalize(full), device)
    problem = ParticleProblem(pos=pos, rep_log_w=rep, exclude_self=exclude_self)
    return Built(problem, k, m, to_tensor(bank.y[:, None, :], device), y=bank.y, pos_weights=np.exp(log_w), h_p=h_p)


def build_gaussian_pair(cfg: Config, device: str) -> Built:
    l0, m = cfg["level0"], int(cfg["gen"]["M"])
    g = l0["gaussian_pair"]
    pts = make_gaussian_pair(tuple(g["mu1"]), tuple(g["mu2"]), g["s"], g["n_pos"], seed=l0["data_seed"])
    pos = TargetSet.uniform(to_tensor(pts, device))
    if float(cfg["h_q"]["ratio"]) > 0:
        raise NotImplementedError("E0.5 uses h_q = 0")
    partner = (1, 0) if cfg["negative"]["type"] == "counterfactual" else None
    problem = ParticleProblem(pos=pos, partner=partner, exclude_self=bool(cfg["h_q"]["exclude_self"]))
    return Built(problem, 2, m, to_tensor(pts, device))


def _single_bank_metrics(built: Built, traj: Trajectory) -> Metrics:
    assert built.y is not None and built.pos_weights is not None and built.h_p is not None
    a = M.assignment_matrix(traj.final, built.y)
    tv = M.tv_rows(a, built.pos_weights)
    self_ratio = np.diag(a)
    ne = n_eff(np.log(np.clip(built.pos_weights, 1e-300, None)))
    return {
        "tv_mean": float(tv.mean()), "tv_max": float(tv.max()),
        "self_mean": float(self_ratio.mean()), "self_min": float(self_ratio.min()),
        "own_dist_mean": float(M.own_distance(traj.final, built.y).mean()),
        "n_eff_mean": float(ne.mean()), "n_eff_min": float(ne.min()), "n_eff_max": float(ne.max()),
        "h_p_mean": float(built.h_p.mean()),
        "tv_per_condition": tv.tolist(), "self_per_condition": self_ratio.tolist(),
    }


def _gaussian_pair_metrics(cfg: Config, sigma: float, traj: Trajectory) -> Metrics:
    g, lam = cfg["level0"]["gaussian_pair"], float(cfg["negative"]["lambda"])
    out: Metrics = {"lambda": lam}
    pred = [prop_e_equilibrium(tuple(g["mu1"]), tuple(g["mu2"]), g["s"], sigma, lam),
            prop_e_equilibrium(tuple(g["mu2"]), tuple(g["mu1"]), g["s"], sigma, lam)]
    errs_m, errs_v = [], []
    for c, p in enumerate(pred):
        mean, var = M.moments(traj.final[c])
        out[f"mean_c{c}"], out[f"var_c{c}"] = mean.tolist(), var
        if p is not None and np.all(np.isfinite(mean)):
            out[f"pred_mean_c{c}"], out[f"pred_particle_var_c{c}"] = list(p.mean), p.particle_var
            errs_m.append(float(np.linalg.norm(mean - np.array(p.mean)) / np.linalg.norm(p.mean)))
            errs_v.append(abs(var + sigma**2 - p.smoothed_var) / p.smoothed_var)
    if errs_m:
        out["rel_err_mean"], out["rel_err_var"] = max(errs_m), max(errs_v)
    return out


def run_particles(run: Run) -> Metrics:
    cfg, device = run.config, default_device()
    l0 = cfg["level0"]
    sigma = frozen_sigma(run.exp)
    built = (build_single_bank if l0["problem"] == "single_bank" else build_gaussian_pair)(cfg, device)
    match cfg["field"]:
        case "affd":
            field = affd_field(built.problem, AffinityDrift(drift_spec(cfg, sigma)))
        case "release":
            field = release_field(built.class_positives, release_drift(cfg))
        case other:
            raise ValueError(f"unknown field {other!r}")
    system = ParticleSystem(field, eta=float(l0["eta"]))
    x0 = to_tensor(init_particles(built.k, built.m, seed=run.seed, box=l0["init_box"]), device)

    max_steps = min(int(l0["max_steps"]), SMOKE_MAX_STEPS) if run.smoke else int(l0["max_steps"])
    traj = system.run(
        x0, max_steps=max_steps, tol=float(l0["tol"]), stable_window=int(l0["stable_window"]),
        check_every=int(l0["check_every"]), diverge_bound=float(l0["diverge_bound"]), compile=bool(l0["compile"]),
        on_check=lambda t, row: run.log(t, **row),
    )
    last = traj.checks[-1]
    metrics: Metrics = {
        "sigma": sigma, "steps": traj.steps, "stable": traj.stable, "diverged": traj.diverged,
        "field": cfg["field"], "converged_at": traj.converged_at, "final_rms_v": last["rms_v"],
        "final_rms_v_plus": last["rms_v_plus"], "final_rms_v_minus": last["rms_v_minus"],
        "final_rms_v_neg": last["rms_v_neg"], "compiled": traj.compiled,
        "wall_time_s": traj.wall_time, "notes": traj.notes,
    }
    if l0["problem"] == "single_bank":
        metrics |= _single_bank_metrics(built, traj)
    else:
        metrics |= _gaussian_pair_metrics(cfg, sigma, traj)

    arrays: dict[str, np.ndarray] = {
        "final": traj.final,
        "snapshot_steps": np.array(sorted(traj.snapshots)),
        "snapshots": np.stack([traj.snapshots[t] for t in sorted(traj.snapshots)]),
        "check_steps": np.array([c["step"] for c in traj.checks]),
        "means": np.stack(traj.means),
    }
    if built.y is not None:
        arrays |= {"y": built.y, "pos_weights": built.pos_weights, "h_p": built.h_p}
    run.save_arrays(**arrays)
    return metrics


RUNNERS: Mapping[str, Any] = {"lemma1": run_lemma1, "particles": run_particles}
