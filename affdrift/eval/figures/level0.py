"""Level 0 figures: FC.1, FC.2 and F0.1–F0.5 (plan §10.2–10.3).

Each builder takes the experiment, its complete runs and the options from
``figures:`` in ``experiment.yaml``, and returns what
:class:`~affdrift.eval.figures.writer.FigureWriter` saves.  Panels are drawn
from seed 0 unless a builder says it averages seeds; nothing is hand-picked.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm

from affdrift.config import Experiment
from affdrift.core.bandwidth import n_eff
from affdrift.core.drift import AffinityDrift, TargetSet
from affdrift.eval.figures.style import BASELINE, arm_color
from affdrift.eval.figures.writer import Panel, Render
from affdrift.eval.metrics import assignment_matrix
from affdrift.runtime import ROOT, to_numpy, to_tensor
from affdrift.tracking.store import RunRecord

type Runs = Mapping[str, Mapping[int, RunRecord]]


@dataclass(frozen=True, slots=True)
class Built:
    render: Render
    panels: list[Panel]
    blind: bool
    sources: dict[str, str]
    extra: str = ""


def _probe() -> dict[str, Any]:
    return json.loads((ROOT / "probe" / "probe_L0.json").read_text())


def _seed0(runs: Runs, arm: str) -> RunRecord:
    return runs[arm][min(runs[arm])]


def _sources(runs: Runs, arms: Sequence[str]) -> dict[str, str]:
    return {r.name: r.config_hash for a in arms for r in runs[a].values()}


def _grid(n: int, ncols: int = 4, size: tuple[float, float] = (3.0, 2.6), **kw: Any) -> tuple[Figure, np.ndarray]:
    ncols = min(n, ncols)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


# ── FC.1 ────────────────────────────────────────────────────────────────────


def fc1(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts.get("arms", runs))
    panels = [Panel.of(a, [r.log for _, r in sorted(runs[a].items())]) for a in arms]
    styles = {"rms_v": ("-", "V"), "rms_v_plus": ("--", "V⁺"), "rms_v_minus": (":", "V⁻"), "rms_v_neg": ("-.", "V_neg")}

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = _grid(len(ps), sharey=True)
        for ax, p in zip(axes.flat, ps):
            for logs in p.data:
                steps = [row["step"] for row in logs]
                for key, (ls, label) in styles.items():
                    vals = np.array([row.get(key, np.nan) for row in logs], dtype=float)
                    if np.nanmax(vals, initial=0) <= 0:
                        continue
                    ax.plot(steps, vals, ls, color=p.color, lw=1.3 if key == "rms_v" else 0.8, alpha=0.75,
                            label=label if logs is p.data[0] else None)
            ax.set(yscale="log", xscale="symlog", title=p.label, xlabel="step")
        axes.flat[0].set_ylabel("RMS ‖·‖ (raw)")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"FC.1 · {exp.id} · per-term field norms, all seeds")
        return fig

    return Built(render, panels, blind=False, sources=_sources(runs, arms))


# ── FC.2 ────────────────────────────────────────────────────────────────────


def fc2(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    rec = _seed0(runs, opts["arm"])
    ne = n_eff(np.log(np.clip(rec.arrays["pos_weights"], 1e-300, None)))
    n_star = rec.config["h_p"]["n_star"]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.hist(ps[0].data, bins=20, color=ps[0].color)
        ax.axvline(n_star, color="tab:gray", ls="--", lw=0.8, label=f"N* = {n_star}")
        if BASELINE in exp.arms:
            ax.axvline(1, color=arm_color(BASELINE), lw=1.5, label="drifting (baseline): N_eff = 1")
        ax.set(xlabel="N_eff(c_k)", ylabel="conditions", title=f"FC.2 · {exp.id} · {ps[0].label}")
        ax.legend()
        return fig

    return Built(render, [Panel.of(opts["arm"], ne)], blind=False, sources={rec.name: rec.config_hash},
                 extra=f"- N_eff: min {ne.min():.2f}, mean {ne.mean():.2f}, max {ne.max():.2f}\n")


# ── F0.1 ────────────────────────────────────────────────────────────────────


def f01(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    panels = []
    for a in arms:
        r = _seed0(runs, a)
        steps, snaps = r.arrays["snapshot_steps"], r.arrays["snapshots"]
        end = int(steps[-1])
        mid = int(np.argmin(np.abs(steps - end / 2)))
        panels.append(Panel.of(a, {"snaps": [(0, snaps[0]), (int(steps[mid]), snaps[mid]), (end, snaps[-1])],
                                   "y": r.arrays["y"]}))
    cmap = plt.get_cmap("viridis")

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(len(ps), 3, figsize=(9, 3 * len(ps)), squeeze=False, sharex=True, sharey=True)
        for row, p in zip(axes, ps):
            y = p.data["y"]
            colors = cmap(np.linspace(0, 1, len(y)))
            for ax, (t, x) in zip(row, p.data["snaps"]):
                for k in range(len(y)):
                    ax.scatter(x[k, :, 0], x[k, :, 1], s=1.5, color=colors[k], alpha=0.5, lw=0)
                ax.scatter(y[:, 0], y[:, 1], s=28, c=colors, edgecolors="black", linewidths=0.8, marker="X", zorder=3)
                ax.set_title(f"{p.label} · t = {t}", color=p.color)
                ax.set_aspect("equal")
        fig.suptitle(f"F0.1 · {exp.id} · particles coloured by condition ψ_k; ✕ = y_k (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


# ── F0.2 / F0.3 ─────────────────────────────────────────────────────────────


def _mean_assignment(runs: Runs, arm: str) -> tuple[np.ndarray, np.ndarray]:
    recs = list(runs[arm].values())
    a = np.mean([assignment_matrix(r.arrays["final"], r.arrays["y"]) for r in recs], axis=0)
    return a, recs[0].arrays["pos_weights"]


def _heat(ax: plt.Axes, m: np.ndarray, title: str, vmax: float, color: str = "black") -> Any:
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, origin="upper")
    ax.set(xlabel="positive j", ylabel="condition k")
    ax.set_title(title, color=color)
    return im


def f02(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    panels = [Panel.of(a, _mean_assignment(runs, a)) for a in arms]

    def render(ps: Sequence[Panel]) -> Figure:
        vmax = max(max(p.data[0].max(), p.data[1].max()) for p in ps)
        fig, axes = plt.subplots(len(ps), 3, figsize=(11, 3.4 * len(ps)), squeeze=False)
        for row, p in zip(axes, ps):
            a_, w_ = p.data
            tv = 0.5 * np.abs(a_ - w_).sum(1).mean()
            _heat(row[0], a_, f"{p.label} · assignment A (seed mean)", vmax, p.color)
            im = _heat(row[1], w_, "w̃_j(c_k) (AffD B0 prediction)", vmax)
            fig.colorbar(im, ax=row[:2], shrink=0.8)
            d = row[2].imshow(a_ - w_, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2)
            row[2].set_title(f"{p.label} · A − w̃ (mean TV {tv:.3f})", color=p.color)
            fig.colorbar(d, ax=row[2], shrink=0.8)
        fig.suptitle(f"F0.2 · {exp.id} · assignment vs NW weights")
        return fig

    return Built(render, panels, blind=False, sources=_sources(runs, arms))


def f03(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    panels = [Panel.of(a, _mean_assignment(runs, a)[0]) for a in arms]
    w = _mean_assignment(runs, arms[0])[1]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(1, len(ps) + 1, figsize=(3.1 * (len(ps) + 1), 3.2))
        vmax = max(max(p.data.max() for p in ps), w.max())
        for ax, p in zip(axes, ps):
            im = _heat(ax, p.data, p.label, vmax, p.color)
        _heat(axes[-1], w, "reference: w̃ (NW mixture)", vmax)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle(f"F0.3 · {exp.id} · final assignment matrices (seed mean)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


# ── F0.4 ────────────────────────────────────────────────────────────────────


def f04(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    panels = []
    for a in arms:
        r = _seed0(runs, a)
        panels.append(Panel.of(a, {"final": r.arrays["final"], "means": r.arrays["means"],
                                   "steps": r.arrays["check_steps"], "m": r.metrics}))

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = _grid(len(ps), ncols=3, size=(3.4, 2.7))
        for ax, p in zip(axes.flat, ps):
            m = p.data["m"]
            if "pred_mean_c0" in m:
                x = p.data["final"][0, :, 0]
                ax.hist(x, bins=60, density=True, color=p.color, alpha=0.6, label="particles c, x₁")
                mu, var = m["pred_mean_c0"][0], m["pred_particle_var_c0"]
                grid = np.linspace(x.min() - 1, x.max() + 1, 300)
                ax.plot(grid, norm.pdf(grid, mu, np.sqrt(var)), color="black", lw=1, label="Prop E")
                ax.set_title(f"{p.label} · err μ {m['rel_err_mean']:.3f}, var {m['rel_err_var']:.3f}", color=p.color)
            else:
                ax.plot(p.data["steps"], np.linalg.norm(p.data["means"][:, 0], axis=-1), color=p.color)
                ax.set(xscale="symlog", yscale="log", xlabel="step", ylabel="‖mean of c‖")
                ax.set_title(f"{p.label} · diverged = {m['diverged']}", color=p.color)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"F0.4 · {exp.id} · equilibrium vs closed form (seed 0); λ ≥ 1: mean trajectory")
        return fig

    return Built(render, panels, blind=False, sources=_sources(runs, arms))


# ── F0.5 ────────────────────────────────────────────────────────────────────


def _grid_field(rec: RunRecord, k: int, sigma: float, n: int = 25) -> dict[str, np.ndarray]:
    """V(·|c_k) on a grid, from the run's final particles, through the same AffinityDrift field."""
    from affdrift.levels.level0 import build_single_bank, drift_spec  # local: figures never import levels at load

    cfg = rec.config
    final, y = rec.arrays["final"], rec.arrays["y"]
    k_all, m = final.shape[:2]
    lim = np.abs(final).max() * 1.1
    gx, gy = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    g = np.stack([gx.ravel(), gy.ravel()], -1)[None]
    built = build_single_bank(cfg, "cpu")
    log_wp = to_numpy(built.problem.pos.log_w)[k : k + 1]
    pos = TargetSet(to_tensor(y[None], "cpu"), to_tensor(log_wp, "cpu"))
    if built.problem.rep_log_w is None:
        gen = TargetSet.uniform(to_tensor(final[k][None], "cpu"))
    else:
        from affdrift.core.kernels import caption_log_weights

        psi = np.linspace(0, 1, k_all)[:, None]
        log_wq = caption_log_weights(psi[k : k + 1], psi, cfg["h_q"]["ratio"] * built.h_p[k])
        gen = TargetSet(to_tensor(final.reshape(1, -1, 2), "cpu"),
                        to_tensor(np.repeat(log_wq - np.log(m), m, axis=1)[:, None, :], "cpu"))
    v = to_numpy(AffinityDrift(drift_spec(cfg, sigma)).field(to_tensor(g, "cpu"), pos, gen).v)[0]
    return {"g": g[0], "v": v, "n": n, "x": final[k], "y": y, "w": np.exp(log_wp[0, 0])}


def _particle_field(rec: RunRecord, k: int) -> dict[str, np.ndarray]:
    """The baseline's field at its own final particles: the release field exists only at generated samples."""
    from affdrift.levels.level0 import build_single_bank, release_drift

    final, y = rec.arrays["final"], rec.arrays["y"]
    built = build_single_bank(rec.config, "cpu")
    v = to_numpy(release_drift(rec.config).drift(to_tensor(final, "cpu"), built.class_positives))
    return {"g": final[k], "v": v[k], "n": None, "x": final[k], "y": y, "w": np.eye(len(y))[k]}


def f05(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    k = int(opts.get("condition", _probe()["quiver_condition"]))
    panels = [
        Panel.of(a, _particle_field(r, k) if r.config["field"] == "release" else _grid_field(r, k, r.metrics["sigma"]))
        for a in arms if (r := _seed0(runs, a))
    ]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(1, len(ps), figsize=(4.4 * len(ps), 4), squeeze=False)
        for ax, p in zip(axes[0], ps):
            d = p.data
            mag = np.linalg.norm(d["v"], axis=-1)
            u = d["v"] / np.maximum(mag[:, None], 1e-12)
            if d["n"] is not None:
                ax.contourf(d["g"][:, 0].reshape(d["n"], d["n"]), d["g"][:, 1].reshape(d["n"], d["n"]),
                            np.log10(mag + 1e-12).reshape(d["n"], d["n"]), levels=20, cmap="Greys", alpha=0.6)
            else:
                ax.text(0.02, 0.98, "field at own particles\n(release V is defined only there)",
                        transform=ax.transAxes, va="top", fontsize=7)
            ax.quiver(d["g"][:, 0], d["g"][:, 1], u[:, 0], u[:, 1], color=p.color, scale=35, width=0.003)
            ax.scatter(d["x"][:, 0], d["x"][:, 1], s=2, color="tab:green", alpha=0.6, label="particles of c_k")
            ax.scatter(d["y"][:, 0], d["y"][:, 1], s=8 + 400 * d["w"], facecolors="none", edgecolors="black",
                       label="y_j (size ∝ w̃_j)")
            ax.set_aspect("equal")
            ax.set_title(f"{p.label} · condition k = {k}", color=p.color)
        axes[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"F0.5 · {exp.id} · direction of V(x|c_k) at the final state; shade = log₁₀‖V‖ (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


BUILDERS: Mapping[str, Callable[[Experiment, Runs, Mapping[str, Any]], Built]] = {
    "FC.1": fc1, "FC.2": fc2, "F0.1": f01, "F0.2": f02, "F0.3": f03, "F0.4": f04, "F0.5": f05,
}
