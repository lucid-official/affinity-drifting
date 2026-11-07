"""Level 1 figures: FC.1–FC.4 and F1.1–F1.7 (plan §10.2, §10.4).

Every panel is drawn from seed 0's saved arrays (or from the config alone, for fields), never from hand-picked
samples. Arm order and colour come from ``style.py``; the baseline ``drifting`` is in every comparison figure
that defines it, and the figure's ``.md`` says why where it is not.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

from affdrift.config import Experiment
from affdrift.core.bandwidth import n_eff
from affdrift.core.kernels import caption_log_weights
from affdrift.eval.figures.level0 import Built, Runs, _grid, _seed0, _sources, fc1
from affdrift.eval.figures.style import BASELINE, arm_color, ordered
from affdrift.eval.figures.writer import Panel
from affdrift.eval.stats import bootstrap_ci
from affdrift.runtime import ROOT

PROBE_COLS = (0, 1, 6, 12)  # two easy, one hard, one tail probe condition


def _bank(cfg: Mapping[str, Any]):  # local import: figures never import levels at module load
    from affdrift.levels.level1 import bank_of

    return bank_of(cfg)


def _arms(runs: Runs, opts: Mapping[str, Any]) -> list[str]:
    return ordered([a for a in opts.get("arms", runs) if a in runs])


def _density_contours(ax: plt.Axes, bank: Any, c: np.ndarray, lim: float = 2.4) -> None:
    g = np.linspace(-lim, lim, 120)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], -1)[None]
    lp = bank.log_density(pts, c[None])[0].reshape(gx.shape)
    ax.contour(gx, gy, np.exp(lp), levels=4, colors="0.7", linewidths=0.6)


def _cond_mean(bank: Any, c: np.ndarray) -> np.ndarray:
    from affdrift.data.toy2d import conditional_mean

    return conditional_mean(c[None], mix_beta=bank.mix_beta)[0]


def _square(ax: plt.Axes, lim: float = 2.4) -> None:
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[])
    ax.set_aspect("equal")


# ── common ──────────────────────────────────────────────────────────────────


def fc2(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    """N_eff over 1,000 bank conditions for the arm's caption weights."""
    from affdrift.levels.level1 import bandwidths

    rec = _seed0(runs, opts["arm"])
    bank = _bank(rec.config)
    idx = np.random.default_rng(0).choice(len(bank.c), 1000, replace=False)
    h = bandwidths(rec.config, bank)[idx]
    ne = n_eff(caption_log_weights(bank.c[idx], bank.c, h))
    n_star = rec.config["h_p"]["n_star"]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.hist(ps[0].data, bins=30, color=ps[0].color)
        ax.axvline(n_star, color="tab:gray", ls="--", lw=0.8, label=f"N* = {n_star}")
        ax.axvline(1, color=arm_color(BASELINE), lw=1.5, label="drifting (baseline): N_eff = 1")
        ax.set(xlabel="N_eff(c)", ylabel="conditions", title=f"FC.2 · {exp.id} · {ps[0].label}")
        ax.legend(fontsize=7)
        return fig

    return Built(render, [Panel.of(opts["arm"], ne)], blind=False, sources={rec.name: rec.config_hash},
                 extra=f"- N_eff over 1,000 bank conditions: min {ne.min():.2f}, median {np.median(ne):.2f}, max {ne.max():.2f}\n")


def fc3(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    """Probe samples (16 ε per condition) at 0, ¼, ½ and all of training, four probe conditions."""
    arms = _arms(runs, opts)
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    c = np.array([p["c"] for p in probe["conditions"]])
    panels = [Panel.of(a, (_seed0(runs, a).arrays["probe_steps"], _seed0(runs, a).arrays["probe_samples"])) for a in arms]
    bank = _bank(_seed0(runs, arms[0]).config)
    cmap = plt.get_cmap("tab10")

    def render(ps: Sequence[Panel]) -> Figure:
        n_ck = len(ps[0].data[0])
        fig, axes = plt.subplots(len(ps), n_ck, figsize=(2.3 * n_ck, 2.3 * len(ps)), squeeze=False)
        for row, p in zip(axes, ps):
            steps, samples = p.data
            for ax, t, s in zip(row, steps, samples):
                for j, k in enumerate(PROBE_COLS):
                    ax.scatter(*bank.modes(c[k:k + 1])[0].T, marker="+", color=cmap(j), s=25, lw=0.8)
                    ax.scatter(s[k, :, 0], s[k, :, 1], s=5, color=cmap(j), alpha=0.8)
                _square(ax)
                ax.set_title(f"{p.label} · step {int(t)}", color=p.color, fontsize=7)
        fig.suptitle(f"FC.3 · {exp.id} · probe conditions {PROBE_COLS} (colours), 16 ε each; + = true modes (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


def fc4(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    """ε₀ → ε₁ in 16 steps: a mixture jumps between modes, a blob deforms smoothly."""
    arms = _arms(runs, opts)
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    c = np.array([p["c"] for p in probe["conditions"]])
    panels = [Panel.of(a, _seed0(runs, a).arrays["fc4"]) for a in arms]
    bank = _bank(_seed0(runs, arms[0]).config)

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(len(ps), len(PROBE_COLS), figsize=(2.4 * len(PROBE_COLS), 2.4 * len(ps)), squeeze=False)
        for row, p in zip(axes, ps):
            for ax, k in zip(row, PROBE_COLS):
                _density_contours(ax, bank, c[k])
                path = p.data[k]
                ax.plot(path[:, 0], path[:, 1], "-", color="0.6", lw=0.6)
                ax.scatter(path[:, 0], path[:, 1], c=np.linspace(0, 1, len(path)), cmap="viridis", s=14, zorder=3)
                _square(ax)
                ax.set_title(f"{p.label} · probe {k}", color=p.color, fontsize=7)
        fig.suptitle(f"FC.4 · {exp.id} · ε interpolation strips (colour = t from ε₀ to ε₁, seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


# ── F1.1 – F1.3 ─────────────────────────────────────────────────────────────


def f11(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = _arms(runs, opts)
    panels = [Panel.of(a, (_seed0(runs, a).arrays["fake_heldout"], _seed0(runs, a).arrays["c_heldout"])) for a in arms]
    bank = _bank(_seed0(runs, arms[0]).config)

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(len(ps), 4, figsize=(9.6, 2.4 * len(ps)), squeeze=False)
        for row, p in zip(axes, ps):
            fake, cs = p.data
            for j, ax in enumerate(row):
                _density_contours(ax, bank, cs[j])
                ax.scatter(fake[j, :, 0], fake[j, :, 1], s=1.5, color=p.color, alpha=0.5)
                ax.scatter(*_cond_mean(bank, cs[j]), marker="*", s=140, color="gold", edgecolors="black", zorder=4)
                _square(ax)
                ax.set_title(f"{p.label} · held-out c{j}", color=p.color, fontsize=7)
        fig.suptitle(f"F1.1 · {exp.id} · 512 samples per held-out condition; contours = p(x|c), ★ = E[x|c] (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


def _attraction_field(cfg: Mapping[str, Any], sigma: float | list[float], c_idx: int, n: int = 25):
    """V⁺(x | c) on a grid for one bank condition, through the same AffinityDrift code as training."""
    from affdrift.core.drift import AffinityDrift, DriftSpec, MultiScaleAffinityDrift, TargetSet
    from affdrift.levels.level1 import Positives, bandwidths
    from affdrift.runtime import numpy_rng, to_numpy, to_tensor

    bank = _bank(cfg)
    g = np.linspace(-2.4, 2.4, n)
    grid = np.stack([a.ravel() for a in np.meshgrid(g, g)], -1)[None]
    sig = sigma if isinstance(sigma, list) else [sigma]
    from affdrift.core.kernels import XKernel

    spec = DriftSpec(sigma=sig[0], kernel=XKernel(cfg["x_kernel"]["kind"]), repulsion=False, drift_norm=False)
    drift = MultiScaleAffinityDrift(spec, sig) if len(sig) > 1 else AffinityDrift(spec)
    pos = Positives(cfg, bank, "cpu", bandwidths(cfg, bank), numpy_rng(0, 0))
    idx = np.array([c_idx])
    v = to_numpy(drift.field(to_tensor(grid, "cpu"), pos(idx, bank.c[idx])).v)[0]
    w = np.exp(to_numpy(pos.log_weights(idx))[0, 0])
    return {"grid": grid[0], "v": v, "n": n, "w": w}


def f12(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = [a for a in _arms(runs, opts) if a != BASELINE]
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    rec0 = _seed0(runs, arms[0])
    bank = _bank(rec0.config)
    c_idx = int(cKDTree(bank.c).query(np.array(probe["conditions"][0]["c"]))[1])
    panels = [Panel.of(a, _attraction_field(_seed0(runs, a).config, _seed0(runs, a).metrics["sigma"], c_idx)) for a in arms]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(1, len(ps), figsize=(3.4 * len(ps), 3.4), squeeze=False)
        for ax, p in zip(axes[0], ps):
            d = p.data
            mag = np.linalg.norm(d["v"], axis=-1, keepdims=True)
            u = d["v"] / np.maximum(mag, 1e-12)
            _density_contours(ax, bank, bank.c[c_idx])
            ax.quiver(d["grid"][:, 0], d["grid"][:, 1], u[:, 0], u[:, 1], color=p.color, scale=30, width=0.004)
            top = np.argsort(d["w"])[-64:]
            ax.scatter(*bank.y[top].T, s=4 + 300 * d["w"][top] / d["w"].max(), facecolors="none", edgecolors="black", lw=0.5)
            _square(ax)
            ax.set_title(f"{p.label}", color=p.color)
        fig.suptitle(f"F1.2 · {exp.id} · direction of V⁺(x|c) for bank condition {c_idx} (nearest to probe 0); "
                     "circles = borrowed positives ∝ weight")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms),
                 extra="- baseline `drifting` 없음: release의 attraction은 생성 샘플 위치에서만 정의되고(이중 softmax), 격자 위의 "
                       "V⁺가 없다. 같은 조건에서 positive가 자기 것 하나뿐이므로 single 패널과 같은 한 점을 향한다.\n")


def f13(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = _arms(runs, opts)
    panels = [Panel.of(a, (_seed0(runs, a).arrays["eps_map"], _seed0(runs, a).arrays["eps_map_c"])) for a in arms]
    bank = _bank(_seed0(runs, arms[0]).config)

    def render(ps: Sequence[Panel]) -> Figure:
        fig, axes = plt.subplots(len(ps), 2, figsize=(5.6, 2.6 * len(ps)), squeeze=False)
        for row, p in zip(axes, ps):
            out, cs = p.data
            for ax, o, cc in zip(row, out, cs):
                n = int(np.sqrt(len(o)))
                mode = np.argmin(((o[:, None, :] - bank.modes(cc[None])[0][None]) ** 2).sum(-1), axis=1)
                ax.imshow(mode.reshape(n, n), cmap="Set2", vmin=0, vmax=7, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5))
                ax.set(xticks=[], yticks=[])
                ax.set_title(f"{p.label} · c = ({cc[0]:.2f}, {cc[1]:.2f})", color=p.color, fontsize=7)
        fig.suptitle(f"F1.3 · {exp.id} · ε₁ × ε₂ slice (other ε dims 0), colour = nearest true mode of f(ε, c) (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms))


# ── F1.4 / F1.5: the capacity phase diagram ─────────────────────────────────


def _axis(opts: Mapping[str, Any]) -> tuple[list[str], list[str], str]:
    if opts.get("axis") == "lipschitz":
        return ["L64", "L16", "L4", "L1"], ["1/64", "1/16", "1/4", "1"], "1/L (smaller L = coarser)"
    return ["d0", "d64", "d16", "d4"], ["0", "1/64", "1/16", "1/4"], "δ"


H_ROWS = ["h0", "h0.5", "h1", "h2", "h4"]


def _cell(runs: Runs, arm: str, metric: str) -> float:
    return float(np.mean([r.metrics[metric] for r in runs[arm].values()])) if arm in runs else np.nan


def f14(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    cols, labels, xlabel = _axis(opts)
    w2 = np.array([[_cell(runs, f"{c}_{h}", "w2_train") for c in cols] for h in H_ROWS])
    mci = np.array([[_cell(runs, f"{c}_{h}", "mci_train") for c in cols] for h in H_ROWS])

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        im = ax.imshow(w2, cmap="viridis_r", origin="lower", aspect="auto")
        for i in range(len(H_ROWS)):
            for j in range(len(cols)):
                ax.text(j, i, f"{w2[i, j]:.2f}\nMCI {mci[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if w2[i, j] > np.nanmean(w2) else "black")
        ax.set_xticks(range(len(cols)), labels)
        ax.set_yticks(range(len(H_ROWS)), ["0 (single)", "0.5", "1 (AffD B0)", "2", "4"])
        ax.set(xlabel=xlabel, ylabel="h_p / h*")
        fig.colorbar(im, ax=ax, label="W2 (training c, seed mean)")
        fig.suptitle(f"F1.4 · {exp.id} · phase diagram")
        return fig

    sources = {r.name: r.config_hash for a in runs for r in runs[a].values()}
    return Built(render, [Panel.of("AffD", w2)], blind=False, sources=sources,
                 extra="- baseline `drifting`은 한 셀(B0 용량)만 있어 heatmap에 넣지 않는다. 값은 report.md에 있다.\n")


def f15(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    cols, labels, xlabel = _axis(opts)
    series = {"single": "h0", "AffD": "h1"}
    data = {}
    for name, h in series.items():
        cis = [bootstrap_ci([r.metrics["mci_train"] for r in runs[f"{c}_{h}"].values()]) if f"{c}_{h}" in runs else None
               for c in cols]
        data[name] = cis
    base = [r.metrics["mci_train"] for r in runs[BASELINE].values()] if BASELINE in runs else None
    base_col = 1 if opts.get("axis") == "lipschitz" else 2  # the baseline runs at the reference capacity

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        x = np.arange(len(cols))
        for p in ps:
            cis = p.data
            m = [c.mean if c else np.nan for c in cis]
            lo = [c.lo if c else np.nan for c in cis]
            hi = [c.hi if c else np.nan for c in cis]
            ax.plot(x, m, "-o", color=p.color, label=p.label)
            ax.fill_between(x, lo, hi, color=p.color, alpha=0.2)
        if base:
            ax.errorbar([base_col], [np.mean(base)], yerr=[[np.mean(base) - min(base)], [max(base) - np.mean(base)]],
                        fmt="s", color=arm_color(BASELINE), label="drifting (baseline)")
        ax.set_xticks(x, labels)
        ax.set(xlabel=xlabel, ylabel="MCI (training c)")
        ax.legend(fontsize=7)
        fig.suptitle(f"F1.5 · {exp.id} · mean-collapse index vs capacity (95% CI over seeds)")
        return fig

    panels = [Panel(arm=name, label=name, color=arm_color(name), data=cis) for name, cis in data.items()]
    sources = {r.name: r.config_hash for a in runs for r in runs[a].values()}
    return Built(render, panels, blind=True, sources=sources)


# ── F1.6 / F1.7 ─────────────────────────────────────────────────────────────


def f16(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = _arms(runs, opts)
    bank = _bank(_seed0(runs, arms[0]).config)
    tree = cKDTree(bank.y)
    panels = [Panel.of(a, tree.query(_seed0(runs, a).arrays["fake_train"].reshape(-1, 2))[0]) for a in arms]
    eps = _seed0(runs, arms[0]).metrics["memorization_eps"]

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        bins = np.geomspace(1e-4, 1.0, 50)
        for p in ps:
            ax.hist(p.data, bins=bins, histtype="step", color=p.color, label=p.label, lw=1.3)
        ax.axvline(eps, color="0.4", ls="--", lw=0.8, label="memorisation ε")
        ax.set(xscale="log", xlabel="distance to nearest training positive", ylabel="samples")
        ax.legend(fontsize=7)
        fig.suptitle(f"F1.6 · {exp.id} · 16 training conditions × 512 samples (seed 0)")
        return fig

    return Built(render, panels, blind=True, sources=_sources(runs, arms),
                 extra="- 배정 heatmap(plan 판독의 둘째 부분)은 조건이 연속이라 L1에서는 σ 평활 TV(`nw_tv_train`)로 대신 보고한다(prereg E1.5).\n")


def f17(exp: Experiment, runs: Runs, opts: Mapping[str, Any]) -> Built:
    arms = list(opts["arms"])
    mult = [float(a.removeprefix("hx")) for a in arms]
    w2 = [bootstrap_ci([r.metrics["w2_heldout"] for r in runs[a].values()]) for a in arms]
    ve = [float(np.mean([r.metrics["raw_drift_energy"] for r in runs[a].values()])) for a in arms]
    fixed = float(np.mean([r.metrics["w2_heldout"] for r in runs["fixed"].values()])) if "fixed" in runs else None
    base = float(np.mean([r.metrics["w2_heldout"] for r in runs[BASELINE].values()])) if BASELINE in runs else None

    def render(ps: Sequence[Panel]) -> Figure:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.errorbar(mult, [c.mean for c in w2], yerr=[[c.mean - c.lo for c in w2], [c.hi - c.mean for c in w2]],
                    fmt="-o", color=arm_color("AffD"), label="W2 (held-out)")
        ax.axvline(1.0, color="0.5", ls=":", label="N_eff rule (h*)")
        if fixed is not None:
            ax.axhline(fixed, color="tab:green", ls="--", lw=0.8, label="fixed h (median h*)")
        if base is not None:
            ax.axhline(base, color=arm_color(BASELINE), ls="-.", lw=0.8, label="drifting (baseline)")
        ax.set(xscale="log", xlabel="h_p / h*", ylabel="W2 (held-out)")
        ax2 = ax.twinx()
        ax2.plot(mult, ve, "-s", color="tab:red", label="raw ‖V‖² (diagnostic)")
        ax2.set(yscale="log", ylabel="raw ‖V‖²")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper center")
        fig.suptitle(f"F1.7 · {exp.id} · W2 and ‖V‖² against h_p (seed mean, 95% CI)")
        return fig

    sources = {r.name: r.config_hash for a in runs for r in runs[a].values()}
    return Built(render, [Panel.of("AffD", w2)], blind=False, sources=sources)


BUILDERS: Mapping[str, Callable[[Experiment, Runs, Mapping[str, Any]], Built]] = {
    "FC.1": fc1, "FC.2": fc2, "FC.3": fc3, "FC.4": fc4,
    "F1.1": f11, "F1.2": f12, "F1.3": f13, "F1.4": f14, "F1.5": f15, "F1.6": f16, "F1.7": f17,
}
