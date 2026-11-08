"""The live preprint: numbers, tables and figures regenerated from experiment results, then compiled.

``paper/`` holds hand-written TeX (``main.tex``, ``sections/``). Everything under ``paper/generated/`` is written
here from ``experiments/*/*/aggregate.json``, the committed run arrays and the analysis files — never typed by hand.
A quantity whose experiment has not been aggregated renders as a grey ``[pending]``; nothing is extrapolated.

:func:`build` runs after every ``scripts/aggregate.py`` (``pipeline.aggregate``), so the PDF follows the results.
"""

import json
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from affdrift.config import EXPERIMENTS_ROOT, Experiment  # noqa: E402
from affdrift.eval.figures.style import BASELINE, arm_color  # noqa: E402
from affdrift.runtime import ROOT, git_commit  # noqa: E402
from affdrift.tracking.store import complete  # noqa: E402

PAPER = ROOT / "paper"
GEN = PAPER / "generated"
FIG = GEN / "figures"
PENDING = r"\pending{}"

COLUMN, FULL = 3.25, 6.875  # CVPR column and text widths, inches
RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
}
#: Display names in the paper: literature terms only, never internal arm labels (CLAUDE.md §8.10).
LABEL = {BASELINE: "Drifting", "single": r"One positive ($h\to 0$)", "AffD": "Affinity Drifting",
         "K4": "Oracle (16 samples)", "K1": "Constant sample kernel", "K2": "Random neighbors",
         "K3": "Noisy positive"}

#: Every pre-registered experiment, described in literature terms for the paper's status table.
DESCRIBE = {
    "E0.1": ("Particles", "Mean shift equals the scaled score difference"),
    "E0.2": ("Particles", r"Negatives with $h'=h$ memorize the positives"),
    "E0.3": ("Particles", r"Negatives with $h'=0$: Nadaraya--Watson equilibrium"),
    "E0.4": ("Particles", r"Equilibrium moves monotonically with $h'$"),
    "E0.5": ("Particles", "Cross-condition negatives: closed-form equilibrium"),
    "E0.5b": ("Particles", "Same, under a Gaussian moment closure"),
    "E0.6": ("Particles", r"Joint vs.\ separate softmax: same equilibrium"),
    "E1.1": ("2-D toy", r"One positive, no negatives: $\ell_2$ regression"),
    "E1.2": ("2-D toy", "One positive regresses to the conditional mean"),
    "E1.3": ("2-D toy", "Affinity Drifting restores the mixture; beats ablations"),
    "E1.4": ("2-D toy", r"Gap grows as resolution in $c$ coarsens (quantization)"),
    "E1.4b": ("2-D toy", r"Same, with a Lipschitz constraint on $c$"),
    "E1.5": ("2-D toy", r"$h'$ selects the equilibrium of a trained generator"),
    "E1.6": ("2-D toy", r"Bandwidth: U-shaped error, minimum near $n^\star$"),
}


# ── reading results ─────────────────────────────────────────────────────────


def _aggregate(eid: str) -> dict[str, Any] | None:
    try:
        path = Experiment.load(eid).dir / "aggregate.json"
    except FileNotFoundError:
        return None
    return json.loads(path.read_text()) if path.exists() else None


def _ci(eid: str, arm: str, metric: str) -> dict[str, float] | None:
    agg = _aggregate(eid)
    return None if agg is None else agg.get("summary", {}).get(arm, {}).get(metric)


def _fmt(x: float, digits: int = 3) -> str:
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        mant, exp = f"{x:.1e}".split("e")
        return rf"{mant}\times10^{{{int(exp)}}}"
    return f"{x:.{digits}g}"


def _num(eid: str, arm: str, metric: str, *, digits: int = 3, ci: bool = False) -> str:
    c = _ci(eid, arm, metric)
    if c is None:
        return PENDING
    s = _fmt(c["mean"], digits)
    return rf"${s}$" if not ci else rf"${s}$ {{\scriptsize$[{_fmt(c['lo'], digits)}, {_fmt(c['hi'], digits)}]$}}"


def _num_one_minus(eid: str, arm: str, metric: str, *, digits: int = 3, ci: bool = False) -> str:
    """1 − metric with its interval flipped (total variance ratio from the stored mean-collapse index)."""
    c = _ci(eid, arm, metric)
    if c is None:
        return PENDING
    s = _fmt(1 - c["mean"], digits)
    return rf"${s}$" if not ci else rf"${s}$ {{\scriptsize$[{_fmt(1 - c['hi'], digits)}, {_fmt(1 - c['lo'], digits)}]$}}"


def _macro_id(eid: str) -> str:
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
             "7": "Seven", "8": "Eight", "9": "Nine", "b": "B", "c": "C", ".": ""}
    return "".join(words.get(ch, ch) for ch in eid)


# ── numbers.tex ─────────────────────────────────────────────────────────────


def numbers() -> dict[str, str]:
    out: dict[str, str] = {}
    for exp in Experiment.all():
        agg = _aggregate(exp.id)
        out[f"Verdict{_macro_id(exp.id)}"] = PENDING if agg is None else rf"\textsc{{{agg['judgement']['verdict'].lower()}}}"
    n = out
    n["EZeroOneErr"] = _num("E0.1", "check", "max_err_f64_cpu", digits=2)
    n["EZeroThreeTV"] = _num("E0.3", "b0", "tv_mean", ci=True)
    n["EZeroThreeDriftingSelf"] = _num("E0.3", BASELINE, "self_mean")
    smoothed = EXPERIMENTS_ROOT / "L0" / "analysis" / "smoothed.json"
    if smoothed.exists():
        s = json.loads(smoothed.read_text())
        vals = s["E0.3: smoothed TV(p̂, q̂) per seed"]["b0"]
        n["EZeroThreeSmoothTV"] = rf"${_fmt(float(np.mean(vals)), 2)}$"
    else:
        n["EZeroThreeSmoothTV"] = PENDING
    n["EZeroFiveLamQuarterErr"] = _num("E0.5", "lam_0.25", "rel_err_mean")
    n["EZeroFiveBMaxErr"] = _num("E0.5b", "lam_0.9", "rel_err_mean", digits=2)
    n["GFourCW"] = _num("G4c", "uncond_lr1e-3", "w2_marginal_heldout", ci=True)
    n["GFourCFloor"] = _num("G4c", "uncond_lr1e-3", "w2_marginal_datadata_heldout")
    for arm, key in ((BASELINE, "Drifting"), ("single", "Single"), ("K4", "Oracle")):
        n[f"EOneTwo{key}TVR"] = _num_one_minus("E1.2", arm, "mci_heldout", ci=True, digits=3)
        n[f"EOneTwo{key}TVRMean"] = _num_one_minus("E1.2", arm, "mci_heldout", digits=3)
        n[f"EOneTwo{key}W"] = _num("E1.2", arm, "w2_heldout")
        n[f"EOneTwo{key}Off"] = _num("E1.2", arm, "off_manifold_heldout")
        n[f"EOneTwo{key}Cov"] = _num("E1.2", arm, "mode_coverage_heldout")
        n[f"EOneTwo{key}Prec"] = _num("E1.2", arm, "precision_heldout")
    for arm, key in (("AffD", "AffD"), ("single", "Single"), ("K1", "KOne"), ("K2", "KTwo"), ("K3", "KThree"),
                     (BASELINE, "Drifting")):
        n[f"EOneThree{key}W"] = _num("E1.3", arm, "w2_heldout", ci=True)
        n[f"EOneThree{key}TVR"] = _num_one_minus("E1.3", arm, "mci_heldout", digits=3)
    for arm, key in (("AffD", "AffD"), ("single", "Single"), ("K2", "KTwo")):
        n[f"EOneThree{key}Mem"] = _num("E1.3", arm, "memorization_train")
        n[f"EOneThree{key}Cov"] = _num("E1.3", arm, "mode_coverage_heldout")
        n[f"EOneThree{key}Off"] = _num("E1.3", arm, "off_manifold_heldout")
    n["EOneThreeAffDRecall"] = _num("E1.3", "AffD", "recall_heldout")
    for arm, key in (("AffD", "AffD"), ("K2", "KTwo")):
        n[f"EOneThree{key}Prec"] = _num("E1.3", arm, "precision_heldout")
    for arm, key in (("d0_h0", "ZeroOne"), ("d0_h1", "ZeroAff"), ("d16_h0", "SixteenOne"), ("d16_h1", "SixteenAff"),
                     ("d4_h0", "FourOne"), ("d4_h1", "FourAff")):
        n[f"EOneFour{key}W"] = _num("E1.4", arm, "w2_train", digits=3)
        n[f"EOneFour{key}TVR"] = _num_one_minus("E1.4", arm, "mci_train", digits=3)
    for arm, key in (("L64_h0", "SixtyFourOne"), ("L64_h1", "SixtyFourAff"), ("L1_h0", "OneOne"), ("L1_h1", "OneAff")):
        n[f"EOneFourB{key}W"] = _num("E1.4b", arm, "w2_train", digits=3)
    for arm, key in (("hq0", "Zero"), ("sym", "Sym")):
        n[f"EOneFive{key}Mem"] = _num("E1.5", arm, "memorization_train", digits=3)
        n[f"EOneFive{key}Own"] = _num("E1.5", arm, "own_positive_share_train", digits=2)
        n[f"EOneFive{key}TV"] = _num("E1.5", arm, "nw_tv_train", digits=3)
        n[f"EOneFive{key}W"] = _num("E1.5", arm, "w2_train", digits=3)
    ref = EXPERIMENTS_ROOT / "L1" / "analysis" / "reference.json"
    r = json.loads(ref.read_text()) if ref.exists() else None
    n["RefNWTrueTV"] = rf"${_fmt(r['E1.5 NW reference']['tv_nw_vs_true'], 3)}$" if r else PENDING
    n["RefTrueTrueTV"] = rf"${_fmt(r['E1.5 NW reference']['tv_true_vs_true'], 3)}$" if r else PENDING
    n["RefMemChance"] = rf"${_fmt(r['memorization chance level']['true_samples'], 3)}$" if r else PENDING
    n["RefOneToMean"] = rf"${_fmt(r['E1.4 one-positive sample mean']['d0_h0']['to_conditional_mean'], 2)}$" if r else PENDING
    n["RefOneToOwn"] = rf"${_fmt(r['E1.4 one-positive sample mean']['d0_h0']['to_own_positive'], 3)}$" if r else PENDING
    for arm, key in (("hx0.25", "Quarter"), ("hx1", "One"), ("hx8", "Eight"), ("fixed", "Fixed")):
        n[f"EOneSix{key}W"] = _num("E1.6", arm, "w2_heldout", digits=3)
        n[f"EOneSix{key}Energy"] = _num("E1.6", arm, "raw_drift_energy", digits=3)
    a, s = _ci("E1.3", "AffD", "w2_heldout"), _ci("E1.3", "single", "w2_heldout")
    n["EOneThreeRatio"] = PENDING if a is None or s is None else rf"${a['mean'] / s['mean']:.2f}$"
    return n


def _write_numbers(n: Mapping[str, str]) -> None:
    lines = [f"% generated by affdrift/reporting/paper.py at {datetime.now(UTC):%Y-%m-%d %H:%M UTC}, "
             f"commit {git_commit()[:10]} — do not edit"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(n.items())]
    lines.append(rf"\newcommand{{\BuildStamp}}{{{datetime.now(UTC):%Y-%m-%d %H:%M} UTC, commit \texttt{{{git_commit()[:7]}}}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


# ── tables ──────────────────────────────────────────────────────────────────

#: Plan claim labels that are not English, as they appear in the paper.
CLAIM_EN = {"단서 1": "Normalisation (cue 1)", "h 선택": "Bandwidth selection"}


def _verdict(eid: str) -> str:
    agg = _aggregate(eid)
    return PENDING if agg is None else rf"\textsc{{{agg['judgement']['verdict'].lower()}}}"


def _write_status_table() -> None:
    rows = [rf"{setting} & {claim} & {_verdict(eid)} \\" for eid, (setting, claim) in DESCRIBE.items()]
    lr = " / ".join(_verdict(e) for e in ("G4", "G4b", "G4c"))
    rows.insert(7, rf"2-D toy & Unconditional model fits the marginal (learning rate) & {lr} \\")
    (GEN / "tab_status.tex").write_text(
        "% generated — do not edit\n\\begin{tabular}{@{}llc@{}}\n\\toprule\nSetting & Pre-registered claim & Outcome \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _write_l1_table() -> None:
    arms = [(BASELINE, "E1.3"), ("single", "E1.3"), ("K1", "E1.3"), ("K2", "E1.3"), ("K3", "E1.3"), ("AffD", "E1.3"),
            ("K4", "E1.2")]
    metrics = [("w2_heldout", r"$W_2$ $\downarrow$"), ("tvr", r"$\operatorname{tr}\Sigma_q/\operatorname{tr}\Sigma_p$"),
               ("precision_heldout", r"Prec. $\uparrow$"), ("recall_heldout", r"Rec. $\uparrow$"),
               ("mode_coverage_heldout", r"Modes $\uparrow$")]
    head = " & ".join(["Method"] + [m[1] for m in metrics]) + r" \\"
    rows = []
    for arm, eid in arms:
        cells = [_num_one_minus(eid, arm, "mci_heldout", digits=2) if m == "tvr" else _num(eid, arm, m, digits=2)
                 for m, _ in metrics]
        name = LABEL.get(arm, arm)
        rows.append((rf"\textbf{{{name}}}" if arm == "AffD" else name) + " & " + " & ".join(cells) + r" \\")
        if arm == "K3":
            rows.append(r"\midrule")
    (GEN / "tab_l1.tex").write_text(
        "% generated — do not edit\n\\begin{tabular}{@{}l" + "c" * len(metrics) + "@{}}\n\\toprule\n" + head
        + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ── figures ─────────────────────────────────────────────────────────────────


def _pending_axis(ax: plt.Axes, what: str) -> None:
    ax.text(0.5, 0.5, f"{what}\n[pending]", ha="center", va="center", transform=ax.transAxes, color="0.5", fontsize=7)
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_color("0.75")


def _arrays(eid: str, arm: str) -> dict[str, np.ndarray] | None:
    """Seed-0 arrays of an arm, only once its experiment has been aggregated (the paper shows judged results only)."""
    if _aggregate(eid) is None:
        return None
    try:
        runs = complete(Experiment.load(eid))
    except FileNotFoundError:
        return None
    if arm not in runs or len(runs[arm]) < len(Experiment.load(eid).seeds):
        return None
    return runs[arm][min(runs[arm])].arrays


def _bank():
    from affdrift.levels.level1 import bank_of

    return bank_of(Experiment.load("E1.2").resolve("single"))


def fig_teaser() -> Figure:
    """Mean collapse vs mixture: samples of four methods on two held-out conditions (seed 0)."""
    from affdrift.data.toy2d import conditional_mean

    bank = _bank()
    cols = [(BASELINE, "E1.2"), ("single", "E1.2"), ("AffD", "E1.3"), ("K4", "E1.2")]
    conds = (0, 2)
    fig, axes = plt.subplots(len(conds), len(cols), figsize=(FULL, 3.15), squeeze=False)
    g = np.linspace(-2.4, 2.4, 140)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], -1)[None]
    for j, (arm, eid) in enumerate(cols):
        arr = _arrays(eid, arm)
        for i, k in enumerate(conds):
            ax = axes[i, j]
            c = bank.c_heldout[k]
            dens = np.exp(bank.log_density(pts, c[None])[0]).reshape(gx.shape)
            ax.contourf(gx, gy, dens, levels=[0.05 * dens.max(), dens.max()], colors=["0.82"])
            ax.contour(gx, gy, dens, levels=[0.05 * dens.max()], colors=["0.55"], linewidths=0.4)
            ax.scatter(*conditional_mean(c[None], mix_beta=bank.mix_beta)[0], marker="*", s=90, facecolors="none",
                       edgecolors="goldenrod", linewidths=0.9, zorder=2)
            if arr is None:
                _pending_axis(ax, LABEL[arm])
            else:
                f = arr["fake_heldout"][k]
                spread = float(np.linalg.norm(f - f.mean(0), axis=1).mean())
                size = 14 if spread < 0.05 else 1.3     # a collapsed condition is drawn large enough to see
                ax.scatter(f[:, 0], f[:, 1], s=size, color=arm_color(arm), alpha=0.6, lw=0, rasterized=True, zorder=3)
            ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xticks=[], yticks=[], aspect="equal")
            if i == 0:
                ax.set_title(LABEL[arm], color="black" if arm != "AffD" else arm_color("AffD"), fontweight="bold" if arm == "AffD" else None)
    return fig


def fig_mechanism() -> Figure:
    """V⁺ at a grid for one condition: single positive (one target) vs AffD (one basin per mode)."""
    from affdrift.eval.figures.level1 import _attraction_field

    exp = Experiment.load("E1.3")
    sigma = json.loads((exp.dir / "sigma.json").read_text())["sigma"]
    bank = _bank()
    probe = json.loads((ROOT / "probe" / "probe_L1.json").read_text())
    from scipy.spatial import cKDTree

    k = int(cKDTree(bank.c).query(np.array(probe["conditions"][0]["c"]))[1])
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.7))
    for ax, arm in zip(axes, ("single", "AffD")):
        d = _attraction_field(exp.resolve(arm), sigma, k, n=21)
        u = d["v"] / np.maximum(np.linalg.norm(d["v"], axis=-1, keepdims=True), 1e-12)
        g = np.linspace(-2.4, 2.4, 140)
        gx, gy = np.meshgrid(g, g)
        dens = np.exp(bank.log_density(np.stack([gx.ravel(), gy.ravel()], -1)[None], bank.c[k][None])[0]).reshape(gx.shape)
        ax.contourf(gx, gy, dens, levels=6, cmap="Greys", alpha=0.4)
        ax.quiver(d["grid"][:, 0], d["grid"][:, 1], u[:, 0], u[:, 1], color=arm_color(arm), scale=28, width=0.006)
        top = [i for i in np.argsort(d["w"])[-48:] if d["w"][i] > 1e-6 * d["w"].max()]
        ax.scatter(*bank.y[top].T, s=3 + 60 * d["w"][top] / d["w"].max(), facecolors="none", edgecolors="black", lw=0.4)
        ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xticks=[], yticks=[], aspect="equal")
        ax.set_title(LABEL[arm])
    return fig


def fig_l0() -> Figure:
    """E0.4: final assignment matrices as h_q grows (seed mean), against the NW weights w̃."""
    from affdrift.eval.metrics import assignment_matrix

    exp = Experiment.load("E0.4")
    runs = complete(exp)
    arms = [BASELINE, "hq_0", "hq_0.5", "hq_1"]
    titles = {BASELINE: "Drifting", "hq_0": r"$h^{\prime}=0$", "hq_0.5": r"$h^{\prime}=h/2$", "hq_1": r"$h^{\prime}=h$"}
    fig, axes = plt.subplots(1, len(arms) + 1, figsize=(FULL, 1.45))
    w = None
    for ax, arm in zip(axes, arms):
        if arm not in runs:
            _pending_axis(ax, titles[arm])
            continue
        recs = list(runs[arm].values())
        a = np.mean([assignment_matrix(r.arrays["final"], r.arrays["y"]) for r in recs], axis=0)
        w = recs[0].arrays["pos_weights"]
        ax.imshow(a, cmap="magma", vmin=0, vmax=0.35)
        ax.set(xticks=[], yticks=[], title=titles[arm])
    if w is not None:
        im = axes[-1].imshow(w, cmap="magma", vmin=0, vmax=0.35)
        axes[-1].set(xticks=[], yticks=[], title="Nadaraya–Watson\n" + r"weights $\tilde w_j(c_k)$")
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    axes[0].set_ylabel("condition $k$")
    return fig


def fig_phase() -> Figure:
    """E1.4: W2 gap between single positive and AffD across the generator's c-resolution δ."""
    from affdrift.eval.stats import bootstrap_ci

    fig, ax = plt.subplots(figsize=(COLUMN, 1.8))
    try:
        runs = complete(Experiment.load("E1.4"))
    except FileNotFoundError:
        runs = {}
    deltas = [("d0", 0.0), ("d64", 1 / 64), ("d16", 1 / 16), ("d4", 1 / 4)]
    ready = _aggregate("E1.4") is not None
    if not ready:
        _pending_axis(ax, "Conditional resolution sweep")
        return fig
    x = np.arange(len(deltas))
    for arm_suffix, name, color in (("h0", "One positive", arm_color("single")), ("h1", "Affinity Drifting", arm_color("AffD"))):
        cis = [bootstrap_ci([r.metrics["w2_train"] for r in runs[f"{d}_{arm_suffix}"].values()]) for d, _ in deltas]
        ax.plot(x, [c.mean for c in cis], "-o", ms=3, color=color, label=name, lw=1)
        ax.fill_between(x, [c.lo for c in cis], [c.hi for c in cis], color=color, alpha=0.2, lw=0)
    if BASELINE in runs:  # original Drifting, run once at the base resolution δ = 1/16
        base = bootstrap_ci([r.metrics["w2_train"] for r in runs[BASELINE].values()])
        ax.axhline(base.mean, color=arm_color(BASELINE), ls="--", lw=0.8, label=r"Drifting ($\delta=1/16$)")
    ax.set_xticks(x, ["0", "1/64", "1/16", "1/4"])
    ax.set(xlabel=r"quantization step $\delta$ of $c$ (coarser $\rightarrow$)", ylabel=r"$W_2$ (training $c$)")
    ax.legend(frameon=False)
    return fig


FIGURES: Mapping[str, Callable[[], Figure]] = {
    "fig_teaser": fig_teaser, "fig_mechanism": fig_mechanism, "fig_l0": fig_l0, "fig_phase": fig_phase,
}


def _write_figures() -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, make in FIGURES.items():
        try:
            with plt.rc_context(RC):
                fig = make()
                fig.savefig(FIG / f"{name}.pdf", dpi=300)
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001 — one figure must not stop the paper
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            with plt.rc_context(RC):
                fig, ax = plt.subplots(figsize=(COLUMN, 1.5))
                _pending_axis(ax, name)
                fig.savefig(FIG / f"{name}.pdf")
                plt.close(fig)
    return failed


# ── build ───────────────────────────────────────────────────────────────────


def build(*, compile_pdf: bool = True) -> dict[str, Any]:
    """Regenerate ``paper/generated/`` and (optionally) compile ``paper/main.pdf``."""
    GEN.mkdir(parents=True, exist_ok=True)
    _write_numbers(numbers())
    _write_status_table()
    _write_l1_table()
    failed = _write_figures()
    result: dict[str, Any] = {"figures_failed": failed, "pdf": None}
    if compile_pdf:
        # tectonic fetches the template's packages into its own cache: no change to the system TeX install
        proc = subprocess.run(["tectonic", "-X", "compile", "main.tex", "--keep-logs"],
                              cwd=PAPER, capture_output=True, text=True)
        result["pdf"] = str(PAPER / "main.pdf") if proc.returncode == 0 else None
        if proc.returncode != 0:
            result["latex_error"] = (proc.stdout + proc.stderr)[-2000:]
    return result
