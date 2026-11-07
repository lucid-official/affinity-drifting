"""The one place arm order and colour are defined (CLAUDE.md §3).

Plan arms keep the plan's order — single, K1, K2, K3, AffD, K4, K5, K6 — and
every Level-specific arm name has an entry here too, so the same arm has the
same colour in every figure of the repository.
"""

from collections.abc import Iterable
from types import MappingProxyType

from affdrift.config import BASELINE_ARM as BASELINE

ARM_COLORS = MappingProxyType({
    # the original Drifting (Lucid release) — in every experiment and every comparison figure
    BASELINE: "#000000",
    # plan §2.3 / §7.1, in the plan's order
    "single": "#7f7f7f",
    "K1": "#d62728",
    "K2": "#ff7f0e",
    "K3": "#bcbd22",
    "AffD": "#1f77b4",
    "K4": "#2ca02c",
    "K5": "#9467bd",
    "K6": "#8c564b",
    # Level 0 — repulsion bandwidth h_q (E0.2–E0.4, E0.6)
    "b0": "#1f77b4",
    "hq0": "#1f77b4",
    "hq_0": "#1f77b4",
    "hq_0.25": "#5b8fc2",
    "hq_0.5": "#a05fa8",
    "hq_1": "#d62728",
    "sym": "#d62728",
    "sym_hp_x0.5": "#ff9896",
    "sym_hp_x2": "#8c1c13",
    "sep_hq0": "#1f77b4",
    "joint_hq0": "#17becf",
    "sep_sym": "#d62728",
    "joint_sym": "#e377c2",
    # Level 0 — counterfactual λ (E0.5)
    "lam_0": "#08306b",
    "lam_0.25": "#2171b5",
    "lam_0.5": "#4292c6",
    "lam_0.75": "#6baed6",
    "lam_0.9": "#9ecae1",
    "lam_1.1": "#e6550d",
    # E0.1 / E1.1
    "check": "#333333",
    # L1 — G4 learning-rate grid (unconditional K5)
    "k5_lr3e-4": "#c5b0d5",
    "k5_lr1e-3": "#9467bd",
    "k5_lr3e-3": "#5b3a8a",
    # L1 — h_p as a multiple of the N_eff bandwidth h* (E1.6); h0 = single colour, h1 = AffD colour
    "hx0.25": "#c6dbef", "hx0.5": "#6baed6", "hx1": "#1f77b4", "hx2": "#08519c", "hx4": "#08306b", "hx8": "#041a3a",
    "fixed": "#2ca02c",
    # L1 — factorials E1.4 (d<δ>) and E1.4b (L<L>): colour = h multiple, identical across capacities
    **{f"{cap}_{h}": colour
       for cap in ("d0", "d64", "d16", "d4", "L64", "L16", "L4", "L1")
       for h, colour in (("h0", "#7f7f7f"), ("h0.5", "#6baed6"), ("h1", "#1f77b4"), ("h2", "#08519c"), ("h4", "#08306b"))},
})
ARM_ORDER = tuple(ARM_COLORS)
BLIND_COLOR = "#555555"


def arm_color(arm: str) -> str:
    try:
        return ARM_COLORS[arm]
    except KeyError:
        raise KeyError(f"arm {arm!r} has no colour: add it to affdrift/eval/figures/style.py") from None


def ordered(arms: Iterable[str]) -> list[str]:
    """``arms`` in the global order."""
    arms = list(arms)
    for a in arms:
        arm_color(a)
    return sorted(arms, key=ARM_ORDER.index)


RC = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}
