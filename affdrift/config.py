r"""Experiment configs: B0, per-experiment Δ, and the single-factor check.

Layout (CLAUDE.md §1)::

    configs/base_B0.yaml                      B0 and its per-Level section
    experiments/<Level>/<ID>/experiment.yaml  the experiment: arms, Δ, judge

An arm's resolved config is ``B0 ⊕ levels[<Level>] ⊕ common ⊕ arm.set``.
:meth:`Experiment.check_delta` enforces CLAUDE.md §2.1 before anything runs:

1. every key that differs from the Level's B0 matches ``allowed_delta``;
2. against the arm's reference arm, the differing keys fall in at most
   ``max_knobs`` of those patterns (1 unless the experiment is a factorial);
3. no arm introduces a key B0 does not have (typo guard).
"""

import fnmatch
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import yaml

from affdrift._typing import Config
from affdrift.runtime import ROOT

BASE_PATH = ROOT / "configs" / "base_B0.yaml"
EXPERIMENTS_ROOT = ROOT / "experiments"
#: Every figure lives here, mirroring the experiment hierarchy (CLAUDE.md §1, §8.7).
FIGURES_ROOT = ROOT / "figures"


#: The original Drifting arm every experiment must carry (CLAUDE.md §2.11).
BASELINE_ARM = "drifting"


class DeltaViolation(ValueError):
    """An arm changes more than its experiment declares (CLAUDE.md §2.1)."""


class MissingBaseline(ValueError):
    """An experiment has no original-Drifting baseline arm (CLAUDE.md §2.11)."""


# ── dict utilities ──────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def expand_dotted(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """``{"a.b": 1}`` → ``{"a": {"b": 1}}``, recursively."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = deep_merge(node.get(leaf, {}), expand_dotted(value)) if isinstance(value, Mapping) else value
    return out


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """New dict: ``over`` merged into ``base``; nested mappings merge, everything else replaces."""
    out = dict(base)
    for key, value in over.items():
        match out.get(key), value:
            case Mapping() as old, Mapping() as new:
                out[key] = deep_merge(old, new)
            case _:
                out[key] = value
    return out


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Dotted-key view of a nested mapping (lists are leaves)."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out |= flatten(value, f"{path}.")
        else:
            out[path] = value
    return out


def diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """``{dotted key: (a value, b value)}`` for every key whose value differs."""
    fa, fb = flatten(a), flatten(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)}


def config_hash(cfg: Config) -> str:
    """First 12 hex digits of SHA-256 over the canonical JSON of ``cfg``."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── experiments ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Delta:
    """What an arm changes, against B0 and against its reference arm."""

    arm: str
    ref: str
    vs_base: dict[str, tuple[Any, Any]]
    vs_ref: dict[str, tuple[Any, Any]]
    knobs: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "reference_arm": self.ref,
            "knobs_vs_reference": list(self.knobs),
            "vs_B0": {k: {"B0": a, "arm": b} for k, (a, b) in self.vs_base.items()},
            "vs_reference": {k: {"reference": a, "arm": b} for k, (a, b) in self.vs_ref.items()},
        }


@dataclass(frozen=True)
class Experiment:
    """One experiment directory, ``experiments/<Level>/<ID>/``."""

    dir: Path
    spec: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, exp_id: str) -> Self:
        matches = sorted(EXPERIMENTS_ROOT.glob(f"L*/{exp_id}/experiment.yaml"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one experiments/L*/{exp_id}/experiment.yaml, found {matches}")
        spec = load_yaml(matches[0])
        if spec.get("id") != exp_id:
            raise ValueError(f"{matches[0]} declares id {spec.get('id')!r}, not {exp_id!r}")
        return cls(matches[0].parent, spec)

    @classmethod
    def all(cls) -> Iterator[Self]:
        for path in sorted(EXPERIMENTS_ROOT.glob("L*/*/experiment.yaml"), key=lambda p: _id_key(p.parent.name)):
            yield cls.load(path.parent.name)

    # ── identity ─────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def level(self) -> str:
        return self.spec["level"]

    @property
    def title(self) -> str:
        return self.spec.get("title", "")

    @property
    def runner(self) -> str:
        return self.spec["runner"]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.spec["seeds"])

    @property
    def arms(self) -> dict[str, dict[str, Any]]:
        return {name: dict(arm or {}) for name, arm in self.spec["arms"].items()}

    @property
    def reference_arm(self) -> str:
        return self.spec.get("reference_arm", next(iter(self.arms)))

    @property
    def figure_dir(self) -> Path:
        """``figures/<Level>/<ID>/`` — experiment-level figures; run-level ones under ``runs/<arm>/seed<k>/``."""
        return FIGURES_ROOT / self.level / self.id

    def run_figure_dir(self, arm: str, seed: int) -> Path:
        return self.figure_dir / "runs" / arm / f"seed{seed}"

    def ref_of(self, arm: str) -> str:
        return self.arms[arm].get("ref", self.reference_arm)

    # ── configs ──────────────────────────────────────────────────────────

    @cached_property
    def base(self) -> dict[str, Any]:
        """B0 resolved for this experiment's Level."""
        b0 = load_yaml(BASE_PATH)
        levels = b0.pop("levels", {})
        return deep_merge(b0, levels.get(self.level, {}))

    def resolve(self, arm: str) -> dict[str, Any]:
        if arm not in self.arms:
            raise KeyError(f"{self.id} has no arm {arm!r}; arms are {list(self.arms)}")
        common = expand_dotted(self.spec.get("common", {}))
        overrides = expand_dotted(self.arms[arm].get("set", {}))
        return deep_merge(deep_merge(self.base, common), overrides)

    def check_baseline(self) -> None:
        """Every experiment runs the original Drifting as arm ``drifting`` with ``field: release``."""
        if BASELINE_ARM not in self.arms:
            raise MissingBaseline(f"{self.id} has no {BASELINE_ARM!r} arm")
        if (field_ := self.resolve(BASELINE_ARM)["field"]) != "release":
            raise MissingBaseline(f"{self.id}/{BASELINE_ARM} has field {field_!r}, not 'release'")

    def check_delta(self, arm: str) -> Delta:
        """Enforce the single-factor rule; raises :class:`DeltaViolation`."""
        resolved, ref = self.resolve(arm), self.ref_of(arm)
        allowed = tuple(self.spec.get("allowed_delta", ()))
        max_knobs = int(self.spec.get("max_knobs", 1))

        base_keys = flatten(self.base).keys()
        if unknown := sorted(flatten(resolved).keys() - base_keys):
            raise DeltaViolation(f"{self.id}/{arm}: keys not in B0 {unknown}")

        vs_base = diff(self.base, resolved)
        if outside := [k for k in vs_base if not _matches(k, allowed)]:
            raise DeltaViolation(f"{self.id}/{arm}: {outside} differ from B0 but are not in allowed_delta {allowed}")

        vs_ref = diff(self.resolve(ref), resolved)
        knobs = tuple(sorted({p for k in vs_ref for p in allowed if fnmatch.fnmatchcase(k, p)}))
        groups = {_first_match(k, allowed) for k in vs_ref}
        if len(groups) > max_knobs:
            raise DeltaViolation(
                f"{self.id}/{arm}: changes {sorted(groups)} against reference arm {ref!r}; at most {max_knobs} allowed"
            )
        return Delta(arm=arm, ref=ref, vs_base=vs_base, vs_ref=vs_ref, knobs=knobs)


def _matches(key: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(key, p) for p in patterns)


def _first_match(key: str, patterns: tuple[str, ...]) -> str:
    return next(p for p in patterns if fnmatch.fnmatchcase(key, p))


def _id_key(exp_id: str) -> tuple[int, ...]:
    """``"E0.10"`` sorts after ``"E0.9"``."""
    digits = exp_id.lstrip("E").split(".")
    return tuple(int("".join(ch for ch in d if ch.isdigit()) or 0) for d in digits)
