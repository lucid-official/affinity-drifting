"""Saving figures with their reading sheet and blind twin (CLAUDE.md §3, §8).

``figures/<Level>/<ID>/<FIG>/`` (run-level: ``figures/<Level>/<ID>/runs/<arm>/seed<k>/<FIG>/``)::

    <FIG>.png         labelled figure
    <FIG>_blind.png   same panels, arm labels replaced and order shuffled
    <FIG>_key.json    blind label → arm
    <FIG>.md          sources, the plan's reading criterion, observation slots

Regenerating a figure rewrites everything above the ``## 관찰`` heading of
the ``.md`` and keeps what people wrote below it.
"""

import json
import string
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from affdrift.eval.figures.catalog import CATALOG  # noqa: E402
from affdrift.eval.figures.style import BLIND_COLOR, RC, arm_color  # noqa: E402
from affdrift.runtime import ROOT, git_commit  # noqa: E402

OBS = "## 관찰"


@dataclass(frozen=True, slots=True)
class Panel:
    """One arm's slot in a comparison figure."""

    arm: str
    label: str
    color: str
    data: Any

    @classmethod
    def of(cls, arm: str, data: Any) -> "Panel":
        return cls(arm=arm, label=arm, color=arm_color(arm), data=data)


type Render = Callable[[Sequence[Panel]], Figure]


def _blind(fig_id: str, panels: Sequence[Panel]) -> tuple[list[Panel], dict[str, str]]:
    rng = np.random.default_rng(zlib.crc32(fig_id.encode()))
    order = rng.permutation(len(panels))
    blinded, key = [], {}
    for letter, i in zip(string.ascii_uppercase, order):
        p = panels[i]
        blinded.append(replace(p, label=f"arm {letter}", color=BLIND_COLOR))
        key[f"arm {letter}"] = p.arm
    return blinded, key


def _sheet(fig_id: str, *, sources: Mapping[str, str], blind: bool, extra: str) -> str:
    spec = CATALOG[fig_id]
    src = "\n".join(f"- `{name}` — config `{h}`" for name, h in sources.items())
    blind_line = (f"먼저 `{fig_id}_blind.png`만 보고 관찰을 적은 뒤 `{fig_id}_key.json`과 라벨 figure를 연다."
                  if blind else "이 figure는 arm 비교가 아니어서 blind 버전이 없다.")
    return (
        f"# {fig_id} — {spec.figure}\n\n"
        f"- 실험: {spec.experiments}\n- 생성 commit: `{git_commit()[:10]}`\n"
        f"- 원천 run:\n{src}\n{extra}\n"
        f"## plan의 판독 기준 (docs/plan.md §10, 사전 등록)\n\n> {spec.reading}\n\n"
        f"## 판독 절차\n\n{blind_line}\n\n"
    )


def _observation_block(blind: bool) -> str:
    lines = [OBS, ""]
    if blind:
        lines += ["### 블라인드 판독 (라벨 공개 전)", "", "1. ", "2. ", "3. ", ""]
    lines += ["### 라벨 공개 후" if blind else "### 관찰 3줄", "", "1. ", "2. ", "3. ", ""]
    return "\n".join(lines)


class FigureWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(
        self,
        fig_id: str,
        render: Render,
        panels: Sequence[Panel],
        *,
        blind: bool,
        sources: Mapping[str, str],
        extra: str = "",
    ) -> Path:
        if fig_id not in CATALOG:
            raise KeyError(f"{fig_id} is not in the figure catalog")
        out = self.root / fig_id
        out.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(RC):
            _save(render(panels), out / f"{fig_id}.png")
            if blind:
                blinded, key = _blind(fig_id, panels)
                _save(render(blinded), out / f"{fig_id}_blind.png")
                (out / f"{fig_id}_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n")
        md = out / f"{fig_id}.md"
        kept = md.read_text().split(OBS, 1)[1] if md.exists() and OBS in md.read_text() else None
        head = _sheet(fig_id, sources=sources, blind=blind, extra=extra)
        md.write_text(head + (OBS + kept if kept is not None else _observation_block(blind)))
        return out / f"{fig_id}.png"


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
