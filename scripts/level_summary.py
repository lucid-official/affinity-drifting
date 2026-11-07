"""Write experiments/<Level>/summary.md from every aggregate.json of the Level (CLAUDE.md §8.8).

The automatic part is regenerated; everything under "## 사람이 적는 부분" is kept.
"""

import argparse
import json
import os
from datetime import UTC, datetime

from affdrift.config import EXPERIMENTS_ROOT, Experiment
from affdrift.runtime import git_commit
from affdrift.tracking import ledger

KEEP = "## 사람이 적는 부분"


def main(level: str) -> None:
    exps = [e for e in Experiment.all() if e.level == level]
    rows, checks, figs = [], [], []
    for exp in exps:
        agg_path = exp.dir / "aggregate.json"
        if not agg_path.exists():
            rows.append(f"| [{exp.id}]({exp.id}/) | {exp.spec.get('claim', '')} | {exp.title} | 미집계 | — |")
            continue
        agg = json.loads(agg_path.read_text())
        j = agg["judgement"]
        fails = [c for c in j["checks"] if c["verdict"] != "PASS"]
        rows.append(f"| [{exp.id}]({exp.id}/) | {exp.spec.get('claim', '')} | {exp.title} | **{j['verdict']}** | "
                    f"[report]({exp.id}/report.md) |")
        checks.append(f"### {exp.id} — {j['verdict']}\n")
        shown = fails or j["checks"]
        checks += [f"- {c['verdict']}: {c['name']} — {c['detail']}" for c in shown[:12]]
        if len(shown) > 12:
            checks.append(f"- … {len(shown) - 12} more in the report")
        summary = agg.get("summary", {})
        if summary:
            metrics = sorted({m for arm in summary.values() for m in arm})[:6]
            checks.append("\n| arm | " + " | ".join(metrics) + " |\n|" + " --- |" * (len(metrics) + 1))
            for arm in exp.arms:
                if arm in summary:
                    cells = [f"{c['mean']:.3g} [{c['lo']:.3g}, {c['hi']:.3g}]" if (c := summary[arm].get(m)) else "—"
                             for m in metrics]
                    checks.append(f"| {arm} | " + " | ".join(cells) + " |")
        checks.append("")
        figs += [f"- {exp.id}: " + ", ".join(f"[{p.parent.name}]({os.path.relpath(p, EXPERIMENTS_ROOT / level)})"
                                            for p in sorted(exp.figure_dir.glob("*/*.png"))
                                            if not p.stem.endswith("_blind"))]
    path = EXPERIMENTS_ROOT / level / "summary.md"
    kept = path.read_text().split(KEEP, 1)[1] if path.exists() and KEEP in path.read_text() else (
        "\n\n### 이상 징후 (사람)\n\n-\n\n### go/no-go (plan §9, 사용자 결정)\n\n-\n")
    text = f"""# {level} 요약

> 자동 생성: `scripts/level_summary.py {level}` · commit `{git_commit()[:10]}` · {datetime.now(UTC):%Y-%m-%d %H:%M UTC}
> 판정은 각 실험의 사전 등록된 `judge:` 규칙으로만 냈다. 해석은 아래 "사람이 적는 부분"과 각 report의 해석 칸에 둔다.

## 판정

| ID | Claim | 질문 | 판정 | 보고서 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## 핵심 수치와 판정 근거 (seed 평균 [95% CI])

{chr(10).join(checks)}
## Figures

{chr(10).join(figs)}

{KEEP}{kept}"""
    path.write_text(text)
    ledger.append("summary", level, [f"`{path.relative_to(EXPERIMENTS_ROOT.parent)}` regenerated"])
    print(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("level", help="e.g. L0")
    main(p.parse_args().level)
