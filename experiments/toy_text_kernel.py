"""§7(a): 텍스트 kernel 유/무 비교.

프롬프트 하나에 샘플 하나인 상황에서 조건부 분포를 얼마나 잘 배우는지 잰다. CPU에서 돌린다.

    python experiments/toy_text_kernel.py --out results/toy_text_kernel
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lucid
import lucid.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toy_common import ToyGenerator, ToyWorld, evaluate  # noqa: E402

from clipdrift.drift import feature_drift_loss  # noqa: E402
from clipdrift.sampling import (  # noqa: E402
    BlindSource,
    KnnSource,
    PairSource,
    PseudoClassSource,
    TextKernel,
    sample_groups,
    text_kernel_for_groups,
)


def build_source(name: str, keys: np.ndarray, seed: int):
    """방법 이름 -> (source, τ_t, 그룹 수, N, P). 방법끼리 스텝당 생성 샘플 수를 맞춘다."""
    if name == "blind":
        return BlindSource(len(keys)), None, 8, 32, 32
    if name == "pair":
        return PairSource(len(keys)), None, 64, 4, 1
    if name.startswith("pseudo"):
        return PseudoClassSource(keys, int(name[6:]), seed), None, 8, 32, 32
    if name.startswith("joint"):
        tau_t = float(name[5:])
        return KnnSource(keys, neighbors=128, queue_size=4096), tau_t, 8, 32, 32
    if name.startswith("wide"):
        # kNN이 큐 전체라 그룹이 무작위다. 조건은 오직 텍스트 kernel에서만 나온다.
        tau_t = float(name[4:])
        return KnnSource(keys, neighbors=4096, queue_size=4096), tau_t, 8, 32, 32
    raise ValueError(name)


def run(world_kind: str, method: str, seed: int, steps: int, shrinkage: float | None) -> dict[str, float]:
    lucid.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = ToyWorld(world_kind, seed=0)
    data = world.draw(8192, rng)
    keys = TextKernel.fit(data.text, shrinkage=shrinkage).transform(data.text)

    source, tau_t, groups_n, num_neg, num_pos = build_source(method, keys, seed)
    if isinstance(source, KnnSource):
        source.warm_up(rng)

    generator = ToyGenerator()
    optimizer = lucid.optim.Adam(generator.parameters(), lr=1e-3)
    points, text = lucid.tensor(data.points), lucid.tensor(data.text)
    started = time.time()
    raw = 0.0
    for _ in range(steps):
        groups = sample_groups(
            source,
            size=len(keys),
            num_groups=groups_n,
            num_pos=num_pos,
            num_neg=num_neg,
            rng=rng,
        )
        prompts = lucid.tensor(groups.prompts, dtype=lucid.int64)
        positives = lucid.tensor(groups.positives, dtype=lucid.int64)
        x = generator(text[prompts]).unsqueeze(1)  # [G, 1, N, 2]
        y = points[positives].unsqueeze(1)
        kernel = None
        if tau_t is not None:
            kernel = text_kernel_for_groups(keys, groups, tau_t, device="cpu")
        loss, stats = feature_drift_loss(x, y, None, kernel)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 2.0)
        optimizer.step()
        raw = float(stats.raw_drift.item())
    result = evaluate(generator, world)
    result.update(raw_drift=raw, seconds=time.time() - started)
    return result


_ALL_METHODS = [
    "blind",
    "pair",
    "pseudo8",
    "pseudo32",
    "joint0.02",
    "joint0.1",
    "joint0.5",
    "joint100",
    "wide0.02",
    "wide0.1",
    "wide0.5",
    "wide2",
    "wide100",
]
METHODS = {"gmm": _ALL_METHODS, "ring": _ALL_METHODS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/toy_text_kernel"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--worlds", nargs="+", default=["gmm", "ring"])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--whiten", action="store_true", help="whiten text keys (shrinkage=0.1)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for kind in args.worlds:
        for method in args.methods or METHODS[kind]:
            for seed in range(args.seeds):
                result = run(kind, method, seed, args.steps, 0.1 if args.whiten else None)
                rows.append({"world": kind, "method": method, "seed": seed, **result})
                print(
                    f"{kind:5s} {method:10s} seed={seed} "
                    f"cond_ed={result['cond_ed']:.4f} marginal_ed={result['marginal_ed']:.4f} "
                    f"spread={result['spread']:.3f} raw={result['raw_drift']:.2e} "
                    f"({result['seconds']:.0f}s)",
                    flush=True,
                )
                (args.out / "rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
