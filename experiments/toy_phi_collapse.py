"""§7(b): φ를 학습 가능하게 뒀을 때 붕괴하는지 확인한다.

세 가지 φ를 비교한다. 모두 φ-공간 drifting loss만 쓴다(원본 공간 항을 넣으면 붕괴가 가려진다).

- frozen          : 무작위 초기화 후 고정 (아이디어 문서 1단계에 해당)
- collude         : φ도 ‖V‖²를 직접 줄임. V를 통과해 미분한다 (문서 §4가 경고한 공모)
- contrastive_ema : φ는 실제 쌍의 InfoNCE로만 갱신하고, drifting에는 EMA φ를 쓴다 (로드맵 4단계)

붕괴 신호는 "φ-공간 raw_drift는 작은데 x-공간 energy distance는 큰" 상태다.

    python experiments/toy_phi_collapse.py --out results/toy_phi_collapse
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toy_common import TEXT_DIM, ToyGenerator, ToyWorld, evaluate  # noqa: E402

from clipdrift.drift import DriftKernel, feature_drift_loss, normalized_drift  # noqa: E402
from clipdrift.sampling import PseudoClassSource, TextKernel, sample_groups  # noqa: E402

PHI_DIM = 8


def make_phi() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, PHI_DIM),
    )


def effective_rank(features: np.ndarray) -> float:
    """(Σs)²/Σs². φ가 소수 축으로 눌리면 1에 가까워진다. 스케일과 무관하다."""
    singular = np.linalg.svd(features - features.mean(0), compute_uv=False)
    return float(singular.sum() ** 2 / (singular**2).sum())


def separation(phi: nn.Module, world: ToyWorld, rng: np.random.Generator) -> float:
    """데이터와 원점 근처 가짜 분포를 φ가 얼마나 구분하는지(φ 표준편차 대비 평균 거리)."""
    data = world.draw(2048, rng).points
    fake = rng.normal(scale=0.2, size=data.shape).astype(np.float32)
    with lucid.no_grad():
        a, b = phi(lucid.tensor(data)).numpy(), phi(lucid.tensor(fake)).numpy()
    return float(np.linalg.norm(a.mean(0) - b.mean(0)) / (a.std(0).mean() + 1e-12))


def run(world_kind: str, mode: str, seed: int, steps: int) -> dict[str, float]:
    lucid.manual_seed(seed)
    rng = np.random.default_rng(seed)
    world = ToyWorld(world_kind, seed=0)
    data = world.draw(8192, rng)
    keys = TextKernel.fit(data.text).transform(data.text)
    source = PseudoClassSource(keys, 8, seed)

    generator = ToyGenerator()
    phi = make_phi()
    opt_gen = lucid.optim.Adam(generator.parameters(), lr=1e-3)
    points, text = lucid.tensor(data.points), lucid.tensor(data.text)

    phi_for_drift = phi
    opt_phi = None
    text_tower = None
    if mode == "collude":
        opt_phi = lucid.optim.Adam(phi.parameters(), lr=1e-3)
    elif mode == "contrastive_ema":
        text_tower = nn.Sequential(nn.Linear(TEXT_DIM, 128), nn.SiLU(), nn.Linear(128, PHI_DIM))
        opt_phi = lucid.optim.Adam(
            list(phi.parameters()) + list(text_tower.parameters()), lr=1e-3
        )
        phi_for_drift = copy.deepcopy(phi)
    elif mode != "frozen":
        raise ValueError(mode)
    drift_params = list(phi_for_drift.parameters())

    started = time.time()
    raw = 0.0
    for _ in range(steps):
        groups = sample_groups(
            source, size=len(keys), num_groups=8, num_pos=32, num_neg=32, rng=rng
        )
        prompts = lucid.tensor(groups.prompts, dtype=lucid.int64)
        positives = lucid.tensor(groups.positives, dtype=lucid.int64)
        y = points[positives]

        # 생성기 갱신: φ는 상수로 취급한다.
        for p in drift_params:
            p.requires_grad = False
        x = generator(text[prompts])
        loss, stats = feature_drift_loss(phi_for_drift(x).unsqueeze(1), phi_for_drift(y).unsqueeze(1))
        opt_gen.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 2.0)
        opt_gen.step()
        raw = float(stats.raw_drift.item())

        if mode == "collude":
            # φ가 ‖V‖²를 직접 줄인다. S 정규화 덕분에 전체 스케일을 줄여서 속일 수는 없다.
            for p in drift_params:
                p.requires_grad = True
            field, _, _ = normalized_drift(
                phi(x.detach()).unsqueeze(1),
                phi(y).unsqueeze(1),
                None,
                DriftKernel(),
                normalize_drift=False,
            )
            phi_loss = (field * field).mean()
            opt_phi.zero_grad()
            phi_loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 2.0)
            opt_phi.step()
        elif mode == "contrastive_ema":
            index = lucid.tensor(rng.integers(len(keys), size=256), dtype=lucid.int64)
            image = F.normalize(phi(points[index]), dim=-1)
            caption = F.normalize(text_tower(text[index]), dim=-1)
            logits = 14.0 * image @ caption.T
            labels = lucid.arange(256, dtype=lucid.int64)
            phi_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt_phi.zero_grad()
            phi_loss.backward()
            opt_phi.step()
            with lucid.no_grad():
                for slow, fast in zip(phi_for_drift.parameters(), phi.parameters()):
                    slow[...] = 0.99 * slow + 0.01 * fast
                    slow.eval()

    result = evaluate(generator, world)
    with lucid.no_grad():
        features = phi_for_drift(points[:2048]).numpy()
    result.update(
        raw_drift=raw,
        effective_rank=effective_rank(features),
        separation=separation(phi_for_drift, world, rng),
        seconds=time.time() - started,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/toy_phi_collapse"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--worlds", nargs="+", default=["gmm"])
    parser.add_argument("--modes", nargs="+", default=["frozen", "collude", "contrastive_ema"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for kind in args.worlds:
        for mode in args.modes:
            for seed in range(args.seeds):
                result = run(kind, mode, seed, args.steps)
                rows.append({"world": kind, "mode": mode, "seed": seed, **result})
                print(
                    f"{kind:5s} {mode:16s} seed={seed} cond_ed={result['cond_ed']:.4f} "
                    f"marginal_ed={result['marginal_ed']:.4f} spread={result['spread']:.3f} "
                    f"raw={result['raw_drift']:.2e} rank={result['effective_rank']:.2f} "
                    f"sep={result['separation']:.2f} ({result['seconds']:.0f}s)",
                    flush=True,
                )
                (args.out / "rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
