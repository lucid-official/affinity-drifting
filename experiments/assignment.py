"""생성 샘플은 어느 학습 이미지로 가나: 텍스트 이웃 그룹 안의 배정 행렬 A와 캡션 가중치 w̃.

affinity-drifting 브랜치 L0(E0.2–E0.4)의 배정 측정을 학습된 생성기에 그대로 옮긴다.

- 그룹: joint 학습과 같은 방식(`KnnSource`: 크기 8192의 무작위 큐에서 앵커의 텍스트 이웃 256개)으로 고른
  학습 항목 32개. 32개가 조건 c_k이자 positive y_k다.
- A_kj: 조건 k의 샘플을 픽셀 공간에서 가장 가까운 y_j에 배정한 비율.
- w̃_j(c_k) ∝ exp(-‖e_k - e_j‖/τ_t): 이 그룹을 positive로 놓았을 때 joint kernel이 매기는 캡션 가중치.
  pair의 목표는 단위행렬(자기 짝 이미지)이다.

함께 재는 것:
- 실제 학습 샘플러에서 프롬프트 자기 짝 이미지가 받는 w̃ 몫. 짝 이미지는 positive 64개 중 하나로 뽑힐 때만 받는다.
- 학습 이미지 전체에 대한 최근접 거리(외우기 확인): 학습 캡션 512개 × 16장. 기준은 실제 사진의 leave-one-out 거리다.

결과는 각 실행의 `assignment.json`과 `results/assignment_reference.json`에 쓴다. 생성 시드는 0으로 고정한다.

    PYTHONPATH=. .venv/bin/python experiments/assignment.py runs/flickr_s10000_*_seed*
"""

import argparse
import json
from pathlib import Path

import lucid
import numpy as np

from clipdrift.caption_eval import load_generator
from clipdrift.data import read_manifest
from clipdrift.evaluate import generate
from clipdrift.sampling import KnnSource, TextKernel, TextKeyedQueue
from clipdrift.text_bank import build_text_bank

GROUPS, MEMBERS, PER_PROMPT = 64, 32, 64
MEMO_PROMPTS, MEMO_PER = 512, 16
SAMPLER_GROUPS = 500
DIST_EDGES = np.arange(0, 81, 1.0)  # 픽셀 L2 거리 히스토그램 구간. 마지막 구간에 넘치는 값을 모은다.


def text_keys(train, config: dict, device: str):
    """`train.train`과 같은 방식으로 텍스트 키를 만든다."""
    bank, _ = build_text_bank(train.captions, num_tokens=config["num_text_tokens"], device=device)
    fit = TextKernel.fit(
        bank.pooled[np.unique(train.caption_ids)],
        shrinkage=config["text_shrinkage"] if config["text_shrinkage"] >= 0 else None,
        rank=config["text_rank"] or None,
    )
    return bank, fit.transform(bank.pooled[train.caption_ids])


def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum((a**2).sum(-1)[:, None] + (b**2).sum(-1)[None] - 2 * a @ b.T, 0))


def make_groups(keys: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    groups = []
    for _ in range(GROUPS):
        queue = TextKeyedQueue(keys, config["queue_size"])
        queue.push(rng.permutation(len(keys))[: queue.capacity])
        pool = queue.neighbors(keys[rng.choice(queue.items)], config["neighbors"])
        groups.append(rng.choice(pool, MEMBERS, replace=False))
    return np.stack(groups)


def caption_weights(keys: np.ndarray, members: np.ndarray, tau: float) -> np.ndarray:
    w = np.exp(-distances(keys[members], keys[members]) / tau)
    return w / w.sum(1, keepdims=True)


def sampler_self_mass(keys: np.ndarray, config: dict, rng: np.random.Generator) -> float:
    """실제 `KnnSource` 그룹에서 프롬프트마다 자기 짝 이미지가 받는 w̃ 몫의 평균."""
    source = KnnSource(keys, neighbors=config["neighbors"], queue_size=config["queue_size"])
    source.warm_up(rng)
    mass = []
    for _ in range(SAMPLER_GROUPS):
        positives, prompts = source.group(config["num_pos"], config["num_neg"], rng)
        w = np.exp(-distances(keys[prompts], keys[positives]) / config["tau_t"])
        w /= w.sum(1, keepdims=True)
        mass.extend((w * (positives[None] == prompts[:, None])).sum(1))
    return float(np.mean(mass))


def tv(a: np.ndarray, b: np.ndarray) -> float:
    """조건별 TV 거리의 평균."""
    return float(0.5 * np.abs(a - b).sum(-1).mean())


def nearest(x: np.ndarray, flat: np.ndarray, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    index, dist = [], []
    for start in range(0, len(x), 1024):
        d = distances(x[start : start + 1024], flat)
        if exclude is not None:
            d[np.arange(len(d)), exclude[start : start + 1024]] = np.inf
        best = d.argmin(1)
        index.append(best)
        dist.append(d[np.arange(len(d)), best])
    return np.concatenate(index), np.concatenate(dist)


def histogram(values: np.ndarray) -> list[int]:
    return np.histogram(np.minimum(values, DIST_EDGES[-1] - 1e-6), bins=DIST_EDGES)[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()

    config = json.loads((args.runs[0] / "config.json").read_text())
    train = read_manifest(Path(config["data"]))
    bank, keys = text_keys(train, config, args.device)
    flat = train.images.reshape(len(train.images), -1)

    groups = make_groups(keys, config, np.random.default_rng(0))
    weights = np.stack([caption_weights(keys, members, config["tau_t"]) for members in groups])
    eye, uniform = np.eye(MEMBERS), np.full((MEMBERS, MEMBERS), 1 / MEMBERS)
    # 예시 그룹(0번)은 텍스트 키의 첫 주성분 순서로 정렬해 w̃의 띠 구조가 보이게 한다.
    example = keys[groups[0]] - keys[groups[0]].mean(0)
    order = np.argsort(np.linalg.svd(example, full_matrices=False)[0][:, 0])

    memo = np.random.default_rng(1).choice(len(flat), MEMO_PROMPTS, replace=False)
    _, real_dist = nearest(flat[memo], flat, exclude=memo)
    reference = {
        "groups": GROUPS,
        "members": MEMBERS,
        "per_prompt": PER_PROMPT,
        "tau_t": config["tau_t"],
        "w_self": float(np.mean(np.diagonal(weights, axis1=1, axis2=2))),
        "tv_uniform_w": tv(uniform, weights),
        "tv_identity_w": tv(eye, weights),
        "sampler_self_mass": sampler_self_mass(keys, config, np.random.default_rng(2)),
        "example_items": groups[0][order].tolist(),
        "example_w": np.round(weights[0][order][:, order], 4).tolist(),
        "memorization": {
            "prompts": MEMO_PROMPTS,
            "per_prompt": MEMO_PER,
            "edges": DIST_EDGES.tolist(),
            "real_loo_hist": histogram(real_dist),
            "real_loo_median": float(np.median(real_dist)),
        },
    }
    Path("results/assignment_reference.json").write_text(json.dumps(reference))
    print({k: v for k, v in reference.items() if k.startswith(("w_", "tv_", "sampler"))}, flush=True)

    for run in args.runs:
        model = load_generator(run, args.device)
        lucid.manual_seed(0)
        assignment = np.zeros((GROUPS, MEMBERS, MEMBERS))
        for g, members in enumerate(groups):
            prompts = np.repeat(train.caption_ids[members], PER_PROMPT)
            x = generate(model, bank.pooled[prompts], bank.tokens[prompts], alpha=1.0, device=args.device)
            best = distances(x.reshape(len(x), -1), flat[members]).argmin(1).reshape(MEMBERS, PER_PROMPT)
            assignment[g] = np.stack([np.bincount(row, minlength=MEMBERS) for row in best]) / PER_PROMPT
        prompts = np.repeat(train.caption_ids[memo], MEMO_PER)
        x = generate(model, bank.pooled[prompts], bank.tokens[prompts], alpha=1.0, device=args.device)
        x = x.reshape(len(x), -1)
        index, dist = nearest(x, flat)
        own = np.sqrt(((x - flat[np.repeat(memo, MEMO_PER)]) ** 2).sum(1))
        columns = np.sort(assignment.sum(1), axis=1)[:, ::-1] / MEMBERS  # 그룹마다 열(학습 이미지)별 배정 몫
        result = {
            "self": float(np.mean(np.diagonal(assignment, axis1=1, axis2=2))),
            "tv_w": tv(assignment, weights),
            "tv_uniform": tv(assignment, uniform),
            "top3": float(columns[:, :3].sum(1).mean()),
            "example": np.round(assignment[0][order][:, order], 4).tolist(),
            "self_hit": float(np.mean(index == np.repeat(memo, MEMO_PER))),
            "nn_median": float(np.median(dist)),
            "own_median": float(np.median(own)),
            "nn_hist": histogram(dist),
            "own_hist": histogram(own),
        }
        (run / "assignment.json").write_text(json.dumps(result))
        print(f"{run.name}: " + " ".join(f"{k}={v:.4f}" for k, v in result.items() if isinstance(v, float)),
              flush=True)


if __name__ == "__main__":
    main()
