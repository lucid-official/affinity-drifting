"""캡션 데이터(예: Flickr30k) 평가. 정답 속성이 없으므로 고정 CLIP 공간에서 잰다.

- `clip_score`: 이미지와 그 캡션의 코사인 유사도 평균.
- `r_precision`: 제 캡션과 무작위 캡션 99개 중 제 캡션이 가장 가까운 비율. 우연 수준은 0.01이다.
- `fd`: CLIP 이미지 임베딩 공간의 Fréchet distance(생성 vs 실제 테스트 이미지). 표본이 1000개라 편향이 있으므로
  실제 학습 이미지와 실제 테스트 이미지 사이의 값(`real.fd`)을 바닥값으로 함께 적는다.
- `diversity`: 같은 캡션으로 만든 샘플끼리의 평균 코사인 거리. 한 점으로 붕괴하면 0이다.
- `layout_corr`, `layout_r_precision`: 4×4로 줄인 색 배치(장면의 대략적인 구성). 프롬프트별 샘플 평균을 짝지어진
  실제 이미지와 비교한다. 상관은 1000개 프롬프트에 걸친 Pearson 상관의 차원 평균이고, R-precision은 짝 이미지와
  무작위 이미지 99개 중 짝이 가장 가까운 비율이다. 이미지가 거칠어 CLIP이 못 잡는 조건 정보를 본다.

평가 이미지가 32×32라 CLIP에는 불리하다. 실제 32×32 이미지의 값이 이 해상도에서의 상한 기준이다.
`retrieval`은 캡션이 가장 비슷한 학습 이미지를 그대로 가져오는 기준선으로, 텍스트만으로 어디까지 맞힐 수
있는지 보여 준다. 테스트 캡션은 학습에 쓰지 않았으므로 텍스트 일반화까지 함께 잰다.

    python -m clipdrift.caption_eval runs/flickr_joint --test data/flickr30k_32/test.jsonl
"""

import argparse
import json
from pathlib import Path

import lucid
import numpy as np
from scipy import linalg

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.data import read_manifest
from clipdrift.evaluate import generate, save_grid
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.text_bank import build_text_bank

CANDIDATES = 100
LAYOUT = 4
SAMPLES_PER_PROMPT = 8


def clip_score(images: np.ndarray, texts: np.ndarray) -> float:
    return float((images * texts).sum(-1).mean())


def _candidates(count: int, rng: np.random.Generator, candidates: int):
    for i in range(count):
        others = rng.choice(np.delete(np.arange(count), i), size=candidates - 1, replace=False)
        yield i, np.concatenate([[i], others])


def r_precision(
    images: np.ndarray, texts: np.ndarray, rng: np.random.Generator, candidates: int = CANDIDATES
) -> float:
    """i번째 이미지에 대해 i번째 캡션과 다른 캡션 `candidates - 1`개를 비교한다."""
    hits = [(texts[pool] @ images[i]).argmax() == 0 for i, pool in _candidates(len(images), rng, candidates)]
    return float(np.mean(hits))


def layout(images: np.ndarray) -> np.ndarray:
    """`[M, C, H, W]` -> 4×4 평균 풀링한 색 배치 `[M, C * 16]`."""
    m, c, h, w = images.shape
    pooled = images.reshape(m, c, LAYOUT, h // LAYOUT, LAYOUT, w // LAYOUT).mean((3, 5))
    return pooled.reshape(m, -1)


def layout_corr_cells(predicted: np.ndarray, real: np.ndarray) -> np.ndarray:
    """프롬프트에 걸친 Pearson 상관을 차원(채널 × 칸)마다 구한다. `[C * 16]`."""
    p = predicted - predicted.mean(0)
    r = real - real.mean(0)
    return (p * r).sum(0) / np.sqrt((p * p).sum(0) * (r * r).sum(0) + 1e-12)


def layout_corr(predicted: np.ndarray, real: np.ndarray) -> float:
    """차원별 상관의 평균."""
    return float(layout_corr_cells(predicted, real).mean())


def layout_r_precision(
    predicted: np.ndarray, real: np.ndarray, rng: np.random.Generator, candidates: int = CANDIDATES
) -> float:
    hits = [
        ((real[pool] - predicted[i]) ** 2).sum(-1).argmin() == 0
        for i, pool in _candidates(len(predicted), rng, candidates)
    ]
    return float(np.mean(hits))


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean_a, mean_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = np.real(linalg.sqrtm(cov_a @ cov_b))
    return float(((mean_a - mean_b) ** 2).sum() + np.trace(cov_a + cov_b - 2.0 * root))


def diversity(embeddings: np.ndarray) -> float:
    """`[P, S, E]` 정규화 임베딩 -> 프롬프트별 샘플 쌍의 평균 코사인 거리."""
    samples = embeddings.shape[1]
    similarity = np.einsum("pse,pte->pst", embeddings, embeddings)
    off_diagonal = similarity.sum((1, 2)) - np.trace(similarity, axis1=1, axis2=2)
    return float(1.0 - (off_diagonal / (samples * (samples - 1))).mean())


def real_reference(
    clip: FrozenCLIP, train_images: np.ndarray, real: np.ndarray, test_text: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """실제 32×32 이미지의 값. 상한 기준과, 캡션을 섞었을 때의 하한 기준."""
    rng = np.random.default_rng(seed)
    subset = train_images[rng.choice(len(train_images), len(real), replace=False)]
    train = clip.encode_images(lucid.tensor(subset))
    shuffled = test_text[rng.permutation(len(test_text))]
    return {
        "clip_score": clip_score(real, test_text),
        "clip_score_shuffled": clip_score(real, shuffled),
        "r_precision": r_precision(real, test_text, np.random.default_rng(seed)),
        "fd": frechet_distance(train, real),
    }


def retrieval_reference(
    clip: FrozenCLIP,
    train_images: np.ndarray,
    train_text: np.ndarray,
    test_images: np.ndarray,
    real: np.ndarray,
    test_text: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    """캡션이 가장 비슷한 학습 이미지 8장을 생성 샘플 대신 쓴다. 생성 없이 텍스트만으로 얻는 수준이다."""
    nearest = np.argsort(-(test_text @ train_text.T), axis=1)[:, :SAMPLES_PER_PROMPT]
    top = clip.encode_images(lucid.tensor(train_images[nearest[:, 0]]))
    predicted = layout(train_images[nearest.reshape(-1)]).reshape(len(test_text), SAMPLES_PER_PROMPT, -1).mean(1)
    target = layout(test_images)
    return {
        "clip_score": clip_score(top, test_text),
        "r_precision": r_precision(top, test_text, np.random.default_rng(seed)),
        "fd": frechet_distance(top, real),
        "layout_corr": layout_corr(predicted, target),
        "layout_r_precision": layout_r_precision(predicted, target, np.random.default_rng(seed)),
    }


def load_generator(run: Path, device: str) -> TextConditionedGenerator:
    model = TextConditionedGenerator(GeneratorConfig(**json.loads((run / "generator.json").read_text())))
    model.load_state_dict(lucid.load(run / "generator_ema.lucid"))
    return model.to(device).eval()


def evaluate_run(
    run: Path,
    clip: FrozenCLIP,
    test,
    real: np.ndarray,
    test_text: np.ndarray,
    bank,
    references: dict[str, dict[str, float]],
    *,
    device: str,
    alphas: tuple[float, ...] = (1.0, 2.0),
    diversity_prompts: int = 50,
    seed: int = 0,
) -> dict[str, object]:
    """프롬프트마다 샘플 8장을 만든다. CLIP 지표는 첫 샘플, 배치 지표는 샘플 평균으로 잰다."""
    model = load_generator(run, device)
    rows = test.caption_ids
    count, per = len(rows), SAMPLES_PER_PROMPT
    target = layout(test.images)
    result: dict[str, object] = dict(references)
    for alpha in alphas:
        lucid.manual_seed(seed)
        repeated = np.repeat(rows, per)
        samples = generate(model, bank.pooled[repeated], bank.tokens[repeated], alpha=alpha, device=device)
        samples = samples.reshape(count, per, *samples.shape[1:])
        embed = clip.encode_images(lucid.tensor(np.ascontiguousarray(samples[:, 0])))
        varied = samples[:diversity_prompts].reshape(-1, *samples.shape[2:])
        varied_embed = clip.encode_images(lucid.tensor(varied)).reshape(diversity_prompts, per, -1)
        predicted = layout(samples.reshape(-1, *samples.shape[2:])).reshape(count, per, -1).mean(1)
        result[f"alpha={alpha}"] = {
            "clip_score": clip_score(embed, test_text),
            "r_precision": r_precision(embed, test_text, np.random.default_rng(seed)),
            "fd": frechet_distance(embed, real),
            "diversity": diversity(varied_embed),
            "layout_corr": layout_corr(predicted, target),
            "layout_r_precision": layout_r_precision(predicted, target, np.random.default_rng(seed)),
        }
        if alpha == alphas[0]:
            save_grid(
                run / "samples_test.png",
                [
                    (test.captions[rows[p]], np.concatenate([test.images[p : p + 1], samples[p]]))
                    for p in range(12)
                ],
                label_width=300,
            )
    (run / "caption_eval.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()

    config = json.loads((args.runs[0] / "config.json").read_text())
    test = read_manifest(args.test)
    clip = FrozenCLIP(device=args.device)
    bank, _ = build_text_bank(test.captions, num_tokens=config["num_text_tokens"], clip=clip, device=args.device)
    test_text = bank.pooled[test.caption_ids]
    real = clip.encode_images(lucid.tensor(test.images))
    train = read_manifest(Path(config["data"]))
    train_bank, _ = build_text_bank(
        train.captions, num_tokens=config["num_text_tokens"], clip=clip, device=args.device
    )
    references = {
        "real": real_reference(clip, train.images, real, test_text),
        "retrieval": retrieval_reference(
            clip, train.images, train_bank.pooled[train.caption_ids], test.images, real, test_text
        ),
    }
    for name, values in references.items():
        print(name, json.dumps(values), flush=True)
    for run in args.runs:
        result = evaluate_run(run, clip, test, real, test_text, bank, references, device=args.device)
        print(run, json.dumps({k: v for k, v in result.items() if k not in references}), flush=True)


if __name__ == "__main__":
    main()
