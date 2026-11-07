"""분포에 맞는 텍스트 찾기 (아이디어 문서 §6).

입력은 이미지 샘플 집합(목표 분포 p)과 후보 텍스트다.

- `alignment_scores`      : E_{y~p}[cos(φ(y), e(t))] - E_{y~p_all}[cos(φ(y), e(t))]  (PMI식 보정)
- `classification_scores` : E_{y~p}[log p(t|y)], p(t|y)는 후보에 대한 softmax
- `drift_scores`          : 텍스트 조건 생성기로 q_t를 만들어 E‖V_{p,q_t}‖²를 비교 (작을수록 좋음)
- `energy_scores`         : 같은 특징 공간의 energy distance (drift_scores의 대조군, 작을수록 좋음)
- `invert_embedding`      : e(t)를 연속 변수로 두고 drifting loss를 줄이는 textual inversion식 탐색

    python -m clipdrift.inverse --checkpoint runs/joint --out results/inverse.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lucid
import numpy as np
from lucid import Tensor

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.data import (
    BACKGROUNDS,
    COLORS,
    HELD_OUT,
    SHAPES,
    caption_for,
    make_shapes,
    render_shape,
)
from clipdrift.drift import DEFAULT_TEMPERATURES, DriftKernel, feature_drift_loss, normalized_drift
from clipdrift.evaluate import attribute_accuracy, generate
from clipdrift.features import pixel_features
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.mae import MAEFeatures, load_mae


def alignment_scores(
    target: np.ndarray, text: np.ndarray, background: np.ndarray | None = None
) -> np.ndarray:
    """`target`, `background`, `text`는 L2 정규화된 임베딩. 범용 캡션의 편향을 배경 평균으로 뺀다."""
    score = (target @ text.T).mean(0)
    if background is not None:
        score = score - (background @ text.T).mean(0)
    return score


def classification_scores(
    target: np.ndarray, text: np.ndarray, logit_scale: float = 100.0
) -> np.ndarray:
    logits = logit_scale * target @ text.T
    logits = logits - logits.max(1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return log_probs.mean(0)


def _as_group(features: dict[str, Tensor]) -> dict[str, Tensor]:
    """`[N, L, C]` -> `[L, N, C]`. 그룹이 하나뿐이라 G 축은 두지 않는다."""
    return {name: value.swapaxes(0, 1) for name, value in features.items()}


@lucid.no_grad()
def _reference_scales(target: dict[str, Tensor]) -> dict[str, Tensor]:
    """후보끼리 ‖V‖²를 비교하려면 S가 고정돼야 한다. 목표 분포 내부 거리로 정한다."""
    scales = {}
    for name, value in target.items():
        distances = lucid.cdist(value, value)
        count = int(value.shape[-2])
        mean = distances.sum() / (int(value.shape[0]) * count * (count - 1))
        scales[name] = lucid.clamp(mean / (int(value.shape[-1]) ** 0.5), min=1e-8)
    return scales


@lucid.no_grad()
def drift_scores(
    model,
    extract,
    target_images: Tensor,
    pooled: np.ndarray,
    tokens: np.ndarray,
    *,
    samples: int,
    alpha: float,
    device: str,
    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURES,
) -> np.ndarray:
    """후보별 E‖V_{p,q_t}‖²를 돌려준다. 값이 작을수록 잘 맞는다고 본다."""
    target = _as_group(extract(target_images))
    scales = _reference_scales(target)
    scores = np.zeros(len(pooled))
    for index in range(len(pooled)):
        images = generate(
            model,
            np.repeat(pooled[index][None], samples, axis=0),
            np.repeat(tokens[index][None], samples, axis=0),
            alpha=alpha,
            device=device,
        )
        generated = _as_group(extract(lucid.tensor(images, device=device)))
        total = 0.0
        for name, phi_x in generated.items():
            _, _, raw = normalized_drift(
                phi_x,
                target[name],
                None,
                DriftKernel(),
                temperatures,
                scale=scales[name],
                normalize_drift=False,
            )
            total += float(raw.item())
        scores[index] = total
    return scores


@lucid.no_grad()
def energy_scores(
    model,
    extract,
    target_images: Tensor,
    pooled: np.ndarray,
    tokens: np.ndarray,
    *,
    samples: int,
    alpha: float,
    device: str,
) -> np.ndarray:
    """정규화된 특징 공간의 energy distance 합. 값이 작을수록 잘 맞는다."""
    target = _as_group(extract(target_images))
    scales = _reference_scales(target)
    scores = np.zeros(len(pooled))
    for index in range(len(pooled)):
        images = generate(
            model,
            np.repeat(pooled[index][None], samples, axis=0),
            np.repeat(tokens[index][None], samples, axis=0),
            alpha=alpha,
            device=device,
        )
        generated = _as_group(extract(lucid.tensor(images, device=device)))
        total = 0.0
        for name, phi_x in generated.items():
            x, y = phi_x / scales[name], target[name] / scales[name]
            cross = lucid.cdist(x, y).mean()
            self_x = lucid.cdist(x, x).mean()
            self_y = lucid.cdist(y, y).mean()
            total += float((2.0 * cross - self_x - self_y).item()) / (int(x.shape[-1]) ** 0.5)
        scores[index] = total
    return scores


def invert_embedding(
    model,
    extract,
    target_images: Tensor,
    init_pooled: np.ndarray,
    init_tokens: np.ndarray,
    *,
    steps: int,
    lr: float,
    samples: int,
    alpha: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """프롬프트 임베딩을 직접 최적화한다. 생성기와 φ는 고정이다(검증되지 않은 확장)."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    pooled = lucid.tensor(init_pooled[None], device=device, requires_grad=True)
    tokens = lucid.tensor(init_tokens[None], device=device, requires_grad=True)
    optimizer = lucid.optim.Adam([pooled, tokens], lr=lr)
    with lucid.no_grad():
        target = _as_group(extract(target_images))
    alpha_tensor = lucid.full((samples,), float(alpha), device=device)
    for _ in range(steps):
        noise, style = model.sample_noise(samples, device)
        images = model(
            noise,
            style,
            pooled.repeat(samples, 1),
            tokens.repeat(samples, 1, 1),
            alpha_tensor,
        )
        total: Tensor | None = None
        for name, phi_x in _as_group(extract(images)).items():
            loss, _ = feature_drift_loss(phi_x, target[name])
            total = loss if total is None else total + loss
        assert total is not None
        total.eval()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    pooled.eval()
    tokens.eval()
    return pooled.detach().numpy()[0], tokens.detach().numpy()[0]


@dataclass(frozen=True)
class Target:
    color: str
    shape: str
    background: str


def _targets(held_out: bool) -> list[Target]:
    return [
        Target(color, shape, background)
        for color in COLORS
        for shape in SHAPES
        for background in BACKGROUNDS
        if ((color, shape) in HELD_OUT) == held_out
    ]


def _rank_of(scores: np.ndarray, index: int, *, smaller_is_better: bool) -> int:
    order = np.argsort(scores if smaller_is_better else -scores)
    return int(np.flatnonzero(order == index)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="generator run directory")
    parser.add_argument("--mae-checkpoint", type=Path, default=Path("checkpoints/mae_shapes.lucid"))
    parser.add_argument("--out", type=Path, default=Path("results/inverse.json"))
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--invert-steps", type=int, default=80)
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()
    device = args.device

    lucid.manual_seed(0)
    rng = np.random.default_rng(0)
    clip = FrozenCLIP(device=device)
    dataset = make_shapes(3000, seed=1)

    # 후보: 학습 조합(seen)과 held-out 조합 전체의 기본 템플릿 캡션.
    candidates = _targets(False) + _targets(True)
    captions = [caption_for(t.color, t.shape, t.background) for t in candidates]
    text, tokens = clip.encode_captions(captions, num_tokens=16)
    background = clip.encode_images(lucid.tensor(dataset.images[:512], device=device))

    results: dict[str, dict[str, list[int]]] = {
        split: {"alignment": [], "alignment_no_pmi": [], "classification": []}
        for split in ("seen", "held_out")
    }
    target_images: dict[Target, np.ndarray] = {}
    for index, target in enumerate(candidates):
        split = "held_out" if (target.color, target.shape) in HELD_OUT else "seen"
        images = np.stack(
            [
                render_shape(target.color, target.shape, target.background, rng=rng)
                for _ in range(args.target_size)
            ]
        )
        target_images[target] = images
        embed = clip.encode_images(lucid.tensor(images, device=device))
        results[split]["alignment"].append(
            _rank_of(alignment_scores(embed, text, background), index, smaller_is_better=False)
        )
        results[split]["alignment_no_pmi"].append(
            _rank_of(alignment_scores(embed, text), index, smaller_is_better=False)
        )
        results[split]["classification"].append(
            _rank_of(classification_scores(embed, text), index, smaller_is_better=False)
        )

    report: dict[str, object] = {"num_candidates": len(candidates)}
    if args.checkpoint is not None:
        # CLIP 임베딩은 이미 계산했으므로 내려서 메모리를 비운다(18GB 환경).
        del clip
        raw_config = json.loads((args.checkpoint / "generator.json").read_text())
        model = TextConditionedGenerator(GeneratorConfig(**raw_config))
        model.load_state_dict(lucid.load(args.checkpoint / "generator_ema.lucid"))
        model = model.to(device).eval()
        mae = MAEFeatures(load_mae(args.mae_checkpoint, device))

        def extract(images: Tensor) -> dict[str, Tensor]:
            return {**pixel_features(images), **mae(images)}

        for split in results:
            results[split]["drift"] = []
            results[split]["energy"] = []
        for index, target in enumerate(candidates):
            split = "held_out" if (target.color, target.shape) in HELD_OUT else "seen"
            images = lucid.tensor(target_images[target], device=device)
            common = dict(samples=args.samples, alpha=1.0, device=device)
            drift = drift_scores(model, extract, images, text, tokens, **common)
            energy = energy_scores(model, extract, images, text, tokens, **common)
            results[split]["drift"].append(_rank_of(drift, index, smaller_is_better=True))
            results[split]["energy"].append(_rank_of(energy, index, smaller_is_better=True))

        # textual inversion: 빈 프롬프트 임베딩에서 출발해 목표 분포로 옮겨 본다.
        inversion = []
        null_pooled, null_tokens = text.mean(0), tokens.mean(0)
        for target in candidates[:: max(len(candidates) // 6, 1)]:
            images = lucid.tensor(target_images[target], device=device)
            pooled_opt, tokens_opt = invert_embedding(
                model,
                extract,
                images,
                null_pooled,
                null_tokens,
                steps=args.invert_steps,
                lr=0.01,
                samples=args.samples,
                alpha=1.0,
                device=device,
            )
            def accuracy(pooled_value, tokens_value):
                generated = generate(
                    model,
                    np.repeat(pooled_value[None], 64, axis=0),
                    np.repeat(tokens_value[None], 64, axis=0),
                    alpha=1.0,
                    device=device,
                )
                return attribute_accuracy(generated, target.color, target.shape, target.background)

            inversion.append(
                {
                    "target": f"{target.color} {target.shape} / {target.background}",
                    "before": accuracy(null_pooled, null_tokens),
                    "after": accuracy(pooled_opt, tokens_opt),
                }
            )
        report["inversion"] = inversion

    def summarize(ranks: list[int]) -> dict[str, float]:
        array = np.array(ranks)
        return {
            "top1": float((array == 0).mean()),
            "top3": float((array < 3).mean()),
            "mean_rank": float(array.mean()),
        }

    report["ranking"] = {
        split: {method: summarize(ranks) for method, ranks in methods.items()}
        for split, methods in results.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
