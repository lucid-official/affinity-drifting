"""생성 이미지 평가와 시각화 (shapes 데이터 전용 정답 속성 사용).

저장된 실행을 다시 평가한다(`summary.json`을 갱신한다):

    python -m clipdrift.evaluate runs/pixel_joint runs/pixel_pseudo
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lucid
import numpy as np
from PIL import Image, ImageDraw

from clipdrift.data import (
    BACKGROUNDS,
    COLORS,
    HELD_OUT,
    SHAPES,
    TEMPLATES,
    caption_for,
    classify_shapes,
    load_dataset,
)
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.text_bank import TextBank, build_text_bank

# 학습 템플릿에 없는 표현. 텍스트 일반화 평가용이다.
UNSEEN_TEMPLATE = "a {background} backdrop behind one {color} {shape}"


@dataclass(frozen=True)
class EvalPrompt:
    text: str
    color: str
    shape: str
    background: str
    split: str  # seen | held_out_combo | unseen_wording


def eval_prompts() -> list[EvalPrompt]:
    prompts = []
    for color in COLORS:
        for shape in SHAPES:
            for background in BACKGROUNDS:
                held_out = (color, shape) in HELD_OUT
                split = "held_out_combo" if held_out else "seen"
                prompts.append(
                    EvalPrompt(
                        caption_for(color, shape, background, TEMPLATES[0]),
                        color,
                        shape,
                        background,
                        split,
                    )
                )
                if not held_out:
                    prompts.append(
                        EvalPrompt(
                            caption_for(color, shape, background, UNSEEN_TEMPLATE),
                            color,
                            shape,
                            background,
                            "unseen_wording",
                        )
                    )
    return prompts


@lucid.no_grad()
def generate(
    model,
    pooled: np.ndarray,
    tokens: np.ndarray,
    *,
    alpha: float,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """프롬프트 임베딩 배열별로 이미지 하나씩 생성한다."""
    chunks = []
    for start in range(0, len(pooled), batch_size):
        batch_pooled = lucid.tensor(pooled[start : start + batch_size], device=device)
        batch_tokens = lucid.tensor(tokens[start : start + batch_size], device=device)
        count = int(batch_pooled.shape[0])
        noise, style = model.sample_noise(count, device)
        alpha_tensor = lucid.full((count,), float(alpha), device=device)
        images = model(noise, style, batch_pooled, batch_tokens, alpha_tensor)
        images.eval()
        chunks.append(images.numpy())
    return np.concatenate(chunks)


def attribute_accuracy(
    images: np.ndarray, color: str, shape: str, background: str
) -> dict[str, float]:
    judged = classify_shapes(images)
    hit_color = np.array([c == color for c in judged["color"]])
    hit_shape = np.array([s == shape for s in judged["shape"]])
    hit_background = np.array([b == background for b in judged["background"]])
    centers = judged["centers"]
    valid = ~np.isnan(centers[:, 0])
    return {
        "color": float(hit_color.mean()),
        "shape": float(hit_shape.mean()),
        "background": float(hit_background.mean()),
        "joint": float((hit_color & hit_shape & hit_background).mean()),
        "center_std": float(np.nanstd(centers[valid], axis=0).mean()) if valid.sum() > 1 else 0.0,
        "coherence": float(judged["coherence"].mean()),
        "iou": float(judged["iou"].mean()),
        "area": float(judged["area"].mean()),
    }


def evaluate_shapes(
    model,
    bank: TextBank,
    *,
    alpha: float = 1.0,
    samples: int = 32,
    device: str,
) -> dict[str, dict[str, float]]:
    """split별 평균 속성 정확도와, 학습 조합의 프롬프트 도형별(`seen/circle` 등) 평균."""
    per_split: dict[str, list[dict[str, float]]] = {}
    for prompt in eval_prompts():
        index = bank.index(prompt.text)
        images = generate(
            model,
            np.repeat(bank.pooled[index][None], samples, axis=0),
            np.repeat(bank.tokens[index][None], samples, axis=0),
            alpha=alpha,
            device=device,
        )
        row = attribute_accuracy(images, prompt.color, prompt.shape, prompt.background)
        per_split.setdefault(prompt.split, []).append(row)
        if prompt.split == "seen":
            per_split.setdefault(f"seen/{prompt.shape}", []).append(row)
    return {
        split: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for split, rows in per_split.items()
    }


def real_baseline(dataset, *, samples: int = 2000) -> dict[str, object]:
    """규칙 판정기의 실제 이미지 값. 판정기 자체의 상한과 목표 분포를 보여 준다."""
    index = np.arange(min(samples, len(dataset.images)))
    judged = classify_shapes(dataset.images[index])
    attrs = [dataset.attributes[i] for i in index]
    hits = {
        key: np.array([judged[key][i] == attrs[i][key] for i in range(len(index))])
        for key in ("color", "shape", "background")
    }
    joint = hits["color"] & hits["shape"] & hits["background"]
    # 지터 범위가 같은 실제 데이터의 프롬프트 내 위치 표준편차(근사).
    centers = judged["centers"]
    labels = np.array([attr["shape"] for attr in attrs])
    return {
        **{key: float(value.mean()) for key, value in hits.items()},
        "joint": float(joint.mean()),
        "center_std": float(np.nanstd(centers, axis=0).mean()),
        "coherence": float(judged["coherence"].mean()),
        "iou": float(judged["iou"].mean()),
        "area_by_shape": {
            shape: float(judged["area"][labels == shape].mean()) for shape in SHAPES
        },
    }


def shape_summary(
    model, bank: TextBank, dataset, *, samples: int, device: str
) -> dict[str, object]:
    """실제 이미지 기준선과 α별 생성 평가를 합친다."""
    summary: dict[str, object] = {"real_baseline": real_baseline(dataset)}
    for alpha in (1.0, 2.0):
        summary[f"alpha={alpha}"] = evaluate_shapes(
            model, bank, alpha=alpha, samples=samples, device=device
        )
    return summary


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip((image + 1.0) * 0.5, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def save_grid(
    path,
    rows: list[tuple[str, np.ndarray]],
    *,
    scale: int = 2,
    label_width: int = 240,
) -> None:
    """(라벨, `[N, C, H, W]`) 행들을 격자 PNG로 저장한다."""
    size = rows[0][1].shape[-1] * scale
    columns = len(rows[0][1])
    canvas = Image.new("RGB", (label_width + columns * (size + 2), len(rows) * (size + 2)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, images) in enumerate(rows):
        y = r * (size + 2)
        draw.text((4, y + size // 2 - 5), label[:40], fill="black")
        for c, image in enumerate(images):
            tile = Image.fromarray(to_uint8(image)).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(tile, (label_width + c * (size + 2), y))
    canvas.save(path)


def reevaluate(run: Path, *, samples: int, device: str, seed: int = 0) -> dict[str, object]:
    """저장된 실행의 EMA 생성기를 현재 지표로 다시 평가해 `summary.json`을 갱신한다."""
    config = json.loads((run / "config.json").read_text())
    dataset = load_dataset(config["data"], count=config["dataset_size"], seed=config["seed"])
    prompts = list(dict.fromkeys([*dataset.captions, *[p.text for p in eval_prompts()]]))
    bank, _ = build_text_bank(prompts, num_tokens=config["num_text_tokens"], device=device)
    model = TextConditionedGenerator(GeneratorConfig(**json.loads((run / "generator.json").read_text())))
    model.load_state_dict(lucid.load(run / "generator_ema.lucid"))
    model = model.to(device).eval()
    lucid.manual_seed(seed)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(shape_summary(model, bank, dataset, samples=samples, device=device))
    path.write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved runs with the current metrics.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()
    for run in args.runs:
        reevaluate(run, samples=args.samples, device=args.device)
        print(f"updated {run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
