"""생성 이미지 평가와 시각화 (shapes 데이터 전용 정답 속성 사용)."""

from dataclasses import dataclass

import lucid
import numpy as np
from PIL import Image, ImageDraw

from clipdrift.data import BACKGROUNDS, COLORS, HELD_OUT, SHAPES, TEMPLATES, caption_for, classify_shapes

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
    }


def evaluate_shapes(
    model,
    bank_lookup,
    *,
    alpha: float = 1.0,
    samples: int = 32,
    device: str,
) -> dict[str, dict[str, float]]:
    """split별 평균 속성 정확도. `bank_lookup(text) -> (pooled, tokens)`."""
    per_split: dict[str, list[dict[str, float]]] = {}
    for prompt in eval_prompts():
        pooled, tokens = bank_lookup(prompt.text)
        images = generate(
            model,
            np.repeat(pooled[None], samples, axis=0),
            np.repeat(tokens[None], samples, axis=0),
            alpha=alpha,
            device=device,
        )
        per_split.setdefault(prompt.split, []).append(
            attribute_accuracy(images, prompt.color, prompt.shape, prompt.background)
        )
    return {
        split: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for split, rows in per_split.items()
    }


def real_baseline(dataset, *, samples: int = 2000) -> dict[str, float]:
    """규칙 판정기의 실제 이미지 정확도. 판정기 자체의 상한을 보여 준다."""
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
    return {
        **{key: float(value.mean()) for key, value in hits.items()},
        "joint": float(joint.mean()),
        "center_std": float(np.nanstd(centers, axis=0).mean()),
        "coherence": float(judged["coherence"].mean()),
    }


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
