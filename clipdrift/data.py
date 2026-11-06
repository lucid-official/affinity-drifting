"""(이미지, 캡션) 데이터셋.

- `shapes`: 검증용 합성 데이터. 정답 속성이 있어 조건 정확도를 규칙으로 잴 수 있고, 일부 색-도형 조합을
  학습에서 빼 두어 텍스트 일반화도 잰다. 이미지 하나에 캡션 하나로, 아이디어 문서 §1의 상황을 만든다.
- 그 외 경로: JSONL 매니페스트 `{"image": 경로, "caption": 문장}` (실제 데이터용).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.12, 0.12),
    "green": (0.10, 0.68, 0.22),
    "blue": (0.15, 0.25, 0.90),
    "yellow": (0.95, 0.84, 0.10),
}
SHAPES = ("circle", "square", "triangle")
BACKGROUNDS: dict[str, tuple[float, float, float]] = {
    "black": (0.05, 0.05, 0.05),
    "white": (0.95, 0.95, 0.95),
}
# 학습에서 제외하는 색-도형 조합. 평가에서만 프롬프트로 쓴다.
HELD_OUT: tuple[tuple[str, str], ...] = (("yellow", "triangle"), ("blue", "circle"))
TEMPLATES = (
    "a {color} {shape} on a {background} background",
    "a photo of a {color} {shape} on {background}",
    "an image showing a {color} {shape} against a {background} background",
    "{background} background with a {color} {shape}",
    "a simple drawing of a {color} {shape}, {background} background",
)


@dataclass(frozen=True)
class CaptionDataset:
    images: np.ndarray  # [M, C, H, W], float32, [-1, 1]
    captions: list[str]  # 전체 고유 캡션
    caption_ids: np.ndarray  # [M], captions의 인덱스
    attributes: list[dict[str, str]] | None = None  # shapes 전용


def render_shape(
    color: str,
    shape: str,
    background: str,
    *,
    rng: np.random.Generator,
    size: int = 32,
    supersample: int = 4,
) -> np.ndarray:
    """위치·크기·톤에 지터를 준 도형을 안티앨리어싱으로 그린다. `[3, H, W]`, `[-1, 1]`."""
    fine = size * supersample
    coord = (np.arange(fine) + 0.5) / fine
    xx, yy = np.meshgrid(coord, coord)
    radius = rng.uniform(0.16, 0.28)
    cx = rng.uniform(radius + 0.04, 1 - radius - 0.04)
    cy = rng.uniform(radius + 0.04, 1 - radius - 0.04)
    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    elif shape == "square":
        mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    elif shape == "triangle":
        # 위로 향한 삼각형. 경계 상자를 채우는 비율이 0.5로 원·사각형과 구분된다.
        height = (yy - (cy - radius)) / (2 * radius)
        mask = (height >= 0) & (height <= 1) & (np.abs(xx - cx) <= radius * height)
    else:
        raise ValueError(shape)
    tone = rng.uniform(-0.04, 0.04)
    fg = np.clip(np.array(COLORS[color]) + tone, 0, 1)
    bg = np.array(BACKGROUNDS[background])
    canvas = np.where(mask[None], fg[:, None, None], bg[:, None, None])
    coarse = canvas.reshape(3, size, supersample, size, supersample).mean(axis=(2, 4))
    return (coarse * 2.0 - 1.0).astype(np.float32)


def caption_for(color: str, shape: str, background: str, template: str = TEMPLATES[0]) -> str:
    return template.format(color=color, shape=shape, background=background)


def make_shapes(count: int = 12000, *, seed: int = 0, size: int = 32) -> CaptionDataset:
    rng = np.random.default_rng(seed)
    combos = [
        (color, shape, background)
        for color in COLORS
        for shape in SHAPES
        for background in BACKGROUNDS
        if (color, shape) not in HELD_OUT
    ]
    images = np.zeros((count, 3, size, size), dtype=np.float32)
    captions: dict[str, int] = {}
    caption_ids = np.zeros(count, dtype=np.int64)
    attributes = []
    for index in range(count):
        color, shape, background = combos[int(rng.integers(len(combos)))]
        images[index] = render_shape(color, shape, background, rng=rng, size=size)
        caption = caption_for(color, shape, background, TEMPLATES[int(rng.integers(len(TEMPLATES)))])
        caption_ids[index] = captions.setdefault(caption, len(captions))
        attributes.append({"color": color, "shape": shape, "background": background})
    return CaptionDataset(images, list(captions), caption_ids, attributes)


def read_manifest(path: Path, *, size: int = 32) -> CaptionDataset:
    """JSONL 매니페스트를 읽어 중앙 크롭 + 리사이즈한 이미지 배열로 올린다."""
    from PIL import Image

    images, caption_ids, captions = [], [], {}
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        with Image.open(base / row["image"]) as image:
            image = image.convert("RGB")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        images.append(array)
        caption_ids.append(captions.setdefault(row["caption"], len(captions)))
    if not images:
        raise ValueError(f"{path} has no samples")
    return CaptionDataset(np.stack(images), list(captions), np.array(caption_ids), None)


def load_dataset(name: str, *, count: int = 12000, seed: int = 0, size: int = 32) -> CaptionDataset:
    if name == "shapes":
        return make_shapes(count, seed=seed, size=size)
    return read_manifest(Path(name), size=size)


def classify_shapes(images: np.ndarray) -> dict[str, list[str] | np.ndarray]:
    """생성/실제 이미지를 규칙으로 속성 판정한다. 입력은 `[M, 3, H, W]`, `[-1, 1]`.

    배경은 네 모서리, 전경은 배경과 충분히 다른 픽셀이다. 도형은 가장 큰 연결 덩어리의 경계 상자 채움 비율로
    나눈다(사각형 ≈1, 원 ≈0.79, 삼각형 ≈0.5). 전경이 거의 없으면 "none"으로 판정해 오답으로 처리한다.
    `coherence`는 전경 중 가장 큰 덩어리의 비율이다. 조각난 이미지는 여기서 드러나므로 도형 정확도와 함께 본다.
    """
    rgb = np.clip((images + 1.0) * 0.5, 0, 1)
    names, palette = list(COLORS), np.array(list(COLORS.values()))
    bg_names, bg_palette = list(BACKGROUNDS), np.array(list(BACKGROUNDS.values()))
    result: dict[str, list[str] | np.ndarray] = {"color": [], "shape": [], "background": []}
    centers = np.full((len(images), 2), np.nan)
    coherence = np.zeros(len(images))
    size = images.shape[-1]
    for index, image in enumerate(rgb):
        corners = np.stack(
            [image[:, 0, 0], image[:, 0, -1], image[:, -1, 0], image[:, -1, -1]]
        ).mean(0)
        background = int(np.argmin(((bg_palette - corners) ** 2).sum(-1)))
        result["background"].append(bg_names[background])
        mask = np.sqrt(((image - bg_palette[background][:, None, None]) ** 2).sum(0)) > 0.35
        if mask.sum() < size * size * 0.01:
            result["color"].append("none")
            result["shape"].append("none")
            continue
        labels, _ = ndimage.label(mask)
        main = labels == np.bincount(labels[labels > 0]).argmax()
        coherence[index] = main.sum() / mask.sum()
        foreground = np.median(image[:, main], axis=1)
        result["color"].append(names[int(np.argmin(((palette - foreground) ** 2).sum(-1)))])
        rows, cols = np.flatnonzero(main.any(1)), np.flatnonzero(main.any(0))
        area = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
        fill = main.sum() / area
        result["shape"].append("square" if fill > 0.89 else "circle" if fill > 0.64 else "triangle")
        grid_y, grid_x = np.nonzero(main)
        centers[index] = grid_x.mean() / size, grid_y.mean() / size
    result["centers"] = centers
    result["coherence"] = coherence
    return result
