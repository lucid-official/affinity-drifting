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


def shape_mask(
    shape: str, cx: float, cy: float, radius: float, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    """단위 정사각형 좌표에서 도형 마스크를 만든다. 삼각형은 위를 향한다."""
    if shape == "circle":
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    if shape == "square":
        return (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    if shape == "triangle":
        height = (yy - (cy - radius)) / (2 * radius)
        return (height >= 0) & (height <= 1) & (np.abs(xx - cx) <= radius * height)
    raise ValueError(shape)


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
    mask = shape_mask(shape, cx, cy, radius, xx, yy)
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


_AREA_FACTOR = {"circle": np.pi, "square": 4.0, "triangle": 2.0}  # 면적 = 계수 × 반지름²


def _best_template(blob: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> tuple[str, float]:
    """덩어리와 가장 잘 겹치는 도형과 그 IoU를 찾는다.

    템플릿마다 면적과 무게중심을 덩어리에 맞춘 뒤 비교하므로 크기와 위치가 아니라 윤곽만으로 판정한다.
    경계 상자 채움 비율과 달리 거친 모서리에 강하다. 그 방식에서는 생성된 사각형의 90% 이상이 원으로 판정됐다.
    """
    pixels = blob.sum()
    area = pixels / blob.size
    center_x, center_y = (xx * blob).sum() / pixels, (yy * blob).sum() / pixels
    step = 1.0 / blob.shape[0]
    offsets = np.arange(-2, 3) * step
    best_shape, best_iou = SHAPES[0], -1.0
    for shape in SHAPES:
        for scale in (0.9, 1.0, 1.1):
            radius = float(np.sqrt(area / _AREA_FACTOR[shape])) * scale
            # 삼각형의 무게중심은 cy보다 r/3 아래에 있다.
            base_y = center_y - (radius / 3 if shape == "triangle" else 0.0)
            for dy in offsets:
                for dx in offsets:
                    template = shape_mask(shape, center_x + dx, base_y + dy, radius, xx, yy)
                    iou = (template & blob).sum() / (template | blob).sum()
                    if iou > best_iou:
                        best_shape, best_iou = shape, float(iou)
    return best_shape, best_iou


def classify_shapes(images: np.ndarray) -> dict[str, list[str] | np.ndarray]:
    """생성/실제 이미지를 규칙으로 속성 판정한다. 입력은 `[M, 3, H, W]`, `[-1, 1]`.

    배경은 네 모서리, 전경은 배경과 충분히 다른 픽셀이다. 도형은 가장 큰 연결 덩어리와 템플릿의 IoU로 고른다.
    전경이 거의 없으면 "none"으로 판정해 오답으로 처리한다.

    `coherence`는 전경 중 가장 큰 덩어리의 비율, `iou`는 고른 템플릿과의 겹침, `area`는 전경 면적 비율이다.
    조각나거나 형태가 뭉개진 이미지는 여기서 드러나므로 도형 정확도와 함께 본다.
    """
    rgb = np.clip((images + 1.0) * 0.5, 0, 1)
    names, palette = list(COLORS), np.array(list(COLORS.values()))
    bg_names, bg_palette = list(BACKGROUNDS), np.array(list(BACKGROUNDS.values()))
    result: dict[str, list[str] | np.ndarray] = {"color": [], "shape": [], "background": []}
    centers = np.full((len(images), 2), np.nan)
    coherence = np.zeros(len(images))
    ious = np.zeros(len(images))
    areas = np.zeros(len(images))
    size = images.shape[-1]
    grid = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(grid, grid)
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
        shape, ious[index] = _best_template(main, xx, yy)
        result["shape"].append(shape)
        areas[index] = mask.mean()
        grid_y, grid_x = np.nonzero(main)
        centers[index] = grid_x.mean() / size, grid_y.mean() / size
    result["centers"] = centers
    result["coherence"] = coherence
    result["iou"] = ious
    result["area"] = areas
    return result
