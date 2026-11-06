"""ResNet-style MAE (논문 A.3). Lucid에는 없어서 논문 설명대로 구현했다.

- 인코더: 3×3 conv + basic block [3, 4, 6, 3], GroupNorm, stage 2~4 첫 블록에서 stride 2
- 디코더: U-Net식 (bilinear 업샘플 -> skip 연결 -> GN -> 3×3 conv 2개)
- 마스킹: 2×2 패치를 50% 확률로 0으로, 마스크 영역 ℓ2 복원 loss

논문 입력은 32×32(잠재 공간 또는 8×8 패치화한 픽셀)이다. 여기서는 32×32 픽셀을 패치화 없이 바로 넣는다.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import numpy as np
from lucid import Tensor

from clipdrift.features import freeze, map_vectors


@dataclass(frozen=True)
class MAEConfig:
    in_channels: int = 3
    base_width: int = 32  # 논문은 256 또는 640. 이 하드웨어와 32×32 데이터에 맞춰 줄였다.
    blocks: tuple[int, ...] = (3, 4, 6, 3)
    groups: int = 8


def _conv_norm(in_c: int, out_c: int, groups: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(groups, out_c)
    )


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, groups: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv_norm(in_c, out_c, groups, stride), nn.ReLU(), _conv_norm(out_c, out_c, groups)
        )
        self.skip = (
            nn.Identity()
            if in_c == out_c and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.GroupNorm(groups, out_c)
            )
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.body(x) + self.skip(x))


class ResNetEncoder(nn.Module):
    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        width = config.base_width
        self.stem = nn.Sequential(
            _conv_norm(config.in_channels, width, config.groups), nn.ReLU()
        )
        stages = []
        in_c = width
        for stage, count in enumerate(config.blocks):
            out_c = width * 2**stage
            blocks = [
                BasicBlock(in_c if i == 0 else out_c, out_c, config.groups, 2 if stage and i == 0 else 1)
                for i in range(count)
            ]
            stages.append(nn.ModuleList(blocks))
            in_c = out_c
        self.stages = nn.ModuleList(stages)

    @override
    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """(stage별 출력, A.5의 "2 블록마다 + 마지막" 특징 맵 목록)."""
        x = self.stem(x)
        stage_outputs, tapped = [], []
        for stage in self.stages:
            for index, block in enumerate(stage, start=1):
                x = block(x)
                if index % 2 == 0 or index == len(stage):
                    tapped.append(x)
            stage_outputs.append(x)
        return stage_outputs, tapped


class UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, in_c + skip_c)
        self.body = nn.Sequential(
            _conv_norm(in_c + skip_c, out_c, groups),
            nn.ReLU(),
            _conv_norm(out_c, out_c, groups),
            nn.ReLU(),
        )

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=tuple(int(v) for v in skip.shape[-2:]), mode="bilinear", align_corners=False)
        return self.body(self.norm(lucid.cat([x, skip], dim=1)))


class ResNetMAE(nn.Module):
    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ResNetEncoder(config)
        width, groups = config.base_width, config.groups
        top = width * 2 ** (len(config.blocks) - 1)
        self.bottleneck = nn.Sequential(_conv_norm(top, top, groups), nn.ReLU())
        ups = []
        in_c = top
        for stage in range(len(config.blocks) - 2, -1, -1):
            out_c = width * 2**stage
            ups.append(UpBlock(in_c, out_c, out_c, groups))
            in_c = out_c
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(in_c, config.in_channels, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        outputs, _ = self.encoder(x)
        y = self.bottleneck(outputs[-1])
        for up, skip in zip(self.ups, reversed(outputs[:-1])):
            y = up(y, skip)
        return self.head(y)


def random_mask(images: Tensor, *, patch: int = 2, ratio: float = 0.5) -> Tensor:
    """마스크 `[B, 1, H, W]`. 1이면 가려진 위치. 2×2 패치를 독립적으로 가린다."""
    batch, _, height, width = (int(v) for v in images.shape)
    drawn = lucid.rand(batch, 1, height // patch, width // patch, device=images.device.type)
    coarse = (drawn < ratio).to(images.dtype)
    return F.interpolate(coarse, size=(height, width), mode="nearest")


def mae_loss(model: ResNetMAE, images: Tensor) -> Tensor:
    mask = random_mask(images)
    recon = model(images * (1.0 - mask))
    per_pixel = ((recon - images) ** 2).mean(1, keepdim=True)
    return (per_pixel * mask).sum() / lucid.clamp(mask.sum(), min=1.0)


class MAEFeatures:
    """동결한 MAE 인코더를 φ로 쓴다. 기울기는 입력 이미지로 흐른다."""

    def __init__(self, model: ResNetMAE) -> None:
        self.encoder = freeze(model.encoder)

    def __call__(self, images: Tensor) -> dict[str, Tensor]:
        _, tapped = self.encoder(images)
        out: dict[str, Tensor] = {}
        for index, feature_map in enumerate(tapped):
            map_vectors(f"mae/{index}", feature_map, out)
        return out


def save_mae(model: ResNetMAE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lucid.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(asdict(model.config)))


def load_mae(path: Path, device: str) -> ResNetMAE:
    raw = json.loads(path.with_suffix(".json").read_text())
    raw["blocks"] = tuple(raw["blocks"])
    model = ResNetMAE(MAEConfig(**raw))
    model.load_state_dict(lucid.load(path))
    return model.to(device).eval()


def train_mae(
    images: np.ndarray,
    *,
    config: MAEConfig,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int = 0,
    log_every: int = 100,
) -> ResNetMAE:
    """`images`는 `[M, C, H, W]`, `[-1, 1]` 범위 float32 배열이다."""
    lucid.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ResNetMAE(config).to(device)
    optimizer = lucid.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    started = time.time()
    for step in range(1, steps + 1):
        index = rng.integers(len(images), size=batch_size)
        batch = lucid.tensor(images[index], device=device)
        loss = mae_loss(model, batch)
        loss.eval()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            print(f"mae step={step} loss={loss.item():.5f} ({time.time() - started:.0f}s)", flush=True)
    return model.eval()


def main() -> None:
    from clipdrift.data import load_dataset

    parser = argparse.ArgumentParser(description="Train the ResNet-style MAE feature encoder.")
    parser.add_argument("--data", default="shapes")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/mae_shapes.lucid"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()
    dataset = load_dataset(args.data)
    model = train_mae(
        dataset.images,
        config=MAEConfig(in_channels=dataset.images.shape[1], base_width=args.base_width),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    save_mae(model, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
