"""텍스트 조건 drifting 학습 루프 (논문 A.9, 아이디어 문서 §3, §5).

    # 로드맵 1단계: 의사 클래스
    python -m clipdrift.train --mode pseudo --features pixel,mae --out runs/pseudo
    # 로드맵 2단계: soft joint kernel + 텍스트 kNN 큐
    python -m clipdrift.train --mode joint --features pixel,mae --out runs/joint
    # 로드맵 3단계: 고정 CLIP 특징 추가
    python -m clipdrift.train --mode joint --features pixel,mae,clip --out runs/joint_clip
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lucid
import lucid.nn as nn
import numpy as np
from lucid import Tensor
from lucid.optim.swa_utils import AveragedModel

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.data import load_dataset
from clipdrift.drift import (
    DEFAULT_TEMPERATURES,
    DriftKernel,
    cfg_log_weights,
    feature_drift_loss,
    sample_alpha,
)
from clipdrift.evaluate import eval_prompts, generate, save_grid, shape_summary
from clipdrift.features import pixel_features, subsample_locations
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.mae import MAEFeatures, load_mae
from clipdrift.sampling import (
    BlindSource,
    Groups,
    KnnSource,
    PairSource,
    PseudoClassSource,
    TextKernel,
    merge_cfg,
    sample_groups,
    text_kernel_for_groups,
)
from clipdrift.text_bank import TextBank, build_text_bank


@dataclass
class TrainConfig:
    data: str = "shapes"
    dataset_size: int = 12000
    mode: str = "joint"  # blind | pair | pseudo | joint
    features: str = "pixel,mae"  # pixel, mae, clip의 조합
    pixel_weight: float = 1.0  # 픽셀 loss 배율. φ의 이미지 공간 기울기가 픽셀 항의 수백 배라 균형을 맞출 때 쓴다.
    mae_checkpoint: str = "checkpoints/mae_shapes.lucid"
    out: str = "runs/default"

    steps: int = 3000
    num_groups: int = 4  # 논문 Nc
    num_neg: int = 16  # 논문 Nneg
    num_pos: int = 64  # 논문 Npos (ablation 기본값). 줄이면 V가 흔들려 이미지가 조각난다.
    num_unc: int = 4  # 논문 Nuncond
    p_null: float = 0.1  # 텍스트 드롭아웃
    max_locations: int = 32  # 특징 하나당 쓰는 위치 수 상한. 0이면 전부(논문).

    tau_t: float = 0.2  # joint kernel의 텍스트 temperature. 분포의 입도를 정한다.
    text_shrinkage: float = -1.0  # 0 이상이면 텍스트 키를 whitening한다. 기본은 끔(TextKernel 참고).
    text_rank: int = 0  # 양수면 텍스트 키를 상위 PCA 축으로 줄인다.
    num_clusters: int = 20  # pseudo 모드의 의사 클래스 수
    neighbors: int = 256  # joint 모드의 kNN 크기
    queue_size: int = 8192

    lr: float = 4e-4
    warmup: int = 300
    weight_decay: float = 0.01
    grad_clip: float = 2.0
    ema: float = 0.999

    hidden: int = 128
    depth: int = 3
    heads: int = 4
    num_text_tokens: int = 16

    device: str = "metal"
    seed: int = 0
    log_every: int = 50
    sample_every: int = 500
    eval_samples: int = 32


def to_grouped(features: dict[str, Tensor], groups: int, count: int) -> dict[str, Tensor]:
    """`{이름: [G*n, L, C]}` -> `{이름: [G, L, n, C]}`."""
    out = {}
    for name, value in features.items():
        locations, dim = int(value.shape[1]), int(value.shape[2])
        out[name] = value.reshape(groups, count, locations, dim).swapaxes(1, 2)
    return out


def extractors_for(config: TrainConfig, clip: FrozenCLIP | None):
    """설정에 맞는 φ 목록을 만든다. CLIP은 함께 쓸 때만 메모리에 남긴다."""
    names = [name.strip() for name in config.features.split(",") if name.strip()]
    extractors = []
    for name in names:
        if name == "pixel":
            extractors.append(pixel_features)
        elif name == "mae":
            extractors.append(MAEFeatures(load_mae(Path(config.mae_checkpoint), config.device)))
        elif name == "clip":
            if clip is None:
                clip = FrozenCLIP(device=config.device)
            extractors.append(clip.features)
        else:
            raise ValueError(f"unknown feature space: {name}")
    return extractors, clip


def extract_all(extractors, images: Tensor) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    for extractor in extractors:
        merged.update(extractor(images))
    return merged


def make_source(config: TrainConfig, keys: np.ndarray, rng: np.random.Generator):
    if config.mode == "blind":
        return BlindSource(len(keys))
    if config.mode == "pair":
        return PairSource(len(keys))
    if config.mode == "pseudo":
        return PseudoClassSource(keys, config.num_clusters, config.seed)
    if config.mode == "joint":
        source = KnnSource(keys, neighbors=config.neighbors, queue_size=config.queue_size)
        source.warm_up(rng)
        return source
    raise ValueError(f"unknown mode: {config.mode}")


def batch_kernel(
    source,
    keys: np.ndarray,
    groups: Groups,
    alpha_grouped: Tensor,
    config: TrainConfig,
) -> DriftKernel:
    kernel = DriftKernel()
    if source.uses_text_kernel:
        kernel = text_kernel_for_groups(keys, groups, config.tau_t, device=config.device)
    if config.num_unc > 0:
        log_w = cfg_log_weights(alpha_grouped, kernel.log_k_neg, config.num_neg, config.num_unc)
        kernel = merge_cfg(kernel, log_w)
    return kernel


def training_step(
    generator: TextConditionedGenerator,
    extractors,
    images: np.ndarray,
    caption_ids: np.ndarray,
    bank: TextBank,
    keys: np.ndarray,
    source,
    config: TrainConfig,
    rng: np.random.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    device = config.device
    groups = sample_groups(
        source,
        size=len(images),
        num_groups=config.num_groups,
        num_pos=config.num_pos,
        num_neg=config.num_neg,
        num_unc=config.num_unc,
        p_null=config.p_null,
        rng=rng,
    )
    g, n, p = config.num_groups, config.num_neg, config.num_pos

    # 프롬프트: 데이터 인덱스 -> 캡션 인덱스. 비조건 그룹은 빈 텍스트(마지막 행)를 쓴다.
    prompt_data = groups.prompts.reshape(-1)
    prompt_ids = np.where(prompt_data >= 0, caption_ids[np.maximum(prompt_data, 0)], -1)
    pooled, tokens = bank.tensors(prompt_ids, device)

    # α는 그룹별로 뽑는다(논문 A.9 2번). 비조건 그룹은 α=1이라 w=0이다.
    alpha_group = sample_alpha(g).to(device)
    if groups.null.any():
        conditional = lucid.tensor((~groups.null).astype(np.float32), device=device)
        alpha_group = 1.0 + (alpha_group - 1.0) * conditional
    alpha_grouped = alpha_group.reshape(g, 1).repeat(1, n)
    alpha = alpha_grouped.reshape(-1)

    noise, style = generator.sample_noise(g * n, device)
    generated = generator(noise, style, pooled, tokens, alpha)

    features_x = to_grouped(extract_all(extractors, generated), g, n)
    with lucid.no_grad():
        positive = lucid.tensor(images[groups.positives.reshape(-1)], device=device)
        features_pos = to_grouped(extract_all(extractors, positive), g, p)
        features_unc = None
        if groups.unconditional is not None:
            unconditional = lucid.tensor(images[groups.unconditional.reshape(-1)], device=device)
            features_unc = to_grouped(extract_all(extractors, unconditional), g, config.num_unc)

    kernel = batch_kernel(source, keys, groups, alpha_grouped, config)
    sets = (features_x, features_pos) if features_unc is None else (features_x, features_pos, features_unc)
    sets = subsample_locations(sets, config.max_locations or None)

    total: Tensor | None = None
    raw: dict[str, Tensor] = {}
    for name, phi_x in sets[0].items():
        phi_unc = sets[2][name] if len(sets) == 3 else None
        loss, stats = feature_drift_loss(
            phi_x, sets[1][name], phi_unc, kernel, DEFAULT_TEMPERATURES
        )
        family = name.split("/")[0]
        if family == "pixel":
            loss = loss * config.pixel_weight
        total = loss if total is None else total + loss
        raw[family] = stats.raw_drift if family not in raw else raw[family] + stats.raw_drift
    assert total is not None
    return total, raw


def _write_samples(model, bank: TextBank, step: int, out: Path, config: TrainConfig) -> None:
    rows = []
    for prompt in eval_prompts():
        if prompt.split == "unseen_wording":
            continue
        index = bank.index(prompt.text)
        count = 10
        images = generate(
            model,
            np.repeat(bank.pooled[index][None], count, axis=0),
            np.repeat(bank.tokens[index][None], count, axis=0),
            alpha=1.0,
            device=config.device,
        )
        tag = "*" if prompt.split == "held_out_combo" else ""
        rows.append((f"{tag}{prompt.color} {prompt.shape} / {prompt.background}", images))
    save_grid(out / f"samples_{step:06d}.png", rows)


def train(config: TrainConfig) -> dict[str, object]:
    lucid.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=1))

    dataset = load_dataset(config.data, count=config.dataset_size, seed=config.seed)
    shapes = dataset.attributes is not None
    extra = [prompt.text for prompt in eval_prompts()] if shapes else []
    prompt_list = list(dict.fromkeys([*dataset.captions, *extra]))
    caption_lookup = {text: i for i, text in enumerate(prompt_list)}
    caption_ids = np.array([caption_lookup[dataset.captions[c]] for c in dataset.caption_ids])

    bank, clip = build_text_bank(
        prompt_list, num_tokens=config.num_text_tokens, device=config.device
    )
    # 키 변환은 학습 캡션으로만 맞춘다. 평가 프롬프트(held-out)가 새어 들어가면 안 된다.
    kernel_fit = TextKernel.fit(
        bank.pooled[np.unique(caption_ids)],
        shrinkage=config.text_shrinkage if config.text_shrinkage >= 0 else None,
        rank=config.text_rank or None,
    )
    keys = kernel_fit.transform(bank.pooled[caption_ids])

    extractors, clip = extractors_for(config, clip)
    if clip is not None and "clip" not in config.features:
        # 텍스트 인코딩만 쓰고 메모리를 비운다(18GB 환경 주의사항).
        clip = None
        gc.collect()

    source = make_source(config, keys, rng)
    gen_config = GeneratorConfig(
        image_size=dataset.images.shape[-1],
        channels=dataset.images.shape[1],
        hidden=config.hidden,
        depth=config.depth,
        heads=config.heads,
        text_pooled_dim=bank.pooled.shape[-1],
        text_token_dim=bank.tokens.shape[-1],
        num_text_tokens=config.num_text_tokens,
    )
    generator = TextConditionedGenerator(gen_config).to(config.device)
    optimizer = lucid.optim.AdamW(
        generator.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay
    )
    scheduler = lucid.optim.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(config.warmup, 1))
    )

    def ema_update(average: Tensor, current: Tensor, count: Tensor | int) -> Tensor:
        # 초반에는 decay를 낮춘다. 0.999로 바로 시작하면 1500스텝 뒤에도 초기 가중치가 22% 남는다.
        steps = int(count)
        decay = min(config.ema, (1.0 + steps) / (10.0 + steps))
        return decay * average + (1.0 - decay) * current

    ema = AveragedModel(generator, avg_fn=ema_update)
    params_ema = list(ema.parameters())

    log_path = out / "log.jsonl"
    log_path.write_text("")
    started = time.time()
    for step in range(1, config.steps + 1):
        loss, raw = training_step(
            generator, extractors, dataset.images, caption_ids, bank, keys, source, config, rng
        )
        loss.eval()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update_parameters(generator)
        # EMA는 loss에 묶이지 않아 평가하지 않으면 지연 그래프가 계속 자란다.
        for parameter in params_ema:
            parameter.eval()

        if step % config.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
                "seconds": round(time.time() - started, 1),
                **{f"raw/{key}": float(value.item()) for key, value in raw.items()},
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row, flush=True)
        if shapes and (step % config.sample_every == 0 or step == config.steps):
            _write_samples(ema.module, bank, step, out, config)

    lucid.save(ema.module.state_dict(), out / "generator_ema.lucid")
    lucid.save(generator.state_dict(), out / "generator_online.lucid")
    (out / "generator.json").write_text(json.dumps(gen_config.to_dict()))

    summary: dict[str, object] = {"seconds": round(time.time() - started, 1), "steps": config.steps}
    if shapes:
        summary.update(
            shape_summary(
                ema.module, bank, dataset, samples=config.eval_samples, device=config.device
            )
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for field, default in asdict(TrainConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
