"""텍스트 조건 1-step 생성기 `f(ε, e(t), α) -> x` (논문 A.2, 아이디어 문서 §3.1).

블록은 Lucid의 `DiTBlock`/`DiTFinalLayer`(adaLN-Zero)를 재사용한다.
- pooled 텍스트 벡터 + α + style 임베딩 -> adaLN 조건
- 텍스트 토큰 시퀀스 + 학습되는 register 토큰 -> in-context 토큰

논문의 SwiGLU, RoPE, RMSNorm, QK-Norm은 쓰지 않는다. Lucid DiTBlock의 표준 구성을 그대로 쓴다.
"""

from dataclasses import asdict, dataclass
from typing import cast, override

import lucid
import lucid.nn as nn
from lucid import Tensor
from lucid.models.generative._common._transformers import (
    DiTBlock,
    DiTFinalLayer,
    sincos_position_embedding,
    timestep_embedding,
    unpatchify,
)


@dataclass(frozen=True)
class GeneratorConfig:
    image_size: int = 32
    channels: int = 3
    patch_size: int = 4
    hidden: int = 256
    depth: int = 6
    heads: int = 4
    mlp_ratio: float = 4.0
    text_pooled_dim: int = 512
    text_token_dim: int = 512
    num_text_tokens: int = 16
    num_registers: int = 8
    num_style_tokens: int = 32  # 논문 A.2
    style_codebook: int = 64

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TextConditionedGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.grid = config.image_size // config.patch_size
        self.patch_embed = nn.Conv2d(
            config.channels, hidden, config.patch_size, stride=config.patch_size
        )
        self.register_buffer("pos_embed", sincos_position_embedding(hidden, self.grid))
        self.pooled_proj = nn.Sequential(
            nn.Linear(config.text_pooled_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.alpha_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.style = nn.Embedding(config.style_codebook, hidden)
        self.token_proj = nn.Linear(config.text_token_dim, hidden)
        self.text_pos = nn.Parameter(lucid.randn(config.num_text_tokens, hidden) * 0.02)
        self.registers = nn.Parameter(lucid.randn(config.num_registers, hidden) * 0.02)
        self.register_cond = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, config.heads, config.mlp_ratio) for _ in range(config.depth)]
        )
        self.final = DiTFinalLayer(hidden, config.patch_size, config.channels)

        # 패치별 독립 노이즈만으로는 "도형 전체의 위치"처럼 이미지 단위의 무작위성을 만들기 어렵다.
        # style 임베딩이 그 전역 노이즈이므로 처음부터 유효한 크기로 둔다(0.02에서는 조각난 이미지가 나왔다).
        nn.init.normal_(self.style.weight, std=0.2)
        for block in self.blocks:
            cast(DiTBlock, block).zero_conditioning()
        self.final.zero_conditioning()
        # DiT는 출력층까지 0으로 시작하지만, drifting에서 모든 x가 같은 점이면 drift도 전부 같아져
        # 다양성이 생기지 않는 안장점에서 출발하게 된다. 출력층만 작은 값으로 시작한다.
        nn.init.normal_(self.final.proj.weight, std=0.01)

    def sample_noise(self, count: int, device: str) -> tuple[Tensor, Tensor]:
        """ε = (가우시안 노이즈, style 인덱스). 논문 A.2의 연속 + 이산 노이즈."""
        cfg = self.config
        noise = lucid.randn(count, cfg.channels, cfg.image_size, cfg.image_size, device=device)
        style = lucid.randint(
            0, cfg.style_codebook, (count, cfg.num_style_tokens), device=device, dtype=lucid.int64
        )
        return noise, style

    @override
    def forward(
        self,
        noise: Tensor,
        style: Tensor,
        text_pooled: Tensor,
        text_tokens: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        cfg = self.config
        batch = int(noise.shape[0])
        # α ∈ [1, 4]를 사인 임베딩이 구분하기 좋은 범위로 늘린다.
        alpha_embed = timestep_embedding((alpha - 1.0) * 300.0, cfg.hidden, dtype=noise.dtype)
        cond = self.pooled_proj(text_pooled) + self.alpha_proj(alpha_embed)
        cond = cond + self.style(style).sum(1)

        image = self.patch_embed(noise).reshape(batch, cfg.hidden, -1).swapaxes(1, 2)
        image = image + self.pos_embed
        text = self.token_proj(text_tokens[:, : cfg.num_text_tokens]) + self.text_pos
        registers = self.registers.unsqueeze(0) + self.register_cond(cond).unsqueeze(1)
        tokens = lucid.cat([text, registers, image], dim=1)
        for block in self.blocks:
            tokens = block(tokens, cond)
        output = self.final(tokens[:, -self.grid * self.grid :], cond)
        return unpatchify(output, self.grid, cfg.patch_size, cfg.channels)
