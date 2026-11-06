"""Lucid CLIP을 얼려서 텍스트 인코더 + 이미지 특징 φ로 쓰는 래퍼 (아이디어 문서 §3.1, §3.3).

Lucid의 `encode_image`/`encode_text`는 최종 벡터만 돌려주므로, 중간 패치 토큰과 텍스트 토큰 시퀀스가
필요한 여기서는 같은 순서로 타워를 직접 통과시킨다. 최종 벡터가 Lucid 결과와 일치하는지는
tests/test_clip_encoder.py에서 확인한다.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import lucid
import lucid.nn.functional as F
import numpy as np
from lucid import Tensor
from lucid.models.multimodal.clip import CLIPModel, CLIPTokenizerFast, clip_vit_base_32

from clipdrift.features import map_vectors

_TOKENIZER_FILES = {
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
}
_TOKENIZER_URL = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/{name}"
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_CACHE = Path.home() / ".cache" / "clipdrift" / "clip_tokenizer"


def load_tokenizer(cache_dir: Path = DEFAULT_CACHE) -> CLIPTokenizerFast:
    """vocab.json + merges.txt를 받아 sha256을 확인하고 토크나이저를 만든다.

    `CLIPTokenizer.from_file`은 Lucid 3.16.3에서 항상 TypeError가 난다
    (`pre_tokenizer` 인자를 생성자가 받지 않음). 그래서 생성자를 직접 호출한다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in _TOKENIZER_FILES.items():
        path = cache_dir / name
        if not path.exists():
            urllib.request.urlretrieve(_TOKENIZER_URL.format(name=name), path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            path.unlink()
            raise RuntimeError(f"{name}: sha256 mismatch ({actual}); removed, retry the download")
    vocab = json.loads((cache_dir / "vocab.json").read_text(encoding="utf-8"))
    lines = (cache_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    merges = [tuple(line.split()) for line in lines if line and not line.startswith("#version")]
    return CLIPTokenizerFast(vocab, merges)


class FrozenCLIP:
    """동결한 CLIP. `nn.Module`로 등록하지 않아 생성기 state_dict에 섞이지 않는다."""

    def __init__(
        self,
        model: CLIPModel | None = None,
        tokenizer: CLIPTokenizerFast | None = None,
        *,
        device: str = "metal",
        layers: tuple[int, ...] = (3, 6, 9, 12),
    ) -> None:
        self.device = device
        self.layers = layers
        self.model = (model if model is not None else clip_vit_base_32(pretrained=True)).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
        self.config = self.model.config
        self._mean = lucid.tensor(_CLIP_MEAN, device=device).reshape(1, 3, 1, 1)
        self._std = lucid.tensor(_CLIP_STD, device=device).reshape(1, 3, 1, 1)

    # ---- text -----------------------------------------------------------------------------

    def tokenize(self, captions: list[str]) -> Tensor:
        length = self.config.context_length
        rows = []
        for caption in captions:
            ids = self.tokenizer.encode(caption)[: length - 2]
            ids = [self.tokenizer._sos_id, *ids, self.tokenizer._eos_id]
            rows.append(ids + [0] * (length - len(ids)))
        return lucid.tensor(rows, dtype=lucid.int64, device=self.device)

    def _text_forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """(L2 정규화된 pooled `[B, E]`, 토큰 시퀀스 `[B, T, W]`)."""
        tower = self.model.textual
        x = tower.token_embedding(input_ids)
        x = x + tower.positional_embedding.reshape(1, int(input_ids.shape[1]), -1)
        x = tower.transformer(x)
        tokens = tower.ln_final(x)
        eos = input_ids.argmax(dim=-1)
        rows = lucid.arange(int(tokens.shape[0]), dtype=lucid.int64, device=self.device)
        pooled = tokens[rows, eos] @ tower.text_projection
        return F.normalize(pooled, dim=-1), tokens

    @lucid.no_grad()
    def encode_captions(
        self, captions: list[str], *, num_tokens: int = 16, batch_size: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """캡션 목록 -> numpy (pooled `[M, E]`, tokens `[M, num_tokens, W]`)."""
        pooled, tokens = [], []
        for start in range(0, len(captions), batch_size):
            ids = self.tokenize(captions[start : start + batch_size])
            batch_pooled, batch_tokens = self._text_forward(ids)
            batch_tokens = batch_tokens[:, :num_tokens]
            batch_pooled.eval()
            batch_tokens.eval()
            pooled.append(batch_pooled.numpy())
            tokens.append(batch_tokens.numpy())
        return np.concatenate(pooled), np.concatenate(tokens)

    # ---- image ----------------------------------------------------------------------------

    def preprocess(self, images: Tensor) -> Tensor:
        """`[-1, 1]` 이미지 -> CLIP 입력. 생성 이미지에도 쓰므로 미분 가능해야 한다."""
        size = self.config.image_size
        resized = F.interpolate(images, size=(size, size), mode="bicubic", align_corners=False)
        return ((resized + 1.0) * 0.5 - self._mean) / self._std

    def image_tower(self, images: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
        """(L2 정규화된 임베딩 `[B, E]`, 레이어별 패치 맵 `[B, W, g, g]`)."""
        tower = self.model.visual
        pixel_values = self.preprocess(images)
        batch = int(pixel_values.shape[0])
        x = tower.conv1(pixel_values)
        x = x.reshape(batch, int(x.shape[1]), -1).swapaxes(1, 2)
        cls = tower.class_embedding.reshape(1, 1, -1).repeat(batch, 1, 1)
        x = lucid.cat([cls, x], dim=1)
        x = x + tower.positional_embedding.reshape(1, tower.num_patches + 1, -1)
        x = tower.ln_pre(x)

        grid = self.config.image_size // self.config.patch_size
        maps: dict[int, Tensor] = {}
        for index, block in enumerate(tower.transformer.resblocks, start=1):
            x = block(x)
            if index in self.layers:
                maps[index] = x[:, 1:].swapaxes(1, 2).reshape(batch, -1, grid, grid)
        embed = F.normalize(tower.ln_post(x[:, 0]) @ tower.proj, dim=-1)
        return embed, maps

    def features(self, images: Tensor) -> dict[str, Tensor]:
        """φ 특징 집합: 공동 임베딩 + 패치 맵의 다중 스케일 벡터(논문 A.5를 패치 토큰에 적용)."""
        embed, maps = self.image_tower(images)
        out = {"clip/embed": embed.unsqueeze(1)}
        for index, feature_map in maps.items():
            map_vectors(f"clip/l{index}", feature_map, out)
        return out

    @lucid.no_grad()
    def encode_images(self, images: Tensor, *, batch_size: int = 64) -> np.ndarray:
        """역방향 탐색(§6)용 공동 임베딩."""
        chunks = []
        for start in range(0, int(images.shape[0]), batch_size):
            embed, _ = self.image_tower(images[start : start + batch_size].to(self.device))
            embed.eval()
            chunks.append(embed.numpy())
        return np.concatenate(chunks)
