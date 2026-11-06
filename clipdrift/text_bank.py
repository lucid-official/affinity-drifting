"""캡션 임베딩 캐시. CLIP을 한 번만 돌리고, φ에 CLIP이 없으면 메모리에서 내릴 수 있게 한다."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lucid
import numpy as np
from lucid import Tensor

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.sampling import NULL_PROMPT

CACHE_DIR = Path("cache")


@dataclass(frozen=True)
class TextBank:
    """고유 프롬프트별 임베딩. 마지막 행은 빈 텍스트(∅)다."""

    prompts: list[str]
    pooled: np.ndarray  # [K + 1, E]
    tokens: np.ndarray  # [K + 1, T, W]

    def tensors(self, indices: np.ndarray, device: str) -> tuple[Tensor, Tensor]:
        """프롬프트 인덱스 -> 텐서. `NULL_PROMPT`(-1)는 마지막 행, 즉 빈 텍스트를 가리킨다."""
        assert NULL_PROMPT == -1
        return (
            lucid.tensor(self.pooled[indices], device=device),
            lucid.tensor(self.tokens[indices], device=device),
        )

    def index(self, prompt: str) -> int:
        return self.prompts.index(prompt)


def build_text_bank(
    prompts: list[str],
    *,
    num_tokens: int,
    clip: FrozenCLIP | None = None,
    device: str = "metal",
    cache_dir: Path = CACHE_DIR,
) -> tuple[TextBank, FrozenCLIP | None]:
    """캐시가 있으면 CLIP을 띄우지 않는다. 필요할 때만 로드해서 함께 돌려준다."""
    digest = hashlib.sha256(json.dumps([prompts, num_tokens]).encode()).hexdigest()[:16]
    path = cache_dir / f"textbank_{digest}.npz"
    if path.exists():
        data = np.load(path)
        return TextBank(prompts, data["pooled"], data["tokens"]), clip
    clip = clip if clip is not None else FrozenCLIP(device=device)
    pooled, tokens = clip.encode_captions([*prompts, ""], num_tokens=num_tokens)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, pooled=pooled, tokens=tokens)
    return TextBank(prompts, pooled, tokens), clip
