"""CLIP 래퍼가 Lucid의 공식 경로와 같은 값을 내는지 확인한다.

사전학습 가중치(577MB)가 캐시에 있어야 하는 테스트는 없으면 건너뛴다.
"""

from pathlib import Path

import lucid
import numpy as np
import pytest
from lucid.models.multimodal.clip import CLIPConfig, CLIPModel

from clipdrift.clip_encoder import DEFAULT_CACHE, FrozenCLIP, load_tokenizer

TINY = CLIPConfig(
    image_size=32, patch_size=16, vision_layers=3, vision_width=32, vision_heads=2,
    context_length=8, vocab_size=64, text_width=32, text_heads=2, text_layers=2, embed_dim=16,
)


class _Tokenizer:
    _sos_id, _eos_id = 62, 63

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 60 for c in text]


@pytest.fixture
def tiny() -> FrozenCLIP:
    lucid.manual_seed(0)
    return FrozenCLIP(CLIPModel(TINY), _Tokenizer(), device="cpu", layers=(1, 3))


def test_image_tower_matches_lucid_encode_image(tiny):
    images = lucid.randn(3, 3, 32, 32)
    embed, maps = tiny.image_tower(images)
    expected = tiny.model.encode_image(tiny.preprocess(images))
    assert (embed - expected).abs().max().item() < 1e-5
    assert maps[3].shape == (3, 32, 2, 2)
    assert set(maps) == {1, 3}


def test_text_forward_matches_lucid_encode_text(tiny):
    ids = tiny.tokenize(["ab", "abc"])
    pooled, tokens = tiny._text_forward(ids)
    expected = tiny.model.encode_text(ids)
    assert (pooled - expected).abs().max().item() < 1e-5
    assert tokens.shape == (2, 8, 32)


def test_long_caption_is_truncated_not_rejected(tiny):
    ids = tiny.tokenize(["x" * 100]).numpy()[0]
    assert ids[0] == 62 and ids[-1] == 63 and len(ids) == 8


def test_gradient_reaches_input_but_not_clip(tiny):
    images = lucid.randn(2, 3, 32, 32, requires_grad=True)
    features = tiny.features(images)
    assert "clip/embed" in features and "clip/l3/loc" in features
    sum(value.sum() for value in features.values()).backward()
    assert float(images.grad.abs().sum().item()) > 0
    assert all(p.grad is None for p in tiny.model.parameters())


def _weights_cached() -> bool:
    return Path.home().joinpath(".cache/lucid/weights/clip_vit_base_32").exists()


@pytest.mark.skipif(not _weights_cached(), reason="CLIP weights not downloaded")
def test_pretrained_clip_ranks_captions_sensibly():
    """실제 가중치: 렌더링한 도형을 올바른 색상 캡션으로 분류한다."""
    from clipdrift.data import render_shape

    tokenizer = load_tokenizer(DEFAULT_CACHE)
    assert tokenizer.tokenize("a photo of a cat", context_length=8)[:7] == [
        49406, 320, 1125, 539, 320, 2368, 49407,
    ]
    clip = FrozenCLIP(tokenizer=tokenizer, device="metal")
    rng = np.random.default_rng(0)
    colors = ["red", "green", "blue", "yellow"]
    images = np.stack([render_shape(c, "square", "white", rng=rng) for c in colors])
    embeds = clip.encode_images(lucid.tensor(images, device="metal"))
    text, _ = clip.encode_captions([f"a {c} square" for c in colors])
    assert (embeds @ text.T).argmax(1).tolist() == [0, 1, 2, 3]
