import numpy as np

from clipdrift.caption_eval import (
    clip_score,
    diversity,
    frechet_distance,
    layout,
    layout_corr,
    layout_r_precision,
    r_precision,
)
from clipdrift.sampling import PairSource
from clipdrift.train import TrainConfig, make_source


def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def test_clip_score_and_r_precision_bounds():
    rng = np.random.default_rng(0)
    texts = unit(rng.normal(size=(300, 32)))
    assert np.isclose(clip_score(texts, texts), 1.0)
    assert r_precision(texts, texts, np.random.default_rng(0)) == 1.0
    unrelated = unit(rng.normal(size=(300, 32)))
    assert r_precision(unrelated, texts, np.random.default_rng(0)) < 0.05  # 우연 수준 0.01


def test_frechet_distance_matches_mean_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4000, 4))
    assert frechet_distance(a, a) < 1e-6
    shift = np.array([1.0, 0.0, 2.0, 0.0])
    assert np.isclose(frechet_distance(a, a + shift), 5.0, atol=1e-6)


def test_diversity_is_zero_for_collapse_and_one_for_orthogonal():
    same = np.tile(unit(np.ones((1, 1, 8))), (3, 4, 1))
    assert np.isclose(diversity(same), 0.0)
    orthogonal = np.tile(np.eye(4)[None], (3, 1, 1))
    assert np.isclose(diversity(orthogonal), 1.0)


def test_layout_pools_blocks_and_scores_matches():
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    images[0, 0, :8, :8] = 1.0  # 첫 이미지의 빨강 채널 왼쪽 위 블록
    pooled = layout(images)
    assert pooled.shape == (2, 48)
    assert pooled[0, 0] == 1.0 and pooled[0].sum() == 1.0 and pooled[1].sum() == 0.0

    rng = np.random.default_rng(0)
    real = rng.normal(size=(300, 48))
    assert np.isclose(layout_corr(real, real), 1.0)
    assert layout_r_precision(real + 0.01 * rng.normal(size=real.shape), real, np.random.default_rng(0)) == 1.0
    unrelated = rng.normal(size=(300, 48))
    assert abs(layout_corr(unrelated, real)) < 0.1
    assert layout_r_precision(unrelated, real, np.random.default_rng(0)) < 0.05


def test_pair_mode_uses_one_image_per_prompt():
    source = make_source(TrainConfig(mode="pair"), np.zeros((10, 2)), np.random.default_rng(0))
    assert isinstance(source, PairSource)
    positives, prompts = source.group(4, 3, np.random.default_rng(0))
    assert len(set(positives)) == 1 and set(positives) == set(prompts)
