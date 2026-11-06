import numpy as np

from clipdrift.inverse import alignment_scores, classification_scores


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_pmi_removes_generic_caption_bias():
    """범용 캡션은 모든 이미지와 비슷하므로 원점수로는 이기지만, 배경을 빼면 진다."""
    rng = np.random.default_rng(0)
    generic = _unit(np.ones(32))
    specific = _unit(generic + 2.0 * _unit(rng.normal(size=32)))
    # 목표 이미지는 범용 방향이 강하고 특정 방향도 약간 있다.
    target = _unit(generic * 3.0 + specific * 0.5 + rng.normal(scale=0.02, size=(64, 32)))
    # 배경 이미지는 범용 방향은 같지만 특정 방향이 없다.
    background = _unit(generic * 3.0 + rng.normal(scale=0.02, size=(256, 32)))
    text = np.stack([generic, specific])
    assert alignment_scores(target, text).argmax() == 0
    assert alignment_scores(target, text, background).argmax() == 1


def test_classification_prefers_matching_caption():
    rng = np.random.default_rng(1)
    text = _unit(rng.normal(size=(5, 32)))
    target = _unit(text[3] + rng.normal(scale=0.01, size=(16, 32)))
    assert classification_scores(target, text).argmax() == 3
    assert (classification_scores(target, text) <= 1e-6).all()
