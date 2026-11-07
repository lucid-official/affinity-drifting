"""이미지 파이프라인 부품 테스트. CLIP 사전학습 가중치 없이 돌도록 작은 모델을 쓴다."""

import lucid
import numpy as np
import pytest

from scipy import ndimage

from clipdrift.data import (
    COLORS,
    SHAPES,
    TEMPLATES,
    classify_shapes,
    make_shapes,
    render_shape,
    shape_mask,
)
from clipdrift.drift import DriftKernel, cfg_log_weights, feature_drift_loss, normalized_drift
from clipdrift.features import map_vectors, pixel_features, subsample_locations
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.mae import MAEConfig, MAEFeatures, ResNetMAE, mae_loss, random_mask
from clipdrift.sampling import (
    NULL_PROMPT,
    KnnSource,
    PseudoClassSource,
    TextKernel,
    TextKeyedQueue,
    sample_groups,
    text_kernel_for_groups,
)


@pytest.fixture(autouse=True)
def _seed():
    lucid.manual_seed(0)


def test_text_kernel_whitens_and_scales():
    rng = np.random.default_rng(0)
    # 한 축으로 강하게 치우친(이방성) 임베딩.
    data = rng.normal(size=(400, 8)) * np.array([10, 5, 1, 1, 1, 0.5, 0.5, 0.1])
    kernel = TextKernel.fit(data, shrinkage=0.0)
    keys = kernel.transform(data)
    cov = np.cov(keys.T)
    # 공분산이 등방적이고 평균 쌍거리는 1이다.
    assert np.diag(cov).max() / np.diag(cov).min() < 1.01
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 1e-3 * np.diag(cov).max()
    dist = np.linalg.norm(keys[:, None] - keys[None], axis=-1)
    assert abs(dist[~np.eye(400, dtype=bool)].mean() - 1.0) < 1e-3


def test_default_kernel_keeps_structure_when_samples_are_few():
    """N < d에서 전체 whitening은 정단체를 만든다. 기본값은 구조를 유지해야 한다."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 64)) * 3.0
    labels = np.repeat(np.arange(5), 8)  # 샘플 40개 < 64차원
    data = centers[labels] + rng.normal(size=(40, 64)) * 0.3
    same = labels[:, None] == labels[None]
    off = ~np.eye(40, dtype=bool)

    def ratio(keys: np.ndarray) -> float:
        dist = np.linalg.norm(keys[:, None] - keys[None], axis=-1)
        return dist[same & off].mean() / dist[~same].mean()

    default = TextKernel.fit(data).transform(data)
    with pytest.warns(UserWarning, match="simplex"):
        whitened = TextKernel.fit(data, shrinkage=0.0).transform(data)
    assert ratio(default) < 0.3
    assert ratio(whitened) > 0.95
    # 기본값도 평균 쌍거리는 1이다.
    dist = np.linalg.norm(default[:, None] - default[None], axis=-1)
    assert abs(dist[off].mean() - 1.0) < 1e-3


def test_text_kernel_rank_keeps_top_axes():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 6)) * np.array([9, 5, 1, 1, 0.1, 0.1])
    kernel = TextKernel.fit(data, rank=2)
    assert kernel.transform(data).shape == (200, 2)
    # 상위 축은 큰 분산 축이다.
    assert np.abs(kernel.projection[:, 0]).argmax() == 0


def test_queue_is_fifo_and_knn_finds_neighbors():
    keys = np.arange(20, dtype=np.float32)[:, None]
    queue = TextKeyedQueue(keys, capacity=8)
    queue.push(np.arange(5))
    queue.push(np.arange(5, 15))
    assert list(queue.items) == list(range(7, 15))
    assert set(queue.neighbors(np.array([10.0]), 3)) == {9, 10, 11}


def test_pseudo_class_groups_are_pure():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 6)) * 10
    labels = rng.integers(4, size=500)
    keys = centers[labels] + rng.normal(size=(500, 6)) * 0.1
    source = PseudoClassSource(keys, 4)
    groups = sample_groups(source, size=500, num_groups=8, num_pos=6, num_neg=6, rng=rng)
    for positive, prompt in zip(groups.positives, groups.prompts):
        assert len(set(labels[positive]) | set(labels[prompt])) == 1


def test_knn_group_shares_text_neighborhood():
    rng = np.random.default_rng(0)
    keys = np.linspace(0, 1, 1000, dtype=np.float32)[:, None]
    source = KnnSource(keys, neighbors=50, queue_size=1000)
    source.warm_up(rng)
    groups = sample_groups(source, size=1000, num_groups=8, num_pos=10, num_neg=10, rng=rng)
    for positive, prompt in zip(groups.positives, groups.prompts):
        span = keys[np.concatenate([positive, prompt])]
        assert span.max() - span.min() < 0.15


def test_null_groups_are_unconditional():
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(100, 4)).astype(np.float32)
    source = KnnSource(keys, neighbors=20, queue_size=100)
    source.warm_up(rng)
    groups = sample_groups(
        source, size=100, num_groups=6, num_pos=4, num_neg=4, p_null=1.0, rng=rng
    )
    assert groups.null.all() and (groups.prompts == NULL_PROMPT).all()
    kernel = text_kernel_for_groups(keys, groups, 0.2, device="cpu")
    assert kernel.log_k_pos.abs().max().item() == 0
    assert kernel.log_k_neg.abs().max().item() == 0


def test_generator_shapes_gradients_and_conditioning():
    cfg = GeneratorConfig(
        image_size=16, patch_size=4, hidden=32, depth=2, heads=2, text_pooled_dim=8,
        text_token_dim=6, num_text_tokens=4, num_registers=2,
    )
    model = TextConditionedGenerator(cfg)
    noise, style = model.sample_noise(3, "cpu")
    pooled, tokens = lucid.randn(3, 8), lucid.randn(3, 4, 6)
    alpha = lucid.full((3,), 2.0)
    output = model(noise, style, pooled, tokens, alpha)
    assert output.shape == (3, 3, 16, 16)
    assert output.std().item() > 0.01  # 초기 출력이 한 점으로 무너져 있지 않다.
    output.mean().backward()
    assert model.final.proj.weight.grad is not None
    # 출력이 노이즈에 따라 달라진다.
    other = model(lucid.randn(*noise.shape), style, pooled, tokens, alpha)
    assert (other - output).abs().max().item() > 1e-3


def test_map_vectors_follows_a5():
    feature_map = lucid.randn(2, 5, 8, 8)
    out: dict[str, lucid.Tensor] = {}
    map_vectors("m", feature_map, out)
    assert out["m/loc"].shape == (2, 64, 5)
    assert out["m/mean"].shape == (2, 1, 5)
    assert out["m/std"].shape == (2, 1, 5)
    assert out["m/mean2"].shape == (2, 16, 5)
    assert out["m/std4"].shape == (2, 4, 5)
    assert abs(out["m/mean"][0, 0, 0].item() - feature_map[0, 0].mean().item()) < 1e-5


def test_subsampling_uses_same_locations():
    a = {"f": lucid.arange(40, dtype=lucid.float32).reshape(1, 40, 1).repeat(2, 1, 1)}
    b = {"f": a["f"] * 2.0}
    sub_a, sub_b = subsample_locations((a, b), 8)
    assert sub_a["f"].shape == (2, 8, 1)
    assert ((sub_b["f"] - sub_a["f"] * 2.0).abs().max().item()) == 0


def test_mae_mask_loss_and_features():
    config = MAEConfig(base_width=8, blocks=(1, 2, 2, 1), groups=4)
    model = ResNetMAE(config)
    images = lucid.randn(2, 3, 32, 32)
    mask = random_mask(images)
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}
    assert 0.2 < float(mask.mean().item()) < 0.8
    # 2×2 패치 단위로 가려진다.
    array = mask.numpy()[0, 0]
    assert (array[::2, ::2] == array[1::2, 1::2]).all()
    assert model(images).shape == images.shape
    loss = mae_loss(model, images)
    loss.backward()
    extractor = MAEFeatures(model)
    x = lucid.randn(2, 3, 32, 32, requires_grad=True)
    features = extractor(x)
    assert {"mae/0/loc", "mae/0/std4"} <= set(features)
    features["mae/0/loc"].sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum().item()) > 0


def test_full_loss_with_text_kernel_and_unconditional_negatives():
    """텍스트 항, CFG, 다중 위치가 모두 켜진 실제 학습 형태."""
    g, l, n, p, u, c = 2, 3, 4, 5, 2, 6
    x = lucid.randn(g, l, n, c, requires_grad=True)
    y, un = lucid.randn(g, l, p, c), lucid.randn(g, l, u, c)
    keys_n, keys_p = lucid.randn(g, n, 4), lucid.randn(g, p, 4)
    from clipdrift.drift import text_log_kernel

    log_k_pos = text_log_kernel(keys_n, keys_p, 0.3).unsqueeze(1)
    log_k_neg = text_log_kernel(keys_n, keys_n, 0.3).unsqueeze(1)
    alpha = lucid.full((g, n), 2.5)
    kernel = DriftKernel(log_k_pos, log_k_neg, cfg_log_weights(alpha, log_k_neg, n, u))
    loss, stats = feature_drift_loss(x, y, un, kernel)
    loss.backward()
    assert x.grad.shape == x.shape
    assert np.isfinite(x.grad.numpy()).all()
    assert stats.raw_drift.item() > 0
    # 음성 혼합까지 포함해도 양/음이 같으면 drift가 0이다.
    same = lucid.cat([x.detach(), un], dim=-2)
    field, _, _ = normalized_drift(
        x.detach(),
        same,
        un,
        DriftKernel(
            lucid.cat([log_k_neg, kernel.log_w_unc], dim=-1) + _self(n, u),
            log_k_neg,
            kernel.log_w_unc,
        ),
        (0.5,),
        normalize_drift=False,
    )
    assert field.abs().max().item() < 1e-4


def _self(n: int, u: int) -> lucid.Tensor:
    from clipdrift.drift import _self_mask

    return _self_mask(n, n + u, device="cpu")


def test_shapes_dataset_and_rule_classifier():
    dataset = make_shapes(400, seed=0)
    assert dataset.images.shape == (400, 3, 32, 32)
    assert dataset.images.min() >= -1 and dataset.images.max() <= 1
    judged = classify_shapes(dataset.images)
    for key in ("color", "shape", "background"):
        hits = [judged[key][i] == dataset.attributes[i][key] for i in range(400)]
        assert np.mean(hits) > 0.97, key
    pairs = {(a["color"], a["shape"]) for a in dataset.attributes}
    assert ("yellow", "triangle") not in pairs and ("blue", "circle") not in pairs
    assert len(dataset.captions) > len(TEMPLATES)


def test_classifier_tolerates_rough_edges():
    """모서리가 깎이고 경계가 지저분해도 도형을 맞혀야 한다. 생성 이미지가 이런 모습이다."""
    size = 32
    grid = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(grid, grid)
    rng = np.random.default_rng(0)
    rough = []
    labels = []
    for index in range(60):
        shape = SHAPES[index % 3]
        radius = rng.uniform(0.2, 0.28)
        cx, cy = rng.uniform(0.35, 0.65, size=2)
        mask = shape_mask(shape, cx, cy, radius, xx, yy)
        # 경계 픽셀 일부를 뒤집어 거칠게 만든다.
        edge = ndimage.binary_dilation(mask) & ~ndimage.binary_erosion(mask)
        mask = np.where(edge & (rng.random(mask.shape) < 0.25), ~mask, mask)
        image = np.where(mask[None], np.array(COLORS["red"])[:, None, None], 0.05)
        rough.append(image * 2.0 - 1.0)
        labels.append(shape)
    judged = classify_shapes(np.stack(rough).astype(np.float32))
    assert np.mean([a == b for a, b in zip(judged["shape"], labels)]) > 0.9


def test_classifier_judges_outline_not_area():
    """면적이 같은 도형끼리도 윤곽으로 구분해야 한다."""
    size = 32
    grid = (np.arange(size) + 0.5) / size
    xx, yy = np.meshgrid(grid, grid)
    target_area = 0.16
    images, labels = [], []
    for shape, factor in (("circle", np.pi), ("square", 4.0), ("triangle", 2.0)):
        mask = shape_mask(shape, 0.5, 0.5, float(np.sqrt(target_area / factor)), xx, yy)
        images.append(np.where(mask[None], np.array(COLORS["red"])[:, None, None], 0.05) * 2.0 - 1.0)
        labels.append(shape)
    judged = classify_shapes(np.stack(images).astype(np.float32))
    assert judged["shape"] == labels
    assert np.ptp(judged["area"]) < 0.02


def test_classifier_rejects_blank_image():
    rng = np.random.default_rng(0)
    image = render_shape("red", "circle", "black", rng=rng)
    blank = np.full_like(image, image[0, 0, 0])
    judged = classify_shapes(np.stack([image, blank]))
    assert judged["color"] == ["red", "none"]


def test_pixel_features_shapes():
    features = pixel_features(lucid.randn(2, 3, 8, 8))
    assert set(features) == {"pixel/raw"}
    assert features["pixel/raw"].shape == (2, 1, 192)
