"""텍스트 키 전처리와 positive/prompt 그룹 샘플링.

한 스텝은 G개 그룹으로 이루어진다. 각 그룹이 논문 Alg.1의 "클래스 하나"에 해당한다.
그룹 안에서 positive와 생성 프롬프트를 **같은 텍스트 이웃에서** 뽑는다. 그래야 생성 쪽과 실제 쪽의
t 주변분포가 맞아 p=q일 때 V=0이 성립한다(아이디어 문서 §3.1).
"""

import warnings
from dataclasses import dataclass
from typing import Protocol

import lucid
import numpy as np
from lucid import Tensor
from sklearn.cluster import KMeans

from clipdrift.drift import DriftKernel, text_log_kernel

NULL_PROMPT = -1  # 빈 텍스트(∅)를 뜻하는 프롬프트 인덱스.


@dataclass(frozen=True)
class TextKernel:
    """텍스트 임베딩의 거리 스케일 고정 (아이디어 문서 §4 이방성).

    CLIP 텍스트 임베딩은 좁은 원뿔에 몰려 거리가 작게 잡힌다. 평균 쌍거리를 1로 맞춰서 kernel이 평평해지는
    것을 막고, τ_t를 "평균 거리 대비 비율"로 해석할 수 있게 한다.

    whitening은 기본으로 끈다. 고유 캡션 수가 임베딩 차원보다 적으면 전체 whitening은 점들을 모든 거리가
    같은 정단체로 만들어 의미 구조를 지운다. shapes 데이터(캡션 100개, 512차원)에서 같은 조합/다른 조합
    거리비가 0.605에서 0.895로 평평해지고, kNN 순도도 0.72에서 0.51로 떨어졌다.
    """

    mean: np.ndarray
    projection: np.ndarray
    scale: float

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        *,
        shrinkage: float | None = None,
        rank: int | None = None,
        scale_samples: int = 2000,
        seed: int = 0,
    ) -> "TextKernel":
        """
        Args:
            shrinkage: None이면 whitening을 하지 않는다. 값을 주면 고유값에 평균 고유값의 그 비율을
                더한 뒤 whitening한다.
            rank: 주어지면 분산이 큰 상위 축만 남긴다(PCA).
        """
        data = np.asarray(embeddings, dtype=np.float64)
        if shrinkage is not None and len(data) <= data.shape[1] and rank is None:
            warnings.warn(
                f"whitening {len(data)} samples in {data.shape[1]} dimensions flattens all "
                "pairwise distances toward a regular simplex; pass rank= or disable whitening",
                stacklevel=2,
            )
        mean = data.mean(0)
        centered = data - mean
        cov = centered.T @ centered / max(len(data) - 1, 1)
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        keep = eigval > eigval.max() * 1e-8
        eigval, eigvec = eigval[keep], eigvec[:, keep]
        if rank is not None:
            eigval, eigvec = eigval[:rank], eigvec[:, :rank]
        projection = eigvec
        if shrinkage is not None:
            projection = eigvec / np.sqrt(eigval + shrinkage * eigval.mean())

        rng = np.random.default_rng(seed)
        subset = centered[rng.permutation(len(data))[:scale_samples]] @ projection
        squared = ((subset[:, None] - subset[None]) ** 2).sum(-1)
        off_diagonal = np.sqrt(squared[~np.eye(len(subset), dtype=bool)])
        scale = float(off_diagonal.mean()) if off_diagonal.size else 1.0
        return cls(mean.astype(np.float32), projection.astype(np.float32), max(scale, 1e-8))

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return (np.asarray(embeddings, dtype=np.float32) - self.mean) @ self.projection / self.scale


@dataclass(frozen=True)
class Groups:
    """데이터 인덱스 묶음. 모두 그룹이 첫 축이다."""

    positives: np.ndarray  # [G, P]
    prompts: np.ndarray  # [G, N], NULL_PROMPT면 빈 텍스트
    unconditional: np.ndarray | None  # [G, U]
    null: np.ndarray  # [G], True면 비조건 그룹


class GroupSource(Protocol):
    uses_text_kernel: bool

    def group(self, num_pos: int, num_neg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(positive 인덱스 [P], prompt 인덱스 [N])를 반환한다."""
        ...


def _pick(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """가능하면 비복원, 풀이 작으면 복원 추출(논문 A.8은 비복원)."""
    return rng.choice(pool, size=count, replace=count > len(pool))


class BlindSource:
    """loss가 텍스트를 보지 않는 기준선. 생성기만 프롬프트를 조건으로 받는다."""

    uses_text_kernel = False

    def __init__(self, size: int) -> None:
        self.size = size

    def group(self, num_pos, num_neg, rng):
        return rng.integers(self.size, size=num_pos), rng.integers(self.size, size=num_neg)


class PairSource:
    """프롬프트 하나에 그 짝 이미지 하나만 쓰는 기준선(N_pos=1 문제, 아이디어 문서 §1)."""

    uses_text_kernel = False

    def __init__(self, size: int) -> None:
        self.size = size

    def group(self, num_pos, num_neg, rng):
        index = int(rng.integers(self.size))
        return np.full(num_pos, index), np.full(num_neg, index)


class PseudoClassSource:
    """로드맵 1단계: 캡션 임베딩을 k-means로 묶어 의사 클래스로 쓴다."""

    uses_text_kernel = False

    def __init__(self, keys: np.ndarray, num_classes: int, seed: int = 0) -> None:
        self.num_classes = num_classes
        self.labels = KMeans(num_classes, n_init=4, random_state=seed).fit_predict(keys)
        self.members = [np.flatnonzero(self.labels == c) for c in range(num_classes)]
        sizes = np.array([len(m) for m in self.members], dtype=np.float64)
        self.weights = sizes / sizes.sum()  # 크기 비례로 뽑아야 t 주변분포가 데이터와 맞는다.

    def group(self, num_pos, num_neg, rng):
        members = self.members[rng.choice(self.num_classes, p=self.weights)]
        return _pick(members, num_pos, rng), _pick(members, num_neg, rng)


class TextKeyedQueue:
    """텍스트 키로 조회하는 전역 FIFO 큐(MoCo식 + kNN, 아이디어 문서 §4).

    이미지 자체가 아니라 데이터 인덱스를 저장한다. 논문 A.8도 데이터 로더가 더 원칙적이라고 밝히며,
    여기서는 임의 접근이 가능한 Dataset을 전제한다.
    """

    def __init__(self, keys: np.ndarray, capacity: int) -> None:
        self.keys = keys
        self.capacity = min(capacity, len(keys))
        self.items = np.zeros(0, dtype=np.int64)

    def push(self, indices: np.ndarray) -> None:
        self.items = np.concatenate([self.items, indices])[-self.capacity :]

    def neighbors(self, anchor: np.ndarray, count: int) -> np.ndarray:
        squared = ((self.keys[self.items] - anchor) ** 2).sum(-1)
        count = min(count, len(self.items))
        return self.items[np.argpartition(squared, count - 1)[:count]]


class KnnSource:
    """로드맵 2단계: 앵커 프롬프트의 텍스트 이웃을 그룹으로 쓰고, joint kernel로 가중한다."""

    uses_text_kernel = True

    def __init__(
        self,
        keys: np.ndarray,
        *,
        neighbors: int = 256,
        queue_size: int = 8192,
        push_per_group: int = 64,
    ) -> None:
        self.queue = TextKeyedQueue(keys, queue_size)
        self.neighbors = neighbors
        self.push_per_group = push_per_group
        self.size = len(keys)

    def group(self, num_pos, num_neg, rng):
        fresh = rng.integers(self.size, size=max(self.push_per_group, 1))
        self.queue.push(fresh)
        anchor = self.queue.keys[rng.choice(self.queue.items)]
        pool = self.queue.neighbors(anchor, self.neighbors)
        return _pick(pool, num_pos, rng), _pick(pool, num_neg, rng)

    def warm_up(self, rng: np.random.Generator) -> None:
        self.queue.push(rng.permutation(self.size)[: self.queue.capacity])


def sample_groups(
    source: GroupSource,
    *,
    size: int,
    num_groups: int,
    num_pos: int,
    num_neg: int,
    num_unc: int = 0,
    p_null: float = 0.0,
    rng: np.random.Generator,
) -> Groups:
    """`p_null` 확률로 그룹을 비조건 그룹으로 바꾼다(텍스트 드롭아웃, 아이디어 문서 §3.4).

    비조건 그룹은 빈 프롬프트로 생성하고 positive를 전체 데이터에서 뽑는다. "∅ 클래스 하나"로 보면
    되므로 kernel 쪽에 예외 처리가 필요 없다.
    """
    positives, prompts, null = [], [], []
    for _ in range(num_groups):
        if rng.random() < p_null:
            positives.append(rng.integers(size, size=num_pos))
            prompts.append(np.full(num_neg, NULL_PROMPT))
            null.append(True)
        else:
            pos, prompt = source.group(num_pos, num_neg, rng)
            positives.append(pos)
            prompts.append(prompt)
            null.append(False)
    unconditional = rng.integers(size, size=(num_groups, num_unc)) if num_unc > 0 else None
    return Groups(np.stack(positives), np.stack(prompts), unconditional, np.array(null))


def text_kernel_for_groups(
    keys: np.ndarray,
    groups: Groups,
    tau_t: float,
    *,
    device: str,
) -> DriftKernel:
    """그룹별 joint kernel의 텍스트 항을 `[G, 1, N, *]`로 만든다. 비조건 그룹은 k_txt=1이다."""
    pos_keys = keys[groups.positives]
    prompt_keys = keys[np.maximum(groups.prompts, 0)]
    prompt = lucid.tensor(prompt_keys, device=device)
    positive = lucid.tensor(pos_keys, device=device)
    log_k_pos = text_log_kernel(prompt, positive, tau_t)
    log_k_neg = text_log_kernel(prompt, prompt, tau_t)
    if groups.null.any():
        keep = lucid.tensor((~groups.null).astype(np.float32), device=device).reshape(-1, 1, 1)
        log_k_pos = log_k_pos * keep
        log_k_neg = log_k_neg * keep
    return DriftKernel(log_k_pos=log_k_pos.unsqueeze(1), log_k_neg=log_k_neg.unsqueeze(1))


def merge_cfg(kernel: DriftKernel, log_w_unc: Tensor | None) -> DriftKernel:
    return DriftKernel(kernel.log_k_pos, kernel.log_k_neg, log_w_unc)
