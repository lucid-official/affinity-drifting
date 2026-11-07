# E0.5 — 사전 등록: Prop E — counterfactual negative의 닫힌 형태

- 작성 2026-11-07 · 실험 엔지니어 Claude · 승인 Chan Lee
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋한다. 커밋 이후 수정하지 않는다(CLAUDE.md §2.9). `sweep.py`는
  두 파일이 커밋되지 않았거나 HEAD와 다르면 실행을 거부한다.

## plan 원문 (docs/plan.md §4, 해당 행 그대로)

| ID | 주장 | B0 대비 Δ | 예측 | 반증 기준 | 지표 |
| --- | --- | --- | --- | --- | --- |
| E0.5 | Prop E | negative = counterfactual 쌍, λ ∈ {0, .25, .5, .75, .9, 1.1} | 아래 닫힌 형태와 일치. λ ≥ 1이면 발산 | 평균·분산 상대 오차 > 5% (λ < 1) | 평형 평균·분산, 발산 여부 |

E0.5는 p_c = N(μ₁, s²), p_c′ = N(μ₂, s²)로 두면 예측이 닫힌 형태로 나온다. σ 평활화를 포함해 s̃² = s² + σ²로 쓴다.

```latex
\log\hat q_c=\log\hat p_c-\lambda\log\hat q_{c'}
\;\Rightarrow\;
\hat q_c=\mathcal N\!\Big(\tfrac{\mu_1-\lambda\mu_2}{1-\lambda},\;(1+\lambda)\,\tilde s^{2}\Big),\qquad \lambda<1
```

## 설정

- L0 공통 설정(plan §4 "설정" 문단): 조건 c_k (k = 1..K, K = 32)를 한 직선 위에 두고, 조건마다 data point y_k ∈ R²를
  하나씩 둔다 (single-positive 체제). particle은 x ← x + η V(x|c)로 갱신하고, ‖V‖ < 1e-6이 되거나 스텝 상한에 닿으면 멈춘다.
- 운영 정의 (plan에 값이 없는 부분, 2026-11-07 승인, docs/deviations.md D4–D6, D10):
  - ψ_k = (k−1)/31, y_k ~ U([−1, 1]²) 최소 간격 0.15, `data_seed = 0`으로 모든 seed에서 고정. seed는 particle 초기값만 바꾼다.
  - particle M = 256, 초기값 U([−1.5, 1.5]²). η = 0.5, 스텝 상한 50,000.
  - 정지: 정규화 전 V의 RMS‖V‖ < 1e-6이 1,000스텝 연속 유지되면 "stable"로 멈춘다. 상한에 닿으면 그 상태로 판정한다.
  - h_p(c_k): N_eff(c_k) ≥ N* = 16을 만족하는 최소값(bisection). h_q = ratio × h_p.
  - σ = 0.5 × y bank 쌍거리의 median. 실험 시작 때 `sigma.json`에 한 번 고정(D5).
  - L0의 AffD repulsion KDE는 자기 자신을 포함한다(D6). drift normalization 끔(D4). G2 비는 로깅만(D7).
  - 배정: particle을 가장 가까운 y_j에 배정. A_kj = 조건 k의 particle 중 y_j에 배정된 비율. w̃_j(c_k) = 정규화한 캡션 가중치.
  - 통계: seed {0,1,2,3,4}, 95% bootstrap CI(10,000 resample), arm 간 비교는 같은 seed의 paired.

## arm (`experiment.yaml`)

| arm | 설정 | 역할 |
| --- | --- | --- |
| `lam_0` | λ = 0 | 판정 (reference) |
| `lam_0.25` … `lam_0.9` | λ ∈ {0.25, 0.5, 0.75, 0.9} | 판정 (수렴) |
| `lam_1.1` | λ = 1.1 | 판정 (발산) |
| `drifting` | `field: release` | baseline |

모든 arm 공통(`common`): Gaussian pair 문제, M = 2048, negative = counterfactual.

## 판정 규칙 (`experiment.yaml`의 `judge:`, 이 규칙으로만 판정)

- 문제: 조건 c, c′. p_c = N(μ₁, s²I), p_c′ = N(μ₂, s²I), μ₁ = (1, 0), μ₂ = (−1, 0), s = 0.5.
  positive는 조건마다 2,048개, 표본 평균과 표본 공분산이 정확히 μ와 s²I가 되게 맞춘다(`data_seed = 0`).
  particle M = 2048(D10). 서로가 서로의 counterfactual negative(가중 λ). h_q = 0. σ는 두 조건 positive 전체에 D5 규칙.
- 예측(plan의 닫힌 형태): q̂_c = N((μ₁ − λμ₂)/(1 − λ), (1 + λ) s̃²), s̃² = s² + σ². q̂는 σ로 평활한 밀도이므로
  particle 분산 + σ²를 (1 + λ) s̃²와 비교한다. c′도 대칭으로 계산하고 두 조건 중 큰 오차를 쓴다.
- `rel_err_mean` = ‖m − m*‖/‖m*‖, `rel_err_var` = |v + σ² − (1+λ)s̃²| / ((1+λ)s̃²), v는 차원 평균 분산.
- λ < 1 arm 각각: 5 seed 평균 `rel_err_mean`, `rel_err_var` 모두 ≤ 0.05이면 PASS, 아니면 **FAIL**.
- λ = 1.1: 5 seed 모두 발산(‖mean‖ > 10⁴ 또는 비유한)해야 PASS, 아니면 **FAIL**.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting을 Lucid release 그대로(`DriftLoss`: Laplace kernel, τ ∈ {0.02, 0.05, 0.2}, 행×열 이중 softmax,
배치별 거리 스케일, 자기 mask) 같은 seed·초기값·η로 돌린다. positive는 원 논문처럼 class-conditional이라 조건 k는
자기 y_k만 본다(캡션 빌려오기 없음). L0 공통 설정 D4(정규화 전 필드)만 똑같이 적용한다.
개발 중 장난감 관찰(EXPERIMENTS.md 2026-11-07 setup): 이 필드는 고정된 유한 표본에서 ‖V‖ ≈ 0에 닿지 않는다.
따라서 baseline은 스텝 상한까지 돈 상태로 보고한다.

기대(기술적, 판정하지 않음): 자기 positive만 보고 counterfactual negative가 없으므로 고정점은 q = p_c: 평균 μ₁, 분산 s². 단 ‖V‖ ≈ 0에는 닿지 않는다.

## figure (plan §10, 판독 기준은 각 `figures/<FIG>/<FIG>.md`에 원문 인용)

- FC.1 (모든 arm), F0.4 (모든 arm). F0.4는 λ별 예측과 맞대는 그림이라 blind 버전 없음.

## 알려진 위험

- 스텝 상한 50k 안에 stable에 못 닿을 수 있다. 그때도 최종 상태로 판정하고 보고서의 이상 징후에 적는다.
- λ = 0.9에서 예측 평균은 ‖m*‖ = 19로 멀다. 상한 안에 거기까지 가지 못하면 FAIL로 나올 수 있고, 그 경우 수렴 궤적을 같이 보고한다.
- 유한 M에서 분산에는 O(1/M) 수준의 편향이 남는다(D6 이후 장난감 문제에서 M = 256이 1% 이내).
