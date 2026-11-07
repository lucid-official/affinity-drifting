# E1.5 — 사전 등록: C4 — 학습된 생성기에서도 h_q가 평형을 고르는가

- 작성 2026-11-08 · 실험 엔지니어 Claude · 결정 위임 Chan Lee (L1 go, D15–D19)
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고 이후 수정하지 않는다(CLAUDE.md §2.9).

## plan 원문 (docs/plan.md §5.2, 해당 행 그대로)

| ID | 주장 | $\Delta$ | 예측 (사전 등록) | 반증 기준 |
| --- | --- | --- | --- | --- |
| E1.5 | C4 | $h_q = h_p$ ↔ $h_q = 0$, $\delta = 0$, 긴 학습 | 대칭은 memorization 비율이 높고, $h_q = 0$은 NW mixture와의 TV $\le 0.1$ | 두 조건의 평형이 같음 |

## 설정

- 데이터(plan §5.1, CLAUDE.md §5, D18): c ~ U([0,1]²), 학습 쌍 N = 10,000(조건당 positive 1개), p(x|c)는 3-mode GMM
  (중심 각도 2πc₁ + 2πj/3, 반지름 1 + c₂, mode 표준편차 0.1, 혼합 logit 2(c₂ − ½)(j − 1)). held-out 조건 256개.
- 생성기: MLP f(ε, g(c)), SiLU, 은닉 256 × 3, 조건 경로 64, ε 4차원. 용량 손잡이 δ(B0: 1/16), L(E1.4b).
- 학습(표준 레시피, D17·D19): Adam β = (0.9, 0.999), lr은 Gate G4에서 정한 값(모든 arm 공통), 상수 lr, 20,000 스텝,
  조건 64개 × 생성 8개, 생성기 가중치 EMA 0.999, 평가는 EMA 가중치로. early stopping 없음. ‖V‖²로 아무것도 고르지 않는다.
- 필드: B0 AffD = 캡션 가중치(N_eff ≥ 16 규칙의 h_p, bank 전체) × Gaussian k_x, σ = bank의 16번째 최근접 이웃 거리 median
  (D15, 실험별 `sigma.json`에 고정), h_q = 0(자기 제외), 항별 drift normalization(E‖V‖²/C = 1), feature scale 고정(D3).
- 평가: 학습 조건 256개와 held-out 조건 256개, 조건당 생성 512개와 참 표본 512개. 판정 지표는 plan §5.1(W2는 정확 OT,
  MCI = 1 − tr Cov_q / tr Cov_p, off-manifold = 참 log 밀도가 그 조건 참 표본의 1% 분위수 미만인 비율, mode coverage =
  최근접 mode가 5% 이상 받은 mode 비율, memorization = 학습 positive 최근접 거리 ≤ 학습 positive 최근접 이웃 거리 median의 ½).
  보고용 표준 지표: precision/recall(k = 3), density/coverage(k = 5), RBF MMD(median heuristic).
- 통계: seed {0,…,4}, seed 평균과 95% bootstrap CI(10,000), arm 비교는 paired.
- **판정 split**: 별도 언급이 없으면 held-out 조건(새 캡션으로의 일반화가 T2I에서의 질문). 학습 조건 값은 함께 보고한다.

## arm

| arm | 설정 | 역할 |
| --- | --- | --- |
| `hq0` | h_q = 0 | 판정 (reference) |
| `sym` | h_q = h_p | 판정 |
| `drifting` | release | baseline |

공통: δ = 0, 학습 3배(60,000 스텝, D19), **이웃 묶음 batch**(앵커 8개 × 각 앵커의 캡션 가중치에서 뽑은 이웃 7개). batch 구성은 모든 arm이 같다.

## 판정 규칙 (`experiment.yaml`의 `judge:`, 이 규칙으로만 판정)

- memorization(학습 조건): sym − hq0의 paired 차이 CI가 0보다 크면 PASS, 작으면 **FAIL**, 걸치면 INCONCLUSIVE.
- NW mixture와의 TV(학습 조건 64개): 생성 표본과 NW mixture Σ w̃_i δ(y_i)를 **같은 σ로 평활한 밀도**끼리 비교한 TV. `hq0`의 seed 평균 ≤ 0.1 → PASS, > 0.1 → **FAIL**.
- 실험 판정은 가장 나쁜 검사.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release 손실 그대로: Laplace kernel τ ∈ {0.02, 0.05, 0.2}, 행×열 softmax, 배치별 거리 스케일,
τ별 drift normalization)을 같은 생성기·학습 레시피·seed로 돌린다. positive는 원 논문처럼 조건 자신의 것 하나뿐이다.

## 알려진 위험

- TV를 평활 밀도로 재는 것은 L0의 교훈(점 단위 배정은 σ 이하의 구조라 필드가 정하지 못한다)에 따른 운영 정의다. 필드의 고정점이 q̂ = p̂(Lemma 1)이므로 같은 해상도에서 비교한다. 결과를 보기 전에 정했다.
- h_q > 0의 repulsion은 batch 안의 다른 조건 생성 표본에만 작용한다. 그래서 이웃 묶음 batch가 필요하다.
