# E1.3 — 사전 등록: C2 — AffD가 mixture를 복원하고 K1–K3를 모두 이기는가

- 작성 2026-11-08 · 실험 엔지니어 Claude · 결정 위임 Chan Lee (L1 go, D15–D19)
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고 이후 수정하지 않는다(CLAUDE.md §2.9).

## plan 원문 (docs/plan.md §5.2, 해당 행 그대로)

| ID | 주장 | $\Delta$ | 예측 (사전 등록) | 반증 기준 |
| --- | --- | --- | --- | --- |
| E1.3 | C2 | AffD ↔ single, K1, K2, K3 (각각) | AffD의 W2가 single의 0.5배 이하이고, K1–K3보다도 낮음 (쌍별 CI가 0을 포함하지 않음) | AffD가 어느 통제군과 동등. 그 통제군이 남긴 요인을 원인으로 보고한다 |

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
| `AffD` | B0 | 판정 (reference) |
| `single` | h_p = 0 | 판정 |
| `K1` | k_x ≡ 1 | 판정 |
| `K2` | 캡션 이웃을 전역 무작위 순열로(같은 N_eff) | 판정 |
| `K3` | single + 빌려온 positive 분산에 맞춘 Gaussian 잡음 | 판정 (짝: single) |
| `drifting` | release | baseline |

## 판정 규칙 (`experiment.yaml`의 `judge:`, 이 규칙으로만 판정)

- 지표: held-out W2.
- AffD / single seed 평균 비 ≤ 0.5 → PASS, 아니면 **FAIL**.
- single, K1, K2, K3 각각: (통제군 − AffD)의 paired 차이 CI가 0보다 크면 PASS, 작으면 **FAIL**, 0을 포함하면 INCONCLUSIVE(동등을 주장하려면 동등 범위가 필요하므로 FAIL로 두지 않는다).
- 실험 판정은 가장 나쁜 검사.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release 손실 그대로: Laplace kernel τ ∈ {0.02, 0.05, 0.2}, 행×열 softmax, 배치별 거리 스케일,
τ별 drift normalization)을 같은 생성기·학습 레시피·seed로 돌린다. positive는 원 논문처럼 조건 자신의 것 하나뿐이다.

## 알려진 위험

- K3의 잡음 분산은 bank 조건 512개에서 AffD가 빌려오는 positive의 차원 평균 분산으로 정한다(실행 전 고정).
- K2는 하나의 전역 순열을 쓴다. 각 행의 가중치 multiset이 그대로라 N_eff가 정확히 같다.
