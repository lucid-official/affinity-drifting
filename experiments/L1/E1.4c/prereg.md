# E1.4c — 사전 등록: C3 재검정 — Fourier feature로 생성기의 c-해상도를 높이면

- 작성 2026-11-09 · 실험 엔지니어 Claude · 승인 Chan Lee (D25: option 1 + “수정된 예측으로”, 손잡이·수준·판정 규칙은 엔지니어 권고안 위임)
- 이 파일과 `experiment.yaml`은 첫 run(smoke 포함) 전에 커밋하고 이후 수정하지 않는다(CLAUDE.md §2.9).

## plan 원문 (docs/plan.md, 해당 행 그대로)

| ID | 주장 | 근거 | Level | 실패 시 |
| --- | --- | --- | --- | --- |
| C3 | C1–C2의 격차는 생성기의 c-해상도가 낮을수록 크고, 무한 용량에서 사라진다 | Prop A + B | L0, L1 | 메커니즘 설명을 수정한다 |

| ID | 주장 | $\Delta$ | 예측 (사전 등록) | 반증 기준 |
| --- | --- | --- | --- | --- |
| E1.4 | C3 | $\delta \times h_p$ ($h_p \in \{0, 0.5, 1, 2, 4\}\cdot\delta$) factorial. E1.4b는 $\delta$ 대신 Lipschitz 상수 $L$로 같은 factorial | single과 AffD의 W2 격차가 $\delta$에 따라(E1.4b에서는 $L$이 작아질수록) 단조 증가. $\delta = 0$에서는 학습 $c$ 기준 격차가 0에 수렴 | 격차가 $\delta$와 무관 |

## 수정된 예측 (D25, 사용자 승인 2026-11-09) — 이 실험은 이것을 판정한다

**왜 수정했나.** 판정 지표는 참 $p(x\mid c)$까지의 W2다. 이 지표로 학습 조건 64개에서 계산하면(탐색적, 판정에 쓰지 않음)
조건부 평균으로 붕괴한 생성기는 $\sqrt{\operatorname{tr}\Sigma_p} \approx 1.41$(실제 single 값과 일치), 자기 positive를 외운
생성기는 $\sqrt{2\operatorname{tr}\Sigma_p} \approx 2.01$이다. 무한 용량에서 single은 암기로 가므로(Prop A) plan C3의
“격차가 무한 용량에서 사라진다”는 이 지표에서 성립할 수 없다. E1.4의 격차도 해상도가 높을수록 컸다(δ = 1/4에서 0.68, δ = 0에서 0.96).
E1.4는 δ = 0에서도 single이 평균으로 붕괴해 해상도 손잡이가 실효 해상도를 바꾸지 못했고, E1.4b는 Lipschitz 제약이 걸리지 않았다.

**예측.** 생성기가 보는 c를 random Fourier feature로 바꿔 c-해상도를 높이면(s가 클수록 높음), single은 조건부 평균에서
자기 positive 쪽으로 이동하고(조작 검사), 학습 조건 W2가 커진다(암기 쪽, 상한 ≈ 2.0). AffD는 그렇지 않으므로
single − AffD의 학습 조건 W2 격차는 해상도와 함께 커진다.

**반증 기준.** single이 자기 positive에 다가가는데도(조작 검사 통과) 학습 조건 W2가 오르지 않거나, 격차가 해상도와 함께 줄어든다.
조작 검사가 실패하면(해상도를 높여도 single이 자기 positive에 다가가지 않으면) 손잡이가 걸리지 않은 것이므로 판정은 INCONCLUSIVE다.

## 설정

- 데이터·학습·평가: E1.4와 같다(B0 L1, D17–D24). c ~ U([0,1]²), 학습 쌍 N = 10,000(조건당 positive 1개), 3-mode GMM,
  MLP 생성기(SiLU, 은닉 256 × 3, 조건 경로 64, ε 4차원), Adam lr 1e-3(G4c), 20,000 스텝, 조건 8개 × 생성 64개,
  EMA 0.999로 평가, Gaussian kernel 3 스케일(D20), 전체 필드 drift normalization(D24), h_q = 0.
- δ = 0(양자화 없음).
- **손잡이(D25):** 조건 경로 g의 입력을 c 대신 $\gamma(c) = [\cos 2\pi Bc, \sin 2\pi Bc]$, $B \in \mathbb R^{64\times 2}$,
  $B_{kl} \sim \mathcal N(0, s^2)$(Tancik et al., 2020)로 바꾼다. B는 run seed로 한 번 뽑는 상수(학습하지 않음). 같은 seed의
  모든 arm은 같은 s에서 같은 B를 쓴다. s ∈ {1, 4, 16, 64}와 raw c(Fourier feature 없음, E1.4의 d0와 같은 생성기).
  학습 조건 간격은 약 0.01이고 s = 64의 전형적 주파수는 단위당 수십–백 주기라, 가장 높은 수준은 조건 간격 정도를 분해한다.
- 평가 split: **판정은 학습 조건**(C3는 학습 c 기준 주장). held-out 값은 함께 보고한다.
- 통계: seed {0,…,4}, seed 평균과 95% bootstrap CI(10,000), arm 비교는 paired.

## arm

s ∈ {raw, 1, 4, 16, 64} × h ∈ {0(single), 1(AffD, h*)} = 10 arm + baseline `drifting`. reference `raw_h1`.
factorial은 판정이 쓰는 두 행(h0, h1)으로 줄였다(D25).

## 판정 규칙 (`experiment.yaml`의 `judge:`, 이 규칙으로만 판정; 가장 나쁜 검사가 전체 판정)

1. **조작 검사:** `own_positive_dist_train`(학습 조건별 생성 표본 평균과 자기 positive 사이 거리의 평균)에서
   raw_h0 − s64_h0의 paired 차이. CI > 0이면 PASS, CI가 0을 포함하거나 0 미만이면 INCONCLUSIVE(`fail_as`).
2. **single의 W2 상승:** 학습 조건 W2에서 s64_h0 − raw_h0의 paired 차이. CI > 0이면 PASS, CI < 0이면 FAIL, 그 밖에는 INCONCLUSIVE.
3. **격차 증가:** 격차 g(s) = W2_train(single) − W2_train(AffD)가 raw → 1 → 4 → 16 → 64 순으로 증가.
   인접 쌍마다 격차 증가의 CI가 통째로 음수면 FAIL, 평균만 음수면 INCONCLUSIVE, 아니면 PASS(E1.4와 같은 규칙).

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release 손실 그대로)을 raw c, δ = 0, 같은 생성기·학습 레시피·seed로 돌린다.

## 알려진 위험

- 20,000 스텝(조건당 약 16번 방문)으로는 높은 s에서도 single이 학습 positive에 충분히 다가가지 못할 수 있다. 그러면 조작 검사가
  INCONCLUSIVE를 낸다. 스텝 수는 G4에서 정한 공통 값이라 바꾸지 않는다(CLAUDE.md §2.8).
- 높은 s에서는 AffD도 나빠질 수 있다(NW mixture의 이산 점을 외우거나 조건 사이를 보간하지 못함). 예측은 격차에 대한 것이므로
  AffD의 W2 자체는 판정하지 않고 보고한다.
- held-out 조건에서는 높은 s가 일반화를 해칠 수 있다. 판정하지 않고 보고한다.
