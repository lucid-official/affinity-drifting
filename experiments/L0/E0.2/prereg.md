# E0.2 — 사전 등록: Prop A — 대칭 h_q = h_p는 memorization

- 작성 2026-11-07 · 실험 엔지니어 Claude · 승인 Chan Lee
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋한다. 커밋 이후 수정하지 않는다(CLAUDE.md §2.9). `sweep.py`는
  두 파일이 커밋되지 않았거나 HEAD와 다르면 실행을 거부한다.

## plan 원문 (docs/plan.md §4, 해당 행 그대로)

| ID | 주장 | B0 대비 Δ | 예측 | 반증 기준 | 지표 |
| --- | --- | --- | --- | --- | --- |
| E0.2 | Prop A | h_q = h_p (대칭) | 모든 유한 h_p에서 particle이 y_k로 수렴 (memorization) | memorization이 아닌 설정에서 ‖V‖ ≈ 0이 안정적으로 유지됨 | 자기 positive까지 평균 거리, 수렴 스텝 |

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
| `sym` | h_q = h_p | 판정 (plan의 Δ) |
| `sym_hp_x0.5` | h_q = h_p, h_p × 0.5 | 판정 ("모든 유한 h_p", D10) |
| `sym_hp_x2` | h_q = h_p, h_p × 2 | 판정 |
| `hq0` | B0 (h_q = 0) | F0.1·F0.5 비교용, 판정 안 함 |
| `drifting` | `field: release` | baseline |

## 판정 규칙 (`experiment.yaml`의 `judge:`, 이 규칙으로만 판정)

- run마다 `self_mean` = 조건별 자기 positive 비율 A_kk의 평균.
- 판정 arm의 run 하나하나에 대해: `self_mean` ≥ 0.95이면 memorization(PASS),
  memorization이 아닌데 stable(‖V‖ < 1e-6이 1,000스텝 유지)이면 **FAIL**(plan의 반증 기준),
  memorization도 stable도 아니면 INCONCLUSIVE.
- 실험 판정은 가장 나쁜 run의 판정(FAIL > INCONCLUSIVE > PASS).

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting을 Lucid release 그대로(`DriftLoss`: Laplace kernel, τ ∈ {0.02, 0.05, 0.2}, 행×열 이중 softmax,
배치별 거리 스케일, 자기 mask) 같은 seed·초기값·η로 돌린다. positive는 원 논문처럼 class-conditional이라 조건 k는
자기 y_k만 본다(캡션 빌려오기 없음). L0 공통 설정 D4(정규화 전 필드)만 똑같이 적용한다.
개발 중 장난감 관찰(EXPERIMENTS.md 2026-11-07 setup): 이 필드는 고정된 유한 표본에서 ‖V‖ ≈ 0에 닿지 않는다.
따라서 baseline은 스텝 상한까지 돈 상태로 보고한다.

기대(기술적, 판정하지 않음): 조건마다 positive가 자기 y_k 하나뿐이므로 고정점은 q_k = δ_{y_k}, 즉 `self_mean` ≈ 1(memorization). 단, ‖V‖ ≈ 0에는 닿지 않는다.

## figure (plan §10, 판독 기준은 각 `figures/<FIG>/<FIG>.md`에 원문 인용)

- FC.1 (모든 arm), FC.2 (`sym`), F0.1 (`drifting`, `hq0`, `sym`), F0.5 (`drifting`, `hq0`, `sym`). F0.1·F0.5는 blind 버전 포함.

## 알려진 위험

- 스텝 상한 50k 안에 stable에 못 닿을 수 있다. 그때도 최종 상태로 판정하고 보고서의 이상 징후에 적는다.
- 대칭 repulsion은 (K, M, K·M) = (32, 256, 8192) 가중치를 쓰므로 run당 시간이 가장 길다.
