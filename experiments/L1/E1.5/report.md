# E1.5 — h_q selects the equilibrium in a trained generator (δ = 0, long training): 결과 보고서

> 자동 생성: `scripts/aggregate.py E1.5` · commit `11e6944352` · 2026-11-08 16:33 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | memorization_train: sym − hq0 > 0 | paired diff 0.001598 [0.0005707, 0.002724] |
| FAIL | hq0:nw_tv_train | seed mean 0.3861 [0.3842, 0.3879] (pass ≤ 0.1, fail > 0.1) |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | memorization_train | own_positive_share_train | nw_tv_train | w2_train | w2_heldout |
| --- | --- | --- | --- | --- | --- |
| hq0 | 0.1663 [0.1656, 0.167] | 0.0004822 [0.0004211, 0.0005676] | 0.3861 [0.3842, 0.3879] | 0.4577 [0.4518, 0.4619] | 0.4498 [0.4442, 0.4565] |
| sym | 0.1679 [0.1671, 0.1685] | 0.0005188 [0.0004761, 0.0005676] | 0.3744 [0.3698, 0.3791] | 0.4663 [0.4554, 0.4775] | 0.4589 [0.4484, 0.4675] |
| drifting | 0.01383 [0.009735, 0.01793] | 0 [0, 0] | 0.9986 [0.9982, 0.999] | 1.427 [1.426, 1.429] | 1.429 [1.428, 1.431] |

누락 run: 0개

## Figures

- (아직 없음: `scripts/make_figures.py E1.5`)

## 이상 징후

- (자동 감지 없음)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
