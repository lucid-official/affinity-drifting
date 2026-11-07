# E0.2 — symmetric h_q = h_p memorises: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.2` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **INCONCLUSIVE**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| INCONCLUSIVE | sym/seed0 | self_mean = 0.2881, stable = False, steps = 50000 |
| INCONCLUSIVE | sym/seed1 | self_mean = 0.3049, stable = False, steps = 50000 |
| INCONCLUSIVE | sym/seed2 | self_mean = 0.3043, stable = False, steps = 50000 |
| INCONCLUSIVE | sym/seed3 | self_mean = 0.2946, stable = False, steps = 50000 |
| INCONCLUSIVE | sym/seed4 | self_mean = 0.2996, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x0.5/seed0 | self_mean = 0.5001, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x0.5/seed1 | self_mean = 0.5243, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x0.5/seed2 | self_mean = 0.5027, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x0.5/seed3 | self_mean = 0.5013, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x0.5/seed4 | self_mean = 0.5015, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x2/seed0 | self_mean = 0.1937, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x2/seed1 | self_mean = 0.1978, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x2/seed2 | self_mean = 0.1819, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x2/seed3 | self_mean = 0.1926, stable = False, steps = 50000 |
| INCONCLUSIVE | sym_hp_x2/seed4 | self_mean = 0.1934, stable = False, steps = 50000 |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | self_mean | self_min | own_dist_mean | tv_mean | steps | final_rms_v | n_eff_mean | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sym | 0.2983 [0.2926, 0.3036] | 0 [0, 0] | 0.7604 [0.7575, 0.7635] | 0.7439 [0.7429, 0.7449] | 5e+04 [5e+04, 5e+04] | 1.261e-05 [9.162e-06, 1.779e-05] | 16 [16, 16] | 493.8 [343.2, 644.3] |
| sym_hp_x0.5 | 0.506 [0.5009, 0.5151] | 0 [0, 0] | 0.5676 [0.5639, 0.5704] | 0.6779 [0.6748, 0.6823] | 5e+04 [5e+04, 5e+04] | 5.581e-06 [3.334e-06, 7.827e-06] | 8.758 [8.758, 8.758] | 824 [742, 878] |
| sym_hp_x2 | 0.1919 [0.1865, 0.1958] | 0 [0, 0] | 0.9249 [0.921, 0.931] | 0.7246 [0.7231, 0.726] | 5e+04 [5e+04, 5e+04] | 9.549e-06 [8.276e-06, 1.078e-05] | 26.28 [26.28, 26.28] | 758.1 [736.4, 787.9] |
| hq0 | 0.07639 [0.07532, 0.07722] | 0.04453 [0.04063, 0.05] | 1.037 [1.037, 1.037] | 0.1469 [0.1446, 0.1491] | 4.899e+04 [4.816e+04, 4.965e+04] | 9.809e-07 [9.799e-07, 9.821e-07] | 16 [16, 16] | 231.3 [196.5, 251.9] |
| drifting | 0.9511 [0.9503, 0.9517] | 0.932 [0.9281, 0.9359] | 0.04447 [0.0441, 0.04492] | 0.8808 [0.8802, 0.8813] | 5e+04 [5e+04, 5e+04] | 1.11e-06 [1.071e-06, 1.143e-06] | 16 [16, 16] | 197.2 [180.7, 220.4] |

누락 run: 0개

## Figures

- [F0.1](../../../figures/L0/E0.2/F0.1/F0.1.png)
- [F0.5](../../../figures/L0/E0.2/F0.5/F0.5.png)
- [FC.1](../../../figures/L0/E0.2/FC.1/FC.1.png)
- [FC.2](../../../figures/L0/E0.2/FC.2/FC.2.png)

## 이상 징후

- `sym/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.39e-06)
- `sym/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.28e-05)
- `sym/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-05)
- `sym/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.12e-05)
- `sym/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 9.35e-06)
- `sym_hp_x0.5/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.05e-06)
- `sym_hp_x0.5/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 5.70e-06)
- `sym_hp_x0.5/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.67e-06)
- `sym_hp_x0.5/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 3.28e-06)
- `sym_hp_x0.5/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.21e-06)
- `sym_hp_x2/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.08e-05)
- `sym_hp_x2/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.60e-06)
- `sym_hp_x2/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.12e-05)
- `sym_hp_x2/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 7.20e-06)
- `sym_hp_x2/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 9.97e-06)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.05e-06)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.16e-06)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.09e-06)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
