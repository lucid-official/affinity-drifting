# E0.3 — h_q = 0 gives the NW mixture w̃: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.3` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| FAIL | b0:tv_mean | seed mean 0.1469 [0.1446, 0.1491] vs fail_above 0.1 |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | tv_mean | tv_max | self_mean | own_dist_mean | steps | final_rms_v | n_eff_mean | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0 | 0.1469 [0.1446, 0.1491] | 0.2101 [0.2016, 0.2199] | 0.07639 [0.07532, 0.07722] | 1.037 [1.037, 1.037] | 4.899e+04 [4.816e+04, 4.965e+04] | 9.809e-07 [9.799e-07, 9.821e-07] | 16 [16, 16] | 152.7 [141.9, 163.4] |
| drifting | 0.8808 [0.8802, 0.8813] | 0.9012 [0.899, 0.9034] | 0.9511 [0.9503, 0.9517] | 0.04447 [0.0441, 0.04492] | 5e+04 [5e+04, 5e+04] | 1.11e-06 [1.071e-06, 1.143e-06] | 16 [16, 16] | 1329 [1148, 1510] |

누락 run: 0개

## Figures

- [F0.1](../../../figures/L0/E0.3/F0.1/F0.1.png)
- [F0.2](../../../figures/L0/E0.3/F0.2/F0.2.png)
- [FC.1](../../../figures/L0/E0.3/FC.1/FC.1.png)
- [FC.2](../../../figures/L0/E0.3/FC.2/FC.2.png)

## 이상 징후

- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.05e-06)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.16e-06)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.09e-06)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
