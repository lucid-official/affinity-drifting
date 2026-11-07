# E0.4 — h_q interpolates monotonically from NW mixture to memorisation: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.4` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | self_mean: hq_0.25 − hq_0 | paired diff 0.001514 [0.0002686, 0.002515] (expected increasing) |
| PASS | self_mean: hq_0.5 − hq_0.25 | paired diff 0.02117 [0.01973, 0.02351] (expected increasing) |
| PASS | self_mean: hq_1 − hq_0.5 | paired diff 0.1992 [0.1919, 0.2054] (expected increasing) |
| PASS | tv_mean: hq_0.25 − hq_0 | paired diff 0.01999 [0.01811, 0.02204] (expected increasing) |
| PASS | tv_mean: hq_0.5 − hq_0.25 | paired diff 0.1143 [0.1103, 0.1174] (expected increasing) |
| PASS | tv_mean: hq_1 − hq_0.5 | paired diff 0.4628 [0.4597, 0.4658] (expected increasing) |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | tv_mean | self_mean | own_dist_mean | steps | final_rms_v | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| hq_0 | 0.1469 [0.1446, 0.1491] | 0.07639 [0.07532, 0.07722] | 1.037 [1.037, 1.037] | 4.899e+04 [4.816e+04, 4.965e+04] | 9.809e-07 [9.799e-07, 9.821e-07] | 151.8 [141.8, 162.2] |
| hq_0.25 | 0.1669 [0.1645, 0.1692] | 0.07791 [0.07661, 0.0792] | 1.037 [1.036, 1.038] | 5e+04 [5e+04, 5e+04] | 1.884e-06 [1.761e-06, 1.998e-06] | 729.7 [657.7, 809.2] |
| hq_0.5 | 0.2812 [0.2776, 0.2844] | 0.09907 [0.09766, 0.1006] | 1.017 [1.014, 1.019] | 5e+04 [5e+04, 5e+04] | 6.117e-06 [5.81e-06, 6.47e-06] | 795.9 [757.8, 838] |
| hq_1 | 0.7439 [0.7429, 0.7449] | 0.2983 [0.2926, 0.3036] | 0.7604 [0.7575, 0.7635] | 5e+04 [5e+04, 5e+04] | 1.261e-05 [9.162e-06, 1.779e-05] | 569.8 [444.2, 695.5] |
| drifting | 0.8808 [0.8802, 0.8813] | 0.9511 [0.9503, 0.9517] | 0.04447 [0.0441, 0.04492] | 5e+04 [5e+04, 5e+04] | 1.11e-06 [1.071e-06, 1.143e-06] | 185 [174, 190.6] |

누락 run: 0개

## Figures

- [F0.3](../../../figures/L0/E0.4/F0.3/F0.3.png)
- [FC.1](../../../figures/L0/E0.4/FC.1/FC.1.png)

## 이상 징후

- `hq_0.25/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.90e-06)
- `hq_0.25/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.80e-06)
- `hq_0.25/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.67e-06)
- `hq_0.25/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.99e-06)
- `hq_0.25/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.05e-06)
- `hq_0.5/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 6.71e-06)
- `hq_0.5/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 5.69e-06)
- `hq_0.5/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 6.37e-06)
- `hq_0.5/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 5.86e-06)
- `hq_0.5/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 5.96e-06)
- `hq_1/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.39e-06)
- `hq_1/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.28e-05)
- `hq_1/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-05)
- `hq_1/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.12e-05)
- `hq_1/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 9.35e-06)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.05e-06)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.16e-06)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.09e-06)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
